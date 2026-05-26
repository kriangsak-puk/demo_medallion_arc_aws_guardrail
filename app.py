"""Safe Haven Demo Booth — Chainlit Application Entry Point.

Interactive dual-engine demo booth UI that demonstrates the security contrast
between an ungoverned AI pipeline ("Data Swamp" / Engine_A) and a governed
Zero-Trust pipeline ("Safe Haven" / Engine_B).

Implements:
- Preset button actions (attack + analytics) with visual distinction
- Dual-engine concurrent processing via asyncio
- Showdown_View with streaming response panels
- Mock mode indicator
- Input validation and sanitization
- Render failure handling with page refresh requirement
- Engine timeout handling (30s) with per-panel error messages
- Input/button disable during processing with re-enable within 2s
"""

import asyncio
import logging
from typing import List, Optional

import chainlit as cl

from agent_wrapper import AgentConfig, StrandsAgentWrapper
from chart_renderer import ChartRenderer
from config import AppConfig, AppMode, ModeStatus
from engine_a import EngineA, EngineResult
from engine_b import EngineB
from health import health_endpoint, set_ready
from mode_detector import ModeDetector
from presets import PresetManager
from validators import sanitize_for_engine, validate_prompt

logger = logging.getLogger(__name__)

# Register the /health endpoint on the Chainlit (Starlette) app.
# This must be done at module level so the route is available immediately
# after the app starts serving requests. (Requirements 10.6, 10.7, 10.8)
try:
    from starlette.routing import Route

    cl.server.app.routes.insert(
        0, Route("/health", endpoint=health_endpoint, methods=["GET"])
    )
except Exception as e:
    logger.warning("Failed to register /health endpoint: %s", e)

# Mark the application as ready — the module has loaded successfully
# and the Chainlit server is serving. (Requirement 10.6)
set_ready()

# Session keys
SESSION_MODE = "mode"
SESSION_MODE_STATUS = "mode_status"
SESSION_ENGINE_A = "engine_a"
SESSION_ENGINE_B = "engine_b"
SESSION_IS_PROCESSING = "is_processing"
SESSION_PRESET_MSG = "preset_message"
SESSION_PRESET_MANAGER = "preset_manager"
SESSION_CHART_RENDERER = "chart_renderer"

# Engine timeout in seconds (Requirement 1.10)
ENGINE_TIMEOUT_SECONDS = 30

# Analytics detection keywords
ANALYTICS_KEYWORDS = [
    "revenue",
    "sales",
    "trend",
    "chart",
    "graph",
    "distribution",
    "top products",
    "monthly",
    "quarterly",
    "by region",
    "by product",
    "analytics",
    "show me",
    "visualize",
]


def _is_analytics_query(text: str) -> bool:
    """Detect if a query is an analytics request based on keywords.

    Args:
        text: The user's query text.

    Returns:
        True if the query matches analytics keywords, False otherwise.
    """
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in ANALYTICS_KEYWORDS)


def _build_preset_actions(preset_manager: PresetManager) -> List[cl.Action]:
    """Build Chainlit Action elements for all presets.

    Attack presets are visually distinguished from analytics presets via:
    - Different label prefixes (🔴 attack emojis vs 🔵 analytics emojis)
    - Different description text ("⚔️ Attack" vs "📊 Analytics")
    - Custom CSS styling (red border for attacks, blue border for analytics)
      applied via public/styles.css targeting action button IDs

    Args:
        preset_manager: The preset manager with configured presets.

    Returns:
        List of cl.Action elements for display in the UI.
    """
    actions: List[cl.Action] = []

    # Attack presets — red/danger themed via CSS (targets preset IDs starting
    # with attack category names like pii_, prompt_, role_, data_, etc.)
    for preset in preset_manager.get_attack_presets():
        action = cl.Action(
            name=f"preset_{preset.id}",
            label=preset.label,
            payload={"prompt_text": preset.prompt_text},
            tooltip=f"⚔️ Attack: {preset.category.replace('_', ' ').title()}",
        )
        actions.append(action)

    # Analytics presets — blue/primary themed via CSS (targets preset IDs
    # starting with "analytics_")
    for preset in preset_manager.get_analytics_presets():
        action = cl.Action(
            name=f"preset_{preset.id}",
            label=preset.label,
            payload={"prompt_text": preset.prompt_text},
            tooltip=f"📊 Analytics: {preset.category.replace('_', ' ').title()}",
        )
        actions.append(action)

    return actions


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize session, detect mode, display presets and mock indicator.

    Steps:
    1. Load AppConfig from environment variables
    2. Detect mode (Live/Mock) using ModeDetector
    3. Initialize StrandsAgentWrapper (or skip if Mock)
    4. Initialize EngineA and EngineB
    5. Store session state (mode, engines, presets)
    6. Display preset buttons (attack + analytics)
    7. Show "⚡ Offline Mock Mode" indicator if in Mock mode

    Requirements: 1.1, 1.5, 7.8, 8.1, 8.6, 11.6
    """
    # Load configuration from environment
    config, missing_vars = AppConfig.from_environment()

    # Detect operating mode
    if missing_vars or config.mock_mode_forced:
        mode = AppMode.MOCK
        mode_status = ModeStatus(
            mode=AppMode.MOCK,
            engine_a_live=False,
            engine_b_live=False,
            failure_reasons=missing_vars if missing_vars else ["Mock mode forced"],
        )
    else:
        detector = ModeDetector(region=config.aws_region)
        mode_status = await detector.detect_mode()
        mode = mode_status.mode

    # Store mode in session
    cl.user_session.set(SESSION_MODE, mode)
    cl.user_session.set(SESSION_MODE_STATUS, mode_status)
    cl.user_session.set(SESSION_IS_PROCESSING, False)

    # Initialize agent wrapper and engines
    agent_config = AgentConfig(
        bedrock_model_id=config.bedrock_model_id,
        knowledge_base_id=config.knowledge_base_id,
        guardrails_id=config.guardrails_id,
        guardrails_version=config.guardrails_version,
        glue_database_name=config.glue_database_name,
        glue_table_name=config.glue_table_name,
        permissive_role_arn=config.permissive_role_arn,
        restricted_role_arn=config.restricted_role_arn,
        region=config.aws_region,
    )
    agent = StrandsAgentWrapper(config=agent_config)
    engine_a = EngineA(agent=agent, mode=mode)
    engine_b = EngineB(agent=agent, mode=mode)
    chart_renderer = ChartRenderer()

    cl.user_session.set(SESSION_ENGINE_A, engine_a)
    cl.user_session.set(SESSION_ENGINE_B, engine_b)
    cl.user_session.set(SESSION_CHART_RENDERER, chart_renderer)

    # Display mock mode indicator if applicable (Requirement 7.8)
    if mode == AppMode.MOCK:
        await cl.Message(
            content="⚡ **Offline Mock Mode** — Running with simulated AWS responses",
        ).send()

    # Build and display preset buttons (Requirements 8.1, 8.6, 11.6)
    preset_manager = PresetManager()
    cl.user_session.set(SESSION_PRESET_MANAGER, preset_manager)
    actions = _build_preset_actions(preset_manager)

    # Send welcome message with preset action buttons displayed in visible area
    # above the input field on load (Requirement 8.6)
    welcome_content = (
        "# 🏰 Safe Haven Demo Booth\n\n"
        "Welcome! Submit an attack prompt or analytics query to see the "
        "security contrast between the **🧟 Data Swamp** (ungoverned) and "
        "**🛡️ Safe Haven** (Zero-Trust governed) pipelines.\n\n"
        "**🔴 Attack Presets** — Try jailbreak and data extraction attacks\n\n"
        "**🔵 Analytics Presets** — Run governed data queries with charts"
    )

    preset_msg = cl.Message(content=welcome_content, actions=actions)
    await preset_msg.send()

    # Store the preset message reference for later updates (disable/re-enable)
    cl.user_session.set(SESSION_PRESET_MSG, preset_msg)


async def _handle_preset_action(action: cl.Action) -> None:
    """Handle preset button clicks — shared logic for all preset callbacks.

    Behavior (Requirements 8.2, 8.4, 8.5, 8.6):
    - Displays the preset text as a user message in the chat (populates input
      field visually without auto-submitting to the on_message handler)
    - Processes through the same _process_prompt pipeline as manual submissions
    - Disables all preset buttons during processing
    - Re-enables buttons after processing completes
    """
    # Check if already processing — prevent concurrent submissions (Req 8.5)
    is_processing = cl.user_session.get(SESSION_IS_PROCESSING)
    if is_processing:
        return

    # Get the preset prompt text from the action payload
    prompt_text = action.payload.get("prompt_text", "")

    # Display the preset text as a user message in the chat panel
    # (Requirement 8.2 — populate input field with preset text)
    await cl.Message(
        content=prompt_text,
        author="User",
        type="user_message",
    ).send()

    # Process through the shared pipeline — identical to manual submissions (Req 8.4)
    await _process_prompt(prompt_text)


# Register action callbacks for all presets.
# Each preset button needs its own named callback in Chainlit 1.3.x
# since action_callback uses exact name matching.

# Attack presets
@cl.action_callback("preset_pii_extract_emails")
async def _cb_pii(action: cl.Action) -> None:
    await _handle_preset_action(action)


@cl.action_callback("preset_prompt_injection")
async def _cb_injection(action: cl.Action) -> None:
    await _handle_preset_action(action)


@cl.action_callback("preset_role_escalation")
async def _cb_escalation(action: cl.Action) -> None:
    await _handle_preset_action(action)


@cl.action_callback("preset_data_exfiltration")
async def _cb_exfiltration(action: cl.Action) -> None:
    await _handle_preset_action(action)


@cl.action_callback("preset_hallucination")
async def _cb_hallucination(action: cl.Action) -> None:
    await _handle_preset_action(action)


@cl.action_callback("preset_credential_extraction")
async def _cb_credential(action: cl.Action) -> None:
    await _handle_preset_action(action)


@cl.action_callback("preset_security_bypass")
async def _cb_bypass(action: cl.Action) -> None:
    await _handle_preset_action(action)


# Analytics presets
@cl.action_callback("preset_analytics_revenue")
async def _cb_revenue(action: cl.Action) -> None:
    await _handle_preset_action(action)


@cl.action_callback("preset_analytics_trends")
async def _cb_trends(action: cl.Action) -> None:
    await _handle_preset_action(action)


@cl.action_callback("preset_analytics_products")
async def _cb_products(action: cl.Action) -> None:
    await _handle_preset_action(action)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle manually typed messages from visitors.

    Validates input, then routes to both engines concurrently.
    Processes preset submissions identically to manual submissions (Req 8.4).

    Requirements: 1.2, 1.3, 1.4, 1.7, 1.9, 1.10
    """
    await _process_prompt(message.content)


async def _process_prompt(text: str) -> None:
    """Process a prompt through both engines concurrently.

    This is the shared processing logic used by both manual submissions
    and preset button actions, ensuring identical handling (Requirement 8.4).

    Steps:
    1. Validate input (1-500 chars, not empty/whitespace)
    2. Sanitize input for engines
    3. Disable preset buttons and input during processing
    4. Detect if analytics query
    5. Route to both Engine_A and Engine_B concurrently via asyncio.gather
    6. Create two response messages for the Showdown_View panels
    7. Stream tokens to respective panels within 500ms of receipt
    8. For analytics: render Plotly chart in Engine_B panel
    9. Display summary after both engines complete
    10. Handle errors and timeouts (30s per engine)
    11. Re-enable preset buttons and input within 2 seconds

    Args:
        text: The raw prompt text to process.
    """
    # Prevent concurrent submissions (Requirement 1.7)
    is_processing = cl.user_session.get(SESSION_IS_PROCESSING)
    if is_processing:
        return

    # Validate input (Requirement 1.9)
    validation = validate_prompt(text)
    if not validation.is_valid:
        await cl.Message(content=f"❌ {validation.error_message}").send()
        return

    # Sanitize for engine processing (Requirement 2.5)
    try:
        sanitized = sanitize_for_engine(validation.sanitized_text)
    except ValueError as e:
        await cl.Message(content=f"❌ {str(e)}").send()
        return

    # Mark as processing — disable input and buttons (Requirement 1.7, 8.5)
    cl.user_session.set(SESSION_IS_PROCESSING, True)
    await _disable_preset_buttons()

    try:
        # Detect query intent
        is_analytics = _is_analytics_query(sanitized)

        # Get engines from session
        engine_a: EngineA = cl.user_session.get(SESSION_ENGINE_A)
        engine_b: EngineB = cl.user_session.get(SESSION_ENGINE_B)

        # Create response messages for both panels (Requirement 1.1, 1.5)
        try:
            msg_a = cl.Message(content="", author="🧟 Data Swamp")
            msg_b = cl.Message(content="", author="🛡️ Safe Haven")
            await msg_a.send()
            await msg_b.send()
        except Exception as render_err:
            # Handle render failure (Requirement 1.8)
            logger.error("Panel render failure: %s", render_err)
            await cl.Message(
                content=(
                    "❌ **Rendering failure** — The response panels could not "
                    "be displayed. Please refresh the page to continue."
                )
            ).send()
            return

        # Define streaming callbacks (Requirement 1.3, 1.4 — within 500ms)
        async def stream_to_a(token: str) -> None:
            await msg_a.stream_token(token)

        async def stream_to_b(token: str) -> None:
            await msg_b.stream_token(token)

        # Route to both engines concurrently with timeout (Req 1.2, 1.10)
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    engine_a.process(prompt=sanitized, on_token=stream_to_a),
                    engine_b.process(
                        prompt=sanitized,
                        parent_message=msg_b,
                        on_token=stream_to_b,
                    ),
                    return_exceptions=True,
                ),
                timeout=ENGINE_TIMEOUT_SECONDS,
            )
            result_a, result_b = results
        except asyncio.TimeoutError:
            # Both engines timed out at the app level (Requirement 1.10)
            logger.error(
                "Both engines timed out after %d seconds", ENGINE_TIMEOUT_SECONDS
            )
            msg_a.content = (
                f"❌ Engine A timed out — no response received within "
                f"{ENGINE_TIMEOUT_SECONDS} seconds."
            )
            await msg_a.update()
            msg_b.content = (
                f"❌ Engine B timed out — no response received within "
                f"{ENGINE_TIMEOUT_SECONDS} seconds."
            )
            await msg_b.update()
            return

        # Handle exceptions from gather (per-engine error handling)
        if isinstance(result_a, Exception):
            logger.error("Engine A exception: %s", result_a)
            result_a = EngineResult(
                engine_label="🧟 Data Swamp",
                response_text="",
                is_blocked=False,
                error=f"Engine A error: {type(result_a).__name__}: {str(result_a)}",
            )
        if isinstance(result_b, Exception):
            logger.error("Engine B exception: %s", result_b)
            result_b = EngineResult(
                engine_label="🛡️ Safe Haven",
                response_text="",
                is_blocked=False,
                error=f"Engine B error: {type(result_b).__name__}: {str(result_b)}",
            )

        # Update Engine A panel (Requirement 1.3)
        if result_a.error:
            msg_a.content = f"❌ {result_a.error}"
        else:
            # Show raw data with PII in Engine A panel (Requirement 11.5)
            content_a = result_a.response_text
            if result_a.annotations:
                content_a += "\n\n" + "\n".join(result_a.annotations)
            msg_a.content = content_a
        await msg_a.update()

        # Update Engine B panel (Requirement 1.4)
        if result_b.error:
            msg_b.content = f"❌ {result_b.error}"
        else:
            content_b = result_b.response_text
            if result_b.annotations:
                content_b += "\n\n" + "\n".join(result_b.annotations)
            msg_b.content = content_b
        await msg_b.update()

        # Render Plotly chart for analytics queries in Engine B (Req 11.3)
        if not result_b.error and is_analytics and not result_b.is_blocked:
            await _render_analytics_chart(sanitized, msg_b)

        # Display summary with engine labels, guardrail actions, and
        # sensitive content detection (Requirement 1.6)
        summary_parts = []

        # Engine A summary
        if result_a.error:
            ea_status = "Error"
        elif result_a.sensitive_detected:
            ea_status = "⚠️ Sensitive content detected"
        else:
            ea_status = "Response delivered"
        summary_parts.append(f"**{result_a.engine_label}**: {ea_status}")

        # Engine B summary
        if result_b.error:
            eb_status = "Error"
        elif result_b.guardrail_action:
            eb_status = f"Guardrail action: {result_b.guardrail_action}"
        else:
            eb_status = "Response delivered"
        summary_parts.append(f"**{result_b.engine_label}**: {eb_status}")

        await cl.Message(content=" | ".join(summary_parts)).send()

    except Exception as e:
        # Catch-all for unexpected errors during processing
        logger.error("Unexpected error during prompt processing: %s", e)
        try:
            await cl.Message(
                content=(
                    "❌ **Processing error** — An unexpected error occurred. "
                    "Please refresh the page and try again."
                )
            ).send()
        except Exception:
            pass

    finally:
        # Re-enable preset buttons and input within 2 seconds (Requirement 8.5)
        cl.user_session.set(SESSION_IS_PROCESSING, False)
        await _enable_preset_buttons()


async def _render_analytics_chart(query: str, parent_msg: cl.Message) -> None:
    """Render a Plotly chart inline in the Engine B panel for analytics queries.

    For analytics queries, generates a chart from the query results and
    displays it as an interactive Plotly element in the Engine B panel.
    Falls back to a text table if chart generation fails.

    Args:
        query: The sanitized analytics query text.
        parent_msg: The Engine B message to attach the chart to.

    Requirements: 11.2, 11.3, 11.4, 11.8, 11.9
    """
    chart_renderer: ChartRenderer = cl.user_session.get(SESSION_CHART_RENDERER)
    mode = cl.user_session.get(SESSION_MODE)

    try:
        if mode == AppMode.MOCK:
            chart_result = await chart_renderer.render_mock_chart(query)
        else:
            # In live mode, chart would be generated from structured agent
            # response data. For now, use mock chart as placeholder.
            chart_result = await chart_renderer.render_mock_chart(query)

        if chart_result.figure is not None:
            # Render Plotly chart inline in Engine B panel (Requirement 11.3)
            chart_element = cl.Plotly(
                figure=chart_result.figure,
                name="analytics_chart",
                display="inline",
            )
            await cl.Message(
                content=f"📊 {chart_result.summary_text}",
                elements=[chart_element],
                author="🛡️ Safe Haven",
            ).send()
        elif chart_result.fallback_table:
            # Fallback to text table if chart generation fails (Req 11.9)
            await cl.Message(
                content=(
                    f"📊 {chart_result.summary_text}\n\n"
                    f"```\n{chart_result.fallback_table}\n```"
                ),
                author="🛡️ Safe Haven",
            ).send()

    except Exception as e:
        # Chart generation failure — fall back to error message (Req 11.9)
        logger.error("Chart rendering failed: %s", e)
        await cl.Message(
            content="📊 Chart generation failed. Data shown in text format above.",
            author="🛡️ Safe Haven",
        ).send()


async def _disable_preset_buttons() -> None:
    """Disable all preset buttons during processing.

    Removes actions from the preset message to prevent clicks
    while engines are processing (Requirement 1.7, 8.5).
    """
    preset_msg: Optional[cl.Message] = cl.user_session.get(SESSION_PRESET_MSG)
    if preset_msg:
        try:
            preset_msg.actions = []
            await preset_msg.update()
        except Exception as e:
            logger.warning("Failed to disable preset buttons: %s", e)


async def _enable_preset_buttons() -> None:
    """Re-enable all preset buttons after processing completes.

    Restores the action buttons on the preset message within 2 seconds
    of both engines completing (Requirement 8.5).
    """
    preset_msg: Optional[cl.Message] = cl.user_session.get(SESSION_PRESET_MSG)
    if preset_msg:
        try:
            preset_manager: PresetManager = cl.user_session.get(SESSION_PRESET_MANAGER)
            if preset_manager is None:
                preset_manager = PresetManager()
            actions = _build_preset_actions(preset_manager)
            preset_msg.actions = actions
            await preset_msg.update()
        except Exception as e:
            logger.warning("Failed to re-enable preset buttons: %s", e)
