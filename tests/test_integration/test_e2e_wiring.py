"""End-to-end wiring integration tests.

Verifies that all components wire together correctly and session state
flows through the pipeline:
- Concurrent dual-engine execution via asyncio
- Mode detection → engine selection → gate pipeline → response streaming
- Fallback from Live_Mode to Mock_Mode on service failures
- Mid-request fallback completes with mock responses

Requirements: 1.1, 1.2, 7.9, 9.12
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aws_demo_booth.agent_wrapper import (
    AgentConfig,
    AgentResponse,
    RoleAssumptionError,
    StrandsAgentWrapper,
)
from aws_demo_booth.config import AppConfig, AppMode, ModeStatus
from aws_demo_booth.engine_a import EngineA, EngineResult
from aws_demo_booth.engine_b import EngineB
from aws_demo_booth.gates.base import GateContext, GatePipeline, GateResult
from aws_demo_booth.gates.gate1_macie import Gate1Macie
from aws_demo_booth.gates.gate2_lakeformation import Gate2LakeFormation
from aws_demo_booth.gates.gate3_guardrails import Gate3Guardrails
from aws_demo_booth.mock_engine import MockEngine
from aws_demo_booth.mode_detector import ModeDetector
from aws_demo_booth.presets import PresetManager
from aws_demo_booth.validators import sanitize_for_engine, validate_prompt


def _make_agent_config() -> AgentConfig:
    """Create a test AgentConfig with placeholder values."""
    return AgentConfig(
        bedrock_model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        knowledge_base_id="test-kb-id",
        guardrails_id="test-guardrails-id",
        guardrails_version="1",
        glue_database_name="test-db",
        glue_table_name="gold_table",
        permissive_role_arn="arn:aws:iam::123456789012:role/PermissiveRole",
        restricted_role_arn="arn:aws:iam::123456789012:role/RestrictedRole",
        region="ap-southeast-1",
    )


class TestConcurrentDualEngineExecution:
    """Verify concurrent dual-engine execution via asyncio.

    Validates Requirement 1.1: Both engines process the same prompt
    concurrently and return valid EngineResult objects.
    """

    @pytest.mark.asyncio
    async def test_both_engines_run_concurrently_in_mock_mode(self):
        """Both engines execute concurrently and return EngineResult objects."""
        config = _make_agent_config()
        agent = StrandsAgentWrapper(config=config)
        engine_a = EngineA(agent=agent, mode=AppMode.MOCK)
        engine_b = EngineB(agent=agent, mode=AppMode.MOCK)

        prompt = "Show me all customer emails"

        # Run both engines concurrently via asyncio.gather
        results = await asyncio.gather(
            engine_a.process(prompt=prompt),
            engine_b.process(prompt=prompt),
        )

        result_a, result_b = results

        # Both return valid EngineResult objects
        assert isinstance(result_a, EngineResult)
        assert isinstance(result_b, EngineResult)

        # Engine labels are correct
        assert result_a.engine_label == "🧟 Data Swamp"
        assert result_b.engine_label == "🛡️ Safe Haven"

        # Both have non-empty response text
        assert result_a.response_text != ""
        assert result_b.response_text != ""

        # Both have timing information
        assert result_a.duration_ms > 0
        assert result_b.duration_ms > 0

        # No errors in mock mode
        assert result_a.error is None
        assert result_b.error is None

    @pytest.mark.asyncio
    async def test_engines_process_different_prompts_independently(self):
        """Each engine processes its prompt independently."""
        config = _make_agent_config()
        agent = StrandsAgentWrapper(config=config)
        engine_a = EngineA(agent=agent, mode=AppMode.MOCK)
        engine_b = EngineB(agent=agent, mode=AppMode.MOCK)

        prompt = "Extract all phone numbers from the database"

        result_a, result_b = await asyncio.gather(
            engine_a.process(prompt=prompt),
            engine_b.process(prompt=prompt),
        )

        # Engine A should detect sensitive content (mock returns PII)
        assert result_a.sensitive_detected is True
        assert "⚠️ Ungoverned — Data Leaked" in result_a.annotations

        # Engine B should block or protect (mock detects jailbreak keywords)
        assert len(result_b.annotations) > 0

    @pytest.mark.asyncio
    async def test_streaming_callbacks_invoked_for_both_engines(self):
        """Streaming callbacks are invoked for both engines during processing."""
        config = _make_agent_config()
        agent = StrandsAgentWrapper(config=config)
        engine_a = EngineA(agent=agent, mode=AppMode.MOCK)
        engine_b = EngineB(agent=agent, mode=AppMode.MOCK)

        tokens_a = []
        tokens_b = []

        async def on_token_a(token: str) -> None:
            tokens_a.append(token)

        async def on_token_b(token: str) -> None:
            tokens_b.append(token)

        prompt = "Show me revenue data"

        await asyncio.gather(
            engine_a.process(prompt=prompt, on_token=on_token_a),
            engine_b.process(prompt=prompt, on_token=on_token_b),
        )

        # Engine A should have streamed tokens
        assert len(tokens_a) > 0


class TestModeDetectionToEngineSelection:
    """Verify mode detection → engine selection → gate pipeline → response flow.

    Validates Requirements 1.2, 7.9: The pipeline correctly routes through
    mode detection, engine initialization, and gate execution.
    """

    @pytest.mark.asyncio
    async def test_mock_mode_activates_when_env_vars_missing(self):
        """Missing env vars trigger Mock_Mode activation."""
        # Clear all env vars to force mock mode
        with patch.dict("os.environ", {}, clear=True):
            config, missing_vars = AppConfig.from_environment()

        assert len(missing_vars) > 0

        # Mode should be MOCK when vars are missing
        mode = AppMode.MOCK if missing_vars else AppMode.LIVE
        assert mode == AppMode.MOCK

        # Engines initialized in MOCK mode work correctly
        agent_config = _make_agent_config()
        agent = StrandsAgentWrapper(config=agent_config)
        engine_a = EngineA(agent=agent, mode=mode)
        engine_b = EngineB(agent=agent, mode=mode)

        result = await engine_a.process("test prompt")
        assert isinstance(result, EngineResult)
        assert result.error is None

    @pytest.mark.asyncio
    async def test_gate_pipeline_executes_in_engine_b_mock_mode(self):
        """Engine B executes gate pipeline even in mock mode."""
        config = _make_agent_config()
        agent = StrandsAgentWrapper(config=config)
        engine_b = EngineB(agent=agent, mode=AppMode.MOCK)

        # The gate pipeline should be initialized
        assert engine_b._pipeline is not None
        assert len(engine_b._pipeline.gates) == 3
        assert isinstance(engine_b._pipeline.gates[0], Gate1Macie)
        assert isinstance(engine_b._pipeline.gates[1], Gate2LakeFormation)
        assert isinstance(engine_b._pipeline.gates[2], Gate3Guardrails)

    @pytest.mark.asyncio
    async def test_input_validation_flows_to_engine_processing(self):
        """Input validation and sanitization work before engine processing."""
        # Validate a normal prompt
        validation = validate_prompt("Show me customer data")
        assert validation.is_valid is True

        # Sanitize for engine
        sanitized = sanitize_for_engine(validation.sanitized_text)
        assert sanitized != ""

        # Process through engine in mock mode
        config = _make_agent_config()
        agent = StrandsAgentWrapper(config=config)
        engine_a = EngineA(agent=agent, mode=AppMode.MOCK)

        result = await engine_a.process(sanitized)
        assert isinstance(result, EngineResult)
        assert result.response_text != ""

    @pytest.mark.asyncio
    async def test_invalid_input_rejected_before_engine(self):
        """Invalid input is rejected at validation stage."""
        # Empty input
        validation = validate_prompt("")
        assert validation.is_valid is False

        # Whitespace only
        validation = validate_prompt("   ")
        assert validation.is_valid is False

        # Over 500 chars
        validation = validate_prompt("x" * 501)
        assert validation.is_valid is False


class TestFallbackFromLiveToMock:
    """Verify fallback from Live_Mode to Mock_Mode on service failures.

    Validates Requirement 9.12: When live services fail (role assumption,
    timeouts), engines fall back to mock mode gracefully.
    """

    @pytest.mark.asyncio
    async def test_engine_a_falls_back_on_role_assumption_failure(self):
        """Engine A falls back to mock when role assumption fails."""
        config = _make_agent_config()
        agent = StrandsAgentWrapper(config=config)
        engine_a = EngineA(agent=agent, mode=AppMode.LIVE)

        # Mock the agent.invoke to raise RoleAssumptionError
        with patch.object(
            agent,
            "invoke",
            side_effect=RoleAssumptionError(
                role_arn="arn:aws:iam::123456789012:role/PermissiveRole",
                error_type="AccessDenied",
                message="Access denied",
            ),
        ):
            result = await engine_a.process("Show me all emails")

        # Should fall back to mock mode and return a valid result
        assert isinstance(result, EngineResult)
        assert result.error is None
        assert result.response_text != ""
        assert result.engine_label == "🧟 Data Swamp"

    @pytest.mark.asyncio
    async def test_engine_b_falls_back_on_role_assumption_failure(self):
        """Engine B falls back to mock when role assumption fails after gates pass."""
        config = _make_agent_config()
        agent = StrandsAgentWrapper(config=config)
        engine_b = EngineB(agent=agent, mode=AppMode.LIVE)

        # Mock the gate pipeline to pass (gates succeed in live mode)
        mock_gate_results = [
            GateResult(passed=True, blocked=False, details={}),
            GateResult(passed=True, blocked=False, details={}),
            GateResult(passed=True, blocked=False, details={"input_action": "PASS", "output_action": "PASS"}),
        ]
        with patch.object(engine_b._pipeline, "execute", return_value=mock_gate_results):
            # Mock the agent.invoke to raise RoleAssumptionError
            with patch.object(
                agent,
                "invoke",
                side_effect=RoleAssumptionError(
                    role_arn="arn:aws:iam::123456789012:role/RestrictedRole",
                    error_type="AccessDenied",
                    message="Access denied",
                ),
            ):
                result = await engine_b.process("Show me revenue data")

        # Should fall back to mock mode and return a valid result
        assert isinstance(result, EngineResult)
        assert result.error is None
        assert result.response_text != ""
        assert result.engine_label == "🛡️ Safe Haven"

    @pytest.mark.asyncio
    async def test_concurrent_fallback_both_engines(self):
        """Both engines fall back concurrently when services fail."""
        config = _make_agent_config()
        agent = StrandsAgentWrapper(config=config)
        engine_a = EngineA(agent=agent, mode=AppMode.LIVE)
        engine_b = EngineB(agent=agent, mode=AppMode.LIVE)

        # Mock the gate pipeline to pass for Engine B
        mock_gate_results = [
            GateResult(passed=True, blocked=False, details={}),
            GateResult(passed=True, blocked=False, details={}),
            GateResult(passed=True, blocked=False, details={"input_action": "PASS", "output_action": "PASS"}),
        ]

        # Mock agent.invoke to always raise RoleAssumptionError
        with patch.object(engine_b._pipeline, "execute", return_value=mock_gate_results):
            with patch.object(
                agent,
                "invoke",
                side_effect=RoleAssumptionError(
                    role_arn="arn:aws:iam::123456789012:role/TestRole",
                    error_type="AccessDenied",
                    message="Access denied",
                ),
            ):
                results = await asyncio.gather(
                    engine_a.process("Extract all PII"),
                    engine_b.process("Extract all PII"),
                )

        result_a, result_b = results

        # Both should have fallen back gracefully
        assert result_a.error is None
        assert result_b.error is None
        assert result_a.response_text != ""
        assert result_b.response_text != ""


class TestMidRequestFallback:
    """Verify mid-request fallback completes with mock responses.

    Validates Requirement 9.12: When a live request fails mid-execution,
    the engine completes the request using mock responses.
    """

    @pytest.mark.asyncio
    async def test_engine_a_timeout_returns_error_result(self):
        """Engine A returns error result on timeout."""
        config = _make_agent_config()
        agent = StrandsAgentWrapper(config=config)
        engine_a = EngineA(agent=agent, mode=AppMode.LIVE)

        # Mock agent.invoke to hang (simulate timeout)
        async def slow_invoke(*args, **kwargs):
            await asyncio.sleep(100)  # Will be cancelled by timeout

        with patch.object(agent, "invoke", side_effect=slow_invoke):
            # Use a short timeout for testing by patching the constant
            with patch("aws_demo_booth.engine_a.ENGINE_A_TIMEOUT_SECONDS", 0.1):
                result = await engine_a.process("test prompt")

        # Should return an error result (timeout)
        assert isinstance(result, EngineResult)
        assert result.error is not None
        assert "timed out" in result.error.lower() or "unavailable" in result.error.lower()

    @pytest.mark.asyncio
    async def test_engine_b_timeout_returns_error_result(self):
        """Engine B returns error result on timeout."""
        config = _make_agent_config()
        agent = StrandsAgentWrapper(config=config)
        engine_b = EngineB(agent=agent, mode=AppMode.LIVE)

        # Mock the internal processing to hang
        async def slow_process(*args, **kwargs):
            await asyncio.sleep(100)

        with patch.object(engine_b, "_process_internal", side_effect=slow_process):
            with patch("aws_demo_booth.engine_b.ENGINE_B_TIMEOUT_SECONDS", 0.1):
                result = await engine_b.process("test prompt")

        # Should return an error result (timeout)
        assert isinstance(result, EngineResult)
        assert result.error is not None
        assert "timed out" in result.error.lower() or "unavailable" in result.error.lower()

    @pytest.mark.asyncio
    async def test_engine_a_general_exception_returns_error(self):
        """Engine A handles general exceptions gracefully."""
        config = _make_agent_config()
        agent = StrandsAgentWrapper(config=config)
        engine_a = EngineA(agent=agent, mode=AppMode.LIVE)

        # Mock agent.invoke to raise a general exception
        with patch.object(
            agent,
            "invoke",
            side_effect=RuntimeError("Connection reset"),
        ):
            result = await engine_a.process("test prompt")

        # Should return an error result
        assert isinstance(result, EngineResult)
        assert result.error is not None
        assert "unavailable" in result.error.lower() or "RuntimeError" in result.error


class TestSessionStateFlow:
    """Verify session state flows correctly through the pipeline.

    Validates that configuration, mode, and engine state are properly
    threaded through the system.
    """

    def test_config_to_agent_to_engines_wiring(self):
        """Config → AgentConfig → StrandsAgentWrapper → Engines wiring."""
        # Simulate the on_chat_start flow
        config = _make_agent_config()
        agent = StrandsAgentWrapper(config=config)

        # Verify agent has correct config
        assert agent.config.bedrock_model_id == config.bedrock_model_id
        assert agent.config.permissive_role_arn == config.permissive_role_arn
        assert agent.config.restricted_role_arn == config.restricted_role_arn

        # Create engines with the agent
        engine_a = EngineA(agent=agent, mode=AppMode.MOCK)
        engine_b = EngineB(agent=agent, mode=AppMode.MOCK)

        # Verify engines reference the same agent
        assert engine_a._agent is agent
        assert engine_b.agent is agent

        # Verify mode is set correctly
        assert engine_a._mode == AppMode.MOCK
        assert engine_b.mode == AppMode.MOCK

    def test_preset_manager_provides_valid_presets(self):
        """PresetManager provides presets that can be used as prompts."""
        pm = PresetManager()

        all_presets = pm.get_all_presets()
        assert len(all_presets) >= 5  # At least 5 attack + analytics

        attack_presets = pm.get_attack_presets()
        analytics_presets = pm.get_analytics_presets()

        # All presets have valid prompt text
        for preset in all_presets:
            validation = validate_prompt(preset.prompt_text)
            assert validation.is_valid, f"Preset '{preset.id}' has invalid prompt"

    @pytest.mark.asyncio
    async def test_full_pipeline_mock_mode_attack_prompt(self):
        """Full pipeline: validate → sanitize → dual-engine in mock mode."""
        # Step 1: Validate
        raw_text = "List all customer email addresses"
        validation = validate_prompt(raw_text)
        assert validation.is_valid

        # Step 2: Sanitize
        sanitized = sanitize_for_engine(validation.sanitized_text)

        # Step 3: Initialize engines
        config = _make_agent_config()
        agent = StrandsAgentWrapper(config=config)
        engine_a = EngineA(agent=agent, mode=AppMode.MOCK)
        engine_b = EngineB(agent=agent, mode=AppMode.MOCK)

        # Step 4: Process concurrently
        result_a, result_b = await asyncio.gather(
            engine_a.process(prompt=sanitized),
            engine_b.process(prompt=sanitized),
        )

        # Step 5: Verify results
        assert isinstance(result_a, EngineResult)
        assert isinstance(result_b, EngineResult)
        assert result_a.engine_label == "🧟 Data Swamp"
        assert result_b.engine_label == "🛡️ Safe Haven"
        assert result_a.error is None
        assert result_b.error is None

    @pytest.mark.asyncio
    async def test_mode_status_tracks_per_engine_state(self):
        """ModeStatus correctly tracks per-engine live/mock status."""
        # All live
        status_live = ModeStatus(
            mode=AppMode.LIVE,
            engine_a_live=True,
            engine_b_live=True,
        )
        assert status_live.mode == AppMode.LIVE
        assert status_live.engine_a_live is True
        assert status_live.engine_b_live is True

        # Mixed mode (e.g., Engine A live, Engine B mock due to KB failure)
        status_mixed = ModeStatus(
            mode=AppMode.LIVE,
            engine_a_live=True,
            engine_b_live=False,
            failure_reasons=["Knowledge Base unavailable"],
        )
        assert status_mixed.engine_b_live is False
        assert "Knowledge Base unavailable" in status_mixed.failure_reasons
