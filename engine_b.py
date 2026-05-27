"""Engine B — Safe Haven Pipeline with 3-Gate sequential defense.

The governed Zero-Trust AI pipeline that uses the Restricted_Role to query
the Gold_Table (PII columns blocked by Lake Formation) and applies Bedrock
Guardrails for both input evaluation and output filtering.

The pipeline executes gates in strict sequential order:
  Gate_1 (Macie display) → Gate_2 (Lake Formation check) → Gate_3 (Guardrails)

If Gate_3 blocks the input, the agent is not invoked and a blocked message
is returned. If processing completes successfully, the response is annotated
with "✅ Safe Haven — Protected". If the guardrail output filter redacts
content, an additional "✅ Output Sanitized by Guardrails" annotation is added.
"""

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, List, Optional

from agent_wrapper import (
    AgentResponse,
    RoleAssumptionError,
    StrandsAgentWrapper,
)
from config import AppMode
from engine_a import EngineResult
from gates.base import GateContext, GatePipeline, GateResult
from gates.gate1_macie import Gate1Macie
from gates.gate2_lakeformation import Gate2LakeFormation
from gates.gate3_guardrails import Gate3Guardrails
from mock_engine import MockEngine

logger = logging.getLogger(__name__)

# Overall timeout for Engine B processing (seconds)
ENGINE_B_TIMEOUT_SECONDS = 180.0


class EngineB:
    """Engine B — Safe Haven governed pipeline with 3-Gate defense.

    Executes the 3-Gate sequential defense pipeline before invoking
    the Strands Agent with the Restricted_Role. Applies guardrails
    for both input evaluation and output filtering.

    The pipeline:
    1. Gate 1 (Macie) — informational display of PII scan results
    2. Gate 2 (Lake Formation) — column-level access control check
    3. Gate 3 (Guardrails) — input evaluation for jailbreak detection
    4. If not blocked: assume Restricted_Role, query Gold_Table, invoke model
    5. Apply guardrails output filtering on response
    6. Annotate result based on guardrail outcome
    """

    def __init__(self, agent: StrandsAgentWrapper, mode: AppMode) -> None:
        """Initialize Engine B.

        Args:
            agent: The Strands Agent wrapper for LLM invocation.
            mode: Current application mode (LIVE or MOCK).
        """
        self.agent = agent
        self.mode = mode
        self._mock_engine = MockEngine()

        # Build the gate pipeline in strict sequential order
        self._pipeline = GatePipeline(
            gates=[
                Gate1Macie(),
                Gate2LakeFormation(),
                Gate3Guardrails(),
            ]
        )

    async def process(
        self,
        prompt: str,
        parent_message: Any = None,
        on_token: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> EngineResult:
        """Process an attack prompt through the Safe Haven pipeline.

        Executes the 3-Gate defense pipeline in strict sequential order,
        then invokes the Strands Agent with the Restricted_Role if not
        blocked. Applies guardrails output filtering and annotates the
        result accordingly.

        Args:
            prompt: The visitor's attack prompt.
            parent_message: The parent Chainlit message for Step_Log display.
            on_token: Optional async callback for streaming response tokens.

        Returns:
            An EngineResult with the processing outcome, annotations,
            and timing information.
        """
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                self._process_internal(prompt, parent_message, on_token),
                timeout=ENGINE_B_TIMEOUT_SECONDS,
            )
            result.duration_ms = (time.time() - start_time) * 1000
            return result

        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                "Engine B timed out after %.1f seconds for prompt: %s",
                ENGINE_B_TIMEOUT_SECONDS,
                prompt[:50],
            )
            return EngineResult(
                engine_label="🛡️ Safe Haven",
                response_text="",
                is_blocked=False,
                error=(
                    "Engine B is unavailable: processing timed out "
                    f"after {int(ENGINE_B_TIMEOUT_SECONDS)} seconds"
                ),
                duration_ms=duration_ms,
            )

    async def _process_internal(
        self,
        prompt: str,
        parent_message: Any,
        on_token: Optional[Callable[[str], Awaitable[None]]],
    ) -> EngineResult:
        """Internal processing logic without the overall timeout wrapper.

        Args:
            prompt: The visitor's attack prompt.
            parent_message: The parent Chainlit message.
            on_token: Optional async callback for streaming tokens.

        Returns:
            An EngineResult with the processing outcome.
        """
        # Build gate context
        context = GateContext(
            prompt=prompt,
            session_mode=self.mode,
            parent_message=parent_message,
        )

        # Execute gate pipeline in strict sequential order
        gate_results: List[GateResult] = await self._pipeline.execute(context)

        # Check for gate errors — abort and report which gate failed
        for i, gate_result in enumerate(gate_results):
            if gate_result.error is not None:
                gate_name = self._pipeline.gates[i].name
                error_msg = f"Gate failed: {gate_name} — {gate_result.error}"
                logger.error(error_msg)
                return EngineResult(
                    engine_label="🛡️ Safe Haven",
                    response_text="",
                    is_blocked=False,
                    error=error_msg,
                )

        # Check if Gate_3 blocked the input
        gate3_result = self._get_gate3_result(gate_results)
        if gate3_result and gate3_result.blocked:
            return EngineResult(
                engine_label="🛡️ Safe Haven",
                response_text="🛡️ **Blocked by Bedrock Guardrails**\n\nYour request was identified as a potential security threat and blocked before reaching the data layer.",
                is_blocked=True,
                guardrail_action="BLOCKED",
                annotations=[
                    "🛡️ **Guardrails**: INPUT BLOCKED — jailbreak/attack detected",
                    "🔒 **Column Security**: Not reached (blocked at guardrail layer)",
                ],
            )

        # Not blocked — invoke the Strands Agent with Restricted_Role
        # In Mock mode, use MockEngine instead of the real agent
        if self.mode == AppMode.MOCK:
            return await self._fallback_to_mock(prompt)

        try:
            agent_response = await self._invoke_agent(prompt, on_token)
        except RoleAssumptionError as e:
            # Fall back to MockEngine on role assumption failure
            logger.error(
                "Engine B role assumption failed (ARN: %s, type: %s), "
                "falling back to Mock_Mode: %s",
                e.role_arn,
                e.error_type,
                str(e),
            )
            return await self._fallback_to_mock(prompt)

        # Handle agent invocation errors
        if agent_response.error:
            return EngineResult(
                engine_label="🛡️ Safe Haven",
                response_text="",
                is_blocked=False,
                error=f"Engine B agent error: {agent_response.error}",
            )

        # Apply guardrails output filtering annotation logic
        annotations: List[str] = []
        guardrail_action = agent_response.guardrail_action

        # Determine output action from Gate_3 details (for output filtering)
        output_action = self._get_output_action(gate3_result, agent_response)

        # Annotate based on guardrail outcome
        # Always annotate with "✅ Safe Haven — Protected" on successful completion
        annotations.append("✅ Safe Haven — Protected")
        annotations.append("🔒 **Column Security**: PII columns blocked by Lake Formation (Restricted_Role)")
        annotations.append("🛡️ **Guardrails**: Bedrock Guardrails active (jailbreak detection + output filtering)")

        # If output was redacted/modified, add sanitization annotation
        if output_action == "MODIFIED":
            annotations.append("✅ Output Sanitized by Guardrails")
            guardrail_action = "MODIFIED"

        # Include generated SQL in annotations if available
        generated_sql = agent_response.metadata.get("generated_sql", "")
        if generated_sql:
            annotations.append(f"📝 **Generated SQL:**\n```sql\n{generated_sql}\n```")

        return EngineResult(
            engine_label="🛡️ Safe Haven",
            response_text=agent_response.content,
            is_blocked=False,
            annotations=annotations,
            guardrail_action=guardrail_action,
            sensitive_detected=False,
        )

    def _get_gate3_result(self, gate_results: List[GateResult]) -> Optional[GateResult]:
        """Extract the Gate_3 result from the pipeline results.

        Gate_3 is the third gate (index 2) in the pipeline. If the
        pipeline was aborted before reaching Gate_3, returns None.

        Args:
            gate_results: List of results from the gate pipeline.

        Returns:
            The Gate_3 GateResult, or None if not reached.
        """
        if len(gate_results) >= 3:
            return gate_results[2]
        return None

    def _get_output_action(
        self,
        gate3_result: Optional[GateResult],
        agent_response: AgentResponse,
    ) -> str:
        """Determine the output guardrail action.

        Checks Gate_3 details for output_action, and also considers
        the agent response's guardrail_action for output filtering.

        Args:
            gate3_result: The Gate_3 result (may contain output_action).
            agent_response: The agent response (may contain guardrail_action).

        Returns:
            The output action string: "PASS", "BLOCKED", or "MODIFIED".
        """
        # Check Gate_3 details for output action
        if gate3_result and gate3_result.details:
            output_action = gate3_result.details.get("output_action", "PASS")
            if output_action in ("BLOCKED", "MODIFIED"):
                return output_action

        # Check agent response guardrail action
        if agent_response.guardrail_action == "MODIFIED":
            return "MODIFIED"

        return "PASS"

    async def _invoke_agent(
        self,
        prompt: str,
        on_token: Optional[Callable[[str], Awaitable[None]]],
    ) -> AgentResponse:
        """Invoke the Strands Agent with the Restricted_Role.

        Assumes the Restricted_Role, queries the Gold_Table (non-PII only),
        and invokes the model with guardrails applied.

        Args:
            prompt: The visitor's attack prompt.
            on_token: Optional async callback for streaming tokens.

        Returns:
            The AgentResponse from the Strands Agent.

        Raises:
            RoleAssumptionError: If the Restricted_Role cannot be assumed.
        """
        return await self.agent.invoke(
            prompt=prompt,
            role_arn=self.agent.config.restricted_role_arn,
            apply_guardrails=True,
            on_token=on_token,
        )

    async def _fallback_to_mock(self, prompt: str) -> EngineResult:
        """Fall back to MockEngine when role assumption fails.

        Args:
            prompt: The visitor's attack prompt.

        Returns:
            An EngineResult with the mock response.
        """
        mock_response = await self._mock_engine.mock_engine_b(prompt)

        # Determine annotations based on mock response content
        annotations: List[str] = []
        is_blocked = "🛡️ Blocked by Bedrock Guardrails" in mock_response

        if is_blocked:
            annotations.append("🛡️ Blocked by Bedrock Guardrails")
        else:
            annotations.append("✅ Safe Haven — Protected")

        return EngineResult(
            engine_label="🛡️ Safe Haven",
            response_text=mock_response,
            is_blocked=is_blocked,
            annotations=annotations,
            guardrail_action="BLOCKED" if is_blocked else "PASS",
        )
