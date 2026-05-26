"""Gate 3 — Bedrock Guardrails evaluation.

Evaluates the visitor's Attack_Prompt through Amazon Bedrock Guardrails
for jailbreak detection (input evaluation) and output filtering (output evaluation).
Blocks agent invocation if input evaluation returns BLOCKED.
"""

import asyncio
import logging
from typing import Any, Dict

from config import AppMode
from gates.base import Gate, GateContext, GateResult
from mock_engine import MockEngine

logger = logging.getLogger(__name__)

# Valid guardrail actions
VALID_ACTIONS = {"PASS", "BLOCKED", "MODIFIED"}


class Gate3Guardrails(Gate):
    """Gate 3: Bedrock Guardrails input/output evaluation.

    Evaluates the Attack_Prompt through Bedrock Guardrails and displays
    the input and output evaluation results in a Step_Log. Blocks agent
    invocation if the input evaluation returns BLOCKED.
    """

    def __init__(self) -> None:
        super().__init__(name="Gate 3 — Bedrock Guardrails", timeout_seconds=10.0)

    def format_guardrails_output(self, input_action: str, output_action: str) -> str:
        """Format the guardrails evaluation result for Step_Log display.

        Args:
            input_action: The input guardrail action (PASS, BLOCKED, or MODIFIED).
            output_action: The output guardrail action (PASS, BLOCKED, or MODIFIED).

        Returns:
            A formatted string showing both input and output evaluation results.
        """
        return f"Input: {input_action}\nOutput: {output_action}"

    def should_expand(self, input_action: str, output_action: str) -> bool:
        """Determine whether the Step_Log should be expanded.

        The Step_Log is expanded if either the input or output action is
        BLOCKED or MODIFIED. It is collapsed only when both are PASS.

        Args:
            input_action: The input guardrail action (PASS, BLOCKED, or MODIFIED).
            output_action: The output guardrail action (PASS, BLOCKED, or MODIFIED).

        Returns:
            True if the Step_Log should be expanded, False if collapsed.
        """
        return input_action != "PASS" or output_action != "PASS"

    async def execute(self, context: GateContext) -> GateResult:
        """Execute Gate 3 guardrails evaluation.

        In MOCK mode, uses MockEngine to simulate guardrail evaluation.
        In LIVE mode, calls the Bedrock Guardrails API (placeholder).

        Displays a Step_Log labeled "[Gate 3] Evaluating Bedrock Guardrails..."
        with input/output evaluation results.

        Args:
            context: The gate context containing prompt, session mode,
                and parent message.

        Returns:
            A GateResult indicating whether the gate passed or blocked.
        """
        try:
            # Create Chainlit Step element for visualization
            step = None
            try:
                import chainlit as cl

                step = cl.Step(
                    name="[Gate 3] Evaluating Bedrock Guardrails...",
                    parent_id=context.parent_message.id
                    if context.parent_message
                    else None,
                )
                await step.send()
            except (ImportError, Exception):
                # Chainlit not available (e.g., in tests) — proceed without UI
                pass

            # Evaluate guardrails with timeout
            guardrail_result = await asyncio.wait_for(
                self._evaluate_guardrails(context),
                timeout=self.timeout_seconds,
            )

            input_action = guardrail_result.get("input_action", "PASS")
            output_action = guardrail_result.get("output_action", "PASS")

            # Format Step_Log content
            log_content = self.format_guardrails_output(input_action, output_action)
            expand = self.should_expand(input_action, output_action)

            # Update Step element if available
            if step is not None:
                try:
                    step.output = log_content
                    step.is_error = input_action == "BLOCKED"
                    # Set expansion state
                    if expand:
                        step.show_input = True
                    await step.update()
                except Exception:
                    pass

            # Determine blocking logic
            if input_action == "BLOCKED":
                return GateResult(
                    passed=False,
                    blocked=True,
                    details={
                        "input_action": "BLOCKED",
                        "output_action": output_action,
                    },
                )

            return GateResult(
                passed=True,
                blocked=False,
                details={
                    "input_action": input_action,
                    "output_action": output_action,
                },
            )

        except asyncio.TimeoutError:
            logger.error("Gate 3 guardrails evaluation timed out after %ss", self.timeout_seconds)
            # Update step with failure message
            if step is not None:
                try:
                    step.output = "Guardrails evaluation failed"
                    step.is_error = True
                    step.show_input = True
                    await step.update()
                except Exception:
                    pass

            return GateResult(
                passed=False,
                blocked=False,
                error="Guardrails evaluation failed",
            )
        except Exception as e:
            logger.error("Gate 3 guardrails evaluation failed: %s", str(e))
            # Update step with failure message
            if step is not None:
                try:
                    step.output = "Guardrails evaluation failed"
                    step.is_error = True
                    step.show_input = True
                    await step.update()
                except Exception:
                    pass

            return GateResult(
                passed=False,
                blocked=False,
                error="Guardrails evaluation failed",
            )

    async def _evaluate_guardrails(self, context: GateContext) -> Dict[str, Any]:
        """Evaluate the prompt through guardrails based on session mode.

        Args:
            context: The gate context containing prompt and session mode.

        Returns:
            A dictionary with input_action and output_action fields.
        """
        if context.session_mode == AppMode.MOCK:
            return await MockEngine().mock_gate3(context.prompt)
        else:
            # LIVE mode: call Bedrock Guardrails API (placeholder)
            return await self._evaluate_live(context.prompt)

    async def _evaluate_live(self, prompt: str) -> Dict[str, Any]:
        """Evaluate prompt through live Bedrock Guardrails API.

        This is a placeholder for the live implementation that will
        call the actual Bedrock Guardrails API.

        Args:
            prompt: The attack prompt to evaluate.

        Returns:
            A dictionary with input_action and output_action fields.
        """
        # TODO: Implement live Bedrock Guardrails API call
        # For now, return PASS for both actions
        return {
            "input_action": "PASS",
            "output_action": "PASS",
        }
