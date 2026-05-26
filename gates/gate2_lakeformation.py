"""Gate 2 — AWS Lake Formation column-level access control check.

Displays the difference between Permissive_Role (Engine_A) and
Restricted_Role (Engine_B) column-level access as enforced by
AWS Lake Formation. Blocks data retrieval on timeout or failure.
"""

import asyncio
import logging
from typing import Any, Dict, List

from aws_demo_booth.config import AppMode
from aws_demo_booth.gates.base import Gate, GateContext, GateResult
from aws_demo_booth.mock_engine import MockEngine

logger = logging.getLogger(__name__)

# Optional Chainlit import — allows patching in tests
try:
    import chainlit as cl
except Exception:
    cl = None  # type: ignore[assignment]


class Gate2LakeFormation(Gate):
    """Gate 2: Evaluates AWS Lake Formation column-level permissions.

    In MOCK mode, uses MockEngine to simulate column access data.
    In LIVE mode, retrieves Lake Formation permissions (placeholder).

    Displays allowed/denied columns, role comparison, and summary.
    Auto-expands when columns are denied, collapses when all permitted.
    Blocks data retrieval on timeout or failure.
    """

    TIMEOUT_SECONDS: float = 10.0

    def __init__(self) -> None:
        super().__init__(
            name="Gate 2: Lake Formation",
            timeout_seconds=self.TIMEOUT_SECONDS,
        )

    async def execute(self, context: GateContext) -> GateResult:
        """Execute Gate 2 Lake Formation column access check.

        Creates a Chainlit Step element labeled
        "[Gate 2] Evaluating AWS Lake Formation...", retrieves column
        access data, formats the output, and sets expansion state based
        on denied columns.

        On timeout or failure, blocks data retrieval by returning
        GateResult with passed=False and blocked=True.

        Args:
            context: The gate context containing prompt, session mode,
                and parent message.

        Returns:
            GateResult indicating success/failure and whether data
            retrieval is blocked.
        """
        step = None
        try:
            # Create Chainlit Step element
            if cl is not None:
                step = cl.Step(
                    name="[Gate 2] Evaluating AWS Lake Formation...",
                    parent_id=context.parent_message.id
                    if context.parent_message
                    else None,
                )
                await step.__aenter__()

            # Retrieve column access data with timeout
            column_data = await asyncio.wait_for(
                self._get_column_access(context.session_mode),
                timeout=self.timeout_seconds,
            )

            allowed: List[str] = column_data.get("allowed", [])
            denied: List[str] = column_data.get("denied", [])

            # Format the Step_Log content
            output = self.format_lakeformation_output(allowed, denied)

            # Determine expansion state
            should_expand = len(denied) > 0

            # Update Step element
            if step is not None:
                step.output = output
                step.is_error = False
                await step.__aexit__(None, None, None)

                # Set expansion state after step completes
                if should_expand:
                    step.show_input = True

            return GateResult(
                passed=True,
                blocked=False,
                details={
                    "allowed": allowed,
                    "denied": denied,
                    "step_output": output,
                    "expanded": should_expand,
                },
            )

        except asyncio.TimeoutError:
            error_msg = (
                "Lake Formation authorization check failed: "
                "evaluation timed out after 10 seconds"
            )
            logger.error(error_msg)

            if step is not None:
                step.output = f"❌ {error_msg}"
                step.is_error = True
                await step.__aexit__(None, None, None)

            return GateResult(
                passed=False,
                blocked=True,
                error=error_msg,
            )

        except Exception as e:
            error_msg = (
                f"Lake Formation authorization check failed: {str(e)}"
            )
            logger.error(error_msg)

            if step is not None:
                try:
                    step.output = f"❌ {error_msg}"
                    step.is_error = True
                    await step.__aexit__(None, None, None)
                except Exception:
                    pass

            return GateResult(
                passed=False,
                blocked=True,
                error=error_msg,
            )

    async def _get_column_access(self, mode: AppMode) -> Dict[str, Any]:
        """Retrieve column access data based on current mode.

        Args:
            mode: Current application mode (LIVE or MOCK).

        Returns:
            Dictionary with 'allowed' and 'denied' column lists.
        """
        if mode == AppMode.MOCK:
            return await MockEngine().mock_gate2()
        else:
            # LIVE mode: placeholder for real Lake Formation API call
            # In production, this would call Lake Formation GetEffectivePermissions
            return await self._get_live_column_access()

    async def _get_live_column_access(self) -> Dict[str, Any]:
        """Retrieve column access from live Lake Formation service.

        Placeholder implementation — will be replaced with actual
        Lake Formation API calls when integrated with AWS services.

        Returns:
            Dictionary with 'allowed' and 'denied' column lists.

        Raises:
            NotImplementedError: Until live AWS integration is implemented.
        """
        # TODO: Implement live Lake Formation permission retrieval
        # using boto3 lakeformation client with Restricted_Role
        raise NotImplementedError(
            "Live Lake Formation integration not yet implemented. "
            "Use MOCK mode for demo."
        )

    def format_lakeformation_output(
        self, allowed: List[str], denied: List[str]
    ) -> str:
        """Format the Lake Formation column access output for the Step_Log.

        Produces a formatted string listing allowed and denied columns,
        role comparison, and a summary line with denied column count.

        This method is separated from execute() to allow independent testing
        without Chainlit dependencies.

        Args:
            allowed: List of column names the Restricted_Role can access.
            denied: List of column names blocked by Lake Formation.

        Returns:
            Formatted string for display in the Step_Log.
        """
        lines: List[str] = []

        # List allowed columns
        for col in allowed:
            lines.append(f"✅ ALLOWED: {col}")

        # List denied columns
        for col in denied:
            lines.append(f"🚫 DENIED: {col}")

        # Role comparison
        lines.append("")
        lines.append(
            "Permissive_Role (Engine_A): ALL columns accessible"
        )
        lines.append(
            "Restricted_Role (Engine_B): PII columns blocked"
        )

        # Summary line
        denied_count = len(denied)
        lines.append("")
        lines.append(
            f"Summary: {denied_count} columns denied and excluded "
            f"from response context"
        )

        return "\n".join(lines)
