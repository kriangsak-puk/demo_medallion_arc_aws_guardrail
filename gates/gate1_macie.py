"""Gate 1 — Amazon Macie Scan Results Display.

Displays pre-computed Amazon Macie PII detection findings as an
informational step in the Engine_B pipeline. This gate never blocks;
it only shows scan results to the visitor.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from config import AppMode
from gates.base import Gate, GateContext, GateResult
from mock_engine import MockEngine


class Gate1Macie(Gate):
    """Gate 1: Display pre-computed Amazon Macie PII scan results.

    This gate is informational only — it never blocks the pipeline.
    It retrieves Macie scan findings (from mock or live source) and
    displays them in a Chainlit Step_Log element.

    The Step_Log is auto-expanded when PII is detected, and left
    collapsed when the data is clean or results are unavailable.
    """

    TIMEOUT_SECONDS: float = 5.0

    def __init__(self) -> None:
        super().__init__(
            name="[Gate 1] Amazon Macie Scan Results",
            timeout_seconds=self.TIMEOUT_SECONDS,
        )

    async def execute(self, context: GateContext) -> GateResult:
        """Execute Gate 1: retrieve and display Macie scan results.

        Args:
            context: The gate context containing prompt, session mode,
                and parent message.

        Returns:
            A GateResult with passed=True and blocked=False (informational gate).
            The details dict contains 'findings_count', 'categories',
            'step_output', and 'expanded'.
        """
        findings_count: Optional[int] = None
        categories: Optional[Dict[str, int]] = None

        try:
            findings_count, categories = await asyncio.wait_for(
                self._get_macie_results(context.session_mode),
                timeout=self.timeout_seconds,
            )
        except (asyncio.TimeoutError, Exception):
            # Results unavailable — proceed without blocking
            findings_count = None
            categories = None

        step_output = self.format_macie_output(findings_count, categories)
        expanded = self._should_expand(findings_count)

        # If Chainlit is available, create the Step element
        await self._display_step(context, step_output, expanded)

        return GateResult(
            passed=True,
            blocked=False,
            details={
                "findings_count": findings_count,
                "categories": categories,
                "step_output": step_output,
                "expanded": expanded,
            },
        )

    def format_macie_output(
        self,
        findings_count: Optional[int],
        categories: Optional[Dict[str, int]],
    ) -> str:
        """Format Macie scan results into a display string.

        This method is separated from execute() to allow independent testing
        without Chainlit dependencies.

        Args:
            findings_count: Number of PII findings, or None if unavailable.
            categories: Dict mapping PII category names to occurrence counts,
                or None if unavailable.

        Returns:
            Formatted string for the Step_Log content.
        """
        if findings_count is None:
            return "Amazon Macie: Results unavailable"

        if findings_count == 0:
            return "Amazon Macie: No PII detected in source data — data is clean"

        lines: List[str] = [
            f"Amazon Macie: {findings_count} PII findings detected in source data"
        ]

        if categories:
            for category_name, count in categories.items():
                lines.append(f"  • {category_name}: {count}")

        return "\n".join(lines)

    def _should_expand(self, findings_count: Optional[int]) -> bool:
        """Determine whether the Step_Log should be auto-expanded.

        Expanded when PII is detected (findings_count > 0).
        Collapsed when clean (findings_count == 0) or unavailable (None).

        Args:
            findings_count: Number of PII findings, or None if unavailable.

        Returns:
            True if the Step_Log should be expanded, False otherwise.
        """
        if findings_count is None:
            return False
        return findings_count > 0

    async def _get_macie_results(
        self, session_mode: AppMode
    ) -> Tuple[int, Dict[str, int]]:
        """Retrieve Macie scan results based on the current mode.

        Args:
            session_mode: Current application mode (LIVE or MOCK).

        Returns:
            Tuple of (findings_count, categories dict).
        """
        if session_mode == AppMode.MOCK:
            mock = MockEngine()
            result = await mock.mock_gate1()
            return result["findings_count"], result["categories"]

        # LIVE mode: placeholder for actual AWS Macie integration
        # In production, this would call the Macie API or read pre-computed
        # results from S3/metadata store.
        return await self._get_live_macie_results()

    async def _get_live_macie_results(self) -> Tuple[int, Dict[str, int]]:
        """Return pre-computed Macie scan results for the Gold Table.

        Macie results are pre-computed (scans already ran on the source data).
        This gate displays those findings informationally. In a full production
        system, this would read findings from S3 or the Macie API.

        Returns:
            Tuple of (findings_count, categories dict).
        """
        # Pre-computed Macie findings for the Gold Table
        # These reflect the PII columns present in the data
        return 7, {
            "EMAIL_ADDRESS": 1,
            "PHONE_NUMBER": 1,
            "NATIONAL_ID": 1,
            "CREDIT_CARD_NUMBER": 1,
            "DATE_OF_BIRTH": 1,
            "ADDRESS": 2,
        }

    async def _display_step(
        self, context: GateContext, output: str, expanded: bool
    ) -> None:
        """Display the Step_Log in Chainlit UI.

        Attempts to create a Chainlit Step element. If Chainlit is not
        available (e.g., during testing), this is a no-op.

        Args:
            context: The gate context with parent_message.
            output: The formatted step output text.
            expanded: Whether the step should be auto-expanded.
        """
        try:
            import chainlit as cl

            async with cl.Step(
                name=self.name,
                parent_id=context.parent_message.id if context.parent_message else None,
            ) as step:
                step.output = output
                step.is_error = False
                # Chainlit Step expansion: show_input controls visibility
                if expanded:
                    step.show_input = True
        except (ImportError, Exception):
            # Chainlit not available or parent_message is a mock — skip UI
            pass
