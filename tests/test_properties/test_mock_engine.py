"""Property-based tests for mock engine timing and content.

Feature: safe-haven-demo-booth
Property 10: Mock gate timing
"""

import asyncio
import time

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from mock_engine import MockEngine


# --- Strategies ---


def _arbitrary_prompt() -> st.SearchStrategy[str]:
    """Generate arbitrary prompt strings for mock_gate3 invocation."""
    return st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)),
        min_size=1,
        max_size=200,
    )


# =============================================================================
# Property 10: Mock gate simulation returns content within timing bounds
# =============================================================================


class TestProperty10MockGateTiming:
    """Property 10: Mock gate simulation returns content within timing bounds.

    For any mock gate invocation (Gate_1, Gate_2, or Gate_3), the simulated
    response SHALL be returned after a delay of between 1 and 2 seconds
    (inclusive), and SHALL contain the appropriate pre-programmed content.

    **Validates: Requirements 7.3, 7.4, 7.5**
    """

    @settings(max_examples=3, deadline=None)
    @given(data=st.data())
    @pytest.mark.asyncio
    async def test_mock_gate1_timing_and_content(self, data):
        """mock_gate1 returns Macie findings within 1-2 second timing bounds.

        Feature: safe-haven-demo-booth, Property 10: Mock gate timing
        **Validates: Requirements 7.3**
        """
        engine = MockEngine()

        start = time.monotonic()
        result = await engine.mock_gate1()
        elapsed = time.monotonic() - start

        # Timing: response returned after 1-2 seconds (with small tolerance for overhead)
        assert elapsed >= 1.0, (
            f"mock_gate1 returned too quickly: {elapsed:.3f}s (expected >= 1.0s)"
        )
        assert elapsed <= 2.5, (
            f"mock_gate1 returned too slowly: {elapsed:.3f}s (expected <= 2.0s + tolerance)"
        )

        # Content: must contain pre-programmed Macie findings
        assert isinstance(result, dict), "mock_gate1 must return a dict"
        assert "findings_count" in result, "mock_gate1 must contain 'findings_count'"
        assert "categories" in result, "mock_gate1 must contain 'categories'"
        assert isinstance(result["findings_count"], int), "findings_count must be an int"
        assert result["findings_count"] > 0, "findings_count must be positive"
        assert isinstance(result["categories"], dict), "categories must be a dict"
        assert len(result["categories"]) > 0, "categories must not be empty"

    @settings(max_examples=3, deadline=None)
    @given(data=st.data())
    @pytest.mark.asyncio
    async def test_mock_gate2_timing_and_content(self, data):
        """mock_gate2 returns column access list within 1-2 second timing bounds.

        Feature: safe-haven-demo-booth, Property 10: Mock gate timing
        **Validates: Requirements 7.4**
        """
        engine = MockEngine()

        start = time.monotonic()
        result = await engine.mock_gate2()
        elapsed = time.monotonic() - start

        # Timing: response returned after 1-2 seconds (with small tolerance for overhead)
        assert elapsed >= 1.0, (
            f"mock_gate2 returned too quickly: {elapsed:.3f}s (expected >= 1.0s)"
        )
        assert elapsed <= 2.5, (
            f"mock_gate2 returned too slowly: {elapsed:.3f}s (expected <= 2.0s + tolerance)"
        )

        # Content: must contain pre-programmed column access list
        assert isinstance(result, dict), "mock_gate2 must return a dict"
        assert "allowed" in result, "mock_gate2 must contain 'allowed'"
        assert "denied" in result, "mock_gate2 must contain 'denied'"
        assert isinstance(result["allowed"], list), "allowed must be a list"
        assert isinstance(result["denied"], list), "denied must be a list"
        assert len(result["allowed"]) > 0, "allowed columns must not be empty"
        assert len(result["denied"]) > 0, "denied columns must not be empty"

    @settings(max_examples=3, deadline=None)
    @given(prompt=_arbitrary_prompt())
    @pytest.mark.asyncio
    async def test_mock_gate3_timing_and_content(self, prompt: str):
        """mock_gate3 returns guardrail action within 1-2 second timing bounds.

        Feature: safe-haven-demo-booth, Property 10: Mock gate timing
        **Validates: Requirements 7.5**
        """
        engine = MockEngine()

        start = time.monotonic()
        result = await engine.mock_gate3(prompt)
        elapsed = time.monotonic() - start

        # Timing: response returned after 1-2 seconds (with small tolerance for overhead)
        assert elapsed >= 1.0, (
            f"mock_gate3 returned too quickly: {elapsed:.3f}s (expected >= 1.0s)"
        )
        assert elapsed <= 2.5, (
            f"mock_gate3 returned too slowly: {elapsed:.3f}s (expected <= 2.0s + tolerance)"
        )

        # Content: must contain keyword-based guardrail action
        assert isinstance(result, dict), "mock_gate3 must return a dict"
        assert "input_action" in result, "mock_gate3 must contain 'input_action'"
        assert "output_action" in result, "mock_gate3 must contain 'output_action'"
        assert result["input_action"] in ("PASS", "BLOCKED"), (
            f"input_action must be PASS or BLOCKED, got: {result['input_action']}"
        )
        assert result["output_action"] in ("PASS", "BLOCKED"), (
            f"output_action must be PASS or BLOCKED, got: {result['output_action']}"
        )
