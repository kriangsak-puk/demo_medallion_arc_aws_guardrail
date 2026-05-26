"""Unit tests for Gate 1 — Amazon Macie Results Display."""

import pytest
import pytest_asyncio

from config import AppMode
from gates.base import GateContext, GateResult
from gates.gate1_macie import Gate1Macie


class TestGate1MacieInit:
    """Tests for Gate1Macie initialization."""

    def test_name(self):
        gate = Gate1Macie()
        assert gate.name == "[Gate 1] Amazon Macie Scan Results"

    def test_timeout(self):
        gate = Gate1Macie()
        assert gate.timeout_seconds == 5.0


class TestFormatMacieOutput:
    """Tests for the format_macie_output method."""

    def test_unavailable_results(self):
        gate = Gate1Macie()
        output = gate.format_macie_output(None, None)
        assert output == "Amazon Macie: Results unavailable"

    def test_zero_findings(self):
        gate = Gate1Macie()
        output = gate.format_macie_output(0, {})
        assert output == "Amazon Macie: No PII detected in source data — data is clean"

    def test_zero_findings_none_categories(self):
        gate = Gate1Macie()
        output = gate.format_macie_output(0, None)
        assert output == "Amazon Macie: No PII detected in source data — data is clean"

    def test_pii_findings_with_categories(self):
        gate = Gate1Macie()
        categories = {
            "EMAIL_ADDRESS": 2,
            "PHONE_NUMBER": 1,
            "NATIONAL_ID": 3,
        }
        output = gate.format_macie_output(6, categories)
        assert "Amazon Macie: 6 PII findings detected in source data" in output
        assert "EMAIL_ADDRESS: 2" in output
        assert "PHONE_NUMBER: 1" in output
        assert "NATIONAL_ID: 3" in output

    def test_single_finding_single_category(self):
        gate = Gate1Macie()
        categories = {"EMAIL_ADDRESS": 1}
        output = gate.format_macie_output(1, categories)
        assert "Amazon Macie: 1 PII findings detected in source data" in output
        assert "EMAIL_ADDRESS: 1" in output

    def test_findings_with_empty_categories(self):
        gate = Gate1Macie()
        output = gate.format_macie_output(3, {})
        assert "Amazon Macie: 3 PII findings detected in source data" in output

    def test_findings_with_none_categories(self):
        gate = Gate1Macie()
        output = gate.format_macie_output(3, None)
        assert "Amazon Macie: 3 PII findings detected in source data" in output


class TestShouldExpand:
    """Tests for the _should_expand method."""

    def test_expand_when_pii_detected(self):
        gate = Gate1Macie()
        assert gate._should_expand(3) is True

    def test_expand_when_single_finding(self):
        gate = Gate1Macie()
        assert gate._should_expand(1) is True

    def test_collapse_when_zero_findings(self):
        gate = Gate1Macie()
        assert gate._should_expand(0) is False

    def test_collapse_when_unavailable(self):
        gate = Gate1Macie()
        assert gate._should_expand(None) is False


class TestGate1MacieExecute:
    """Tests for the execute method in MOCK mode."""

    @pytest.mark.asyncio
    async def test_execute_mock_mode_returns_gate_result(self):
        gate = Gate1Macie()
        context = GateContext(
            prompt="test prompt",
            session_mode=AppMode.MOCK,
            parent_message=None,
        )
        result = await gate.execute(context)

        assert isinstance(result, GateResult)
        assert result.passed is True
        assert result.blocked is False
        assert result.error is None

    @pytest.mark.asyncio
    async def test_execute_mock_mode_has_findings(self):
        gate = Gate1Macie()
        context = GateContext(
            prompt="test prompt",
            session_mode=AppMode.MOCK,
            parent_message=None,
        )
        result = await gate.execute(context)

        # MockEngine.mock_gate1() returns findings_count=7
        assert result.details["findings_count"] == 7
        assert result.details["categories"] is not None
        assert result.details["expanded"] is True

    @pytest.mark.asyncio
    async def test_execute_mock_mode_step_output_format(self):
        gate = Gate1Macie()
        context = GateContext(
            prompt="test prompt",
            session_mode=AppMode.MOCK,
            parent_message=None,
        )
        result = await gate.execute(context)

        step_output = result.details["step_output"]
        assert "Amazon Macie: 7 PII findings detected in source data" in step_output
        assert "EMAIL_ADDRESS" in step_output
        assert "PHONE_NUMBER" in step_output
        assert "NATIONAL_ID" in step_output

    @pytest.mark.asyncio
    async def test_execute_live_mode_unavailable_graceful(self):
        """Live mode raises NotImplementedError, which is caught as unavailable."""
        gate = Gate1Macie()
        context = GateContext(
            prompt="test prompt",
            session_mode=AppMode.LIVE,
            parent_message=None,
        )
        result = await gate.execute(context)

        # Should handle the error gracefully
        assert result.passed is True
        assert result.blocked is False
        assert result.details["findings_count"] is None
        assert result.details["step_output"] == "Amazon Macie: Results unavailable"
        assert result.details["expanded"] is False

    @pytest.mark.asyncio
    async def test_gate_is_informational_never_blocks(self):
        """Gate 1 is informational only — it should never block the pipeline."""
        gate = Gate1Macie()
        context = GateContext(
            prompt="any prompt",
            session_mode=AppMode.MOCK,
            parent_message=None,
        )
        result = await gate.execute(context)

        assert result.blocked is False
        assert result.passed is True
