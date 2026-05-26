"""Unit tests for Gate 2 — Lake Formation column-level access control."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aws_demo_booth.config import AppMode
from aws_demo_booth.gates.base import GateContext, GateResult
from aws_demo_booth.gates.gate2_lakeformation import Gate2LakeFormation


class TestGate2LakeFormationInit:
    """Tests for Gate2LakeFormation initialization."""

    def test_name(self):
        gate = Gate2LakeFormation()
        assert gate.name == "Gate 2: Lake Formation"

    def test_timeout_seconds(self):
        gate = Gate2LakeFormation()
        assert gate.timeout_seconds == 10.0


class TestFormatLakeFormationOutput:
    """Tests for the format_lakeformation_output method."""

    def test_allowed_columns_prefixed(self):
        gate = Gate2LakeFormation()
        output = gate.format_lakeformation_output(
            allowed=["region", "product", "revenue"],
            denied=[],
        )
        assert "✅ ALLOWED: region" in output
        assert "✅ ALLOWED: product" in output
        assert "✅ ALLOWED: revenue" in output

    def test_denied_columns_prefixed(self):
        gate = Gate2LakeFormation()
        output = gate.format_lakeformation_output(
            allowed=[],
            denied=["email", "phone_number", "national_id"],
        )
        assert "🚫 DENIED: email" in output
        assert "🚫 DENIED: phone_number" in output
        assert "🚫 DENIED: national_id" in output

    def test_mixed_allowed_and_denied(self):
        gate = Gate2LakeFormation()
        output = gate.format_lakeformation_output(
            allowed=["region", "product", "revenue", "date"],
            denied=["email", "phone_number", "national_id"],
        )
        assert "✅ ALLOWED: region" in output
        assert "✅ ALLOWED: product" in output
        assert "🚫 DENIED: email" in output
        assert "🚫 DENIED: phone_number" in output

    def test_role_comparison_present(self):
        gate = Gate2LakeFormation()
        output = gate.format_lakeformation_output(
            allowed=["region"],
            denied=["email"],
        )
        assert "Permissive_Role (Engine_A): ALL columns accessible" in output
        assert "Restricted_Role (Engine_B): PII columns blocked" in output

    def test_summary_line_with_denied_count(self):
        gate = Gate2LakeFormation()
        output = gate.format_lakeformation_output(
            allowed=["region", "product"],
            denied=["email", "phone_number", "national_id"],
        )
        assert "Summary: 3 columns denied and excluded from response context" in output

    def test_summary_line_zero_denied(self):
        gate = Gate2LakeFormation()
        output = gate.format_lakeformation_output(
            allowed=["region", "product", "revenue"],
            denied=[],
        )
        assert "Summary: 0 columns denied and excluded from response context" in output

    def test_empty_both_lists(self):
        gate = Gate2LakeFormation()
        output = gate.format_lakeformation_output(allowed=[], denied=[])
        assert "Permissive_Role (Engine_A): ALL columns accessible" in output
        assert "Summary: 0 columns denied and excluded from response context" in output


class TestGate2Execute:
    """Tests for Gate2LakeFormation.execute method."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock GateContext for testing."""
        return GateContext(
            prompt="test prompt",
            session_mode=AppMode.MOCK,
            parent_message=MagicMock(id="msg-123"),
        )

    @pytest.mark.asyncio
    @patch("aws_demo_booth.gates.gate2_lakeformation.cl", None, create=True)
    async def test_execute_mock_mode_success(self, mock_context):
        """Test successful execution in MOCK mode without Chainlit."""
        gate = Gate2LakeFormation()

        # Patch the _get_column_access to avoid actual mock delay
        mock_data = {
            "allowed": ["region", "product", "revenue", "date"],
            "denied": ["email", "phone_number", "national_id"],
        }
        gate._get_column_access = AsyncMock(return_value=mock_data)

        result = await gate.execute(mock_context)

        assert result.passed is True
        assert result.blocked is False
        assert result.error is None
        assert result.details["allowed"] == ["region", "product", "revenue", "date"]
        assert result.details["denied"] == ["email", "phone_number", "national_id"]
        assert result.details["expanded"] is True

    @pytest.mark.asyncio
    @patch("aws_demo_booth.gates.gate2_lakeformation.cl", None, create=True)
    async def test_execute_all_permitted_collapsed(self, mock_context):
        """Test that expansion state is False when all columns permitted."""
        gate = Gate2LakeFormation()

        mock_data = {
            "allowed": ["region", "product", "revenue", "date"],
            "denied": [],
        }
        gate._get_column_access = AsyncMock(return_value=mock_data)

        result = await gate.execute(mock_context)

        assert result.passed is True
        assert result.blocked is False
        assert result.details["expanded"] is False

    @pytest.mark.asyncio
    @patch("aws_demo_booth.gates.gate2_lakeformation.cl", None, create=True)
    async def test_execute_timeout_blocks_data_retrieval(self, mock_context):
        """Test that timeout results in blocked=True."""
        gate = Gate2LakeFormation()

        async def slow_access(mode):
            await asyncio.sleep(20)  # Exceeds 10s timeout
            return {"allowed": [], "denied": []}

        gate._get_column_access = slow_access
        # Override timeout for faster test
        gate.timeout_seconds = 0.1

        result = await gate.execute(mock_context)

        assert result.passed is False
        assert result.blocked is True
        assert result.error is not None
        assert "timed out" in result.error

    @pytest.mark.asyncio
    @patch("aws_demo_booth.gates.gate2_lakeformation.cl", None, create=True)
    async def test_execute_exception_blocks_data_retrieval(self, mock_context):
        """Test that an exception results in blocked=True."""
        gate = Gate2LakeFormation()

        gate._get_column_access = AsyncMock(
            side_effect=RuntimeError("Service unreachable")
        )

        result = await gate.execute(mock_context)

        assert result.passed is False
        assert result.blocked is True
        assert result.error is not None
        assert "Service unreachable" in result.error

    @pytest.mark.asyncio
    @patch("aws_demo_booth.gates.gate2_lakeformation.cl", None, create=True)
    async def test_execute_live_mode_not_implemented(self):
        """Test that LIVE mode raises NotImplementedError (placeholder)."""
        gate = Gate2LakeFormation()
        context = GateContext(
            prompt="test",
            session_mode=AppMode.LIVE,
            parent_message=MagicMock(id="msg-456"),
        )

        result = await gate.execute(context)

        assert result.passed is False
        assert result.blocked is True
        assert "not yet implemented" in result.error
