"""Unit tests for the ModeDetector class."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import AppMode, ModeStatus
from mode_detector import ModeDetector, SERVICE_HEALTH_TIMEOUT


@pytest.fixture
def detector():
    """Create a ModeDetector instance for testing."""
    return ModeDetector(region="ap-southeast-1")


@pytest.fixture
def clean_env(monkeypatch):
    """Remove all AWS credential environment variables."""
    for var in [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    ]:
        monkeypatch.delenv(var, raising=False)


class TestHasAwsCredentials:
    """Tests for _has_aws_credentials method."""

    def test_no_credentials_returns_false(self, detector, clean_env):
        assert detector._has_aws_credentials() is False

    def test_access_key_and_secret_returns_true(self, detector, clean_env, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        assert detector._has_aws_credentials() is True

    def test_access_key_without_secret_returns_false(self, detector, clean_env, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        assert detector._has_aws_credentials() is False

    def test_session_token_returns_true(self, detector, clean_env, monkeypatch):
        monkeypatch.setenv("AWS_SESSION_TOKEN", "FwoGZXIvYXdzEBYaDH...")
        assert detector._has_aws_credentials() is True

    def test_aws_profile_returns_true(self, detector, clean_env, monkeypatch):
        monkeypatch.setenv("AWS_PROFILE", "demo-booth")
        assert detector._has_aws_credentials() is True

    def test_ecs_container_credentials_returns_true(self, detector, clean_env, monkeypatch):
        monkeypatch.setenv(
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "/v2/credentials/uuid-here",
        )
        assert detector._has_aws_credentials() is True


class TestCheckServiceHealth:
    """Tests for check_service_health method."""

    @pytest.mark.asyncio
    async def test_healthy_service_returns_true(self, detector):
        with patch.object(detector, "_ping_service", new_callable=AsyncMock) as mock_ping:
            mock_ping.return_value = True
            result = await detector.check_service_health("sts")
            assert result is True
            mock_ping.assert_called_once_with("sts")

    @pytest.mark.asyncio
    async def test_unhealthy_service_retries_once(self, detector):
        with patch.object(detector, "_ping_service", new_callable=AsyncMock) as mock_ping:
            mock_ping.return_value = False
            result = await detector.check_service_health("sts")
            assert result is False
            assert mock_ping.call_count == 2

    @pytest.mark.asyncio
    async def test_first_failure_then_success_returns_true(self, detector):
        with patch.object(detector, "_ping_service", new_callable=AsyncMock) as mock_ping:
            mock_ping.side_effect = [False, True]
            result = await detector.check_service_health("sts")
            assert result is True
            assert mock_ping.call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_retries_once(self, detector):
        with patch.object(detector, "_ping_service", new_callable=AsyncMock) as mock_ping:
            mock_ping.side_effect = asyncio.TimeoutError()
            result = await detector.check_service_health("sts")
            assert result is False
            assert mock_ping.call_count == 2

    @pytest.mark.asyncio
    async def test_exception_retries_once(self, detector):
        with patch.object(detector, "_ping_service", new_callable=AsyncMock) as mock_ping:
            mock_ping.side_effect = RuntimeError("Connection refused")
            result = await detector.check_service_health("sts")
            assert result is False
            assert mock_ping.call_count == 2


class TestDetectMode:
    """Tests for detect_mode method."""

    @pytest.mark.asyncio
    async def test_no_credentials_returns_mock_mode(self, detector, clean_env):
        result = await detector.detect_mode()
        assert result.mode == AppMode.MOCK
        assert result.engine_a_live is False
        assert result.engine_b_live is False
        assert len(result.failure_reasons) == 1
        assert "No AWS credentials" in result.failure_reasons[0]

    @pytest.mark.asyncio
    async def test_all_services_healthy_returns_live_mode(self, detector, clean_env, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")

        with patch.object(detector, "check_service_health", new_callable=AsyncMock) as mock_health:
            mock_health.return_value = True
            result = await detector.detect_mode()

        assert result.mode == AppMode.LIVE
        assert result.engine_a_live is True
        assert result.engine_b_live is True
        assert result.failure_reasons == []

    @pytest.mark.asyncio
    async def test_sts_failure_returns_mock_mode(self, detector, clean_env, monkeypatch):
        monkeypatch.setenv("AWS_PROFILE", "demo-booth")

        with patch.object(detector, "check_service_health", new_callable=AsyncMock) as mock_health:
            mock_health.return_value = False
            result = await detector.detect_mode()

        assert result.mode == AppMode.MOCK
        assert result.engine_a_live is False
        assert result.engine_b_live is False
        assert any("STS" in r for r in result.failure_reasons)

    @pytest.mark.asyncio
    async def test_bedrock_failure_returns_mock_mode(self, detector, clean_env, monkeypatch):
        monkeypatch.setenv("AWS_PROFILE", "demo-booth")

        async def health_side_effect(service_name):
            if service_name == "sts":
                return True
            return False

        with patch.object(detector, "check_service_health", side_effect=health_side_effect):
            result = await detector.detect_mode()

        assert result.mode == AppMode.MOCK
        assert result.engine_a_live is False
        assert result.engine_b_live is False
        assert any("Bedrock" in r for r in result.failure_reasons)

    @pytest.mark.asyncio
    async def test_glue_failure_returns_mock_mode(self, detector, clean_env, monkeypatch):
        monkeypatch.setenv("AWS_PROFILE", "demo-booth")

        async def health_side_effect(service_name):
            if service_name in ("sts", "bedrock"):
                return True
            return False

        with patch.object(detector, "check_service_health", side_effect=health_side_effect):
            result = await detector.detect_mode()

        assert result.mode == AppMode.MOCK
        assert result.engine_a_live is False
        assert result.engine_b_live is False
        assert any("Glue" in r for r in result.failure_reasons)

    @pytest.mark.asyncio
    async def test_failure_reasons_include_service_name_and_error_type(
        self, detector, clean_env, monkeypatch
    ):
        monkeypatch.setenv("AWS_PROFILE", "demo-booth")

        with patch.object(detector, "check_service_health", new_callable=AsyncMock) as mock_health:
            mock_health.return_value = False
            result = await detector.detect_mode()

        # Failure reasons should mention the service name
        assert len(result.failure_reasons) > 0
        assert any("STS" in r or "service" in r.lower() for r in result.failure_reasons)
