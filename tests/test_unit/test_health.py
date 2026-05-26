"""Unit tests for the health check endpoint module.

Tests the health endpoint logic including:
- Returns HTTP 200 with {"status": "healthy"} when app is ready
- Returns HTTP 503 with {"status": "unhealthy", "reason": "..."} when not ready
- Readiness state management (set_ready, set_not_ready, is_ready)
- ASGI endpoint handler behavior

Requirements: 10.6, 10.7, 10.8
"""

import json
import time
from unittest.mock import patch

import pytest

from aws_demo_booth.health import (
    INIT_TIMEOUT_SECONDS,
    get_health_status,
    health_endpoint,
    is_ready,
    reset_for_testing,
    set_not_ready,
    set_ready,
)


@pytest.fixture(autouse=True)
def reset_health_state():
    """Reset health module state before each test."""
    reset_for_testing()
    yield
    reset_for_testing()


class TestReadinessState:
    """Tests for readiness state management."""

    def test_initial_state_is_not_ready(self):
        """App starts in not-ready state."""
        assert is_ready() is False

    def test_set_ready_marks_app_as_ready(self):
        """set_ready() transitions to ready state."""
        set_ready()
        assert is_ready() is True

    def test_set_not_ready_marks_app_as_not_ready(self):
        """set_not_ready() transitions back to not-ready state."""
        set_ready()
        assert is_ready() is True
        set_not_ready()
        assert is_ready() is False

    def test_set_not_ready_with_reason(self):
        """set_not_ready() accepts an optional reason string."""
        set_ready()
        set_not_ready(reason="Service unavailable")
        assert is_ready() is False

    def test_multiple_set_ready_calls_are_idempotent(self):
        """Calling set_ready() multiple times keeps app ready."""
        set_ready()
        set_ready()
        set_ready()
        assert is_ready() is True


class TestGetHealthStatus:
    """Tests for get_health_status() function."""

    def test_returns_200_when_ready(self):
        """Returns HTTP 200 and healthy status when app is ready."""
        set_ready()
        status_code, body = get_health_status()
        assert status_code == 200
        assert body == {"status": "healthy"}

    def test_returns_503_when_not_ready(self):
        """Returns HTTP 503 and unhealthy status when app is not ready."""
        status_code, body = get_health_status()
        assert status_code == 503
        assert body["status"] == "unhealthy"
        assert "reason" in body

    def test_unhealthy_reason_during_initialization(self):
        """Reason indicates app is still initializing when within timeout."""
        status_code, body = get_health_status()
        assert status_code == 503
        assert "initializing" in body["reason"].lower()

    def test_unhealthy_reason_after_timeout(self):
        """Reason indicates timeout exceeded when past INIT_TIMEOUT_SECONDS."""
        # Simulate startup time being far in the past
        with patch(
            "aws_demo_booth.health._startup_time",
            time.time() - INIT_TIMEOUT_SECONDS - 1,
        ):
            status_code, body = get_health_status()
        assert status_code == 503
        assert "did not complete" in body["reason"].lower()
        assert str(INIT_TIMEOUT_SECONDS) in body["reason"]

    def test_healthy_response_body_is_json_serializable(self):
        """Healthy response body can be serialized to JSON."""
        set_ready()
        _, body = get_health_status()
        json_str = json.dumps(body)
        assert json.loads(json_str) == body

    def test_unhealthy_response_body_is_json_serializable(self):
        """Unhealthy response body can be serialized to JSON."""
        _, body = get_health_status()
        json_str = json.dumps(body)
        assert json.loads(json_str) == body


class TestHealthEndpoint:
    """Tests for the ASGI health_endpoint handler."""

    @pytest.mark.asyncio
    async def test_endpoint_returns_200_when_ready(self):
        """ASGI handler sends 200 response when app is ready."""
        set_ready()
        responses = []

        async def mock_send(message):
            responses.append(message)

        await health_endpoint({}, None, mock_send)

        # Check response start
        assert responses[0]["type"] == "http.response.start"
        assert responses[0]["status"] == 200

        # Check content-type header
        headers = dict(responses[0]["headers"])
        assert headers[b"content-type"] == b"application/json"

        # Check response body
        assert responses[1]["type"] == "http.response.body"
        body = json.loads(responses[1]["body"])
        assert body == {"status": "healthy"}

    @pytest.mark.asyncio
    async def test_endpoint_returns_503_when_not_ready(self):
        """ASGI handler sends 503 response when app is not ready."""
        responses = []

        async def mock_send(message):
            responses.append(message)

        await health_endpoint({}, None, mock_send)

        # Check response start
        assert responses[0]["type"] == "http.response.start"
        assert responses[0]["status"] == 503

        # Check response body
        assert responses[1]["type"] == "http.response.body"
        body = json.loads(responses[1]["body"])
        assert body["status"] == "unhealthy"
        assert "reason" in body

    @pytest.mark.asyncio
    async def test_endpoint_content_length_header_matches_body(self):
        """Content-Length header matches actual body size."""
        set_ready()
        responses = []

        async def mock_send(message):
            responses.append(message)

        await health_endpoint({}, None, mock_send)

        headers = dict(responses[0]["headers"])
        content_length = int(headers[b"content-length"])
        actual_length = len(responses[1]["body"])
        assert content_length == actual_length

    @pytest.mark.asyncio
    async def test_endpoint_sends_two_messages(self):
        """ASGI handler sends exactly two messages (start + body)."""
        set_ready()
        responses = []

        async def mock_send(message):
            responses.append(message)

        await health_endpoint({}, None, mock_send)
        assert len(responses) == 2

    @pytest.mark.asyncio
    async def test_endpoint_transitions_with_state_change(self):
        """Endpoint reflects state changes between calls."""
        responses_before = []
        responses_after = []

        async def mock_send_before(message):
            responses_before.append(message)

        async def mock_send_after(message):
            responses_after.append(message)

        # First call — not ready
        await health_endpoint({}, None, mock_send_before)
        assert responses_before[0]["status"] == 503

        # Set ready
        set_ready()

        # Second call — ready
        await health_endpoint({}, None, mock_send_after)
        assert responses_after[0]["status"] == 200


class TestInitTimeoutConstant:
    """Tests for the initialization timeout constant."""

    def test_init_timeout_is_30_seconds(self):
        """INIT_TIMEOUT_SECONDS is set to 30 per requirement 10.6."""
        assert INIT_TIMEOUT_SECONDS == 30
