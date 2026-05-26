"""Health check endpoint for the Safe Haven Demo Booth.

Exposes a /health endpoint on port 8000 that returns:
- HTTP 200 with {"status": "healthy"} when the app is initialized and ready
- HTTP 503 with {"status": "unhealthy", "reason": "..."} when not ready

The readiness flag is set to True after the application module loads
successfully and the Chainlit app is ready to accept connections.

Requirements: 10.6, 10.7, 10.8
"""

import json
import logging
import time

logger = logging.getLogger(__name__)

# Module-level readiness state
_app_ready: bool = False
_startup_time: float = time.time()

# Maximum allowed initialization time (seconds)
INIT_TIMEOUT_SECONDS: int = 30


def set_ready() -> None:
    """Mark the application as ready to accept connections.

    Called after the Chainlit app module has loaded successfully
    and the application is initialized.
    """
    global _app_ready
    _app_ready = True
    logger.info("Application health check: marked as ready")


def set_not_ready(reason: str = "") -> None:
    """Mark the application as not ready.

    Args:
        reason: Optional reason for the unhealthy state.
    """
    global _app_ready
    _app_ready = False
    if reason:
        logger.warning("Application health check: marked as not ready - %s", reason)


def is_ready() -> bool:
    """Check if the application is ready to accept connections.

    Returns:
        True if the app is initialized and ready, False otherwise.
    """
    return _app_ready


def get_health_status() -> tuple:
    """Get the current health status of the application.

    Returns:
        A tuple of (status_code, response_body) where:
        - status_code is 200 if healthy, 503 if unhealthy
        - response_body is a dict with status and optional reason
    """
    if _app_ready:
        return 200, {"status": "healthy"}

    # Determine the reason for being unhealthy
    elapsed = time.time() - _startup_time
    if elapsed > INIT_TIMEOUT_SECONDS:
        reason = (
            f"Initialization did not complete within "
            f"{INIT_TIMEOUT_SECONDS} seconds"
        )
    else:
        reason = "Application is still initializing"

    return 503, {"status": "unhealthy", "reason": reason}


async def health_endpoint(scope, receive, send):
    """ASGI endpoint handler for /health.

    This is a raw ASGI handler that can be mounted on the Chainlit
    (Starlette) application as a custom route.

    Args:
        scope: ASGI scope dict.
        receive: ASGI receive callable.
        send: ASGI send callable.
    """
    status_code, body = get_health_status()
    body_bytes = json.dumps(body).encode("utf-8")

    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(body_bytes)).encode("utf-8")],
            ],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": body_bytes,
        }
    )


def reset_for_testing() -> None:
    """Reset module state for testing purposes.

    This should only be called in test code to reset the module-level
    state between test cases.
    """
    global _app_ready, _startup_time
    _app_ready = False
    _startup_time = time.time()
