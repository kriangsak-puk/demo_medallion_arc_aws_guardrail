"""Mode detection module.

Detects whether the application should run in Live_Mode or Mock_Mode
based on AWS credential availability and service health checks.
"""

import asyncio
import logging
import os
from typing import Optional

import boto3
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
)

from aws_demo_booth.config import AppMode, ModeStatus

logger = logging.getLogger(__name__)

# Timeout for service health checks in seconds
SERVICE_HEALTH_TIMEOUT = 5


class ModeDetector:
    """Detects application operating mode based on credentials and service health.

    Checks for AWS credentials in the environment. If absent, returns Mock mode.
    If credentials are present, performs service health checks with a 5-second
    timeout and a single retry before falling back to Mock mode for the
    affected engine.
    """

    def __init__(self, region: str = "ap-southeast-1") -> None:
        self._region = region

    def _has_aws_credentials(self) -> bool:
        """Check if AWS credentials are available in the environment.

        Checks for:
        - AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (explicit credentials)
        - AWS_SESSION_TOKEN (temporary credentials)
        - AWS_PROFILE (named profile)
        - AWS_CONTAINER_CREDENTIALS_RELATIVE_URI (ECS container credentials)
        """
        # Explicit access key credentials
        if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get(
            "AWS_SECRET_ACCESS_KEY"
        ):
            return True

        # Session token (temporary credentials from STS)
        if os.environ.get("AWS_SESSION_TOKEN"):
            return True

        # Named profile
        if os.environ.get("AWS_PROFILE"):
            return True

        # ECS container credentials endpoint
        if os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"):
            return True

        return False

    async def check_service_health(self, service_name: str) -> bool:
        """Check health of a specific AWS service with 5-second timeout.

        Attempts to call the service. If the call fails or times out,
        retries once. Returns True if the service is reachable, False otherwise.

        Args:
            service_name: The AWS service name to check (e.g., 'sts', 'bedrock').

        Returns:
            True if the service responded successfully, False otherwise.
        """
        for attempt in range(2):  # Initial attempt + 1 retry
            try:
                result = await asyncio.wait_for(
                    self._ping_service(service_name),
                    timeout=SERVICE_HEALTH_TIMEOUT,
                )
                if result:
                    return True
            except asyncio.TimeoutError:
                logger.warning(
                    "Service health check timed out for '%s' (attempt %d/2)",
                    service_name,
                    attempt + 1,
                )
            except Exception as e:
                logger.warning(
                    "Service health check failed for '%s' (attempt %d/2): %s: %s",
                    service_name,
                    attempt + 1,
                    type(e).__name__,
                    str(e),
                )

        return False

    async def _ping_service(self, service_name: str) -> bool:
        """Ping an AWS service to verify connectivity.

        Runs the blocking boto3 call in a thread executor to avoid
        blocking the event loop.

        Args:
            service_name: The AWS service name to ping.

        Returns:
            True if the service responded, False otherwise.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._ping_service_sync, service_name)

    def _ping_service_sync(self, service_name: str) -> bool:
        """Synchronous service ping using boto3.

        Uses STS GetCallerIdentity as the health check since it's
        lightweight and validates credentials + connectivity.

        Args:
            service_name: The AWS service name to check.

        Returns:
            True if the call succeeded, False otherwise.
        """
        try:
            if service_name == "sts":
                client = boto3.client("sts", region_name=self._region)
                client.get_caller_identity()
                return True
            elif service_name == "bedrock":
                client = boto3.client("bedrock", region_name=self._region)
                client.list_foundation_models(maxResults=1)
                return True
            elif service_name == "bedrock-runtime":
                # For bedrock-runtime, we just verify the client can be created
                # and credentials are valid via STS
                client = boto3.client("sts", region_name=self._region)
                client.get_caller_identity()
                return True
            elif service_name == "glue":
                client = boto3.client("glue", region_name=self._region)
                client.get_databases(MaxResults=1)
                return True
            else:
                # Default: use STS as a generic connectivity check
                client = boto3.client("sts", region_name=self._region)
                client.get_caller_identity()
                return True
        except (NoCredentialsError, ClientError, BotoCoreError) as e:
            logger.debug(
                "Service ping failed for '%s': %s: %s",
                service_name,
                type(e).__name__,
                str(e),
            )
            return False

    async def detect_mode(self) -> ModeStatus:
        """Detect the application operating mode.

        Logic:
        1. If no AWS credentials are found in the environment, return Mock mode.
        2. If credentials are present, check service health with 5-second timeout.
        3. On failure, retry once. If retry also fails, switch to Mock mode for
           the affected engine and log the failure reason.

        Returns:
            ModeStatus with the detected mode, per-engine live/mock status,
            and any failure reasons.
        """
        failure_reasons: list[str] = []

        # Step 1: Check for AWS credentials
        if not self._has_aws_credentials():
            logger.info(
                "No AWS credentials found in environment. Launching in Mock_Mode."
            )
            return ModeStatus(
                mode=AppMode.MOCK,
                engine_a_live=False,
                engine_b_live=False,
                failure_reasons=["No AWS credentials found in environment"],
            )

        # Step 2: Credentials present — check service health
        engine_a_live = True
        engine_b_live = True

        # Check STS (needed by both engines for role assumption)
        sts_healthy = await self.check_service_health("sts")
        if not sts_healthy:
            reason = "STS service unreachable — cannot assume IAM roles"
            logger.error(reason)
            failure_reasons.append(reason)
            engine_a_live = False
            engine_b_live = False

        # Check Bedrock (needed by both engines for model inference)
        if engine_a_live or engine_b_live:
            bedrock_healthy = await self.check_service_health("bedrock")
            if not bedrock_healthy:
                reason = "Bedrock service unreachable — model inference unavailable"
                logger.error(reason)
                failure_reasons.append(reason)
                engine_a_live = False
                engine_b_live = False

        # Check Glue (needed by both engines for Gold_Table queries)
        if engine_a_live or engine_b_live:
            glue_healthy = await self.check_service_health("glue")
            if not glue_healthy:
                reason = "Glue service unreachable — cannot query Gold_Table"
                logger.error(reason)
                failure_reasons.append(reason)
                engine_a_live = False
                engine_b_live = False

        # Determine overall mode
        if not engine_a_live and not engine_b_live:
            mode = AppMode.MOCK
        elif not engine_a_live or not engine_b_live:
            # Partial failure — still considered LIVE overall but with
            # per-engine mock fallback
            mode = AppMode.LIVE
        else:
            mode = AppMode.LIVE

        return ModeStatus(
            mode=mode,
            engine_a_live=engine_a_live,
            engine_b_live=engine_b_live,
            failure_reasons=failure_reasons,
        )
