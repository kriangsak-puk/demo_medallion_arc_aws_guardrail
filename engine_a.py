"""Engine A — Data Swamp Pipeline.

The ungoverned AI pipeline that uses the Permissive_Role to query the
Gold_Table (sees all columns including PII) and does NOT apply Guardrails.
Demonstrates the risks of operating without data security controls.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

from agent_wrapper import (
    AgentResponse,
    RoleAssumptionError,
    StrandsAgentWrapper,
)
from config import AppMode
from mock_engine import MockEngine

logger = logging.getLogger(__name__)

# Engine A timeout in seconds
ENGINE_A_TIMEOUT_SECONDS = 180


@dataclass
class EngineResult:
    """Result from an engine pipeline execution.

    Attributes:
        engine_label: Display label for the engine panel (e.g., "🧟 Data Swamp").
        response_text: The generated response text.
        is_blocked: Whether the response was blocked by guardrails.
        annotations: List of annotation labels (e.g., ["⚠️ Ungoverned — Data Leaked"]).
        guardrail_action: The guardrail action taken (None for Engine A).
        sensitive_detected: Whether sensitive content was detected in the response.
        error: Error message if the engine failed, None otherwise.
        duration_ms: Processing duration in milliseconds.
    """

    engine_label: str
    response_text: str
    is_blocked: bool
    annotations: List[str] = field(default_factory=list)
    guardrail_action: Optional[str] = None
    sensitive_detected: bool = False
    error: Optional[str] = None
    duration_ms: float = 0.0


# PII detection patterns
_PHONE_PATTERNS = [
    # International format: +XX-XX-XXX-XXXX, +XX-XXX-XXX-XXXX
    re.compile(r"\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{2,4}[-.\s]?\d{3,4}"),
    # US format: (XXX) XXX-XXXX
    re.compile(r"\(\d{3}\)\s?\d{3}[-.\s]?\d{4}"),
]

_EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
)

# Thai national ID format: X-XXXX-XXXXX-XX-X
_NATIONAL_ID_PATTERN = re.compile(
    r"\d{1}-\d{4}-\d{5}-\d{2}-\d{1}"
)

# Company codes: XXXX-INTERNAL-XXXX-CONFIDENTIAL
_COMPANY_CODE_PATTERN = re.compile(
    r"[A-Z0-9]{2,}-INTERNAL-[A-Z0-9]{2,}-CONFIDENTIAL",
    re.IGNORECASE,
)


class EngineA:
    """Engine A — the ungoverned 'Data Swamp' pipeline.

    Uses the Permissive_Role to query the Gold_Table with access to all
    columns including PII. Does NOT apply Bedrock Guardrails. Detects
    and annotates sensitive content in responses to highlight data leakage.
    """

    ENGINE_LABEL = "🧟 Data Swamp"

    def __init__(self, agent: StrandsAgentWrapper, mode: AppMode) -> None:
        """Initialize Engine A.

        Args:
            agent: The Strands Agent wrapper for LLM invocation.
            mode: Current application operating mode (LIVE or MOCK).
        """
        self._agent = agent
        self._mode = mode
        self._mock_engine = MockEngine()

    async def process(
        self,
        prompt: str,
        on_token: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> EngineResult:
        """Process an attack prompt through the ungoverned pipeline.

        In Live mode:
        1. Assume Permissive_Role via STS
        2. Query Gold_Table (all columns including PII)
        3. Invoke Bedrock model WITHOUT guardrails
        4. Stream response tokens to UI via on_token callback
        5. Detect sensitive content in response
        6. Annotate if PII detected

        In Mock mode:
        - Use MockEngine.mock_engine_a() for pre-programmed response

        Falls back to Mock_Mode on role assumption failure.

        Args:
            prompt: The visitor's attack prompt (already validated/sanitized).
            on_token: Optional async callback invoked for each response
                token during streaming.

        Returns:
            An EngineResult with the response, annotations, and metadata.
        """
        start_time = time.perf_counter()

        if self._mode == AppMode.MOCK:
            return await self._process_mock(prompt, on_token, start_time)

        return await self._process_live(prompt, on_token, start_time)

    async def _process_mock(
        self,
        prompt: str,
        on_token: Optional[Callable[[str], Awaitable[None]]],
        start_time: float,
    ) -> EngineResult:
        """Process prompt using mock engine responses.

        Args:
            prompt: The visitor's attack prompt.
            on_token: Optional streaming callback.
            start_time: Processing start timestamp.

        Returns:
            EngineResult with mock response and annotations.
        """
        response_text = await self._mock_engine.mock_engine_a(prompt)

        # Stream tokens to UI if callback provided
        if on_token and response_text:
            for token in response_text.split():
                await on_token(token + " ")

        # Detect sensitive content and annotate
        detected_types = self.detect_sensitive_content(response_text)
        annotations: List[str] = []
        if detected_types:
            annotations.append("⚠️ Ungoverned — Data Leaked")

        duration_ms = (time.perf_counter() - start_time) * 1000

        return EngineResult(
            engine_label=self.ENGINE_LABEL,
            response_text=response_text,
            is_blocked=False,
            annotations=annotations,
            guardrail_action=None,
            sensitive_detected=bool(detected_types),
            error=None,
            duration_ms=duration_ms,
        )

    async def _process_live(
        self,
        prompt: str,
        on_token: Optional[Callable[[str], Awaitable[None]]],
        start_time: float,
    ) -> EngineResult:
        """Process prompt using live AWS services.

        Assumes Permissive_Role, invokes Bedrock model without guardrails,
        streams response, and annotates PII leakage.

        Falls back to mock mode on role assumption failure.

        Args:
            prompt: The visitor's attack prompt.
            on_token: Optional streaming callback.
            start_time: Processing start timestamp.

        Returns:
            EngineResult with live response and annotations.
        """
        try:
            # Invoke agent with Permissive_Role, NO guardrails
            # Apply 30-second timeout (Requirement 2.6)
            agent_response: AgentResponse = await asyncio.wait_for(
                self._agent.invoke(
                    prompt=prompt,
                    role_arn=self._agent.config.permissive_role_arn,
                    apply_guardrails=False,
                    on_token=on_token,
                ),
                timeout=ENGINE_A_TIMEOUT_SECONDS,
            )
        except RoleAssumptionError as e:
            # Fall back to Mock_Mode on role assumption failure (Requirement 9.9)
            logger.error(
                "Engine A role assumption failed (ARN: %s, error: %s). "
                "Falling back to Mock_Mode.",
                e.role_arn,
                e.error_type,
            )
            return await self._process_mock(prompt, on_token, start_time)
        except asyncio.TimeoutError:
            # 30-second timeout exceeded (Requirement 2.6)
            duration_ms = (time.perf_counter() - start_time) * 1000
            error_msg = (
                "Engine A is unavailable — response timed out after "
                f"{ENGINE_A_TIMEOUT_SECONDS} seconds."
            )
            logger.error("Engine A timeout after %ds", ENGINE_A_TIMEOUT_SECONDS)
            return EngineResult(
                engine_label=self.ENGINE_LABEL,
                response_text="",
                is_blocked=False,
                annotations=[],
                guardrail_action=None,
                sensitive_detected=False,
                error=error_msg,
                duration_ms=duration_ms,
            )
        except Exception as e:
            # General error handling (Requirement 2.6)
            duration_ms = (time.perf_counter() - start_time) * 1000
            error_msg = f"Engine A is unavailable — {type(e).__name__}: {str(e)}"
            logger.error("Engine A error: %s", error_msg)
            return EngineResult(
                engine_label=self.ENGINE_LABEL,
                response_text="",
                is_blocked=False,
                annotations=[],
                guardrail_action=None,
                sensitive_detected=False,
                error=error_msg,
                duration_ms=duration_ms,
            )

        # Check for agent-level errors
        if agent_response.error:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return EngineResult(
                engine_label=self.ENGINE_LABEL,
                response_text="",
                is_blocked=False,
                annotations=[],
                guardrail_action=None,
                sensitive_detected=False,
                error=f"Engine A is unavailable — {agent_response.error}",
                duration_ms=duration_ms,
            )

        # Detect sensitive content and annotate (Requirement 2.4)
        response_text = agent_response.content
        detected_types = self.detect_sensitive_content(response_text)
        annotations: List[str] = []
        if detected_types:
            annotations.append("⚠️ Ungoverned — Data Leaked")

        # Include generated SQL in annotations if available
        generated_sql = agent_response.metadata.get("generated_sql", "")
        if generated_sql:
            annotations.append(f"📝 **Generated SQL:**\n```sql\n{generated_sql}\n```")

        duration_ms = (time.perf_counter() - start_time) * 1000

        return EngineResult(
            engine_label=self.ENGINE_LABEL,
            response_text=response_text,
            is_blocked=False,
            annotations=annotations,
            guardrail_action=None,
            sensitive_detected=bool(detected_types),
            error=None,
            duration_ms=duration_ms,
        )

    def detect_sensitive_content(self, response: str) -> List[str]:
        """Detect PII patterns in a response string.

        Checks for phone numbers, email addresses, national IDs, and
        proprietary company codes. Returns a list of detected pattern
        type names.

        This method is independently testable and does not depend on
        any external services.

        Args:
            response: The response text to scan for sensitive content.

        Returns:
            A list of detected pattern type names (e.g., ["phone", "email"]).
            Empty list if no sensitive content is detected.
        """
        detected: List[str] = []

        # Check phone numbers
        for pattern in _PHONE_PATTERNS:
            if pattern.search(response):
                detected.append("phone")
                break

        # Check email addresses
        if _EMAIL_PATTERN.search(response):
            detected.append("email")

        # Check national IDs (Thai format)
        if _NATIONAL_ID_PATTERN.search(response):
            detected.append("national_id")

        # Check company codes
        if _COMPANY_CODE_PATTERN.search(response):
            detected.append("company_code")

        return detected
