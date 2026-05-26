"""Property-based tests for annotations and sensitive content detection.

Feature: safe-haven-demo-booth
Property 3: Sensitive content detection annotates Engine_A responses containing PII patterns
Property 5: Guardrail blocking prevents agent invocation
Property 6: Engine_B output annotation reflects guardrail outcome
"""

import asyncio
import string
from typing import Any, Awaitable, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agent_wrapper import AgentConfig, AgentResponse
from config import AppMode
from engine_a import EngineA, EngineResult
from engine_b import EngineB
from gates.base import Gate, GateContext, GatePipeline, GateResult
from mock_engine import MockEngine


# =============================================================================
# Property 3: Sensitive content detection annotates Engine_A responses
# containing PII patterns
# =============================================================================

# --- Strategies for Property 3 ---


def _international_phone() -> st.SearchStrategy[str]:
    """Generate international phone numbers matching +XX-XX-XXX-XXXX format."""
    return st.builds(
        lambda cc, area, mid, last: f"+{cc}-{area}-{mid}-{last}",
        cc=st.from_regex(r"[0-9]{1,3}", fullmatch=True),
        area=st.from_regex(r"[0-9]{1,4}", fullmatch=True),
        mid=st.from_regex(r"[0-9]{2,4}", fullmatch=True),
        last=st.from_regex(r"[0-9]{3,4}", fullmatch=True),
    )


def _us_phone() -> st.SearchStrategy[str]:
    """Generate US phone numbers matching (XXX) XXX-XXXX format."""
    return st.builds(
        lambda area, mid, last: f"({area}) {mid}-{last}",
        area=st.from_regex(r"[0-9]{3}", fullmatch=True),
        mid=st.from_regex(r"[0-9]{3}", fullmatch=True),
        last=st.from_regex(r"[0-9]{4}", fullmatch=True),
    )


def _email_address() -> st.SearchStrategy[str]:
    """Generate email addresses matching standard email regex."""
    return st.builds(
        lambda local, domain, tld: f"{local}@{domain}.{tld}",
        local=st.from_regex(r"[a-zA-Z0-9._%+\-]{1,20}", fullmatch=True),
        domain=st.from_regex(r"[a-zA-Z0-9.\-]{1,15}", fullmatch=True),
        tld=st.from_regex(r"[a-zA-Z]{2,6}", fullmatch=True),
    )


def _thai_national_id() -> st.SearchStrategy[str]:
    """Generate Thai national IDs matching X-XXXX-XXXXX-XX-X format."""
    return st.builds(
        lambda a, b, c, d, e: f"{a}-{b}-{c}-{d}-{e}",
        a=st.from_regex(r"[0-9]{1}", fullmatch=True),
        b=st.from_regex(r"[0-9]{4}", fullmatch=True),
        c=st.from_regex(r"[0-9]{5}", fullmatch=True),
        d=st.from_regex(r"[0-9]{2}", fullmatch=True),
        e=st.from_regex(r"[0-9]{1}", fullmatch=True),
    )


def _company_code() -> st.SearchStrategy[str]:
    """Generate company codes matching XXXX-INTERNAL-XXXX-CONFIDENTIAL format."""
    return st.builds(
        lambda prefix, suffix: f"{prefix}-INTERNAL-{suffix}-CONFIDENTIAL",
        prefix=st.from_regex(r"[A-Z0-9]{2,6}", fullmatch=True),
        suffix=st.from_regex(r"[A-Z0-9]{2,6}", fullmatch=True),
    )


def _pii_pattern() -> st.SearchStrategy[str]:
    """Generate one of the PII patterns (phone, email, national ID, company code)."""
    return st.one_of(
        _international_phone(),
        _us_phone(),
        _email_address(),
        _thai_national_id(),
        _company_code(),
    )


def _surrounding_text() -> st.SearchStrategy[str]:
    """Generate surrounding text that does NOT accidentally contain PII patterns.

    Uses a restricted alphabet that avoids @ signs, + signs, parentheses,
    and digits to prevent accidental PII pattern matches.
    """
    safe_chars = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?:;'\"\n\t"
    return st.text(alphabet=safe_chars, min_size=0, max_size=80)


def _response_with_pii() -> st.SearchStrategy[str]:
    """Generate response strings that contain at least one embedded PII pattern."""
    return st.builds(
        lambda prefix, pii, suffix: f"{prefix} {pii} {suffix}",
        prefix=_surrounding_text(),
        pii=_pii_pattern(),
        suffix=_surrounding_text(),
    )


def _clean_response() -> st.SearchStrategy[str]:
    """Generate response strings that do NOT contain any PII patterns.

    Uses a restricted alphabet that cannot form phone numbers, emails,
    national IDs, or company codes.
    """
    # Exclude digits, @, +, (), and hyphen patterns that could form PII
    safe_chars = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?:;'\"\n\t"
    return (
        st.text(alphabet=safe_chars, min_size=1, max_size=200)
        .filter(lambda s: s.strip())  # Ensure non-empty after trim
    )


class TestProperty3SensitiveContentDetection:
    """Property 3: Sensitive content detection annotates Engine_A responses containing PII patterns.

    For any Engine_A response string that matches one or more sensitive content
    patterns (phone numbers, email addresses, national IDs, or proprietary
    company codes), the annotation logic SHALL produce the warning label
    "⚠️ Ungoverned — Data Leaked". For any response that does not match any
    sensitive pattern, no such annotation SHALL be applied.

    **Validates: Requirements 2.4**
    """

    @settings(max_examples=100, deadline=None)
    @given(response=_response_with_pii())
    def test_pii_detected_returns_non_empty_list(self, response: str):
        """detect_sensitive_content returns non-empty list when PII is present.

        For any response string containing an embedded PII pattern (phone,
        email, national ID, or company code), the detection method SHALL
        return a non-empty list of detected pattern types.

        Feature: safe-haven-demo-booth, Property 3: Sensitive content detection
        **Validates: Requirements 2.4**
        """
        engine = EngineA.__new__(EngineA)
        detected = engine.detect_sensitive_content(response)

        assert len(detected) > 0, (
            f"Expected non-empty detection list for response containing PII, "
            f"but got empty list. Response: {response!r}"
        )

    @settings(max_examples=100, deadline=None)
    @given(response=_clean_response())
    def test_no_pii_returns_empty_list(self, response: str):
        """detect_sensitive_content returns empty list when no PII is present.

        For any response string that does not match any sensitive content
        pattern, the detection method SHALL return an empty list.

        Feature: safe-haven-demo-booth, Property 3: Sensitive content detection
        **Validates: Requirements 2.4**
        """
        engine = EngineA.__new__(EngineA)
        detected = engine.detect_sensitive_content(response)

        assert len(detected) == 0, (
            f"Expected empty detection list for clean response, "
            f"but got {detected}. Response: {response!r}"
        )

    @settings(max_examples=100, deadline=None)
    @given(response=_response_with_pii())
    def test_annotation_applied_when_pii_detected(self, response: str):
        """Annotation "⚠️ Ungoverned — Data Leaked" is applied when PII is detected.

        For any Engine_A response that matches one or more sensitive content
        patterns, the annotation logic SHALL produce the warning label
        "⚠️ Ungoverned — Data Leaked".

        Feature: safe-haven-demo-booth, Property 3: Sensitive content detection
        **Validates: Requirements 2.4**
        """
        engine = EngineA.__new__(EngineA)
        detected = engine.detect_sensitive_content(response)

        # Simulate the annotation logic from EngineA._process_mock / _process_live
        annotations: List[str] = []
        if detected:
            annotations.append("⚠️ Ungoverned — Data Leaked")

        assert "⚠️ Ungoverned — Data Leaked" in annotations, (
            f"Expected '⚠️ Ungoverned — Data Leaked' annotation when PII detected, "
            f"but annotations were: {annotations}. Detected types: {detected}. "
            f"Response: {response!r}"
        )

    @settings(max_examples=100, deadline=None)
    @given(response=_clean_response())
    def test_no_annotation_when_no_pii(self, response: str):
        """No annotation is applied when no PII is detected.

        For any response that does not match any sensitive pattern, no
        "⚠️ Ungoverned — Data Leaked" annotation SHALL be applied.

        Feature: safe-haven-demo-booth, Property 3: Sensitive content detection
        **Validates: Requirements 2.4**
        """
        engine = EngineA.__new__(EngineA)
        detected = engine.detect_sensitive_content(response)

        # Simulate the annotation logic
        annotations: List[str] = []
        if detected:
            annotations.append("⚠️ Ungoverned — Data Leaked")

        assert "⚠️ Ungoverned — Data Leaked" not in annotations, (
            f"Expected no '⚠️ Ungoverned — Data Leaked' annotation for clean response, "
            f"but annotations were: {annotations}. Detected types: {detected}. "
            f"Response: {response!r}"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        phone=_international_phone(),
        email=_email_address(),
        national_id=_thai_national_id(),
        company_code=_company_code(),
    )
    def test_all_pii_types_detected_when_all_present(
        self, phone: str, email: str, national_id: str, company_code: str
    ):
        """All PII types are detected when all are present in the response.

        When a response contains phone numbers, emails, national IDs, AND
        company codes, detect_sensitive_content SHALL return all four types.

        Feature: safe-haven-demo-booth, Property 3: Sensitive content detection
        **Validates: Requirements 2.4**
        """
        response = f"Contact: {phone}, email: {email}, ID: {national_id}, code: {company_code}"
        engine = EngineA.__new__(EngineA)
        detected = engine.detect_sensitive_content(response)

        assert "phone" in detected, (
            f"Expected 'phone' in detected types. Got: {detected}. Response: {response!r}"
        )
        assert "email" in detected, (
            f"Expected 'email' in detected types. Got: {detected}. Response: {response!r}"
        )
        assert "national_id" in detected, (
            f"Expected 'national_id' in detected types. Got: {detected}. Response: {response!r}"
        )
        assert "company_code" in detected, (
            f"Expected 'company_code' in detected types. Got: {detected}. Response: {response!r}"
        )


# =============================================================================
# Property 5: Guardrail blocking prevents agent invocation
# =============================================================================

# --- Strategies for Property 5 ---

# The jailbreak keywords from MockEngine that trigger BLOCKED in mock mode
JAILBREAK_KEYWORDS: List[str] = MockEngine.JAILBREAK_KEYWORDS


def _prompt_with_jailbreak_keyword() -> st.SearchStrategy[str]:
    """Generate prompts that contain at least one jailbreak keyword.

    Constructs prompts by combining arbitrary text with a randomly
    selected jailbreak keyword, ensuring the keyword appears in the
    prompt and will trigger the guardrail blocking logic.
    """
    return st.builds(
        lambda prefix, keyword, suffix: f"{prefix} {keyword} {suffix}",
        prefix=st.text(
            alphabet=st.characters(
                blacklist_categories=("Cs",),
                blacklist_characters="\x00",
            ),
            min_size=0,
            max_size=50,
        ),
        keyword=st.sampled_from(JAILBREAK_KEYWORDS),
        suffix=st.text(
            alphabet=st.characters(
                blacklist_categories=("Cs",),
                blacklist_characters="\x00",
            ),
            min_size=0,
            max_size=50,
        ),
    )


# --- Fixture to patch out mock gate delays ---


@pytest.fixture(autouse=True)
def patch_mock_gate_delays(monkeypatch):
    """Patch asyncio.sleep in mock_engine to eliminate gate delays during tests.

    The MockEngine gates use random 1-2 second delays which would make
    100 iterations extremely slow. We patch asyncio.sleep to be instant.
    """
    import mock_engine as mock_engine_module

    original_sleep = asyncio.sleep

    async def fast_sleep(delay):
        """Replace sleep with a near-instant delay for test speed."""
        await original_sleep(0)

    monkeypatch.setattr(mock_engine_module.asyncio, "sleep", fast_sleep)


# --- Mock Agent Wrapper ---


class TrackingAgentWrapper:
    """A mock StrandsAgentWrapper that tracks whether invoke() was called.

    Used to verify that when guardrails block a prompt, the agent's
    invoke() method is never called.
    """

    def __init__(self) -> None:
        self.invoke_called = False
        self.invoke_call_count = 0
        self.config = AgentConfig(
            bedrock_model_id="test-model",
            knowledge_base_id="test-kb",
            guardrails_id="test-guardrails",
            guardrails_version="1",
            glue_database_name="test-db",
            glue_table_name="test-table",
            permissive_role_arn="arn:aws:iam::123456789012:role/PermissiveRole",
            restricted_role_arn="arn:aws:iam::123456789012:role/RestrictedRole",
        )

    async def invoke(
        self,
        prompt: str,
        role_arn: str,
        apply_guardrails: bool,
        on_token: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> AgentResponse:
        """Track that invoke was called (should NOT happen for blocked prompts)."""
        self.invoke_called = True
        self.invoke_call_count += 1
        return AgentResponse(
            content="This should not be reached",
            metadata={},
            guardrail_action="PASS",
        )

    async def retrieve_context(self, prompt: str, role_arn: str) -> str:
        """Mock retrieve_context (should NOT be called for blocked prompts)."""
        return ""


# =============================================================================
# Property 5: Guardrail blocking prevents agent invocation
# =============================================================================


class TestProperty5GuardrailBlockingPreventsAgentInvocation:
    """Property 5: Guardrail blocking prevents agent invocation.

    For any Attack_Prompt where Guardrails input evaluation returns a
    "BLOCKED" action, the Strands_Agent SHALL NOT be invoked for response
    generation, and the Engine_B panel SHALL display
    "🛡️ Blocked by Bedrock Guardrails".

    **Validates: Requirements 3.4**
    """

    @settings(max_examples=100, deadline=None)
    @given(prompt=_prompt_with_jailbreak_keyword())
    @pytest.mark.asyncio
    async def test_blocked_prompt_does_not_invoke_agent(self, prompt: str):
        """When guardrails block a prompt, the Strands_Agent is NOT invoked.

        For any prompt containing a jailbreak keyword, the mock guardrail
        returns BLOCKED, and the agent's invoke() method SHALL NOT be called.

        Feature: safe-haven-demo-booth, Property 5: Guardrail blocking
        **Validates: Requirements 3.4**
        """
        # Create a tracking agent wrapper to detect if invoke() is called
        tracking_agent = TrackingAgentWrapper()

        # Create EngineB in MOCK mode (uses MockEngine for gate evaluation)
        engine_b = EngineB(agent=tracking_agent, mode=AppMode.MOCK)

        # Process the prompt through Engine B
        result: EngineResult = await engine_b.process(
            prompt=prompt,
            parent_message=None,
            on_token=None,
        )

        # Verify: the agent's invoke() was NOT called
        assert not tracking_agent.invoke_called, (
            f"Agent invoke() was called for blocked prompt: '{prompt[:80]}...'. "
            f"When guardrails return BLOCKED, the agent SHALL NOT be invoked."
        )
        assert tracking_agent.invoke_call_count == 0, (
            f"Agent invoke() was called {tracking_agent.invoke_call_count} time(s) "
            f"for blocked prompt. Expected 0 calls."
        )

    @settings(max_examples=100, deadline=None)
    @given(prompt=_prompt_with_jailbreak_keyword())
    @pytest.mark.asyncio
    async def test_blocked_prompt_returns_is_blocked_true(self, prompt: str):
        """When guardrails block a prompt, the result has is_blocked=True.

        For any prompt containing a jailbreak keyword, the Engine_B result
        SHALL have is_blocked=True indicating the prompt was blocked.

        Feature: safe-haven-demo-booth, Property 5: Guardrail blocking
        **Validates: Requirements 3.4**
        """
        tracking_agent = TrackingAgentWrapper()
        engine_b = EngineB(agent=tracking_agent, mode=AppMode.MOCK)

        result: EngineResult = await engine_b.process(
            prompt=prompt,
            parent_message=None,
            on_token=None,
        )

        # Verify: result indicates blocking
        assert result.is_blocked is True, (
            f"Expected is_blocked=True for prompt containing jailbreak keyword, "
            f"but got is_blocked={result.is_blocked}. Prompt: '{prompt[:80]}...'"
        )

    @settings(max_examples=100, deadline=None)
    @given(prompt=_prompt_with_jailbreak_keyword())
    @pytest.mark.asyncio
    async def test_blocked_prompt_displays_blocked_message(self, prompt: str):
        """When guardrails block a prompt, the blocked message is displayed.

        For any prompt where guardrails return BLOCKED, the Engine_B panel
        SHALL display "🛡️ Blocked by Bedrock Guardrails" in the response_text.

        Feature: safe-haven-demo-booth, Property 5: Guardrail blocking
        **Validates: Requirements 3.4**
        """
        tracking_agent = TrackingAgentWrapper()
        engine_b = EngineB(agent=tracking_agent, mode=AppMode.MOCK)

        result: EngineResult = await engine_b.process(
            prompt=prompt,
            parent_message=None,
            on_token=None,
        )

        # Verify: response text contains the blocked message
        assert "\U0001f6e1\ufe0f Blocked by Bedrock Guardrails" in result.response_text, (
            f"Expected '\U0001f6e1\ufe0f Blocked by Bedrock Guardrails' in response_text, "
            f"but got: '{result.response_text[:100]}'. "
            f"Prompt: '{prompt[:80]}...'"
        )



# --- Helper gates for Property 6 ---


class FakePassGate(Gate):
    """A gate that always passes without blocking."""

    def __init__(self, name: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(name=name, timeout_seconds=10.0)
        self._details = details or {}

    async def execute(self, context: GateContext) -> GateResult:
        return GateResult(passed=True, blocked=False, details=self._details)


class FakeGate3(Gate):
    """A fake Gate_3 that returns configurable input/output actions."""

    def __init__(self, input_action: str = "PASS", output_action: str = "PASS") -> None:
        super().__init__(name="[Gate 3] Evaluating Bedrock Guardrails...", timeout_seconds=10.0)
        self.input_action = input_action
        self.output_action = output_action

    async def execute(self, context: GateContext) -> GateResult:
        blocked = self.input_action == "BLOCKED"
        return GateResult(
            passed=not blocked,
            blocked=blocked,
            details={
                "input_action": self.input_action,
                "output_action": self.output_action,
            },
        )


# --- Strategies for Property 6 ---

# Characters that won't trigger jailbreak/PII keywords in MockEngine
_JAILBREAK_KEYWORDS_LOWER = [kw.lower() for kw in MockEngine.JAILBREAK_KEYWORDS]
_PII_KEYWORDS_LOWER = [kw.lower() for kw in MockEngine.PII_KEYWORDS]
_ALL_BLOCKED_KEYWORDS = _JAILBREAK_KEYWORDS_LOWER + _PII_KEYWORDS_LOWER


def _safe_prompt_strategy() -> st.SearchStrategy[str]:
    """Generate prompts that do NOT contain any jailbreak or PII keywords.

    These prompts should pass through Mock Engine_B without being blocked,
    resulting in a "sanitized" response with the Safe Haven annotation.
    """
    # Use a restricted alphabet to avoid accidentally forming blocked keywords
    safe_chars = "abcdfghklmqtuvwxyz0123456789 .,!?"
    return (
        st.text(alphabet=safe_chars, min_size=5, max_size=100)
        .filter(lambda s: s.strip())
        .filter(
            lambda s: not any(kw in s.lower() for kw in _ALL_BLOCKED_KEYWORDS)
        )
    )


def _guardrail_output_action_strategy() -> st.SearchStrategy[str]:
    """Generate guardrail output actions: PASS or MODIFIED (non-blocked)."""
    return st.sampled_from(["PASS", "MODIFIED"])


def _response_text_strategy() -> st.SearchStrategy[str]:
    """Generate arbitrary response text from the agent."""
    return st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)),
        min_size=1,
        max_size=200,
    )


# =============================================================================
# Property 6: Engine_B output annotation reflects guardrail outcome
# =============================================================================


class TestProperty6EngineBOutputAnnotation:
    """Property 6: Engine_B output annotation reflects guardrail outcome.

    For any Engine_B response that completes without being blocked, the output
    SHALL be annotated with "✅ Safe Haven — Protected". For any response where
    the Guardrails output filter detects and redacts content, the output SHALL
    additionally be annotated with "✅ Output Sanitized by Guardrails".

    **Validates: Requirements 3.6, 3.7**
    """

    @settings(max_examples=100, deadline=None)
    @given(prompt=_safe_prompt_strategy())
    @pytest.mark.asyncio
    async def test_mock_mode_safe_prompt_has_protected_annotation(self, prompt: str):
        """In Mock mode, safe prompts (no blocked keywords) produce
        "✅ Safe Haven — Protected" annotation.

        For any prompt that does not contain jailbreak or PII keywords,
        Engine_B in Mock mode SHALL annotate the response with
        "✅ Safe Haven — Protected".

        Feature: safe-haven-demo-booth, Property 6: Engine_B output annotation
        **Validates: Requirements 3.6, 3.7**
        """
        # Create a tracking agent (won't be invoked in mock mode fallback)
        tracking_agent = TrackingAgentWrapper()

        engine = EngineB(agent=tracking_agent, mode=AppMode.MOCK)

        # Replace the gate pipeline with instant fake gates to avoid
        # 1-2 second mock delays per gate (we're testing annotation logic,
        # not gate timing)
        fake_gate1 = FakePassGate(name="[Gate 1] Amazon Macie Scan Results")
        fake_gate2 = FakePassGate(name="[Gate 2] Evaluating AWS Lake Formation...")
        fake_gate3 = FakeGate3(input_action="PASS", output_action="PASS")
        engine._pipeline = GatePipeline(gates=[fake_gate1, fake_gate2, fake_gate3])

        result = await engine.process(prompt=prompt, parent_message=None)

        # Non-blocked result should have the Safe Haven annotation
        assert not result.is_blocked, (
            f"Safe prompt should not be blocked, got is_blocked=True for: {prompt!r}"
        )
        assert "✅ Safe Haven — Protected" in result.annotations, (
            f"Non-blocked Engine_B result must contain '✅ Safe Haven — Protected' "
            f"annotation. Got annotations: {result.annotations} for prompt: {prompt!r}"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        prompt=_response_text_strategy(),
        output_action=_guardrail_output_action_strategy(),
        response_text=_response_text_strategy(),
    )
    @pytest.mark.asyncio
    async def test_live_mode_output_action_determines_sanitized_annotation(
        self, prompt: str, output_action: str, response_text: str
    ):
        """In Live mode with mocked gates, when Gate_3 output_action is MODIFIED,
        the result SHALL contain both "✅ Safe Haven — Protected" and
        "✅ Output Sanitized by Guardrails" annotations.

        When output_action is PASS, only "✅ Safe Haven — Protected" should
        be present (no sanitization annotation).

        Feature: safe-haven-demo-booth, Property 6: Engine_B output annotation
        **Validates: Requirements 3.6, 3.7**
        """
        # Create a mock agent that returns a successful response
        mock_agent = MagicMock()
        mock_agent.config = AgentConfig(
            bedrock_model_id="test-model",
            knowledge_base_id="test-kb",
            guardrails_id="test-guard",
            guardrails_version="1",
            glue_database_name="test-db",
            glue_table_name="test-table",
            permissive_role_arn="arn:aws:iam::123456789012:role/Permissive",
            restricted_role_arn="arn:aws:iam::123456789012:role/Restricted",
        )

        # Mock the agent invoke to return a successful response with PASS guardrail
        mock_agent.invoke = AsyncMock(
            return_value=AgentResponse(
                content=response_text,
                metadata={"model_id": "test-model"},
                guardrail_action="PASS",
                error=None,
            )
        )

        engine = EngineB(agent=mock_agent, mode=AppMode.LIVE)

        # Replace the pipeline with fake gates that don't block
        # Gate_3 returns the specified output_action
        fake_gate1 = FakePassGate(name="[Gate 1] Amazon Macie Scan Results")
        fake_gate2 = FakePassGate(name="[Gate 2] Evaluating AWS Lake Formation...")
        fake_gate3 = FakeGate3(input_action="PASS", output_action=output_action)

        engine._pipeline = GatePipeline(gates=[fake_gate1, fake_gate2, fake_gate3])

        result = await engine.process(prompt=prompt, parent_message=None)

        # Should not be blocked
        assert not result.is_blocked, (
            f"Non-blocked gate pipeline should not produce blocked result. "
            f"output_action={output_action}"
        )
        # Should not have an error
        assert result.error is None, (
            f"Expected no error, got: {result.error}"
        )

        # Always annotated with "✅ Safe Haven — Protected" on non-blocked completion
        assert "✅ Safe Haven — Protected" in result.annotations, (
            f"Non-blocked Engine_B result must contain '✅ Safe Haven — Protected'. "
            f"Got annotations: {result.annotations}, output_action={output_action}"
        )

        # If output was MODIFIED, additionally annotated with sanitization message
        if output_action == "MODIFIED":
            assert "✅ Output Sanitized by Guardrails" in result.annotations, (
                f"When output_action is MODIFIED, result must contain "
                f"'✅ Output Sanitized by Guardrails'. "
                f"Got annotations: {result.annotations}"
            )
        else:
            # When output_action is PASS, sanitization annotation should NOT be present
            assert "✅ Output Sanitized by Guardrails" not in result.annotations, (
                f"When output_action is PASS, result must NOT contain "
                f"'✅ Output Sanitized by Guardrails'. "
                f"Got annotations: {result.annotations}"
            )

    @settings(max_examples=100, deadline=None)
    @given(
        prompt=_response_text_strategy(),
        response_text=_response_text_strategy(),
    )
    @pytest.mark.asyncio
    async def test_live_mode_agent_guardrail_modified_has_sanitized_annotation(
        self, prompt: str, response_text: str
    ):
        """When the agent response itself has guardrail_action=MODIFIED
        (output filtering detected content), the result SHALL contain
        "✅ Output Sanitized by Guardrails" in addition to the protected annotation.

        This tests the case where Gate_3 output_action is PASS but the agent
        response guardrail_action indicates output was modified during filtering.

        Feature: safe-haven-demo-booth, Property 6: Engine_B output annotation
        **Validates: Requirements 3.6, 3.7**
        """
        # Create a mock agent that returns a MODIFIED guardrail action
        mock_agent = MagicMock()
        mock_agent.config = AgentConfig(
            bedrock_model_id="test-model",
            knowledge_base_id="test-kb",
            guardrails_id="test-guard",
            guardrails_version="1",
            glue_database_name="test-db",
            glue_table_name="test-table",
            permissive_role_arn="arn:aws:iam::123456789012:role/Permissive",
            restricted_role_arn="arn:aws:iam::123456789012:role/Restricted",
        )

        # Agent response with guardrail_action=MODIFIED (output was redacted)
        mock_agent.invoke = AsyncMock(
            return_value=AgentResponse(
                content=response_text,
                metadata={"model_id": "test-model"},
                guardrail_action="MODIFIED",
                error=None,
            )
        )

        engine = EngineB(agent=mock_agent, mode=AppMode.LIVE)

        # Replace pipeline with fake gates — Gate_3 output_action is PASS
        # but the agent itself reports MODIFIED
        fake_gate1 = FakePassGate(name="[Gate 1] Amazon Macie Scan Results")
        fake_gate2 = FakePassGate(name="[Gate 2] Evaluating AWS Lake Formation...")
        fake_gate3 = FakeGate3(input_action="PASS", output_action="PASS")

        engine._pipeline = GatePipeline(gates=[fake_gate1, fake_gate2, fake_gate3])

        result = await engine.process(prompt=prompt, parent_message=None)

        # Should not be blocked
        assert not result.is_blocked
        assert result.error is None

        # Must have both annotations
        assert "✅ Safe Haven — Protected" in result.annotations, (
            f"Non-blocked result must have '✅ Safe Haven — Protected'. "
            f"Got: {result.annotations}"
        )
        assert "✅ Output Sanitized by Guardrails" in result.annotations, (
            f"When agent guardrail_action is MODIFIED, result must have "
            f"'✅ Output Sanitized by Guardrails'. Got: {result.annotations}"
        )


    @settings(max_examples=100, deadline=None)
    @given(prompt=_prompt_with_jailbreak_keyword())
    @pytest.mark.asyncio
    async def test_blocked_prompt_does_not_have_protected_annotation(self, prompt: str):
        """Blocked prompts do NOT receive the "✅ Safe Haven — Protected" annotation.

        For any prompt that is blocked by guardrails (jailbreak keyword detected),
        the Engine_B result SHALL NOT contain "✅ Safe Haven — Protected" in its
        annotations. Only non-blocked completions receive this annotation.

        Feature: safe-haven-demo-booth, Property 6: Engine_B output annotation
        **Validates: Requirements 3.6, 3.7**
        """
        tracking_agent = TrackingAgentWrapper()
        engine = EngineB(agent=tracking_agent, mode=AppMode.MOCK)

        result = await engine.process(prompt=prompt, parent_message=None)

        # Blocked prompts should be marked as blocked
        assert result.is_blocked is True, (
            f"Expected is_blocked=True for prompt with jailbreak keyword, "
            f"but got is_blocked={result.is_blocked}. Prompt: '{prompt[:80]}...'"
        )

        # Blocked prompts must NOT have the "✅ Safe Haven — Protected" annotation
        assert "✅ Safe Haven — Protected" not in result.annotations, (
            f"Blocked prompt must NOT contain '✅ Safe Haven — Protected' annotation. "
            f"Got annotations: {result.annotations}. Prompt: '{prompt[:80]}...'"
        )

        # Blocked prompts should have the blocked annotation instead
        assert "🛡️ Blocked by Bedrock Guardrails" in result.annotations, (
            f"Blocked prompt should have '🛡️ Blocked by Bedrock Guardrails' annotation. "
            f"Got annotations: {result.annotations}. Prompt: '{prompt[:80]}...'"
        )
