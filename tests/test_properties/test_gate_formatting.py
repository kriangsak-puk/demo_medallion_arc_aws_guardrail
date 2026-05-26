"""Property-based tests for gate formatting (Properties 7, 8, 9).

Feature: safe-haven-demo-booth
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from aws_demo_booth.gates.gate2_lakeformation import Gate2LakeFormation
from aws_demo_booth.gates.gate3_guardrails import Gate3Guardrails, VALID_ACTIONS


# --- Strategies ---

_guardrail_action = st.sampled_from(sorted(VALID_ACTIONS))


def _column_name() -> st.SearchStrategy[str]:
    """Generate realistic column names like 'col_abc123'."""
    return st.from_regex(r"col_[a-z0-9]{3,8}", fullmatch=True)


def _non_overlapping_column_lists() -> st.SearchStrategy[tuple]:
    """Generate two non-overlapping lists of column names (1-10 each)."""
    return st.lists(
        _column_name(), min_size=1, max_size=10, unique=True
    ).flatmap(
        lambda allowed: st.tuples(
            st.just(allowed),
            st.lists(
                _column_name().filter(lambda c: c not in allowed),
                min_size=1,
                max_size=10,
                unique=True,
            ),
        )
    )


# =============================================================================
# Property 8: Lake Formation column access formatting
# =============================================================================


class TestProperty8LakeFormationFormatting:
    """Property 8: Lake Formation column access formatting.

    For any set of allowed columns and denied columns returned by Gate_2
    evaluation, the Step_Log SHALL list each allowed column prefixed with
    "✅ ALLOWED", each denied column prefixed with "🚫 DENIED", and SHALL
    include a summary line stating the count of denied columns excluded
    from the response context.

    **Validates: Requirements 5.2, 5.3, 5.6**
    """

    @settings(max_examples=100, deadline=None)
    @given(columns=_non_overlapping_column_lists())
    def test_allowed_columns_have_correct_prefix(self, columns):
        """Each allowed column appears with '✅ ALLOWED: {col}' prefix.

        Feature: safe-haven-demo-booth, Property 8: Lake Formation formatting
        **Validates: Requirements 5.2**
        """
        allowed, denied = columns
        gate = Gate2LakeFormation()

        output = gate.format_lakeformation_output(allowed, denied)

        for col in allowed:
            expected_line = f"✅ ALLOWED: {col}"
            assert expected_line in output, (
                f"Expected '{expected_line}' in output but not found.\n"
                f"Output:\n{output}"
            )

    @settings(max_examples=100, deadline=None)
    @given(columns=_non_overlapping_column_lists())
    def test_denied_columns_have_correct_prefix(self, columns):
        """Each denied column appears with '🚫 DENIED: {col}' prefix.

        Feature: safe-haven-demo-booth, Property 8: Lake Formation formatting
        **Validates: Requirements 5.3**
        """
        allowed, denied = columns
        gate = Gate2LakeFormation()

        output = gate.format_lakeformation_output(allowed, denied)

        for col in denied:
            expected_line = f"🚫 DENIED: {col}"
            assert expected_line in output, (
                f"Expected '{expected_line}' in output but not found.\n"
                f"Output:\n{output}"
            )

    @settings(max_examples=100, deadline=None)
    @given(columns=_non_overlapping_column_lists())
    def test_summary_line_contains_denied_count(self, columns):
        """Summary line states the count of denied columns excluded from context.

        Feature: safe-haven-demo-booth, Property 8: Lake Formation formatting
        **Validates: Requirements 5.6**
        """
        allowed, denied = columns
        gate = Gate2LakeFormation()

        output = gate.format_lakeformation_output(allowed, denied)

        expected_summary = (
            f"Summary: {len(denied)} columns denied and excluded "
            f"from response context"
        )
        assert expected_summary in output, (
            f"Expected summary '{expected_summary}' in output but not found.\n"
            f"Output:\n{output}"
        )


# =============================================================================
# Property 9: Guardrails Step_Log formatting and expansion state
# =============================================================================


class TestProperty9GuardrailsFormatting:
    """Property 9: Guardrails Step_Log formatting and expansion state.

    For any guardrail evaluation result with an input action and an output action
    (each being "PASS", "BLOCKED", or "MODIFIED"), the Gate_3 Step_Log SHALL display
    "Input: {input_action}" and "Output: {output_action}". The Step_Log SHALL be
    rendered in expanded state if either action is "BLOCKED" or "MODIFIED", and in
    collapsed state if and only if both actions are "PASS".

    **Validates: Requirements 6.2, 6.3, 6.4, 6.5**
    """

    def setup_method(self):
        """Create a Gate3Guardrails instance for each test."""
        self.gate = Gate3Guardrails()

    @settings(max_examples=100)
    @given(input_action=_guardrail_action, output_action=_guardrail_action)
    def test_output_contains_input_action(self, input_action: str, output_action: str):
        """Formatted output contains 'Input: {input_action}'.

        Feature: safe-haven-demo-booth, Property 9: Guardrails formatting
        **Validates: Requirements 6.2**
        """
        result = self.gate.format_guardrails_output(input_action, output_action)
        expected = f"Input: {input_action}"
        assert expected in result, (
            f"Expected {expected!r} in output, got: {result!r}"
        )

    @settings(max_examples=100)
    @given(input_action=_guardrail_action, output_action=_guardrail_action)
    def test_output_contains_output_action(self, input_action: str, output_action: str):
        """Formatted output contains 'Output: {output_action}'.

        Feature: safe-haven-demo-booth, Property 9: Guardrails formatting
        **Validates: Requirements 6.3**
        """
        result = self.gate.format_guardrails_output(input_action, output_action)
        expected = f"Output: {output_action}"
        assert expected in result, (
            f"Expected {expected!r} in output, got: {result!r}"
        )

    @settings(max_examples=100)
    @given(input_action=_guardrail_action, output_action=_guardrail_action)
    def test_expanded_when_either_not_pass(self, input_action: str, output_action: str):
        """Step_Log is expanded when either action is BLOCKED or MODIFIED.

        Feature: safe-haven-demo-booth, Property 9: Guardrails formatting
        **Validates: Requirements 6.4**
        """
        expand = self.gate.should_expand(input_action, output_action)

        if input_action != "PASS" or output_action != "PASS":
            assert expand is True, (
                f"Expected expanded=True for input={input_action!r}, "
                f"output={output_action!r}, got {expand}"
            )

    @settings(max_examples=100)
    @given(input_action=_guardrail_action, output_action=_guardrail_action)
    def test_collapsed_only_when_both_pass(self, input_action: str, output_action: str):
        """Step_Log is collapsed if and only if both actions are PASS.

        Feature: safe-haven-demo-booth, Property 9: Guardrails formatting
        **Validates: Requirements 6.5**
        """
        expand = self.gate.should_expand(input_action, output_action)

        if input_action == "PASS" and output_action == "PASS":
            assert expand is False, (
                f"Expected expanded=False when both actions are PASS, got {expand}"
            )
        else:
            assert expand is True, (
                f"Expected expanded=True for input={input_action!r}, "
                f"output={output_action!r}, got {expand}"
            )
