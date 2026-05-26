"""Property-based tests for input validation and sanitization.

Feature: safe-haven-demo-booth
"""

import re
import string

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from aws_demo_booth.validators import sanitize_for_engine


# --- Constants matching the implementation ---

SHELL_METACHARACTERS = set(";|&$`")

SQL_KEYWORDS = [
    "SELECT",
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "UNION",
    "ALTER",
    "CREATE",
    "EXEC",
    "EXECUTE",
    "TRUNCATE",
    "MERGE",
]

SQL_KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(SQL_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

SCRIPT_TAG_PATTERN = re.compile(
    r"<\s*script[^>]*>.*?<\s*/\s*script\s*>|<\s*script[^>]*/\s*>",
    re.IGNORECASE | re.DOTALL,
)


# --- Strategies ---


def _safe_text() -> st.SearchStrategy[str]:
    """Generate text that does NOT contain any dangerous patterns.

    Safe text avoids shell metacharacters, SQL keywords as whole words,
    and script tag patterns.
    """
    # Use alphabet that excludes shell metacharacters and angle brackets
    safe_chars = "".join(
        ch
        for ch in string.ascii_letters + string.digits + " .,!?-_()[]{}:'\"\n\t"
        if ch not in SHELL_METACHARACTERS and ch not in "<>"
    )
    return st.text(alphabet=safe_chars, min_size=1, max_size=100).filter(
        lambda s: not SQL_KEYWORD_PATTERN.search(s)
    )


def _script_tag() -> st.SearchStrategy[str]:
    """Generate a <script>...</script> tag with random content."""
    content = st.text(
        alphabet=string.ascii_letters + string.digits + " .,!?",
        min_size=0,
        max_size=30,
    )
    return content.map(lambda c: f"<script>{c}</script>")


def _shell_metachar_string() -> st.SearchStrategy[str]:
    """Generate a string containing one or more shell metacharacters."""
    metachar = st.sampled_from(list(SHELL_METACHARACTERS))
    return metachar.map(lambda ch: ch)


def _sql_keyword() -> st.SearchStrategy[str]:
    """Generate a SQL keyword (random case)."""

    def randomize_case(kw: str) -> st.SearchStrategy[str]:
        # Return the keyword in various cases
        return st.sampled_from([kw.upper(), kw.lower(), kw.capitalize()])

    return st.sampled_from(SQL_KEYWORDS).flatmap(randomize_case)


def _input_with_dangerous_patterns() -> st.SearchStrategy[str]:
    """Generate input strings that embed dangerous patterns among safe text."""
    safe = _safe_text()
    script = _script_tag()
    metachar = _shell_metachar_string()
    sql_kw = _sql_keyword()

    # Build a string by interleaving safe text with at least one dangerous pattern
    return st.tuples(safe, st.one_of(script, metachar, sql_kw), safe).map(
        lambda parts: parts[0] + parts[1] + parts[2]
    )


# =============================================================================
# Property 2: Input sanitization removes dangerous patterns while preserving
# safe content
# =============================================================================


class TestProperty2InputSanitization:
    """Property 2: Input sanitization removes dangerous patterns while preserving safe content.

    **Validates: Requirements 2.5**
    """

    @settings(max_examples=20)
    @given(text=_input_with_dangerous_patterns())
    def test_no_script_tags_in_output(self, text: str):
        """Sanitized output contains no script tags.

        Feature: safe-haven-demo-booth, Property 2: Input sanitization
        **Validates: Requirements 2.5**
        """
        assume(len(text) <= 1000)
        result = sanitize_for_engine(text)
        assert not SCRIPT_TAG_PATTERN.search(result), (
            f"Script tag found in sanitized output: {result!r}"
        )

    @settings(max_examples=20)
    @given(text=_input_with_dangerous_patterns())
    def test_no_shell_metacharacters_in_output(self, text: str):
        """Sanitized output contains no shell metacharacters.

        Feature: safe-haven-demo-booth, Property 2: Input sanitization
        **Validates: Requirements 2.5**
        """
        assume(len(text) <= 1000)
        result = sanitize_for_engine(text)
        found_metachars = SHELL_METACHARACTERS.intersection(set(result))
        assert not found_metachars, (
            f"Shell metacharacters {found_metachars} found in sanitized output: {result!r}"
        )

    @settings(max_examples=20)
    @given(text=_input_with_dangerous_patterns())
    def test_no_sql_keywords_in_output(self, text: str):
        """Sanitized output contains no SQL keywords as whole words.

        Feature: safe-haven-demo-booth, Property 2: Input sanitization
        **Validates: Requirements 2.5**
        """
        assume(len(text) <= 1000)
        result = sanitize_for_engine(text)
        match = SQL_KEYWORD_PATTERN.search(result)
        assert match is None, (
            f"SQL keyword '{match.group()}' found in sanitized output: {result!r}"
        )

    @settings(max_examples=20)
    @given(safe_prefix=_safe_text(), safe_suffix=_safe_text())
    def test_safe_substrings_preserved_in_order(self, safe_prefix: str, safe_suffix: str):
        """Safe substrings are preserved in their original order after sanitization.

        Feature: safe-haven-demo-booth, Property 2: Input sanitization
        **Validates: Requirements 2.5**
        """
        # Insert a dangerous pattern between two safe strings
        dangerous = "<script>alert('xss')</script>"
        text = safe_prefix + dangerous + safe_suffix
        assume(len(text) <= 1000)

        result = sanitize_for_engine(text)

        # Both safe parts should appear in the result in order
        prefix_idx = result.find(safe_prefix)
        suffix_idx = result.find(safe_suffix, prefix_idx + len(safe_prefix) if prefix_idx >= 0 else 0)

        assert prefix_idx >= 0, (
            f"Safe prefix {safe_prefix!r} not found in result {result!r}"
        )
        assert suffix_idx >= prefix_idx + len(safe_prefix), (
            f"Safe suffix {safe_suffix!r} not found after prefix in result {result!r}"
        )

    @settings(max_examples=20)
    @given(
        text=st.text(
            alphabet=st.characters(blacklist_categories=("Cs",)),
            min_size=1001,
            max_size=1500,
        )
    )
    def test_rejects_input_exceeding_1000_chars(self, text: str):
        """Input exceeding 1000 characters is rejected with ValueError.

        Feature: safe-haven-demo-booth, Property 2: Input sanitization
        **Validates: Requirements 2.5**
        """
        with pytest.raises(ValueError):
            sanitize_for_engine(text)

    @settings(max_examples=20)
    @given(text=_safe_text())
    def test_safe_content_passes_through_unchanged(self, text: str):
        """Input with no dangerous patterns passes through unchanged.

        Feature: safe-haven-demo-booth, Property 2: Input sanitization
        **Validates: Requirements 2.5**
        """
        assume(len(text) <= 1000)
        result = sanitize_for_engine(text)
        assert result == text, (
            f"Safe input was modified: {text!r} -> {result!r}"
        )
