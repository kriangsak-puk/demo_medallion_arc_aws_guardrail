"""Unit tests for the input validator and sanitizer module."""

import pytest

from validators import (
    MAX_ENGINE_LENGTH,
    MAX_PROMPT_LENGTH,
    ValidationResult,
    sanitize_for_engine,
    validate_prompt,
)


class TestValidatePrompt:
    """Tests for validate_prompt()."""

    def test_accepts_simple_valid_prompt(self):
        result = validate_prompt("Hello, show me the data")
        assert result.is_valid is True
        assert result.error_message is None
        assert result.sanitized_text == "Hello, show me the data"

    def test_accepts_single_character(self):
        result = validate_prompt("a")
        assert result.is_valid is True
        assert result.sanitized_text == "a"

    def test_accepts_exactly_500_chars(self):
        text = "x" * 500
        result = validate_prompt(text)
        assert result.is_valid is True
        assert result.sanitized_text == text

    def test_trims_whitespace_before_validation(self):
        result = validate_prompt("  hello world  ")
        assert result.is_valid is True
        assert result.sanitized_text == "hello world"

    def test_rejects_empty_string(self):
        result = validate_prompt("")
        assert result.is_valid is False
        assert result.error_message is not None
        assert "empty" in result.error_message.lower() or "whitespace" in result.error_message.lower()

    def test_rejects_whitespace_only(self):
        result = validate_prompt("   \t\n  ")
        assert result.is_valid is False
        assert result.error_message is not None

    def test_rejects_over_500_chars_after_trim(self):
        text = "x" * 501
        result = validate_prompt(text)
        assert result.is_valid is False
        assert "500" in result.error_message

    def test_trims_then_checks_length(self):
        # 498 chars + surrounding spaces = within limit after trim
        text = "  " + "x" * 498 + "  "
        result = validate_prompt(text)
        assert result.is_valid is True
        assert len(result.sanitized_text) == 498

    def test_rejects_over_500_after_trim(self):
        # 501 chars + spaces, still over 500 after trim
        text = "  " + "x" * 501 + "  "
        result = validate_prompt(text)
        assert result.is_valid is False

    def test_returns_empty_sanitized_text_on_rejection(self):
        result = validate_prompt("")
        assert result.sanitized_text == ""


class TestSanitizeForEngine:
    """Tests for sanitize_for_engine()."""

    def test_passes_through_safe_text(self):
        text = "Show me revenue by region"
        assert sanitize_for_engine(text) == text

    def test_strips_script_tags(self):
        text = "Hello <script>alert('xss')</script> world"
        result = sanitize_for_engine(text)
        assert "<script>" not in result.lower()
        assert "</script>" not in result.lower()
        assert "Hello" in result
        assert "world" in result

    def test_strips_script_tags_with_attributes(self):
        text = 'before <script type="text/javascript">code</script> after'
        result = sanitize_for_engine(text)
        assert "<script" not in result.lower()
        assert "before" in result
        assert "after" in result

    def test_strips_shell_metacharacters(self):
        text = "ls; cat /etc/passwd | grep root & echo $HOME `whoami`"
        result = sanitize_for_engine(text)
        assert ";" not in result
        assert "|" not in result
        assert "&" not in result
        assert "$" not in result
        assert "`" not in result

    def test_strips_sql_keywords(self):
        text = "Please SELECT all data FROM the table"
        result = sanitize_for_engine(text)
        assert "SELECT" not in result
        # "FROM" is not in our keyword list, so it stays
        assert "Please" in result
        assert "all data" in result

    def test_strips_sql_keywords_case_insensitive(self):
        text = "drop table users; delete from accounts"
        result = sanitize_for_engine(text)
        assert "drop" not in result.lower()
        assert "delete" not in result.lower()

    def test_strips_multiple_sql_keywords(self):
        text = "UNION SELECT * FROM users; DROP TABLE accounts"
        result = sanitize_for_engine(text)
        assert "UNION" not in result
        assert "SELECT" not in result
        assert "DROP" not in result

    def test_preserves_safe_content_order(self):
        text = "first SELECT second DROP third"
        result = sanitize_for_engine(text)
        # Safe words should remain in order
        first_idx = result.index("first")
        second_idx = result.index("second")
        third_idx = result.index("third")
        assert first_idx < second_idx < third_idx

    def test_rejects_over_1000_chars(self):
        text = "x" * 1001
        with pytest.raises(ValueError, match="1000"):
            sanitize_for_engine(text)

    def test_accepts_exactly_1000_chars(self):
        text = "x" * 1000
        result = sanitize_for_engine(text)
        assert result == text

    def test_combined_dangerous_patterns(self):
        text = "Hello <script>alert(1)</script>; SELECT * | cat $file"
        result = sanitize_for_engine(text)
        assert "<script>" not in result.lower()
        assert ";" not in result
        assert "|" not in result
        assert "$" not in result
        assert "SELECT" not in result
        assert "Hello" in result

    def test_sql_keyword_only_matches_whole_words(self):
        # "selection" contains "select" but should NOT be stripped
        text = "The selection process is important"
        result = sanitize_for_engine(text)
        assert "selection" in result
