"""Input validation and sanitization for the Safe Haven Demo Booth.

Provides prompt validation (length, whitespace checks) and sanitization
(stripping dangerous patterns like script tags, shell metacharacters, SQL keywords)
before prompts are sent to either engine.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationResult:
    """Result of prompt validation."""

    is_valid: bool
    error_message: Optional[str]
    sanitized_text: str


# --- Constants ---

MAX_PROMPT_LENGTH = 500
MAX_ENGINE_LENGTH = 1000

# Pattern to match <script>...</script> tags (including self-closing and with attributes)
_SCRIPT_TAG_PATTERN = re.compile(
    r"<\s*script[^>]*>.*?<\s*/\s*script\s*>|<\s*script[^>]*/\s*>",
    re.IGNORECASE | re.DOTALL,
)

# Shell metacharacters to strip
_SHELL_METACHARACTERS = set(";|&$`")

# SQL keywords to strip (case-insensitive, whole words only)
_SQL_KEYWORDS = [
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

_SQL_KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(_SQL_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def validate_prompt(text: str) -> ValidationResult:
    """Validate a visitor-submitted prompt.

    Accepts prompts that are 1-500 characters after trimming whitespace.
    Rejects empty strings, whitespace-only strings, and strings exceeding
    500 characters after trim.

    Args:
        text: The raw input text from the visitor.

    Returns:
        ValidationResult with is_valid flag, optional error message,
        and the trimmed text if valid (empty string if invalid).
    """
    trimmed = text.strip()

    if not trimmed:
        return ValidationResult(
            is_valid=False,
            error_message="Prompt cannot be empty or whitespace-only.",
            sanitized_text="",
        )

    if len(trimmed) > MAX_PROMPT_LENGTH:
        return ValidationResult(
            is_valid=False,
            error_message=(
                f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters "
                f"(got {len(trimmed)} after trimming)."
            ),
            sanitized_text="",
        )

    return ValidationResult(
        is_valid=True,
        error_message=None,
        sanitized_text=trimmed,
    )


def sanitize_for_engine(text: str) -> str:
    """Sanitize a prompt before passing it to an engine.

    Strips script tags, shell metacharacters, and SQL keywords from the input.
    Raises ValueError if the input exceeds 1000 characters.

    Args:
        text: The prompt text to sanitize.

    Returns:
        The sanitized text with dangerous patterns removed.

    Raises:
        ValueError: If the input exceeds 1000 characters.
    """
    if len(text) > MAX_ENGINE_LENGTH:
        raise ValueError(
            f"Input exceeds maximum engine length of {MAX_ENGINE_LENGTH} characters "
            f"(got {len(text)})."
        )

    # Strip script tags
    result = _SCRIPT_TAG_PATTERN.sub("", text)

    # Strip shell metacharacters
    result = "".join(ch for ch in result if ch not in _SHELL_METACHARACTERS)

    # Strip SQL keywords (whole words, case-insensitive)
    result = _SQL_KEYWORD_PATTERN.sub("", result)

    return result
