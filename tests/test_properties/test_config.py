"""Property-based tests for configuration and environment variable detection.

Feature: safe-haven-demo-booth
"""

import logging
import os
import re
from unittest.mock import patch

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from config import AppConfig, REQUIRED_ENV_VARS
from presets import PresetManager

# Logger used by the config module
CONFIG_LOGGER_NAME = "config"


# --- Strategies ---

# All required env vars as a frozen set for subset generation
ALL_REQUIRED = frozenset(REQUIRED_ENV_VARS)


def _incomplete_env_subset() -> st.SearchStrategy[frozenset]:
    """Generate a proper subset of REQUIRED_ENV_VARS (at least one missing).

    Returns a frozenset of variable names that will be PRESENT in the environment.
    The complement (ALL_REQUIRED - subset) will be the missing variables.
    """
    # Generate subsets of indices to include (0 to len-1 vars present)
    # We need at least one missing, so max present is len(ALL_REQUIRED) - 1
    return st.frozensets(
        st.sampled_from(sorted(ALL_REQUIRED)),
        min_size=0,
        max_size=len(ALL_REQUIRED) - 1,
    )


def _env_var_value() -> st.SearchStrategy[str]:
    """Generate a non-empty value for an environment variable."""
    return st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_:/.",
        min_size=1,
        max_size=50,
    )


# =============================================================================
# Property 13: Missing environment variable detection and mock mode activation
# =============================================================================


class TestProperty13MissingEnvVarDetection:
    """Property 13: Missing environment variable detection and mock mode activation.

    For any subset of required environment variables that is incomplete (one or
    more missing), the application SHALL log an error message listing each missing
    variable by name and SHALL switch to Mock_Mode.

    **Validates: Requirements 10.2**
    """

    @settings(max_examples=20)
    @given(present_vars=_incomplete_env_subset(), values=st.data())
    def test_missing_vars_detected_exactly(self, present_vars: frozenset, values):
        """The returned missing_vars list contains exactly the variables not set.

        Feature: safe-haven-demo-booth, Property 13: Missing env var detection
        **Validates: Requirements 10.2**
        """
        # Build environment with only the present vars set to non-empty values
        env = {}
        for var_name in present_vars:
            env[var_name] = values.draw(_env_var_value(), label=var_name)

        expected_missing = ALL_REQUIRED - present_vars

        with patch.dict(os.environ, env, clear=True):
            _config, missing_vars = AppConfig.from_environment()

        assert set(missing_vars) == expected_missing, (
            f"Expected missing: {sorted(expected_missing)}, "
            f"got: {sorted(missing_vars)}"
        )

    @settings(max_examples=20)
    @given(present_vars=_incomplete_env_subset(), values=st.data())
    def test_error_log_lists_each_missing_variable(
        self, present_vars: frozenset, values
    ):
        """An error log message is produced listing each missing variable by name.

        Feature: safe-haven-demo-booth, Property 13: Missing env var detection
        **Validates: Requirements 10.2**
        """
        env = {}
        for var_name in present_vars:
            env[var_name] = values.draw(_env_var_value(), label=var_name)

        expected_missing = ALL_REQUIRED - present_vars
        # Ensure at least one is missing (guaranteed by strategy, but be explicit)
        assume(len(expected_missing) > 0)

        # Use a logging handler directly instead of caplog fixture
        config_logger = logging.getLogger(CONFIG_LOGGER_NAME)

        # Use a simple list-based handler to capture log records
        log_records: list[logging.LogRecord] = []

        class ListHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                log_records.append(record)

        list_handler = ListHandler()
        list_handler.setLevel(logging.ERROR)
        config_logger.addHandler(list_handler)

        try:
            with patch.dict(os.environ, env, clear=True):
                _config, _missing_vars = AppConfig.from_environment()

            # Combine all log messages
            log_text = " ".join(h.getMessage() for h in log_records)

            # Verify each missing variable name appears in the error log
            for var_name in expected_missing:
                assert var_name in log_text, (
                    f"Missing variable '{var_name}' not found in error log: {log_text!r}"
                )
        finally:
            config_logger.removeHandler(list_handler)

    @settings(max_examples=20)
    @given(present_vars=_incomplete_env_subset(), values=st.data())
    def test_mock_mode_should_activate_when_vars_missing(
        self, present_vars: frozenset, values
    ):
        """When missing_vars is non-empty, the caller should switch to Mock_Mode.

        The from_environment() method returns a non-empty missing_vars list,
        signaling to the caller that Mock_Mode must be activated.

        Feature: safe-haven-demo-booth, Property 13: Missing env var detection
        **Validates: Requirements 10.2**
        """
        env = {}
        for var_name in present_vars:
            env[var_name] = values.draw(_env_var_value(), label=var_name)

        expected_missing = ALL_REQUIRED - present_vars
        assume(len(expected_missing) > 0)

        with patch.dict(os.environ, env, clear=True):
            _config, missing_vars = AppConfig.from_environment()

        # Non-empty missing_vars signals Mock_Mode activation
        assert len(missing_vars) > 0, (
            "missing_vars should be non-empty when required vars are absent, "
            "signaling Mock_Mode activation"
        )
        # The missing_vars count matches expected
        assert len(missing_vars) == len(expected_missing)


# =============================================================================
# Property 12: Preset label constraints
# =============================================================================

# Regex pattern to detect at least one emoji character.
# Covers common emoji Unicode ranges including emoticons, symbols, dingbats,
# and supplemental symbols.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport and Map
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed characters
    "\U0000FE00-\U0000FE0F"  # Variation Selectors
    "\U00002600-\U000026FF"  # Misc symbols (☀, ⚡, etc.)
    "\U0000200D"             # Zero Width Joiner
    "\U00002B50-\U00002B55"  # Stars and circles
    "]"
)


def _contains_emoji(text: str) -> bool:
    """Return True if text contains at least one emoji character."""
    return bool(_EMOJI_PATTERN.search(text))


class TestProperty12PresetLabelConstraints:
    """Property 12: Preset label constraints.

    For all configured attack prompt presets, each preset label SHALL have a
    length of no more than 30 characters and SHALL contain at least one emoji
    character indicating the attack type.

    Feature: safe-haven-demo-booth, Property 12: Preset label constraints
    **Validates: Requirements 8.3**
    """

    def test_all_preset_labels_within_30_chars(self):
        """Every preset label must be at most 30 characters long.

        Feature: safe-haven-demo-booth, Property 12: Preset label constraints
        **Validates: Requirements 8.3**
        """
        manager = PresetManager()
        presets = manager.get_all_presets()

        assert len(presets) > 0, "PresetManager should have at least one preset"

        for preset in presets:
            assert len(preset.label) <= 30, (
                f"Preset '{preset.id}' label is {len(preset.label)} chars "
                f"(max 30): {preset.label!r}"
            )

    def test_all_preset_labels_contain_at_least_one_emoji(self):
        """Every preset label must contain at least one emoji character.

        Feature: safe-haven-demo-booth, Property 12: Preset label constraints
        **Validates: Requirements 8.3**
        """
        manager = PresetManager()
        presets = manager.get_all_presets()

        assert len(presets) > 0, "PresetManager should have at least one preset"

        for preset in presets:
            assert _contains_emoji(preset.label), (
                f"Preset '{preset.id}' label has no emoji: {preset.label!r}"
            )

    def test_preset_count_within_bounds(self):
        """The preset manager should have 5-10 attack presets and 2-4 analytics presets.

        Feature: safe-haven-demo-booth, Property 12: Preset label constraints
        **Validates: Requirements 8.3**
        """
        manager = PresetManager()
        attack_presets = manager.get_attack_presets()
        analytics_presets = manager.get_analytics_presets()

        assert 5 <= len(attack_presets) <= 10, (
            f"Expected 5-10 attack presets, got {len(attack_presets)}"
        )
        assert 2 <= len(analytics_presets) <= 4, (
            f"Expected 2-4 analytics presets, got {len(analytics_presets)}"
        )

    @settings(max_examples=1)
    @given(data=st.data())
    def test_property_all_presets_satisfy_label_constraints(self, data):
        """Property test: all configured presets satisfy label constraints.

        This is a static validation property — we iterate over all presets and
        verify the constraints hold universally. Using @given(data=st.data())
        with max_examples=1 since no random generation is needed.

        Feature: safe-haven-demo-booth, Property 12: Preset label constraints
        **Validates: Requirements 8.3**
        """
        manager = PresetManager()
        presets = manager.get_all_presets()

        for preset in presets:
            # Constraint 1: label length ≤ 30 characters
            assert len(preset.label) <= 30, (
                f"Preset '{preset.id}' label exceeds 30 chars "
                f"({len(preset.label)}): {preset.label!r}"
            )
            # Constraint 2: label contains at least one emoji
            assert _contains_emoji(preset.label), (
                f"Preset '{preset.id}' label missing emoji: {preset.label!r}"
            )


# =============================================================================
# Property 14: Analytics query chart rendering produces valid visualization
# =============================================================================

import asyncio
from typing import Any, Dict, List

import pytest

from chart_renderer import ChartRenderer, ChartResult

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None


# --- Strategies for Property 14 ---

# Valid chart types as defined in the spec
VALID_CHART_TYPES = ["bar", "line", "pie"]

# Keywords that influence chart type detection
ALL_CHART_KEYWORDS = [
    "by region", "by product", "compare", "top",  # bar
    "trend", "monthly", "over time", "growth", "quarterly",  # line
    "distribution", "proportion", "share", "breakdown", "percentage",  # pie
]


def _chart_column_name() -> st.SearchStrategy[str]:
    """Generate valid column names for tabular data."""
    return st.from_regex(r"[a-z][a-z_]{0,14}", fullmatch=True)


def _chart_numeric_value() -> st.SearchStrategy:
    """Generate numeric values for chart data (integers and floats)."""
    return st.one_of(
        st.integers(min_value=0, max_value=10000),
        st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    )


def _chart_label_value() -> st.SearchStrategy[str]:
    """Generate label values for the x-axis / category column."""
    return st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-",
        min_size=1,
        max_size=20,
    )


def _chart_tabular_data(min_rows: int = 1, max_rows: int = 10) -> st.SearchStrategy[List[Dict[str, Any]]]:
    """Generate random tabular data sets with 2+ columns and 1+ rows.

    Each row is a dictionary with the same set of keys. The first column
    contains label values, the second column contains numeric values.
    """
    return _chart_column_name().flatmap(
        lambda x_col: _chart_column_name().filter(lambda y: y != x_col).flatmap(
            lambda y_col: st.lists(
                st.fixed_dictionaries({
                    x_col: _chart_label_value(),
                    y_col: _chart_numeric_value(),
                }),
                min_size=min_rows,
                max_size=max_rows,
            )
        )
    )


def _chart_query_string() -> st.SearchStrategy[str]:
    """Generate random analytics query strings.

    Mixes arbitrary text with optional chart-type keywords to exercise
    the chart type detection logic.
    """
    base_text = st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ?",
        min_size=3,
        max_size=80,
    )
    keyword = st.sampled_from(ALL_CHART_KEYWORDS + ["sales", "revenue", "data", "show me"])

    return st.one_of(
        base_text,
        st.builds(lambda prefix, kw: f"{prefix} {kw}", prefix=base_text, kw=keyword),
    )


class TestProperty14ChartRendering:
    """Property 14: Analytics query chart rendering produces valid visualization.

    For any analytics query that retrieves tabular data from the Gold_Table,
    the chart renderer SHALL produce a valid Plotly figure with at least one
    data trace, a non-empty title, and axis labels. The chart type SHALL be
    one of "bar", "line", or "pie". If chart generation fails, the fallback
    SHALL be a formatted text table containing all retrieved data rows.

    Feature: safe-haven-demo-booth, Property 14: Analytics query chart rendering
    **Validates: Requirements 11.2, 11.3, 11.8, 11.9**
    """

    @settings(max_examples=100, deadline=None)
    @given(query=_chart_query_string(), data=_chart_tabular_data())
    @pytest.mark.asyncio
    async def test_render_chart_produces_figure_with_at_least_one_trace(
        self, query: str, data: List[Dict[str, Any]]
    ):
        """render_chart produces a valid Plotly figure with at least one data trace.

        Feature: safe-haven-demo-booth, Property 14: Analytics query chart rendering
        **Validates: Requirements 11.2, 11.3**
        """
        renderer = ChartRenderer()
        result: ChartResult = await renderer.render_chart(query=query, data=data)

        assert result.figure is not None, (
            f"Expected a valid Plotly figure for query '{query}' with {len(data)} rows, "
            f"but got None. Error: {result.error}"
        )
        assert len(result.figure.data) >= 1, (
            f"Expected at least one data trace in the figure, "
            f"but got {len(result.figure.data)} traces."
        )

    @settings(max_examples=100, deadline=None)
    @given(query=_chart_query_string(), data=_chart_tabular_data())
    @pytest.mark.asyncio
    async def test_render_chart_produces_non_empty_title(
        self, query: str, data: List[Dict[str, Any]]
    ):
        """render_chart produces a figure with a non-empty title.

        Feature: safe-haven-demo-booth, Property 14: Analytics query chart rendering
        **Validates: Requirements 11.2, 11.3**
        """
        renderer = ChartRenderer()
        result: ChartResult = await renderer.render_chart(query=query, data=data)

        assert result.figure is not None, (
            f"Expected a valid figure but got None. Error: {result.error}"
        )
        title = result.figure.layout.title
        title_text = title.text if hasattr(title, "text") else str(title)
        assert title_text and title_text.strip(), (
            f"Expected non-empty title on the figure, but got: '{title_text}'. "
            f"Query: '{query}'"
        )

    @settings(max_examples=100, deadline=None)
    @given(query=_chart_query_string(), data=_chart_tabular_data())
    @pytest.mark.asyncio
    async def test_render_chart_type_is_bar_line_or_pie(
        self, query: str, data: List[Dict[str, Any]]
    ):
        """Chart type SHALL be one of "bar", "line", or "pie".

        Feature: safe-haven-demo-booth, Property 14: Analytics query chart rendering
        **Validates: Requirements 11.8**
        """
        renderer = ChartRenderer()
        result: ChartResult = await renderer.render_chart(query=query, data=data)

        assert result.chart_type in VALID_CHART_TYPES, (
            f"Expected chart_type to be one of {VALID_CHART_TYPES}, "
            f"but got '{result.chart_type}'. Query: '{query}'"
        )

    @settings(max_examples=100, deadline=None)
    @given(query=_chart_query_string(), data=_chart_tabular_data())
    @pytest.mark.asyncio
    async def test_render_chart_has_axis_labels(
        self, query: str, data: List[Dict[str, Any]]
    ):
        """Chart figure SHALL have axis labels for bar and line charts.

        Feature: safe-haven-demo-booth, Property 14: Analytics query chart rendering
        **Validates: Requirements 11.2, 11.3**
        """
        renderer = ChartRenderer()
        result: ChartResult = await renderer.render_chart(query=query, data=data)

        assert result.figure is not None, (
            f"Expected a valid figure but got None. Error: {result.error}"
        )
        # For bar and line charts, axis labels should be present
        if result.chart_type in ("bar", "line"):
            x_title = result.figure.layout.xaxis.title
            y_title = result.figure.layout.yaxis.title
            x_text = x_title.text if hasattr(x_title, "text") else str(x_title)
            y_text = y_title.text if hasattr(y_title, "text") else str(y_title)
            assert x_text and x_text.strip(), (
                f"Expected non-empty x-axis label for {result.chart_type} chart, "
                f"but got: '{x_text}'"
            )
            assert y_text and y_text.strip(), (
                f"Expected non-empty y-axis label for {result.chart_type} chart, "
                f"but got: '{y_text}'"
            )

    @settings(max_examples=100, deadline=None)
    @given(query=_chart_query_string(), data=_chart_tabular_data())
    @pytest.mark.asyncio
    async def test_fallback_table_contains_all_data_rows(
        self, query: str, data: List[Dict[str, Any]]
    ):
        """Fallback table contains all retrieved data rows when data is valid.

        Feature: safe-haven-demo-booth, Property 14: Analytics query chart rendering
        **Validates: Requirements 11.9**
        """
        renderer = ChartRenderer()
        result: ChartResult = await renderer.render_chart(query=query, data=data)

        assert result.fallback_table, (
            f"Expected non-empty fallback_table when data has {len(data)} rows, "
            f"but got empty string."
        )
        # Verify all data rows are represented in the fallback table
        for row in data:
            for value in row.values():
                assert str(value) in result.fallback_table, (
                    f"Expected value '{value}' to appear in fallback_table, "
                    f"but it was not found."
                )

    @settings(max_examples=100, deadline=None)
    @given(query=_chart_query_string())
    @pytest.mark.asyncio
    async def test_empty_data_returns_error_with_fallback(self, query: str):
        """Empty data returns error with fallback (no figure produced).

        When chart generation receives empty data, the renderer SHALL
        return a ChartResult with figure=None and a non-None error message.

        Feature: safe-haven-demo-booth, Property 14: Analytics query chart rendering
        **Validates: Requirements 11.9**
        """
        renderer = ChartRenderer()
        result: ChartResult = await renderer.render_chart(query=query, data=[])

        assert result.figure is None, (
            f"Expected figure=None for empty data, but got a figure."
        )
        assert result.error is not None, (
            f"Expected non-None error for empty data, but got None."
        )

    @settings(max_examples=100, deadline=None)
    @given(query=_chart_query_string(), label=_chart_label_value())
    @pytest.mark.asyncio
    async def test_single_column_data_returns_error_with_fallback(
        self, query: str, label: str
    ):
        """Data with fewer than 2 columns returns error with fallback.

        When chart generation receives data with only one column (insufficient
        for x/y axis mapping), the renderer SHALL return figure=None with error.

        Feature: safe-haven-demo-booth, Property 14: Analytics query chart rendering
        **Validates: Requirements 11.9**
        """
        single_col_data = [{"category": label}]

        renderer = ChartRenderer()
        result: ChartResult = await renderer.render_chart(query=query, data=single_col_data)

        assert result.figure is None, (
            f"Expected figure=None for single-column data, but got a figure."
        )
        assert result.error is not None, (
            f"Expected non-None error for single-column data, but got None."
        )
