"""Property-based tests for chart rendering.

Feature: safe-haven-demo-booth
Property 14: Analytics query chart rendering produces valid visualization

For any analytics query that retrieves tabular data from the Gold_Table, the chart
renderer SHALL produce a valid Plotly figure with at least one data trace, a non-empty
title, and axis labels. The chart type SHALL be one of "bar", "line", or "pie". If
chart generation fails, the fallback SHALL be a formatted text table containing all
retrieved data rows.
"""

import asyncio
from typing import Any, Dict, List

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from aws_demo_booth.chart_renderer import ChartRenderer, ChartResult

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


# =============================================================================
# Strategies for Property 14
# =============================================================================

# Valid chart types as defined in the spec
VALID_CHART_TYPES = ["bar", "line", "pie"]

# Keywords that influence chart type detection
ALL_CHART_KEYWORDS = [
    "by region", "by product", "compare", "top",  # bar
    "trend", "monthly", "over time", "growth", "quarterly",  # line
    "distribution", "proportion", "share", "breakdown", "percentage",  # pie
]


def _column_name() -> st.SearchStrategy[str]:
    """Generate valid column names for tabular data.

    Column names are non-empty strings of lowercase letters and underscores,
    representing typical database column names.
    """
    return st.from_regex(r"[a-z][a-z_]{0,14}", fullmatch=True)


def _numeric_value() -> st.SearchStrategy:
    """Generate numeric values for chart data (integers and floats)."""
    return st.one_of(
        st.integers(min_value=0, max_value=10000),
        st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    )


def _label_value() -> st.SearchStrategy[str]:
    """Generate label values for the x-axis / category column."""
    return st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-",
        min_size=1,
        max_size=20,
    )


def _tabular_data(min_rows: int = 1, max_rows: int = 10) -> st.SearchStrategy[List[Dict[str, Any]]]:
    """Generate random tabular data sets with 2+ columns and 1-10 rows.

    Each row is a dictionary with the same set of keys. The first column
    contains label values, the second column contains numeric values.
    Additional columns may also be present.
    """
    return _column_name().flatmap(
        lambda x_col: _column_name().filter(lambda y: y != x_col).flatmap(
            lambda y_col: st.lists(
                st.fixed_dictionaries({
                    x_col: _label_value(),
                    y_col: _numeric_value(),
                }),
                min_size=min_rows,
                max_size=max_rows,
            )
        )
    )


def _query_string() -> st.SearchStrategy[str]:
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


def _query_with_keyword(keywords: List[str]) -> st.SearchStrategy[str]:
    """Generate query strings that contain at least one keyword from the given list."""
    return st.builds(
        lambda prefix, kw, suffix: f"{prefix} {kw} {suffix}",
        prefix=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz ",
            min_size=1,
            max_size=20,
        ),
        kw=st.sampled_from(keywords),
        suffix=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz ?",
            min_size=0,
            max_size=20,
        ),
    )


# =============================================================================
# Property 14: Analytics query chart rendering produces valid visualization
# =============================================================================


class TestProperty14ChartRenderingValidVisualization:
    """Property 14: Analytics query chart rendering produces valid visualization.

    For any analytics query that retrieves tabular data from the Gold_Table,
    the chart renderer SHALL produce a valid Plotly figure with at least one
    data trace, a non-empty title, and axis labels. The chart type SHALL be
    one of "bar", "line", or "pie". If chart generation fails, the fallback
    SHALL be a formatted text table containing all retrieved data rows.

    **Validates: Requirements 11.2, 11.3, 11.8, 11.9**
    """

    @settings(max_examples=100, deadline=None)
    @given(query=_query_string(), data=_tabular_data())
    @pytest.mark.asyncio
    async def test_valid_data_produces_figure_with_trace(self, query: str, data: List[Dict[str, Any]]):
        """Chart renderer produces a valid Plotly figure with at least one data trace.

        For any analytics query with valid tabular data (2+ columns, 1-10 rows),
        the chart renderer SHALL produce a ChartResult with a non-None figure
        containing at least one data trace.

        Feature: safe-haven-demo-booth, Property 14: Chart rendering
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
            f"but got {len(result.figure.data)} traces. Query: '{query}'"
        )

    @settings(max_examples=100, deadline=None)
    @given(query=_query_string(), data=_tabular_data())
    @pytest.mark.asyncio
    async def test_valid_data_produces_non_empty_title(self, query: str, data: List[Dict[str, Any]]):
        """Chart renderer produces a figure with a non-empty title.

        For any analytics query with valid tabular data, the rendered chart
        SHALL have a non-empty title derived from the query context.

        Feature: safe-haven-demo-booth, Property 14: Chart rendering
        **Validates: Requirements 11.2, 11.3**
        """
        renderer = ChartRenderer()
        result: ChartResult = await renderer.render_chart(query=query, data=data)

        assert result.figure is not None, (
            f"Expected a valid figure but got None. Error: {result.error}"
        )
        title = result.figure.layout.title
        # Plotly stores title as either a string or a Title object with .text
        title_text = title.text if hasattr(title, "text") else str(title)
        assert title_text and title_text.strip(), (
            f"Expected non-empty title on the figure, but got: '{title_text}'. "
            f"Query: '{query}'"
        )

    @settings(max_examples=100, deadline=None)
    @given(query=_query_string(), data=_tabular_data())
    @pytest.mark.asyncio
    async def test_chart_type_is_valid(self, query: str, data: List[Dict[str, Any]]):
        """Chart type SHALL be one of "bar", "line", or "pie".

        For any analytics query with valid tabular data, the ChartResult
        chart_type field SHALL be one of the three valid chart types.

        Feature: safe-haven-demo-booth, Property 14: Chart rendering
        **Validates: Requirements 11.8**
        """
        renderer = ChartRenderer()
        result: ChartResult = await renderer.render_chart(query=query, data=data)

        assert result.chart_type in VALID_CHART_TYPES, (
            f"Expected chart_type to be one of {VALID_CHART_TYPES}, "
            f"but got '{result.chart_type}'. Query: '{query}'"
        )

    @settings(max_examples=100, deadline=None)
    @given(query=_query_string(), data=_tabular_data())
    @pytest.mark.asyncio
    async def test_fallback_table_is_non_empty_when_data_provided(self, query: str, data: List[Dict[str, Any]]):
        """Fallback table is non-empty when data is provided.

        For any analytics query with valid tabular data, the ChartResult
        fallback_table field SHALL be a non-empty string containing all
        retrieved data rows as a formatted text table.

        Feature: safe-haven-demo-booth, Property 14: Chart rendering
        **Validates: Requirements 11.9**
        """
        renderer = ChartRenderer()
        result: ChartResult = await renderer.render_chart(query=query, data=data)

        assert result.fallback_table, (
            f"Expected non-empty fallback_table when data has {len(data)} rows, "
            f"but got empty string. Query: '{query}'"
        )
        # Verify all data rows are represented in the fallback table
        for row in data:
            for value in row.values():
                assert str(value) in result.fallback_table, (
                    f"Expected value '{value}' to appear in fallback_table, "
                    f"but it was not found. Table:\n{result.fallback_table}"
                )

    @settings(max_examples=100, deadline=None)
    @given(query=_query_string())
    @pytest.mark.asyncio
    async def test_empty_data_falls_back_to_error(self, query: str):
        """Empty data input falls back with error (no figure produced).

        When chart generation receives empty data, the renderer SHALL
        return a ChartResult with figure=None and a non-None error,
        indicating fallback to text table display.

        Feature: safe-haven-demo-booth, Property 14: Chart rendering
        **Validates: Requirements 11.9**
        """
        renderer = ChartRenderer()
        result: ChartResult = await renderer.render_chart(query=query, data=[])

        assert result.figure is None, (
            f"Expected figure=None for empty data, but got a figure. Query: '{query}'"
        )
        assert result.error is not None, (
            f"Expected non-None error for empty data, but got None. Query: '{query}'"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        query=_query_string(),
        label=_label_value(),
    )
    @pytest.mark.asyncio
    async def test_single_column_data_falls_back_to_error(self, query: str, label: str):
        """Data with fewer than 2 columns falls back with error.

        When chart generation receives data with only one column (insufficient
        for x/y axis mapping), the renderer SHALL return a ChartResult with
        figure=None and a non-None error.

        Feature: safe-haven-demo-booth, Property 14: Chart rendering
        **Validates: Requirements 11.9**
        """
        # Single-column data: only one key per row
        single_col_data = [{"category": label}]

        renderer = ChartRenderer()
        result: ChartResult = await renderer.render_chart(query=query, data=single_col_data)

        assert result.figure is None, (
            f"Expected figure=None for single-column data, but got a figure. "
            f"Query: '{query}'"
        )
        assert result.error is not None, (
            f"Expected non-None error for single-column data, but got None. "
            f"Query: '{query}'"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        query=_query_with_keyword(["by region", "by product", "compare", "top"]),
        data=_tabular_data(),
    )
    @pytest.mark.asyncio
    async def test_bar_keywords_produce_bar_chart(self, query: str, data: List[Dict[str, Any]]):
        """Queries with bar chart keywords produce bar chart type.

        When the query contains keywords indicating categorical comparison
        (by region, by product, compare, top), the chart type SHALL be "bar".

        Feature: safe-haven-demo-booth, Property 14: Chart rendering
        **Validates: Requirements 11.8**
        """
        # Ensure no pie or line keywords are present to avoid ambiguity
        query_lower = query.lower()
        assume(not any(kw in query_lower for kw in [
            "distribution", "proportion", "share", "breakdown", "percentage",
            "trend", "monthly", "over time", "growth", "quarterly",
        ]))

        renderer = ChartRenderer()
        result: ChartResult = await renderer.render_chart(query=query, data=data)

        assert result.chart_type == "bar", (
            f"Expected chart_type='bar' for query with bar keywords, "
            f"but got '{result.chart_type}'. Query: '{query}'"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        query=_query_with_keyword(["trend", "monthly", "over time", "growth", "quarterly"]),
        data=_tabular_data(),
    )
    @pytest.mark.asyncio
    async def test_line_keywords_produce_line_chart(self, query: str, data: List[Dict[str, Any]]):
        """Queries with line chart keywords produce line chart type.

        When the query contains keywords indicating time-series trends
        (trend, monthly, over time, growth, quarterly), the chart type
        SHALL be "line".

        Feature: safe-haven-demo-booth, Property 14: Chart rendering
        **Validates: Requirements 11.8**
        """
        # Ensure no pie keywords are present (pie is checked first in detection)
        query_lower = query.lower()
        assume(not any(kw in query_lower for kw in [
            "distribution", "proportion", "share", "breakdown", "percentage",
        ]))

        renderer = ChartRenderer()
        result: ChartResult = await renderer.render_chart(query=query, data=data)

        assert result.chart_type == "line", (
            f"Expected chart_type='line' for query with line keywords, "
            f"but got '{result.chart_type}'. Query: '{query}'"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        query=_query_with_keyword(["distribution", "proportion", "share", "breakdown", "percentage"]),
        data=_tabular_data(),
    )
    @pytest.mark.asyncio
    async def test_pie_keywords_produce_pie_chart(self, query: str, data: List[Dict[str, Any]]):
        """Queries with pie chart keywords produce pie chart type.

        When the query contains keywords indicating proportional distribution
        (distribution, proportion, share, breakdown, percentage), the chart
        type SHALL be "pie".

        Feature: safe-haven-demo-booth, Property 14: Chart rendering
        **Validates: Requirements 11.8**
        """
        renderer = ChartRenderer()
        result: ChartResult = await renderer.render_chart(query=query, data=data)

        assert result.chart_type == "pie", (
            f"Expected chart_type='pie' for query with pie keywords, "
            f"but got '{result.chart_type}'. Query: '{query}'"
        )

    @settings(max_examples=100, deadline=None)
    @given(
        query=_query_string(),
        data=_tabular_data(),
        chart_type=st.sampled_from(VALID_CHART_TYPES),
    )
    @pytest.mark.asyncio
    async def test_explicit_chart_type_is_respected(self, query: str, data: List[Dict[str, Any]], chart_type: str):
        """When chart_type is explicitly specified, it SHALL be used regardless of query.

        For any explicit chart_type parameter passed to render_chart, the
        resulting ChartResult SHALL use that chart type.

        Feature: safe-haven-demo-booth, Property 14: Chart rendering
        **Validates: Requirements 11.8**
        """
        renderer = ChartRenderer()
        result: ChartResult = await renderer.render_chart(
            query=query, data=data, chart_type=chart_type
        )

        assert result.chart_type == chart_type, (
            f"Expected chart_type='{chart_type}' when explicitly specified, "
            f"but got '{result.chart_type}'. Query: '{query}'"
        )
