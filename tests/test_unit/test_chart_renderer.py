"""Unit tests for the ChartRenderer and ChartResult.

Tests cover:
- ChartResult dataclass creation
- detect_chart_type() keyword matching logic
- build_bar_chart(), build_line_chart(), build_pie_chart() Plotly figure generation
- build_fallback_table() formatted text table output
- render_chart() async method with valid data, empty data, and error cases
- render_mock_chart() pre-programmed mock data for Mock_Mode
- _generate_summary() key insight text generation
"""

import pytest
import pytest_asyncio

from chart_renderer import (
    BAR_KEYWORDS,
    LINE_KEYWORDS,
    MOCK_MONTHLY_TRENDS,
    MOCK_PRODUCT_DISTRIBUTION,
    MOCK_REVENUE_BY_REGION,
    PIE_KEYWORDS,
    ChartRenderer,
    ChartResult,
)

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


@pytest.fixture
def renderer() -> ChartRenderer:
    return ChartRenderer()


@pytest.fixture
def sample_bar_data():
    return [
        {"region": "APAC", "revenue": 1200000},
        {"region": "EU", "revenue": 420000},
        {"region": "NA", "revenue": 890000},
    ]


@pytest.fixture
def sample_line_data():
    return [
        {"month": "Jan", "value": 100},
        {"month": "Feb", "value": 120},
        {"month": "Mar", "value": 150},
        {"month": "Apr", "value": 180},
    ]


@pytest.fixture
def sample_pie_data():
    return [
        {"product": "Widget Pro", "share": 45},
        {"product": "DataSync", "share": 30},
        {"product": "CloudBase", "share": 25},
    ]


# =============================================================================
# ChartResult dataclass tests
# =============================================================================


class TestChartResult:
    """Tests for the ChartResult dataclass."""

    def test_create_chart_result_with_figure(self):
        result = ChartResult(
            figure="mock_figure",
            chart_type="bar",
            summary_text="APAC leads with 1.2M",
            fallback_table="| region | revenue |",
            error=None,
        )
        assert result.figure == "mock_figure"
        assert result.chart_type == "bar"
        assert result.summary_text == "APAC leads with 1.2M"
        assert result.fallback_table == "| region | revenue |"
        assert result.error is None

    def test_create_chart_result_with_error(self):
        result = ChartResult(
            figure=None,
            chart_type="bar",
            summary_text="Chart generation failed.",
            fallback_table="| data |",
            error="Plotly is not installed",
        )
        assert result.figure is None
        assert result.error == "Plotly is not installed"

    def test_chart_type_values(self):
        for chart_type in ["bar", "line", "pie"]:
            result = ChartResult(
                figure=None,
                chart_type=chart_type,
                summary_text="",
                fallback_table="",
                error=None,
            )
            assert result.chart_type == chart_type


# =============================================================================
# detect_chart_type() tests
# =============================================================================


class TestDetectChartType:
    """Tests for chart type detection based on query keywords."""

    def test_bar_keyword_by_region(self, renderer: ChartRenderer):
        assert renderer.detect_chart_type("Show sales by region", ["region", "sales"]) == "bar"

    def test_bar_keyword_by_product(self, renderer: ChartRenderer):
        assert renderer.detect_chart_type("Revenue by product", ["product", "revenue"]) == "bar"

    def test_bar_keyword_compare(self, renderer: ChartRenderer):
        assert renderer.detect_chart_type("Compare these categories", ["category", "amount"]) == "bar"

    def test_bar_keyword_top(self, renderer: ChartRenderer):
        assert renderer.detect_chart_type("Show top performers", ["name", "score"]) == "bar"

    def test_line_keyword_trend(self, renderer: ChartRenderer):
        assert renderer.detect_chart_type("Show the trend", ["date", "value"]) == "line"

    def test_line_keyword_monthly(self, renderer: ChartRenderer):
        assert renderer.detect_chart_type("Monthly revenue report", ["month", "revenue"]) == "line"

    def test_line_keyword_over_time(self, renderer: ChartRenderer):
        assert renderer.detect_chart_type("Sales over time", ["period", "sales"]) == "line"

    def test_line_keyword_growth(self, renderer: ChartRenderer):
        assert renderer.detect_chart_type("Show growth metrics", ["year", "growth"]) == "line"

    def test_line_keyword_quarterly(self, renderer: ChartRenderer):
        assert renderer.detect_chart_type("Quarterly performance", ["quarter", "perf"]) == "line"

    def test_pie_keyword_distribution(self, renderer: ChartRenderer):
        assert renderer.detect_chart_type("Product distribution", ["product", "count"]) == "pie"

    def test_pie_keyword_proportion(self, renderer: ChartRenderer):
        assert renderer.detect_chart_type("Show proportion of sales", ["category", "amount"]) == "pie"

    def test_pie_keyword_share(self, renderer: ChartRenderer):
        assert renderer.detect_chart_type("Market share analysis", ["company", "share"]) == "pie"

    def test_pie_keyword_breakdown(self, renderer: ChartRenderer):
        assert renderer.detect_chart_type("Cost breakdown", ["category", "cost"]) == "pie"

    def test_pie_keyword_percentage(self, renderer: ChartRenderer):
        assert renderer.detect_chart_type("Show percentage of total", ["item", "pct"]) == "pie"

    def test_default_to_bar_when_no_keywords(self, renderer: ChartRenderer):
        assert renderer.detect_chart_type("Show me the data", ["col_a", "col_b"]) == "bar"

    def test_case_insensitive_detection(self, renderer: ChartRenderer):
        assert renderer.detect_chart_type("SHOW MONTHLY DATA", ["month", "val"]) == "line"

    def test_pie_takes_priority_over_bar(self, renderer: ChartRenderer):
        """Pie keywords are checked first, so they take priority."""
        assert renderer.detect_chart_type("distribution by region", ["region", "count"]) == "pie"

    def test_pie_takes_priority_over_line(self, renderer: ChartRenderer):
        """Pie keywords are checked first, so they take priority over line."""
        assert renderer.detect_chart_type("growth distribution", ["period", "value"]) == "pie"

    def test_line_takes_priority_over_bar(self, renderer: ChartRenderer):
        """Line keywords are checked before bar keywords."""
        assert renderer.detect_chart_type("monthly comparison by region trend", ["month", "val"]) == "line"


# =============================================================================
# build_bar_chart() tests
# =============================================================================


class TestBuildBarChart:
    """Tests for bar chart generation."""

    def test_builds_valid_bar_chart(self, renderer: ChartRenderer, sample_bar_data):
        fig = renderer.build_bar_chart(sample_bar_data, "region", "revenue", "Revenue by Region")
        assert fig is not None
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Bar)

    def test_bar_chart_has_correct_data(self, renderer: ChartRenderer, sample_bar_data):
        fig = renderer.build_bar_chart(sample_bar_data, "region", "revenue", "Revenue by Region")
        bar_trace = fig.data[0]
        assert list(bar_trace.x) == ["APAC", "EU", "NA"]
        assert list(bar_trace.y) == [1200000, 420000, 890000]

    def test_bar_chart_has_title(self, renderer: ChartRenderer, sample_bar_data):
        fig = renderer.build_bar_chart(sample_bar_data, "region", "revenue", "Revenue by Region")
        title_text = fig.layout.title.text if hasattr(fig.layout.title, "text") else str(fig.layout.title)
        assert "Revenue by Region" in title_text

    def test_bar_chart_has_axis_labels(self, renderer: ChartRenderer, sample_bar_data):
        fig = renderer.build_bar_chart(sample_bar_data, "region", "revenue", "Revenue by Region")
        assert fig.layout.xaxis.title.text == "region"
        assert fig.layout.yaxis.title.text == "revenue"

    def test_bar_chart_handles_missing_values(self, renderer: ChartRenderer):
        data = [{"region": "APAC"}, {"region": "EU", "revenue": 420000}]
        fig = renderer.build_bar_chart(data, "region", "revenue", "Test")
        assert list(fig.data[0].x) == ["APAC", "EU"]
        assert list(fig.data[0].y) == [0, 420000]


# =============================================================================
# build_line_chart() tests
# =============================================================================


class TestBuildLineChart:
    """Tests for line chart generation."""

    def test_builds_valid_line_chart(self, renderer: ChartRenderer, sample_line_data):
        fig = renderer.build_line_chart(sample_line_data, "month", "value", "Monthly Trends")
        assert fig is not None
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Scatter)

    def test_line_chart_has_correct_data(self, renderer: ChartRenderer, sample_line_data):
        fig = renderer.build_line_chart(sample_line_data, "month", "value", "Monthly Trends")
        scatter_trace = fig.data[0]
        assert list(scatter_trace.x) == ["Jan", "Feb", "Mar", "Apr"]
        assert list(scatter_trace.y) == [100, 120, 150, 180]

    def test_line_chart_uses_lines_and_markers(self, renderer: ChartRenderer, sample_line_data):
        fig = renderer.build_line_chart(sample_line_data, "month", "value", "Monthly Trends")
        assert fig.data[0].mode == "lines+markers"

    def test_line_chart_has_title(self, renderer: ChartRenderer, sample_line_data):
        fig = renderer.build_line_chart(sample_line_data, "month", "value", "Monthly Trends")
        title_text = fig.layout.title.text if hasattr(fig.layout.title, "text") else str(fig.layout.title)
        assert "Monthly Trends" in title_text

    def test_line_chart_has_axis_labels(self, renderer: ChartRenderer, sample_line_data):
        fig = renderer.build_line_chart(sample_line_data, "month", "value", "Monthly Trends")
        assert fig.layout.xaxis.title.text == "month"
        assert fig.layout.yaxis.title.text == "value"


# =============================================================================
# build_pie_chart() tests
# =============================================================================


class TestBuildPieChart:
    """Tests for pie chart generation."""

    def test_builds_valid_pie_chart(self, renderer: ChartRenderer, sample_pie_data):
        fig = renderer.build_pie_chart(sample_pie_data, "product", "share", "Product Distribution")
        assert fig is not None
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Pie)

    def test_pie_chart_has_correct_labels(self, renderer: ChartRenderer, sample_pie_data):
        fig = renderer.build_pie_chart(sample_pie_data, "product", "share", "Product Distribution")
        pie_trace = fig.data[0]
        assert list(pie_trace.labels) == ["Widget Pro", "DataSync", "CloudBase"]

    def test_pie_chart_has_correct_values(self, renderer: ChartRenderer, sample_pie_data):
        fig = renderer.build_pie_chart(sample_pie_data, "product", "share", "Product Distribution")
        pie_trace = fig.data[0]
        assert list(pie_trace.values) == [45, 30, 25]

    def test_pie_chart_has_title(self, renderer: ChartRenderer, sample_pie_data):
        fig = renderer.build_pie_chart(sample_pie_data, "product", "share", "Product Distribution")
        title_text = fig.layout.title.text if hasattr(fig.layout.title, "text") else str(fig.layout.title)
        assert "Product Distribution" in title_text


# =============================================================================
# build_fallback_table() tests
# =============================================================================


class TestBuildFallbackTable:
    """Tests for fallback text table generation."""

    def test_empty_data_returns_empty_string(self, renderer: ChartRenderer):
        assert renderer.build_fallback_table([]) == ""

    def test_empty_columns_returns_empty_string(self, renderer: ChartRenderer):
        assert renderer.build_fallback_table([{}]) == ""

    def test_single_row_table(self, renderer: ChartRenderer):
        data = [{"name": "Alice", "score": 95}]
        table = renderer.build_fallback_table(data)
        assert "name" in table
        assert "score" in table
        assert "Alice" in table
        assert "95" in table

    def test_multi_row_table(self, renderer: ChartRenderer, sample_bar_data):
        table = renderer.build_fallback_table(sample_bar_data)
        assert "region" in table
        assert "revenue" in table
        assert "APAC" in table
        assert "EU" in table
        assert "NA" in table
        assert "1200000" in table

    def test_table_has_header_and_separator(self, renderer: ChartRenderer, sample_bar_data):
        table = renderer.build_fallback_table(sample_bar_data)
        lines = table.split("\n")
        assert len(lines) >= 3  # header + separator + at least 1 data row
        assert "|" in lines[0]  # header has pipes
        assert "-" in lines[1]  # separator has dashes

    def test_table_contains_all_rows(self, renderer: ChartRenderer, sample_bar_data):
        table = renderer.build_fallback_table(sample_bar_data)
        lines = table.split("\n")
        # header + separator + 3 data rows = 5 lines
        assert len(lines) == 5


# =============================================================================
# render_chart() async method tests
# =============================================================================


class TestRenderChart:
    """Tests for the main render_chart() async method."""

    @pytest.mark.asyncio
    async def test_renders_bar_chart_for_categorical_query(self, renderer: ChartRenderer, sample_bar_data):
        result = await renderer.render_chart("Revenue by region", sample_bar_data)
        assert result.figure is not None
        assert result.chart_type == "bar"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_renders_line_chart_for_trend_query(self, renderer: ChartRenderer, sample_line_data):
        result = await renderer.render_chart("Show monthly trend", sample_line_data)
        assert result.figure is not None
        assert result.chart_type == "line"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_renders_pie_chart_for_distribution_query(self, renderer: ChartRenderer, sample_pie_data):
        result = await renderer.render_chart("Product distribution", sample_pie_data)
        assert result.figure is not None
        assert result.chart_type == "pie"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_explicit_chart_type_overrides_detection(self, renderer: ChartRenderer, sample_bar_data):
        result = await renderer.render_chart("Revenue by region", sample_bar_data, chart_type="pie")
        assert result.chart_type == "pie"
        assert result.figure is not None

    @pytest.mark.asyncio
    async def test_empty_data_returns_error(self, renderer: ChartRenderer):
        result = await renderer.render_chart("Show data", [])
        assert result.figure is None
        assert result.error is not None
        assert "No data" in result.error

    @pytest.mark.asyncio
    async def test_single_column_data_returns_error(self, renderer: ChartRenderer):
        result = await renderer.render_chart("Show data", [{"only_col": "value"}])
        assert result.figure is None
        assert result.error is not None
        assert "two columns" in result.error

    @pytest.mark.asyncio
    async def test_summary_text_is_non_empty(self, renderer: ChartRenderer, sample_bar_data):
        result = await renderer.render_chart("Revenue by region", sample_bar_data)
        assert result.summary_text
        assert len(result.summary_text) > 0

    @pytest.mark.asyncio
    async def test_fallback_table_always_present_with_data(self, renderer: ChartRenderer, sample_bar_data):
        result = await renderer.render_chart("Revenue by region", sample_bar_data)
        assert result.fallback_table
        assert "APAC" in result.fallback_table

    @pytest.mark.asyncio
    async def test_title_derived_from_query(self, renderer: ChartRenderer, sample_bar_data):
        result = await renderer.render_chart("Revenue by region?", sample_bar_data)
        title_text = result.figure.layout.title.text if hasattr(result.figure.layout.title, "text") else str(result.figure.layout.title)
        assert "Revenue by region" in title_text

    @pytest.mark.asyncio
    async def test_empty_query_uses_default_title(self, renderer: ChartRenderer, sample_bar_data):
        result = await renderer.render_chart("", sample_bar_data)
        title_text = result.figure.layout.title.text if hasattr(result.figure.layout.title, "text") else str(result.figure.layout.title)
        assert title_text == "Analytics Chart"


# =============================================================================
# render_mock_chart() tests
# =============================================================================


class TestRenderMockChart:
    """Tests for mock chart generation in Mock_Mode."""

    @pytest.mark.asyncio
    async def test_mock_revenue_by_region_default(self, renderer: ChartRenderer):
        """Default mock query returns revenue by region bar chart."""
        result = await renderer.render_mock_chart("Show me revenue data")
        assert result.figure is not None
        assert result.chart_type == "bar"
        assert "APAC" in result.summary_text

    @pytest.mark.asyncio
    async def test_mock_monthly_trends(self, renderer: ChartRenderer):
        """Query with line keywords returns monthly trends line chart."""
        result = await renderer.render_mock_chart("Show monthly trend")
        assert result.figure is not None
        assert result.chart_type == "line"
        assert "growth" in result.summary_text.lower() or "trend" in result.summary_text.lower()

    @pytest.mark.asyncio
    async def test_mock_product_distribution(self, renderer: ChartRenderer):
        """Query with pie keywords returns product distribution pie chart."""
        result = await renderer.render_mock_chart("Product distribution breakdown")
        assert result.figure is not None
        assert result.chart_type == "pie"
        assert "Widget Pro" in result.summary_text

    @pytest.mark.asyncio
    async def test_mock_chart_has_valid_figure(self, renderer: ChartRenderer):
        """Mock chart always produces a valid figure with at least one trace."""
        result = await renderer.render_mock_chart("Revenue by region")
        assert result.figure is not None
        assert len(result.figure.data) >= 1

    @pytest.mark.asyncio
    async def test_mock_chart_has_fallback_table(self, renderer: ChartRenderer):
        """Mock chart always includes a fallback table."""
        result = await renderer.render_mock_chart("Revenue by region")
        assert result.fallback_table
        assert len(result.fallback_table) > 0

    @pytest.mark.asyncio
    async def test_mock_chart_no_error(self, renderer: ChartRenderer):
        """Mock chart generation should not produce errors."""
        result = await renderer.render_mock_chart("Any query")
        assert result.error is None


# =============================================================================
# _generate_summary() tests
# =============================================================================


class TestGenerateSummary:
    """Tests for summary text generation."""

    def test_bar_chart_summary_shows_leader(self, renderer: ChartRenderer, sample_bar_data):
        summary = renderer._generate_summary(sample_bar_data, "region", "revenue", "bar")
        assert "APAC" in summary
        assert "1200000" in summary

    def test_line_chart_summary_shows_growth(self, renderer: ChartRenderer, sample_line_data):
        summary = renderer._generate_summary(sample_line_data, "month", "value", "line")
        assert "growth" in summary.lower() or "decline" in summary.lower()

    def test_pie_chart_summary_shows_percentage(self, renderer: ChartRenderer, sample_pie_data):
        summary = renderer._generate_summary(sample_pie_data, "product", "share", "pie")
        assert "Widget Pro" in summary
        assert "%" in summary

    def test_empty_data_summary(self, renderer: ChartRenderer):
        summary = renderer._generate_summary([], "x", "y", "bar")
        assert "No data" in summary

    def test_line_chart_growth_calculation(self, renderer: ChartRenderer):
        data = [{"month": "Jan", "value": 100}, {"month": "Dec", "value": 200}]
        summary = renderer._generate_summary(data, "month", "value", "line")
        assert "100%" in summary
        assert "growth" in summary

    def test_line_chart_decline_calculation(self, renderer: ChartRenderer):
        data = [{"month": "Jan", "value": 200}, {"month": "Dec", "value": 100}]
        summary = renderer._generate_summary(data, "month", "value", "line")
        assert "50%" in summary
        assert "decline" in summary
