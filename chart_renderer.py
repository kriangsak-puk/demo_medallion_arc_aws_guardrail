"""Chart renderer for analytics query visualization.

Generates Plotly charts from structured query results returned by the
Strands Agent. Supports bar, line, and pie charts with automatic type
detection based on query context and data shape. Falls back to formatted
text tables when chart generation fails.

Includes mock chart generation for Mock_Mode with pre-programmed data
representing typical business analytics.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Keywords for chart type detection
BAR_KEYWORDS: List[str] = ["by region", "by product", "compare", "top"]
LINE_KEYWORDS: List[str] = ["trend", "monthly", "over time", "growth", "quarterly"]
PIE_KEYWORDS: List[str] = [
    "distribution",
    "proportion",
    "share",
    "breakdown",
    "percentage",
]

# Mock data for Mock_Mode chart generation
MOCK_REVENUE_BY_REGION: Dict[str, int] = {
    "APAC": 1200000,
    "EU": 420000,
    "NA": 890000,
}

MOCK_MONTHLY_TRENDS: Dict[str, int] = {
    "Jan": 100,
    "Feb": 120,
    "Mar": 150,
    "Apr": 180,
}

MOCK_PRODUCT_DISTRIBUTION: Dict[str, int] = {
    "Widget Pro": 45,
    "DataSync": 30,
    "CloudBase": 25,
}


@dataclass
class ChartResult:
    """Result of a chart rendering operation.

    Attributes:
        figure: The Plotly figure object, or None if chart generation failed.
        chart_type: The type of chart rendered ("bar", "line", or "pie").
        summary_text: A key insight text describing the chart data.
        fallback_table: A formatted text table as fallback when chart fails.
        error: Error message if chart generation failed, None otherwise.
    """

    figure: Optional[Any]  # Optional[go.Figure] — Any to avoid import issues
    chart_type: str
    summary_text: str
    fallback_table: str
    error: Optional[str]


class ChartRenderer:
    """Generates Plotly charts from structured query results.

    Supports automatic chart type detection based on query keywords and
    data column names. Provides fallback to formatted text tables when
    Plotly is unavailable or chart generation fails.
    """

    def detect_chart_type(self, query: str, columns: List[str]) -> str:
        """Infer the appropriate chart type from query context and data columns.

        Detection logic:
        - "bar" for categorical comparisons (keywords: by region, by product, compare, top)
        - "line" for time-series (keywords: trend, monthly, over time, growth, quarterly)
        - "pie" for proportional distributions (keywords: distribution, proportion, share, breakdown, percentage)
        - Defaults to "bar" if unclear

        Args:
            query: The analytics query string from the visitor.
            columns: List of column names in the retrieved data.

        Returns:
            One of "bar", "line", or "pie".
        """
        query_lower = query.lower()

        # Check for pie chart keywords first (distribution/proportion)
        for keyword in PIE_KEYWORDS:
            if keyword in query_lower:
                return "pie"

        # Check for line chart keywords (time-series)
        for keyword in LINE_KEYWORDS:
            if keyword in query_lower:
                return "line"

        # Check for bar chart keywords (categorical)
        for keyword in BAR_KEYWORDS:
            if keyword in query_lower:
                return "bar"

        # Default to bar chart if unclear
        return "bar"

    def build_bar_chart(
        self, data: List[Dict[str, Any]], x_col: str, y_col: str, title: str
    ) -> "go.Figure":
        """Build a bar chart for categorical comparisons.

        Args:
            data: List of dictionaries containing the data rows.
            x_col: Column name for the x-axis (categories).
            y_col: Column name for the y-axis (values).
            title: Chart title.

        Returns:
            A Plotly Figure object with the bar chart.

        Raises:
            RuntimeError: If Plotly is not available.
        """
        if go is None:
            raise RuntimeError("Plotly is not installed")

        x_values = [row.get(x_col, "") for row in data]
        y_values = [row.get(y_col, 0) for row in data]

        fig = go.Figure(
            data=[go.Bar(x=x_values, y=y_values, name=y_col)],
            layout=go.Layout(
                title=title,
                xaxis_title=x_col,
                yaxis_title=y_col,
            ),
        )
        return fig

    def build_line_chart(
        self, data: List[Dict[str, Any]], x_col: str, y_col: str, title: str
    ) -> "go.Figure":
        """Build a line chart for time-series trends.

        Args:
            data: List of dictionaries containing the data rows.
            x_col: Column name for the x-axis (time periods).
            y_col: Column name for the y-axis (values).
            title: Chart title.

        Returns:
            A Plotly Figure object with the line chart.

        Raises:
            RuntimeError: If Plotly is not available.
        """
        if go is None:
            raise RuntimeError("Plotly is not installed")

        x_values = [row.get(x_col, "") for row in data]
        y_values = [row.get(y_col, 0) for row in data]

        fig = go.Figure(
            data=[go.Scatter(x=x_values, y=y_values, mode="lines+markers", name=y_col)],
            layout=go.Layout(
                title=title,
                xaxis_title=x_col,
                yaxis_title=y_col,
            ),
        )
        return fig

    def build_pie_chart(
        self, data: List[Dict[str, Any]], labels_col: str, values_col: str, title: str
    ) -> "go.Figure":
        """Build a pie chart for proportional distributions.

        Args:
            data: List of dictionaries containing the data rows.
            labels_col: Column name for the pie slice labels.
            values_col: Column name for the pie slice values.
            title: Chart title.

        Returns:
            A Plotly Figure object with the pie chart.

        Raises:
            RuntimeError: If Plotly is not installed.
        """
        if go is None:
            raise RuntimeError("Plotly is not installed")

        labels = [row.get(labels_col, "") for row in data]
        values = [row.get(values_col, 0) for row in data]

        fig = go.Figure(
            data=[go.Pie(labels=labels, values=values)],
            layout=go.Layout(title=title),
        )
        return fig

    def build_fallback_table(self, data: List[Dict[str, Any]]) -> str:
        """Build a formatted text table from data when chart generation fails.

        Creates a markdown-style table with column headers and aligned rows.

        Args:
            data: List of dictionaries containing the data rows.

        Returns:
            A formatted text table string. Returns empty string if data is empty.
        """
        if not data:
            return ""

        # Get all column names from the first row
        columns = list(data[0].keys())
        if not columns:
            return ""

        # Calculate column widths
        col_widths = {col: len(str(col)) for col in columns}
        for row in data:
            for col in columns:
                value_len = len(str(row.get(col, "")))
                if value_len > col_widths[col]:
                    col_widths[col] = value_len

        # Build header
        header = "| " + " | ".join(
            str(col).ljust(col_widths[col]) for col in columns
        ) + " |"
        separator = "|" + "|".join(
            "-" * (col_widths[col] + 2) for col in columns
        ) + "|"

        # Build rows
        rows = []
        for row in data:
            row_str = "| " + " | ".join(
                str(row.get(col, "")).ljust(col_widths[col]) for col in columns
            ) + " |"
            rows.append(row_str)

        return "\n".join([header, separator] + rows)

    async def render_chart(
        self,
        query: str,
        data: List[Dict[str, Any]],
        chart_type: Optional[str] = None,
    ) -> ChartResult:
        """Render a chart from structured query results.

        Detects the appropriate chart type from query context if not specified,
        builds the Plotly figure, and generates a summary text. Falls back to
        a formatted text table if chart generation fails.

        Args:
            query: The analytics query string from the visitor.
            data: List of dictionaries containing the retrieved data rows.
            chart_type: Optional explicit chart type ("bar", "line", "pie").
                If None, the type is inferred from the query and data columns.

        Returns:
            A ChartResult containing the figure, chart type, summary text,
            fallback table, and any error information.
        """
        fallback_table = self.build_fallback_table(data)

        # Validate input data
        if not data:
            return ChartResult(
                figure=None,
                chart_type="bar",
                summary_text="No data available for visualization.",
                fallback_table=fallback_table,
                error="No data provided for chart generation.",
            )

        columns = list(data[0].keys())
        if len(columns) < 2:
            return ChartResult(
                figure=None,
                chart_type="bar",
                summary_text="Insufficient columns for chart generation.",
                fallback_table=fallback_table,
                error="At least two columns are required for chart generation.",
            )

        # Detect chart type if not specified
        if chart_type is None:
            chart_type = self.detect_chart_type(query, columns)

        # Use first column as labels/x-axis, second as values/y-axis
        x_col = columns[0]
        y_col = columns[1]

        # Generate title from query
        title = query.strip().rstrip("?").strip()
        if not title:
            title = "Analytics Chart"

        # Build the chart
        try:
            if chart_type == "line":
                figure = self.build_line_chart(data, x_col, y_col, title)
            elif chart_type == "pie":
                figure = self.build_pie_chart(data, x_col, y_col, title)
            else:
                # Default to bar
                chart_type = "bar"
                figure = self.build_bar_chart(data, x_col, y_col, title)
        except Exception as e:
            logger.error("Chart generation failed: %s", str(e))
            return ChartResult(
                figure=None,
                chart_type=chart_type,
                summary_text="Chart generation failed. Showing data as table.",
                fallback_table=fallback_table,
                error=str(e),
            )

        # Generate summary text
        summary_text = self._generate_summary(data, x_col, y_col, chart_type)

        return ChartResult(
            figure=figure,
            chart_type=chart_type,
            summary_text=summary_text,
            fallback_table=fallback_table,
            error=None,
        )

    async def render_mock_chart(self, query: str) -> ChartResult:
        """Generate a pre-programmed chart for Mock_Mode.

        Selects mock data based on query keywords and generates an
        appropriate chart. Uses the same chart type detection logic
        as live mode.

        Mock data sets:
        - Revenue by region: {"APAC": 1200000, "EU": 420000, "NA": 890000}
        - Monthly trends: {"Jan": 100, "Feb": 120, "Mar": 150, "Apr": 180}
        - Product distribution: {"Widget Pro": 45, "DataSync": 30, "CloudBase": 25}

        Args:
            query: The analytics query string from the visitor.

        Returns:
            A ChartResult with pre-programmed mock chart data.
        """
        query_lower = query.lower()

        # Select mock data based on query context
        if any(kw in query_lower for kw in LINE_KEYWORDS):
            # Monthly trends data
            data = [
                {"month": k, "value": v} for k, v in MOCK_MONTHLY_TRENDS.items()
            ]
            chart_type = "line"
            summary_text = "Monthly trend shows consistent growth from 100 to 180 units (80% increase over 4 months)."
        elif any(kw in query_lower for kw in PIE_KEYWORDS):
            # Product distribution data
            data = [
                {"product": k, "share": v}
                for k, v in MOCK_PRODUCT_DISTRIBUTION.items()
            ]
            chart_type = "pie"
            summary_text = "Widget Pro leads with 45% market share, followed by DataSync (30%) and CloudBase (25%)."
        else:
            # Default: Revenue by region
            data = [
                {"region": k, "revenue": v}
                for k, v in MOCK_REVENUE_BY_REGION.items()
            ]
            chart_type = "bar"
            summary_text = "APAC region leads with ฿1.2M in revenue, followed by NA (฿890K) and EU (฿420K)."

        return await self.render_chart(query, data, chart_type=chart_type)

    def _generate_summary(
        self,
        data: List[Dict[str, Any]],
        x_col: str,
        y_col: str,
        chart_type: str,
    ) -> str:
        """Generate a key insight text from the chart data.

        Args:
            data: The data rows used for the chart.
            x_col: The x-axis / labels column name.
            y_col: The y-axis / values column name.
            chart_type: The chart type being rendered.

        Returns:
            A summary string describing the key insight.
        """
        if not data:
            return "No data available."

        # Find the maximum value entry
        try:
            max_row = max(data, key=lambda row: float(row.get(y_col, 0)))
            max_label = str(max_row.get(x_col, "Unknown"))
            max_value = max_row.get(y_col, 0)
        except (ValueError, TypeError):
            return f"Data contains {len(data)} entries across {x_col}."

        if chart_type == "pie":
            total = sum(float(row.get(y_col, 0)) for row in data)
            if total > 0:
                percentage = (float(max_value) / total) * 100
                return f"{max_label} leads with {percentage:.0f}% of total {y_col}."
            return f"{max_label} has the highest {y_col}."
        elif chart_type == "line":
            first_val = data[0].get(y_col, 0)
            last_val = data[-1].get(y_col, 0)
            try:
                first_num = float(first_val)
                last_num = float(last_val)
                if first_num > 0:
                    change = ((last_num - first_num) / first_num) * 100
                    direction = "growth" if change > 0 else "decline"
                    return f"Trend shows {abs(change):.0f}% {direction} from {data[0].get(x_col)} to {data[-1].get(x_col)}."
            except (ValueError, TypeError):
                pass
            return f"Trend data spans {len(data)} periods."
        else:
            # Bar chart
            return f"{max_label} leads with {max_value} in {y_col}."
