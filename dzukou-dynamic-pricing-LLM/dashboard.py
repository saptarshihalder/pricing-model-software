import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OVERVIEW_CSV = BASE_DIR / "Dzukou_Pricing_Overview_With_Names - Copy.csv"
RECOMMENDED_CSV = BASE_DIR / "recommended_prices.csv"
OUT_HTML = BASE_DIR / "dashboard.html"


def load_data():
    """Load overview and recommendation data and compute deltas."""
    overview = pd.read_csv(OVERVIEW_CSV, encoding="cp1252")
    recommended = pd.read_csv(RECOMMENDED_CSV)

    # Trim whitespace from column names and key fields so merges work
    overview = overview.rename(columns=lambda c: c.strip())
    recommended = recommended.rename(columns=lambda c: c.strip())

    for df in (overview, recommended):
        df["Product Name"] = df["Product Name"].str.strip()
        df["Product ID"] = df["Product ID"].astype(str).str.strip()

    # Clean currency values
    overview["Current Price"] = (
        overview["Current Price"].astype(str).str.replace(
            "€", "").str.replace(
            ",", "").astype(float))
    overview["Unit Cost"] = (
        overview["Unit Cost"].astype(str).str.replace(
            "€", "").str.replace(
            ",", "").astype(float))

    # Merge dataframes
    df = recommended.merge(
        overview[["Product Name", "Product ID", "Current Price", "Unit Cost"]],
        on=["Product Name", "Product ID"],
        how="left",
    )

    # Convert string values to numeric
    df["Recommended Price"] = df["Recommended Price"].astype(float)
    df["Price Delta"] = df["Recommended Price"] - df["Current Price"]
    df["Price Delta %"] = (df["Price Delta"] / df["Current Price"]) * 100

    # Handle profit delta
    if "Profit Delta" in df.columns:
        df["Profit Delta"] = df["Profit Delta"].astype(float)

    # Parse AB P-Value properly (handle both decimal and scientific notation)
    if "AB P-Value" in df.columns:
        df["AB P-Value"] = pd.to_numeric(df["AB P-Value"], errors="coerce")

        def _sig_label(val):
            if pd.notna(val) and val < 0.05:
                return "Significant"
            return "Not Significant"

        df["AB Significance"] = df["AB P-Value"].apply(_sig_label)

    return df


def build_dashboard(df, out_path=OUT_HTML):
    """Generate an enhanced HTML dashboard.

    The output includes improved visuals and metrics.
    """
    # Calculate summary metrics
    total_profit_increase = df["Profit Delta"].sum()
    avg_price_increase = df["Price Delta %"].mean()
    products_with_increase = (df["Price Delta"] > 0).sum()
    products_with_decrease = (df["Price Delta"] < 0).sum()

    # Statistical significance metrics
    if "AB P-Value" in df.columns:
        significant_changes = (df["AB P-Value"] < 0.05).sum()
        avg_p_value = df["AB P-Value"].mean()
    else:
        significant_changes = 0
        avg_p_value = 1.0

    # Create profit delta visualization with gradient colors
    fig_delta = go.Figure()
    fig_delta.add_trace(
        go.Bar(
            x=df["Product Name"],
            y=df["Profit Delta"],
            marker=dict(
                color=df["Profit Delta"],
                colorscale=[[0, "#e74c3c"], [0.5, "#f39c12"], [1, "#27ae60"]],
                line=dict(width=1, color="rgba(255,255,255,0.3)"),
                colorbar=dict(title="Profit Δ"),
            ),
            text=[f"€{x:.2f}" for x in df["Profit Delta"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Profit Delta: €%{y:.2f}<extra></extra>"
            ),
        )
    )
    fig_delta.update_layout(
        title=dict(
            text="Profit Delta by Product",
            font=dict(
                size=24,
                family="Arial")),
        xaxis=dict(
            title="Products",
            tickangle=-45),
        yaxis=dict(
            title="Profit Delta (€)",
            gridcolor="rgba(128,128,128,0.2)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=500,
        margin=dict(
            t=80,
            b=120),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14),
    )

    # Price comparison chart
    fig_price = go.Figure()
    fig_price.add_trace(
        go.Bar(
            name="Current Price",
            x=df["Product Name"],
            y=df["Current Price"],
            marker=dict(color="#3498db", opacity=0.8),
            text=[f"€{x:.2f}" for x in df["Current Price"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Current: €%{y:.2f}<extra></extra>"
            ),
        )
    )
    fig_price.add_trace(
        go.Bar(
            name="Recommended Price",
            x=df["Product Name"],
            y=df["Recommended Price"],
            marker=dict(color="#e74c3c", opacity=0.8),
            text=[f"€{x:.2f}" for x in df["Recommended Price"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Recommended: €%{y:.2f}<extra></extra>"
            ),
        )
    )
    fig_price.update_layout(
        barmode="group",
        title=dict(
            text="Current vs Recommended Prices",
            font=dict(
                size=24,
                family="Arial")),
        xaxis=dict(
            title="Products",
            tickangle=-45),
        yaxis=dict(
            title="Price (€)",
            gridcolor="rgba(128,128,128,0.2)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=500,
        margin=dict(
            t=80,
            b=120),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.8)"),
    )

    # Price change percentage with trend line
    fig_percentage = go.Figure()
    fig_percentage.add_trace(
        go.Scatter(
            x=df["Product Name"],
            y=df["Price Delta %"],
            mode="lines+markers",
            line=dict(color="#9b59b6", width=3, shape="spline"),
            marker=dict(
                size=12,
                color=df["Price Delta %"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="% Change"),
            ),
            text=[f"{x:.1f}%" for x in df["Price Delta %"]],
            textposition="top center",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Price Change: %{y:.1f}%<extra></extra>"
            ),
        )
    )
    fig_percentage.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        opacity=0.5)
    fig_percentage.update_layout(
        title=dict(
            text="Price Change Percentage by Product",
            font=dict(
                size=24,
                family="Arial")),
        xaxis=dict(
            title="Products",
            tickangle=-45),
        yaxis=dict(
            title="Price Change (%)",
            gridcolor="rgba(128,128,128,0.2)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
        margin=dict(
            t=80,
            b=120),
    )

    # AB Test P-Value visualization
    if "AB P-Value" in df.columns:
        fig_pvalue = go.Figure()
        colors = [
            '#27ae60' if p < 0.05 else '#e74c3c'
            for p in df["AB P-Value"]
        ]
        fig_pvalue.add_trace(
            go.Bar(
                x=df["Product Name"],
                y=-np.log10(df["AB P-Value"] + 1e-10),
                marker=dict(color=colors, opacity=0.8),
                text=[
                    f"p={p:.4f}" if p > 0.0001 else f"p={p:.2e}"
                    for p in df["AB P-Value"]
                ],
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>P-Value: %{text}<br>"
                    "-log10(p): %{y:.2f}<extra></extra>"
                ),
            )
        )
        fig_pvalue.add_hline(
            y=-np.log10(0.05),
            line_dash="dash",
            line_color="red",
            opacity=0.7,
            annotation_text="p=0.05 threshold",
            annotation_position="top right",
        )
        fig_pvalue.update_layout(
            title=dict(
                text="A/B Test Statistical Significance",
                font=dict(
                    size=24,
                    family="Arial")),
            xaxis=dict(
                title="Products",
                tickangle=-45),
            yaxis=dict(
                title="-log10(P-Value)",
                gridcolor="rgba(128,128,128,0.2)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(
                t=80,
                b=120),
        )
    else:
        fig_pvalue = None

    # Format table data for HTML
    table_data = df.copy()
    for col in [
        "Current Price",
        "Recommended Price",
        "Price Delta",
            "Profit Delta"]:
        if col in table_data.columns:
            table_data[col] = table_data[col].apply(lambda x: f"€{x:.2f}")

    if "Price Delta %" in table_data.columns:
        table_data["Price Delta %"] = table_data["Price Delta %"].apply(
            lambda x: f"{x:.1f}%")

    if "AB P-Value" in table_data.columns:
        table_data["AB P-Value"] = table_data["AB P-Value"].apply(
            lambda x: f"{x:.4f}" if x > 0.0001 else f"{x:.2e}"
        )

    table_html = table_data.to_html(
        index=False,
        classes="table table-hover table-striped",
        table_id="data-table",
        escape=False,
    )

    # Build HTML content
    pvalue_section = ""
    if fig_pvalue:
        to_html = fig_pvalue.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={'displayModeBar': False},
        )
        pvalue_section = (
            "<div class='mb-5'>"
            f"{to_html}"
            "</div>"
        )

    html = f"""
<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='utf-8'>
    <meta name='viewport'
          content='width=device-width, initial-scale=1.0'>
    <title>Dzukou Pricing Dashboard</title>
    <link rel='stylesheet'
          href='https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'>
    <link
        href='https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;
        600;700&display=swap'
        rel='stylesheet'>
    <style>
        body{{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Poppins', sans-serif;
            color: #2c3e50;
            min-height: 100vh;
        }}
        .main-container{{
            background: rgba(255,255,255,0.95);
            margin: 20px auto;
            max-width: 1400px;
            border-radius: 25px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            padding: 40px;
        }}
        .metric-card{{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
            height: 100%;
        }}
        .metric-card:hover{{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        }}
        .metric-value{{
            font-size: 2.5rem;
            font-weight: 700;
            color: #2c3e50;
        }}
        .metric-label{{
            font-size: 1rem;
            color: #7f8c8d;
            margin-top: 10px;
        }}
        h1{{
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 30px;
        }}
        .table{{
            font-size: 0.9rem;
        }}
        .table th{{
            background-color: #667eea;
            color: white;
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        .plotly-graph-div{{
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        }}
    </style>
</head>
<body>
<div class='main-container'>
    <div class='mb-5 text-center'>
        <h1>🎯 Dzukou Pricing Dashboard</h1>
        <p class='lead'>
            Data-driven pricing recommendations for optimal profitability
        </p>
    </div>

    <div class='row text-center mb-5'>
        <div class='col-md-3 mb-3'>
            <div class='metric-card'>
                <div class='metric-value'>€{total_profit_increase:,.2f}</div>
                <div class='metric-label'>Total Profit Increase</div>
            </div>
        </div>
        <div class='col-md-3 mb-3'>
            <div class='metric-card'>
                <div class='metric-value'>{avg_price_increase:.1f}%</div>
                <div class='metric-label'>Average Price Change</div>
            </div>
        </div>
        <div class='col-md-3 mb-3'>
            <div class='metric-card'>
                <div class='metric-value'>
                    {products_with_increase}/{len(df)}
                </div>
                <div class='metric-label'>Products with Price Increase</div>
            </div>
        </div>
        <div class='col-md-3 mb-3'>
            <div class='metric-card'>
                <div class='metric-value'>
                    {significant_changes}/{len(df)}
                </div>
                <div class='metric-label'>
                    Statistically Significant Changes
                </div>
            </div>
        </div>
    </div>

    <div class='mb-5'>
        {
            fig_delta.to_html(
                full_html=False,
                include_plotlyjs='cdn',
                config={'displayModeBar': False},
            )
        }
    </div>

    <div class='mb-5'>
        {
            fig_price.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={'displayModeBar': False},
            )
        }
    </div>

    <div class='mb-5'>
        {
            fig_percentage.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={'displayModeBar': False},
            )
        }
    </div>

    {pvalue_section}

    <div class='mb-3'>
        <h3>📊 Detailed Product Analysis</h3>
    </div>
    <div class='table-responsive'>
        {table_html}
    </div>
</div>

<script>
    // Add DataTables for better table interaction if desired
    document.addEventListener('DOMContentLoaded', function() {{
        // Simple table sorting on header click
        const table = document.getElementById('data-table');
        const headers = table.querySelectorAll('th');

        headers.forEach((header, index) => {{
            header.style.cursor = 'pointer';
            header.addEventListener('click', () => {{
                sortTable(index);
            }});
        }});
    }});

    function sortTable(column) {{
        // Basic table sorting implementation
        const table = document.getElementById('data-table');
        const tbody = table.tBodies[0];
        const rows = Array.from(tbody.rows);

        rows.sort((a, b) => {{
            const aText = a.cells[column].textContent;
            const bText = b.cells[column].textContent;

            const aNum = parseFloat(aText.replace(/[€%,]/g, ''));
            const bNum = parseFloat(bText.replace(/[€%,]/g, ''));

            if (!isNaN(aNum) && !isNaN(bNum)) {{
                return aNum - bNum;
            }}
            return aText.localeCompare(bText);
        }});

        rows.forEach(row => tbody.appendChild(row));
    }}
</script>
</body>
</html>
"""

    Path(out_path).write_text(html, encoding="utf-8")
    print(f"Dashboard saved to {str(out_path)}")
    print(f"Total profit increase: €{total_profit_increase:,.2f}")
    print(
        f"Products with significant changes: {significant_changes}/{len(df)}")


def main():
    """Main function to generate the dashboard."""
    df = load_data()
    build_dashboard(df)


if __name__ == "__main__":
    main()
