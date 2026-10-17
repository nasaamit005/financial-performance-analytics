
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

daily_file = REPORTS / "portfolio_daily_returns.csv"
summary_file = REPORTS / "portfolio_summary.csv"

if not daily_file.exists() or not summary_file.exists():
    raise FileNotFoundError(
        "Run src/portfolio_analysis.py first to generate the CSV files."
    )

daily = pd.read_csv(daily_file, parse_dates=["Date"])
summary = pd.read_csv(summary_file)

fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=(
        "Portfolio Growth: Starting Value ₹10,000",
        "Risk and Return Comparison",
    ),
    vertical_spacing=0.16,
    specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
)

# Chart 1: Growth of the portfolio and benchmark.
fig.add_trace(
    go.Scatter(
        x=daily["Date"],
        y=daily["Portfolio_Growth_10000"],
        mode="lines",
        name="Equal-Weighted Stocks",
        line=dict(width=3),
        hovertemplate="Date: %{x|%d %b %Y}<br>Value: ₹%{y:,.2f}<extra></extra>",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=daily["Date"],
        y=daily["Benchmark_Growth_10000"],
        mode="lines",
        name="S&P 500",
        line=dict(width=3, dash="dash"),
        hovertemplate="Date: %{x|%d %b %Y}<br>Value: ₹%{y:,.2f}<extra></extra>",
    ),
    row=1,
    col=1,
)

# Chart 2: Annualized return versus volatility.
for _, row in summary.iterrows():
    fig.add_trace(
        go.Bar(
            x=[row["Portfolio"]],
            y=[row["CAGR_Pct"]],
            name=f'{row["Portfolio"]} CAGR',
            text=[f'{row["CAGR_Pct"]:.2f}%'],
            textposition="outside",
            hovertemplate=(
                "Portfolio: %{x}<br>"
                "CAGR: %{y:.2f}%<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=[row["Portfolio"]],
            y=[row["Annualized_Volatility_Pct"]],
            name=f'{row["Portfolio"]} Volatility',
            mode="markers",
            marker=dict(size=13, symbol="diamond"),
            hovertemplate=(
                "Portfolio: %{x}<br>"
                "Annualized volatility: %{y:.2f}%<extra></extra>"
            ),
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

fig.update_layout(
    title="Financial Performance Analytics | Portfolio vs S&P 500",
    template="plotly_white",
    height=850,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=70, r=70, t=120, b=80),
)

fig.update_yaxes(
    title_text="Portfolio Value (₹)",
    tickprefix="₹",
    tickformat=",.0f",
    row=1,
    col=1,
)

fig.update_yaxes(
    title_text="CAGR (%)",
    ticksuffix="%",
    row=2,
    col=1,
    secondary_y=False,
)

fig.update_yaxes(
    title_text="Annualized Volatility (%)",
    ticksuffix="%",
    row=2,
    col=1,
    secondary_y=True,
)

fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_xaxes(title_text="Portfolio", row=2, col=1)

output_file = REPORTS / "portfolio_comparison.html"
fig.write_html(output_file, include_plotlyjs=True, auto_open=True)

print(f"Interactive chart saved to: {output_file}")
