
from pathlib import Path
import html
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "all_tickers_clean.csv"
REPORTS = ROOT / "reports"
CHARTS = REPORTS / "charts"
OUTPUT_FILE = CHARTS / "financial_performance_dashboard.html"

CHARTS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# THEME
# ---------------------------------------------------------
BG = "#0B1220"
PANEL = "#121C2E"
TEXT = "#E5EDF8"
MUTED = "#9BAAC0"
ACCENT = "#38BDF8"

px.defaults.template = "plotly_dark"


def load_csv(path):
    """Load a CSV safely; return None if it is unavailable."""
    if not path.exists():
        print(f"Optional report not found: {path.name}")
        return None

    try:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        print(f"Loaded: {path.name} ({len(df):,} rows)")
        return df
    except Exception as exc:
        print(f"Could not load {path.name}: {exc}")
        return None


def find_col(df, candidates):
    """Find a column without case or whitespace sensitivity."""
    if df is None:
        return None

    lookup = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.strip().lower() in lookup:
            return lookup[candidate.strip().lower()]

    return None


def fmt_pct(value, decimals=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}%"


def fmt_num(value, decimals=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def style_fig(fig, height=420):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Arial"),
        height=height,
        margin=dict(l=45, r=25, t=65, b=45),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_size=12,
            font_family="Arial"
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor="#263349")
    fig.update_yaxes(showgrid=True, gridcolor="#263349")
    return fig


def figure_div(fig, include_js=False):
    return plot(
        fig,
        output_type="div",
        include_plotlyjs=True if include_js else False,
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True
        }
    )


# ---------------------------------------------------------
# LOAD EXISTING PROJECT DATA
# ---------------------------------------------------------
prices = load_csv(DATA_FILE)
risk = load_csv(REPORTS / "risk_metrics_summary.csv")
performance = load_csv(REPORTS / "performance_summary.csv")
portfolio = load_csv(REPORTS / "portfolio_summary.csv")
optimization = load_csv(REPORTS / "portfolio_optimization_summary.csv")
weights = load_csv(REPORTS / "optimized_portfolio_weights.csv")
benchmark = load_csv(REPORTS / "benchmark_metrics_summary.csv")
forecast = load_csv(REPORTS / "forecast_evaluation.csv")
forecast_backtest = load_csv(
    ROOT / "data" / "processed" / "forecast_backtest.csv"
)

if prices is None:
    raise FileNotFoundError(
        f"Required cleaned dataset not found: {DATA_FILE}"
    )

# Normalize common column names.
date_col = find_col(prices, ["Date", "Datetime"])
ticker_col = find_col(prices, ["Ticker", "Symbol"])
price_col = find_col(prices, ["Adj_Close", "Adj Close", "Close"])

if not all([date_col, ticker_col, price_col]):
    raise ValueError(
        "The cleaned price CSV must contain Date, Ticker, "
        "and Adj_Close or Close. Found: "
        f"{prices.columns.tolist()}"
    )

prices[date_col] = pd.to_datetime(prices[date_col], errors="coerce")
prices[price_col] = pd.to_numeric(prices[price_col], errors="coerce")
prices = prices.dropna(subset=[date_col, ticker_col, price_col])
prices = prices[prices[price_col] > 0]
prices = prices.sort_values([ticker_col, date_col])

# Use the latest available data date.
latest_date = prices[date_col].max()
first_date = prices[date_col].min()

tickers = sorted(prices[ticker_col].astype(str).unique().tolist())
stock_tickers = [t for t in tickers if t != "^GSPC"]

# Build aligned daily price series.
price_matrix = prices.pivot_table(
    index=date_col,
    columns=ticker_col,
    values=price_col,
    aggfunc="last"
).sort_index()

# Keep columns with usable starting and ending prices.
price_matrix = price_matrix.dropna(axis=1, how="all")
price_matrix = price_matrix.ffill()

returns = price_matrix.pct_change(fill_method=None)
returns = returns.replace([np.inf, -np.inf], np.nan)

# ---------------------------------------------------------
# CALCULATE KEY PERFORMANCE INDICATORS
# ---------------------------------------------------------
daily_obs = returns.dropna(how="all").shape[0]
trading_days = 252

cumulative_returns = (
    price_matrix.div(price_matrix.bfill().iloc[0]).iloc[-1] - 1
) * 100

daily_mean = returns.mean()
daily_std = returns.std()

annualized_return = (
    (1 + daily_mean) ** trading_days - 1
) * 100

annualized_volatility = daily_std * np.sqrt(trading_days) * 100

sharpe = (
    daily_mean / daily_std.replace(0, np.nan)
) * np.sqrt(trading_days)

drawdown_matrix = (
    price_matrix / price_matrix.cummax() - 1
) * 100
max_drawdown = drawdown_matrix.min()

# Portfolio series: equal-weighted stock returns, excluding benchmark.
valid_stocks = [t for t in stock_tickers if t in returns.columns]

if valid_stocks:
    portfolio_returns = returns[valid_stocks].mean(axis=1)
    portfolio_growth = (1 + portfolio_returns.fillna(0)).cumprod()
    portfolio_total_return = (portfolio_growth.iloc[-1] - 1) * 100
    portfolio_volatility = (
        portfolio_returns.std() * np.sqrt(trading_days) * 100
    )
    portfolio_sharpe = (
        portfolio_returns.mean()
        / portfolio_returns.std()
        * np.sqrt(trading_days)
        if portfolio_returns.std() > 0 else np.nan
    )
    portfolio_drawdown = (
        portfolio_growth / portfolio_growth.cummax() - 1
    ).min() * 100
else:
    portfolio_total_return = np.nan
    portfolio_volatility = np.nan
    portfolio_sharpe = np.nan
    portfolio_drawdown = np.nan

# Prefer the already-generated risk report for its historical metrics.
risk_ticker_col = find_col(risk, ["Ticker", "Symbol"])
risk_return_col = find_col(
    risk, ["Cumulative_Return_%", "Cumulative Return %", "Total_Return_%"]
)
risk_vol_col = find_col(
    risk, ["Annualized_Volatility_%", "Annualized Volatility %"]
)
risk_sharpe_col = find_col(risk, ["Sharpe_Ratio", "Sharpe Ratio"])
risk_dd_col = find_col(risk, ["Maximum_Drawdown_%", "Maximum Drawdown %"])

# ---------------------------------------------------------
# FIGURE 1: NORMALIZED PERFORMANCE
# ---------------------------------------------------------
normalized = price_matrix.div(price_matrix.bfill().iloc[0]) * 100

fig_performance = go.Figure()

for ticker in normalized.columns:
    fig_performance.add_trace(
        go.Scatter(
            x=normalized.index,
            y=normalized[ticker],
            mode="lines",
            name=str(ticker),
            line=dict(width=2),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "Growth of 100: %{y:.2f}<extra></extra>"
            )
        )
    )

fig_performance.update_layout(
    title="Growth of 100 — Stock and Benchmark Comparison",
    xaxis_title="Date",
    yaxis_title="Value of 100 invested",
    hovermode="x unified"
)
style_fig(fig_performance, 480)

# ---------------------------------------------------------
# FIGURE 2: RISK AND RETURN
# ---------------------------------------------------------
risk_return = pd.DataFrame({
    "Ticker": cumulative_returns.index.astype(str),
    "Cumulative Return (%)": cumulative_returns.values,
    "Annualized Volatility (%)": annualized_volatility.reindex(
        cumulative_returns.index
    ).values,
    "Sharpe Ratio": sharpe.reindex(cumulative_returns.index).values,
    "Maximum Drawdown (%)": max_drawdown.reindex(
        cumulative_returns.index
    ).values
}).replace([np.inf, -np.inf], np.nan)

fig_risk_return = px.scatter(
    risk_return,
    x="Annualized Volatility (%)",
    y="Cumulative Return (%)",
    color="Ticker",
    size="Sharpe Ratio",
    hover_data=[
        "Sharpe Ratio",
        "Maximum Drawdown (%)"
    ],
    title="Historical Risk vs Return"
)
style_fig(fig_risk_return, 420)

# ---------------------------------------------------------
# FIGURE 3: DRAWDOWN OVER TIME
# ---------------------------------------------------------
fig_drawdown = go.Figure()

for ticker in drawdown_matrix.columns:
    fig_drawdown.add_trace(
        go.Scatter(
            x=drawdown_matrix.index,
            y=drawdown_matrix[ticker],
            mode="lines",
            name=str(ticker),
            line=dict(width=1.8),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "Drawdown: %{y:.2f}%<extra></extra>"
            )
        )
    )

fig_drawdown.update_layout(
    title="Historical Drawdowns",
    xaxis_title="Date",
    yaxis_title="Drawdown (%)",
    hovermode="x unified"
)
style_fig(fig_drawdown, 420)

# ---------------------------------------------------------
# FIGURE 4: ROLLING 30-TRADING-DAY VOLATILITY
# ---------------------------------------------------------
rolling_vol = returns.rolling(30).std() * np.sqrt(trading_days) * 100

fig_vol = go.Figure()

for ticker in rolling_vol.columns:
    fig_vol.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol[ticker],
            mode="lines",
            name=str(ticker),
            line=dict(width=1.7),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "Annualized rolling volatility: %{y:.2f}%"
                "<extra></extra>"
            )
        )
    )

fig_vol.update_layout(
    title="Rolling 30-Day Annualized Volatility",
    xaxis_title="Date",
    yaxis_title="Volatility (%)",
    hovermode="x unified"
)
style_fig(fig_vol, 420)

# ---------------------------------------------------------
# FIGURE 5: PORTFOLIO OPTIMIZATION WEIGHTS
# ---------------------------------------------------------
fig_weights = go.Figure()

if weights is not None and not weights.empty:
    weight_data = weights.copy()

    # Actual columns in optimized_portfolio_weights.csv
    weight_data["Weight_%"] = pd.to_numeric(
        weight_data["Weight_%"], errors="coerce"
    )
    weight_data = weight_data.dropna(
        subset=["Strategy", "Ticker", "Weight_%"]
    )

    for strategy in weight_data["Strategy"].unique():
        subset = weight_data[weight_data["Strategy"] == strategy]

        fig_weights.add_trace(
            go.Bar(
                x=subset["Ticker"],
                y=subset["Weight_%"],
                name=strategy,
                customdata=subset[["Investment_At_10000"]],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Ticker: %{x}<br>"
                    "Portfolio weight: %{y:.2f}%<br>"
                    "Investment per $10,000: $%{customdata[0]:,.2f}"
                    "<extra></extra>"
                )
            )
        )

    fig_weights.update_layout(
        title="Optimized Portfolio Weights by Strategy",
        xaxis_title="Stock",
        yaxis_title="Portfolio Weight (%)",
        barmode="group"
    )
    fig_weights.update_yaxes(ticksuffix="%")
else:
    fig_weights.add_annotation(
        text="Portfolio weights data is unavailable.",
        x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False
    )

style_fig(fig_weights, 420)

# ---------------------------------------------------------
# FIGURE 6: FORECAST EVALUATION
# ---------------------------------------------------------
fig_forecast = go.Figure()

if forecast is not None and not forecast.empty:
    fc_data = forecast.copy()

    numeric_cols = [
        "Naive_MAE",
        "Drift_MAE",
        "Naive_RMSE",
        "Drift_RMSE",
        "MAE_Improvement_Pct",
        "RMSE_Improvement_Pct"
    ]

    for col in numeric_cols:
        if col in fc_data.columns:
            fc_data[col] = pd.to_numeric(
                fc_data[col], errors="coerce"
            )

    for metric in ["MAE", "RMSE"]:
        for model in ["Naive", "Drift"]:
            col = f"{model}_{metric}"

            if col in fc_data.columns:
                fig_forecast.add_trace(
                    go.Bar(
                        x=fc_data["Ticker"],
                        y=fc_data[col],
                        name=f"{model} {metric}",
                        hovertemplate=(
                            "Ticker: %{x}<br>"
                            f"{metric}: %{{y:,.2f}} price units"
                            "<extra></extra>"
                        )
                    )
                )

    fig_forecast.update_layout(
        title="Naive vs Drift Forecast Errors",
        xaxis_title="Stock",
        yaxis_title="Error (price units)",
        barmode="group"
    )

    fig_forecast.update_xaxes(
        categoryorder="array",
        categoryarray=fc_data["Ticker"].tolist()
    )
else:
    fig_forecast.add_annotation(
        text="Forecast evaluation data is unavailable.",
        x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False
    )

style_fig(fig_forecast, 420)


# ---------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------
def metric_card(label, value, detail=""):
    return f"""
    <div class="kpi">
      <div class="kpi-label">{html.escape(label)}</div>
      <div class="kpi-value">{html.escape(str(value))}</div>
      <div class="kpi-detail">{html.escape(str(detail))}</div>
    </div>
    """


kpis = [
    metric_card(
        "Stocks Analyzed",
        len(stock_tickers),
        "Excludes the S&P 500 benchmark"
    ),
    metric_card(
        "Portfolio Total Return",
        fmt_pct(portfolio_total_return),
        "Equal-weighted stock daily returns"
    ),
    metric_card(
        "Portfolio Annualized Volatility",
        fmt_pct(portfolio_volatility),
        "Based on daily returns and 252 trading days"
    ),
    metric_card(
        "Portfolio Sharpe Ratio",
        fmt_num(portfolio_sharpe, 3),
        "Assumes zero risk-free rate"
    ),
    metric_card(
        "Portfolio Maximum Drawdown",
        fmt_pct(portfolio_drawdown),
        "Largest historical peak-to-trough decline"
    ),
    metric_card(
        "Latest Data Date",
        latest_date.strftime("%d %b %Y"),
        f"Data starts {first_date.strftime('%d %b %Y')}"
    )
]

# ---------------------------------------------------------
# ASSEMBLE ONE INTERACTIVE HTML DASHBOARD
# ---------------------------------------------------------
divs = [
    figure_div(fig_performance, include_js=True),
    figure_div(fig_risk_return),
    figure_div(fig_drawdown),
    figure_div(fig_vol),
    figure_div(fig_weights),
    figure_div(fig_forecast)
]

sections = [
    ("Performance Comparison", divs[0]),
    ("Risk vs Return", divs[1]),
    ("Drawdown Analysis", divs[2]),
    ("Rolling Volatility", divs[3]),
    ("Portfolio Optimization", divs[4]),
    ("Forecast Evaluation", divs[5])
]

section_html = "\n".join(
    f'<section class="chart-panel"><h2>{title}</h2>{content}</section>'
    for title, content in sections
)

html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Financial Performance Analytics Dashboard</title>
<style>
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: {BG};
  color: {TEXT};
  font-family: Arial, sans-serif;
}}
header {{
  padding: 30px 4%;
  background: linear-gradient(120deg, #14233B, #0B1220);
  border-bottom: 1px solid #263349;
}}
header h1 {{ margin: 0 0 8px; font-size: 30px; }}
header p {{ color: {MUTED}; margin: 0; line-height: 1.6; }}
main {{ width: 94%; max-width: 1600px; margin: 24px auto; }}
.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 14px;
  margin-bottom: 24px;
}}
.kpi {{
  background: {PANEL};
  border: 1px solid #263349;
  border-radius: 12px;
  padding: 20px;
  min-height: 125px;
}}
.kpi-label {{ color: {MUTED}; font-size: 13px; }}
.kpi-value {{ color: {ACCENT}; font-size: 27px; font-weight: 700; margin: 12px 0 8px; }}
.kpi-detail {{ color: {MUTED}; font-size: 11px; line-height: 1.5; }}
.chart-panel {{
  background: {PANEL};
  border: 1px solid #263349;
  border-radius: 12px;
  margin: 18px 0;
  padding: 14px;
  overflow: hidden;
}}
.chart-panel h2 {{ font-size: 18px; margin: 8px 12px 0; }}
footer {{
  padding: 20px 4%;
  color: {MUTED};
  font-size: 12px;
  border-top: 1px solid #263349;
  line-height: 1.7;
}}
@media (max-width: 600px) {{
  header h1 {{ font-size: 23px; }}
  main {{ width: 98%; }}
  .chart-panel {{ padding: 4px; }}
}}
</style>
</head>
<body>
<header>
  <h1>Financial Performance Analytics</h1>
  <p>Interactive stock performance, portfolio risk, optimization and forecasting</p>
  <p>Historical data: {first_date:%d %b %Y} – {latest_date:%d %b %Y}</p>
</header>
<main>
  <div class="kpi-grid">
    {''.join(kpis)}
  </div>
  {section_html}
</main>
<footer>
  <strong>Methodology:</strong> Historical results are descriptive, not predictions.
  Annualization uses 252 trading days. The dashboard's equal-weighted portfolio
  is calculated from daily stock returns and is not necessarily identical to
  the optimized portfolio from the optimization script. Forecast errors are
  shown in the price units used by the source report. Historical results do
  not guarantee future performance.
</footer>
</body>
</html>
"""

OUTPUT_FILE.write_text(html_page, encoding="utf-8")

print("\n" + "=" * 60)
print("FINANCIAL PERFORMANCE DASHBOARD CREATED")
print("=" * 60)
print(f"Dashboard: {OUTPUT_FILE}")
print(f"Stocks: {len(stock_tickers)}")
print(f"Data range: {first_date.date()} to {latest_date.date()}")
print(f"Price observations: {len(prices):,}")
print("\nOpen the HTML file in your browser to explore the dashboard.")
