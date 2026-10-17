
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ==================================================
# 1. PATHS AND SETTINGS
# ==================================================
ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "all_tickers_clean.csv"
CHART_DIR = ROOT / "reports" / "charts"
REPORT_DIR = ROOT / "reports"

CHART_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK = "^GSPC"
TRADING_DAYS = 252
ROLLING_WINDOW = 63
RISK_FREE_RATE = 0.0  # Annual assumption; replace if appropriate


# ==================================================
# 2. LOAD DATA
# ==================================================
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)


def normalize(value):
    return "".join(
        char.lower() for char in str(value) if char.isalnum()
    )


def find_column(names):
    lookup = {normalize(col): col for col in df.columns}
    for name in names:
        if normalize(name) in lookup:
            return lookup[normalize(name)]
    return None


date_col = find_column(["Date", "Datetime"])
ticker_col = find_column(["Ticker", "Symbol"])
price_col = find_column(
    ["Adj Close", "Adjusted Close", "Adj_Close", "Close"]
)

if not all([date_col, ticker_col, price_col]):
    raise ValueError(
        "Required columns not found. Dataset columns: "
        f"{df.columns.tolist()}"
    )

df = df.rename(
    columns={
        date_col: "Date",
        ticker_col: "Ticker",
        price_col: "Price",
    }
)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Ticker"] = df["Ticker"].astype(str).str.strip()
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

df = (
    df.dropna(subset=["Date", "Ticker", "Price"])
    .query("Price > 0")
    .sort_values(["Ticker", "Date"])
    .drop_duplicates(["Ticker", "Date"], keep="last")
)

if BENCHMARK not in set(df["Ticker"]):
    raise ValueError(
        f"Benchmark {BENCHMARK} was not found. "
        f"Tickers in data: {sorted(df['Ticker'].unique())}"
    )


# ==================================================
# 3. CREATE DAILY RETURNS AND ALIGN DATES
# ==================================================
price_wide = df.pivot(
    index="Date", columns="Ticker", values="Price"
).sort_index()

returns_wide = price_wide.pct_change(fill_method=None)

if BENCHMARK not in returns_wide.columns:
    raise ValueError("Benchmark returns could not be calculated.")

# Keep benchmark and stock returns on the same dates.
valid_market = returns_wide[BENCHMARK].notna()
returns_wide = returns_wide.loc[valid_market].copy()

stocks = sorted(
    ticker
    for ticker in returns_wide.columns
    if ticker != BENCHMARK
)

if not stocks:
    raise ValueError("No stocks were found apart from the benchmark.")

# Annual risk-free rate converted to a daily rate.
daily_rf = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
market_excess = returns_wide[BENCHMARK] - daily_rf

summary_rows = []
rolling_rows = []
relative_rows = []

for ticker in stocks:
    paired = pd.concat(
        [returns_wide[ticker], returns_wide[BENCHMARK]],
        axis=1,
        keys=["Stock", "Market"],
    ).dropna()

    if len(paired) < 2:
        print(f"Skipping {ticker}: insufficient overlapping data.")
        continue

    stock_returns = paired["Stock"]
    market_returns = paired["Market"]

    stock_excess = stock_returns - daily_rf
    market_excess_pair = market_returns - daily_rf

    market_variance = market_excess_pair.var(ddof=1)

    if market_variance > 0:
        beta = (
            stock_excess.cov(market_excess_pair) / market_variance
        )
        daily_alpha = (
            stock_excess.mean()
            - beta * market_excess_pair.mean()
        )
        annual_alpha = daily_alpha * TRADING_DAYS * 100
    else:
        beta = np.nan
        annual_alpha = np.nan

    elapsed_years = (
        (paired.index[-1] - paired.index[0]).days / 365.25
    )

    total_growth = (1 + stock_returns).prod()
    annual_return = (
        total_growth ** (1 / elapsed_years) - 1
        if elapsed_years > 0 and total_growth > 0
        else np.nan
    )

    annual_volatility = (
        stock_returns.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100
    )

    sharpe = (
        stock_excess.mean() / stock_returns.std(ddof=1)
        * np.sqrt(TRADING_DAYS)
        if stock_returns.std(ddof=1) > 0
        else np.nan
    )

    market_total = (1 + market_returns).prod()
    market_annual_return = (
        market_total ** (1 / elapsed_years) - 1
        if elapsed_years > 0 and market_total > 0
        else np.nan
    )

    summary_rows.append(
        {
            "Ticker": ticker,
            "Benchmark": BENCHMARK,
            "Start_Date": paired.index[0].date(),
            "End_Date": paired.index[-1].date(),
            "Overlapping_Trading_Days": len(paired),
            "Cumulative_Return_%": (total_growth - 1) * 100,
            "Annualized_Return_%": annual_return * 100,
            "Benchmark_Annualized_Return_%":
                market_annual_return * 100,
            "Annualized_Excess_Return_pp":
                (annual_return - market_annual_return) * 100,
            "Annualized_Volatility_%": annual_volatility,
            "Beta": beta,
            "Annualized_Regression_Alpha_%": annual_alpha,
            "Sharpe_Ratio": sharpe,
        }
    )

    # Rolling beta based on trailing 63 paired daily observations.
    rolling_cov = (
        stock_excess.rolling(
            ROLLING_WINDOW, min_periods=40
        ).cov(market_excess_pair)
    )
    rolling_var = market_excess_pair.rolling(
        ROLLING_WINDOW, min_periods=40
    ).var()

    rolling_beta = rolling_cov / rolling_var

    for date, value in rolling_beta.items():
        rolling_rows.append(
            {"Date": date, "Ticker": ticker, "Rolling_Beta": value}
        )

    # Cumulative growth from the first shared return date.
    stock_growth = (1 + stock_returns).cumprod() * 100
    market_growth = (1 + market_returns).cumprod() * 100

    for date, value in stock_growth.items():
        relative_rows.append(
            {
                "Date": date,
                "Ticker": ticker,
                "Growth_Index": value,
            }
        )

    # Add the benchmark once for each date.
    if ticker == stocks[0]:
        for date, value in market_growth.items():
            relative_rows.append(
                {
                    "Date": date,
                    "Ticker": BENCHMARK,
                    "Growth_Index": value,
                }
            )


if not summary_rows:
    raise ValueError("No valid stock-versus-benchmark results.")

metrics = pd.DataFrame(summary_rows).sort_values(
    "Annualized_Excess_Return_pp", ascending=False
)

rolling_df = pd.DataFrame(rolling_rows)
relative_df = pd.DataFrame(relative_rows)

metrics.to_csv(
    REPORT_DIR / "benchmark_metrics_summary.csv", index=False
)
rolling_df.to_csv(
    REPORT_DIR / "rolling_beta_data.csv", index=False
)
relative_df.to_csv(
    REPORT_DIR / "relative_performance_data.csv", index=False
)


# ==================================================
# 4. BUILD THE INTERACTIVE DASHBOARD
# ==================================================
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Growth of 100 — Stocks vs S&P 500",
        "Annualized Return vs Annualized Volatility",
        "Beta vs Annualized Excess Return",
        "Rolling 63-Day Beta",
    ),
    horizontal_spacing=0.12,
    vertical_spacing=0.15,
)

# Chart 1: relative cumulative performance.
for ticker in [BENCHMARK] + stocks:
    data = relative_df[relative_df["Ticker"] == ticker]
    if data.empty:
        continue

    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Growth_Index"],
            name=ticker,
            legendgroup=ticker,
            mode="lines",
            line=dict(width=2.5 if ticker == BENCHMARK else 1.8),
            hovertemplate=(
                f"<b>{ticker}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Growth index: %{y:.2f}"
                "<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

# Chart 2: annualized return vs volatility.
for _, row in metrics.iterrows():
    fig.add_trace(
        go.Scatter(
            x=[row["Annualized_Volatility_%"]],
            y=[row["Annualized_Return_%"]],
            name=row["Ticker"],
            legendgroup=row["Ticker"],
            showlegend=False,
            mode="markers+text",
            text=[row["Ticker"]],
            textposition="top center",
            marker=dict(
                size=15,
                line=dict(width=1, color="white"),
            ),
            customdata=[[
                row["Beta"],
                row["Annualized_Excess_Return_pp"],
                row["Sharpe_Ratio"],
            ]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Annualized volatility: %{x:.2f}%<br>"
                "Annualized return: %{y:.2f}%<br>"
                "Beta: %{customdata[0]:.3f}<br>"
                "Excess return: %{customdata[1]:.2f} pp<br>"
                "Sharpe ratio: %{customdata[2]:.3f}"
                "<extra></extra>"
            ),
        ),
        row=1, col=2,
    )

# Chart 3: beta vs excess return.
for _, row in metrics.iterrows():
    fig.add_trace(
        go.Scatter(
            x=[row["Beta"]],
            y=[row["Annualized_Excess_Return_pp"]],
            name=row["Ticker"],
            legendgroup=row["Ticker"],
            showlegend=False,
            mode="markers+text",
            text=[row["Ticker"]],
            textposition="top center",
            marker=dict(
                size=15,
                line=dict(width=1, color="white"),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Beta: %{x:.3f}<br>"
                "Annualized excess return: %{y:.2f} percentage points"
                "<extra></extra>"
            ),
        ),
        row=2, col=1,
    )

# Chart 4: rolling beta.
for ticker in stocks:
    data = rolling_df[rolling_df["Ticker"] == ticker]

    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Rolling_Beta"],
            name=ticker,
            legendgroup=ticker,
            showlegend=False,
            mode="lines",
            line=dict(width=1.8),
            hovertemplate=(
                f"<b>{ticker}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Rolling beta: %{y:.3f}"
                "<extra></extra>"
            ),
        ),
        row=2, col=2,
    )

fig.update_layout(
    title=dict(
        text=(
            "MARKET BENCHMARK INTELLIGENCE"
            "<br><sup>Relative Performance • Beta • Alpha • Rolling Sensitivity</sup>"
        ),
        x=0.03,
        xanchor="left",
    ),
    template="plotly_dark",
    height=1000,
    hovermode="closest",
    legend=dict(
        title="Click to show/hide series",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    margin=dict(l=70, r=35, t=135, b=75),
)

fig.update_xaxes(showgrid=True, gridcolor="rgba(150,150,150,0.15)")
fig.update_yaxes(showgrid=True, gridcolor="rgba(150,150,150,0.15)")

fig.update_yaxes(title_text="Growth index (start = 100)", row=1, col=1)
fig.update_xaxes(title_text="Annualized volatility (%)", row=1, col=2)
fig.update_yaxes(title_text="Annualized return (%)", row=1, col=2)
fig.update_xaxes(title_text="Beta", row=2, col=1)
fig.update_yaxes(
    title_text="Annualized excess return (percentage points)",
    row=2, col=1,
)
fig.update_xaxes(title_text="Date", row=2, col=2)
fig.update_yaxes(title_text="Rolling beta", row=2, col=2)

dashboard_file = CHART_DIR / "benchmark_analysis_dashboard.html"

fig.write_html(
    dashboard_file,
    include_plotlyjs=True,
    full_html=True,
    config={
        "responsive": True,
        "displaylogo": False,
        "scrollZoom": True,
    },
)

print("\n" + "=" * 65)
print("BENCHMARK ANALYSIS COMPLETED")
print("=" * 65)
print(f"Dashboard: {dashboard_file}")
print(f"Metrics:   {REPORT_DIR / 'benchmark_metrics_summary.csv'}")
print("\nResults ranked by annualized excess return:")
print(metrics.round(3).to_string(index=False))
print(
    "\nNote: Alpha is the annualized daily regression intercept, "
    "not a guaranteed future return."
)
