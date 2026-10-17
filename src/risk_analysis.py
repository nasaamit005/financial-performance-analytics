
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "all_tickers_clean.csv"
CHART_DIR = ROOT / "reports" / "charts"
REPORT_DIR = ROOT / "reports"

CHART_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.0  # Assumption: 0% annual risk-free rate


# --------------------------------------------------
# 2. LOAD AND IDENTIFY COLUMNS
# --------------------------------------------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}\n"
        "Check that all_tickers_clean.csv exists in data/processed."
    )

df = pd.read_csv(INPUT_FILE)


def find_column(possible_names):
    """Find a column while ignoring spaces, underscores and case."""
    normalize = lambda value: "".join(
        char.lower() for char in str(value) if char.isalnum()
    )

    lookup = {normalize(col): col for col in df.columns}

    for name in possible_names:
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
        "Could not identify required columns.\n"
        f"Found columns: {df.columns.tolist()}\n"
        "Required: Date, Ticker and Adj Close (or Close)."
    )

df = df.rename(
    columns={
        date_col: "Date",
        ticker_col: "Ticker",
        price_col: "Price",
    }
)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Ticker"] = df["Ticker"].astype(str).str.strip()

df = (
    df.dropna(subset=["Date", "Ticker", "Price"])
    .query("Price > 0")
    .sort_values(["Ticker", "Date"])
    .drop_duplicates(["Ticker", "Date"], keep="last")
    .reset_index(drop=True)
)

if df.empty:
    raise ValueError("No valid price data remains after cleaning.")


# --------------------------------------------------
# 3. CALCULATE RISK METRICS PER TICKER
# --------------------------------------------------
frames = []
summary = []

for ticker, group in df.groupby("Ticker", sort=True):
    group = group.sort_values("Date").copy()

    # Daily percentage returns
    group["Daily_Return"] = group["Price"].pct_change()
    group["Growth_Base_100"] = (
        group["Price"] / group["Price"].iloc[0] * 100
    )

    # Rolling 21-trading-day annualized volatility
    group["Rolling_Volatility"] = (
        group["Daily_Return"]
        .rolling(21, min_periods=15)
        .std()
        * np.sqrt(TRADING_DAYS)
        * 100
    )

    # Drawdown from the previous running price peak
    running_peak = group["Price"].cummax()
    group["Drawdown"] = (group["Price"] / running_peak - 1) * 100

    returns = group["Daily_Return"].dropna()

    if len(returns) < 2:
        print(f"Skipping {ticker}: insufficient return observations.")
        continue

    daily_volatility = returns.std()
    annual_volatility = daily_volatility * np.sqrt(TRADING_DAYS)

    # Geometric annualized return
    elapsed_years = (
        (group["Date"].iloc[-1] - group["Date"].iloc[0]).days / 365.25
    )

    if elapsed_years > 0:
        total_growth = group["Price"].iloc[-1] / group["Price"].iloc[0]
        annual_return = (
            total_growth ** (1 / elapsed_years) - 1
        )
    else:
        annual_return = np.nan

    # Historical daily 95% VaR and CVaR, expressed as loss percentages
    fifth_percentile = returns.quantile(0.05)
    var_95 = -fifth_percentile * 100
    cvar_95 = -returns[returns <= fifth_percentile].mean() * 100

    # Sharpe ratio using the stated annual risk-free-rate assumption
    daily_rf = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
    excess_returns = returns - daily_rf

    sharpe = (
        excess_returns.mean() / daily_volatility * np.sqrt(TRADING_DAYS)
        if daily_volatility > 0
        else np.nan
    )

    frames.append(group)

    summary.append(
        {
            "Ticker": ticker,
            "Start_Date": group["Date"].iloc[0].date(),
            "End_Date": group["Date"].iloc[-1].date(),
            "Observations": len(returns),
            "Cumulative_Return_%": (
                group["Price"].iloc[-1] / group["Price"].iloc[0] - 1
            ) * 100,
            "Annualized_Return_%": annual_return * 100,
            "Annualized_Volatility_%": annual_volatility * 100,
            "Sharpe_Ratio": sharpe,
            "Maximum_Drawdown_%": group["Drawdown"].min(),
            "Historical_VaR_95_%": var_95,
            "Historical_CVaR_95_%": cvar_95,
        }
    )

if not frames:
    raise ValueError("Not enough valid data to calculate risk metrics.")

analysis = pd.concat(frames, ignore_index=True)
metrics = pd.DataFrame(summary)

metrics = metrics.sort_values(
    "Annualized_Volatility_%", ascending=False
)

metrics_file = REPORT_DIR / "risk_metrics_summary.csv"
metrics.to_csv(metrics_file, index=False)


# --------------------------------------------------
# 4. BUILD A PROFESSIONAL INTERACTIVE DASHBOARD
# --------------------------------------------------
tickers = sorted(analysis["Ticker"].unique())

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Growth of 100 — Adjusted Price",
        "Rolling 21-Day Annualized Volatility",
        "Drawdown from Previous Peak",
        "Daily Return Distribution",
    ),
    vertical_spacing=0.13,
    horizontal_spacing=0.09,
)

for ticker in tickers:
    data = analysis[analysis["Ticker"] == ticker]

    # View 1: normalized performance
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Growth_Base_100"],
            name=ticker,
            legendgroup=ticker,
            mode="lines",
            line=dict(width=2),
            customdata=np.column_stack([data["Price"]]),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Growth index: %{y:.2f}<br>"
                "Price: %{customdata[0]:.2f}"
                "<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

    # View 2: changing volatility
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Rolling_Volatility"],
            name=ticker,
            legendgroup=ticker,
            showlegend=False,
            mode="lines",
            line=dict(width=1.8),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Annualized volatility: %{y:.2f}%"
                "<extra></extra>"
            ),
        ),
        row=1, col=2,
    )

    # View 3: underwater drawdown chart
    fig.add_trace(
        go.Scatter(
            x=data["Date"],
            y=data["Drawdown"],
            name=ticker,
            legendgroup=ticker,
            showlegend=False,
            mode="lines",
            line=dict(width=1.5),
            fill="tozeroy",
            opacity=0.55,
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Drawdown: %{y:.2f}%"
                "<extra></extra>"
            ),
        ),
        row=2, col=1,
    )

    # View 4: daily return distribution
    daily_returns = data["Daily_Return"].dropna() * 100

    fig.add_trace(
        go.Histogram(
            x=daily_returns,
            name=ticker,
            legendgroup=ticker,
            showlegend=False,
            opacity=0.45,
            nbinsx=70,
            hovertemplate=(
                f"<b>{ticker}</b><br>"
                "Daily return range: %{x:.2f}%<br>"
                "Trading days: %{y}"
                "<extra></extra>"
            ),
        ),
        row=2, col=2,
    )

fig.update_layout(
    title=dict(
        text=(
            "FINANCIAL RISK INTELLIGENCE"
            "<br><sup>Performance • Volatility • Drawdown • Return Distribution</sup>"
        ),
        x=0.03,
        xanchor="left",
    ),
    template="plotly_dark",
    height=950,
    hovermode="x unified",
    barmode="overlay",
    legend=dict(
        title="Click a ticker to isolate it",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    margin=dict(l=65, r=35, t=130, b=75),
)

fig.update_xaxes(showgrid=True, gridcolor="rgba(150,150,150,0.15)")
fig.update_yaxes(showgrid=True, gridcolor="rgba(150,150,150,0.15)")

fig.update_yaxes(title_text="Growth index (start = 100)", row=1, col=1)
fig.update_yaxes(title_text="Annualized volatility (%)", row=1, col=2)
fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
fig.update_xaxes(title_text="Daily return (%)", row=2, col=2)
fig.update_yaxes(title_text="Number of trading days", row=2, col=2)

dashboard_file = CHART_DIR / "risk_analysis_dashboard.html"

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

print("\n" + "=" * 60)
print("RISK ANALYSIS COMPLETED")
print("=" * 60)
print(f"Interactive dashboard: {dashboard_file}")
print(f"Risk metrics CSV:      {metrics_file}")
print("\nRisk metrics ranked by annualized volatility:")
print(metrics.round(3).to_string(index=False))
print("\nOpen the HTML dashboard in your browser to explore the charts.")
