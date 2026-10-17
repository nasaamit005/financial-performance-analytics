
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize

# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "all_tickers_clean.csv"
REPORT_DIR = ROOT / "reports"
CHART_DIR = REPORT_DIR / "charts"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK = "^GSPC"
TRADING_DAYS = 252
RISK_FREE_RATE = 0.0
NUM_SIMULATIONS = 5000
RANDOM_SEED = 42
INITIAL_INVESTMENT = 10000.0

# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

# Normalize headers without assuming their capitalization.
df.columns = [
    str(column).strip().lower().replace(" ", "_")
    for column in df.columns
]

required_columns = {"date", "ticker"}
missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {missing}. "
        f"Available columns: {df.columns.tolist()}"
    )

if "adj_close" in df.columns:
    PRICE_COLUMN = "adj_close"
elif "close" in df.columns:
    PRICE_COLUMN = "close"
else:
    raise ValueError("The CSV must contain Adj_Close or Close.")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
df[PRICE_COLUMN] = pd.to_numeric(
    df[PRICE_COLUMN], errors="coerce"
)

df = df.dropna(subset=["date", "ticker", PRICE_COLUMN])
df = df[df[PRICE_COLUMN] > 0]
df = df.drop_duplicates(
    subset=["date", "ticker"], keep="last"
)

prices = (
    df.pivot(index="date", columns="ticker", values=PRICE_COLUMN)
    .sort_index()
)

if BENCHMARK not in prices.columns:
    raise ValueError(
        f"Benchmark {BENCHMARK} not found. "
        f"Available tickers: {prices.columns.tolist()}"
    )

stocks = [ticker for ticker in prices.columns if ticker != BENCHMARK]

if len(stocks) < 2:
    raise ValueError("At least two individual stocks are required.")

# Use the same dates for a fair comparison.
prices = prices[[*stocks, BENCHMARK]].dropna()

if len(prices) < 100:
    raise ValueError("Insufficient complete price observations.")

returns = prices.pct_change(fill_method=None).dropna()
stock_returns = returns[stocks]
benchmark_returns = returns[BENCHMARK]

if stock_returns.empty:
    raise ValueError("Daily stock returns could not be calculated.")

print("=" * 70)
print("PORTFOLIO OPTIMIZATION")
print("=" * 70)
print(f"Price column: {PRICE_COLUMN}")
print(f"Period: {prices.index.min().date()} to {prices.index.max().date()}")
print(f"Observations: {len(stock_returns)}")
print(f"Stocks: {', '.join(stocks)}")

# ============================================================
# 3. PORTFOLIO MATHEMATICS
# ============================================================

mean_annual = stock_returns.mean() * TRADING_DAYS
cov_annual = stock_returns.cov() * TRADING_DAYS
n_stocks = len(stocks)

def expected_return(weights):
    return float(np.dot(weights, mean_annual.values))

def annual_volatility(weights):
    variance = float(
        weights @ cov_annual.values @ weights
    )
    return float(np.sqrt(max(variance, 0.0)))

def sharpe_ratio(weights):
    volatility = annual_volatility(weights)
    if volatility <= 1e-12:
        return -1e6
    return (expected_return(weights) - RISK_FREE_RATE) / volatility

def max_drawdown(daily_returns):
    growth = (1 + daily_returns).cumprod()
    drawdown = growth / growth.cummax() - 1
    return float(drawdown.min())

equal_weights = np.repeat(1.0 / n_stocks, n_stocks)
bounds = [(0.0, 1.0)] * n_stocks
constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

# ============================================================
# 4. OPTIMIZE MINIMUM VOLATILITY AND MAXIMUM SHARPE
# ============================================================

min_vol_result = minimize(
    annual_volatility,
    equal_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 3000, "ftol": 1e-12},
)

max_sharpe_result = minimize(
    lambda w: -sharpe_ratio(w),
    equal_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 3000, "ftol": 1e-12},
)

if not min_vol_result.success:
    raise RuntimeError(
        f"Minimum volatility optimization failed: "
        f"{min_vol_result.message}"
    )

if not max_sharpe_result.success:
    raise RuntimeError(
        f"Maximum Sharpe optimization failed: "
        f"{max_sharpe_result.message}"
    )

strategies = {
    "Equal Weighted": equal_weights,
    "Minimum Volatility": min_vol_result.x,
    "Maximum Sharpe": max_sharpe_result.x,
}

# ============================================================
# 5. SAVE ASSET ALLOCATIONS
# ============================================================

allocation_rows = []

for strategy, weights in strategies.items():
    for ticker, weight in zip(stocks, weights):
        allocation_rows.append({
            "Strategy": strategy,
            "Ticker": ticker,
            "Weight_%": weight * 100,
            "Investment_At_10000": weight * INITIAL_INVESTMENT,
        })

allocations = pd.DataFrame(allocation_rows)

allocations.to_csv(
    REPORT_DIR / "optimized_portfolio_weights.csv",
    index=False,
)

# ============================================================
# 6. STRATEGY PERFORMANCE
# ============================================================

strategy_returns = pd.DataFrame(index=stock_returns.index)

for strategy, weights in strategies.items():
    strategy_returns[strategy] = stock_returns @ weights

strategy_returns[BENCHMARK] = benchmark_returns

summary_rows = []

for strategy in strategy_returns.columns:
    daily = strategy_returns[strategy].dropna()
    growth = (1 + daily).cumprod()
    years = len(daily) / TRADING_DAYS

    total_return = (growth.iloc[-1] - 1) * 100
    cagr = (
        (growth.iloc[-1] ** (1 / years) - 1) * 100
        if years > 0 and growth.iloc[-1] > 0
        else np.nan
    )
    volatility = daily.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (
        (daily.mean() * TRADING_DAYS - RISK_FREE_RATE) / volatility
        if volatility > 0
        else np.nan
    )

    summary_rows.append({
        "Strategy": strategy,
        "Total_Return_%": total_return,
        "CAGR_%": cagr,
        "Annualized_Volatility_%": volatility * 100,
        "Sharpe_Ratio": sharpe,
        "Maximum_Drawdown_%": max_drawdown(daily) * 100,
        "Observations": len(daily),
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    REPORT_DIR / "portfolio_optimization_summary.csv",
    index=False,
)

strategy_returns.to_csv(
    REPORT_DIR / "optimized_strategy_daily_returns.csv",
    index_label="Date",
)

# ============================================================
# 7. SIMULATE PORTFOLIOS FOR THE EFFICIENT FRONTIER
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

random_weights = rng.dirichlet(
    np.ones(n_stocks),
    size=NUM_SIMULATIONS,
)

frontier_rows = []

for weights in random_weights:
    ret = expected_return(weights)
    vol = annual_volatility(weights)
    sr = (
        (ret - RISK_FREE_RATE) / vol
        if vol > 1e-12
        else np.nan
    )

    frontier_rows.append({
        "Annualized_Return_%": ret * 100,
        "Annualized_Volatility_%": vol * 100,
        "Sharpe_Ratio": sr,
    })

frontier = pd.DataFrame(frontier_rows)

frontier.to_csv(
    REPORT_DIR / "efficient_frontier.csv",
    index=False,
)

# ============================================================
# 8. HISTORICAL GROWTH OF $10,000 IN EACH STOCK
# ============================================================

growth_index = prices[stocks].div(prices[stocks].iloc[0])
investment_values = growth_index * INITIAL_INVESTMENT

investment_rows = []

for ticker in stocks:
    final_value = investment_values[ticker].iloc[-1]

    investment_rows.append({
        "Ticker": ticker,
        "Initial_Investment": INITIAL_INVESTMENT,
        "Final_Investment_Value": final_value,
        "Profit_Loss": final_value - INITIAL_INVESTMENT,
        "Total_Return_%": (
            final_value / INITIAL_INVESTMENT - 1
        ) * 100,
    })

investment_summary = pd.DataFrame(investment_rows).sort_values(
    "Total_Return_%", ascending=False
)

investment_summary.to_csv(
    REPORT_DIR / "individual_stock_investment_summary.csv",
    index=False,
)

# ============================================================
# 9. INTERACTIVE PLOTLY DASHBOARD
# ============================================================

fig = make_subplots(
    rows=3,
    cols=2,
    subplot_titles=[
        "$10,000 Stock Investment Growth",
        "Efficient Frontier",
        "Portfolio Allocation by Strategy",
        "Cumulative Portfolio Performance",
        "Individual Stock Returns",
        "Strategy Risk vs Return",
    ],
    vertical_spacing=0.11,
    horizontal_spacing=0.10,
)

# Chart 1: investment growth.
for ticker in stocks:
    fig.add_trace(
        go.Scatter(
            x=investment_values.index,
            y=investment_values[ticker],
            name=ticker,
            mode="lines",
            hovertemplate=(
                f"<b>{ticker}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Value: $%{y:,.2f}<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

# Chart 2: simulated efficient frontier.
fig.add_trace(
    go.Scatter(
        x=frontier["Annualized_Volatility_%"],
        y=frontier["Annualized_Return_%"],
        mode="markers",
        name="Simulated portfolios",
        marker=dict(
            size=5,
            color=frontier["Sharpe_Ratio"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Sharpe ratio"),
            opacity=0.7,
        ),
        customdata=frontier[["Sharpe_Ratio"]].values,
        hovertemplate=(
            "Volatility: %{x:.2f}%<br>"
            "Return: %{y:.2f}%<br>"
            "Sharpe: %{customdata[0]:.3f}<extra></extra>"
        ),
    ),
    row=1, col=2,
)

for strategy, weights in strategies.items():
    fig.add_trace(
        go.Scatter(
            x=[annual_volatility(weights) * 100],
            y=[expected_return(weights) * 100],
            name=strategy,
            mode="markers+text",
            text=[strategy],
            textposition="top center",
            marker=dict(size=12, symbol="diamond"),
            hovertemplate=(
                f"<b>{strategy}</b><br>"
                "Volatility: %{x:.2f}%<br>"
                "Expected return: %{y:.2f}%<extra></extra>"
            ),
        ),
        row=1, col=2,
    )

# Chart 3: allocations.
for strategy in strategies:
    subset = allocations[allocations["Strategy"] == strategy]

    fig.add_trace(
        go.Bar(
            x=subset["Ticker"],
            y=subset["Weight_%"],
            name=strategy,
            hovertemplate=(
                f"<b>{strategy}</b><br>"
                "Ticker: %{x}<br>"
                "Weight: %{y:.2f}%<extra></extra>"
            ),
        ),
        row=2, col=1,
    )

# Chart 4: portfolio growth.
for strategy in strategy_returns.columns:
    growth = (1 + strategy_returns[strategy]).cumprod() * 100

    fig.add_trace(
        go.Scatter(
            x=growth.index,
            y=growth,
            name=strategy,
            mode="lines",
            hovertemplate=(
                f"<b>{strategy}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Growth index: %{y:.2f}<extra></extra>"
            ),
        ),
        row=2, col=2,
    )

# Chart 5: cumulative stock returns.
for ticker in stocks:
    returns_index = growth_index[ticker] * 100

    fig.add_trace(
        go.Scatter(
            x=returns_index.index,
            y=returns_index,
            name=ticker,
            mode="lines",
            showlegend=False,
            hovertemplate=(
                f"<b>{ticker}</b><br>"
                "Date: %{x|%d %b %Y}<br>"
                "Growth index: %{y:.2f}<extra></extra>"
            ),
        ),
        row=3, col=1,
    )

# Chart 6: strategy CAGR and volatility.
for _, row in summary.iterrows():
    fig.add_trace(
        go.Scatter(
            x=[row["Annualized_Volatility_%"]],
            y=[row["CAGR_%"]],
            mode="markers+text",
            text=[row["Strategy"]],
            textposition="top center",
            name=row["Strategy"],
            showlegend=False,
            marker=dict(size=14),
            customdata=[[
                row["Sharpe_Ratio"],
                row["Maximum_Drawdown_%"],
            ]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Volatility: %{x:.2f}%<br>"
                "CAGR: %{y:.2f}%<br>"
                "Sharpe: %{customdata[0]:.3f}<br>"
                "Max drawdown: %{customdata[1]:.2f}%"
                "<extra></extra>"
            ),
        ),
        row=3, col=2,
    )

fig.update_layout(
    title=dict(
        text=(
            "PORTFOLIO OPTIMIZATION INTELLIGENCE"
            "<br><sup>Asset Allocation • Efficient Frontier • "
            "Historical Risk and Return</sup>"
        ),
        x=0.03,
        xanchor="left",
    ),
    template="plotly_dark",
    height=1500,
    barmode="group",
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    margin=dict(l=70, r=45, t=150, b=70),
)

fig.update_xaxes(showgrid=True, gridcolor="rgba(150,150,150,0.15)")
fig.update_yaxes(showgrid=True, gridcolor="rgba(150,150,150,0.15)")

fig.update_yaxes(title_text="Investment value ($)", row=1, col=1)
fig.update_xaxes(title_text="Volatility (%)", row=1, col=2)
fig.update_yaxes(title_text="Expected return (%)", row=1, col=2)
fig.update_yaxes(title_text="Allocation (%)", row=2, col=1)
fig.update_yaxes(title_text="Growth index (start = 100)", row=2, col=2)
fig.update_yaxes(title_text="Growth index (start = 100)", row=3, col=1)
fig.update_xaxes(title_text="Volatility (%)", row=3, col=2)
fig.update_yaxes(title_text="CAGR (%)", row=3, col=2)

dashboard_file = CHART_DIR / "portfolio_optimization_dashboard.html"

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

# ============================================================
# 10. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PORTFOLIO OPTIMIZATION COMPLETED")
print("=" * 70)

print("\nPORTFOLIO STRATEGY COMPARISON")
print(summary.round(3).to_string(index=False))

print("\nOPTIMIZED PORTFOLIO WEIGHTS (%)")
print(
    allocations.pivot(
        index="Ticker",
        columns="Strategy",
        values="Weight_%",
    ).round(2).to_string()
)

print("\nINDIVIDUAL STOCK INVESTMENT SUMMARY")
print(investment_summary.round(2).to_string(index=False))

print("\nOUTPUT FILES")
for path in [
    REPORT_DIR / "optimized_portfolio_weights.csv",
    REPORT_DIR / "portfolio_optimization_summary.csv",
    REPORT_DIR / "optimized_strategy_daily_returns.csv",
    REPORT_DIR / "efficient_frontier.csv",
    REPORT_DIR / "individual_stock_investment_summary.csv",
    dashboard_file,
]:
    print(f"- {path}")

print(
    "\nNote: Estimates use historical data and are not guarantees "
    "of future performance."
)
