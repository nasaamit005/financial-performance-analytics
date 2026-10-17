
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "all_tickers_clean.csv"
REPORT_DIR = ROOT / "reports"

BENCHMARK = "^GSPC"
TRADING_DAYS = 252


def calculate_metrics(returns):
    returns = returns.dropna()

    if returns.empty:
        return {}

    wealth = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS

    total_return = wealth.iloc[-1] - 1
    cagr = wealth.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    volatility = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = (
        returns.mean() / returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
        if returns.std(ddof=1) > 0
        else np.nan
    )

    drawdown = wealth / wealth.cummax() - 1

    return {
        "Total_Return_Pct": total_return * 100,
        "CAGR_Pct": cagr * 100,
        "Annualized_Volatility_Pct": volatility * 100,
        "Sharpe_Ratio_RF_Zero": sharpe,
        "Max_Drawdown_Pct": drawdown.min() * 100,
        "Observations": len(returns),
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])

    price_col = (
        "Adj_Close" if "Adj_Close" in df.columns else "Close"
    )

    required = {"Date", "Ticker", price_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", price_col])
    df = df[df[price_col] > 0]
    df = df.sort_values(["Date", "Ticker"])
    df = df.drop_duplicates(["Date", "Ticker"], keep="last")

    prices = df.pivot(
        index="Date",
        columns="Ticker",
        values=price_col,
    ).sort_index()

    if BENCHMARK not in prices.columns:
        raise ValueError(f"Benchmark {BENCHMARK} not found.")

    stock_tickers = [
        ticker for ticker in ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]
        if ticker in prices.columns
    ]

    if len(stock_tickers) < 2:
        raise ValueError("At least two stock tickers are required.")

    # Use dates with complete price observations for all selected assets.
    prices = prices[[*stock_tickers, BENCHMARK]].dropna()

    daily_returns = prices.pct_change(fill_method=None).dropna()

    # Equal weight across the five stocks, rebalanced daily.
    portfolio_returns = daily_returns[stock_tickers].mean(axis=1)
    benchmark_returns = daily_returns[BENCHMARK]

    result = pd.DataFrame({
        "Date": daily_returns.index,
        "Portfolio_Return": portfolio_returns.to_numpy(),
        "Benchmark_Return": benchmark_returns.to_numpy(),
    })

    result["Portfolio_Growth_10000"] = (
        10000 * (1 + result["Portfolio_Return"]).cumprod()
    )
    result["Benchmark_Growth_10000"] = (
        10000 * (1 + result["Benchmark_Return"]).cumprod()
    )

    portfolio_metrics = calculate_metrics(portfolio_returns)
    benchmark_metrics = calculate_metrics(benchmark_returns)

    summary = pd.DataFrame([
        {"Portfolio": "Equal-Weighted Stocks", **portfolio_metrics},
        {"Portfolio": "S&P 500 Benchmark", **benchmark_metrics},
    ])

    # Estimate beta using daily portfolio and benchmark returns.
    benchmark_variance = benchmark_returns.var(ddof=1)
    if benchmark_variance > 0:
        beta = portfolio_returns.cov(benchmark_returns) / benchmark_variance
    else:
        beta = np.nan

    summary["Beta_vs_SP500"] = [
        beta,
        1.0,
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    daily_path = REPORT_DIR / "portfolio_daily_returns.csv"
    summary_path = REPORT_DIR / "portfolio_summary.csv"

    result.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nPortfolio comparison:")
    print(summary.round(3).to_string(index=False))
    print("\nSaved files:")
    print(daily_path)
    print(summary_path)


if __name__ == "__main__":
    main()
