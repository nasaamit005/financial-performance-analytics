
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

INPUT_FILE = Path("data/processed/all_tickers_clean.csv")
OUTPUT_DIR = Path("reports")
CHART_DIR = OUTPUT_DIR / "charts"

BENCHMARK = "^GSPC"
TRADING_DAYS = 252
RISK_FREE_RATE = 0.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

def analyze_performance():

    print("\n" + "=" * 65)
    print("FINANCIAL PERFORMANCE ANALYTICS")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Please run your data cleaning script first."
        )

    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])

    required_columns = {"Date", "Ticker"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    price_col = (
        "Adj_Close"
        if "Adj_Close" in df.columns
        else "Close"
        if "Close" in df.columns
        else None
    )

    if price_col is None:
        raise ValueError(
            "No price column found. Expected Adj_Close or Close."
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[price_col] = pd.to_numeric(
        df[price_col], errors="coerce"
    )

    df = df.dropna(
        subset=["Date", "Ticker", price_col]
    )

    df = df[df[price_col] > 0].copy()

    df["Ticker"] = df["Ticker"].astype(str)

    df = df.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    if df.duplicated(["Ticker", "Date"]).any():
        raise ValueError(
            "Duplicate ticker/date rows found. "
            "Please clean the data before analysis."
        )

    if df.empty:
        raise ValueError("No valid price observations found.")

    print(f"Input file: {INPUT_FILE}")
    print(f"Price column: {price_col}")
    print(f"Observations: {len(df):,}")
    print(f"Tickers: {df['Ticker'].nunique()}")
    print(
        f"Date range: {df['Date'].min().date()} "
        f"to {df['Date'].max().date()}"
    )


    # ========================================================
    # 3. DAILY RETURNS AND GROWTH OF $1
    # ========================================================

    df["Daily_Return"] = (
        df.groupby("Ticker")[price_col]
        .pct_change(fill_method=None)
    )

    df["Growth_of_1"] = (
        (1 + df["Daily_Return"])
        .groupby(df["Ticker"])
        .cumprod()
    )

    df["Cumulative_Return"] = df["Growth_of_1"] - 1

    df.to_csv(
        OUTPUT_DIR / "daily_returns.csv",
        index=False
    )

    print("\nDaily returns saved.")


    # ========================================================
    # 4. PERFORMANCE SUMMARY
    # ========================================================

    summary_rows = []

    for ticker, group in df.groupby("Ticker"):

        group = group.sort_values("Date")

        prices = group[price_col].dropna()
        returns = group["Daily_Return"].dropna()

        if len(prices) < 2:
            continue

        first_date = group["Date"].iloc[0]
        last_date = group["Date"].iloc[-1]

        first_price = prices.iloc[0]
        last_price = prices.iloc[-1]

        cumulative_return = (
            last_price / first_price
        ) - 1

        elapsed_days = (
            last_date - first_date
        ).days

        if elapsed_days > 0 and first_price > 0:
            cagr = (
                (last_price / first_price)
                ** (365.25 / elapsed_days)
            ) - 1
        else:
            cagr = np.nan

        if len(returns) > 1:
            daily_volatility = returns.std(ddof=1)
            annualized_volatility = (
                daily_volatility * np.sqrt(TRADING_DAYS)
            )

            daily_excess_return = (
                returns - RISK_FREE_RATE / TRADING_DAYS
            )

            if daily_volatility > 0:
                sharpe_ratio = (
                    daily_excess_return.mean()
                    / daily_volatility
                ) * np.sqrt(TRADING_DAYS)
            else:
                sharpe_ratio = np.nan
        else:
            annualized_volatility = np.nan
            sharpe_ratio = np.nan

        # Maximum drawdown
        growth = (1 + returns).cumprod()

        running_peak = growth.cummax()

        drawdown = (growth / running_peak) - 1

        max_drawdown = (
            drawdown.min()
            if not drawdown.empty
            else np.nan
        )

        # Downside deviation: daily returns below zero
        negative_returns = returns[returns < 0]

        if len(returns) > 0:
            downside_deviation = np.sqrt(
                np.mean(np.minimum(returns, 0) ** 2)
            ) * np.sqrt(TRADING_DAYS)
        else:
            downside_deviation = np.nan

        # Sortino ratio
        if (
            pd.notna(downside_deviation)
            and downside_deviation > 0
        ):
            annualized_mean_excess = (
                returns.mean() - RISK_FREE_RATE / TRADING_DAYS
            ) * TRADING_DAYS

            sortino_ratio = (
                annualized_mean_excess / downside_deviation
            )
        else:
            sortino_ratio = np.nan

        summary_rows.append({
            "Ticker": ticker,
            "Start_Date": first_date,
            "End_Date": last_date,
            "Observations": len(group),
            "First_Price": first_price,
            "Last_Price": last_price,
            "Cumulative_Return": cumulative_return,
            "CAGR": cagr,
            "Annualized_Volatility": annualized_volatility,
            "Sharpe_Ratio": sharpe_ratio,
            "Maximum_Drawdown": max_drawdown,
            "Downside_Deviation": downside_deviation,
            "Sortino_Ratio": sortino_ratio,
            "Negative_Days": len(negative_returns),
        })

    summary = pd.DataFrame(summary_rows)

    if summary.empty:
        raise ValueError(
            "Could not calculate summary metrics. "
            "Check the available price history."
        )

    summary.to_csv(
        OUTPUT_DIR / "performance_summary.csv",
        index=False
    )

    print("\nPERFORMANCE SUMMARY")
    print("-" * 65)

    display_columns = [
        "Ticker",
        "Cumulative_Return",
        "CAGR",
        "Annualized_Volatility",
        "Sharpe_Ratio",
        "Maximum_Drawdown",
        "Sortino_Ratio",
    ]

    print(
        summary[display_columns]
        .to_string(
            index=False,
            formatters={
                "Cumulative_Return": "{:.2%}".format,
                "CAGR": "{:.2%}".format,
                "Annualized_Volatility": "{:.2%}".format,
                "Sharpe_Ratio": "{:.2f}".format,
                "Maximum_Drawdown": "{:.2%}".format,
                "Sortino_Ratio": "{:.2f}".format,
            }
        )
    )


    # ========================================================
    # 5. BENCHMARK COMPARISON AND CORRELATIONS
    # ========================================================

    return_table = df.pivot(
        index="Date",
        columns="Ticker",
        values="Daily_Return"
    ).sort_index()

    if BENCHMARK in return_table.columns:

        benchmark_returns = return_table[BENCHMARK]

        asset_columns = [
            ticker
            for ticker in return_table.columns
            if ticker != BENCHMARK
        ]

        comparison_rows = []

        for ticker in asset_columns:

            paired = pd.concat(
                [
                    return_table[ticker].rename("Asset_Return"),
                    benchmark_returns.rename("Benchmark_Return"),
                ],
                axis=1,
            ).dropna()

            if paired.empty:
                continue

            active_returns = (
                paired["Asset_Return"]
                - paired["Benchmark_Return"]
            )

            comparison_rows.append({
                "Ticker": ticker,
                "Benchmark": BENCHMARK,
                "Observations": len(paired),
                "Correlation": paired[
                    "Asset_Return"
                ].corr(paired["Benchmark_Return"]),
                "Average_Daily_Active_Return": active_returns.mean(),
                "Annualized_Active_Return_Approx": (
                    active_returns.mean() * TRADING_DAYS
                ),
                "Tracking_Error": (
                    active_returns.std(ddof=1)
                    * np.sqrt(TRADING_DAYS)
                    if len(active_returns) > 1
                    else np.nan
                ),
            })

        benchmark_comparison = pd.DataFrame(
            comparison_rows
        )

        benchmark_comparison.to_csv(
            OUTPUT_DIR / "benchmark_comparison_daily.csv",
            index=False
        )

        print("\nBenchmark comparison saved.")

    else:
        print(
            f"\nBenchmark {BENCHMARK} not found. "
            "Skipping benchmark comparison."
        )

    correlations = return_table.corr()

    correlations.to_csv(
        OUTPUT_DIR / "return_correlations.csv"
    )

    print("Return correlations saved.")


    # ========================================================
    # 6. MATPLOTLIB STATIC CHARTS
    # ========================================================

    # Cumulative performance chart
    plt.figure(figsize=(14, 7))

    for ticker, group in df.groupby("Ticker"):
        plt.plot(
            group["Date"],
            group["Growth_of_1"],
            label=ticker,
            linewidth=1.8
        )

    plt.title("Cumulative Stock Performance")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "cumulative_performance.png",
        dpi=160
    )

    plt.close()

    # Annualized volatility chart
    volatility_data = summary.dropna(
        subset=["Annualized_Volatility"]
    ).sort_values("Annualized_Volatility")

    plt.figure(figsize=(11, 6))

    plt.bar(
        volatility_data["Ticker"],
        volatility_data["Annualized_Volatility"]
    )

    plt.title("Annualized Volatility by Stock")
    plt.xlabel("Ticker")
    plt.ylabel("Annualized Volatility")
    plt.gca().yaxis.set_major_formatter(
        plt.matplotlib.ticker.PercentFormatter(1.0)
    )

    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "annualized_volatility.png",
        dpi=160
    )

    plt.close()


    # ========================================================
    # 7. INTERACTIVE PLOTLY PERFORMANCE CHART
    # ========================================================

    performance_fig = px.line(
        df.dropna(subset=["Growth_of_1"]),
        x="Date",
        y="Growth_of_1",
        color="Ticker",
        title="Interactive Stock Performance Comparison",
        labels={
            "Date": "Date",
            "Growth_of_1": "Growth of $1",
            "Ticker": "Stock",
        },
        template="plotly_dark",
        hover_data={
            "Cumulative_Return": ":.2%",
            "Growth_of_1": ":.3f",
        },
    )

    performance_fig.update_layout(
        height=700,
        hovermode="x unified",
        legend_title_text="Stock",
        xaxis_rangeslider_visible=True,
        margin=dict(l=60, r=40, t=80, b=60),
    )

    performance_fig.write_html(
        str(
            CHART_DIR
            / "cumulative_performance_interactive.html"
        ),
        include_plotlyjs=True
    )


    # ========================================================
    # 8. INTERACTIVE ANNUALIZED VOLATILITY CHART
    # ========================================================

    volatility_fig = px.bar(
        summary.sort_values("Annualized_Volatility"),
        x="Ticker",
        y="Annualized_Volatility",
        color="Ticker",
        title="Annualized Volatility by Stock",
        labels={
            "Ticker": "Stock",
            "Annualized_Volatility": "Annualized Volatility",
        },
        template="plotly_dark",
        hover_data={
            "Observations": True,
            "Annualized_Volatility": ":.2%",
        },
    )

    volatility_fig.update_layout(
        height=600,
        showlegend=False,
    )

    volatility_fig.update_yaxes(
        tickformat=".1%"
    )

    volatility_fig.write_html(
        str(
            CHART_DIR
            / "annualized_volatility_interactive.html"
        ),
        include_plotlyjs=True
    )


    # ========================================================
    # 9. INTERACTIVE MAXIMUM DRAWDOWN CHART
    # ========================================================

    drawdown_frames = []

    for ticker, group in df.groupby("Ticker"):

        group = group.sort_values("Date").copy()

        growth = (
            (1 + group["Daily_Return"].fillna(0))
            .cumprod()
        )

        peak = growth.cummax()

        group["Drawdown"] = (growth / peak) - 1

        drawdown_frames.append(
            group[["Date", "Ticker", "Drawdown"]]
        )

    drawdown_df = pd.concat(
        drawdown_frames,
        ignore_index=True
    )

    drawdown_fig = px.line(
        drawdown_df,
        x="Date",
        y="Drawdown",
        color="Ticker",
        title="Interactive Portfolio Drawdown Over Time",
        labels={
            "Date": "Date",
            "Drawdown": "Drawdown",
            "Ticker": "Stock",
        },
        template="plotly_dark",
    )

    drawdown_fig.update_layout(
        height=650,
        hovermode="x unified",
        xaxis_rangeslider_visible=True,
    )

    drawdown_fig.update_yaxes(
        tickformat=".1%"
    )

    drawdown_fig.write_html(
        str(
            CHART_DIR
            / "maximum_drawdown_interactive.html"
        ),
        include_plotlyjs=True
    )


    # ========================================================
    # 10. INTERACTIVE CORRELATION HEATMAP
    # ========================================================

    correlation_fig = go.Figure(
        data=go.Heatmap(
            z=correlations.values,
            x=correlations.columns.tolist(),
            y=correlations.index.tolist(),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=np.round(
                correlations.values, 2
            ),
            texttemplate="%{text}",
            hovertemplate=(
                "Stock 1: %{x}<br>"
                "Stock 2: %{y}<br>"
                "Correlation: %{z:.3f}"
                "<extra></extra>"
            ),
            colorbar=dict(
                title="Correlation"
            ),
        )
    )

    correlation_fig.update_layout(
        title="Interactive Stock Return Correlation Heatmap",
        template="plotly_dark",
        height=650,
        xaxis_title="Stock",
        yaxis_title="Stock",
    )

    correlation_fig.write_html(
        str(
            CHART_DIR
            / "return_correlation_heatmap_interactive.html"
        ),
        include_plotlyjs=True
    )


    # ========================================================
    # 11. INTERACTIVE ROLLING VOLATILITY CHART
    # ========================================================

    rolling_frames = []

    for ticker, group in df.groupby("Ticker"):

        group = group.sort_values("Date").copy()

        group["Rolling_Volatility_30D"] = (
            group["Daily_Return"]
            .rolling(window=30, min_periods=20)
            .std()
            * np.sqrt(TRADING_DAYS)
        )

        rolling_frames.append(
            group[
                ["Date", "Ticker", "Rolling_Volatility_30D"]
            ]
        )

    rolling_df = pd.concat(
        rolling_frames,
        ignore_index=True
    )

    rolling_fig = px.line(
        rolling_df.dropna(
            subset=["Rolling_Volatility_30D"]
        ),
        x="Date",
        y="Rolling_Volatility_30D",
        color="Ticker",
        title="30-Trading-Day Rolling Annualized Volatility",
        labels={
            "Date": "Date",
            "Rolling_Volatility_30D": "Annualized Volatility",
            "Ticker": "Stock",
        },
        template="plotly_dark",
    )

    rolling_fig.update_layout(
        height=650,
        hovermode="x unified",
        xaxis_rangeslider_visible=True,
    )

    rolling_fig.update_yaxes(
        tickformat=".1%"
    )

    rolling_fig.write_html(
        str(
            CHART_DIR
            / "rolling_volatility_interactive.html"
        ),
        include_plotlyjs=True
    )


    # ========================================================
    # 12. REPORT OUTPUTS
    # ========================================================

    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("\nCSV reports:")
    for file_path in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  {file_path}")

    print("\nCharts:")
    for file_path in sorted(CHART_DIR.iterdir()):
        if file_path.is_file():
            print(f"  {file_path}")

    print("\nNext step: open the interactive HTML charts in your browser.")

    return summary


if __name__ == "__main__":
    analyze_performance()
