
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/processed/all_tickers_clean.csv")
OUTPUT_DIR = Path("reports")
CHART_DIR = OUTPUT_DIR / "charts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CALCULATE TECHNICAL INDICATORS
# ============================================================

def calculate_indicators(group):
    """Calculate RSI, MACD and Bollinger Bands for one ticker."""

    group = group.sort_values("Date").copy()
    price = pd.to_numeric(group["Adj_Close"], errors="coerce")

    # RSI (14 periods)
    change = price.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss
    group["RSI_14"] = 100 - (100 / (1 + rs))

    group.loc[
        (avg_loss == 0) & (avg_gain > 0), "RSI_14"
    ] = 100

    group.loc[
        (avg_loss == 0) & (avg_gain == 0), "RSI_14"
    ] = 50

    # MACD (12, 26, 9)
    ema_12 = price.ewm(
        span=12, adjust=False, min_periods=12
    ).mean()

    ema_26 = price.ewm(
        span=26, adjust=False, min_periods=26
    ).mean()

    group["MACD"] = ema_12 - ema_26

    group["MACD_Signal"] = group["MACD"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()

    group["MACD_Histogram"] = (
        group["MACD"] - group["MACD_Signal"]
    )

    # Bollinger Bands (20 periods, +/- 2 standard deviations)
    group["SMA_20"] = price.rolling(
        window=20, min_periods=20
    ).mean()

    rolling_std = price.rolling(
        window=20, min_periods=20
    ).std()

    group["BB_Upper"] = group["SMA_20"] + 2 * rolling_std
    group["BB_Lower"] = group["SMA_20"] - 2 * rolling_std

    band_width = group["BB_Upper"] - group["BB_Lower"]

    group["BB_Percent"] = (
        (price - group["BB_Lower"]) / band_width
    ).where(band_width != 0)

    return group


# ============================================================
# 3. STATIC CHARTS (PNG)
# ============================================================

def create_static_charts(result, ticker="NVDA"):
    """Save PNG charts for reports and project documentation."""

    stock = result[result["Ticker"] == ticker].sort_values("Date")

    if stock.empty:
        print(f"No data found for {ticker}; static charts skipped.")
        return

    # Price and Bollinger Bands
    plt.figure(figsize=(13, 6))
    plt.plot(stock["Date"], stock["Adj_Close"], label="Adjusted Close")
    plt.plot(stock["Date"], stock["SMA_20"], label="20-Day SMA")
    plt.plot(stock["Date"], stock["BB_Upper"], label="Upper Band")
    plt.plot(stock["Date"], stock["BB_Lower"], label="Lower Band")
    plt.title(f"{ticker}: Price and Bollinger Bands")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Price")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(CHART_DIR / f"{ticker.lower()}_bollinger_bands.png", dpi=150)
    plt.close()

    # RSI
    plt.figure(figsize=(13, 4))
    plt.plot(stock["Date"], stock["RSI_14"], label="RSI (14)")
    plt.axhline(70, linestyle="--", label="Reference: 70")
    plt.axhline(30, linestyle="--", label="Reference: 30")
    plt.ylim(0, 100)
    plt.title(f"{ticker}: Relative Strength Index")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(CHART_DIR / f"{ticker.lower()}_rsi.png", dpi=150)
    plt.close()

    # MACD
    plt.figure(figsize=(13, 4))
    plt.plot(stock["Date"], stock["MACD"], label="MACD")
    plt.plot(stock["Date"], stock["MACD_Signal"], label="Signal Line")
    plt.bar(stock["Date"], stock["MACD_Histogram"], label="Histogram", width=1)
    plt.axhline(0, linewidth=0.8)
    plt.title(f"{ticker}: MACD")
    plt.xlabel("Date")
    plt.ylabel("MACD Value")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(CHART_DIR / f"{ticker.lower()}_macd.png", dpi=150)
    plt.close()

    print(f"Static PNG charts saved for {ticker}.")


# ============================================================
# 4. INTERACTIVE PRICE AND BOLLINGER BAND CHART
# ============================================================

def create_interactive_price_chart(stock, ticker):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stock["Date"],
        y=stock["Adj_Close"],
        mode="lines",
        name="Adjusted Close",
        customdata=stock[["RSI_14", "MACD"]].to_numpy(),
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            "Adjusted Close: %{y:.2f}<br>"
            "RSI: %{customdata[0]:.2f}<br>"
            "MACD: %{customdata[1]:.3f}"
            "<extra></extra>"
        )
    ))

    for column, label in [
        ("SMA_20", "20-Day SMA"),
        ("BB_Upper", "Upper Bollinger Band"),
        ("BB_Lower", "Lower Bollinger Band"),
    ]:
        fig.add_trace(go.Scatter(
            x=stock["Date"],
            y=stock[column],
            mode="lines",
            name=label,
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                + label + ": %{y:.2f}<extra></extra>"
            )
        ))

    fig.update_layout(
        title=f"{ticker}: Price and Bollinger Bands",
        xaxis_title="Date",
        yaxis_title="Adjusted Price",
        hovermode="x unified",
        template="plotly_white",
        legend_title="Indicator",
        xaxis_rangeslider_visible=True
    )

    fig.write_html(
        str(CHART_DIR / f"{ticker.lower()}_interactive_bollinger.html"),
        include_plotlyjs=True
    )


# ============================================================
# 5. INTERACTIVE RSI CHART
# ============================================================

def create_interactive_rsi_chart(stock, ticker):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stock["Date"],
        y=stock["RSI_14"],
        mode="lines",
        name="RSI (14)",
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            "RSI: %{y:.2f}<extra></extra>"
        )
    ))

    fig.add_hline(
        y=70, line_dash="dash",
        annotation_text="Overbought reference: 70"
    )
    fig.add_hline(
        y=30, line_dash="dash",
        annotation_text="Oversold reference: 30"
    )

    fig.update_layout(
        title=f"{ticker}: Relative Strength Index",
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis_range=[0, 100],
        hovermode="x unified",
        template="plotly_white"
    )

    fig.write_html(
        str(CHART_DIR / f"{ticker.lower()}_interactive_rsi.html"),
        include_plotlyjs=True
    )


# ============================================================
# 6. INTERACTIVE MACD CHART
# ============================================================

def create_interactive_macd_chart(stock, ticker):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("MACD and Signal Line", "MACD Histogram")
    )

    fig.add_trace(go.Scatter(
        x=stock["Date"],
        y=stock["MACD"],
        mode="lines",
        name="MACD",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>MACD: %{y:.4f}<extra></extra>"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=stock["Date"],
        y=stock["MACD_Signal"],
        mode="lines",
        name="Signal Line",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Signal: %{y:.4f}<extra></extra>"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=stock["Date"],
        y=stock["MACD_Histogram"],
        name="Histogram",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Histogram: %{y:.4f}<extra></extra>"
    ), row=2, col=1)

    fig.update_layout(
        title=f"{ticker}: MACD Analysis",
        template="plotly_white",
        hovermode="x unified"
    )

    fig.write_html(
        str(CHART_DIR / f"{ticker.lower()}_interactive_macd.html"),
        include_plotlyjs=True
    )


# ============================================================
# 7. INTERACTIVE STOCK PERFORMANCE COMPARISON
# ============================================================

def create_performance_comparison(result):
    """Compare price-index performance from a common base of 100."""

    comparison = result.sort_values(["Ticker", "Date"]).copy()

    comparison["Indexed_Performance"] = (
        comparison["Adj_Close"]
        / comparison.groupby("Ticker")["Adj_Close"].transform("first")
        * 100
    )

    fig = go.Figure()

    for ticker, group in comparison.groupby("Ticker", sort=True):
        group = group.sort_values("Date")

        fig.add_trace(go.Scatter(
            x=group["Date"],
            y=group["Indexed_Performance"],
            mode="lines",
            name=ticker,
            customdata=group["Indexed_Performance"] - 100,
            hovertemplate=(
                "Ticker: " + ticker + "<br>"
                "Date: %{x|%Y-%m-%d}<br>"
                "Growth of 100: %{y:.2f}<br>"
                "Cumulative price return: %{customdata:.2f}%"
                "<extra></extra>"
            )
        ))

    fig.add_hline(
        y=100,
        line_dash="dash",
        annotation_text="Starting value: 100"
    )

    fig.update_layout(
        title="Interactive Stock vs S&P 500 Performance",
        xaxis_title="Date",
        yaxis_title="Growth of 100",
        hovermode="x unified",
        template="plotly_white",
        legend_title="Ticker",
        xaxis_rangeslider_visible=True
    )

    fig.write_html(
        str(CHART_DIR / "interactive_performance_comparison.html"),
        include_plotlyjs=True
    )


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE.resolve()}\n"
            "Run src/clean_data.py first."
        )

    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])

    required_columns = {"Date", "Ticker", "Adj_Close"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Input dataset is empty.")

    df["Adj_Close"] = pd.to_numeric(
        df["Adj_Close"], errors="coerce"
    )

    df = df.dropna(subset=["Date", "Ticker", "Adj_Close"])
    df = df.drop_duplicates(
        subset=["Ticker", "Date"], keep="last"
    )
    df = df.sort_values(["Ticker", "Date"])

    if df.empty:
        raise ValueError("No valid rows remain after data validation.")

    print(f"Rows loaded: {len(df):,}")
    print(f"Tickers: {', '.join(sorted(df['Ticker'].unique()))}")

    # Correct pandas-compatible processing: no include_groups argument.
    indicator_groups = []

    for ticker, group in df.groupby("Ticker", sort=True):
        print(f"Calculating indicators for {ticker}...")
        indicator_groups.append(calculate_indicators(group))

    result = pd.concat(indicator_groups, ignore_index=True)
    result = result.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    # Save the complete technical-indicator dataset.
    output_file = OUTPUT_DIR / "technical_indicators.csv"
    result.to_csv(output_file, index=False)
    print(f"\nDataset saved: {output_file.resolve()}")

    # Display latest indicator values.
    display_columns = [
        "Date", "Ticker", "Adj_Close", "RSI_14",
        "MACD", "MACD_Signal", "BB_Upper", "BB_Lower"
    ]

    latest = (
        result.dropna(
            subset=["RSI_14", "MACD_Signal", "BB_Upper", "BB_Lower"]
        )
        .groupby("Ticker")
        .tail(1)
        .sort_values("Ticker")
    )

    print("\nLATEST TECHNICAL INDICATORS")
    if latest.empty:
        print("No complete indicator values are available.")
    else:
        print(latest[display_columns].round(3).to_string(index=False))

    # Save static charts.
    create_static_charts(result, ticker="NVDA")

    # Generate interactive charts for every ticker.
    for ticker, stock in result.groupby("Ticker", sort=True):
        stock = stock.sort_values("Date")

        create_interactive_price_chart(stock, ticker)
        create_interactive_rsi_chart(stock, ticker)
        create_interactive_macd_chart(stock, ticker)

        print(f"Interactive charts created for {ticker}.")

    # Compare all tickers on the same indexed scale.
    create_performance_comparison(result)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print(f"All outputs: {OUTPUT_DIR.resolve()}")
    print(f"Interactive charts: {CHART_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
