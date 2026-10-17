
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "data" / "processed" / "all_tickers_clean.csv"

BENCHMARK = "^GSPC"
TRADING_DAYS = 252

st.set_page_config(
    page_title="Financial Performance Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

@st.cache_data
def load_data(file_path, modified_time):
    data = pd.read_csv(file_path, parse_dates=["Date"])

    price_column = (
        "Adj_Close" if "Adj_Close" in data.columns else "Close"
    )

    required = {"Date", "Ticker", price_column}
    if not required.issubset(data.columns):
        raise ValueError(
            f"Dataset must contain these columns: {required}"
        )

    data["Ticker"] = data["Ticker"].astype(str)
    data[price_column] = pd.to_numeric(
        data[price_column], errors="coerce"
    )
    data = data.dropna(
        subset=["Date", "Ticker", price_column]
    )
    data = data[data[price_column] > 0]
    data = data.sort_values(["Ticker", "Date"])

    if data.duplicated(["Ticker", "Date"]).any():
        raise ValueError(
            "Duplicate ticker/date records found in the dataset."
        )

    data["Daily_Return"] = (
        data.groupby("Ticker")[price_column]
        .pct_change(fill_method=None)
    )

    return data, price_column


st.title("📈 Financial Performance Analytics")
st.caption(
    "Historical stock performance, benchmark comparison, "
    "risk metrics, and portfolio insights"
)

if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.info(
        "Check that data/processed/all_tickers_clean.csv "
        "exists in your project."
    )
    st.stop()

try:
    df, price_col = load_data(
        str(DATA_FILE),
        DATA_FILE.stat().st_mtime,
    )
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

all_tickers = sorted(df["Ticker"].unique().tolist())

if not all_tickers:
    st.error("No tickers were found in the dataset.")
    st.stop()


# ------------------------------------------------------------
# 3. SIDEBAR FILTERS
# ------------------------------------------------------------

st.sidebar.title("Dashboard Controls")

default_tickers = [
    ticker for ticker in all_tickers
    if ticker != BENCHMARK
]

if not default_tickers:
    default_tickers = all_tickers

selected_tickers = st.sidebar.multiselect(
    "Select stocks",
    options=all_tickers,
    default=default_tickers,
)

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

selected_dates = st.sidebar.date_input(
    "Analysis date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if not selected_tickers:
    st.warning("Select at least one ticker from the sidebar.")
    st.stop()

if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
    st.info("Select both a start date and an end date.")
    st.stop()

start_date, end_date = selected_dates

if start_date > end_date:
    st.error("The start date must be on or before the end date.")
    st.stop()

start_timestamp = pd.Timestamp(start_date)
end_timestamp = (
    pd.Timestamp(end_date) + pd.Timedelta(days=1)
    - pd.Timedelta(nanoseconds=1)
)

filtered = df[
    df["Ticker"].isin(selected_tickers)
    & (df["Date"] >= start_timestamp)
    & (df["Date"] <= end_timestamp)
].copy()

if filtered.empty:
    st.warning("No observations match the selected filters.")
    st.stop()

filtered = filtered.sort_values(["Ticker", "Date"])


# ------------------------------------------------------------
# 4. CALCULATE METRICS FOR THE SELECTED PERIOD
# ------------------------------------------------------------

filtered["Period_Growth"] = (
    filtered.groupby("Ticker")[price_col]
    .transform(lambda prices: prices / prices.iloc[0])
)

filtered["Period_Return"] = filtered["Period_Growth"] - 1

filtered["Drawdown"] = filtered.groupby("Ticker")[
    "Period_Growth"
].transform(lambda growth: growth / growth.cummax() - 1)

filtered["Rolling_Volatility"] = filtered.groupby("Ticker")[
    "Daily_Return"
].transform(
    lambda returns: (
        returns.rolling(30, min_periods=20).std()
        * np.sqrt(TRADING_DAYS)
    )
)

summary_rows = []

for ticker, group in filtered.groupby("Ticker"):
    group = group.sort_values("Date")
    prices = group[price_col]
    returns = group["Daily_Return"].dropna()

    total_return = (
        prices.iloc[-1] / prices.iloc[0] - 1
        if len(prices) >= 2 else np.nan
    )

    elapsed_days = (
        group["Date"].iloc[-1] - group["Date"].iloc[0]
    ).days

    if elapsed_days > 0 and prices.iloc[0] > 0:
        cagr = (
            (prices.iloc[-1] / prices.iloc[0])
            ** (365.25 / elapsed_days)
        ) - 1
    else:
        cagr = np.nan

    volatility = (
        returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
        if len(returns) > 1 else np.nan
    )

    sharpe = (
        returns.mean() / returns.std(ddof=1)
        * np.sqrt(TRADING_DAYS)
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else np.nan
    )

    growth = group["Period_Growth"]
    drawdown = growth / growth.cummax() - 1
    max_drawdown = drawdown.min()

    summary_rows.append({
        "Ticker": ticker,
        "Start Price": prices.iloc[0],
        "Latest Price": prices.iloc[-1],
        "Period Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": max_drawdown,
        "Observations": len(group),
    })

summary = pd.DataFrame(summary_rows)


# ------------------------------------------------------------
# 5. KPI CARDS
# ------------------------------------------------------------

st.subheader("Performance Overview")

preferred = [
    ticker for ticker in selected_tickers
    if ticker != BENCHMARK
]
kpi_ticker = preferred[0] if preferred else selected_tickers[0]

selected_metric = summary[
    summary["Ticker"] == kpi_ticker
].iloc[0]

st.caption(
    f"KPIs shown for {kpi_ticker} | "
    f"{start_date} to {end_date}"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Period Return",
    f"{selected_metric['Period Return']:.2%}"
    if pd.notna(selected_metric["Period Return"]) else "N/A",
)

c2.metric(
    "Annualized Volatility",
    f"{selected_metric['Annualized Volatility']:.2%}"
    if pd.notna(selected_metric["Annualized Volatility"]) else "N/A",
)

c3.metric(
    "Sharpe Ratio",
    f"{selected_metric['Sharpe Ratio']:.2f}"
    if pd.notna(selected_metric["Sharpe Ratio"]) else "N/A",
)

c4.metric(
    "Maximum Drawdown",
    f"{selected_metric['Maximum Drawdown']:.2%}"
    if pd.notna(selected_metric["Maximum Drawdown"]) else "N/A",
)


# ------------------------------------------------------------
# 6. STOCK PERFORMANCE COMPARISON
# ------------------------------------------------------------

st.divider()
st.subheader("1. Stock Performance Comparison")

fig_performance = px.line(
    filtered,
    x="Date",
    y="Period_Growth",
    color="Ticker",
    title="Growth of $1 — rebased at the start of the selected period",
    labels={
        "Date": "Date",
        "Period_Growth": "Growth of $1",
        "Ticker": "Stock",
    },
    template="plotly_dark",
)

fig_performance.update_layout(
    height=520,
    hovermode="x unified",
    legend_title_text="Ticker",
    xaxis_rangeslider_visible=True,
)

st.plotly_chart(fig_performance, width="stretch")


# ------------------------------------------------------------
# 7. RETURN COMPARISON
# ------------------------------------------------------------

st.subheader("2. Return Comparison")

return_chart = summary.sort_values(
    "Period Return", ascending=False
)

fig_returns = px.bar(
    return_chart,
    x="Ticker",
    y="Period Return",
    color="Period Return",
    color_continuous_scale="RdYlGn",
    text=return_chart["Period Return"].map(
        lambda value: f"{value:.1%}" if pd.notna(value) else "N/A"
    ),
    title="Total Return for Selected Period",
    template="plotly_dark",
)

fig_returns.update_layout(height=430)
fig_returns.update_yaxes(tickformat=".1%")
fig_returns.update_traces(textposition="outside")

st.plotly_chart(fig_returns, use_container_width=True)


# ------------------------------------------------------------
# 8. VOLATILITY AND DRAWDOWN
# ------------------------------------------------------------

left, right = st.columns(2)

with left:
    fig_vol = px.bar(
        summary.sort_values("Annualized Volatility"),
        x="Ticker",
        y="Annualized Volatility",
        color="Ticker",
        title="Annualized Volatility",
        template="plotly_dark",
        hover_data=["Observations"],
    )
    fig_vol.update_yaxes(tickformat=".1%")
    fig_vol.update_layout(height=430, showlegend=False)
    st.plotly_chart(fig_vol, use_container_width=True)

with right:
    fig_drawdown = px.line(
        filtered,
        x="Date",
        y="Drawdown",
        color="Ticker",
        title="Drawdown from Previous Peak",
        template="plotly_dark",
        labels={"Drawdown": "Drawdown"},
    )
    fig_drawdown.update_yaxes(tickformat=".1%")
    fig_drawdown.update_layout(
        height=430,
        hovermode="x unified",
    )
    st.plotly_chart(fig_drawdown, use_container_width=True)


# ------------------------------------------------------------
# 9. ROLLING VOLATILITY
# ------------------------------------------------------------

st.subheader("3. Rolling 30-Trading-Day Volatility")

fig_rolling = px.line(
    filtered.dropna(subset=["Rolling_Volatility"]),
    x="Date",
    y="Rolling_Volatility",
    color="Ticker",
    title="Annualized Rolling Volatility",
    template="plotly_dark",
    labels={
        "Rolling_Volatility": "Annualized Volatility",
        "Date": "Date",
    },
)

fig_rolling.update_yaxes(tickformat=".1%")
fig_rolling.update_layout(
    height=470,
    hovermode="x unified",
)

st.plotly_chart(fig_rolling, use_container_width=True)


# ------------------------------------------------------------
# 10. RETURN CORRELATION HEATMAP
# ------------------------------------------------------------

st.subheader("4. Daily Return Correlation")

return_pivot = filtered.pivot(
    index="Date",
    columns="Ticker",
    values="Daily_Return",
)

correlation = return_pivot.corr()

fig_corr = go.Figure(
    data=go.Heatmap(
        z=correlation.values,
        x=correlation.columns.tolist(),
        y=correlation.index.tolist(),
        zmin=-1,
        zmax=1,
        colorscale="RdBu",
        reversescale=True,
        text=np.round(correlation.values, 2),
        texttemplate="%{text}",
        hovertemplate=(
            "Stock: %{x}<br>"
            "Compared with: %{y}<br>"
            "Correlation: %{z:.3f}"
            "<extra></extra>"
        ),
        colorbar={"title": "Correlation"},
    )
)

fig_corr.update_layout(
    title="Correlation of Daily Returns",
    template="plotly_dark",
    height=520,
)

st.plotly_chart(fig_corr, use_container_width=True)


# ------------------------------------------------------------
# 11. METRICS TABLE AND DOWNLOAD
# ------------------------------------------------------------

st.divider()
st.subheader("5. Detailed Performance Metrics")

display_summary = summary.copy()

for col in [
    "Period Return",
    "CAGR",
    "Annualized Volatility",
    "Maximum Drawdown",
]:
    display_summary[col] = display_summary[col].map(
        lambda value: f"{value:.2%}" if pd.notna(value) else "N/A"
    )

display_summary["Sharpe Ratio"] = display_summary[
    "Sharpe Ratio"
].map(
    lambda value: f"{value:.2f}" if pd.notna(value) else "N/A"
)

st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True,
)

csv_data = summary.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Performance Summary CSV",
    data=csv_data,
    file_name="dashboard_performance_summary.csv",
    mime="text/csv",
)

st.caption(
    "Data source: your cleaned historical price dataset. "
    "Historical results are not predictions of future returns. "
    "The Sharpe ratio assumes a zero risk-free rate. "
    "CAGR is annualized from the selected period's endpoints."
)

