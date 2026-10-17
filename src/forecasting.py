
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "all_tickers_clean.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

TEST_SIZE = 0.20


def forecast_naive(train, steps):
    """Predict every future price using the last training price."""
    return np.repeat(float(train.iloc[-1]), steps)


def forecast_drift(train, steps):
    """Extend the average price change observed in training."""
    if len(train) < 2:
        return forecast_naive(train, steps)

    drift = (float(train.iloc[-1]) - float(train.iloc[0])) / (
        len(train) - 1
    )
    return float(train.iloc[-1]) + drift * np.arange(1, steps + 1)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"])

    required = {"Date", "Ticker"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns: {required}"
        )

    price_col = (
        "Adj_Close" if "Adj_Close" in df.columns else "Close"
    )

    if price_col not in df.columns:
        raise ValueError("No Adj_Close or Close price column found.")

    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["Date", "Ticker", price_col])
    df = df[df[price_col] > 0]
    df = df.sort_values(["Ticker", "Date"])

    predictions = []
    metrics = []

    for ticker, group in df.groupby("Ticker"):
        group = (
            group.drop_duplicates(subset=["Date"], keep="last")
            .sort_values("Date")
            .reset_index(drop=True)
        )

        prices = group[price_col]
        split = int(len(prices) * (1 - TEST_SIZE))

        if split < 2 or len(prices) - split < 2:
            print(f"Skipping {ticker}: insufficient observations.")
            continue

        train = prices.iloc[:split]
        test = prices.iloc[split:]

        models = {
            "Naive": forecast_naive(train, len(test)),
            "Drift": forecast_drift(train, len(test)),
        }

        for model_name, forecast in models.items():
            actual = test.to_numpy()

            mae = mean_absolute_error(actual, forecast)
            rmse = np.sqrt(mean_squared_error(actual, forecast))

            metrics.append({
                "Ticker": ticker,
                "Model": model_name,
                "Train_End": group.loc[split - 1, "Date"],
                "Test_Start": group.loc[split, "Date"],
                "Test_End": group.loc[len(group) - 1, "Date"],
                "Train_Rows": len(train),
                "Test_Rows": len(test),
                "MAE": mae,
                "RMSE": rmse,
            })

            for date, actual_price, predicted_price in zip(
                group["Date"].iloc[split:],
                actual,
                forecast,
            ):
                predictions.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Model": model_name,
                    "Actual_Price": actual_price,
                    "Predicted_Price": predicted_price,
                })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.DataFrame(predictions)

    metrics_df.to_csv(
        OUTPUT_DIR / "forecast_metrics.csv", index=False
    )
    predictions_df.to_csv(
        OUTPUT_DIR / "forecast_backtest.csv", index=False
    )

    print("\nForecast evaluation:")
    if not metrics_df.empty:
        print(
            metrics_df.sort_values(["Ticker", "RMSE"]).to_string(
                index=False
            )
        )
        print("\nSaved:")
        print(OUTPUT_DIR / "forecast_metrics.csv")
        print(OUTPUT_DIR / "forecast_backtest.csv")
    else:
        print("No forecasts were generated.")


if __name__ == "__main__":
    main()
