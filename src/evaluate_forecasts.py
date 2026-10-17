
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "forecast_metrics.csv"
OUTPUT_FILE = ROOT / "reports" / "forecast_evaluation.csv"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Run forecasting.py first. Missing: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {"Ticker", "Model", "MAE", "RMSE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in ["MAE", "RMSE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["MAE", "RMSE"]].isna().any().any():
        raise ValueError("MAE or RMSE contains missing/non-numeric values.")

    if (df[["MAE", "RMSE"]] < 0).any().any():
        raise ValueError("Forecast errors cannot be negative.")

    mae = df.pivot(index="Ticker", columns="Model", values="MAE")
    rmse = df.pivot(index="Ticker", columns="Model", values="RMSE")

    results = pd.DataFrame(index=mae.index)
    results["Naive_MAE"] = mae["Naive"]
    results["Drift_MAE"] = mae["Drift"]
    results["Naive_RMSE"] = rmse["Naive"]
    results["Drift_RMSE"] = rmse["Drift"]

    results["MAE_Improvement_Pct"] = (
        (results["Naive_MAE"] - results["Drift_MAE"])
        / results["Naive_MAE"] * 100
    )

    results["RMSE_Improvement_Pct"] = (
        (results["Naive_RMSE"] - results["Drift_RMSE"])
        / results["Naive_RMSE"] * 100
    )

    results["Best_Baseline_By_RMSE"] = np.where(
        results["Drift_RMSE"] < results["Naive_RMSE"],
        "Drift",
        "Naive",
    )

    results = results.reset_index()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print("\nForecast evaluation summary:")
    print(results.round(2).to_string(index=False))
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
