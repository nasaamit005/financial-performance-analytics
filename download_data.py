from pathlib import Path
import pandas as pd
import yfinance as yf

# Stocks and market benchmark
TICKERS = ["AMZN", "AAPL", "MSFT", "GOOGL", "NVDA", "^GSPC"]

START_DATE = "2016-01-01"
END_DATE = "2026-10-16"

# Save CSV files inside data/raw
OUT_DIR = Path("data") / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

all_data = []

for ticker in TICKERS:
    print(f"Downloading {ticker}...")

    try:
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df.empty:
            print(f"No data returned for {ticker}")
            continue

        # Handle Yahoo Finance column formats
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df.insert(1, "Ticker", ticker)
        df.columns = [str(c).replace(" ", "_") for c in df.columns]

        filename = ticker.replace("^", "") + "_daily.csv"
        output_file = OUT_DIR / filename
        df.to_csv(output_file, index=False)

        all_data.append(df)
        print(f"Saved: {output_file} ({len(df)} rows)")

    except Exception as error:
        print(f"Error downloading {ticker}: {error}")

if all_data:
    combined = pd.concat(all_data, ignore_index=True)
    combined.to_csv(OUT_DIR / "all_tickers_daily.csv", index=False)
    print("\nAll downloads completed.")
    print(f"Combined dataset: {OUT_DIR / 'all_tickers_daily.csv'}")
else:
    print("\nNo data downloaded. Check your internet connection.")
