from pathlib import Path
import pandas as pd

# Project paths
INPUT_FILE = Path("data/raw/all_tickers_daily.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "all_tickers_clean.csv"

def clean_stock_data():
    # 1. Check that the input file exists
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}. "
            "Run the downloader first."
        )

    # 2. Load the raw dataset
    df = pd.read_csv(INPUT_FILE)
    print("=" * 60)
    print("FINANCIAL DATA QUALITY REPORT")
    print("=" * 60)
    print(f"\nOriginal rows: {len(df):,}")
    print(f"Original columns: {df.columns.tolist()}")

    # 3. Standardize column names
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
    )

    # 4. Check required columns
    required = [
        "Date", "Ticker", "Open", "High",
        "Low", "Close", "Adj_Close", "Volume"
    ]
    missing_columns = [
        col for col in required if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # 5. Convert data types
    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    )
    df["Ticker"] = df["Ticker"].astype("string").str.strip()

    numeric_columns = [
        "Open", "High", "Low", "Close",
        "Adj_Close", "Volume"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6. Report missing values before cleaning
    print("\nMissing values before cleaning:")
    print(df.isna().sum().to_string())

    # 7. Remove exact duplicate rows
    exact_duplicates = df.duplicated().sum()
    df = df.drop_duplicates()

    # 8. Remove duplicate ticker-date records
    ticker_date_duplicates = df.duplicated(
        subset=["Date", "Ticker"]
    ).sum()

    df = df.drop_duplicates(
        subset=["Date", "Ticker"], keep="first"
    )

    # 9. Remove rows missing essential identifiers/prices
    before_required_filter = len(df)
    df = df.dropna(
        subset=["Date", "Ticker", "Close"]
    )
    removed_required = before_required_filter - len(df)

    # 10. Flag impossible or invalid market records
    invalid_prices = (
        (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
        | (df["High"] < df["Low"])
        | (df["High"] < df[["Open", "Close"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close"]].min(axis=1))
        | (df["Volume"] < 0)
    )

    invalid_count = int(invalid_prices.sum())
    if invalid_count:
        print(f"\nRemoving invalid market records: {invalid_count}")
        df = df.loc[~invalid_prices].copy()

    # 11. Sort by ticker and date
    df = df.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    # 12. Create output directory and save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # 13. Final report
    print("\nCleaning summary")
    print("-" * 40)
    print(f"Exact duplicate rows removed: {exact_duplicates}")
    print(
        "Duplicate ticker-date rows removed: "
        f"{ticker_date_duplicates}"
    )
    print(f"Rows removed for missing essentials: {removed_required}")
    print(f"Final rows: {len(df):,}")
    print(f"Final missing values:\n{df.isna().sum().to_string()}")
    print(f"\nTickers: {df['Ticker'].unique().tolist()}")
    print(f"Date range: {df['Date'].min().date()} to "
          f"{df['Date'].max().date()}")
    print(f"\nClean dataset saved to: {OUTPUT_FILE}")
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))

    return df


if __name__ == "__main__":
    clean_stock_data()
