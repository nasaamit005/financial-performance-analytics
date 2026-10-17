
from pathlib import Path

import pandas as pd
import pytest


# Project root is the parent of the tests folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "all_tickers_clean.csv"


@pytest.fixture(scope="module")
def stock_data():
    """Load the cleaned stock dataset once for the test module."""
    assert DATA_FILE.exists(), f"Cleaned dataset not found: {DATA_FILE}"
    return pd.read_csv(DATA_FILE)


def test_dataset_is_not_empty(stock_data):
    """The cleaned dataset must contain observations."""
    assert not stock_data.empty


def test_required_columns_exist(stock_data):
    """Essential fields must be present."""
    required_columns = {
        "Date",
        "Ticker",
        "Adj_Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
    }
    assert required_columns.issubset(stock_data.columns)


def test_no_missing_values(stock_data):
    """The cleaned dataset must not contain missing values."""
    assert stock_data.isna().sum().sum() == 0


def test_no_duplicate_rows(stock_data):
    """Entire rows must be unique."""
    assert not stock_data.duplicated().any()


def test_unique_ticker_date_pairs(stock_data):
    """Each ticker should have no more than one record per date."""
    assert not stock_data.duplicated(["Date", "Ticker"]).any()


def test_prices_are_positive(stock_data):
    """Available price fields must contain positive values."""
    price_columns = ["Adj_Close", "Close", "High", "Low", "Open"]

    for column in price_columns:
        assert (stock_data[column] > 0).all(), (
            f"Non-positive values found in {column}"
        )


def test_dates_are_valid(stock_data):
    """Dates must be parseable and span a valid range."""
    dates = pd.to_datetime(stock_data["Date"], errors="coerce")

    assert dates.notna().all()
    assert dates.min() <= dates.max()


def test_expected_tickers_exist(stock_data):
    """All six expected assets must be represented."""
    expected_tickers = {"AAPL", "AMZN", "GOOGL", "MSFT", "NVDA", "^GSPC"}
    actual_tickers = set(stock_data["Ticker"].unique())

    assert expected_tickers.issubset(actual_tickers)
