Financial Performance Analytics & Forecasting

Overview

A Python-based financial analytics project that evaluates historical stock performance, measures investment risk, compares portfolio strategies against the S&P 500, and evaluates baseline price forecasts.

The project analyzes Apple (AAPL), Amazon (AMZN), Alphabet (GOOGL), Microsoft (MSFT), NVIDIA (NVDA), and the S&P 500 benchmark (^GSPC).

Analysis period: January 4, 2016 – October 14, 2026.

Objectives

Measure historical returns and compound annual growth rates (CAGR).

Compare volatility, Sharpe ratio, Sortino ratio, and maximum drawdown.

Calculate historical Value at Risk (VaR) and Conditional VaR (CVaR).

Compare an equal-weighted portfolio with the S&P 500.

Explore minimum-volatility and maximum-Sharpe portfolio strategies.

Evaluate naive and drift forecasting baselines using MAE and RMSE.

Present findings through interactive Plotly dashboards and business insights.

Technology Stack

Python

pandas and NumPy

Matplotlib and Seaborn

Plotly

scikit-learn and statsmodels

SQL and SQLAlchemy

Jupyter Notebook

pytest

Project Structure

financial-performance-analytics/
├── data/
│   ├── raw/
│   └── processed/
├── dashboard/
├── notebooks/
├── reports/
│   ├── charts/
│   └── business_insights.md
├── sql/
├── src/
├── tests/
├── .gitignore
├── requirements.txt
└── README.md

Setup

1. Clone the repository

git clone https://github.com/nasaamit005/nasaamit005.git
cd nasaamit005

2. Create a virtual environment

Windows PowerShell:

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

3. Install dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

4. Run data-quality tests

python -m pytest tests -v

Key Deliverables

Cleaned historical stock data

Stock performance and risk metrics

Portfolio and benchmark comparisons

Portfolio optimization reports

Forecast evaluation results

Interactive HTML visualizations

Business insights and recommendations

Start with the interactive dashboard:

reports/charts/financial_performance_dashboard.html

Open the HTML file in a modern web browser.

Selected Historical Findings

In the analyzed period, the equal-weighted portfolio reported a CAGR of approximately 35.18%, compared with approximately 13.31% for the S&P 500. The equal-weighted portfolio also exhibited higher annualized volatility: approximately 27.20%, compared with 17.80% for the benchmark.

The drift forecasting baseline achieved lower MAE and RMSE than the naive baseline for all six analyzed series in the reported test period.

These are historical results, not predictions or guarantees of future performance. Portfolio optimization results require additional out-of-sample validation before they can support investment decisions.

Limitations

The analysis focuses on five technology stocks and one benchmark.

Historical results may not represent future market conditions.

Transaction costs, taxes, slippage, and rebalancing assumptions may affect implementable returns.

Forecasting and optimization results require rigorous out-of-sample testing.

Financial analysis is for educational and analytical purposes, not personalized investment advice.

Author

GitHub: @nasaamit005