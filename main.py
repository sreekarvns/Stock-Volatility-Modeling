
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance
import arch
import hmmlearn
import statsmodels

from plot_acf_analysis import plot_acf_analysis
from fit_volatility_models import fit_volatility_models
from evaluate_and_plot_volatility import evaluate_models
from detect_market_regimes import detect_market_regimes
from plot_volatility_with_ci import plot_volatility_with_ci

def load_csv_data(csv_file_path):
    """
    Loads OHLC stock data from a CSV file, validates, and prepares returns.
    Returns a clean DataFrame with Date, Open, High, Low, Close, Simple_Returns, Log_Returns.
    """
    try:
        df = pd.read_csv(csv_file_path)
        # Parse Date column
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # Drop rows with invalid dates
        df = df.dropna(subset=['Date'])
        # Sort by date ascending
        df = df.sort_values('Date').reset_index(drop=True)
        # Remove duplicates
        df = df.drop_duplicates(subset=['Date'])
        # Extract OHLC columns
        ohlc_cols = ['Date', 'Open', 'High', 'Low', 'Close']
        df = df[ohlc_cols]
        # Check for missing values
        df = df.dropna()
        # Calculate returns
        df['Simple_Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        # Remove rows with NaN returns
        df = df.dropna(subset=['Simple_Returns', 'Log_Returns'])
        # Data quality report
        print(f"Data loaded from {csv_file_path}")
        print(f"Rows: {len(df)}")
        print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Duplicate dates: {df.duplicated(subset=['Date']).sum()}")
        if len(df) < 2000:
            print("Warning: Less than 2000 trading days. Statistical validity may be affected.")
        return df
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return pd.DataFrame()

def main():
    # Specify CSV file path
    csv_file_path = 'INFY.csv'  # Change to 'RELIANCE.csv' if desired

    # Load and validate data
    df = load_csv_data(csv_file_path)
    if df.empty:
        print("No data to process. Exiting.")
        return

    # Plot ACF analysis
    plot_acf_analysis(df['Log_Returns'])

    # Fit volatility models
    models = fit_volatility_models(df['Log_Returns'])

    # Evaluate and plot volatility models with full diagnostics (use 20-day window for realized volatility)
    metrics = evaluate_models(models, df['Log_Returns'], realized_window=20)

    # Print Hybrid GARCH-LSTM metrics and summary to terminal
    if metrics and 'Hybrid GARCH-LSTM' in metrics:
        print("\n=== HYBRID GARCH-LSTM PERFORMANCE ===")
        for k, v in metrics['Hybrid GARCH-LSTM'].items():
            print(f"{k.upper()}: {v:.4f}" if isinstance(v, float) else f"{k.upper()}: {v}")
        print("\nHybrid GARCH-LSTM metrics are also saved in the output summary table and visualizations.")
    else:
        print("\nHybrid GARCH-LSTM metrics not found in evaluation results.")

    # Detect market regimes with HMM
    hmm_model, states = detect_market_regimes(df['Log_Returns'])

    # Plot volatility with confidence intervals (GARCH(1,1))
    if 'GARCH(1,1)' in models:
        plot_volatility_with_ci(models['GARCH(1,1)'], df['Log_Returns'])
    else:
        print("GARCH(1,1) model not found for CI plot.")

    # Hybrid GARCH-LSTM metrics and comparison are now handled in evaluate_and_plot_volatility.py

if __name__ == "__main__":
    main()
