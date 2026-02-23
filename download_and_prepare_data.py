import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def download_and_prepare_data(ticker='NIFTY-50', years=10):
    """
    Downloads daily OHLC data for the given ticker using yfinance,
    calculates simple and log returns, and returns a clean DataFrame.
    Args:
        ticker (str): Ticker symbol (default 'NIFTY-50' or '^NSEI')
        years (int): Number of years of data to download (default 10)
    Returns:
        pd.DataFrame: DataFrame with Date, Close, Simple_Returns, Log_Returns
    """
    # Map 'NIFTY-50' to Yahoo Finance ticker
    if ticker.upper() in ['NIFTY-50', 'NIFTY50', 'NIFTY']:
        yf_ticker = '^NSEI'
    else:
        yf_ticker = ticker

    end_date = datetime.today()
    start_date = end_date - timedelta(days=years*365)

    try:
        df = yf.download(yf_ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1d')
        if df.empty:
            raise ValueError(f"No data found for ticker {yf_ticker}.")
        df = df.reset_index()
        df = df[['Date', 'Close']].dropna()
        df['Simple_Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df = df.dropna(subset=['Simple_Returns', 'Log_Returns'])
        df = df[['Date', 'Close', 'Simple_Returns', 'Log_Returns']]
        df = df.reset_index(drop=True)
    except Exception as e:
        print(f"Error downloading or processing data: {e}")
        return pd.DataFrame(columns=['Date', 'Close', 'Simple_Returns', 'Log_Returns'])

    return df
