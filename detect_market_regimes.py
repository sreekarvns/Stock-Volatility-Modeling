import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

def detect_market_regimes(log_returns):
    """
    Fits a 2-state Gaussian HMM to log-returns and visualizes regimes.
    Args:
        log_returns (pd.Series): Series of log-returns
    Returns:
        model: Fitted HMM model
        states: Decoded hidden states
    """
    if not isinstance(log_returns, pd.Series):
        raise ValueError("log_returns must be a pandas Series.")
    X = log_returns.dropna().values.reshape(-1, 1)
    dates = log_returns.dropna().index
    try:
        model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(X)
        states = model.predict(X)
        # Print transition matrix and means
        print("Transition matrix:\n", model.transmat_)
        print("Means for each state:", model.means_.flatten())
        # Plot
        plt.figure(figsize=(14, 6))
        plt.plot(dates, X.flatten(), label='Log-Returns', color='black')
        for state in [0, 1]:
            mask = (states == state)
            plt.fill_between(dates, X.flatten().min(), X.flatten().max(), where=mask, alpha=0.2 if state == 0 else 0.4,
                             color='blue' if state == 0 else 'red',
                             label=f"State {state} ({'Calm' if state == 0 else 'Volatile'})")
        plt.title('Market Regimes Detected by HMM')
        plt.xlabel('Date')
        plt.ylabel('Log-Returns')
        plt.legend()
        plt.tight_layout()
        # Save to output directory
        import os
        output_dir = os.path.abspath(os.path.join(os.getcwd(), 'output'))
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'hmm_regimes.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"HMM regime plot saved: {save_path}")
        plt.show()
        return model, states
    except Exception as e:
        print(f"Error fitting HMM: {e}")
        return None, None
