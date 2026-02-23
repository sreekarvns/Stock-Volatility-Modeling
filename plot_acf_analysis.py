import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd

def plot_acf_analysis(returns):
    """
    Plots ACF for returns and squared returns side by side.
    Args:
        returns (pd.Series): Series of returns (simple or log)
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("Input must be a pandas Series.")
    squared_returns = returns ** 2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(returns.dropna(), lags=40, ax=axes[0])
    axes[0].set_title('ACF of Returns')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('Autocorrelation')
    plot_acf(squared_returns.dropna(), lags=40, ax=axes[1])
    axes[1].set_title('ACF of Squared Returns')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('Autocorrelation')
    plt.tight_layout()
    # Save to output directory
    import os
    output_dir = os.path.abspath(os.path.join(os.getcwd(), 'output'))
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'acf_plot.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"ACF plot saved: {save_path}")
    plt.show()
