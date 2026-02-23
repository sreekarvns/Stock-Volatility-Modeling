import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_volatility_with_ci(garch_model, log_returns):
    """
    Plots realized volatility and predicted GARCH volatility with 95% confidence intervals.
    Args:
        garch_model: Fitted GARCH(1,1) model
        log_returns (pd.Series): Actual log-returns
    """
    # Extract predicted volatility
    pred_vol = pd.Series(np.sqrt(garch_model.conditional_volatility), index=garch_model.conditional_volatility.index)
    realized_vol = log_returns.rolling(window=20).std()
    # 95% CI for volatility forecast (approximate, assuming normality)
    # Standard error for volatility: sigma / sqrt(2T), T=20 for rolling window
    T = 20
    se = pred_vol / np.sqrt(2 * T)
    ci_upper = pred_vol + norm.ppf(0.975) * se
    ci_lower = pred_vol - norm.ppf(0.975) * se
    # Align for plotting
    aligned = pd.concat([realized_vol, pred_vol, ci_lower, ci_upper], axis=1, join='inner')
    aligned.columns = ['realized', 'predicted', 'ci_lower', 'ci_upper']
    plt.figure(figsize=(14, 6))
    plt.plot(aligned.index, aligned['realized'], label='Realized Volatility (20d)', color='black', linewidth=2)
    plt.plot(aligned.index, aligned['predicted'], label='Predicted Volatility (GARCH)', color='blue')
    plt.fill_between(aligned.index, aligned['ci_lower'], aligned['ci_upper'], color='blue', alpha=0.2, label='95% CI')
    plt.title('Predicted vs Realized Volatility with 95% CI')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.tight_layout()
    plt.show()
