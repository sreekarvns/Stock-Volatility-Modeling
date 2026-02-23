"""
create_hybrid_visualizations.py

Contains functions to generate publication-quality visualizations and a summary table
for comparing volatility models, with a focus on demonstrating Hybrid GARCH-LSTM superiority.

Visualization functions:
1. plot_metric_barh: Horizontal bar chart of all models by main metric (e.g., RMSE)
2. plot_volatility_timeseries: Time series plot of realized vs. predicted volatility (all models)
3. plot_grouped_metric_bar: Grouped bar chart of all metrics for all models
4. plot_scatter_grid: Grid of scatter plots (predicted vs. realized volatility, all models)
5. plot_metric_box_violin: Box/violin plot of error distributions (all models)
6. plot_summary_table: Styled summary table highlighting best model(s)

All functions accept model results in a standard format (e.g., DataFrame of metrics, dict of predictions).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. Horizontal bar chart of main metric (e.g., RMSE)
def plot_metric_barh(metrics_df, metric='RMSE', highlight_model='Hybrid GARCH-LSTM', ax=None, save_path=None):
    """
    Plots a horizontal bar chart of the main metric for all models.
    Args:
        metrics_df: DataFrame with model names as index and metrics as columns
        metric: Metric to plot (e.g., 'RMSE')
        highlight_model: Model to highlight (e.g., 'Hybrid GARCH-LSTM')
        ax: Optional matplotlib axis
        save_path: If provided, saves the figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    order = metrics_df[metric].sort_values().index
    colors = ["#FFD700" if m == highlight_model else "#4682B4" for m in order]
    sns.barplot(
        y=metrics_df.loc[order].index,
        x=metrics_df.loc[order, metric],
        palette=colors,
        ax=ax
    )
    ax.set_xlabel(metric)
    ax.set_ylabel('Model')
    ax.set_title(f'{metric} Comparison Across Models')
    for i, v in enumerate(metrics_df.loc[order, metric]):
        ax.text(v, i, f'{v:.3f}', va='center', ha='left', fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return ax

# 2. Time series plot of realized vs. predicted volatility (all models)
def plot_volatility_timeseries(realized, predictions_dict, highlight_model='Hybrid GARCH-LSTM', ax=None, save_path=None):
    """
    Plots time series of realized volatility and all model predictions.
    Args:
        realized: pd.Series of realized volatility (index: datetime)
        predictions_dict: dict {model_name: pd.Series}
        highlight_model: Model to highlight
        ax: Optional matplotlib axis
        save_path: If provided, saves the figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure
    ax.plot(realized.index, realized, label='Realized', color='black', linewidth=2)
    for model, preds in predictions_dict.items():
        if model == highlight_model:
            ax.plot(preds.index, preds, label=model, linewidth=2.5, linestyle='-', color='#FFD700')
        else:
            ax.plot(preds.index, preds, label=model, linewidth=1.5, linestyle='--', alpha=0.7)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility')
    ax.set_title('Volatility Forecasts vs. Realized')
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return ax

# 3. Grouped bar chart of all metrics for all models
def plot_grouped_metric_bar(metrics_df, metrics=None, highlight_model='Hybrid GARCH-LSTM', ax=None, save_path=None):
    """
    Plots grouped bar chart of all metrics for all models.
    Args:
        metrics_df: DataFrame with model names as index and metrics as columns
        metrics: List of metrics to plot (default: all columns)
        highlight_model: Model to highlight
        ax: Optional matplotlib axis
        save_path: If provided, saves the figure
    """
    if metrics is None:
        metrics = metrics_df.columns.tolist()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    width = 0.8 / len(metrics)
    x = np.arange(len(metrics_df))
    for i, metric in enumerate(metrics):
        color = '#FFD700' if metric == highlight_model else None
        ax.bar(x + i * width, metrics_df[metric], width=width, label=metric)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(metrics_df.index, rotation=30, ha='right')
    ax.set_ylabel('Metric Value')
    ax.set_title('Model Comparison Across Metrics')
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return ax

# 4. Grid of scatter plots (predicted vs. realized volatility, all models)
def plot_scatter_grid(realized, predictions_dict, highlight_model='Hybrid GARCH-LSTM', save_path=None):
    """
    Plots a grid of scatter plots for each model: predicted vs. realized volatility.
    Args:
        realized: pd.Series of realized volatility
        predictions_dict: dict {model_name: pd.Series}
        highlight_model: Model to highlight
        save_path: If provided, saves the figure
    """
    n_models = len(predictions_dict)
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()
    for i, (model, preds) in enumerate(predictions_dict.items()):
        ax = axes[i]
        color = '#FFD700' if model == highlight_model else '#4682B4'
        ax.scatter(realized, preds, alpha=0.7, color=color, edgecolor='k', s=30)
        ax.plot([realized.min(), realized.max()], [realized.min(), realized.max()], 'r--', lw=1)
        ax.set_title(model)
        ax.set_xlabel('Realized')
        ax.set_ylabel('Predicted')
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return axes

# 5. Box/violin plot of error distributions (all models)
def plot_metric_box_violin(errors_dict, plot_type='box', highlight_model='Hybrid GARCH-LSTM', ax=None, save_path=None):
    """
    Plots box or violin plot of error distributions for all models.
    Args:
        errors_dict: dict {model_name: np.array or pd.Series of errors}
        plot_type: 'box' or 'violin'
        highlight_model: Model to highlight
        ax: Optional matplotlib axis
        save_path: If provided, saves the figure
    """
    df = pd.DataFrame(errors_dict)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure
    if plot_type == 'box':
        sns.boxplot(data=df, ax=ax, palette=["#FFD700" if c == highlight_model else "#4682B4" for c in df.columns])
    else:
        sns.violinplot(data=df, ax=ax, palette=["#FFD700" if c == highlight_model else "#4682B4" for c in df.columns])
    ax.set_title('Error Distribution Across Models')
    ax.set_ylabel('Error')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return ax

# 6. Styled summary table highlighting best model(s)
def plot_summary_table(metrics_df, main_metric='RMSE', highlight_model='Hybrid GARCH-LSTM', save_path=None):
    """
    Plots a styled summary table of all metrics, highlighting the best model(s).
    Args:
        metrics_df: DataFrame with model names as index and metrics as columns
        main_metric: Metric to highlight best model (e.g., 'RMSE')
        highlight_model: Model to highlight
        save_path: If provided, saves the figure
    """
    styled = metrics_df.style.background_gradient(axis=0, cmap='YlGn').highlight_max(axis=0, color='#FFD700')\
        .apply(lambda s: ['font-weight: bold; color: #FFD700' if v == highlight_model else '' for v in s.index], axis=0)
    if save_path:
        # Save as image using dataframe_image if available
        try:
            import dataframe_image as dfi
            dfi.export(styled, save_path)
        except ImportError:
            print('Install dataframe_image to save styled tables as images.')
    return styled
