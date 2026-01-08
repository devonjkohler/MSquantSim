from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from table_evaluator import TableEvaluator
from joblib import Parallel, delayed

def plot_corr(
    df: pd.DataFrame,
    size: float = 10,
    font_scale: float = 1.0,
    method: str = "pearson",
    cmap: str = "RdBu_r",
    annotate: bool = False,
    annot_fmt: str = ".2f",
):
    """
    Professional correlation heatmap (lower triangle), colorblind-friendly.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (non-numeric columns are ignored).
    size : float
        Figure width/height in inches.
    font_scale : float
        Scales all text sizes.
    method : {"pearson", "spearman", "kendall"}
        Correlation method.
    cmap : str
        Matplotlib diverging colormap.
    annotate : bool
        If True, write correlation values (can get busy for many features).
    annot_fmt : str
        Format string for annotations.

    Returns
    -------
    fig, ax
    """
    # Keep numeric columns only
    x = df.select_dtypes(include=[np.number]).copy()
    if x.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns to plot.")

    corr = x.corr(method=method)

    # Mask upper triangle + diagonal for a cleaner plot
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_plot = corr.mask(mask)

    # Figure + axes
    base_font = 10 * font_scale
    fig, ax = plt.subplots(figsize=(size, size), dpi=150)

    im = ax.imshow(
        corr_plot.to_numpy(),
        vmin=-1,
        vmax=1,
        cmap=cmap,
        interpolation="nearest",
    )

    # Ticks + labels
    labels = corr.columns.to_list()
    n = len(labels)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=base_font)
    ax.set_yticklabels(labels, fontsize=base_font)

    # Put x labels on top (common for correlation matrices)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)

    # Subtle grid to separate cells
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Optional annotations (only for visible lower triangle)
    if annotate:
        for i in range(n):
            for j in range(n):
                val = corr_plot.iat[i, j]
                if pd.notna(val):
                    ax.text(
                        j,
                        i,
                        format(val, annot_fmt),
                        ha="center",
                        va="center",
                        fontsize=0.85 * base_font,
                    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=0.95 * base_font)
    cbar.set_label("Correlation", fontsize=base_font)

    ax.set_title(
        f"Correlation matrix ({method})",
        fontsize=1.2 * base_font,
        pad=12,
    )

    fig.tight_layout()
    return fig, ax

def calculate_similarity_score(
        simulated_data,
        real_data,
        group_col='Condition'
):
    """
    Calculate a similarity score comparing simulated and real tabular data.

    This is a thin wrapper around a TableEvaluator that computes how closely
    simulated_data matches real_data with respect to a grouping/target column.

    Args:
        simulated_data (pandas.DataFrame): Simulated dataset to evaluate.
        real_data (pandas.DataFrame): Real/ground-truth dataset to compare against.
        group_col (str, optional): Name of the categorical/target column used by
            the TableEvaluator for grouping or conditional evaluation. Defaults to
            'Condition'.

    Returns:
        Any: The raw output returned by TableEvaluator.evaluate(..., return_outputs=True).
        The exact structure depends on the TableEvaluator implementation (commonly
        a dict or object containing numeric similarity metrics, per-column reports,
        and any diagnostic outputs).

    Raises:
        ValueError: If group_col is not present in either simulated_data or real_data.
        NameError/ImportError: If TableEvaluator is not available in the runtime
            (i.e., not imported or defined).

    Example:
        >>> # simulated_df and real_df are pandas.DataFrame instances and both contain
        >>> # a column named 'Condition'
        >>> results = calculate_similarity_score(simulated_df, real_df, group_col='Condition')
    """

    # Initialize TableEvaluator
    table_evaluator = TableEvaluator(real_data, simulated_data, 
                                     cat_cols=[group_col])

    # Run evaluation
    similarity_results = table_evaluator.evaluate(
        target_col=group_col, notebook = False, 
        verbose = False, return_outputs = True)

    return similarity_results

def _evaluate_single_sim_pair(
    healthy_df,
    disease_df,
    real_data,
    condition_column,
    condition_label_healthy,
    condition_label_disease,
):
    """
    Evaluate one pair of simulated DataFrames and return a flat dict of metric->value.
    """
    h = healthy_df.copy()
    d = disease_df.copy()

    # Replace inf/-inf with NaN and fill NaNs with column mean (numeric columns only)
    for df in (h, d):
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            # Convert infinities to NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # Compute mean excluding NaNs
            mean_val = df[col].mean(skipna=True)
            if pd.notna(mean_val):
                df[col].fillna(mean_val, inplace=True)
    
    h.loc[:, condition_column] = condition_label_healthy
    d.loc[:, condition_column] = condition_label_disease

    merged_sim_data = pd.concat([h, d], axis=0, ignore_index=True)

    res = calculate_similarity_score(merged_sim_data, real_data, group_col=condition_column)
    res = res["Overview Results"]
    
    flat = {}
    for key, value in res.items():
        if isinstance(value, dict) and "result" in value:
            flat[key] = value["result"]
        else:
            flat[key] = value
    
    return flat

def similarity_across_simulations(sim_data_healthy, 
                                  sim_data_disease,
                                  real_data,
                                  condition_column,
                                  condition_label_healthy,
                                  condition_label_disease):
    
    """
    Compute and aggregate similarity scores between simulated and real datasets across multiple simulation replicates.
    Parallelized using joblib.
    """
    # Basic checks
    if len(sim_data_healthy) != len(sim_data_disease):
        raise ValueError("sim_data_healthy and sim_data_disease must have the same length.")
    if len(sim_data_healthy) == 0:
        return {}

    # Parallel evaluation across replicates
    results_list = Parallel(n_jobs=-1)(
        delayed(_evaluate_single_sim_pair)(
            sim_data_healthy[i],
            sim_data_disease[i],
            real_data,
            condition_column,
            condition_label_healthy,
            condition_label_disease,
        ) for i in tqdm(range(len(sim_data_healthy)), desc="Evaluating simulations")
    )

    # Aggregate results: key -> list of values across replicates
    all_sim_results = {}
    for res in results_list:
        for key, val in res.items():
            all_sim_results.setdefault(key, []).append(val)

    return all_sim_results