from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
from sklearn.decomposition import PCA


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
        csv_path1,
        csv_path2,
        reference_data=None,
        cat_cols=['Condition'],
        target_col='Condition',
        condition_mapping={'left_only': 0, 'right_only': 1, 'both': 2},
        plot_settings=None
):
    """
    Load two CSV datasets, merge them, and evaluate using TableEvaluator.

    Parameters:
    -----------
    csv_path1 : str
        Path to the first CSV file
    csv_path2 : str
        Path to the second CSV file
    reference_data : pandas DataFrame, optional
        Reference data to compare with the merged dataset. If None,
        the merged dataset will be used as reference data.
    cat_cols : list, default ['Condition']
        List of categorical columns for TableEvaluator
    target_col : str, default 'Condition'
        Target column for evaluation
    condition_mapping : dict, default {'left_only': 0, 'right_only': 1, 'both': 2}
        Mapping for the _merge indicator to Condition values
    plot_settings : dict, optional
        Dictionary of settings for plots in TableEvaluator (e.g., {'figsize': (10, 5)})

    Returns:
    --------
    table_evaluator : TableEvaluator
        The initialized TableEvaluator object after running evaluate()
    merged_df : pandas DataFrame
        The merged dataframe with Condition column
    """
    # Print information about files being loaded
    print(f"Loading datasets:\n - {os.path.basename(csv_path1)}\n - {os.path.basename(csv_path2)}")

    # Load the CSV files
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)

    print(f"Loaded shapes: {df1.shape}, {df2.shape}")

    # Merge datasets
    merged_df = pd.merge(df1, df2, how='outer', indicator=True)

    # Create Condition column based on the merge indicator
    merged_df['Condition'] = merged_df['_merge'].map(condition_mapping)

    # Drop the _merge column
    merged_df = merged_df.drop('_merge', axis=1)

    # If reference_data is None, use merged_df as both datasets
    if reference_data is None:
        reference_data = merged_df.copy()

    # Print information about merged data
    condition_counts = merged_df['Condition'].value_counts().to_dict()
    print(f"Merged data shape: {merged_df.shape}")
    print(f"Condition counts: {condition_counts}")

    # Initialize TableEvaluator
    table_evaluator = TableEvaluator(reference_data, merged_df, cat_cols=cat_cols)

    # Configure plot settings if provided
    if plot_settings:
        # Apply plot settings (assuming TableEvaluator has methods to set these)
        if 'figsize' in plot_settings:
            # This would depend on how TableEvaluator implements figure size settings
            # You may need to adjust based on actual TableEvaluator API
            plt.rcParams['figure.figsize'] = plot_settings['figsize']

        # Add other plot settings as needed

    # Run evaluation
    table_evaluator.evaluate(target_col=target_col)

    # Return the table_evaluator and merged_df for further analysis if needed
    return table_evaluator, merged_df