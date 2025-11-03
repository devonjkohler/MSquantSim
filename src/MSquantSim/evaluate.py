import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from table_evaluator import TableEvaluator
import pandas as pd
import os

def plot_corr(df, size=10,font_scale=0.7):
    '''Plot a graphical correlation matrix for a dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    plt.rcParams.update({'font.size': 8 * font_scale})

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Compute the correlation matrix for the received dataframe
    corr = df.corr()
    corr = corr[corr <= 0.99]
    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr, cmap=cmap)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6 * font_scale)
    ax.set_yticklabels(corr.columns, fontsize=6 * font_scale)
    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)
    cbar.ax.tick_params(labelsize=6 * font_scale)

    # Adjust layout to make room for labels
    plt.tight_layout()

    return

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
