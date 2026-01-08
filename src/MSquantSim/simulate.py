
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib


import pandas as pd
import numpy as np
import os

from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

from copulas.multivariate import GaussianMultivariate, VineCopula

import warnings
warnings.filterwarnings('ignore')

def copula_model(data, samples):
    # Create and fit the Gaussian copula model
    model = GaussianMultivariate()
    model.fit(data)

    # Sample from the model
    synthetic_samples = model.sample(samples)
    
    return synthetic_samples

def tvae_model(data, metadata, samples):
    # Create and fit the TVAE model
    model = TVAESynthesizer(metadata)
    model.fit(data)
    
    # Sample from the model
    synthetic_samples = model.sample(num_rows=samples)
    
    return synthetic_samples

def per_protein_model(data, means, variances, samples):
    synthetic_samples = pd.DataFrame()

    # Generate data for each protein/column
    for column in data.columns:
        mean = means[column]
        std_dev = np.sqrt(variances[column])
        synthetic_samples[column] = np.random.normal(
            loc=mean, scale=std_dev, size=samples)

    return synthetic_samples

def simulate_using_copula(dataset, num_simulations=1, samples_per_simulation=None,
                          output_dir=None, output_prefix='simulated',
                          return_data=True, random_state=None):
    """
    Simulate protein abundance data using Gaussian Copula models.

    Parameters:
    -----------
    dataset : pandas DataFrame
        Input dataset containing protein abundance data
    num_simulations : int, default 1
        Number of simulation iterations to perform
    samples_per_simulation : int, default None
        Number of samples to generate in each simulation
        If None, uses the same number of samples as the input dataset
    output_dir : str, default None
        Directory to save CSV files. If None, files are not saved
    output_prefix : str, default 'simulated'
        Prefix for output filenames
    return_data : bool, default True
        Whether to return the simulated data
    random_state : int, default None
        Random seed for reproducibility

    Returns:
    --------
    list or None
        If return_data=True: List of pandas DataFrames containing simulated data
        If return_data=False: None
    """

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Default to the same number of samples as the input dataset
    if samples_per_simulation is None:
        samples_per_simulation = len(dataset)
    
    if num_simulations == 1:
        simulation = copula_model(dataset, samples_per_simulation)
    else:
        simulation = Parallel(n_jobs=-2)(
            delayed(copula_model)(dataset, samples_per_simulation)
            for _ in tqdm(range(num_simulations), 
                          desc="Running copula simulations"))
        
    # Create output directory if it doesn't exist
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if num_simulations == 1:
            output_path = os.path.join(output_dir, f"{output_prefix}_0.csv")
            simulation.to_csv(output_path, index=False)
        else:
            # Save each simulated dataset to CSV
            for i in range(num_simulations):
                output_path = os.path.join(output_dir, f"{output_prefix}_{i}.csv")
                simulation[i].to_csv(output_path, index=False)

    if return_data:
        return simulation
  
def simulate_using_tvae(dataset, metadata=None, num_simulations=1,
                        samples_per_simulation=None,
                          output_dir=None, output_prefix='simulated',
                          return_data=True, random_state=None):
    """
    Generate synthetic data using TVAE synthesizer and save to CSV files.

    Parameters:
    -----------
    dataset : pandas DataFrame
        Input dataset containing protein abundance data
    metadata : sdv.metadata.SingleTableMetadata
        The metadata object describing the data structure
    num_simulations : int, default 1
        Number of simulation iterations to perform
    samples_per_simulation : int, default None
        Number of samples to generate in each simulation
        If None, uses the same number of samples as the input dataset
    output_dir : str, default None
        Directory to save CSV files. If None, files are not saved
    output_prefix : str, default 'simulated'
        Prefix for output filenames
    return_data : bool, default True
        Whether to return the simulated data
    random_state : int, default None
        Random seed for reproducibility

    Returns:
    --------
    list or None
        If return_data=True: List of pandas DataFrames containing simulated data
        If return_data=False: None
    """

    if metadata is None:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=dataset)
        
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Default to the same number of samples as the input dataset
    if samples_per_simulation is None:
        samples_per_simulation = len(dataset)
    
    if num_simulations == 1:
        simulation = tvae_model(dataset, metadata, samples_per_simulation)
    else:
        simulation = Parallel(n_jobs=-2)(
            delayed(tvae_model)(dataset, metadata, samples_per_simulation)
            for _ in tqdm(range(num_simulations), 
                          desc="Running tvae simulations"))

    # Create output directory if it doesn't exist
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if num_simulations == 1:
            output_path = os.path.join(output_dir, f"{output_prefix}_0.csv")
            simulation.to_csv(output_path, index=False)
        else:
            # Save each simulated dataset to CSV
            for i in range(num_simulations):
                output_path = os.path.join(output_dir, f"{output_prefix}_{i}.csv")
                simulation[i].to_csv(output_path, index=False)

    if return_data:
        return simulation

def simulate_using_per_protein(dataset, num_simulations=1,
                               samples_per_simulation=None,
                               output_dir=None, output_prefix='simulated',
                               return_data=True, random_state=None):
    """
    Simulate synthetic data using per-protein Gaussian distributions.

    This function estimates the mean and variance for each protein (column) in the dataset
    and generates synthetic data by sampling from normal distributions with those parameters.

    Parameters:
    -----------
    dataset : pandas DataFrame
        Input dataset containing protein abundance data
    num_simulations : int, default 1
        Number of simulation iterations to perform
    samples_per_simulation : int, default None
        Number of samples to generate in each simulation
        If None, uses the same number of samples as the input dataset
    output_dir : str, default None
        Directory to save CSV files. If None, files are not saved
    output_prefix : str, default 'simulated'
        Prefix for output filenames
    return_data : bool, default True
        Whether to return the simulated data
    random_state : int, default None
        Random seed for reproducibility

    Returns:
    --------
    list or None
        If return_data=True: List of pandas DataFrames containing simulated data
        If return_data=False: None
    """

    # Estimate the mean and variance for each column (protein)
    means = dataset.mean()
    variances = dataset.var()

    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Default to the same number of samples as the input dataset
    if samples_per_simulation is None:
        samples_per_simulation = len(dataset)
    
    if num_simulations == 1:
        simulation = per_protein_model(dataset, means, variances, samples_per_simulation)
    else:
        simulation = Parallel(n_jobs=-2)(
            delayed(per_protein_model)(dataset, means, variances, samples_per_simulation)
            for _ in tqdm(range(num_simulations), 
                          desc="Running per-protein simulations"))

    # Create output directory if it doesn't exist
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if num_simulations == 1:
            output_path = os.path.join(output_dir, f"{output_prefix}_0.csv")
            simulation.to_csv(output_path, index=False)
        else:
            # Save each simulated dataset to CSV
            for i in range(num_simulations):
                output_path = os.path.join(output_dir, f"{output_prefix}_{i}.csv")
                simulation[i].to_csv(output_path, index=False)

    if return_data:
        return simulation