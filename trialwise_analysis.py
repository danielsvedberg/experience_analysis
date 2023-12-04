import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import pandas as pd
from scipy.stats import sem
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import random

def model(x, a, b, c):
    return a / (1 + np.exp(-b * x + c))

def generate_synthetic_data(time_steps=30, subjects=3, jitter_strength=0.1):
    """
    Generates synthetic data in a pandas DataFrame in long-form format.

    Args:
    time_steps (int): Number of time steps in the series.
    subjects (int): Number of different subjects/replicates.
    jitter_strength (float): Strength of the random jitter added to each subject's data.

    Returns:
    pandas.DataFrame: A DataFrame containing the synthetic data in long-form.
    """
    # Time points
    t = np.linspace(0, 1, time_steps)
    # Sigmoid function for the base curve
    a = 1
    b = 5
    c = 0.5
    base_curve = model(t, a, b, c)

    # Create an empty list to store data
    data = []

    # Generate data for each subject
    for subject in range(subjects):
        # Add random jitter
        jitter = np.random.normal(0, jitter_strength, time_steps)
        subject_data = np.clip(base_curve + jitter, 0, 1)  # Ensure data is bounded between 0 and 1

        # Append to data list
        for time_step in range(time_steps):
            data.append({
                'Time': time_step,
                'Subject': f'Subject_{subject+1}',
                'Value': subject_data[time_step]
            })

    # Convert to DataFrame
    longform_data = pd.DataFrame(data)

    return longform_data

def shuffle_time(df, subject_cols='Subject', time_col='Time'):
    newdf = []
    for name, group in df.groupby(subject_cols):
        nt = list(group[time_col])
        random.shuffle(nt)
        group[time_col] = nt
        newdf.append(group)
    newdf = pd.concat(newdf)
    return newdf 

def nonlinear_regression(data, subject_cols=['Subject'], time_col='Time', value_col='Value'):
    """
    Performs nonlinear regression on the synthetic data for each subject.

    Args:
    data (pandas.DataFrame): The synthetic data in long-form format.

    Returns:
    dict: A dictionary containing the fitted parameters for each subject.
    """

    # Define the nonlinear model (exponential decay)

    # Initialize a dictionary to store results
    fitted_params = {}
    r2_scores = {}

    # Get unique subjects
    # Fit the model for each subject
    for subject, subject_data in data.groupby(subject_cols):

        # Extract time and values
        time = subject_data[time_col]
        values = subject_data[value_col]

        # Fit the model
        params, _ = curve_fit(model, time, values, bounds=([0,0,0],[1,'inf', 'inf']), maxfev=100000)

        # Store the fitted parameters
        fitted_params[subject] = params
        
        y_pred = model(subject_data[time_col], *params)
        r2 = r2_score(subject_data[value_col], y_pred)
        
        r2_scores[subject] = r2

    return fitted_params, r2_scores

def bootstrap_stats(matrix, n_bootstraps=1000, axis=0):
    # Number of rows and columns in the matrix
    n_rows, n_cols = matrix.shape

    # Initialize arrays to store bootstrap means and standard deviations
    boot_means = np.zeros((n_bootstraps, n_cols))

    # Perform bootstrap resampling
    for i in range(n_bootstraps):
        # Sample with replacement from the rows of the matrix
        sample = matrix[np.random.randint(0, n_rows, size=n_rows)]
        
        # Calculate mean and standard deviation for each column in the sample
        boot_means[i, :] = np.mean(sample, axis= axis)

    bootmean = np.mean(boot_means, axis = axis)
    bootci = np.percentile(boot_means, [2.5, 97.5], axis = axis)
    
    params = {'bootmean': bootmean, 'bootci': bootci}
    return params

"""
def nonlinear_metaregression(data, subject_cols=['Subject'], grouping_cols=['exp_group', 'time_group'], time_col='Time', value_col='Value'):
    fitted_params, r2 = nonlinear_regression(data, subject_cols, time_col, value_col)
    df = pd.Series(fitted_params).reset_index()
    cols = subject_cols + ['params']
    df = df.set_axis(cols, axis=1)

    grpavgdf = df.groupby(grouping_cols).mean().reset_index()

    bootmetaparam= []
    bootmetaci = []
    r2_bootmean = []
    r2_bootci = []
    groups = []
    for i, group in df.groupby(grouping_cols):
        matrix = np.array(list(group.params))
        metaparams = bootstrap_stats(matrix, n_bootstraps=1000)
        r2arr = np.array(list(r2.values()))
        r2arr = r2arr.reshape(-1,1)
        meanr2 = bootstrap_stats(r2arr, axis=0)

        groups.append(i)
        bootmetaparam.append(metaparams['bootmean'])
        bootmetaci.append(metaparams['bootci'])
        r2_bootmean.append(meanr2['bootmean'])
        r2_bootci.append(meanr2['bootci'])

    metadf = pd.DataFrame({'group': groups, 'bootmetaparam':bootmetaparam, 'bootmetaci': bootmetaci, 'r2_bootmean': r2_bootmean, 'r2_bootci': r2_bootci})
    metadf = metadf.set_index('group')

    y_pred = model(data[time_col], *metaparams['bootmean'])
    metar2 = r2_score(data[value_col], y_pred)
    
    return metaparams, metar2
"""

def generate_metafit(metaparams, time_steps = 30):
    t = np.linspace(0, time_steps-1, time_steps)

    # Create an empty list to store data
    data = []
    
    meanparams = metaparams['bootmean']
    matrix = metaparams['bootci']
    
    # Generate all combinations of column entries
    transposed_matrix = matrix.T
    column_combinations = list(product(*transposed_matrix))
    
    res = []
    for row in column_combinations:
        ci_data = model(t, *row)
        res.append(ci_data)
    res = np.vstack(res)
    est_high = np.max(res,axis=0)
    est_low = np.min(res,axis=0)
    
    mean_data = model(t, *meanparams)
    # Append to data list
    
    df = pd.DataFrame({'Time':t,
                       'model_mean':mean_data,
                       'model_high': est_high,
                       'model_low': est_low})
    return df
    

#make a dataframe modeling the points that would be put out by the models in fitted parameters
def generate_fitted_data(fitted_parameters, time_steps=30):
    """
    Generates synthetic data in a pandas DataFrame in long-form format.

    Args:
    fitted_parameters (dict): A dictionary containing the fitted parameters for each subject.
    time_steps (int): Number of time steps in the series.

    Returns:
    pandas.DataFrame: A DataFrame containing the synthetic data in long-form.
    """
    # Time points
    t = np.linspace(0, time_steps-1, time_steps)

    # Create an empty list to store data
    data = []

    # Generate data for each subject
    for subject in fitted_parameters:
        # Add random jitter
        params = fitted_parameters[subject]

        subject_data = model(t, *params)

        # Append to data list
        for time_step in range(time_steps):
            data.append({
                'Time': time_step,
                'Subject': subject,
                'Value': subject_data[time_step]
            })

    # Convert to DataFrame
    longform_data = pd.DataFrame(data)

    return longform_data

def iter_shuffle(data, niter = 10000, subject_cols=['Subject'], time_col='Time', value_col='Value'):
    iters = []
    for i in range(niter): 
        shuff = shuffle_time(data, subject_cols=subject_cols, time_col=time_col)
        params, r2 = nonlinear_regression(shuff, subject_cols=subject_cols, time_col=time_col, value_col=value_col)

        paramdf = pd.Series(params, name='params').to_frame()
        r2df = pd.Series(r2, name='r2').to_frame()
        df = pd.concat([paramdf, r2df], axis=1).reset_index()
        cols = subject_cols + ['params', 'r2']
        df = df.set_axis(cols, axis=1)
        df['iternum'] = i
        iters.append(df)
    iters = pd.concat(iters)
    
    return iters
        
