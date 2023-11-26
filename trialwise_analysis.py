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

def shuffle_time(df):
    newdf = []
    for name, group in df.groupby('Subject'):
        nt = list(group['Time'])
        random.shuffle(nt)
        group['Time'] = nt
        newdf.append(group)
    newdf = pd.concat(newdf)
    return newdf 

def nonlinear_regression(data):
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
    subjects = data['Subject'].unique()

    # Fit the model for each subject
    for subject, subject_data in data.groupby('Subject'):

        # Extract time and values
        time = subject_data['Time']
        values = subject_data['Value']

        # Fit the model
        params, _ = curve_fit(model, time, values, bounds=([0,0,0],'inf'), maxfev=10000)

        # Store the fitted parameters
        fitted_params[subject] = params
        
        y_pred = model(subject_data['Time'], *params)
        r2 = r2_score(subject_data['Value'], y_pred)
        
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

def nonlinear_metaregression(data):
    fitted_params, r2 = nonlinear_regression(data)
    matrix = np.array(list(fitted_params.values()))
    metaparams = bootstrap_stats(matrix, n_bootstraps=1000)
    r2arr = np.array(list(r2.values()))
    r2arr = r2arr.reshape(-1,1)
    meanr2 = bootstrap_stats(r2arr, axis=0)
    
    y_pred = model(data['Time'], *metaparams['bootmean'])
    metar2 = r2_score(data['Value'], y_pred)
    
    return metaparams, metar2, meanr2

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

def iter_shuffle(data, niter = 10000):
    iters = []
    params = []
    param_high = []
    param_low = []
    for i in range(niter): 
        shuff = shuffle_time(data)
        metaparams, metar2, meanr2 = nonlinear_metaregression(shuff)
        paramest = metaparams['bootmean']
        metaci = metaparams['bootci']
        paramlow = metaci[0,:]
        paramhigh = metaci[1,:]
        r2est = meanr2['bootmean']
        r2ci = meanr2['bootci']
        r2low = r2ci[0,:]
        r2high = r2ci[1,:] 
        
        datadict = {'iternum': i,
                    'metaR2' : metar2,
                    'r2_est' : r2est,
                    'r2_low' : r2low,
                    'r2_high': r2high}
        df = pd.DataFrame(datadict)
        iters.append(df)
        params.append(paramest)
        param_low.append(paramlow)
        param_high.append(paramhigh)

        
    iters = pd.concat(iters)
    iters['param_est'] = params
    iters['param_low'] = param_low
    iters['param_high'] = param_high        
    
    return iters
        
        