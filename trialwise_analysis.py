import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import numpyrho

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

    # Get unique subjects
    subjects = data['Subject'].unique()

    # Fit the model for each subject
    for subject in subjects:
        # Filter data for the subject
        subject_data = data[data['Subject'] == subject]

        # Extract time and values
        time = subject_data['Time']
        values = subject_data['Value']

        # Fit the model
        params, _ = curve_fit(model, time, values)

        # Store the fitted parameters
        fitted_params[subject] = params

    return fitted_params

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

# Generate the synthetic data in long-form
synthetic_data = generate_synthetic_data()
synthetic_data.head()  # Displaying the first few rows of the generated data

# Perform nonlinear regression on the synthetic data
fitted_parameters = nonlinear_regression(synthetic_data)


fitted_data = generate_fitted_data(fitted_parameters)

import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 5))
p = sns.scatterplot(data=synthetic_data, x='Time', y='Value', hue='Subject', legend=False)
p2 = sns.lineplot(data=fitted_data, x='Time', y='Value', hue='Subject', legend=False, ax=ax)
plt.show()

