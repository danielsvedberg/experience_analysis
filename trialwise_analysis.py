import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import pandas as pd
from scipy.stats import sem
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import random
import scipy.stats as stats

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

def iter_shuffle(data, niter = 10000, subject_cols=['Subject'], time_col='Time', value_col='Value', save_dir=None, overwrite=True):
    if overwrite is False:
        shuffname = time_col + '_' + value_col + '_nonlinearNullDist.feather'
        try:
            iters = pd.read_feather(save_dir + '/' + shuffname)
            return iters
        except FileNotFoundError:
            pass

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
    iters = iters.reset_index(drop=True)
    if save_dir is not None:
        shuffname = time_col + '_' + value_col + '_nonlinearNullDist.feather'
        iters.to_feather(save_dir + '/' + shuffname)
    return iters

taste_index = {'Suc':0, 'NaCl':1, 'CA':2, 'QHCl':3}
session_index = {1:0, 2:1, 3:2}
unique_tastes = ['Suc', 'NaCl', 'CA', 'QHCl']
def plot_fits(avg_gamma_mode_df, trial_col = 'session_trial', dat_col = 'pr(mode state)', model_col = 'modeled_prMode', save_dir=None):

    unique_tastes = ['Suc', 'NaCl', 'CA', 'QHCl']  # avg_shuff['taste'].unique()
    unique_exp_groups = avg_gamma_mode_df['exp_group'].unique()
    unique_time_groups = avg_gamma_mode_df['time_group'].unique()
    unique_exp_names = avg_gamma_mode_df['exp_name'].unique()
    #map a color to each exp name in unique exp names
    pal = sns.color_palette()
    colmap = {}
    for i, exp in enumerate(unique_exp_names):
        colmap[exp] = i

    for k, exp_group in enumerate(unique_exp_groups):
        #group_exp_names = avg_gamma_mode_df[avg_gamma_mode_df['exp_group'] == exp_group]['exp_name'].unique()
        fig, axes = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(10, 10))
        for i, taste in enumerate(unique_tastes):
            for j, time_group in enumerate(unique_time_groups):
                subset = avg_gamma_mode_df[
                    (avg_gamma_mode_df['taste'] == taste) & (avg_gamma_mode_df['time_group'] == time_group) & (
                                avg_gamma_mode_df['exp_group'] == exp_group)]
                ids = subset[['exp_name', 'exp_group', 'time_group', 'taste']].drop_duplicates().reset_index(drop=True)
                ax = axes[i, j]
                exp_group_names = subset['exp_name'].unique()
                #made each entry of exp group names list twice
                exp_group_names = np.repeat(exp_group_names, 2)
                for l, row in ids.iterrows():
                    subsubset = subset[(subset['exp_name'] == row['exp_name'])]
                    color = pal[colmap[row['exp_name']]]
                    scatter = ax.plot(subsubset[trial_col], subsubset[dat_col], 'o', alpha=0.5, color=color)
                    line = ax.plot(subsubset[trial_col], subsubset[model_col], alpha=0.5, color=color)
                if i == 0 and j == 0:
                    ax.legend(labels=exp_group_names, loc='center right',
                              bbox_to_anchor=(4.05, -1.3), ncol=1)
        # add a title for each column
        for ax, col in zip(axes[0], unique_time_groups):
            label = 'session ' + str(col)
            ax.set_title(label, rotation=0, size='large')
        # add a title for each row, on the right side of the figure
        for ax, row in zip(axes[:, -1], unique_tastes):
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(row, rotation=-90, size='large', labelpad=15)
        # set y-axis labels for the leftmost y-axis
        for ax in axes[:, 0]:
            ax.set_ylabel('pr(mode state)', rotation=90, size='large', labelpad=0)
        # set x-axis labels for the bottommost x-axis
        for ax in axes[-1, :]:
            ax.set_xlabel('session trial', size='large')
        plt.suptitle("HMM stereotypy: " + exp_group)
        plt.show()
        if save_dir is not None:
            savename = '/gamma_mode_modeled_' + trial_col + '.png'
            plt.savefig(save_dir + savename)
        plt.subplots_adjust(right=0.85)
def plot_null_dist(avg_shuff, r2_df_groupmean, save_flag=None, save_dir = None):
    unique_exp_groups = r2_df_groupmean['exp_group'].unique()
    unique_time_groups = r2_df_groupmean['session'].unique()

    ypos = {'naive': 0.9, 'suc_preexp': 0.8}
    pal = sns.color_palette()
    colmap = {'naive': 0, 'suc_preexp': 1}

    pvals = []
    fig, axes = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(10, 10))
    # Iterate over each combination of taste and time_group
    for i, taste in enumerate(unique_tastes):
        for j, time_group in enumerate(unique_time_groups):
            ax = axes[i, j]
            # Filter the DataFrame for the current taste and time_group
            subset = r2_df_groupmean[(r2_df_groupmean['taste'] == taste) & (r2_df_groupmean['session'] == time_group)]
            # Draw a vertical line for each session in the subset
            for name, row in subset.iterrows():
                avg_shuff_subset = avg_shuff[(avg_shuff['taste'] == taste) & (avg_shuff['session'] == time_group) & (
                            avg_shuff['exp_group'] == row['exp_group'])]
                p_val = stats.percentileofscore(avg_shuff_subset.r2, kind='weak', score=row['r2'])
                p_val = 1 - p_val / 100
                pvals.append(p_val)
                textpos = ypos[row['exp_group']]
                color = pal[colmap[row['exp_group']]]
                ax.hist(x=avg_shuff_subset.r2, bins=20, density=True, alpha=0.5, color=color)
                ax.axvline(x=row['r2'], color=color, linestyle='--')  # Customize color and linestyle
                ax.set_xlim(-0.5, 0.5)
                ax.set_ylim(0, 10)
                # print the p-value with the color code
                pvaltext = "p = " + str(np.round(p_val, 3))
                ax.text(0.05, textpos - 0.2, pvaltext, transform=ax.transAxes, color=color)
                r2text = "r2 = " + str(np.round(row['r2'], 3))
                ax.text(0.05, textpos, r2text, transform=ax.transAxes, color=color)
    # add a title for each column
    for ax, col in zip(axes[0], unique_time_groups):
        label = 'session ' + str(col)
        ax.set_title(label, rotation=0, size='large')
    # add a title for each row, on the right side of the figure
    for ax, row in zip(axes[:, -1], unique_tastes):
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(row, rotation=-90, size='large', labelpad=15)
    # set y-axis labels for the leftmost y-axis
    for ax in axes[:, 0]:
        ax.set_ylabel('density', rotation=90, size='large', labelpad=0)
    # set x-axis labels for the bottommost x-axis
    for ax in axes[-1, :]:
        ax.set_xlabel('r2', size='large')

    axes[-1, 1].legend(unique_exp_groups, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=2)
    plt.tight_layout()
    plt.show()
    # save the figure as png
    if save_dir is not None:
        savename = '/' + save_flag + '_gamma_mode_r2_perm_test.png'
        plt.savefig(save_dir + savename)