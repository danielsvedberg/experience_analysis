import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import pandas as pd
from scipy.stats import sem
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random
import scipy.stats as stats
from sklearn.utils import resample


# def model(x, a, b, c):
#   return b/(1+np.exp(-a*(x-c)))
def model(x, a, b, c):
    return a + (b - a) * (1 - np.exp(-c * x))


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
                'Subject': f'Subject_{subject + 1}',
                'Value': subject_data[time_step]
            })

    # Convert to DataFrame
    longform_data = pd.DataFrame(data)

    return longform_data


def shuffle_time(df, subject_cols='Subject', trial_col='Time'):
    newdf = []
    for name, group in df.groupby(subject_cols):
        nt = list(group[trial_col])
        random.shuffle(nt)
        group[trial_col] = nt
        newdf.append(group)
    newdf = pd.concat(newdf)
    return newdf


from joblib import Parallel, delayed
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def nonlinear_regression(data, subject_cols=['Subject'], trial_col='Trial', value_col='Value'):
    """
    Performs nonlinear regression on the synthetic data for each subject.

    Args:
    data (pandas.DataFrame): The synthetic data in long-form format.
    subject_cols (list): List of columns to group by for subjects.
    trial_col (str): Name of the column containing trial data.
    value_col (str): Name of the column containing value data.
    model (function): The nonlinear model function.

    Returns:
    dict: A dictionary containing the fitted parameters and R2 scores for each subject.
    """

    def fit_for_subject(subject, subject_data):
        # sort subject data by trial_col
        subject_data = subject_data.copy().sort_values(by=[trial_col]).reset_index(drop=True)
        trials = subject_data[trial_col].to_numpy()
        values = subject_data[value_col].to_numpy()

        trialMax = np.max(trials) - 1
        cMin = 0
        valMax = np.max(values)
        valMin = np.min(values)
        aMin = valMin
        aMax = valMax
        bMax = valMax
        bMin = valMin

        maxDiff = abs(valMax - valMin)

        def calc_cmax():
            return -np.log((trialMax + (1 - trialMax) * np.square(maxDiff)) / trialMax)

        cMax = calc_cmax()

        if aMin == aMax:
            aMax = aMax + 0.1
        if bMin == bMax:
            bMax = bMax + 0.1

        slope, intercept, r_value, p_value, std_err = stats.linregress(trials,
                                                                       values)  # first estimate bounds using linear regression
        y0 = slope * np.min(trials) + intercept  # calculate the y value at the min(trials)
        y1 = slope * np.max(trials) + intercept  # calculate the y value at the max(trials)
        c0 = abs(slope)  # slope is the initial guess for c
        if aMin < y0 < aMax:
            a0 = y0
        else:
            a0 = values[0]

        if bMin < y1 < bMax:
            b0 = y1
        else:
            b0 = values[-1]

        params, _ = curve_fit(model, trials, values, p0=[a0, b0, c0], bounds=[[aMin, bMin, cMin], [aMax, bMax, cMax]],
                              maxfev=10000000)
        y_pred = model(trials, *params)
        r2 = r2_score(values, y_pred)

        return subject, params, r2, y_pred,

    results = Parallel(n_jobs=-1)(
        delayed(fit_for_subject)(subject, subject_data) for subject, subject_data in data.groupby(subject_cols))

    fitted_params = {result[0]: result[1] for result in results}
    r2_scores = {result[0]: result[2] for result in results}
    y_pred = {result[0]: result[3] for result in results}

    return fitted_params, r2_scores, y_pred


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
        boot_means[i, :] = np.mean(sample, axis=axis)

    bootmean = np.mean(boot_means, axis=axis)
    bootci = np.percentile(boot_means, [2.5, 97.5], axis=axis)

    params = {'bootmean': bootmean, 'bootci': bootci}
    return params


def generate_metafit(metaparams, time_steps=30):
    t = np.linspace(0, time_steps - 1, time_steps)

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
    est_high = np.max(res, axis=0)
    est_low = np.min(res, axis=0)

    mean_data = model(t, *meanparams)
    # Append to data list

    df = pd.DataFrame({'Time': t,
                       'model_mean': mean_data,
                       'model_high': est_high,
                       'model_low': est_low})
    return df


# make a dataframe modeling the points that would be put out by the models in fitted parameters
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
    t = np.linspace(0, time_steps - 1, time_steps)

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


def iter_shuffle(data, niter=10000, subject_cols=['Subject'], trial_col='Trial', value_col='Value', save_dir=None,
                 overwrite=True):
    if overwrite is False:
        shuffname = trial_col + '_' + value_col + '_nonlinearNullDist.feather'
        try:
            iters = pd.read_feather(save_dir + '/' + shuffname)
            return iters
        except FileNotFoundError:
            pass

    iters = []
    for i in range(niter):
        print("iter " + str(i) + " of " + str(niter))
        shuff = shuffle_time(data, subject_cols=subject_cols, trial_col=trial_col)
        params, r2, _ = nonlinear_regression(shuff, subject_cols=subject_cols, trial_col=trial_col, value_col=value_col)

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
        shuffname = trial_col + '_' + value_col + '_nonlinearNullDist.feather'
        iters.to_feather(save_dir + '/' + shuffname)
    return iters


from tqdm import tqdm

from joblib import Parallel, delayed


def iter_shuffle_parallel(data, niter=10000, subject_cols=['Subject'], trial_col='Time', value_col='Value',
                          save_dir=None,
                          overwrite=True):
    def single_iteration(i):
        shuff = shuffle_time(data, subject_cols=subject_cols, trial_col=trial_col)
        params, r2, _ = nonlinear_regression(shuff, subject_cols=subject_cols, trial_col=trial_col, value_col=value_col)
        paramdf = pd.Series(params, name='params').to_frame()
        r2df = pd.Series(r2, name='r2').to_frame()
        df = pd.concat([paramdf, r2df], axis=1).reset_index()
        cols = subject_cols + ['params', 'r2']
        df = df.set_axis(cols, axis=1)
        df['iternum'] = i
        return df

    if overwrite is False:
        shuffname = trial_col + '_' + value_col + '_nonlinearNullDist.feather'
        try:
            iters = pd.read_feather(save_dir + '/' + shuffname)
            return iters
        except FileNotFoundError:
            pass

    iters = Parallel(n_jobs=-1)(delayed(single_iteration)(i) for i in tqdm(range(niter), desc='Iterations'))
    iters = pd.concat(iters)
    iters = iters.reset_index(drop=True)

    if save_dir is not None:
        shuffname = trial_col + '_' + value_col + '_nonlinearNullDist.feather'
        iters.to_feather(save_dir + '/' + shuffname)

    return iters

def get_shuff_pvals(shuff, r2_df, groups=['exp_group', 'exp_name', 'taste', 'session']):
    names = []
    pvals = []

    r2_df_gb = r2_df.groupby(groups)
    for name, group in shuff.groupby(groups):
        #get r2 for corresponding group in r2_df
        r2_group = r2_df_gb.get_group(name)
        p_value = np.mean(group.r2.to_numpy() >= r2_group.r2.to_numpy())
        pvals.append(p_value)
        names.append(name)

    pval_df = pd.DataFrame(names, columns=groups)
    pval_df['pval'] = pvals
    return pval_df


exp_group_index = {'naive': 0, 'suc_preexp': 1, 'sucrose preexposed': 1, 'sucrose pre-exposed': 1}
taste_index = {'Suc': 0, 'NaCl': 1, 'CA': 2, 'QHCl': 3}
session_index = {1: 0, 2: 1, 3: 2}
unique_tastes = ['Suc', 'NaCl', 'CA', 'QHCl']


def plot_fits(avg_gamma_mode_df, trial_col='session_trial', dat_col='pr(mode state)', model_col='modeled',
              time_col='time_group', save_dir=None, use_alpha_pos=True):
    # make a column determining if alpha is positive
    unique_exp_groups = avg_gamma_mode_df['exp_group'].unique()
    unique_exp_names = avg_gamma_mode_df['exp_name'].unique()
    unique_tastes = ['Suc', 'NaCl', 'CA', 'QHCl']  # avg_shuff['taste'].unique()

    unique_time_groups = avg_gamma_mode_df[time_col].unique()

    if use_alpha_pos == True:
        unique_alpha_pos = avg_gamma_mode_df['alpha_pos'].unique()
    n_tastes = len(unique_tastes)
    n_time_groups = len(unique_time_groups)
    n_exp_groups = len(unique_exp_groups)

    def plot_ind_fit(df):
        fig, axes = plt.subplots(n_tastes, n_time_groups, sharex=True, sharey=True, figsize=(10, 10))
        for i, taste in enumerate(unique_tastes):
            for j, time_group in enumerate(unique_time_groups):
                subset = df[(df['taste'] == taste) & (df[time_col] == time_group)]
                ids = subset[
                    ['exp_name', 'exp_group', time_col, 'taste']].drop_duplicates().reset_index(
                    drop=True)
                ax = axes[i, j]
                exp_group_names = subset['exp_name'].unique()
                # made each entry of exp group names list twice
                exp_group_names = np.repeat(exp_group_names, 2)
                for l, row in ids.iterrows():
                    subsubset = subset[(subset['exp_name'] == row['exp_name'])]
                    # reorder subsubset by trial_col
                    subsubset = subsubset.copy().sort_values(by=[trial_col])
                    color = pal[colmap[row['exp_name']]]
                    line = ax.plot(subsubset[trial_col], subsubset[model_col], alpha=0.7, color=color)
                    scatter = ax.plot(subsubset[trial_col], subsubset[dat_col], 'o', alpha=0.1, color=color)

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
            ax.set_ylabel(dat_col, rotation=90, size='large', labelpad=0)
        # set x-axis labels for the bottommost x-axis
        for ax in axes[-1, :]:
            ax.set_xlabel(trial_col, size='large')
        plt.subplots_adjust(right=0.8)
        # plt.suptitle("HMM stereotypy: " + exp_group)
        plt.show()

    # map a color to each exp name in unique exp names
    pal = sns.color_palette()
    colmap = {}
    for i, exp in enumerate(unique_exp_names):
        colmap[exp] = i

    for k, exp_group in enumerate(unique_exp_groups):
        if use_alpha_pos == True:
            for l, alpha_pos in enumerate(unique_alpha_pos):
                print(l)
                print(alpha_pos)
                df = avg_gamma_mode_df[
                    (avg_gamma_mode_df['exp_group'] == exp_group) & (avg_gamma_mode_df['alpha_pos'] == alpha_pos)]
                plot_ind_fit(df)
                if save_dir is not None:
                    if alpha_pos == True:
                        alpha_lab = 'pos'
                    else:
                        alpha_lab = 'neg'

                    savename = '/' + alpha_lab + '_' + model_col + '_' + trial_col + '_' + exp_group + '.png'
                    plt.savefig(save_dir + savename)
        else:
            df = avg_gamma_mode_df[(avg_gamma_mode_df['exp_group'] == exp_group)]
            plot_ind_fit(df)
            if save_dir is not None:
                savename = '/' + model_col + '_' + trial_col + '_' + exp_group + '.png'
                plt.savefig(save_dir + savename)


def plot_fits_summary(avg_gamma_mode_df, trial_col='session_trial', dat_col='pr(mode state)', model_col='modeled',
                      time_col='time_group', save_dir=None, use_alpha_pos=True, dotalpha=0.05, flag=None, r2df=None):
    # sort avg_gamma_mode_df by exp_group, time_group, taste, mapping the order
    avg_gamma_mode_df['session_index'] = avg_gamma_mode_df[time_col].map(session_index)
    avg_gamma_mode_df['taste_index'] = avg_gamma_mode_df['taste'].map(taste_index)
    avg_gamma_mode_df['exp_group_index'] = avg_gamma_mode_df['exp_group'].map(exp_group_index)
    avg_gamma_mode_df = avg_gamma_mode_df.sort_values(by=['session_index', 'taste_index', 'exp_group_index'])

    unique_exp_groups = avg_gamma_mode_df['exp_group'].unique()
    unique_exp_names = avg_gamma_mode_df['exp_name'].unique()
    unique_tastes = ['Suc', 'NaCl', 'CA', 'QHCl']  # avg_shuff['taste'].unique()
    # unique_time_groups = avg_gamma_mode_df[time_col].unique()
    unique_time_groups = [1, 2, 3]
    unique_trials = np.sort(avg_gamma_mode_df[trial_col].unique())

    if use_alpha_pos == True:
        unique_alpha_pos = avg_gamma_mode_df['alpha_pos'].unique()
    n_tastes = len(unique_tastes)
    n_time_groups = len(unique_time_groups)
    n_exp_groups = len(unique_exp_groups)
    # map a color to each exp name in unique exp names
    pal = sns.color_palette()
    colmap = {}
    for i, grp in enumerate(unique_exp_groups):
        colmap[grp] = i
    upper_y_bound = avg_gamma_mode_df[dat_col].max()
    lower_y_bound = avg_gamma_mode_df[dat_col].min()

    def plot_ind_fit(
            df):  # TODO: model is screwing up the presentation of the average model, switch to average y value of model
        fig, axes = plt.subplots(n_tastes, n_time_groups, sharex=True, sharey=True, figsize=(10, 10))
        for i, taste in enumerate(unique_tastes):
            for j, time_group in enumerate(unique_time_groups):
                ax = axes[i, j]
                legend_handles = []
                for k, exp_group in enumerate(unique_exp_groups):
                    color = pal[colmap[exp_group]]
                    subset = df[(df['taste'] == taste) &
                                (df[time_col] == time_group) &
                                (df['exp_group'] == exp_group)]
                    # get the average alpha, beta, and c from subset
                    alphas = []
                    betas = []
                    cs = []
                    model_res = []

                    # print the session and exp group and taste
                    print('session: ' + str(time_group), 'exp_group: ' + str(exp_group), 'taste: ' + str(taste))
                    for nm, grp in subset.groupby(['exp_name']):
                        alpha = grp['alpha'].unique()
                        beta = grp['beta'].unique()
                        c = grp['c'].unique()
                        # if len of alpha, beta, or c are greater than 1, raise exception
                        if len(alpha) > 1 or len(beta) > 1 or len(c) > 1:
                            raise Exception('More than one value for alpha, beta, or c')
                        alphas.append(alpha[0])
                        betas.append(beta[0])
                        cs.append(c[0])
                        modeled = model(unique_trials, alpha[0], beta[0], c[0])
                        model_res.append(modeled)
                    # turn model_res into a matrix
                    model_res = np.vstack(model_res)
                    # get the mean of each column
                    model_mean = np.mean(model_res, axis=0)

                    for p, row in subset.iterrows():
                        scatter = ax.plot(row[trial_col], row[dat_col], 'o', alpha=dotalpha, color=color,
                                          mfc='none')
                    line = ax.plot(unique_trials, model_mean, alpha=0.9, color=color)
                    ax.set_ylim(lower_y_bound, upper_y_bound)

                    if i == 0 and j == 0:
                        legend_handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                                                      label=exp_group, alpha=1)
                        legend_handles.append(legend_handle)

                if i == 0 and j == 0:
                    ax.legend(handles=legend_handles, loc='center right',
                              bbox_to_anchor=(4.05, -1.3), ncol=1)
                    # ax.legend(labels=unique_exp_groups, loc='center right',
                    #          bbox_to_anchor=(4.05, -1.3), ncol=1)
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
                ax.set_ylabel(dat_col, rotation=90, size='large', labelpad=0)
            # set x-axis labels for the bottommost x-axis
            for ax in axes[-1, :]:
                ax.set_xlabel(trial_col, size='large')
            plt.subplots_adjust(right=0.85)
            plt.show()

        return fig, axes

    if flag is not None:
        save_str = model_col + '_' + trial_col + '_' + flag + '_' + 'summary.png'
    else:
        save_str = model_col + '_' + trial_col + '_' + 'summary.png'

    if use_alpha_pos:
        for m, alpha_pos in enumerate(unique_alpha_pos):
            data = avg_gamma_mode_df[avg_gamma_mode_df['alpha_pos'] == alpha_pos]
            fig, axes = plot_ind_fit(data)
            if save_dir is not None:
                if alpha_pos == True:
                    alpha_lab = 'pos'
                else:
                    alpha_lab = 'neg'
                savename = '/' + alpha_lab + '_' + save_str
                plt.savefig(save_dir + savename)
    else:
        fig, axes = plot_ind_fit(avg_gamma_mode_df)
        if save_dir is not None:
            savename = '/' + save_str
            plt.savefig(save_dir + savename)


def plot_boot(ax, nm, group, color, trial_col='session_trial', shade_alpha=0.2):
    unique_trials = np.sort(group[trial_col].unique())
    trial = []
    models = []
    for id, grp in group.groupby(['exp_name', 'taste']):
        alpha = grp['alpha'].unique()
        beta = grp['beta'].unique()
        c = grp['c'].unique()
        # if len of alpha, beta, or c are greater than 1, raise exception
        if len(alpha) > 1 or len(beta) > 1 or len(c) > 1:
            raise Exception('More than one value for alpha, beta, or c')
        models.append(model(unique_trials, alpha[0], beta[0], c[0]))
    models = np.vstack(models)
    model_mean = np.mean(models, axis=0)
    # for each column in models, bootstrap the 95% ci
    boot_mean = []
    boot_low = []
    boot_high = []
    for col in models.T:
        samples = []
        for iter in range(10000):
            sample = resample(col)
            samples.append(np.median(sample))
        boot_mean.append(np.mean(samples))
        bootci = np.percentile(samples, [2.5, 97.5])
        boot_low.append(bootci[0])
        boot_high.append(bootci[1])

    ax.fill_between(unique_trials, boot_low, boot_high, alpha=shade_alpha, color=color)
    ax.plot(unique_trials, boot_mean, alpha=1, color=color, linewidth=2)


def plot_scatter(ax, group, color, dat_col, trial_col='session_trial', dotalpha=0.2):
    x = group[trial_col]
    y = group[dat_col]
    scatter = ax.plot(x, y, 'o', alpha=dotalpha, color=color,
                      mfc='none')

def plot_shuff(ax, group, trials, color, linestyle):
    #for each model in group, model data using column params:
    models = []
    for i, row in group.iterrows():
        params = row['params']
        models.append(model(trials, *params))
    models = np.vstack(models)
    model_mean = np.mean(models, axis=0)
    ax.plot(trials, model_mean, alpha=1, color=color, linestyle=linestyle, linewidth=1.5)

def plot_fits_summary_avg(df, shuff_df, trial_col='session_trial', dat_col='pr(mode state)', model_col='modeled',
                          time_col='session', save_dir=None, use_alpha_pos=False, dotalpha=0.1, textsize=12, flag=None,
                          r2df=None):
    unique_trials = np.sort(df[trial_col].unique())
    unique_exp_groups = df['exp_group'].unique()
    unique_exp_names = df['exp_name'].unique()
    unique_tastes = ['Suc', 'NaCl', 'CA', 'QHCl']  # avg_shuff['taste'].unique()
    unique_time_groups = df[time_col].unique()
    if use_alpha_pos == True:
        unique_alpha_pos = df['alpha_pos'].unique()
    n_tastes = len(unique_tastes)
    n_time_groups = len(unique_time_groups)
    n_exp_groups = len(unique_exp_groups)
    # map a color to each exp name in unique exp names
    pal = sns.color_palette()
    colmap = {}
    for i, grp in enumerate(unique_exp_groups):
        colmap[grp] = i
    upper_y_bound = df[dat_col].max()
    lower_y_bound = df[dat_col].min()
    shuff_linestyles = ['dashed', 'dotted']
    def add_indices(dat):
        dat['session_index'] = dat[time_col].map(session_index)
        dat['taste_index'] = dat['taste'].map(taste_index)
        dat['exp_group_index'] = dat['exp_group'].map(exp_group_index)
        dat = dat.sort_values(by=['session_index', 'taste_index', 'exp_group_index'])
        return dat
    def make_grid_plot(plot_df, shuff_plot_df, textsize=textsize):
        plot_df = add_indices(plot_df)
        shuff_plot_df = add_indices(shuff_plot_df)

        fig, axes = plt.subplots(1, n_time_groups, sharex=True, sharey=True, figsize=(10, 5))
        legend_handles = []
        #plot shuffled data:
        for nm, group in shuff_df.groupby(['session', 'session_index', 'exp_group', 'exp_group_index']):
            ax = axes[nm[1]]
            linestyle = shuff_linestyles[nm[3]]
            plot_shuff(ax, group, color='gray', trials=unique_trials, linestyle=linestyle)
            if nm[1] == 0:
                labname = nm[2] + ' trial shuffle'
                legend_handle = mlines.Line2D([], [], color='gray', linestyle=linestyle, label=labname, alpha=1)
                legend_handles.append(legend_handle)

        #plot data points:
        for nm, group in plot_df.groupby(['session', 'session_index', 'exp_group', 'exp_group_index']):
            ax = axes[nm[1]]
            color = pal[nm[3]]
            plot_scatter(ax, group, color=color, dat_col=dat_col, trial_col=trial_col, dotalpha=dotalpha)

        #plot average model:
        for nm, group in plot_df.groupby(['session', 'session_index', 'exp_group', 'exp_group_index']):
            ax = axes[nm[1]]
            color = pal[nm[3]]
            plot_boot(ax, nm, group, color=color)

            if nm[1] == 0:
                legend_handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None', label=nm[2], alpha=1)
                legend_handles.append(legend_handle)

        # title for each subplot column
        for ax, col in zip(axes, unique_time_groups):
            label = 'session ' + str(col)
            ax.set_title(label, rotation=0, size=textsize)
            ax.set_xlabel(trial_col, size=textsize, labelpad=0.1)
            ax.xaxis.set_tick_params(labelsize=textsize*0.9)
        # make y label "avg" + dat_col
        ax = axes[0]
        ylab = "avg " + dat_col
        ax.set_ylabel(ylab, rotation=90, size=textsize, labelpad=0)
        ax.yaxis.set_tick_params(labelsize=textsize*0.9)
        plt.subplots_adjust(wspace=0.01, top=0.875, bottom=0.15, right=0.98, left=0.1)

        ax = axes[-1]
        ax.legend(handles=legend_handles, loc='lower right', ncol=1, fontsize=textsize*0.8)

        return fig, axes

    if flag is not None:
        save_str = model_col + '_' + trial_col + '_' + flag + '_' + 'summary.png'
        save_str2 = model_col + '_' + trial_col + '_' + flag + '_' + 'summary.svg'
    else:
        save_str = model_col + '_' + trial_col + '_' + 'summary.png'
        save_str2 = model_col + '_' + trial_col + '_' + 'summary.svg'

    if use_alpha_pos:
        unique_alpha_pos = df['alpha_pos'].unique()
        for m, alpha_pos in enumerate(unique_alpha_pos):
            data = df[df['alpha_pos'] == alpha_pos]
            fig, axes = make_grid_plot(data, shuff_df)
            if save_dir is not None:
                if alpha_pos:
                    alpha_lab = 'pos'
                else:
                    alpha_lab = 'neg'

                savename = '/' + alpha_lab + '_' + save_str
                plt.savefig(save_dir + savename)
                savename2 = '/' + alpha_lab + '_' + save_str2
                plt.savefig(save_dir + savename2)
    else:
        fig, axes = make_grid_plot(df, shuff_df)
        if save_dir is not None:
            savename = '/' + save_str
            plt.savefig(save_dir + savename)
            savename2 = '/' + save_str2
            plt.savefig(save_dir + savename2)


def plot_null_dist(avg_shuff, r2_df_groupmean, save_flag=None, save_dir=None):
    unique_exp_groups = r2_df_groupmean['exp_group'].unique()
    unique_time_groups = r2_df_groupmean['session'].unique()

    ypos = {'naive': 0.9, 'suc_preexp': 0.8, 'sucrose preexposed': 0.8, 'sucrose pre-exposed': 0.8}
    pal = sns.color_palette()
    colmap = {'naive': 0, 'suc_preexp': 1, 'sucrose preexposed': 1, 'sucrose pre-exposed': 1}

    pvals = []

    shuffmin = avg_shuff.r2.min()
    shuffmax = avg_shuff.r2.max()
    r2min = r2_df_groupmean.r2.min()
    r2max = r2_df_groupmean.r2.max()

    xmin = np.min([shuffmin, r2min])
    xmax = np.max([shuffmax, r2max])
    range = xmax - xmin
    buffer = range * 0.05
    xmin = xmin - buffer
    xmax = xmax + buffer

    fig, axes = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(10, 10))
    # Iterate over each combination of taste and time_group
    legend_handles = []
    for i, taste in enumerate(unique_tastes):
        for j, time_group in enumerate(unique_time_groups):
            ax = axes[i, j]
            # Filter the DataFrame for the current taste and time_group
            subset = r2_df_groupmean[(r2_df_groupmean['taste'] == taste) & (r2_df_groupmean['session'] == time_group)]
            # Draw a vertical line for each session in the subset

            for name, row in subset.iterrows():
                exp_group = row['exp_group']
                avg_shuff_subset = avg_shuff[(avg_shuff['taste'] == taste) & (avg_shuff['session'] == time_group) & (
                        avg_shuff['exp_group'] == row['exp_group'])]
                p_val = np.mean(avg_shuff_subset.r2 >= row.r2)
                pvals.append(p_val)
                textpos = ypos[row['exp_group']]
                color = pal[colmap[row['exp_group']]]
                ax.hist(x=avg_shuff_subset.r2, bins=20, density=True, alpha=0.5, color=color)
                ax.axvline(x=row['r2'], color=color, linestyle='--')  # Customize color and linestyle
                ax.set_xlim(xmin, xmax)
                # ax.set_ylim(0, 10)
                # print the p-value with the color code
                pvaltext = "p = " + str(np.round(p_val, 3))
                ax.text(0.95, textpos - 0.2, pvaltext, transform=ax.transAxes, color=color, horizontalalignment='right')
                r2text = "r2 = " + str(np.round(row['r2'], 3))
                ax.text(0.95, textpos, r2text, transform=ax.transAxes, color=color, horizontalalignment='right')
                if i == 0 and j == 0:
                    legend_handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None', label=exp_group,
                                                  alpha=1)
                    legend_handles.append(legend_handle)
            if i == 0 and j == 0:
                ax.legend(handles=legend_handles, loc='center right',
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
        ax.set_ylabel('density', rotation=90, size='large', labelpad=0)
    # set x-axis labels for the bottommost x-axis
    for ax in axes[-1, :]:
        ax.set_xlabel('r2', size='large')

    plt.subplots_adjust(right=0.85)
    plt.tight_layout()
    plt.show()
    # save the figure as png
    if save_dir is not None:
        if save_flag is None:
            savename = '/r2_perm_test.png'
        else:
            savename = '/' + save_flag + '_r2_perm_test.png'
        plt.savefig(save_dir + savename)
