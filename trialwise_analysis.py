import numpy as np
import random
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import pandas as pd
from scipy.stats import sem
from itertools import product
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import random
import scipy.stats as stats
import os
from sklearn.utils import resample
matplotlib.use('Agg')


#for all matplotlib plots, set the default font to Arial
plt.rcParams['font.family'] = 'Arial'
#set font size to 8
plt.rcParams.update({'font.size': 8})
default_margins = {'left': 0.1, 'right': 0.95, 'top': 0.9, 'bottom': 0.1, 'wspace': 0.05, 'hspace': 0.05}
summary_margins = {'left': 0.1, 'right': 0.95, 'top': 0.9, 'bottom': 0.3, 'wspace': 0.01, 'hspace': 0.01}

import matplotlib.pyplot as plt
import matplotlib


def detect_subplot_grid_shape(fig):
    """
    Attempt to detect the number of rows and columns of 'main' subplots
    in a figure by examining their positions in normalized figure coordinates.

    This assumes a rectangular grid of subplots laid out by something
    like plt.subplots(nrows, ncols) (i.e. uniform sized Axes).

    Returns
    -------
    (nrows, ncols) : tuple of int
    """
    # Get all Axes
    axes = fig.get_axes()

    # Filter to only "real" subplot Axes. We skip things like colorbars or inset_axes
    # A simple heuristic: check if it's an instance of SubplotBase.
    # Alternatively, you could skip any axes that appear to be legends or colorbars.
    real_subplots = []
    for ax in axes:
        # We also skip invisible or twin axes (to avoid duplicates).
        if (ax.get_visible() and
                isinstance(ax, matplotlib.axes.SubplotBase)):
            real_subplots.append(ax)

    if not real_subplots:
        return 0, 0  # No valid subplots found

    # Extract bounding boxes in normalized figure coords
    # We'll look at the center or the lower-left corner.
    # For consistent subplots, either approach works if they're arranged in a grid.
    x_positions = []
    y_positions = []
    for ax in real_subplots:
        bbox = ax.get_position()  # returns Bbox(x0, y0, x1, y1) in [0..1]
        # Let's use the center to avoid minor floating offsets on edges
        x_center = 0.5 * (bbox.x0 + bbox.x1)
        y_center = 0.5 * (bbox.y0 + bbox.y1)
        x_positions.append(x_center)
        y_positions.append(y_center)

    # Group unique centers for x and y, which correspond to columns and rows respectively
    # We'll define a small tolerance to group subplots in the same row/column
    # if their centers differ by less than some tiny threshold.
    def unique_positions(vals, tol=1e-3):
        # Sort them
        vals_sorted = sorted(vals)
        unique_vals = []
        current_group = None
        for v in vals_sorted:
            if current_group is None:
                current_group = v
                unique_vals.append(v)
            else:
                # if difference is large enough, we treat as a new group
                if abs(v - current_group) > tol:
                    unique_vals.append(v)
                    current_group = v
        return unique_vals

    unique_x = unique_positions(x_positions, tol=1e-3)
    unique_y = unique_positions(y_positions, tol=1e-3)

    ncols = len(unique_x)
    nrows = len(unique_y)

    return (nrows, ncols)


def adjust_figure_for_panel_size_auto(fig, panel_width=1.2, panel_height=1.2, do_second_tight=True):
    """
    Adjust an existing figure so that after tight_layout() the full grid of subplots
    has a total dimension of (ncols * panel_width) x (nrows * panel_height) inches,
    where nrows and ncols are automatically detected.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object (already containing subplots).
    panel_width : float
        Desired width (in inches) of each subplot *panel*.
    panel_height : float
        Desired height (in inches) of each subplot *panel*.
    do_second_tight : bool, optional
        If True, applies tight_layout() again after resizing the figure.
        This can slightly refine the result.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The same figure object, but resized so that the total subplot grid
        matches the target dimension (ncols * panel_width x nrows * panel_height).
    """
    # 1. Apply tight_layout so that subplot bounding boxes are computed
    fig.tight_layout()
    fig.canvas.draw()  # force a draw so that layout info is up to date

    # 2. Detect the grid shape from the Axes
    nrows, ncols = detect_subplot_grid_shape(fig)
    if nrows == 0 or ncols == 0:
        # Could not detect a valid grid - return unchanged
        return fig

    # 3. Current figure size in inches
    init_width, init_height = fig.get_size_inches()

    # 4. Collect bounding boxes of all "real" subplots

    ax = fig.get_axes()[0]
    bb = ax.get_position()
    xW = bb.x1 - bb.x0
    yH = bb.y1 - bb.y0


    subplot_width = xW * init_width
    subplot_height = yH * init_height

    panels_width = ncols * subplot_width
    panels_height = nrows * subplot_height

    padding_width = init_width - panels_width
    padding_height = init_height - panels_height

    new_panels_width = ncols * panel_width
    new_panels_height = nrows * panel_height

    new_fig_width = new_panels_width + padding_width
    new_fig_height = new_panels_height + padding_height

    fig.set_size_inches(new_fig_width, new_fig_height, forward=True)

    # 8. Optionally do a second pass
    if do_second_tight:
        fig.tight_layout()

    return fig

# def model(x, a, b, c):
#   return b/(1+np.exp(-a*(x-c)))
# def model(x, a, b, c):
#     return b + (a - b) * np.exp(-c * x)
def model(x,a,b,c,d):
    return a + (b-a)/(1+np.exp(-c*(x-d)))


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


def shuffle_time(df, subject_cols='Subject', trial_col='session trial'):
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


def calc_cmax(dy, dx, nTrials):
    # this ensures that the first time step does not account for more than [ntrials/(ntrials-1)]% of the total change
    if dy == 0:
        print('dy is zero, setting cmax to near-zero')
        cmax = 1e-10
    elif nTrials <= 1:
        raise Exception('nTrials is 1, cannot calculate cmax')
    else:
        dy = abs(dy)
        dx = abs(dx)
        maxDeltaY = dy/dx * (dx - (dx/nTrials))  # the average change in y per time step multiplied by n-1 time steps
        cmax = -np.log((maxDeltaY - dy)/-dy)  # the value of c that would produce maxDeltaY

    if np.isnan(cmax):
        print('dy: ' + str(dy))
        print('dx: ' + str(dx))
        print('nTrials: ' + str(nTrials))
        print('maxDeltaY: ' + str(maxDeltaY))
        raise Exception('cmax is nan')
    return cmax


def nonlinear_regression(data, subject_cols=['Subject'], trial_col='Trial', value_col='Value', ymin=None, ymax=None,
                         parallel=True):
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

    def parse_bounds(paramMax, paramMin):
        buffer = 1e-10
        if paramMax == paramMin:
            print('bound collision detected, adjusting max/min values')
            if ymax is None and ymin is None:
                paramMin = paramMin - buffer
                paramMax = paramMax + buffer
                print('new bounds: ' + str(paramMin) + ' ' + str(paramMax))
            else:
                if ymax is not None:
                    if paramMin == ymax:
                        paramMin = paramMin - buffer
                if ymin is not None:
                    if paramMax == ymin:
                        paramMax = paramMax + buffer
                print('new bounds: ' + str(paramMin) + ' ' + str(paramMax))
        if paramMax == paramMin:
            raise Exception('bounds still equal after adjustment')
        #if either paramMin or paramMax is nan, raise exception
        if np.isnan(paramMin) or np.isnan(paramMax):
            raise Exception('paramMin or paramMax is nan')
        return paramMin, paramMax

    def fit_for_subject(subject, subject_data):
        # sort subject data by trial_col
        subject_data = subject_data.copy().sort_values(by=[trial_col]).reset_index(drop=True)
        trials = subject_data[trial_col].to_numpy()
        values = subject_data[value_col].to_numpy()

        #if there are nans in values, remove them and the corresponding indices of trials
        nan_indices = np.argwhere(np.isnan(values))
        if len(nan_indices) > 0:
            trials = np.delete(trials, nan_indices)
            values = np.delete(values, nan_indices)

        nTrials = len(trials)
        if nTrials < 4:
            print('Not enough data points to fit model for subject ' + str(subject))
            return subject, (np.nan, np.nan, np.nan, np.nan), np.nan, np.nan

        valMax = np.max(values)
        valMin = np.min(values)
        if valMin == valMax:
            a0 = valMin
            b0 = valMax
            c0 = 0
            d0 = np.median(trials)
            params = [a0, b0, c0, d0]
            y_pred = model(trials, *params)
            r2 = r2_score(values, y_pred)
            return subject, params, r2, y_pred

        trialMax = np.max(trials)
        trialMin = np.min(trials)

        aMin = valMin
        aMax = valMax
        bMax = valMax
        bMin = valMin
        dMin = trials[1]
        dMax = trials[-2]
        d0 = np.median(trials)

        dy = abs(valMax - valMin)
        dx = abs(trialMax - trialMin)
        #if dy, dx, or nTrials are nan, raise exception
        if np.isnan(dy) or np.isnan(dx) or np.isnan(nTrials):
            print('valmax: ' + str(valMax))
            print('valmin: ' + str(valMin))
            print('values: ' + str(values))
            raise Exception('dy, dx, or nTrials is nan')

        cMax = calc_cmax(dy, dx, nTrials)
        cMin = 0

        #aMin, aMax = parse_bounds(aMax, aMin)
        #bMin, bMax = parse_bounds(bMax, bMin)
        cMin, cMax = parse_bounds(cMax, cMin)

        slope, intercept, r_value, p_value, std_err = stats.linregress(trials,
                                                                       values)  # first estimate bounds using linear regression
        y0 = slope * np.min(trials) + intercept  # calculate the y value at the min(trials)
        y1 = slope * np.max(trials) + intercept  # calculate the y value at the max(trials)
        c0 = abs(slope)  # slope is the initial guess for c

        if aMin <= y0 <= aMax:
            a0 = y0
        elif aMin <= values[0] <= aMax:
            a0 = values[0]
        else:
            #figure out if a0 is closer to aMin or aMax
            if abs(aMin - y0) < abs(aMax - y0):
                a0 = aMin
            else:
                a0 = aMax

        if bMin <= y1 <= bMax:
            b0 = y1
        elif bMin <= values[-1] <= bMax:
            b0 = values[-1]
        else:
            #figure out if b0 is closer to bMin or bMax
            if abs(bMin - y1) < abs(bMax - y1):
                b0 = bMin
            else:
                b0 = bMax

        if c0 > cMax:
            c0 = cMax
        elif c0 < cMin:
            c0 = cMin

        if not (cMin <= c0 <= cMax):
            raise Exception('c0 is not between cMin and cMax')

        if d0 > dMax or d0 < dMin:
            #d0 becomes the midpoint between dMin and dMax
            d0 = int(dMin + (dMax - dMin) / 2)


        params, _ = curve_fit(model, trials, values, p0=[a0, b0, c0, d0], bounds=[[aMin, bMin, cMin, dMin], [aMax, bMax, cMax, dMax]],
                              maxfev=10000000)
        y_pred = model(trials, *params)
        r2 = r2_score(values, y_pred)

        return subject, params, r2, y_pred

    if parallel:
        results = Parallel(n_jobs=-1)(
            delayed(fit_for_subject)(subject, subject_data) for subject, subject_data in data.groupby(subject_cols))
    else:
        results = []
        for subject, subject_data in data.groupby(subject_cols):
            results.append(fit_for_subject(subject, subject_data))

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

        # Calculate mean and standard deviation for each column in the sample, ignoring NaNs
        boot_means[i, :] = np.nanmean(sample, axis=axis)

    bootmean = np.nanmean(boot_means, axis=axis)
    bootci = np.nanpercentile(boot_means, [2.5, 97.5], axis=axis)

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

def calc_pred_change(trials, params):
    max_trial = max(trials)
    max_pr = model(max_trial, *params)
    min_trial = min(trials)
    min_pr = model(min_trial, *params)
    return max_pr - min_pr

def iter_shuffle(data, nIter=10000, subject_cols=['Subject'], trial_col='Trial', value_col='Value', ymin=None,
                 ymax=None, save_dir=None, overwrite=True, parallel=True, flag=None):

    shuffname = trial_col + '_' + value_col + '_nonlinearNullDist.feather'
    if flag:
        shuffname = flag + '_' + shuffname

    if overwrite is False:
        try:
            iters = pd.read_feather(save_dir + '/' + shuffname)
            #check if iters is actually a dataframe
            if type(iters) != pd.core.frame.DataFrame:
                raise ValueError('iters is not a dataframe, try re-running with overwrite=True')
            elif iters is None:
                raise ValueError('iters is None, try re-running with overwrite=True')
            else:
                return iters
        except FileNotFoundError:
            print('file ' + shuffname + ' not found, overwriting')
            overwrite = True
    if overwrite is True:
        if parallel:
            iters = iter_shuffle_parallel(data, nIter=nIter, subject_cols=subject_cols, trial_col=trial_col,
                                          value_col=value_col, save_dir=save_dir, ymin=ymin, ymax=ymax)
        else:
            iters = []
            for i in range(nIter):
                print("iter " + str(i) + " of " + str(nIter))
                shuff = shuffle_time(data, subject_cols=subject_cols, trial_col=trial_col)
                params, r2, _ = nonlinear_regression(shuff, subject_cols=subject_cols, trial_col=trial_col, value_col=value_col, ymin=ymin, ymax=ymax)

                paramdf = pd.Series(params, name='params').to_frame()
                r2df = pd.Series(r2, name='r2').to_frame()
                df = pd.concat([paramdf, r2df], axis=1).reset_index()
                cols = subject_cols + ['params', 'r2']
                df = df.set_axis(cols, axis=1)
                df['iternum'] = i
                iters.append(df)

            iters = pd.concat(iters, ignore_index=True)
        if save_dir is not None:
            iters.to_feather(save_dir + '/' + shuffname)
        return iters


from joblib import Parallel, delayed

def iter_shuffle_parallel(data, nIter=10000, subject_cols=['Subject'], trial_col='Time', value_col='Value', ymin=None,
                          ymax=None, save_dir=None, overwrite=True):
    def single_iteration(i):
        shuff = shuffle_time(data, subject_cols=subject_cols, trial_col=trial_col)
        params, r2, _ = nonlinear_regression(shuff, subject_cols=subject_cols, trial_col=trial_col, value_col=value_col,
                                             ymin=ymin, ymax=ymax)
        paramdf = pd.Series(params, name='params').to_frame()
        r2df = pd.Series(r2, name='r2').to_frame()
        df = pd.concat([paramdf, r2df], axis=1).reset_index()
        cols = subject_cols + ['params', 'r2']
        df = df.set_axis(cols, axis=1)
        df['iternum'] = i
        return df

    shuffname = trial_col + '_' + value_col + '_nonlinearNullDist.feather'
    if overwrite is False:
        try:
            iters = pd.read_feather(save_dir + '/' + shuffname)
            return iters
        except FileNotFoundError:
            pass

    iters = Parallel(n_jobs=-1)(delayed(single_iteration)(i) for i in range(nIter))
    iters = pd.concat(iters, ignore_index=True)

    if save_dir is not None:
        iters.to_feather(save_dir + '/' + shuffname)

    return iters


def get_shuff_pvals(shuff, r2_df, groups=['exp_group', 'exp_name', 'taste', 'session']):
    names = []
    pvals = []

    r2_df_gb = r2_df.groupby(groups)
    for name, group in shuff.groupby(groups):
        # get r2 for corresponding group in r2_df
        r2_group = r2_df_gb.get_group(name)
        pval = pval_from_null(group.r2.to_numpy(), r2_group.r2.to_numpy())
        pvals.append(pval)
        names.append(name)

    pval_df = pd.DataFrame(names, columns=groups)
    pval_df['pval'] = pvals
    return pval_df


exp_group_index = {'naive': 0, 'suc_preexp': 1, 'sucrose preexposed': 1, 'sucrose pre-exposed': 1}
taste_index = {'Suc': 0, 'NaCl': 1, 'CA': 2, 'QHCl': 3}
#session_index = {1: 0, 3: 1} #2: 1, 3: 2}
unique_tastes = ['Suc', 'NaCl', 'CA', 'QHCl']

#plot line graphs of each fit for each animal
def plot_fits(avg_gamma_mode_df, trial_col='session_trial', dat_col='pr(mode state)', model_col='modeled',
              time_col='time_group', save_dir=None, use_alpha_pos=False, flag=None):
    # make a column determining if alpha is positive
    unique_exp_groups = avg_gamma_mode_df['exp_group'].unique()
    unique_exp_names = avg_gamma_mode_df['exp_name'].unique()
    unique_tastes = ['Suc', 'NaCl', 'CA', 'QHCl']  # avg_shuff['taste'].unique()

    #unique_time_groups = np.sort(avg_gamma_mode_df[time_col].unique())
    unique_time_groups = [1, 2, 3]
    if use_alpha_pos == True:
        unique_alpha_pos = avg_gamma_mode_df['alpha_pos'].unique()
    n_tastes = len(unique_tastes)
    n_time_groups = len(unique_time_groups)
    n_exp_groups = len(unique_exp_groups)
    pal = sns.color_palette()
    colmap = {}
    n_time_groups = len(unique_time_groups)
    width = 2*n_time_groups + 2

    for i, exp in enumerate(unique_exp_names):
        colmap[exp] = i
    def plot_ind_fit(df):

        fig, axes = plt.subplots(n_tastes, n_time_groups, sharex=True, sharey=False, figsize=(width, 12))
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

                #if i == 0 and j == 0:
                #    ax.legend(labels=exp_group_names, loc='center right',
                #              bbox_to_anchor=(4.05, -1.3), ncol=1)
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
        plt.tight_layout()
        # plt.suptitle("HMM stereotypy: " + exp_group)


    # map a color to each exp name in unique exp names


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

                    savename = '/' + alpha_lab + '_' + model_col + '_' + trial_col + '_' + exp_group + '.svg'
                    plt.savefig(save_dir + savename)
        else:
            df = avg_gamma_mode_df[(avg_gamma_mode_df['exp_group'] == exp_group)]
            plot_ind_fit(df)
            if save_dir is not None:
                savename = model_col + '_' + trial_col + '_' + exp_group + '.svg'
                if flag is not None:
                    savename = flag + '_' + savename

                plt.savefig(save_dir + '/' + savename)

#plot line graph of fits for each taste
def plot_fits_summary(avg_gamma_mode_df, trial_col='session_trial', dat_col='pr(mode state)', model_col='modeled',
                      time_col='time_group', save_dir=None, use_alpha_pos=True, dotalpha=0.05, flag=None, nIter=100, r2df=None):
    # sort avg_gamma_mode_df by exp_group, time_group, taste, mapping the order
    sessions = avg_gamma_mode_df[time_col].unique()
    sessions.sort()
    n_sessions = len(sessions)
    session_idx = np.arange(n_sessions)
    #turn all session values into integers
    session_idx = session_idx.astype(int)
    #make a dictionary mapping session to index
    session_index = dict(zip(sessions, session_idx))


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
    ymax = avg_gamma_mode_df[dat_col].max()
    ymin = avg_gamma_mode_df[dat_col].min()
    bound01 = False
    if ymax <= 1 and ymin >= 0:
        bound01 = True


    plotwidth = 3*n_time_groups + 1
    def plot_ind_fit(df):
        fig, axes = plt.subplots(n_tastes, n_time_groups, sharex=True, sharey=True)
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
                    ds = []
                    model_res = []

                    # print the session and exp group and taste
                    print('session: ' + str(time_group), 'exp_group: ' + str(exp_group), 'taste: ' + str(taste))
                    for nm, grp in subset.groupby(['exp_name']):
                        alpha = grp['alpha'].unique()
                        beta = grp['beta'].unique()
                        c = grp['c'].unique()
                        d = grp['d'].unique()
                        # if len of alpha, beta, or c are greater than 1, raise exception
                        if len(alpha) > 1 or len(beta) > 1 or len(c) > 1:
                            raise Exception('More than one value for alpha, beta, or c')
                        alphas.append(alpha[0])
                        betas.append(beta[0])
                        cs.append(c[0])
                        ds.append(d[0])
                        modeled = model(unique_trials, alpha[0], beta[0], c[0], d[0])
                        model_res.append(modeled)
                    # turn model_res into a matrix
                    model_res = np.vstack(model_res)
                    # get the mean of each column
                    model_mean = np.nanmean(model_res, axis=0)

                    for p, row in subset.iterrows():
                        scatter = ax.plot(row[trial_col], row[dat_col], 'o', alpha=dotalpha, color=color,
                                          mfc='none')
                    line = ax.plot(unique_trials, model_mean, alpha=0.9, color=color)
                    if bound01:
                        ax.set_ylim(0, 1)

                    if i == 0 and j == 0:
                        legend_handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                                                      label=exp_group, alpha=1)
                        legend_handles.append(legend_handle)

                #if i == 0 and j == 0:
                    #ax.legend(handles=legend_handles, loc='center right',
                    #          bbox_to_anchor=(4.05, -1.3), ncol=1)
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
            plt.tight_layout()

        return fig, axes

    if flag is not None:
        save_str = model_col + '_' + trial_col + '_' + flag + '_' + 'summary.svg'
    else:
        save_str = model_col + '_' + trial_col + '_' + 'summary.svg'

    fig, axes = plot_ind_fit(avg_gamma_mode_df)
    if save_dir is not None:
        savename = '/' + save_str
        plt.savefig(save_dir + savename)

def mean_resample(col):
    s = resample(col)
    return np.nanmean(s)

def calc_boot(group, trial_col, nIter=1000, parallel=False):
    unique_trials = np.sort(group[trial_col].unique())
    params = group[['alpha','beta', 'c', 'd']]
    params = params.drop_duplicates(ignore_index=True) #get each unique model

    models = []
    #for id, grp in group.groupby(['exp_name', 'taste']):
    for i, row in params.iterrows():
        alpha = row['alpha']#grp['alpha'].unique()
        beta = row['beta']#grp['beta'].unique()
        c = row['c']#grp['c'].unique()
        d = row['d']#grp['d'].unique()
        models.append(model(unique_trials, alpha, beta, c, d))
    models = np.vstack(models)
    if len(models) == len(group):
        raise Exception('trials mishandled')
    # for each column in models, bootstrap the 95% ci
    boot_mean = []
    boot_low = []
    boot_high = []

    for col in models.T:
        if parallel:
            samples = Parallel(n_jobs=-1)(delayed(mean_resample)(col) for i in range(nIter))
        else:
            samples = []
            for iter in range(nIter):
                sample = resample(col)
                samples.append(np.nanmean(sample))
        boot_mean.append(np.nanmean(samples))
        bootci = np.nanpercentile(samples, [2.5, 97.5])
        boot_low.append(bootci[0])
        boot_high.append(bootci[1])

    return boot_mean, boot_low, boot_high


def plot_boot(ax, nm, group, color, trial_col='taste_trial', shade_alpha=0.2, nIter=1000, parallel=False):
    unique_trials = np.sort(group[trial_col].unique())+1
    boot_mean, boot_low, boot_high = calc_boot(group, trial_col, nIter=nIter, parallel=parallel)
    ax.fill_between(unique_trials, boot_low, boot_high, alpha=shade_alpha, color=color)
    ax.plot(unique_trials, boot_mean, alpha=1, color=color, linewidth=2)


def plot_scatter(ax, group, color, dat_col, trial_col='taste_trial', dotalpha=0.2):
    x = group[trial_col]+1
    y = group[dat_col]
    scatter = ax.plot(x, y, 'o', alpha=dotalpha, color=color,
                      mfc='none')


def plot_shuff(ax, group, trials, color, linestyle):
    # for each model in group, model data using column params:
    models = []
    trs = []
    for i, row in group.iterrows():
        params = row['params']
        models.append(model(trials, *params))
        trs.append(trials)
    models = np.vstack(models)
    trs = np.vstack(trs)
    #make models and trials 1d
    models = models.flatten()
    trials = trs.flatten()+1
    #model_mean = np.nanmean(models, axis=0)
    sns.lineplot(trials, models, alpha=0.5, color=color, linestyle=linestyle, ax=ax)
    #ax.plot(trials, model_mean, alpha=1, color=color, linestyle=linestyle, linewidth=1.5)

#plots the line graph of the average model for each session, with the data points overlaid, and the 95% confidence interval for the model
def plot_fits_summary_avg(df, shuff_df, trial_col='session_trial', dat_col='pr(mode state)', time_col='session',
                          save_dir=None, use_alpha_pos=False, dotalpha=0.1, textsize=20, flag=None, nIter=1000,
                          parallel=True, r2df=None, ymin=None, ymax=None):
    bound01 = False
    if ymin is not None and ymax is not None:
        if ymax <= 1 and ymin >= 0:
            bound01 = True


    sessions = df[time_col].unique()
    sessions.sort()
    n_sessions = len(sessions)
    session_idx = np.arange(n_sessions)
    # turn all session values into integers
    session_idx = session_idx.astype(int)
    # make a dictionary mapping session to index
    session_index = dict(zip(sessions, session_idx))

    # get the 97.5 and 2.5 percentile of df[dat_col], assign them to ymax and ymin
    ntiles = df[dat_col].quantile([0.01, 0.99])
    # ymin = ntiles[0.01]
    # ymax = ntiles[0.99]

    unique_trials = np.sort(df[trial_col].unique())
    unique_exp_groups = df['exp_group'].unique()
    unique_exp_names = df['exp_name'].unique()
    unique_tastes = ['Suc', 'NaCl', 'CA', 'QHCl']  # avg_shuff['taste'].unique()
    unique_time_groups = np.sort(df[time_col].unique())
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
        plotwidth = 3*n_time_groups + 1
        fig, axes = plt.subplots(1, n_time_groups, sharex=True, sharey=True)
        legend_handles = []
        # plot shuffled data:
        for nm, group in shuff_plot_df.groupby([time_col, 'session_index', 'exp_group', 'exp_group_index']):
            ax = axes[nm[1]]
            linestyle = shuff_linestyles[nm[3]]
            plot_shuff(ax, group, color='gray', trials=unique_trials, linestyle=linestyle)
            #if nm[1] == 0:
            #    labname = nm[2] + ' trial shuffle'
            #    legend_handle = mlines.Line2D([], [], color='gray', linestyle=linestyle, label=labname, alpha=1)
            #    legend_handles.append(legend_handle)

        # plot data points:
        # for nm, group in plot_df.groupby(['session', 'session_index', 'exp_group', 'exp_group_index']):
        #     ax = axes[nm[1]]
        #     color = pal[nm[3]]
        #     plot_scatter(ax, group, color=color, dat_col=dat_col, trial_col=trial_col, dotalpha=dotalpha)

        # plot average model:
        for nm, group in plot_df.groupby([time_col, 'session_index', 'exp_group', 'exp_group_index']):
            ax = axes[nm[1]]
            color = pal[nm[3]]
            plot_boot(ax, nm, group, color=color, nIter=nIter, trial_col=trial_col, parallel=parallel)

            #if nm[1] == 0:
            #    legend_handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None', label=nm[2], alpha=1)
            #    legend_handles.append(legend_handle)

        # title for each subplot column
        for ax, col in zip(axes, unique_time_groups):
            label = 'session ' + str(col)
            ax.set_title(label, rotation=0, size=textsize)
            ax.set_xlabel('trial', size=textsize, labelpad=0.1)
            ax.xaxis.set_tick_params(labelsize=textsize)
            if bound01:
                ax.set_ylim(0, 1)
            ax.set_xlim(1,30)
        # make y label "avg" + dat_col
        ax = axes[0]
        ylab = "avg " + dat_col
        ax.set_ylabel(ylab, rotation=90, size=textsize, labelpad=0)
        ax.yaxis.set_tick_params(labelsize=textsize)
        plt.tight_layout()
        #plt.tight_layout()

        #ax = axes[-1]
        #ax.legend(handles=legend_handles, loc='best', ncol=1, fontsize=textsize)

        return fig, axes


    save_str = dat_col + '_' + trial_col + '_' + 'summary'

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
                save_fig(fig, save_dir, 'line_graphs', savename, flag)
    else:
        fig, axes = make_grid_plot(df, shuff_df)
        fig = adjust_figure_for_panel_size_auto(fig)

        save_fig(fig, save_dir, 'line_graphs', save_str, flag)


def get_pval_stars(pval, adjustment=None):
    if adjustment is not None:
        pval = pval * adjustment
    if pval <= 0.001:
        return '***'
    elif pval <= 0.01:
        return '**'
    elif pval <= 0.05:
        return '*'
    else:
        return ''

def pval_from_null(null_dist, test_stat):
    #check if test_stat or null dist contain nans or inf
    if np.isnan(test_stat) or np.isinf(test_stat):
        raise Exception('Test statistic contains nan or inf')

    test_stat = float(test_stat)
    #check if null dist is longer than 1:
    if len(null_dist) < 100:
        print('Warning: Null distribution is too small')
    null_dist = np.array(null_dist)
    null_mean = np.nanmean(null_dist)
    if null_mean < test_stat:
        pval = np.nanmean(null_dist >= test_stat)
    else:
        pval = np.nanmean(null_dist <= test_stat)
    return pval

def bootstrap_mean_ci(data, n_bootstrap=100, ci=95):
    low = (100 - ci) / 2
    high = 100 - low

    bootstrap_means = np.array(
        [np.nanmean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)])
    lower_bound = np.nanpercentile(bootstrap_means, low)
    upper_bound = np.nanpercentile(bootstrap_means, high)
    return np.nanmean(bootstrap_means), lower_bound, upper_bound

def plot_bars(ax, r2_data, shuff_r2_data, label, bar_pos, bar_width, nIter, indices, color, n_comp=1, two_tailed=True, boot_data=True):
    #get the lower 0.025 and upper 0.975 percentiles for the shuff_r2
    #these data will form the floating bars representing the null distribution

    if two_tailed==True:
        pct_low = 2.5
        pct_high = 97.5
        low_adj = pct_low/n_comp
        high_adj = 100 - low_adj
    else:
        pct_low = 0
        pct_high = 95
        low_adj = 0
        high_adj = 100 - (5/n_comp)


    lower_bound = np.nanpercentile(shuff_r2_data, low_adj)
    upper_bound = np.nanpercentile(shuff_r2_data, high_adj)

    if boot_data:
        # Calculate the mean and confidence interval for the r2 data
        mean_r2, ci_lower, ci_upper = bootstrap_mean_ci(r2_data, n_bootstrap=nIter)
    else:
        ci_lower = np.nanpercentile(r2_data, pct_low)
        ci_upper = np.nanpercentile(r2_data, pct_high)
        mean_r2 = np.nanmean(r2_data)

        #set ci_lower, ci_upper, and mean_r2 to the mean of

    # Calculate the p-value for the r2 data
    p_val = pval_from_null(shuff_r2_data, mean_r2)
    p_val = p_val * n_comp
    if two_tailed: #adjust p-value for two-tailed test
        p_val = p_val * 2

    p_val_str = get_pval_stars(p_val)

    # plot the floating bar representing null distribution
    ax.bar(bar_pos, upper_bound - lower_bound, bar_width, bottom=lower_bound, color='gray',
                alpha=0.5,
                label=label if all(idx == 0 for idx in indices) else "")
    
    # Plot the mean r2 values
    ax.hlines(mean_r2, bar_pos - bar_width / 2, bar_pos + bar_width / 2,
                   color=color, lw=2)
    # Plot error bars for the confidence interval of the mean r2 values
    ax.errorbar(bar_pos, mean_r2, yerr=[[mean_r2 - ci_lower], [ci_upper - mean_r2]],
                     fmt='none', color=color, capsize=5)

    # print the p-value above the bar
    ax.text(bar_pos + 0.1, ci_upper + 0.01, p_val_str, ha='center', va='bottom',
                 color=color, fontsize=16, rotation='vertical')
    # print the actual p value rounded to 3 decimal places below the star
    ax.text(bar_pos + 0.1, ci_upper - 0.01, str(round(p_val, 3)),
            fontsize=8, ha='left', va='bottom', rotation='vertical')
    #ax.set_aspect('equal', adjustable='box')
#plots the bar graphs for the value col  
def plot_r2_pval_summary(shuff_r2_df, r2_df, stat_col=None, save_flag=None, save_dir=None, two_tailed=False, textsize=8, nIter=1000, n_comp=1):
    if stat_col is None:
        stat_col = 'r2'
    sessions = r2_df['session'].unique()
    sessions.sort()
    n_sessions = len(sessions)
    session_idx = np.arange(n_sessions)
    # turn all session values into integers
    session_idx = session_idx.astype(int)
    # make a dictionary mapping session to index
    session_index = dict(zip(sessions, session_idx))
    r2_df['session_index'] = r2_df['session'].map(session_index)
    shuff_r2_df['session_index'] = shuff_r2_df['session'].map(session_index)
    pal = sns.color_palette()
    colmap = {'naive': 0, 'suc_preexp': 1, 'sucrose preexposed': 1, 'sucrose pre-exposed': 1}

    tastes = ['Suc', 'NaCl', 'CA', 'QHCl']
    n_tastes = len(tastes) + 1

    # bootstrap mean r2 value with replacement for nIter iterations
    groups = ['exp_group', 'session', 'taste']
    # Get unique sessions
    sessions = shuff_r2_df['session'].unique()
    sessions.sort()
    n_sessions = len(sessions)

    # Iterate over each session and create a subplot
    exp_groups = r2_df['exp_group'].unique()
    # Width of each bar
    figW = 5
    figH = 1
    print(figW, figH)
    bar_width = 0.75
    for k, exp_group in enumerate(exp_groups):
        # Set up the subplots
        fig, axes = plt.subplots(1, n_sessions, sharey=True)
        #fig.set_figwidth(figW)
        for i, session in enumerate(sessions):
            ax = axes[i]
            # Plot bars for each taste
            for j, taste in enumerate(tastes):
                r2_data = r2_df[(r2_df['exp_group'] == exp_group) & (r2_df['session'] == session) & (r2_df['taste'] == taste)][stat_col]
                shuff_r2_data = shuff_r2_df[(shuff_r2_df['exp_group'] == exp_group) & (shuff_r2_df['session'] == session) & (shuff_r2_df['taste'] == taste)]
                shuff_r2_data = shuff_r2_data.groupby(['iternum']).mean().reset_index()[stat_col] #average across animals
                bar_pos = j# * bar_width
                indices = [j]
                color = pal[colmap[exp_group]]
                label = exp_group
                plot_bars(ax, r2_data, shuff_r2_data, label, bar_pos, bar_width, nIter, indices, color, two_tailed=two_tailed, n_comp=n_comp)

            # Plot overall percentile bars and mean r2 values
            r2_data = r2_df[(r2_df['exp_group'] == exp_group) & (r2_df['session'] == session)]
            overall_r2 = r2_data.groupby(['exp_name']).mean().reset_index()[stat_col] #average across tastes
            shuff_r2_data = shuff_r2_df[(shuff_r2_df['exp_group'] == exp_group) & (shuff_r2_df['session'] == session)]
            overall_shuff_r2 = shuff_r2_data.groupby(['iternum']).mean().reset_index()[stat_col] #average acros animals and tastes
            bar_pos = len(tastes)
            indices = [i]
            color = pal[colmap[exp_group]]
            label = exp_group
            plot_bars(ax, overall_r2, overall_shuff_r2, label, bar_pos, bar_width, nIter, indices, color, two_tailed=two_tailed, n_comp=n_comp)

            # Only set ylabel for the first subplot
            if i == 0:
                if stat_col == 'r2':
                    ax.set_ylabel('avg. r2 value', size=textsize)
                else:
                    ax.set_ylabel(stat_col, size=textsize)
                ax.yaxis.set_tick_params(labelsize=textsize)

            if stat_col == 'r2':
                ax.set_ylim(0, 1)
            # else:
            #     ax.set_ylim(ymin, ymax)
            # Set the x-ticks
            ax.set_xticks(np.arange(len(tastes) + 1))
            ax.set_xticklabels(tastes + ['Combined'], rotation=80, size=textsize)
            ax.xaxis.set_tick_params(labelsize=textsize)
            ax.set_title('session ' + str(session), rotation=0, size=textsize)

        plt.tight_layout()
        fig = adjust_figure_for_panel_size_auto(fig)
        savename = exp_group + '_' + stat_col + '_summary_bar_plot'
        foldername = 'day_avgs'
        save_fig(fig, save_dir, foldername, savename, save_flag=save_flag)

def save_fig(fig, save_dir, folder_name, savename, save_flag=None):
    if save_flag is not None:
        savename = save_flag + '_' + savename

    exts = ['.png', '.svg']
    for ext in exts:
        new_save_dir = os.path.join(save_dir, ext)
        new_save_dir = os.path.join(new_save_dir, folder_name)
        if not os.path.exists(new_save_dir):
            os.makedirs(new_save_dir)
        filename = savename + ext
        fig.savefig(os.path.join(new_save_dir, filename))


#this function plots just the average value + tests across tastes
def plot_r2_pval_avg(shuff_r2_df, r2_df, stat_col=None, save_flag=None, save_dir=None, two_tailed=False, textsize=8, nIter=100, n_comp=1):
    if stat_col is None:
        stat_col = 'r2'
    sessions = r2_df['session'].unique()
    sessions.sort()
    n_sessions = len(sessions)
    session_idx = np.arange(n_sessions)
    # turn all session values into integers
    session_idx = session_idx.astype(int)
    # make a dictionary mapping session to index
    session_index = dict(zip(sessions, session_idx))
    r2_df['session_index'] = r2_df['session'].map(session_index)
    shuff_r2_df['session_index'] = shuff_r2_df['session'].map(session_index)
    pal = sns.color_palette()
    colmap = {'naive': 0, 'suc_preexp': 1, 'sucrose preexposed': 1, 'sucrose pre-exposed': 1}

    tastes = ['Suc', 'NaCl', 'CA', 'QHCl']

    # bootstrap mean r2 value with replacement for nIter iterations
    groups = ['exp_group', 'session', 'taste']
    # Get unique sessions
    sessions = shuff_r2_df['session'].unique()
    sessions.sort()
    n_sessions = len(sessions)
    n_comp = n_sessions
    # Iterate over each session and create a subplot
    exp_groups = r2_df['exp_group'].unique()
    bar_width = 0.75
    figW = (1.2/5) * n_sessions
    for k, exp_group in enumerate(exp_groups):
        fig, ax = plt.subplots(1, 1, sharey=True)
        for i, session in enumerate(sessions):
            r2_data = r2_df[(r2_df['exp_group'] == exp_group) & (r2_df['session'] == session)][stat_col]
            shuff_r2_data = shuff_r2_df[(shuff_r2_df['exp_group'] == exp_group) & (shuff_r2_df['session'] == session)]
            shuff_r2_data = shuff_r2_data.groupby(['iternum']).mean().reset_index()[stat_col]
            bar_pos = i #+ k * bar_width
            indices = [i]
            color = pal[colmap[exp_group]]
            label = exp_group
            plot_bars(ax, r2_data, shuff_r2_data, label, bar_pos, bar_width, nIter, indices, color, two_tailed=two_tailed, n_comp=n_comp)

        # Only set ylabel for the first subplot
        if stat_col == 'r2':
            ax.set_ylabel('avg. r2 value', size=textsize)
        else:
            ax.set_ylabel(stat_col, size=textsize)
        ax.yaxis.set_tick_params(labelsize=textsize)

        if stat_col == 'r2':
            ax.set_ylim(0, 1)
        # else:
        #     ax.set_ylim(ymin, ymax)
        # Set the x-ticks
        ax.set_xticks(np.arange(len(sessions)))
        sessstr = [str(sess) for sess in sessions]
        ax.set_xticklabels(sessstr, size=textsize)
        ax.xaxis.set_tick_params(labelsize=textsize)
        #ax.xlabel('session', size=textsize)
        #set the x axis label to be 'session'
        ax.set_xlabel('session', size=textsize)

        handles = [mlines.Line2D([], [], color='gray', marker='s', linestyle='None', label='trial-shuffle 95% CI')]
        handles.append(mlines.Line2D([], [], color=pal[colmap[exp_group]], marker='s', linestyle='None', label='avg. of models'))
        #apply tight layout
        #plt.tight_layout()
        fig = adjust_figure_for_panel_size_auto(fig, panel_width=figW)
        # Add the legend to the figure
        #ax.legend(handles=handles, loc='best', fontsize=textsize * 0.8)
        #plt.subplots_adjust(**summary_margins)

        savename = exp_group + '_' + stat_col + '_avg_bar_plot'
        save_fig(fig, save_dir, 'day_avgs', savename, save_flag=save_flag)


difference_index = {'1-2': 0, '2-3': 1, '1-3': 2}
#plots difference in value col (normally r2) across sessions
def plot_daywise_r2_pval_diffs(shuff_r2_diffs, r2_diffs, stat_col=None, save_flag=None, save_dir=None, textsize=8, nIter=100, n_comp=None, ymin=None, ymax=None):
    if stat_col is None:
        stat_col = 'r2'
    diff_col = stat_col + ' difference'
    exp_groups = r2_diffs['exp_group'].unique()
    #r2_diffs['session_index'] = r2_diffs['session'].map(session_index)
    #shuff_r2_diffs['session_index'] = shuff_r2_diffs['session'].map(session_index)

    pal = sns.color_palette()
    colmap = {'naive': 0, 'suc_preexp': 1, 'sucrose preexposed': 1, 'sucrose pre-exposed': 1}

    tastes = ['Suc', 'NaCl', 'CA', 'QHCl']

    # bootstrap mean r2 diff value with replacement for nIter iterations
    groups = ['exp_group', 'taste', 'Session Difference']
    # Get unique session differences
    sess_diffs = shuff_r2_diffs['Session Difference'].unique()
    #order sess_diffs according to difference_index
    sess_diffs = sorted(sess_diffs, key=lambda x: difference_index[x])
    n_diffs = len(sess_diffs)

    if n_comp is None:
        n_comp = n_diffs

    avg_shuff = shuff_r2_diffs.groupby(groups + ['iternum']).mean().reset_index()
    avg_r2 = r2_diffs.groupby(groups).mean().reset_index()

    if ymin is None:
        ymin = min(avg_shuff[diff_col].min(), avg_r2[diff_col].min())
    if ymax is None:
        ymax = max(avg_shuff[diff_col].max(), avg_r2[diff_col].max())
    #if ymax > 2 or ymin < -2: set ymax and ymin to 2 and -2
    if stat_col == 'r2':
        if ymax > 1:
            ymax = 1
        if ymin < -1:
            ymin = -1

    # Set up the subplots
    # Iterate over each session and create a subplot
    # Width of each bar
    n_sessions = len(sess_diffs)
    n_tastes = len(tastes) + 1
    bar_width = 0.75
    for k, exp_group in enumerate(exp_groups):
        fig, axes = plt.subplots(1, n_diffs, sharey=True)
        for i, sess_diff in enumerate(sess_diffs):
            if n_diffs == 1:
                ax = axes
            else:
                ax = axes[i]
        # Plot bars for each taste
            for j, taste in enumerate(tastes):
                r2_data = r2_diffs[(r2_diffs['exp_group'] == exp_group) & (r2_diffs['Session Difference'] == sess_diff) & (r2_diffs['taste'] == taste)][diff_col]
                shuff_r2_data = shuff_r2_diffs[(shuff_r2_diffs['exp_group'] == exp_group) & (shuff_r2_diffs['Session Difference'] == sess_diff) & (shuff_r2_diffs['taste'] == taste)][diff_col]
                bar_pos = j #+ k * bar_width
                indices = [j]
                color = pal[colmap[exp_group]]
                label = exp_group
                plot_bars(ax, r2_data, shuff_r2_data, label, bar_pos, bar_width, nIter, indices, color, two_tailed=True, n_comp=n_comp)

        # Plot overall percentile bars and mean r2 values
            r2_data = r2_diffs[(r2_diffs['exp_group'] == exp_group) & (r2_diffs['Session Difference'] == sess_diff)]
            shuff_r2_data = shuff_r2_diffs[(shuff_r2_diffs['exp_group'] == exp_group) & (shuff_r2_diffs['Session Difference'] == sess_diff)]
            #get the mean of shuff_r2_data grouped by iternum
            overall_r2 = r2_data[diff_col]
            #overall_r2 = r2_data.groupby(['exp_name']).mean().reset_index()[diff_col] #average across tastes
            overall_shuff_r2_data = shuff_r2_data.groupby('iternum').mean().reset_index()[diff_col]
            bar_pos = len(tastes)
            indices = [i]
            color = pal[colmap[exp_group]]
            label = exp_group
            plot_bars(ax, overall_r2, overall_shuff_r2_data, label, bar_pos, bar_width, nIter, indices, color, two_tailed=True, n_comp=n_comp)

            # Only set ylabel for the first subplot
            if i == 0:
                ax.yaxis.set_tick_params(labelsize=textsize)

            ax.set_ylim(ymin, ymax)
            # Set the x-ticks
            ax.set_xticks(np.arange(len(tastes) + 1))
            ax.set_xticklabels(tastes + ['Combined'], rotation=60, size=textsize)
            ax.xaxis.set_tick_params(labelsize=textsize)
        if n_diffs > 1:
            for ax, col in zip(axes, sess_diffs):
                label = 'sessions ' + str(col)
                ax.set_title(label, rotation=0, size=textsize)
        else:
            label = 'sessions ' + str(sess_diffs[0])
            axes.set_title(label, rotation=0, size=textsize)

        handles = [mlines.Line2D([], [], color='gray', marker='s', linestyle='None', label='trial-shuffle 95% CI')]
        handles.append(mlines.Line2D([], [], color=pal[colmap[exp_group]], marker='s', linestyle='None', label='avg. of models'))

        lab = '$\Delta $ ' + stat_col
        # Add the legend to the figure
        if n_diffs > 1:
            #axes[-1].legend(handles=handles, loc='best', fontsize=textsize)
            axes[0].set_ylabel(lab, size=textsize)
        else:
            #axes.legend(handles=handles, loc='best', fontsize=textsize)
            axes.set_ylabel(lab, size=textsize)

        fig = adjust_figure_for_panel_size_auto(fig)
        #plt.tight_layout()

    ####################################################################################################
    # save the figure as png
        savename = exp_group + '_' + stat_col + '_day_diff_plot'
        save_fig(fig, save_dir, 'session_diffs', savename, save_flag=save_flag)


def plot_daywise_avg_diffs(shuff_r2_diffs, r2_diffs, stat_col=None, save_flag=None, save_dir=None, textsize=8, nIter=100, n_comp=None, ymin=None, ymax=None):
    if stat_col is None:
        stat_col = 'r2'
    diff_col = stat_col + ' difference'
    exp_groups = r2_diffs['exp_group'].unique()
    #r2_diffs['session_index'] = r2_diffs['session'].map(session_index)
    #shuff_r2_diffs['session_index'] = shuff_r2_diffs['session'].map(session_index)

    pal = sns.color_palette()
    colmap = {'naive': 0, 'suc_preexp': 1, 'sucrose preexposed': 1, 'sucrose pre-exposed': 1}

    # bootstrap mean r2 diff value with replacement for nIter iterations
    groups = ['exp_group', 'taste', 'Session Difference']
    # Get unique session differences
    sess_diffs = shuff_r2_diffs['Session Difference'].unique()
    #order sess_diffs according to difference_index
    sess_diffs = sorted(sess_diffs, key=lambda x: difference_index[x])
    n_diffs = len(sess_diffs)

    if n_comp is None:
        n_comp = n_diffs

    figW = (1.2/5) * n_diffs

    avg_shuff = shuff_r2_diffs.groupby(groups + ['iternum']).mean().reset_index()
    avg_r2 = r2_diffs.groupby(groups).mean().reset_index()

    if ymin is None:
        ymin = min(avg_shuff[diff_col].min(), avg_r2[diff_col].min())
    if ymax is None:
        ymax = max(avg_shuff[diff_col].max(), avg_r2[diff_col].max())
    #if ymax > 2 or ymin < -2: set ymax and ymin to 2 and -2
    if stat_col == 'r2':
        if ymax > 1:
            ymax = 1
        if ymin < -1:
            ymin = -1

    # Set up the subplots
    # Iterate over each session and create a subplot
    # Width of each bar
    bar_width = 0.75
    for k, exp_group in enumerate(exp_groups):
        fig, ax = plt.subplots(1, 1)
        for i, sess_diff in enumerate(sess_diffs):
        # Plot overall percentile bars and mean r2 values
            r2_data = r2_diffs[(r2_diffs['exp_group'] == exp_group) & (r2_diffs['Session Difference'] == sess_diff)]
            shuff_r2_data = shuff_r2_diffs[(shuff_r2_diffs['exp_group'] == exp_group) & (shuff_r2_diffs['Session Difference'] == sess_diff)]
            #get the mean of shuff_r2_data grouped by iternum
            overall_r2 = r2_data[diff_col]#.groupby(['exp_name']).mean().reset_index()[diff_col] #average across tastes
            overall_shuff_r2_data = shuff_r2_data.groupby('iternum').mean().reset_index()[diff_col]
            bar_pos = i #* 0.25 + bar_width
            indices = [i]
            color = pal[colmap[exp_group]]
            label = exp_group
            plot_bars(ax, overall_r2, overall_shuff_r2_data, label, bar_pos, bar_width, nIter, indices, color, two_tailed=True, n_comp=n_comp)

        ax.set_ylabel(diff_col, size=textsize)
        ax.yaxis.set_tick_params(labelsize=textsize)

        ax.set_ylim(ymin, ymax)
        # Set the x-tick labels
        ax.set_xticks(np.arange(len(sess_diffs)))
        ax.set_xticklabels(sess_diffs, size=textsize)
        ax.set_xlabel('session', size=textsize)
        lab = '$\Delta $ ' + stat_col
        ax.set_ylabel(lab, size=textsize)

        fig = adjust_figure_for_panel_size_auto(fig, panel_width=figW)

    ####################################################################################################
    # save the figure as png
        savename = exp_group + '_' + stat_col + '_avg_day_diff_plot'
        save_fig(fig, save_dir, 'session_diffs', savename, save_flag=save_flag)


#plots difference in value col (normally r2) between groups
#shuff_r2_df should already be averaged across groups (no individual exp names)
def plot_r2_pval_diffs_summary(shuff_r2_df, r2_df, stat_col=None, save_flag=None, save_dir=None, textsize=8, nIter=100, n_comp=None, ymin=None, ymax=None):
    #shuff r2 df should be averaged across exp_names
    if stat_col is None:
        stat_col = 'r2'
    if ymin is None:
        ymin = min(shuff_r2_df[stat_col].min(), r2_df[stat_col].min())
    if ymax is None:
        ymax = max(shuff_r2_df[stat_col].max(), r2_df[stat_col].max())

    sessions = r2_df['session'].unique()
    sessions.sort()
    n_sessions = len(sessions)
    if n_comp is None:
        n_comp = n_sessions

    session_idx = np.arange(n_sessions)
    # turn all session values into integers
    session_idx = session_idx.astype(int)
    # make a dictionary mapping session to index
    session_index = dict(zip(sessions, session_idx))

    unique_exp_groups = r2_df['exp_group'].unique()
    unique_time_groups = r2_df['session'].unique()
    r2_df['session_index'] = r2_df['session'].map(session_index)
    shuff_r2_df['session_index'] = shuff_r2_df['session'].map(session_index)
    pal = sns.color_palette()
    colmap = {'naive': 0, 'suc_preexp': 1, 'sucrose preexposed': 1, 'sucrose pre-exposed': 1}

    tastes = ['Suc', 'NaCl', 'CA', 'QHCl']

    # make a new df called shuff_diff, wherein for each grouping of session, taste, and iternum of shuff_r2_df, calculate the difference in r2 between each of the two exp_groups
    grouping_cols = ['session', 'taste', 'iternum']
    shuff_diffs = []
    indices = []
    for name, group in shuff_r2_df.groupby(grouping_cols):
        # get the r2 values for each exp_group
        diffs = []
        for exp_group in unique_exp_groups:
            r2 = float(group[group['exp_group'] == exp_group][stat_col])
            diffs.append(r2) # append to shuff_diff
        diff = abs(diffs[0] - diffs[1]) # calculate the difference in r2 between the two exp_groups
        shuff_diffs.append(diff) # append to shuff_diff
        indices.append(list(name))
    shuff_diff = pd.DataFrame(indices, columns=grouping_cols)
    shuff_diff['r2_diff'] = shuff_diffs

    overall_shuff_diffs = []
    overall_indices = []
    for name, group in shuff_r2_df.groupby(['session', 'iternum']):
        # get the r2 values for each exp_group
        diffs = []
        for exp_group in unique_exp_groups:
            r2 = group[group['exp_group'] == exp_group][stat_col].mean() #average across tastes
            diffs.append(r2) # append to shuff_diff
        diff = abs(diffs[0] - diffs[1]) # calculate the difference in r2 between the two exp_groups
        overall_shuff_diffs.append(diff) # append to shuff_diff
        overall_indices.append(list(name))
    overall_shuff_diff = pd.DataFrame(overall_indices, columns=['session', 'iternum'])
    overall_shuff_diff['r2_diff'] = overall_shuff_diffs

    # bootstrap mean r2 value with replacement for nIter iterations
    groups = ['exp_group', 'session', 'taste']
    boot_means = []
    ids = []
    iteridx = []
    for i in range(nIter):
        print(i)
        for name, group in r2_df.groupby(groups):
            k = len(group)
            mean = np.nanmean(random.choices(group[stat_col].to_numpy(), k=k))
            boot_means.append(mean)
            ids.append(name)
            iteridx.append(i)
    boot_mean_r2 = pd.DataFrame(ids, columns=groups)
    boot_mean_r2[stat_col] = boot_means
    boot_mean_r2['iternum'] = iteridx
    #get taste-averaged r2 values
    overall_boot_means = []
    overall_ids = []
    for name, group in boot_mean_r2.groupby(['exp_group', 'session', 'iternum']): #get each group
        mean = group[stat_col].mean() #average across tastes
        overall_boot_means.append(mean)
        overall_ids.append(name)
    overall_boot_mean_r2 = pd.DataFrame(overall_ids, columns=['exp_group', 'session', 'iternum'])
    overall_boot_mean_r2[stat_col] = overall_boot_means

    # get the mean r2 difference in the bootstrapped r2 data
    groups = ['session', 'taste', 'iternum']
    boot_r2_diffs = []
    ids = []
    for nm, group in boot_mean_r2.groupby(groups):
        diffs = []
        for exp_group in unique_exp_groups:
            r2 = float(group[group['exp_group'] == exp_group][stat_col]) #get the value of r2 for the exp group
            diffs.append(r2)
        diff = abs(diffs[0] - diffs[1]) #get the diffs between exp groups
        boot_r2_diffs.append(diff)
        ids.append(nm)
    boot_mean_r2_diff = pd.DataFrame(ids, columns=groups)
    boot_mean_r2_diff['r2_diff'] = boot_r2_diffs

    # get the mean r2 difference in the bootstrapped r2 data averaged across tastes
    overall_boot_r2_diffs = []
    overall_ids = []
    for nm, group in overall_boot_mean_r2.groupby(['session', 'iternum']):
        diffs = []
        for exp_group in unique_exp_groups:
            r2 = float(group[group['exp_group'] == exp_group][stat_col]) #get the singular value of r2 for the exp group
            diffs.append(r2)
        diff = abs(diffs[0] - diffs[1])
        overall_boot_r2_diffs.append(diff)
        overall_ids.append(nm)
    overall_boot_mean_r2_diff = pd.DataFrame(overall_ids, columns=['session', 'iternum'])
    overall_boot_mean_r2_diff['r2_diff'] = overall_boot_r2_diffs


    # Get unique sessions
    sessions = shuff_r2_df['session'].unique()
    sessions.sort()
    n_sessions = len(sessions)
    n_tastes = len(tastes) + 1

    # Set up the subplots
    figW = 1
    figH = 1
    fig, axes = plt.subplots(1, n_sessions, figsize=(figW, figH), sharey=True)
    # Iterate over each session and create a subplot
    exp_groups = r2_df['exp_group'].unique()
    # Width of each bar
    bar_width = 0.25
    for i, session in enumerate(sessions):
        ax = axes[i]
        #plot the data for each taste
        for j, taste in enumerate(tastes):
            for k, exp_group in enumerate(exp_groups):
                r2_data = \
                boot_mean_r2[(boot_mean_r2['exp_group'] == exp_group) & (boot_mean_r2['session'] == session) & (boot_mean_r2['taste'] == taste)][stat_col]
                shuff_r2_data = shuff_r2_df[
                    (shuff_r2_df['exp_group'] == exp_group) & (shuff_r2_df['session'] == session) & (
                                shuff_r2_df['taste'] == taste)][stat_col]

                bar_pos = j #+ k * bar_width
                indices = [j, k]
                color = pal[colmap[exp_group]]
                label = exp_group
                plot_bars(ax, r2_data, shuff_r2_data, label, bar_pos, bar_width, nIter, indices, color, n_comp=n_comp, boot_data=False)
            # plot the data for difference between exp_groups
            r2_data = \
            boot_mean_r2_diff[(boot_mean_r2_diff['session'] == session) & (boot_mean_r2_diff['taste'] == taste)][
                'r2_diff']
            shuff_r2_data = shuff_diff[(shuff_diff['session'] == session) & (shuff_diff['taste'] == taste)][
                'r2_diff']
            bar_pos = len(tastes)+1 #+ k * bar_width
            indices = [j, k + 1]
            label = '|$ \Delta $|'
            plot_bars(ax, r2_data, shuff_r2_data, label, bar_pos, bar_width, nIter, indices, color='black', n_comp=n_comp, boot_data=False)

        #plot the data for the average across tastes
        for k, exp_group in enumerate(exp_groups):
            r2_data = overall_boot_mean_r2[(overall_boot_mean_r2['exp_group'] == exp_group) & (overall_boot_mean_r2['session'] == session)][stat_col]
            shuff_r2_data = shuff_r2_df[(shuff_r2_df['exp_group'] == exp_group) & (shuff_r2_df['session'] == session)]
            shuff_r2_data = shuff_r2_data.groupby(['iternum']).mean().reset_index()[stat_col] #average across tastes
            bar_pos = len(tastes) + 1#k * bar_width
            indices = [i]
            color = pal[colmap[exp_group]]
            label = exp_group
            plot_bars(ax, r2_data, shuff_r2_data, label, bar_pos, bar_width, nIter, indices, color, n_comp=n_comp, boot_data=False)
        # plot the overall difference between exp_groups
        r2_data = overall_boot_mean_r2_diff[(overall_boot_mean_r2_diff['session'] == session)]['r2_diff']
        shuff_r2_data = overall_shuff_diff[(overall_shuff_diff['session'] == session)]['r2_diff']
        bar_pos = len(tastes) + (k + 1) * bar_width
        indices = [i]
        label = '|$\Delta$|'
        plot_bars(ax, r2_data, shuff_r2_data, label, bar_pos, bar_width, nIter, indices, color='black', boot_data=False, n_comp=n_comp)

        # Only set ylabel for the first subplot
        if i == 0:
            axes[i].set_ylabel('r2 Value', size=textsize)
            axes[i].yaxis.set_tick_params(labelsize=textsize)

        axes[i].set_ylim(0, 1)
        # Set the x-ticks
        axes[i].set_xticks(np.arange(len(tastes) + 1) + bar_width / 2)
        axes[i].set_xticklabels(tastes + ['Combined'], rotation=60, size=textsize)
        axes[i].xaxis.set_tick_params(labelsize=textsize)

        for ax in axes:
            ax.set_ylim(ymin, ymax)

    handles = [mlines.Line2D([], [], color='gray', marker='s', linestyle='None', label='trial-shuffle 95% CI')]
    for k, group in enumerate(exp_groups):
        label = group
        handles.append(mlines.Line2D([], [], color=pal[colmap[group]], marker='s', linestyle='None', label=label))
    handles.append(mlines.Line2D([], [], color='black', marker='s', linestyle='None', label='|$\Delta$|'))

    # Add the legend to the figure
    #axes[-1].legend(handles=handles, loc='best', fontsize=textsize)
    #plt.tight_layout()
    plt.tight_layout()

    ####################################################################################################
    # save the figure as png
    savename = stat_col + '_diffs_summary_bar_plot'
    save_fig(fig, save_dir, 'group_diffs', savename, save_flag=save_flag)


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
    n_time_groups = len(unique_time_groups)
    plotwidth = n_time_groups * 3 + 1
    fig, axes = plt.subplots(4, n_time_groups, sharex=True, sharey=True, figsize=(plotwidth, 10))
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
                pval = pval_from_null(avg_shuff_subset.r2, float(row.r2))
                pvals.append(pval)
                textpos = ypos[row['exp_group']]
                color = pal[colmap[row['exp_group']]]
                ax.hist(x=avg_shuff_subset.r2, bins=20, density=True, alpha=0.5, color=color)
                ax.axvline(x=row['r2'], color=color, linestyle='--')  # Customize color and linestyle
                ax.set_xlim(xmin, xmax)
                # ax.set_ylim(0, 10)
                # print the p-value with the color code
                pvaltext = "p = " + str(np.round(pval, 3))
                ax.text(0.95, textpos - 0.2, pvaltext, transform=ax.transAxes, color=color, horizontalalignment='right')

                r2text = "r2 = " + str(np.round(row['r2'], 3))
                ax.text(0.95, textpos, r2text, transform=ax.transAxes, color=color, horizontalalignment='right')
                if i == 0 and j == 0:
                    legend_handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None', label=exp_group,
                                                  alpha=1)
                    legend_handles.append(legend_handle)
            if i == 0 and j == 0:
                legend_handles.append(mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='|$\Delta$|'))
                #ax.legend(handles=legend_handles, loc='center right', bbox_to_anchor=(4.05, -1.3), ncol=1)
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

    plt.tight_layout()
    #plt.tight_layout()
    # save the figure as png
    if save_dir is not None:
        if save_flag is None:
            savename = '/r2_perm_test.png'
        else:
            savename = '/' + save_flag + '_r2_perm_test.png'
        plt.savefig(save_dir + savename)

def preprocess_nonlinear_regression(df, subject_cols, group_cols, trial_col, value_col, ymin=None, ymax=None, parallel=True, overwrite=False, nIter=10000, save_dir=None, flag=None):
    if type(subject_cols) is list:
        groupings = subject_cols + group_cols
    elif type(subject_cols) is str:
        groupings = [subject_cols] + group_cols
    else:
        raise ValueError('subject_cols must be a string or a list of strings')
    params, r2, y_pred = nonlinear_regression(df, subject_cols=groupings, trial_col=trial_col, value_col=value_col,
                                                 parallel=parallel, ymin=ymin, ymax=ymax)
    r2_df = pd.Series(r2).reset_index()  # turn dict into a series with multi-index
    r2_df = r2_df.reset_index(drop=True)  # reset the index so it becomes a dataframe
    colnames = groupings + ['r2']
    column_mapping = dict(zip(r2_df.columns, colnames))
    r2_df = r2_df.rename(columns=column_mapping)

    identifiers = []
    alpha = []
    beta = []
    c = []
    d = []
    for i, group in df.groupby(groupings):
        pr = params[i]
        identifiers.append(i)
        alpha.append(pr[0])
        beta.append(pr[1])
        c.append(pr[2])
        d.append(pr[3])

    ids_df = pd.DataFrame(identifiers, columns=groupings)
    params_df = ids_df.copy()
    params_df['alpha'] = alpha
    params_df['beta'] = beta
    params_df['c'] = c
    params_df['d'] = d

    # merge the params with df
    df2 = df.merge(params_df, on=groupings)
    df3 = df2.merge(r2_df, on=groupings)
    # for each row of df3, get the modeled value for that row, by passing the trial number and alpha, beta, c values to the model function
    modeled = []
    for i, row in df3.iterrows():
        pr = row[['alpha', 'beta', 'c', 'd']]
        mod = model(row[trial_col], *pr)
        modeled.append(mod)
    df3['modeled'] = modeled
    # get the null distribution of r2 values
    shuffle = iter_shuffle(df3, nIter=nIter, subject_cols=groupings, trial_col=trial_col, value_col=value_col,
                            ymin=ymin, ymax=ymax, save_dir=save_dir, overwrite=overwrite, parallel=parallel, flag=flag)

    if shuffle is [] or shuffle is None:
        raise ValueError('shuffle is None')
    return df3, shuffle

def plot_nonlinear_regression_stats(df3, shuff, subject_cols, group_cols, trial_col, value_col, flag=None, nIter=100, textsize=8, ymin=None, ymax=None, save_dir=None):
    if type(subject_cols) is list:
        gr = subject_cols + group_cols
    elif type(subject_cols) is str:
        gr = [subject_cols] + group_cols
    avg_shuff = shuff.groupby(gr + ['iternum']).mean().reset_index() #average across exp_names
    avg_df3 = df3.groupby(gr).mean().reset_index() #trial average df3

    for exp_group, group in avg_df3.groupby(['exp_group']):
        avg_group_shuff = avg_shuff.groupby('exp_group').get_group(exp_group)

        if flag is not None:
            save_flag = trial_col + '_' + value_col + '_' + flag
        else:
            save_flag = trial_col + '_' + value_col + '_' + exp_group + '_only'
        #this plots the average, 95% CI, null dist and pval for each taste and average across tastes
        plot_r2_pval_summary(avg_group_shuff, group, save_flag=save_flag, save_dir=save_dir, textsize=textsize, nIter=nIter)
        #this plots the average, 95% CI, null dist and pval the average across tastes
        plot_r2_pval_avg(avg_group_shuff, group, save_flag=save_flag, save_dir=save_dir, textsize=textsize, nIter=nIter)

#TODO: fix issus with this:
def get_session_differences(df3, shuff, stat_col='r2', group_cols=['exp_group', 'taste'], subject_cols = ['exp_name'], parallel=False):
    diff_col = stat_col + ' difference'
    relevant_cols = [stat_col] + group_cols + subject_cols
    df4 = df3[relevant_cols].drop_duplicates(ignore_index=True).reset_index(drop=True)
    #make a new dataframe called day_diffs that contains the difference in r2 between day 1 and day 2, and day 1 and day 3, and day 2 and day 3
    def calculate_differences(group_name, group_df):
        sessions = group_df['session'].unique()
        sessions.sort()
        res = []
        for i in sessions[:-1]:
            for j in sessions[1:]:
                if i != j:
                    idf = group_df[group_df['session'] == i]
                    jdf = group_df[group_df['session'] == j]
                    if len(idf) > 1 or len(jdf) > 1:
                        raise ValueError('more than one row for a session')
                    diff = idf.at[idf.index[0], stat_col] - jdf.at[jdf.index[0], stat_col]
                    session_diff = f"{i}-{j}"
                    res.append({'Group': group_name, 'Session Difference': session_diff, diff_col: diff})
        return res

    group_columns = group_cols.copy() #+ subject_cols
    #remove session from group_columns
    #check if group_columns contains 'session', if not, add it back in
    grouped = df4.groupby(group_columns).mean().reset_index()
    group_columns.remove('session')
    grouped = grouped.groupby(group_columns)
    if parallel==True:
        results = Parallel(n_jobs=-1)(delayed(calculate_differences)(group_name, group_df.sort_values('session')) for group_name, group_df in grouped)
    else:
        results = []
        for group_name, group_df in grouped:
            results.append(calculate_differences(group_name, group_df.sort_values('session')))
    flat_results = [item for sublist in results for item in sublist]
    r2_diffs = pd.DataFrame(flat_results)
    expanded_groups = pd.DataFrame(r2_diffs['Group'].tolist(), columns=group_columns)
    # Concatenate the new columns with the original DataFrame
    r2_diffs = pd.concat([expanded_groups, r2_diffs.drop('Group', axis=1)], axis=1)

    # Apply function to shuff_r2_df
    shuff_grpcols = group_cols.copy() + ['iternum']
    grpd_shuff = shuff.groupby(shuff_grpcols).mean().reset_index()
    shuff_grpcols.remove('session')
    grpd_shuff = grpd_shuff.groupby(shuff_grpcols)
    results = Parallel(n_jobs=-1)(delayed(calculate_differences)(group_name, group_df.sort_values('session')) for group_name, group_df in grpd_shuff)
    flat_results = [item for sublist in results for item in sublist]
    shuff_r2_diffs = pd.DataFrame(flat_results)
    expanded_groups = pd.DataFrame(shuff_r2_diffs['Group'].tolist(), columns=shuff_grpcols)
    # Concatenate the new columns with the original DataFrame
    shuff_r2_diffs = pd.concat([expanded_groups, shuff_r2_diffs.drop('Group', axis=1)], axis=1)
    #mean groups should be group_columns minus 'exp_name', plus 'Session Difference'
    mean_groups = shuff_grpcols + ['Session Difference']
    #mean_groups.remove('session')
    shuff_r2_diffs = shuff_r2_diffs.groupby(mean_groups).mean().reset_index()
    return r2_diffs, shuff_r2_diffs

def plot_session_differences(df3, shuff, subject_cols, group_cols, trial_col, value_col, stat_col=None, flag=None, nIter=100, textsize=20, ymin=None, ymax=None, save_dir=None):
    if stat_col is None:
        stat_col = 'r2'

    r2_diffs, shuff_r2_diffs = get_session_differences(df3, shuff, stat_col=stat_col, group_cols=group_cols, subject_cols=subject_cols)

    if flag is not None:
        save_flag = trial_col + '_' + value_col + '_' + flag
    else:
        save_flag = trial_col + '_' + value_col
    plot_daywise_r2_pval_diffs(shuff_r2_diffs, r2_diffs, stat_col=stat_col, save_flag=save_flag, save_dir=save_dir, textsize=textsize, nIter=nIter, ymin=ymin, ymax=ymax)
    plot_daywise_avg_diffs(shuff_r2_diffs, r2_diffs, stat_col=stat_col, save_flag=save_flag, save_dir=save_dir, textsize=textsize, nIter=nIter, ymin=ymin, ymax=ymax)

def get_pred_change(df3, shuff, subject_cols, group_cols, trial_col):
    groups = subject_cols + group_cols
    trials = df3[trial_col].unique()

    avg_df3 = df3.groupby(groups).mean().reset_index() #trial average df3
    def add_pred_change(df):
        pred_change = []
        for i, row in df.iterrows():
            params = row[['alpha', 'beta', 'c', 'd']]
            pred_change.append(calc_pred_change(trials, params))
        df['pred. change'] = pred_change
        return df

    pred_change_df = add_pred_change(avg_df3)

    pred_change_shuff = []
    for i, row in shuff.iterrows():
        params = row['params']
        pred_change_shuff.append(calc_pred_change(trials, params))
    shuff['pred. change'] = pred_change_shuff
    return pred_change_df, shuff

def get_params_df(df3, shuff, subject_cols, group_cols, trial_col):
    groups = [subject_cols] + group_cols
    trials = df3[trial_col].unique()

    avg_df3 = df3.groupby(groups).mean().reset_index() #trial average df3
    def add_pred_change(df):
        pred_change = []
        for i, row in df.iterrows():
            params = row[['alpha', 'beta', 'c', 'd']]
            pred_change.append(calc_pred_change(trials, params))
        df['pred. change'] = pred_change
        return df

    pred_change_df = add_pred_change(avg_df3)

    pred_change_shuff = []
    for i, row in shuff.iterrows():
        params = row['params']
        pred_change_shuff.append(calc_pred_change(trials, params))
    shuff['pred. change'] = pred_change_shuff
    return pred_change_df, shuff

def plot_predicted_change(pred_change_df, pred_change_shuff, group_cols, trial_col, value_col, flag=None, nIter=100, textsize=20, ymin=None, ymax=None, save_dir=None):

    avg_shuff = pred_change_shuff.groupby(group_cols + ['iternum']).mean().reset_index()

    if flag is not None:
        save_flag = trial_col + '_' + value_col + '_' + flag
    else:
        save_flag = trial_col + '_' + value_col
    plot_r2_pval_summary(avg_shuff, pred_change_df, stat_col='pred. change', save_flag=save_flag, save_dir=save_dir, two_tailed=True, textsize=textsize, nIter=nIter)
    plot_r2_pval_avg(avg_shuff, pred_change_df, stat_col='pred. change', save_flag=save_flag, save_dir=save_dir, textsize=textsize, two_tailed=True, nIter=nIter)

def plot_nonlinear_line_graphs(df3, shuff, subject_cols, group_cols, trial_col, value_col, save_dir=None, flag=None, nIter=100, parallel=True, ymin=None, ymax=None, textsize=20):
    groups = [subject_cols] + group_cols
    if ymin is None and ymax is None:
        ymin = min(df3[value_col])
        ymax = max(df3[value_col])

    plot_fits_summary_avg(df3, shuff_df=shuff, dat_col=value_col, trial_col=trial_col, save_dir=save_dir,
                             use_alpha_pos=False, textsize=textsize, dotalpha=0.15, flag=flag, nIter=nIter,
                             parallel=parallel, ymin=ymin, ymax=ymax)
    for exp_group, group in df3.groupby(['exp_group']):
        group_shuff = shuff.groupby('exp_group').get_group(exp_group)
        if flag is not None:
            save_flag = exp_group + '_' + flag
        else:
            save_flag = exp_group
        plot_fits_summary_avg(group, shuff_df=group_shuff, dat_col=value_col, trial_col=trial_col,
                                 save_dir=save_dir, use_alpha_pos=False, textsize=textsize, dotalpha=0.15,
                                 flag=save_flag, nIter=nIter, parallel=parallel, ymin=ymin, ymax=ymax)

        #plot_fits_summary(group, trial_col=trial_col, dat_col=value_col, save_dir=save_dir, use_alpha_pos=False, time_col='session', flag=save_flag)

def plot_nonlinear_regression_comparison(df3, shuff, stat_col, subject_cols, group_cols, trial_col, value_col, save_dir=None, flag=None, nIter=100, ymin=None, ymax=None, textsize=20):
    if type(subject_cols) is list:
        groups = subject_cols + group_cols
    elif type(subject_cols) is str:
        groups = [subject_cols] + group_cols
    avg_shuff = shuff.groupby(group_cols + ['iternum']).mean().reset_index()
    avg_df3 = df3.groupby(groups).mean().reset_index() #trial average df3
    # plot the r2 values for each session with the null distribution
    if flag is not None:
        save_flag = trial_col + '_' + value_col + '_' + flag
    else:
        save_flag = trial_col + '_' + value_col
    plot_r2_pval_diffs_summary(avg_shuff, avg_df3, stat_col, save_flag=save_flag, save_dir=save_dir, textsize=textsize, nIter=nIter, ymin=ymin, ymax=ymax)#re-run and replot with nIter=nIter


def plotting_pipeline(df3, shuff, trial_col, value_col, group_cols, subject_cols, ymin=None, ymax=None, nIter=10000, save_dir=None, flag=None):
    if subject_cols is None:
        subject_cols = ['exp_name']
    if group_cols is None:
        group_cols=['exp_group', 'session', 'taste']

    args = {'subject_cols': subject_cols, 'group_cols': group_cols, 'trial_col': trial_col, 'value_col': value_col, 'nIter': nIter, 'save_dir':save_dir,
            'textsize':8, 'flag': flag, 'ymin':ymin, 'ymax':ymax}
    
    plot_nonlinear_line_graphs(df3, shuff, **args)
    #plot the stats quantificaiton of the r2 values
    plot_nonlinear_regression_stats(df3, shuff, **args)
    #plot the stats quantification of the r2 values with head to head of naive vs sucrose preexposed
    #plot_nonlinear_regression_comparison(df3, shuff, stat_col='r2', **args)
    #plot the sessionwise differences in the r2 values
    plot_session_differences(df3, shuff, **args)
    plt.close('all')
    r2_pred_change, shuff_r2_pred_change = get_pred_change(df3, shuff, subject_cols=subject_cols, group_cols=group_cols, trial_col=trial_col)

    # plot the within session difference between naive and preexp for pred. change
    ymin = min(r2_pred_change['pred. change'].min(), shuff_r2_pred_change['pred. change'].min())
    ymax = max(r2_pred_change['pred. change'].max(), shuff_r2_pred_change['pred. change'].max())

    args2 = args
    #remove 'subject cols' from args2
    args2.pop('subject_cols')
    # plot the predicted change in value col over the course of the session, with stats
    plot_predicted_change(r2_pred_change, shuff_r2_pred_change, **args2)

    # plot the session differences in the predicted change of value col
    args3 = {'subject_cols':subject_cols, 'group_cols':group_cols, 'trial_col':trial_col, 'value_col':value_col,
            'nIter':nIter, 'save_dir':save_dir, 'textsize':8, 'flag':flag}
    plot_session_differences(r2_pred_change, shuff_r2_pred_change, stat_col='pred. change', **args3)
    plt.close('all')
