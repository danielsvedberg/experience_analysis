import blechpy
import numpy as np
import blechpy.dio.h5io as h5io
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score
import os
import matplotlib.gridspec as gridspec
import pingouin as pg
import matplotlib
import math
matplotlib.use('Agg')

#for all matplotlib plots, set the default font to Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.major.width'] = 0.72
plt.rcParams['ytick.major.width'] = 0.72
plt.rcParams['xtick.minor.width'] = 0.72
plt.rcParams['ytick.minor.width'] = 0.72
#set font size to 8
dfontsize = 8
plt.rcParams.update({'font.size': dfontsize})
default_margins = {'left': 0.1, 'right': 0.95, 'top': 0.9, 'bottom': 0.1, 'wspace': 0.05, 'hspace': 0.05}
summary_margins = {'left': 0.1, 'right': 0.95, 'top': 0.9, 'bottom': 0.3, 'wspace': 0.01, 'hspace': 0.01}
dPanW = 0.8
dPanH = 0.8
dPad = round(0.2/(8*(1/72)), 2)
dh_pad = round(0.09/(8*(1/72)), 2)

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

# get the size of a single panel in inches
def get_panel_frac(fig):
    # 4. Collect bounding boxes of all "real" subplots
    ax = fig.get_axes()[0]
    bb = ax.get_position()
    xW = bb.x1 - bb.x0
    yH = bb.y1 - bb.y0

    return xW, yH

def check_panel_size(fig, panel_width, panel_height):
    init_width, init_height = fig.get_size_inches()
    xW, yH = get_panel_frac(fig)
    subplot_width = xW * init_width
    subplot_height = yH * init_height
    panel_width_error = abs(panel_width - subplot_width)
    panel_height_error = abs(panel_height - subplot_height)

    if panel_width_error > 0.01 or panel_height_error > 0.01:
        print('Panel width error: ' + str(panel_width_error))
        print('Panel height error: ' + str(panel_height_error))
        return False
    else:
        return True

def round_up_1d(num):
    return math.ceil(num * 10) / 10

def adjust_figure_for_panel_size_auto(fig, panel_width=dPanW, panel_height=dPanH, do_second_tight=True):
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
    # 1. round up panel width and height to 1 decimal place
    panel_width = round_up_1d(panel_width)
    panel_height = round_up_1d(panel_height)


    # 2. Detect the grid shape from the Axes
    nrows, ncols = detect_subplot_grid_shape(fig)
    if nrows == 0 or ncols == 0:
        # Could not detect a valid grid - return unchanged
        return fig

    #desired margins in inches:
    left = 0.6
    right = 0.1
    bottom = 0.3
    top = 0.2
    wspace = 0.1
    hspace = 0.1

    h_padding = top + bottom + (hspace * (nrows - 1))
    v_padding = left + right + (wspace * (ncols - 1))

    target_fig_size = (ncols * panel_width + v_padding, nrows * panel_height + h_padding)

    #now get the fraction of the figure that will be each padding
    left_frac = left / target_fig_size[0]
    right_frac = right / target_fig_size[0]
    bottom_frac = bottom / target_fig_size[1]
    top_frac = top / target_fig_size[1]
    wspace_frac = wspace/panel_width
    hspace_frac = hspace/panel_height

    #now, set the figure size to the target size
    fig.set_size_inches(target_fig_size)

    #and adjust the margins to the desired values
    fig.subplots_adjust(left=left_frac, right=1-right_frac, top=1-top_frac, bottom=bottom_frac, wspace=wspace_frac, hspace=hspace_frac)

    check = check_panel_size(fig, panel_width, panel_height)
    if check:
        return fig
    else:
        raise ValueError('Panel size not correct after adjustment')


def average_difference(u, v):
    return np.mean(np.abs(u - v))

def get_trial_info(dat):
    dintrials = dat.dig_in_trials
    dintrials['taste_trial'] = 1
    # groupby name and cumsum taste trial
    dintrials['taste_trial'] = dintrials.groupby('name')['taste_trial'].cumsum()
    # rename column trial_num to 'session_trial'
    dintrials = dintrials.rename(columns={'trial_num': 'session_trial', 'name': 'taste'})
    # select just the columns 'taste_trial', 'taste', 'session_trial', 'channel', and 'on_time'
    dintrials = dintrials[['taste_trial', 'taste', 'session_trial', 'channel', 'on_time']]
    return dintrials

def get_avg_intertrial_distances(rec_dir, bsln_sub=True, metric = 'euclidean'):
    df_list = []
    dat = blechpy.load_dataset(rec_dir)
    dintrials = get_trial_info(dat)
    #time_array, rate_array = h5io.get_rate_data(rec_dir)
    time_array, rate_array = h5io.get_psths(rec_dir)
    ts = 100
    te = 3000
    sidx = np.where(time_array >= ts)
    eidx = np.where(time_array <= te)
    idx = np.intersect1d(sidx, eidx)
    bslnidx = np.where(time_array < 0)[0]
    for din, rate in rate_array.items(): #loop through dins
        baseline = rate[:, :, bslnidx].mean(axis=2)
        rate = rate[:, :, idx]

        euc_dist_mat = np.zeros((rate.shape[1], rate.shape[2]))  # Trials x Bins

        for j in range(rate.shape[2]):  # Loop over bins
            if bsln_sub:
                binrate = rate[:, :, j] - baseline
            else:
                binrate = rate[:, :, j]
            euc_dist = squareform(pdist(binrate.T, metric=metric))
            it_euc_dist = np.mean(euc_dist, axis=1)
            euc_dist_mat[:, j] = it_euc_dist

        avg_euc_dist = np.mean(euc_dist_mat, axis=1)

        df = pd.DataFrame({
            'euclidean_distance': avg_euc_dist,
            'rec_dir': rec_dir,
            'channel': int(din[-1]),  # get the din number from string din
            'taste_trial': np.arange(rate.shape[1])
        })
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    # remove all rows where taste == 'Spont'
    df = df.loc[df['taste'] != 'Spont']
    df['euclidean_distance'] = df['euclidean_distance'].transform(lambda x: (x - x.mean()) / x.std())
    # add index info to df from dintrials using merge on taste_trial and channel
    df = pd.merge(df, dintrials, on=['taste_trial', 'channel'])

    # subtract the min of 'session_trial' from 'session_trial' to get the session_trial relative to the start of the recording
    df['session_trial'] = df['session_trial'] - df['session_trial'].min()
    return df

#TODO 090524: get this into something coherent I can run for the graph
def get_neuron_intertrial_correlations(rec_dir, bsln_sub=False):
    df_list = []
    dat = blechpy.load_dataset(rec_dir)
    dintrials = get_trial_info(dat)
    #time_array, rate_array = h5io.get_rate_data(rec_dir)
    time_array, rate_array = h5io.get_psths(rec_dir)
    ts = 100
    te = 3000
    sidx = np.where(time_array >= ts)
    eidx = np.where(time_array <= te)
    idx = np.intersect1d(sidx, eidx)
    bslnidx = np.where(time_array < 0)[0]


    for din, rate in rate_array.items(): #loop through dins
        baseline = rate[:, :, bslnidx].mean(axis=2)
        rate = rate[:, :, idx]
        if bsln_sub:
            rate = rate - baseline[:, :, np.newaxis]

        n_trials = rate.shape[1]
        neuron_idx = []
        i_indices = []
        j_indices = []
        corrs = []
        pvals = []

        for n in range(rate.shape[0]):  # Loop over neurons
            for i in range(1, n_trials):
                for j in range(i):
                    i_indices.append(i)
                    j_indices.append(j)
                    rate_i = rate[n, i, :]
                    rate_j = rate[n, j, :]
                    corr, pval = stats.pearsonr(rate_i, rate_j)
                    corrs.append(corr)
                    pvals.append(pval)
                    neuron_idx.append(n)


        df = pd.DataFrame({
            'correlation': corrs,
            'pval': pvals,
            'taste_trial': i_indices,
            'comp_trial' : j_indices,
            'i_trial': i_indices,
            'j_trial': j_indices,
            'neuron': neuron_idx})
        df['rec_dir'] = rec_dir
        df['channel'] = int(din[-1])  # get the din number from string din
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    # remove all rows where taste == 'Spont'
    df = df.loc[df['channel'] != 4]
    # add index info to df from dintrials using merge on taste_trial and channel
    df = pd.merge(df, dintrials, on=['taste_trial', 'channel'])
    # subtract the min of 'session_trial' from 'session_trial' to get the session_trial relative to the start of the recording
    df['session_trial'] = df['session_trial'] - df['session_trial'].min()
    return df

def get_neuron_intertrialgroup_correlations(rec_dir, bsln_sub=False, chunk_size = 10):
    df_list = []
    dat = blechpy.load_dataset(rec_dir)
    dintrials = get_trial_info(dat)
    #time_array, rate_array = h5io.get_rate_data(rec_dir)
    time_array, rate_array = h5io.get_psths(rec_dir)
    ts = 100
    te = 3000
    sidx = np.where(time_array >= ts)
    eidx = np.where(time_array <= te)
    idx = np.intersect1d(sidx, eidx)
    bslnidx = np.where(time_array < 0)[0]


    for din, rate in rate_array.items(): #loop through dins
        baseline = rate[:, :, bslnidx].mean(axis=2)
        rate = rate[:, :, idx]
        if bsln_sub:
            rate = rate - baseline[:, :, np.newaxis]

        n_trials = rate.shape[1]
        # Split the second dimension into chunks, even if it isn't perfectly divisible
        split_rate = np.array_split(rate, range(chunk_size, n_trials, chunk_size), axis=1)

        # Take the mean of each chunk along the second dimension (axis=1)
        rate = np.stack([np.mean(chunk, axis=1) for chunk in split_rate], axis=1)
        n_chunks = rate.shape[1]

        # create a list of strings called 'chunk ranges' i.e. '1-10' for the first chunk, '11-20' for the second, '21-30, etc
        chunk_ranges = [str((i+1)*chunk_size - chunk_size+1) + '-' + str((i+1)*chunk_size) for i in range(n_chunks)]
        #for n in range(rate.shape[0]):
        neuron_idx = []
        i_indices = []
        j_indices = []
        corrs = []
        pvals = []
        trial_groups = []
        comp_trial_groups = []

        for n in range(rate.shape[0]):  # Loop over neurons
            for i in range(1, n_chunks):
                for j in range(i):
                    i_indices.append(i)
                    j_indices.append(j)
                    rate_i = rate[n, i, :]
                    rate_j = rate[n, j, :]
                    trial_group = chunk_ranges[j]
                    comp_trial_group = chunk_ranges[i]

                    corr, pval = stats.pearsonr(rate_i, rate_j)
                    corrs.append(corr)
                    pvals.append(pval)
                    neuron_idx.append(n)
                    trial_groups.append(trial_group)
                    comp_trial_groups.append(comp_trial_group)


        df = pd.DataFrame({
            'correlation': corrs,
            'pval': pvals,
            'trial_group' : trial_groups,
            'comp_trial_group' : comp_trial_groups,
            'i_index': i_indices,
            'j_index': j_indices,
            'neuron': neuron_idx})
        df['rec_dir'] = rec_dir
        df['channel'] = int(din[-1])  # get the din number from string din
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    # remove all rows where taste == 'Spont'
    df = df.loc[df['channel'] != 4]
    # add index info to df from dintrials using merge on taste_trial and channel
    dintrials = dintrials[['channel', 'taste']]
    #remove duplicates from dintrials
    dintrials = dintrials.drop_duplicates()
    df = pd.merge(df, dintrials, on=['channel'])
    return df


def make_consensus_matrix2(rec_info, shuffle=False):
    #bins = np.arange(210, 500)
    tmin = 100
    tmax = 3000
    df_list = []
    matrices = []
    names = []
    df = None
    for name, group in rec_info.groupby(['exp_group', 'rec_num']):
        group = group.reset_index(drop=True)
        top_branch_dist = []
        dinnums = []
        cluster_arrays = []
        rec_dirs = []
        trial_counts = []
        names.append(name)
        consensus_matrix = np.empty((30, 30, len(group), 4))
        consensus_matrix[:] = np.nan

        for exp_num, row in group.iterrows():
            rec_dir = row['rec_dir']
            #time_array, rate_array = h5io.get_rate_data(rec_dir)
            time_array, rate_array = h5io.get_psths(rec_dir)
            #get indices of time array where time is between 100 and 3000
            sidx = np.where(time_array >= tmin)
            eidx = np.where(time_array <= tmax)
            idx = np.intersect1d(sidx, eidx)
            time_array = time_array[idx]
            for din, rate in rate_array.items():
                dinnum = int(din[-1])
                if din != 'dig_in_4':
                    rate = rate[:, :, idx]
                    dinnums.append(dinnum)
                    # downsample rate from 7000 bins to 700 by averaging every 10 bins
                    #rate = rate.reshape(rate.shape[0], rate.shape[1], -1, 10).mean(axis=3)
                    n_trials = rate.shape[1]
                    trial_counts.append(n_trials)

                    if shuffle:
                        # create a trial index that is resampled with replacement
                        trial_index = np.arange(n_trials)
                        resampled_index = np.random.choice(trial_index, size=n_trials, replace=True)
                        # recreate rate with resampled index
                        rate = rate[:, resampled_index, :]
                    # # zscore the rate
                    # rate = (rate - np.nanmean(rate)) / np.nanstd(rate)
                    # rate = rate[:, :, idx]

                    #create empty distance matrix
                    dms = np.zeros((n_trials, n_trials, len(time_array)))
                    for b, bn in enumerate(time_array):
                        X = rate[:, :, b].T
                        dms[:,:,b] = squareform(pdist(X, metric=average_difference))

                    dm = np.mean(dms, axis=2)
                    #fold the distance matrix to make it symmetrical
                    dm = (dm + dm.T) / 2
                    linkages = linkage(squareform(dm), method='ward')
                    distances = linkages[:, 2]
                    top_branch_dist.append(distances[-2])

                    min_t = distances.min()
                    max_t = distances.max()

                    best_score = -1
                    cluster_labels = None


                    for t in np.linspace(min_t, max_t, 1000):
                        clabs = fcluster(linkages, t=t, criterion='distance')
                        if (len(np.unique(clabs)) > 1) and (len(np.unique(clabs)) < len(rate)):
                            score = silhouette_score(dm, clabs)
                            if score > best_score:
                                best_score = score
                                cluster_labels = clabs

                    cluster_arrays.append(cluster_labels)
                    rec_dirs.append(rec_dir)
                    for i in range(n_trials):
                        for j in range(n_trials):
                            if j > i:
                                if cluster_labels[i] != cluster_labels[j]:
                                    consensus_matrix[i, j, exp_num, dinnum] = 1
                                else:
                                    consensus_matrix[i, j, exp_num, dinnum] = 0
                            elif j == i:
                                consensus_matrix[i, j, exp_num, dinnum] = 0

                    #check whether the diagonal is 0
                    dia = np.diag(consensus_matrix[:, :, exp_num, dinnum])
                    checksum = np.nansum(dia)
                    if checksum != 0:
                        raise ValueError('Diagonal is not 0 for exp_num: ', exp_num, 'dinnum: ', dinnum)



                    #make a df with the cluster labels and top branch distance and dinnum
        cluster_dict = {'channel': dinnums, 'top_branch_dist': top_branch_dist}
        cluster_df = pd.DataFrame(cluster_dict)
        cluster_df['cluster_labels'] = cluster_arrays

        cluster_df['exp_group'] = name[0]
        cluster_df['session'] = name[1]
        cluster_df['rec_dir'] = rec_dirs
        cluster_df['n_trials'] = trial_counts
        df_list.append(cluster_df)
        df = pd.concat(df_list, ignore_index=True)
        consensus_matrix = np.nanmean(consensus_matrix, axis=(2, 3))
        # replace all nan with 0
        consensus_matrix = np.nan_to_num(consensus_matrix)
        # fold the consensus matrix to make it symmetrical
        consensus_matrix = (consensus_matrix + consensus_matrix.T) / 2

        matrices.append(consensus_matrix)

    return matrices, names, df

def make_consensus_matrix_shuffle(rec_info, n_iter=10, parallel=True, overwrite=False, save_dir=None):
    def shuff_consensus(rec_info, iternum):
        matrices, names, df = make_consensus_matrix2(rec_info, shuffle=True)
        df['iter'] = iternum
        return matrices, names, df, iternum

    if overwrite:
        if save_dir is None:
            raise ValueError('save_dir must be specified if overwrite is True')
        else:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
    else:
        if save_dir:
            print('loading dfs from pickle file...')
            #load dfs from pickle
            dfs = pd.read_pickle(save_dir + '/shuffled_cluster_dfs.pkl')
            #load matrices from npy
            matrices = np.load(save_dir + '/shuffled_matrices.npy')
            #load names_df from csv
            names_df = pd.read_csv(save_dir + '/shuffled_names.csv')
            return matrices, names_df, dfs
        else:
            print('no save_dir specified, will not save dfs to pickle')

    if parallel:
        results = Parallel(n_jobs=-1)(delayed(shuff_consensus)(rec_info, i) for i in range(n_iter))
    else:
        results = [shuff_consensus(rec_info, i) for i in range(n_iter)]

    matrices, names, dfs, iters = zip(*results)
    dfs = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
    matrices = np.array(matrices)

    names_dfs = []
    for i, name in enumerate(names):
        exp_group, session = zip(*name)
        names_dfs.append(pd.DataFrame({'exp_group': exp_group, 'session': session, 'iter': iters[i]}))
    names_df = pd.concat(names_dfs, ignore_index=True).reset_index(drop=True)

    if overwrite and save_dir is not None:
        #save dfs to pickle
        dfs.to_pickle(save_dir + '/shuffled_cluster_dfs.pkl')
        #save matrices to npy
        np.save(save_dir + '/shuffled_matrices.npy', matrices)
        #save names_df to csv
        names_df.to_csv(save_dir + '/shuffled_names.csv')

    return matrices, names_df, dfs

def get_consensus_thresholds(matrices, names):
    dfs = []
    linkages = []
    for i, mat in enumerate(matrices):
        distvec = squareform(mat)
        linkage_mat = linkage(distvec, method='ward')
        linkages.append(linkage_mat)
        distances = linkage_mat[:, 2]
        t_min, t_max = distances.min(), distances.max()
        tvals = np.linspace(t_min, t_max, 1000)

        scores = []
        best_t = []
        cluster_labels = []
        for t in tvals:
            clabs = fcluster(linkage_mat, t=t, criterion='distance')
            if len(np.unique(clabs)) == 1 or len(np.unique(clabs)) == len(mat):
                print("no score")
            else:
                score = silhouette_score(mat, clabs)
                print('t:', t, 'score:', score)
                best_t.append(t)
                scores.append(score)
                cluster_labels.append(clabs)

        #       make a dataframe with best_t and scores and get the row with the best t
        score_df = pd.DataFrame({'t': best_t, 'score': scores, 'cluster_labels': cluster_labels})
        best_t = score_df.loc[score_df['score'].idxmax()]
        best_t['exp_group'] = names[i][0]
        best_t['session'] = names[i][1]
        dfs.append(best_t)

    best_t_df = pd.concat(dfs, axis=1).T
    thresholds = list(best_t_df['t'])
    return thresholds

def get_AB_clustering(df):
    #takes the df that is the result of make_consensus_matrix2
    # for each row in df, get the largest set of cluster labels in the array stored in each row of cluster labels
    # then make a new column called 'cluster_A' and store the length of the largest set of cluster labels in each index
    # then make a new column called 'cluster_B' and store the length of the second largest set of cluster labels in each index
    # then make a new column called 'cluster_A_avg_trial' and store the average index of the largest set of cluster labels in each index
    # then make a new column called 'cluster_B_avg_trial' and store the average index of the second largest set of cluster labels in each index
    clust_A_idxs = []
    clust_B_idxs = []
    clust_A_size = []
    clust_B_size = []
    clust_A_avg_trial = []
    clust_B_avg_trial = []
    for i, row in df.iterrows():
        cluster_labels = row['cluster_labels']
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)

        # Get the indices of the two largest clusters
        indices_largest_clusters = np.argsort(-counts)[:2]

        # Extract labels of the two largest clusters
        largest_clusters_labels = unique_labels[indices_largest_clusters]

        average_indices = []
        for label in largest_clusters_labels:
            # Find indices (positions) of data points belonging to the current cluster
            indices = np.where(cluster_labels == label)[0]

            # Calculate the mean index for the current cluster
            mean_index = np.mean(indices)

            # Append the mean index to the list
            average_indices.append(mean_index)
        # get the index of avg_indices that is the smallest
        smallest_index = np.argmin(average_indices)
        largest_index = np.argmax(average_indices)

        # in clust A idxs, store the indices of cluster_labels that are equal to the label of the smallest index
        clust_A_idxs.append(np.where(cluster_labels == largest_clusters_labels[smallest_index])[0])
        clust_B_idxs.append(np.where(cluster_labels == largest_clusters_labels[largest_index])[0])
        clust_A_size.append(counts[indices_largest_clusters[smallest_index]])
        clust_B_size.append(counts[indices_largest_clusters[largest_index]])
        clust_A_avg_trial.append(average_indices[smallest_index])
        clust_B_avg_trial.append(average_indices[largest_index])

    df['clust_A_idxs'] = clust_A_idxs
    df['clust_B_idxs'] = clust_B_idxs
    df['clust_A_size'] = clust_A_size
    df['clust_B_size'] = clust_B_size
    df['clust_A_avg_trial'] = clust_A_avg_trial
    df['clust_B_avg_trial'] = clust_B_avg_trial
    return(df)

def longform_AB_clustering(df, shuffle=False):
    # longform df by melting clust_A_size and clust_B_size as well as clust_A_avg_trial and clust_B_avg_trial
    if shuffle:
        target_cols = ['exp_group', 'session', 'channel', 'exp_name', 'iter']
    else:
        target_cols = ['exp_group', 'session', 'channel', 'exp_name']

    df_clust_size = pd.melt(df, id_vars=target_cols,
                            value_vars=['clust_A_size', 'clust_B_size'], var_name='cluster', value_name='size')
    # refactor values in cluster column so instead of 'clust_A_size' and 'clust_B_size' it is 'A' and 'B'
    df_clust_size['cluster'] = df_clust_size['cluster'].str.replace('_size', '')

    df_clust_trial = pd.melt(df, id_vars=target_cols,
                             value_vars=['clust_A_avg_trial', 'clust_B_avg_trial'], var_name='cluster',
                             value_name='avg_trial')
    # refactor values in cluster column so instead of 'clust_A_avg_trial' and 'clust_B_avg_trial' it is 'A' and 'B'
    df_clust_trial['cluster'] = df_clust_trial['cluster'].str.replace('_avg_trial', '')
    # merge
    target_cols = target_cols + ['cluster']
    df_clust = pd.merge(df_clust_size, df_clust_trial, on=target_cols)
    # replace 'clust_A' and 'clust_B' with 'A' and 'B'
    df_clust['cluster'] = df_clust['cluster'].str.replace('clust_', '')
    # relabel A in to 'early' and B into 'late'
    df_clust['cluster'] = df_clust['cluster'].str.replace('A', 'early')
    df_clust['cluster'] = df_clust['cluster'].str.replace('B', 'late')

    return df_clust


def get_AB_cluster_labels(df, shuffle=False):
    df_list = []
    if shuffle:
        target_cols = ['exp_group', 'session', 'channel', 'exp_name', 'iter']
    else:
        target_cols = ['exp_group', 'session', 'channel', 'exp_name']

    for i, row in df.iterrows():
        clust_A_idxs = row['clust_A_idxs']
        clust_B_idxs = row['clust_B_idxs']
        clust_idxs = np.concatenate([clust_A_idxs, clust_B_idxs])
        clust_A_labels = np.repeat('early', len(clust_A_idxs))
        clust_B_labels = np.repeat('late', len(clust_B_idxs))
        clust_labels = np.concatenate([clust_A_labels, clust_B_labels])
        newdf = pd.DataFrame({'clust_idx': clust_idxs, 'cluster': clust_labels})
        newdf[target_cols] = row[target_cols]
        df_list.append(newdf)
    newdf = pd.concat(df_list, ignore_index=True)

    if shuffle:
        #refactor cluster column so early becomes 'early shuffle' and late becomes 'late shuffle'
        newdf['cluster'] = newdf['cluster'] + ' shuffle'

    return newdf


def get_intra_inter_distances(df):
    dflist = []
    # loop through every row in df
    for nm, group in df.groupby(['exp_name']):

        inter_distances = []
        intra_A_distances = []
        intra_B_distances = []
        intra_distances = []
        all_trials_distances = []

        bins = np.arange(210, 500)
        nbins = len(bins)
        an_dist_mats = np.empty((len(group), nbins, 30, 30))
        an_dist_mats[:] = np.nan
        n_trials_list = []
        group = group.reset_index(drop=True)
        for i, row in group.iterrows():
            # load the rec_dir
            rec_dir = row['rec_dir']
            # create a string for the dig in from the channel
            din = 'dig_in_' + str(row['channel'])
            # get the rate arrays
            time_array, rate_array = h5io.get_rate_data(rec_dir, din=row['channel'])
            # get the number of trials
            n_trials = rate_array.shape[1]
            n_trials_list.append(n_trials)
            # downsample rate from 7000 bins to 700 by averaging every 10 bins
            rate = rate_array.reshape(rate_array.shape[0], rate_array.shape[1], -1, 10).mean(axis=3)
            # zscore the rate
            rate = (rate - np.mean(rate)) / np.std(rate)
            # iterate through each bin in bins with enumerate and get the average distance matrix across the bins
            for bdx, b in enumerate(bins):
                # get the rate for the current bin
                X = rate[:, :, b].T
                # calculate the pairwise distance matrix for the rate
                dm = squareform(pdist(X, metric=average_difference))

                an_dist_mats[i, bdx, 0:dm.shape[0], 0:dm.shape[0]] = dm

        # average an_dist_mats across the bins
        an_dist_mats = np.nanmean(an_dist_mats, axis=1)

        for i, row in group.iterrows():
            n_trials = n_trials_list[i]
            avg_dm = an_dist_mats[i, 0:n_trials, 0:n_trials]
            intra_A_dm = avg_dm[np.ix_(row['clust_A_idxs'], row['clust_A_idxs'])]
            intra_B_dm = avg_dm[np.ix_(row['clust_B_idxs'], row['clust_B_idxs'])]
            # get the upper triangle of the intra_A_dm and intra_B_dm and linearize
            intra_A_dm = intra_A_dm[np.triu_indices(intra_A_dm.shape[0], k=1)]
            intra_B_dm = intra_B_dm[np.triu_indices(intra_B_dm.shape[0], k=1)]
            AB_distances = avg_dm[np.ix_(row['clust_A_idxs'], row['clust_B_idxs'])]
            all_trial_dm = avg_dm[np.ix_(np.arange(n_trials), np.arange(n_trials))]
            all_trial_dm = all_trial_dm[np.triu_indices(all_trial_dm.shape[0], k=1)]
            # linearize AB_distances
            AB_distances = AB_distances.flatten()
            # append the linearized AB_distances to the list
            inter_distances.append(AB_distances)
            intra_A_distances.append(intra_A_dm)
            intra_B_distances.append(intra_B_dm)
            intra_distances.append(np.concatenate([intra_A_dm, intra_B_dm]))
            all_trials_distances.append(all_trial_dm)

        # make a dataframe with the inter_distances
        group['inter_distances'] = inter_distances
        group['intra_A_distances'] = intra_A_distances
        group['intra_B_distances'] = intra_B_distances
        group['all_trial_distances'] = all_trials_distances
        group['intra_distances'] = intra_distances
        dflist.append(group)
    # concatenate the list of dataframes into one dataframe
    df = pd.concat(dflist, ignore_index=True)

    # spread the distances into longform
    df_inter = df[['exp_group', 'session', 'channel', 'exp_name', 'inter_distances']].explode('inter_distances')
    # rename inter_distances to distances and make a new column called 'type' and store 'inter' in each index
    df_inter = df_inter.rename(columns={'inter_distances': 'distance'})
    df_inter['type'] = 'inter-cluster'
    df_intra = df[['exp_group', 'session', 'channel', 'exp_name', 'intra_distances']].explode('intra_distances')
    df_intra = df_intra.rename(columns={'intra_distances': 'distance'})
    df_intra['type'] = 'intra-cluster'
    df_all = df[['exp_group', 'session', 'channel', 'exp_name', 'all_trial_distances']].explode('all_trial_distances')
    df_all = df_all.rename(columns={'all_trial_distances': 'distance'})
    df_all['type'] = 'unclustered'
    df_long = pd.concat([df_inter, df_intra, df_all], ignore_index=True)

    # make df_long['session' and 'channel'] int
    df_long['session'] = df_long['session'].astype(int)
    df_long['channel'] = df_long['channel'].astype(int)
    # make AB_distances float
    df_long['distance'] = df_long['distance'].astype(float)
    return df_long

exp_group_order = {'naive': 0, 'suc_preexp': 1}
session_order = {1: 0, 2: 1, 3: 2}
exp_group_names = ['Naive', 'Suc. Pre-exposed']
exp_group_colors = ['Blues', 'Oranges']
exp_group_color_map = {'naive':'Blues', 'suc_preexp': 'Oranges'}

def plot_correlation_matrices(matrices, names, save=False, save_dir=None, flag=None):
    # get the maximum and minimum for every 3 entries of matrices
    max_vals = []
    min_vals = []

    #turn list of tuples "names" into separate lists for each entry in the tuple
    names_list = list(map(list, zip(*names)))
    exp_groups = set(names_list[0])
    sessions = set(names_list[1])
    n_exp_groups = len(exp_groups)
    n_sessions = len(sessions)

    for i in range(0, len(matrices), n_sessions):
        print(i)
        max_val = max([np.max(m) for m in matrices[i:i + 3]])
        min_val = min([np.min(m) for m in matrices[i:i + 3]])
        max_vals.append(max_val)
        min_vals.append(min_val)

    # Adjust the figure layout
    fig = plt.figure()#figsize=(10.5, 6))
    gs = gridspec.GridSpec(2, 4, width_ratios=[0.1, 1, 1, 1],
                           hspace=0.05)  # Extra column for the color bars on the left

    # Create axes for the plots, adjusting for the extra column for the color bars
    axs = [[fig.add_subplot(gs[i, j + 1]) for j in range(3)] for i in range(2)]

    # Create separate color bars for each row in the first column
    cbar_axes = [fig.add_subplot(gs[i, 0]) for i in range(2)]

    for i, mat in enumerate(matrices):
        name = names[i]
        exp_group = name[0]
        session = name[1]
        j = exp_group_order[exp_group]
        k = session_order[session]
        ax = axs[j][k]  # Adjust for the shifted indexing due to color bar column
        cmap = plt.get_cmap(exp_group_colors[j])
        cax = ax.matshow(mat, vmin=min_vals[j], vmax=max_vals[j], origin='lower', cmap=cmap)
        if k != 0:
            ax.set_yticks([])
        else:
            # ax.set_ylabel('Trial', fontsize=8)
            ax.set_yticks([0, 9, 19, 29])
            ax.set_yticklabels(ax.get_yticks(), fontsize=8)
        if j != 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Trial', fontsize=8)
            ax.set_xticks([0, 9, 19, 29])
            ax.set_xticklabels(ax.get_xticks(), fontsize=8)
            ax.xaxis.set_ticks_position('bottom')
        if j == 0:
            ax.set_title('Session ' + str(session), pad=-6, fontsize=8)
        # Set the ylabel on the right side of the last column of the data plots
        if k == 2:  # Corrected condition to match the last column of the data plots
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(exp_group_names[j], rotation=270, labelpad=20, fontsize=8)

        # Add one color bar per row in the first column
        if k == 1:  # This ensures color bars are added once per row
            cb = fig.colorbar(cax, cax=cbar_axes[j], orientation='vertical')
            cbar_axes[j].yaxis.set_ticks_position('left')
            cbar_axes[j].set_ylabel('distance index', fontsize=8, rotation=90, labelpad=-80)

    plt.tight_layout()
    plt.show()

    plot_name = save_dir + os.sep + '_consensus_matrix'
    if flag is not None:
        plot_name = plot_name + '_' + flag

    if save:
        fig.savefig(plot_name + '.png')
        fig.savefig(plot_name + '.svg')

def plot_correlation_matrices_single(matrices, names, save=False, save_dir=None, flag=None):
    # get the maximum and minimum for every 3 entries of matrices
    exp_group_name = names[0][0]
    max_vals = []
    min_vals = []

    for i in range(0, len(matrices)):
        print(i)
        max_val = max([np.max(m) for m in matrices[i:i + 3]])
        min_val = min([np.min(m) for m in matrices[i:i + 3]])
        max_vals.append(max_val)
        min_vals.append(min_val)

    max_val = max(max_vals)
    min_val = min(min_vals)

    # Adjust the figure layout
    fig = plt.figure(figsize=(10, 5.5))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 0.05],
                           hspace=0)  # Extra column for the color bars on the left

    # Create axes for the plots, adjusting for the extra column for the color bars
    axs = [fig.add_subplot(gs[0,j]) for j in range(3)]

    # Create separate color bars for each row in the first column
    cbar_axes = fig.add_subplot(gs[1,:])

    for i, mat in enumerate(matrices):
        name = names[i]
        exp_group = name[0]
        session = name[1]
        k = session_order[session]
        ax = axs[k]  # Adjust for the shifted indexing due to color bar column
        cmap = plt.get_cmap(exp_group_color_map[exp_group_name])
        cax = ax.matshow(mat, vmin=min_val, vmax=max_val, origin='lower', cmap=cmap)
        if k != 0:
            ax.set_yticks([])
        else:
            # ax.set_ylabel('Trial', fontsize=8)
            ax.set_yticks([0, 9, 19, 29])
            ax.set_yticklabels(ax.get_yticks(), fontsize=8)
            ax.set_ylabel('Trial', fontsize=8)


        ax.set_xlabel('Trial', fontsize=8)
        ax.set_xticks([0, 9, 19, 29])
        ax.set_xticklabels(ax.get_xticks(), fontsize=8)
        ax.xaxis.set_ticks_position('bottom')

        ax.set_title('Session ' + str(session), pad=-6, fontsize=8)
        # Set the ylabel on the right side of the last column of the data plots
        if k == 2:  # Corrected condition to match the last column of the data plots
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(exp_group_name, rotation=270, labelpad=20, fontsize=8)

        # Add one color bar per row in the first column
        if k == 1:  # This ensures color bars are added once per row
            cb = fig.colorbar(cax, cax=cbar_axes, orientation='horizontal')
            cbar_axes.yaxis.set_ticks_position('left')
            cbar_axes.set_xlabel('distance index', fontsize=8)#, labelpad=-80)

    plt.tight_layout()
    plt.show()
    plot_name = save_dir + os.sep + exp_group_name + '_consensus_matrix'
    if flag is not None:
        plot_name = plot_name + '_' + flag
    if save:
        fig.savefig(plot_name + '.png')
        fig.savefig(plot_name + '.svg')

def plot_heirarchical_clustering(matrices, names, threshold=None, save=False, save_dir=None, flag=None):
    linkages = []
    max_vals = []
    for i, m in enumerate(matrices):
        m = squareform(m)#(mat + mat.T) / 2)
        consensus_linkage_matrix = linkage(m, method='ward')
        linkages.append(consensus_linkage_matrix)
        # get the maximum value of the linkage matrix
        max_val = consensus_linkage_matrix[-1, 2]
        max_vals.append(max_val)
    # get the maximum of every 3 entries of max_vals
    max_vals = [max(max_vals[i:i + 3]) for i in range(0, len(max_vals), 3)]

    # now plot the dendograms =
    # make a 2x3 grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 6))
    # iterate over the matrices and names
    leaves = []
    for i, link in enumerate(linkages):
        name = names[i]
        exp_group = name[0]
        session = name[1]

        j = exp_group_order[exp_group]
        k = session_order[session]
        max_y = max_vals[j] * 1.1
        print(max_y)
        # fold mat to make it symmetrical

        ax = axs[j, k]
        if threshold is not None:
            leaf = dendrogram(link, ax=ax, leaf_rotation=90, leaf_font_size=8, get_leaves=True, color_threshold=threshold[i])
        else:
            leaf = dendrogram(link, ax=ax, leaf_rotation=90, leaf_font_size=8, get_leaves=True)
        leaves.append(leaf)
        # set max of y axis to max_y
        ax.set_ylim(0, max_y)

        # for the first row, set the title to the session number
        if j == 0:
            ax.set_title('Session ' + str(session))
        # for the last column, set the ylabel to the exp_group on the right axis
        if k == 2:
            ax.set_ylabel(exp_group_names[j], rotation=-90, labelpad=20, fontsize=8)
            ax.yaxis.set_label_position("right")

        # for the first column, set the ylabel to 'Cluster distan"
        if k == 0:
            ax.set_ylabel('Cluster distance', fontsize=8)
        else:
            ax.set_yticks([])

        # for the last row, set the xlabel to 'Trial'
        if j == 1:
            ax.set_xlabel('Trial', fontsize=8)
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    plt.show()
    # save the figure
    plot_name = save_dir + os.sep + '_dendograms'
    if flag is not None:
        plot_name = plot_name + '_' + flag
    if save:
        fig.savefig(plot_name + '.png')
        fig.savefig(plot_name + '.svg')

    return(fig, leaves)


def plot_heirarchical_clustering_single(matrices, names, threshold=None, save=False, save_dir=None, flag=None):
    linkages = []
    max_vals = []
    for i, m in enumerate(matrices):
        m = squareform(m)#(mat + mat.T) / 2)
        consensus_linkage_matrix = linkage(m, method='ward')
        linkages.append(consensus_linkage_matrix)
        # get the maximum value of the linkage matrix
        max_val = consensus_linkage_matrix[-1, 2]
        max_vals.append(max_val)
    # get the maximum of every 3 entries of max_vals
    #max_vals = [max(max_vals[i:i + 3]) for i in range(0, len(max_vals), 3)]
    max_vals = max(max_vals)
    # now plot the dendograms =
    # make a 2x3 grid of subplots
    fig, axs = plt.subplots(1, 3, figsize=(8, 4), sharey=False)
    # iterate over the matrices and names
    leaves = []
    for i, link in enumerate(linkages):
        name = names[i]
        exp_group = name[0]
        session = name[1]

        k = session_order[session]
        max_y = max_vals * 1.1
        print(max_y)
        # fold mat to make it symmetrical

        ax = axs[k]
        if threshold is not None:
            leaf = dendrogram(link, ax=ax, leaf_rotation=90, leaf_font_size=7, get_leaves=True, color_threshold=threshold[i])
        else:
            leaf = dendrogram(link, ax=ax, leaf_rotation=90, leaf_font_size=7, get_leaves=True)
        leaves.append(leaf)
        # set max of y-axis to max_y
        ax.set_ylim(0, max_y)

        ax.set_title('Session ' + str(session), fontsize=8)
        # for the last column, set the ylabel to the exp_group on the right axis
        #if k == 2:
        #ax.set_ylabel(exp_group, rotation=-90, labelpad=20, fontsize=8)
        #ax.yaxis.set_label_position("right")

        # for the first column, set the ylabel to 'Cluster distan"
        if k == 0:
            ax.set_ylabel('cluster\ndistance', fontsize=8)
            #set the y axis tick labels to a fontsize of 8
            ax.yaxis.set_tick_params(labelsize=8)
        else:
            ax.set_yticks([])

        # for the last row, set the xlabel to 'Trial'
        ax.set_xlabel('Trial', fontsize=8)
    #plt.tight_layout()
    fig = adjust_figure_for_panel_size_auto(fig, panel_width=2.4)
    #plt.subplots_adjust(wspace=0.05, hspace=0.2)
    #plt.show()
    # save the figure
    plot_name = save_dir + os.sep + 'naive' + '_dendograms'
    if flag is not None:
        plot_name = plot_name + '_' + flag
    if save:
        fig.savefig(plot_name + '.png')
        fig.savefig(plot_name + '.svg')

    return(fig, leaves)

def plot_cluster_sizes(df_AB_long, save_dir=None):
    # make a bar plot of the size of the two largest clusters for each channel
    g = sns.catplot(data=df_AB_long, kind='bar', x='cluster', y='size', row='exp_group', col='session',
                    margin_titles=True,
                    linewidth=2, edgecolor='black', facecolor=(0, 0, 0, 0))
    # map a catplot with stripplot to the same axes
    g.map_dataframe(sns.stripplot, x='cluster', y='size', dodge=True, color='black')
    # relabel the y axis to "cluster size (trials)"
    g.set_ylabels('cluster size (trials)')
    # remove 'exp group' from the row labels
    g.set_titles(row_template='{row_name}', col_template='{col_var} {col_name}')

    plt.show()
    # save the plot
    g.savefig(save_dir + '/cluster_size_barplot.png')
    g.savefig(save_dir + '/cluster_size_barplot.svg')

def plot_cluster_sizes_w_shuff(df_AB_long, shuff_AB_long, save_dir=None):
    # make a bar plot of the size of the two largest clusters for each channel
    g = sns.catplot(data=df_AB_long, kind='bar', x='cluster', y='size', row='exp_group', col='session',
                    margin_titles=True,
                    linewidth=2, edgecolor='black', facecolor=(0, 0, 0, 0))
    # map a catplot with stripplot to the same axes
    g.map_dataframe(sns.stripplot, x='cluster', y='size', dodge=True, color='black')
    # relabel the y axis to "cluster size (trials)"
    g.set_ylabels('cluster size (trials)')
    # remove 'exp group' from the row labels
    g.set_titles(row_template='{row_name}', col_template='{col_var} {col_name}')

    plt.show()
    # save the plot
    g.savefig(save_dir + '/cluster_size_barplot.png')
    g.savefig(save_dir + '/cluster_size_barplot.svg')

def plot_cluster_sizes_w_shuff_single(df_AB_long, shuff_AB_long, save_dir=None):
    # make a bar plot of the size of the two largest clusters for each channel
    cluster_x_map = {'early': 1, 'late': 2, 'total': 0, 'diff': 3}
    bonf_corr = 4 # 2 comparisons across 3 days 6 total comparisons
    conf = (0.05 / bonf_corr)/2
    ntiles = [conf, 1 - conf]

    #prepare to make a pvalue table
    vals = []
    pvals = []
    sessions = []
    clusters = []
    exp_groups = []

    fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    for session in [1,2,3]:
        ax = axs[session - 1]
        session_df = df_AB_long[df_AB_long['session'] == session]
        session_shuff = shuff_AB_long[shuff_AB_long['session'] == session]

        #session_df['size'] = session_df['size']/30 * 100
        #session_shuff['size'] = session_shuff['size']/30 * 100

        grp_cols = ['exp_group', 'session', 'channel', 'exp_name', 'n_trials']
        #make a 'total' cluster that is the sum of the early and late clusters
        total_df = session_df.groupby(grp_cols).sum().reset_index()
        total_df['cluster'] = 'total'

        #get the difference between the early and late clusters
        out = session_df.set_index(grp_cols+['cluster']).unstack('cluster').swaplevel(axis=1).sort_index(axis=1)
        diff_df = out['late'] - out['early']
        diff_df = diff_df.reset_index()
        diff_df['cluster'] = 'diff'

        session_df = pd.concat([session_df, total_df, diff_df], ignore_index=True)
        session_df['size'] = session_df['size']/session_df['n_trials']*100
        session_df = session_df.groupby(['exp_group', 'session', 'cluster'])['size'].mean().reset_index()

        #do the same for shuffles
        grp_cols = ['exp_group', 'session', 'channel', 'exp_name', 'n_trials', 'iter']
        total_shuff = session_shuff.groupby(grp_cols).mean().reset_index()
        total_shuff['cluster'] = 'total'

        #get the difference between the early and late clusters
        out = session_shuff.set_index(grp_cols+['cluster']).unstack('cluster').swaplevel(axis=1).sort_index(axis=1)
        diff_shuff = out['late'] - out['early']
        diff_shuff = diff_shuff.reset_index()
        diff_shuff['cluster'] = 'diff'

        session_shuff = pd.concat([session_shuff, total_shuff, diff_shuff], ignore_index=True)
        session_shuff['size'] = session_shuff['size']/session_shuff['n_trials']*100

        session_shuff = session_shuff.groupby(['exp_group', 'session', 'cluster', 'iter'])['size'].mean().reset_index()

        shuff_quantiles = session_shuff.groupby(['cluster'])['size'].quantile(ntiles).unstack().reset_index()
        # rename the columns of shuff_quantiles to 'lower' and 'upper'
        shuff_quantiles.columns = ['cluster', 'lower', 'upper']
        #if lower is negative, set it to 0, and set height to upper for that row
        #shuff_quantiles.loc[shuff_quantiles['lower'] < 0, 'lower'] = 0
        shuff_quantiles['height'] = shuff_quantiles['upper'] - shuff_quantiles['lower']
        print(shuff_quantiles)
        #reorder rows of shuff quantiles so that cluster order is 'early', 'late', 'total', 'diff'
        shuff_quantiles = shuff_quantiles.set_index('cluster').reindex(['total', 'early', 'late', 'diff']).reset_index()
        # set the height of the bars to be the difference between the upper and lower quantiles
        print(shuff_quantiles)

        # make a floating barplot with cluster on the x-axis, 0.025 the bottom of each bar, and 0.975 the top of each bar
        ax.bar(x=shuff_quantiles['cluster'], height=shuff_quantiles['height'], bottom=shuff_quantiles['lower'],
               color='gray', alpha=0.5)
        sns.barplot(data=session_df, x='cluster', y='size', ax=ax, capsize=0.1, errwidth=2, edgecolor='black',
                    order=['total','early', 'late', 'diff'], facecolor=(0, 0, 0, 0))

        #reorder session_df so that order of clusters is 'total', 'early', 'late', 'diff'

        for cluster in ['total', 'early', 'late', 'diff']:
            group = session_df[session_df['cluster'] == cluster]
            shuff_group = session_shuff[session_shuff['cluster'] == cluster]
            mean_shuff = shuff_group['size'].mean()
            mean = group['size'].mean()
            if mean > mean_shuff:
                pval = 1-(np.mean(mean > shuff_group['size']))
            else:
                pval = 1-(np.mean(mean < shuff_group['size']))
            pval = pval * bonf_corr
            stars = get_pval_stars(pval)

            if pval < 0.05:
                xpos = cluster_x_map[cluster]
                #plot a text from stars above the bar in bold
                ax.text(xpos, 75, stars, fontsize=8, ha='center', fontweight='bold')
            #make the pval table
            vals.append(mean)
            pvals.append(pval)
            sessions.append(session)
            clusters.append(cluster)
            exp_groups.append('naive')


        #set y lim
        ax.set_ylim(-20, 100)
        if session == 1:
            ax.set_ylabel('% of trials', fontsize=8)
            yticks = [-20, 0,20,40,60,80,100]
            ax.set_yticks(yticks)
            ax.set_yticklabels(ax.get_yticks(), fontsize=8)
        else:
            ax.set_ylabel('')
        ax.set_title('Session ' + str(session), fontsize=8)
        ax.set_xticklabels(['2\nlargest', 'early', 'late', 'late\n-early'], fontsize=8, rotation=80, ha='right', rotation_mode='anchor')
        ax.set_xlabel('Cluster', fontsize=8)

    fig = adjust_figure_for_panel_size_auto(fig)
    plt.show()
    # save the plot
    plt.savefig(save_dir + '/cluster_size_barplot_naive.png')
    plt.savefig(save_dir + '/cluster_size_barplot_naive.svg')

    #save the pval table as csv
    pval_df = pd.DataFrame({'exp_group': exp_groups, 'session': sessions, 'cluster': clusters, 'mean': vals, 'pval': pvals})
    savename = save_dir + '/cluster_size_barplot_naive_pval_table.csv'
    pval_df.to_csv(savename, index=False)



def get_pval_stars(pval, bonf_corr=None):
    if bonf_corr is not None:
        pval = pval * bonf_corr
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return None

def plot_cluster_avg_trial_naive(newdf, save_dir=None, flag=None):
    aovs = []
    for name, group in newdf.groupby(['session']):
        aov = pg.rm_anova(data=group, dv='clust_idx', within=['channel', 'cluster'], subject='exp_name')
        aov['session'] = name
        aovs.append(aov)
    aovs = pd.concat(aovs, ignore_index=True)
    aovs['p-GG-corr'] = aovs['p-GG-corr'] * 3 #bonfferoni correcting for 3 sessions
    aovs = aovs.loc[aovs['Source'] == 'cluster']

    fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
    for session in [1, 2, 3]:
        ax = axs[session - 1]
        session_df = newdf[newdf['session'] == session]
        sns.barplot(data=session_df, x='cluster', y='clust_idx', ax=ax, capsize=0.1, errwidth=2, edgecolor='black', facecolor=(0, 0, 0, 0))
        sns.swarmplot(data=session_df, x='cluster', y='clust_idx', ax=ax, color='black', alpha=0.5, size=3)

        sess_aov = aovs[aovs['session'] == session]
        pval = sess_aov['p-GG-corr'].values[0]
        if pval < 0.05:
            stars = get_pval_stars(pval)
            #put a horizontal line from 0 to 1 at 25
            ax.plot([0, 1], [25, 25], lw=2, c='black')
            ax.text(0.5, 26, stars, fontsize=8, ha='center')

        ax.set_title('Session ' + str(session))
        if session == 1:
            ax.set_ylabel('Trial')
        else:
            ax.set_ylabel('')
        ax.set_xlabel('Cluster')
        ax.set_ylim(-1, 30)
        ax.set_yticks([0, 10, 20, 30])
        ax.set_yticklabels(ax.get_yticks(), fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_xlabel('Cluster', fontsize=8)
    plt.tight_layout()
    plt.show()
    # save the plot
    savename = save_dir + '/cluster_avg_trial_naive'
    if flag is not None:
        savename = savename + '_' + flag
    plt.savefig(savename + '.png')
    plt.savefig(savename + '.svg')

def plot_cluster_avg_trial_naive_w_shuff(newdf, shuff_df, save_dir=None, flag=None):
    #relabel cluster column so 'early shuffle' is 'early\nshuffle' and 'late shuffle' is 'late\nshuffle'
    #make a new column called 'half which is the 'cluster' column with ' shuffle' removed
    newdf = newdf.copy()
    cluster_x_map = {'early': 0, 'late': 1, 'diff': 2}
    shuff_df['cluster'] = shuff_df['cluster'].str.replace('early shuffle', 'early')
    shuff_df['cluster'] = shuff_df['cluster'].str.replace('late shuffle', 'late')

    meandf = newdf.groupby(['session', 'channel', 'cluster', 'exp_name']).agg({'clust_idx': 'mean'}).reset_index()
    newdf = newdf.groupby(['session', 'channel', 'cluster', 'exp_name']).agg({'clust_idx': 'mean'}).reset_index()
    #group shuff_df by cluster, channel, exp_name, and iter, get the mean of clust_idx
    mean_shuff_df = shuff_df.groupby(['session','cluster', 'iter']).agg({'clust_idx': 'mean'}).reset_index()

    bonf_corr = 3 # 2 comparisons across 3 days 6 total comparisons
    conf = (0.05 / bonf_corr)/2 #two-sided comparison
    ntiles = [conf, 1 - conf]

    #set up lists to make a pvalue and grouping table
    vals = []
    pvals = []
    sessions = []
    clusters = []
    exp_groups = []

    fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    for session in [1, 2, 3]:
        ax = axs[session - 1]
        session_df = newdf[newdf['session'] == session]
        mean_session_df = meandf[meandf['session'] == session]
        session_shuff = mean_shuff_df[mean_shuff_df['session'] == session]
        #for each grouping of cluster in session_shuff, get the 97.5% and 2.5% quantiles of clust_idx

        grp_cols = ['session', 'cluster', 'channel', 'exp_name']
        out = mean_session_df.set_index(grp_cols).unstack('cluster').swaplevel(axis=1).sort_index(axis=1)
        diff_df = out['late'] - out['early']
        diff_df = diff_df.reset_index()
        diff_df['cluster'] = 'diff'
        session_df = pd.concat([session_df, diff_df], ignore_index=True)

        shuff_grp_cols = ['session', 'cluster', 'iter']
        out = session_shuff.set_index(shuff_grp_cols).unstack('cluster').swaplevel(axis=1).sort_index(axis=1)
        diff_shuff = out['late'] - out['early']
        diff_shuff = diff_shuff.reset_index()
        diff_shuff['cluster'] = 'diff'

        session_shuff = pd.concat([session_shuff, diff_shuff], ignore_index=True)

        shuff_quantiles = session_shuff.groupby('cluster')['clust_idx'].quantile(ntiles).unstack().reset_index()
        #rename the columns of shuff_quantiles to 'lower' and 'upper'
        shuff_quantiles.columns = ['cluster', 'lower', 'upper']
        shuff_quantiles['height'] = shuff_quantiles['upper'] - shuff_quantiles['lower']
        print(shuff_quantiles)

        #set the order of cluster to cluster_x_map
        shuff_quantiles = shuff_quantiles.set_index('cluster').reindex(['early', 'late', 'diff']).reset_index()

        #make a floating barplot with cluster on the x-axis, 0.025 the bottom of each bar, and 0.975 the top of each bar
        ax.bar(x=shuff_quantiles['cluster'], height=shuff_quantiles['height'], bottom=shuff_quantiles['lower'], color='gray', alpha=0.5)
        sns.barplot(data=session_df, x='cluster', y='clust_idx', ax=ax, capsize=0.1, errwidth=2, edgecolor='black',
                    order=['early','late', 'diff'], facecolor=(0, 0, 0, 0))
        sns.swarmplot(data=session_df, x='cluster', y='clust_idx', ax=ax, color='tab:blue', alpha=0.5, size=2,
                      order=['early', 'late', 'diff'])

        for cluster, group in session_df.groupby('cluster'):
            shuff_group = session_shuff[session_shuff['cluster'] == cluster]
            mean_shuff = shuff_group['clust_idx'].mean()
            mean = group['clust_idx'].mean()
            if mean > mean_shuff:
                pval = 1-(np.mean(mean > shuff_group['clust_idx']))
            else:
                pval = 1-(np.mean(mean < shuff_group['clust_idx']))
            pval = pval * bonf_corr
            stars = get_pval_stars(pval)

            if pval < 0.05:
                xpos = cluster_x_map[cluster]
                #plot a text from stars above the bar in bold
                ax.text(xpos, 20, stars, fontsize=8, ha='center', fontweight='bold')

            #record the pval, session, cluster, and exp_group
            vals.append(mean)
            pvals.append(pval)
            sessions.append(session)
            clusters.append(cluster)
            exp_groups.append('naive')

        ax.set_title('Session ' + str(session), fontsize=8)
        if session == 1:
            ax.set_ylabel('Trial', fontsize=8)
        else:
            ax.set_ylabel('')
        ax.set_xlabel('Cluster', fontsize=8)
        ax.set_ylim(-1, 30)
        ax.set_yticks([0, 10, 20, 30])
        ax.set_yticklabels(ax.get_yticks(), fontsize=8)
        ax.set_xticklabels(['early', 'late', 'late\n- early'], fontsize=8, rotation=45)
        ax.set_xlabel('Cluster', fontsize=8)
    fig = adjust_figure_for_panel_size_auto(fig)
    plt.show()

    # save the plot
    savename = save_dir + '/cluster_avg_trial_naive_w_shuff'
    if flag is not None:
        savename = savename + '_' + flag
    plt.savefig(savename + '.png')
    plt.savefig(savename + '.svg')

    #create and save the pval table
    pval_df = pd.DataFrame({'mean': vals, 'pval': pvals, 'session': sessions, 'cluster': clusters, 'exp_group': exp_groups})
    pval_df['pval'] = pval_df['pval'].astype(float)
    pval_df['session'] = pval_df['session'].astype(int)
    pval_df['cluster'] = pval_df['cluster'].astype(str)
    pval_df['exp_group'] = pval_df['exp_group'].astype(str)
    save_name = save_dir + '/cluster_avg_trial_naive_w_shuff_pvals'
    if flag is not None:
        save_name = save_name + '_' + flag
    pval_df.to_csv(save_name + '.csv', index=False)


def plot_cluster_distances(intra_inter_df, save_dir=None):
    # plot a bar plot of the distances
    colors = {'naive': 'blue', 'suc_preexp': 'orange'}
    g = sns.catplot(data=intra_inter_df, kind='bar', x='type', y='distance', row='exp_group', col='session',
                    margin_titles=True, linewidth=2, aspect=1, color='white', edgecolor='black')
    # g.map_dataframe(sns.stripplot, x='type', y='distance', dodge=True, alpha=1, jitter=0.4, palette='colorblind')
    g.set_titles(row_template='{row_name}')
    g.set_ylabels('distance index')
    plt.show()
    # save the plot
    g.savefig(save_dir + '/cluster_distance_bars.png')
    g.savefig(save_dir + '/cluster_distance_bars.svg')

def plot_cluster_distances_naive(intra_inter_df, save_dir=None):
    intra_inter_df = intra_inter_df[intra_inter_df['exp_group'] == 'naive']
    #refactor type column so 'inter-cluster' is 'inter\ncluster' and 'intra-cluster' is 'intra\cluster' and 'unclustered' is 'un-\nclustered'
    intra_inter_df['type'] = intra_inter_df['type'].str.replace('inter-cluster', 'inter-\ncluster')
    intra_inter_df['type'] = intra_inter_df['type'].str.replace('intra-cluster', 'intra-\ncluster')
    intra_inter_df['type'] = intra_inter_df['type'].str.replace('unclustered', 'un-\nclustered')
    # plot a bar plot of the distances
    colors = {'naive': 'blue', 'suc_preexp': 'orange'}
    g = sns.catplot(data=intra_inter_df, kind='bar', x='type', y='distance', row='exp_group', col='session',
                    margin_titles=True, linewidth=2, aspect=1, color='white', edgecolor='black')
    # g.map_dataframe(sns.stripplot, x='type', y='distance', dodge=True, alpha=1, jitter=0.4, palette='colorblind')
    g.set_titles(row_template='{row_name}')
    g.set_ylabels('distance index')
    g.set_xlabels('')
    plt.tight_layout()
    plt.show()
    # save the plot
    g.savefig(save_dir + '/cluster_distance_bars_naive.png')
    g.savefig(save_dir + '/cluster_distance_bars_naive.svg')
