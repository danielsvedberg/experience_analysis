import blechpy
import numpy as np
import blechpy.dio.h5io as h5io
import pandas as pd
from joblib import Parallel, delayed
import trialwise_analysis as ta
import analysis as ana
import matplotlib.pyplot as plt
import feather
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
rec_info = proj.rec_info.copy()  # get the rec_info table
rec_dirs = rec_info['rec_dir']

PA = ana.ProjectAnalysis(proj)


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


# %% calculate and plot euclidean and cosine distances for each taste trial
def process_rec_dir(rec_dir):
    df_list = []
    dat = blechpy.load_dataset(rec_dir)
    dintrials = get_trial_info(dat)
    time_array, rate_array = h5io.get_rate_data(rec_dir)
    for din, rate in rate_array.items():
        avg_firing_rate = np.mean(rate, axis=1)  # Neurons x Bins
        cos_sim_mat = np.zeros((rate.shape[1], rate.shape[2]))  # Trials x Bins
        euc_dist_mat = np.zeros((rate.shape[1], rate.shape[2]))  # Trials x Bins

        for i in range(rate.shape[1]):  # Loop over trials
            for j in range(rate.shape[2]):  # Loop over bins
                trial_rate_bin = rate[:, i, j]
                avg_firing_rate_bin = avg_firing_rate[:, j]

                # Cosine similarity
                cos_sim = np.dot(trial_rate_bin, avg_firing_rate_bin) / (
                        np.linalg.norm(trial_rate_bin) * np.linalg.norm(avg_firing_rate_bin))
                cos_sim_mat[i, j] = cos_sim

                # Euclidean distance
                euc_dist = np.linalg.norm(trial_rate_bin - avg_firing_rate_bin)
                euc_dist_mat[i, j] = euc_dist
        # zscore every entry of euc_dist_mat
        euc_dist_mat = (euc_dist_mat - np.mean(euc_dist_mat)) / np.std(euc_dist_mat)

        avg_cos_sim = np.mean(cos_sim_mat[:, 2000:5000], axis=1)
        avg_euc_dist = np.mean(euc_dist_mat[:, 2000:5000], axis=1)

        df = pd.DataFrame({
            'cosine_similarity': avg_cos_sim,
            'euclidean_distance': avg_euc_dist,
            'rec_dir': rec_dir,
            'channel': int(din[-1]),  # get the din number from string din
            'taste_trial': np.arange(rate.shape[1])
        })
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    # add index info to df from dintrials using merge on taste_trial and channel
    df = pd.merge(df, dintrials, on=['taste_trial', 'channel'])
    # remove all rows where taste == 'Spont'
    df = df.loc[df['taste'] != 'Spont']
    # subtract the min of 'session_trial' from 'session_trial' to get the session_trial relative to the start of the recording
    df['session_trial'] = df['session_trial'] - df['session_trial'].min()
    return df


# Parallelize processing of each rec_dir
num_cores = -1  # Use all available cores
final_dfs = Parallel(n_jobs=num_cores)(delayed(process_rec_dir)(rec_dir) for rec_dir in rec_dirs)

# Concatenate all resulting data frames into one
final_df = pd.concat(final_dfs, ignore_index=True)

# merge in rec_info into final_df
final_df = pd.merge(final_df, rec_info, on='rec_dir')
final_df['session'] = final_df['rec_num']

subject_col = 'exp_name'
group_cols = ['exp_group', 'session', 'taste']
trial_col = 'session_trial'
value_col = 'euclidean_distance'
preprodf, shuffle = ta.preprocess_nonlinear_regression(final_df, subject_col, group_cols, trial_col, value_col,
                                                       nIter=10000, save_dir=PA.save_dir, overwrite=False)

flag = 'test'
nIter = 10000
textsize = 20
parallel = True
yMin = preprodf[value_col].min()
yMax = preprodf[value_col].max()
ta.plot_fits_summary_avg(preprodf, shuff_df=shuffle, dat_col=value_col, trial_col=trial_col, save_dir=PA.save_dir,
                         use_alpha_pos=False, textsize=textsize, dotalpha=0.15, flag=flag, nIter=nIter,
                         parallel=parallel, yMin=yMin, yMax=yMax)
for exp_group, group in preprodf.groupby(['exp_group']):
    group_shuff = shuffle.groupby('exp_group').get_group(exp_group)
    if flag is not None:
        save_flag = exp_group + '_' + flag
    else:
        save_flag = exp_group
    ta.plot_fits_summary_avg(group, shuff_df=group_shuff, dat_col=value_col, trial_col=trial_col,
                             save_dir=PA.save_dir, use_alpha_pos=False, textsize=textsize, dotalpha=0.15,
                             flag=save_flag, nIter=nIter, parallel=parallel, yMin=yMin, yMax=yMax)

ta.plot_fits_summary(preprodf, dat_col=value_col, trial_col=trial_col, save_dir=PA.save_dir, time_col='session',
                     use_alpha_pos=False, dotalpha=0.15, flag=flag)

ta.plot_nonlinear_regression_stats(preprodf, shuffle, subject_col=subject_col, group_cols=group_cols,
                                   trial_col=trial_col, value_col=value_col, save_dir=PA.save_dir, flag=flag,
                                   textsize=textsize, nIter=nIter)

pred_change_df, pred_change_shuff = ta.get_pred_change(preprodf, shuffle, subject_col=subject_col,
                                                       group_cols=group_cols, trial_col=trial_col)
ta.plot_predicted_change(pred_change_df, pred_change_shuff, group_cols, value_col=value_col, trial_col=trial_col,
                         save_dir=PA.save_dir, flag=flag, textsize=textsize, nIter=nIter)

# %% plotting functions
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

exp_group_order = {'naive': 0, 'suc_preexp': 1}
session_order = {1: 0, 2: 1, 3: 2}
exp_group_names = ['Naive', 'Suc. Pre-exposed']
exp_group_colors = ['Blues', 'Oranges']

def plot_correlation_matrices(matrices, names, save=False):
    # get the maximum and minimum for every 3 entries of matrices
    max_vals = []
    min_vals = []
    for i in range(0, len(matrices), 3):
        max_val = max([np.max(mat) for mat in matrices[i:i + 3]])
        min_val = min([np.min(mat) for mat in matrices[i:i + 3]])
        max_vals.append(max_val)
        min_vals.append(min_val)

    # Adjust the figure layout
    fig = plt.figure(figsize=(10.5, 6))
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
            # ax.set_ylabel('Trial', fontsize=20)
            ax.set_yticks([0, 9, 19, 29])
            ax.set_yticklabels(ax.get_yticks(), fontsize=14)
        if j != 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Trial', fontsize=20)
            ax.set_xticks([0, 9, 19, 29])
            ax.set_xticklabels(ax.get_xticks(), fontsize=14)
            ax.xaxis.set_ticks_position('bottom')
        if j == 0:
            ax.set_title('Session ' + str(session), pad=-6, fontsize=20)
        # Set the ylabel on the right side of the last column of the data plots
        if k == 2:  # Corrected condition to match the last column of the data plots
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(exp_group_names[j], rotation=270, labelpad=20, fontsize=20)

        # Add one color bar per row in the first column
        if k == 1:  # This ensures color bars are added once per row
            cb = fig.colorbar(cax, cax=cbar_axes[j], orientation='vertical')
            cbar_axes[j].yaxis.set_ticks_position('left')
            cbar_axes[j].set_ylabel('distance index', fontsize=20, rotation=90, labelpad=-80)

    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig(PA.save_dir + '/consensus_matrix.png')
        fig.savefig(PA.save_dir + '/consensus_matrix.svg')


def plot_heirarchical_clustering(matrices, names, threshold=None, save=False):
    linkages = []
    max_vals = []
    for i, mat in enumerate(matrices):
        mat = squareform(mat)#(mat + mat.T) / 2)
        consensus_linkage_matrix = linkage(mat, method='ward')
        linkages.append(consensus_linkage_matrix)
        # get the maximum value of the linkage matrix
        max_val = consensus_linkage_matrix[-1, 2]
        max_vals.append(max_val)
    # get the maximum of every 3 entries of max_vals
    max_vals = [max(max_vals[i:i + 3]) for i in range(0, len(max_vals), 3)]

    # now plot the dendograms
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
            ax.set_ylabel(exp_group_names[j], rotation=-90, labelpad=20, fontsize=20)
            ax.yaxis.set_label_position("right")

        # for the first column, set the ylabel to 'Cluster distan"
        if k == 0:
            ax.set_ylabel('Cluster distance', fontsize=20)
        else:
            ax.set_yticks([])

        # for the last row, set the xlabel to 'Trial'
        if j == 1:
            ax.set_xlabel('Trial', fontsize=20)
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    plt.show()
    # save the figure
    if save:
        fig.savefig(PA.save_dir + '/dendograms.png')
        fig.savefig(PA.save_dir + '/dendograms.svg')
    return(fig, leaves)


# %% average euclidean distance matrix (oldest approach)
# make a matrix of the Euclidean distance for each taste trial for each taste and session, then take the average
def make_euc_dist_matrix(rec_dir):
    bins = np.arange(2000, 5000)
    df_list = []
    dat = blechpy.load_dataset(rec_dir)
    dintrials = get_trial_info(dat)
    dintrials['taste_trial'] = dintrials['taste_trial'] - 1
    time_array, rate_array = h5io.get_rate_data(rec_dir)
    # set up a dict to store a matrix for each key in rate_array
    euc_dist_mats = []
    dins = []
    rec_dirs = []
    for din, rate in rate_array.items():
        if din != 'dig_in_4':
            dinnum = int(din[-1])
            euc_distances = np.empty((30, 30, len(bins)))
            n_trials = rate.shape[1]
            # fill with nan
            euc_distances[:] = np.nan
            for bidx, b in enumerate(bins):
                X = rate[:, :, b].T
                euc_dist = pdist(X)
                euc_dist = squareform(euc_dist)
                euc_distances[0:n_trials, 0:n_trials, bidx] = euc_dist
            # average across bins
            euc_distances = np.nanmean(euc_distances, axis=2)
            # fold the distance matrix to make it symmetrical
            euc_distances = (euc_distances + euc_distances.T) / 2
            # z-score the euc_distances
            euc_distances = (euc_distances - np.mean(euc_distances)) / np.std(euc_distances)
            # add the negative of the lowest value to all entries of euc_distances
            # euc_distances = euc_distances - np.min(euc_distances)
            euc_dist_mats.append(euc_distances)
            dins.append(dinnum)
            rec_dirs.append(rec_dir)
    # make a dataframe with rec_dir and euc_distances and dins
    euc_distances = pd.DataFrame({'rec_dir': rec_dirs, 'euc_dist': euc_dist_mats, 'din': dins})

    return euc_distances

# Parallelize processing of each rec_dir
num_cores = -1  # Use all available cores
distances = Parallel(n_jobs=num_cores)(delayed(make_euc_dist_matrix)(rec_dir) for rec_dir in rec_dirs)

# Concatenate all resulting data frames into one
final_df = pd.concat(distances, ignore_index=True)

final_df = pd.merge(final_df, rec_info, on='rec_dir')
final_df['session'] = final_df['rec_num']

matrices = []
names = []
for name, group in final_df.groupby(['exp_group', 'session']):
    names.append(name)
    euc_dist_mats = np.empty((30, 30, len(group)))
    euc_dist_mats[:] = np.nan
    group = group.reset_index(drop=True)
    for i, row in group.iterrows():
        euc_dist_mats[:, :, i] = row['euc_dist']
    euc_dist_mats = np.nanmean(euc_dist_mats, axis=2)
    # replace all nan with 0
    matrices.append(euc_dist_mats)

plot_correlation_matrices(matrices, names)
plot_heirarchical_clustering(matrices, names)


# %% consensus clustering (first attempt)

def cluster_data(sample, t):
    distance_matrix = pdist(sample)
    linkage_matrix = linkage(distance_matrix, method='average')
    cluster_labels = fcluster(linkage_matrix, t=t, criterion='distance')
    return cluster_labels


def opti_t_bin(X):
    linkages = linkage(pdist(X), method='average')
    distances = linkages[:, 2]
    min_t = distances.min()
    max_t = distances.max()
    t_range = np.linspace(min_t, max_t, 100)
    best_score = -1
    best_t = None  # Initialize best_t to ensure it has a value even if all scores are below -1
    for t in t_range:
        cluster_labels = cluster_data(X, t)
        if len(np.unique(cluster_labels)) == 1 or len(np.unique(cluster_labels)) == len(X):
            continue  # Silhouette score is not meaningful for a single cluster
        score = silhouette_score(X, cluster_labels)
        if score > best_score:
            best_score = score
            best_t = t
            print('Best t:', best_t, 'Best score:', best_score)
    return best_t


def optimize_bin(rate, bin_value):
    X = rate[:, :, bin_value].T
    return opti_t_bin(X)


def optimize_t(rec_dir):
    bins = np.arange(200, 500)
    optimal_ts = []
    time_array, rate_array = h5io.get_rate_data(rec_dir)
    for din, rate in rate_array.items():
        if din is not 'dig_in_4':
            # downsample by averaging every 10 bins
            rate = rate.reshape(rate.shape[0], rate.shape[1], -1, 10).mean(axis=3)
            # z-score the rate
            # rate = (rate - rate.mean()) / rate.std()
            # average rate across bins
            X = rate[:, :, bins].mean(axis=2).T
            optimized = opti_t_bin(X)
            # optimized = Parallel(n_jobs=-1)(delayed(optimize_bin)(rate, b) for b in bins)
            # replace all entries in list optimal_ts that are None with Nan
            if optimized is None:
                optimized = np.nan
            # optimized = #[np.nan if x is None else x for x in optimized]
            optimal_ts.append(np.nanmean(optimized))
    optimal_ts = np.nanmean(optimal_ts)
    return optimal_ts


rec_dirs = rec_info['rec_dir']
ts = []
for rec_dir in rec_dirs:
    print(rec_dir)
    ts.append(optimize_t(rec_dir))

# make a dataframe with rec_dir and ts
ts_df = pd.DataFrame({'rec_dir': rec_dirs, 'best_t_val': ts})
rec_info = proj.rec_info.copy()
# merge ts_df with rec_info
rec_info = pd.merge(rec_info, ts_df, on='rec_dir')


def make_consensus_matrix(rec_info):
    bins = np.arange(210, 500)
    matrices = []
    names = []
    for name, group in rec_info.groupby(['exp_group', 'rec_num']):
        names.append(name)
        consensus_matrix = np.empty((30, 30, 4, len(bins)))
        consensus_matrix[:] = np.nan
        for _, row in group.iterrows():
            rec_dir = row['rec_dir']
            time_array, rate_array = h5io.get_rate_data(rec_dir)
            for din, rate in rate_array.items():
                dinnum = int(din[-1])
                # downsample rate from 7000 bins to 700 by averaging every 10 bins
                rate = rate.reshape(rate.shape[0], rate.shape[1], -1, 10).mean(axis=3)
                if din != 'dig_in_4':
                    n_trials = rate.shape[1]
                    for bidx, b in enumerate(bins):
                        X = rate[:, :, b].T
                        cluster_labels = cluster_data(X, t=row['best_t_val'])
                        for i in range(n_trials):
                            for j in range(n_trials):
                                if j > i:
                                    if cluster_labels[i] != cluster_labels[j]:
                                        consensus_matrix[i, j, dinnum, bidx] = 1
                                    else:
                                        consensus_matrix[i, j, dinnum, bidx] = 0
                                if j == i:
                                    consensus_matrix[i, j, dinnum, bidx] = 0

        consensus_matrix = np.nanmean(consensus_matrix, axis=3)
        consensus_matrix = np.nanmean(consensus_matrix, axis=2)
        # replace all nan with 0
        consensus_matrix = np.nan_to_num(consensus_matrix)
        #make the matrix symmetrical
        consensus_matrix = (consensus_matrix + consensus_matrix.T) / 2

        if np.isnan(consensus_matrix).all():
            raise ValueError('The entire consensus matrix is nan')
        matrices.append(consensus_matrix)
    return matrices, names

matrices, names = make_consensus_matrix(rec_info)

plot_correlation_matrices(matrices, names)
plot_heirarchical_clustering(matrices, names)

# sweep over different values of t to get the best silhouette score
mat = matrices[0]
mat = ((mat + mat.T) / 2)
distvec = 1 - squareform(mat)
linkages = linkage(distvec, method='average')
tvals = np.linspace(0.0001, 1, 1000)
best_t = []
scores = []
cluster_labels = []
for t in tvals:
    clabs = fcluster(linkages, t=t, criterion='distance')
    if len(np.unique(clabs)) == 1 or len(np.unique(clabs)) == len(mat):
        print("no score")
    else:
        score = silhouette_score(mat, clabs)
        print('t:', t, 'score:', score)
        best_t.append(t)
        scores.append(score)
        cluster_labels.append(clabs)

# make a dataframe with best_t and scores and get the row with the best t
score_df = pd.DataFrame({'t': best_t, 'score': scores, 'cluster_labels': cluster_labels})
best_t = score_df.loc[score_df['score'].idxmax()]



# %% consensus clustering (second attempt) with averaging distances for each trial and then performing consensus clustering
##THIS THAT GOOD SHIT RIGHT HERE 02/22/24
def make_consensus_matrix2(rec_info):
    bins = np.arange(210, 500)
    matrices = []
    names = []
    for name, group in rec_info.groupby(['exp_group', 'rec_num']):
        names.append(name)
        consensus_matrix = np.empty((30, 30, len(group), 4))
        consensus_matrix[:] = np.nan
        group = group.reset_index(drop=True)
        for exp_num, row in group.iterrows():
            rec_dir = row['rec_dir']
            time_array, rate_array = h5io.get_rate_data(rec_dir)
            for din, rate in rate_array.items():
                dinnum = int(din[-1])
                if din != 'dig_in_4':
                    # downsample rate from 7000 bins to 700 by averaging every 10 bins
                    rate = rate.reshape(rate.shape[0], rate.shape[1], -1, 10).mean(axis=3)
                    n_trials = rate.shape[1]
                    #create empty distance matrix
                    dms = np.zeros((n_trials, n_trials, len(bins)))
                    for b, bn in enumerate(bins):
                        X = rate[:, :, bn].T
                        dms[:,:,b] = squareform(pdist(X))

                    dm = np.mean(dms, axis=2)
                    #fold the distance matrix to make it symmetrical
                    dm = (dm + dm.T) / 2
                    linkages = linkage(squareform(dm), method='ward')
                    distances = linkages[:, 2]
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

                    for i in range(n_trials):
                        for j in range(n_trials):
                            if j > i:
                                if cluster_labels[i] != cluster_labels[j]:
                                    consensus_matrix[i, j, exp_num, dinnum] = 1
                                else:
                                    consensus_matrix[i, j, exp_num, dinnum] = 0
                            elif j == i:
                                consensus_matrix[i, j, exp_num, dinnum] = 0

        consensus_matrix = np.nanmean(consensus_matrix, axis=(2, 3))
        # replace all nan with 0
        consensus_matrix = np.nan_to_num(consensus_matrix)
        # fold the consensus matrix to make it symmetrical
        consensus_matrix = (consensus_matrix + consensus_matrix.T) / 2
        matrices.append(consensus_matrix)
    return matrices, names

matrices, names = make_consensus_matrix2(rec_info)

plot_correlation_matrices(matrices, names)
plot_heirarchical_clustering(matrices, names)

# sweep over different values of t to get the best silhouette score
dfs = []
for i, mat in enumerate(matrices):
    distvec = squareform(mat)
    linkage_mat = linkage(distvec, method='ward')
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
# plot the dendograms
fig, leaves = plot_heirarchical_clustering(matrices, names, threshold=thresholds)

