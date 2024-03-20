import blechpy
import numpy as np
import blechpy.dio.h5io as h5io
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import gridspec

import analysis as ana
import matplotlib.pyplot as plt
import feather
import seaborn as sns
import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score
import matplotlib.gridspec as gridspec

exp_group_order = {'naive': 0, 'suc_preexp': 1, 'True': 0, 'False': 1}
session_order = {1: 0, 2: 1, 3: 2}
exp_group_names = ['Naive', 'Suc. Pre-exposed']
exp_group_colors = ['Blues', 'Oranges']
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


def plot_correlation_matrices(matrices, names, save_dir=None, save=False):
    # get the maximum and minimum for every 3 entries of matrices
    max_vals = []
    min_vals = []
    for i in range(0, len(matrices), 3):
        max_val = max([np.max(m) for m in matrices[i:i + 3]])
        min_val = min([np.min(m) for m in matrices[i:i + 3]])
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
        fig.savefig(save_dir + '/consensus_matrix.png')
        fig.savefig(save_dir + '/consensus_matrix.svg')



def plot_heirarchical_clustering(matrices, names, save_dir= None, threshold=None, save=False):
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
        fig.savefig(save_dir + '/dendograms.png')
        fig.savefig(save_dir + '/dendograms.svg')
    return(fig, leaves)



def average_difference(u, v):
    return np.mean(np.abs(u - v))
def make_consensus_matrix2(rec_info):
    bins = np.arange(210, 500)
    df_list = []
    matrices = []
    names = []
    for name, group in rec_info.groupby(['exp_group', 'rec_num']):
        top_branch_dist = []
        dinnums = []
        cluster_arrays = []
        rec_dirs = []
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
                    dinnums.append(dinnum)
                    # downsample rate from 7000 bins to 700 by averaging every 10 bins
                    rate = rate.reshape(rate.shape[0], rate.shape[1], -1, 10).mean(axis=3)
                    # zscore the rate
                    rate = (rate - np.mean(rate)) / np.std(rate)
                    n_trials = rate.shape[1]
                    #create empty distance matrix
                    dms = np.zeros((n_trials, n_trials, len(bins)))
                    for b, bn in enumerate(bins):
                        X = rate[:, :, bn].T
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

                    #make a df with the cluster labels and top branch distance and dinnum
        cluster_dict = {'channel': dinnums, 'top_branch_dist': top_branch_dist}
        cluster_df = pd.DataFrame(cluster_dict)
        cluster_df['cluster_labels'] = cluster_arrays
        cluster_df['exp_group'] = name[0]
        cluster_df['session'] = name[1]
        cluster_df['rec_dir'] = rec_dirs
        df_list.append(cluster_df)
        df = pd.concat(df_list, ignore_index=True)
        consensus_matrix = np.nanmean(consensus_matrix, axis=(2, 3))
        # replace all nan with 0
        consensus_matrix = np.nan_to_num(consensus_matrix)
        # fold the consensus matrix to make it symmetrical
        consensus_matrix = (consensus_matrix + consensus_matrix.T) / 2
        matrices.append(consensus_matrix)
    return matrices, names, df




