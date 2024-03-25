import blechpy
import numpy as np
import blechpy.dio.h5io as h5io
import pandas as pd
from joblib import Parallel, delayed
import trialwise_analysis as ta
import analysis as ana
import matplotlib.pyplot as plt
import feather
import seaborn as sns
import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score
import os
import matplotlib.gridspec as gridspec
import pingouin as pg


def average_difference(u, v):
    return np.mean(np.abs(u - v))
def make_consensus_matrix2(rec_info, shuffle=False):
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
                    n_trials = rate.shape[1]
                    if shuffle:
                        # create a trial index that is resampled with replacement
                        trial_index = np.arange(n_trials)
                        resampled_index = np.random.choice(trial_index, size=n_trials, replace=True)
                        # recreate rate with resampled index
                        rate = rate[:, resampled_index, :]
                    # zscore the rate
                    rate = (rate - np.mean(rate)) / np.std(rate)

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
            # ax.set_ylabel('Trial', fontsize=20)
            ax.set_yticks([0, 9, 19, 29])
            ax.set_yticklabels(ax.get_yticks(), fontsize=18)
            ax.set_ylabel('Trial', fontsize=20)


        ax.set_xlabel('Trial', fontsize=20)
        ax.set_xticks([0, 9, 19, 29])
        ax.set_xticklabels(ax.get_xticks(), fontsize=18)
        ax.xaxis.set_ticks_position('bottom')

        ax.set_title('Session ' + str(session), pad=-6, fontsize=20)
        # Set the ylabel on the right side of the last column of the data plots
        if k == 2:  # Corrected condition to match the last column of the data plots
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(exp_group_name, rotation=270, labelpad=20, fontsize=20)

        # Add one color bar per row in the first column
        if k == 1:  # This ensures color bars are added once per row
            cb = fig.colorbar(cax, cax=cbar_axes, orientation='horizontal')
            cbar_axes.yaxis.set_ticks_position('left')
            cbar_axes.set_xlabel('distance index', fontsize=20)#, labelpad=-80)

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
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
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
            leaf = dendrogram(link, ax=ax, leaf_rotation=90, leaf_font_size=8, get_leaves=True, color_threshold=threshold[i])
        else:
            leaf = dendrogram(link, ax=ax, leaf_rotation=90, leaf_font_size=8, get_leaves=True)
        leaves.append(leaf)
        # set max of y-axis to max_y
        ax.set_ylim(0, max_y)

        # for the first row, set the title to the session number
        ax.set_title('Session ' + str(session))
        # for the last column, set the ylabel to the exp_group on the right axis
        if k == 2:
            ax.set_ylabel(exp_group, rotation=-90, labelpad=20, fontsize=20)
            ax.yaxis.set_label_position("right")

        # for the first column, set the ylabel to 'Cluster distan"
        if k == 0:
            ax.set_ylabel('Cluster distance', fontsize=20)
        else:
            ax.set_yticks([])

        # for the last row, set the xlabel to 'Trial'
        ax.set_xlabel('Trial', fontsize=20)
    plt.tight_layout()
    #plt.subplots_adjust(wspace=0.05, hspace=0.2)
    plt.show()
    # save the figure
    plot_name = save_dir + os.sep + exp_group + '_dendograms'
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
    cluster_x_map = {'early': 0, 'late': 1, 'total': 2}
    bonf_corr = 6 # 2 comparisons across 3 days 6 total comparisons
    conf = 0.05 / bonf_corr
    ntiles = [conf, 1 - conf]

    fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    for session in [1,2,3]:
        ax = axs[session - 1]
        session_df = df_AB_long[df_AB_long['session'] == session]
        session_shuff = shuff_AB_long[shuff_AB_long['session'] == session]
        session_df['size'] = session_df['size']/30 * 100
        session_shuff['size'] = session_shuff['size']/30 * 100


        #make a 'total' cluster that is the sum of the early and late clusters
        total_df = session_df.groupby(['exp_group', 'session', 'channel','exp_name']).sum().reset_index()
        total_df['cluster'] = 'total'
        session_df = pd.concat([session_df, total_df], ignore_index=True)

        total_shuff = session_shuff.groupby(['exp_group', 'session', 'channel','exp_name', 'iter']).sum().reset_index()
        total_shuff['cluster'] = 'total'
        session_shuff = pd.concat([session_shuff, total_shuff], ignore_index=True)

        session_shuff = session_shuff.groupby(['exp_group', 'session', 'channel', 'cluster', 'iter'])['size'].mean().reset_index()

        shuff_quantiles = session_shuff.groupby('cluster')['size'].quantile(ntiles).unstack().reset_index()
        # rename the columns of shuff_quantiles to 'lower' and 'upper'
        shuff_quantiles.columns = ['cluster', 'lower', 'upper']
        shuff_quantiles['height'] = shuff_quantiles['upper'] - shuff_quantiles['lower']

        # make a floating barplot with cluster on the x-axis, 0.025 the bottom of each bar, and 0.975 the top of each bar
        ax.bar(x=shuff_quantiles['cluster'], height=shuff_quantiles['height'], bottom=shuff_quantiles['lower'],
               color='gray', alpha=0.5)
        sns.barplot(data=session_df, x='cluster', y='size', ax=ax, capsize=0.1, errwidth=2, edgecolor='black',
                    order=['early', 'late', 'total'], facecolor=(0, 0, 0, 0))

        for cluster, group in session_df.groupby('cluster'):
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
                ax.text(xpos, 75, stars, fontsize=20, ha='center', fontweight='bold')
        #set y lim
        ax.set_ylim(0, 100)
        if session == 1:
            ax.set_ylabel('% of trials', fontsize=20)
            yticks = [0,20,40,60,80,100]
            ax.set_yticks(yticks)
            ax.set_yticklabels(ax.get_yticks(), fontsize=17)
        else:
            ax.set_ylabel('')
        ax.set_title('Session ' + str(session))
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=17)

    plt.tight_layout()
    plt.show()
    # save the plot
    plt.savefig(save_dir + '/cluster_size_barplot_naive.png')
    plt.savefig(save_dir + '/cluster_size_barplot_naive.svg')


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
            ax.text(0.5, 26, stars, fontsize=20, ha='center')

        ax.set_title('Session ' + str(session))
        if session == 1:
            ax.set_ylabel('Trial')
        else:
            ax.set_ylabel('')
        ax.set_xlabel('Cluster')
        ax.set_ylim(-1, 30)
        ax.set_yticks([0, 10, 20, 30])
        ax.set_yticklabels(ax.get_yticks(), fontsize=17)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=17)
        ax.set_xlabel('Cluster', fontsize=20)
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
    cluster_x_map = {'early': 0, 'late': 1}
    shuff_df['cluster'] = shuff_df['cluster'].str.replace('early shuffle', 'early')
    shuff_df['cluster'] = shuff_df['cluster'].str.replace('late shuffle', 'late')
    #group shuff_df by cluster, channel, exp_name, and iter, get the mean of clust_idx
    shuff_df = shuff_df.groupby(['session','channel', 'cluster', 'iter']).agg({'clust_idx': 'mean'}).reset_index()

    bonf_corr = 6 # 2 comparisons across 3 days 6 total comparisons
    conf = 0.05 / bonf_corr
    ntiles = [conf, 1 - conf]

    fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
    for session in [1, 2, 3]:
        ax = axs[session - 1]
        session_df = newdf[newdf['session'] == session]
        session_shuff = shuff_df[shuff_df['session'] == session]
        #for each grouping of cluster in session_shuff, get the 97.5% and 2.5% quantiles of clust_idx

        shuff_quantiles = session_shuff.groupby('cluster')['clust_idx'].quantile(ntiles).unstack().reset_index()
        #rename the columns of shuff_quantiles to 'lower' and 'upper'
        shuff_quantiles.columns = ['cluster', 'lower', 'upper']
        shuff_quantiles['height'] = shuff_quantiles['upper'] - shuff_quantiles['lower']
        #make a floating barplot with cluster on the x-axis, 0.025 the bottom of each bar, and 0.975 the top of each bar
        ax.bar(x=shuff_quantiles['cluster'], height=shuff_quantiles['height'], bottom=shuff_quantiles['lower'], color='gray', alpha=0.5)
        sns.barplot(data=session_df, x='cluster', y='clust_idx', ax=ax, capsize=0.1, errwidth=2, edgecolor='black',
                    order=['early','late'], facecolor=(0, 0, 0, 0))
        sns.swarmplot(data=session_df, x='cluster', y='clust_idx', ax=ax, color='tab:blue', alpha=0.5, size=2,
                      order=['early', 'late'])

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
                ax.text(xpos, 20, stars, fontsize=20, ha='center', fontweight='bold')

        ax.set_title('Session ' + str(session))
        if session == 1:
            ax.set_ylabel('Trial')
        else:
            ax.set_ylabel('')
        ax.set_xlabel('Cluster')
        ax.set_ylim(-1, 30)
        ax.set_yticks([0, 10, 20, 30])
        ax.set_yticklabels(ax.get_yticks(), fontsize=17)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=17)
        ax.set_xlabel('Cluster', fontsize=20)
    plt.tight_layout()
    plt.show()

    # save the plot
    savename = save_dir + '/cluster_avg_trial_naive_w_shuff'
    if flag is not None:
        savename = savename + '_' + flag
    plt.savefig(savename + '.png')
    plt.savefig(savename + '.svg')

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
