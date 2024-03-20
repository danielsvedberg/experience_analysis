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


#new file called sterotypuy functions, witha ll the funcitons, and then push that. 

licks = np.array([10,7,8,9,30,4])
#make licks a column vector
licks = licks.reshape(-1, 1)
test = pdist(licks)
test = squareform(test)


proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
rec_info = proj.rec_info.copy()  # get the rec_info table
rec_dirs = rec_info['rec_dir']

PA = ana.ProjectAnalysis(proj)

all_units, held_df = PA.get_unit_info(overwrite=False)
anID = 'DS46'
test = held_df.query('exp_name == @anID')
test = test.loc[any([test['unit1'].isin(['unit025', 'unit026', 'unit031', 'unit032']), test['unit2'].isin(['unit025', 'unit026', 'unit031', 'unit032'])])]
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
        fig.savefig(PA.save_dir + '/consensus_matrix.png')
        fig.savefig(PA.save_dir + '/consensus_matrix.svg')


def plot_heirarchical_clustering(matrices, names, threshold=None, save=False):
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
        fig.savefig(PA.save_dir + '/dendograms.png')
        fig.savefig(PA.save_dir + '/dendograms.svg')
    return(fig, leaves)

# %% consensus clustering (second attempt) with averaging distances for each trial and then performing consensus clustering
##THIS THAT GOOD SHIT RIGHT HERE 02/22/24

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

matrices, names, df = make_consensus_matrix2(rec_info)

plot_correlation_matrices(matrices, names, save=True)
plot_heirarchical_clustering(matrices, names)

# sweep over different values of t to get the best silhouette score
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
# plot the dendograms
fig, leaves = plot_heirarchical_clustering(matrices, names, threshold=thresholds, save=True)


#merge df with rec_info
df = pd.merge(df, rec_info, on=(['rec_dir', 'exp_group']))

#group df by rec_dir and scale top_branch_dist from 0 to 1 in each group
df['top_branch_dist'] = df.groupby('rec_dir')['top_branch_dist'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

#for each row in df, get the largest set of cluster labels in the array stored in each row of cluster labels
#then make a new column called 'cluster_A' and store the length of the largest set of cluster labels in each index
#then make a new column called 'cluster_B' and store the length of the second largest set of cluster labels in each index
#then make a new column called 'cluster_A_avg_trial' and store the average index of the largest set of cluster labels in each index
#then make a new column called 'cluster_B_avg_trial' and store the average index of the second largest set of cluster labels in each index

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
    #get the index of avg_indices that is the smallest
    smallest_index = np.argmin(average_indices)
    largest_index = np.argmax(average_indices)

    #in clust A idxs, store the indices of cluster_labels that are equal to the label of the smallest index
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

#longform df by melting clust_A_size and clust_B_size as well as clust_A_avg_trial and clust_B_avg_trial
df_clust_size = pd.melt(df, id_vars=['exp_group', 'session', 'channel', 'exp_name'], value_vars=['clust_A_size', 'clust_B_size'], var_name='cluster', value_name='size')
#refactor values in cluster column so instead of 'clust_A_size' and 'clust_B_size' it is 'A' and 'B'
df_clust_size['cluster'] = df_clust_size['cluster'].str.replace('_size', '')

df_clust_trial = pd.melt(df, id_vars=['exp_group', 'session', 'channel', 'exp_name'], value_vars=['clust_A_avg_trial', 'clust_B_avg_trial'], var_name='cluster', value_name='avg_trial')
#refactor values in cluster column so instead of 'clust_A_avg_trial' and 'clust_B_avg_trial' it is 'A' and 'B'
df_clust_trial['cluster'] = df_clust_trial['cluster'].str.replace('_avg_trial', '')
#merge
df_clust = pd.merge(df_clust_size, df_clust_trial, on=['exp_group', 'session', 'channel', 'exp_name', 'cluster'])
#replace 'clust_A' and 'clust_B' with 'A' and 'B'
df_clust['cluster'] = df_clust['cluster'].str.replace('clust_', '')
#relabel A in to 'early' and B into 'late'
df_clust['cluster'] = df_clust['cluster'].str.replace('A', 'early')
df_clust['cluster'] = df_clust['cluster'].str.replace('B', 'late')

#make a bar plot of the size of the two largest clusters for each channel
#TODO statistics for this
g = sns.catplot(data=df_clust, kind='bar', x='cluster', y='size', row='exp_group', col='session', margin_titles=True,
                linewidth = 2, edgecolor='black', facecolor = (0, 0, 0, 0))
#map a catplot with stripplot to the same axes
g.map_dataframe(sns.stripplot, x='cluster', y='size', dodge=True, color='black')
#relabel the y axis to "cluster size (trials)"
g.set_ylabels('cluster size (trials)')
#remove 'exp group' from the row labels
g.set_titles(row_template = '{row_name}', col_template = '{col_var} {col_name}')

plt.show()
#save the plot
g.savefig(PA.save_dir + '/cluster_size_barplot.png')
g.savefig(PA.save_dir + '/cluster_size_barplot.svg')

df_list = []
for i, row in df.iterrows():
    clust_A_idxs = row['clust_A_idxs']
    clust_B_idxs = row['clust_B_idxs']
    clust_idxs = np.concatenate([clust_A_idxs, clust_B_idxs])
    clust_A_labels = np.repeat('early', len(clust_A_idxs))
    clust_B_labels = np.repeat('late', len(clust_B_idxs))
    clust_labels = np.concatenate([clust_A_labels, clust_B_labels])
    newdf = pd.DataFrame({'clust_idx': clust_idxs, 'cluster': clust_labels})
    newdf[['exp_group', 'session', 'channel', 'exp_name']] = row[['exp_group', 'session', 'channel', 'exp_name']]
    df_list.append(newdf)
newdf = pd.concat(df_list, ignore_index=True)

#plot a violin plot of the cluster idx
#make the plot twice as wide as it is tall
g = sns.catplot(data=newdf, kind='violin', x='session', y='clust_idx', row='exp_group', col='cluster', margin_titles=True,
                color='white', linewidth=2, edgecolor='black', saturation=1, aspect=1.5)
g.map_dataframe(sns.swarmplot, x='session', y='clust_idx', dodge=True, color='black', alpha=0.25)
g.set_ylabels('trial number')
g.set_titles(row_template = '{row_name}', col_template = '{col_name} {col_var}')
plt.show()
#save the plot
g.savefig(PA.save_dir + '/cluster_idx_violinplot.png')
g.savefig(PA.save_dir + '/cluster_idx_violinplot.svg')

#compute intra and inter-cluster distances
dflist = []
#loop through every row in df
for nm, group in df.groupby(['exp_name']):

    inter_distances = []
    intra_A_distances = []
    intra_B_distances = []
    all_trials_distances = []

    bins = np.arange(210, 500)
    nbins = len(bins)
    an_dist_mats = np.empty((len(group), nbins, 30, 30))
    an_dist_mats[:] = np.nan
    n_trials_list = []
    group = group.reset_index(drop=True)
    for i, row in group.iterrows():
        #load the rec_dir
        rec_dir = row['rec_dir']
        #create a string for the dig in from the channel
        din = 'dig_in_' + str(row['channel'])
        #get the rate arrays
        time_array, rate_array = h5io.get_rate_data(rec_dir, din=row['channel'])
        #get the number of trials
        n_trials = rate_array.shape[1]
        n_trials_list.append(n_trials)
        #downsample rate from 7000 bins to 700 by averaging every 10 bins
        rate = rate_array.reshape(rate_array.shape[0], rate_array.shape[1], -1, 10).mean(axis=3)
        #zscore the rate
        rate = (rate - np.mean(rate)) / np.std(rate)
        #iterate through each bin in bins with enumerate and get the average distance matrix across the bins
        for bdx, b in enumerate(bins):
            #get the rate for the current bin
            X = rate[:, :, b].T
            #calculate the pairwise distance matrix for the rate
            dm = squareform(pdist(X, metric=average_difference))

            an_dist_mats[i, bdx, 0:dm.shape[0], 0:dm.shape[0]] = dm

    #average an_dist_mats across the bins
    an_dist_mats = np.nanmean(an_dist_mats, axis=1)

    for i, row in group.iterrows():
        n_trials = n_trials_list[i]
        avg_dm = an_dist_mats[i, 0:n_trials, 0:n_trials]
        intra_A_dm = avg_dm[np.ix_(row['clust_A_idxs'], row['clust_A_idxs'])]
        intra_B_dm = avg_dm[np.ix_(row['clust_B_idxs'], row['clust_B_idxs'])]
        #get the upper triangle of the intra_A_dm and intra_B_dm and linearize
        intra_A_dm = intra_A_dm[np.triu_indices(intra_A_dm.shape[0], k=1)]
        intra_B_dm = intra_B_dm[np.triu_indices(intra_B_dm.shape[0], k=1)]
        AB_distances = avg_dm[np.ix_(row['clust_A_idxs'], row['clust_B_idxs'])]
        all_trial_dm = avg_dm[np.ix_(np.arange(n_trials), np.arange(n_trials))]
        all_trial_dm = all_trial_dm[np.triu_indices(all_trial_dm.shape[0], k=1)]
        #linearize AB_distances
        AB_distances = AB_distances.flatten()
        #append the linearized AB_distances to the list
        inter_distances.append(AB_distances)
        intra_A_distances.append(intra_A_dm)
        intra_B_distances.append(intra_B_dm)
        all_trials_distances.append(all_trial_dm)
    #make a dataframe with the inter_distances
    group['AB_distances'] = inter_distances
    group['intra_A_distances'] = intra_A_distances
    group['intra_B_distances'] = intra_B_distances
    group['all_trial_distances'] = all_trials_distances
    dflist.append(group)
#concatenate the list of dataframes into one dataframe
df = pd.concat(dflist, ignore_index=True)

#spread the distances into longform
df_long = df[['exp_group', 'session', 'channel', 'exp_name', 'AB_distances']].explode('AB_distances')
#make df_long['session' and 'channel'] int
df_long['session'] = df_long['session'].astype(int)
df_long['channel'] = df_long['channel'].astype(int)
#make AB_distances float
df_long['AB_distances'] = df_long['AB_distances'].astype(float)
df_long = df_long.groupby(['exp_group', 'session','channel','exp_name']).mean().reset_index()
#plot a violin plot of the distances
colors = {'naive': 'blue', 'suc_preexp': 'orange'}
g=sns.catplot(data=df_long, kind='box', x='session', y='AB_distances', row='exp_group', margin_titles=True, linewidth=2, aspect=3, color='white')
g.map_dataframe(sns.stripplot, x='session', y='AB_distances', dodge=True, hue='exp_name', alpha=1, jitter=0.4, palette='colorblind')
g.set_titles(row_template = '{row_name}')
g.set_ylabels('early-late cluster distance')
plt.show()
#save the plot
g.savefig(PA.save_dir + '/early_late_cluster_distance_violinplot.png')
g.savefig(PA.save_dir + '/early_late_cluster_distance_violinplot.svg')

#make df_long2 melting intra_A_distances and intra_B_distances
intra_dist_df = df[['exp_group', 'session', 'channel', 'exp_name', 'intra_A_distances', 'intra_B_distances']]
intra_dist_df = pd.melt(intra_dist_df, id_vars=['exp_group', 'session', 'channel', 'exp_name'], value_vars=['intra_A_distances', 'intra_B_distances'], var_name='cluster', value_name='intra_cluster_distance')
#refactor cluster column so instead of 'intra_A_distances' and 'intra_B_distances' it is 'A' and 'B'
intra_dist_df['cluster'] = intra_dist_df['cluster'].str.replace('_distances', '')
intra_dist_df['cluster'] = intra_dist_df['cluster'].str.replace('intra_', '')
#explode intra_cluster_distance
intra_dist_df = intra_dist_df.explode('intra_cluster_distance')
#make intra_cluster_distance float
intra_dist_df['intra_cluster_distance'] = intra_dist_df['intra_cluster_distance'].astype(float)
#refector A and B to 'early' and 'late'
intra_dist_df['cluster'] = intra_dist_df['cluster'].str.replace('A', 'early')
intra_dist_df['cluster'] = intra_dist_df['cluster'].str.replace('B', 'late')
#refactor exp_group to 'Naive' and 'Suc. Pre-exposed'
intra_dist_df['exp_group'] = intra_dist_df['exp_group'].str.replace('naive', 'Naive')
intra_dist_df['exp_group'] = intra_dist_df['exp_group'].str.replace('suc_preexp', 'Suc. Pre-exposed')

intra_dist_df = intra_dist_df.groupby(['exp_group', 'session','channel','exp_name', 'cluster']).mean().reset_index()
#make a channel-exp_name joint column
intra_dist_df['channel_exp_name'] = intra_dist_df['channel'].astype(str) + '_' + intra_dist_df['exp_name']

g = sns.relplot(data=intra_dist_df, kind='line', x='session', y='intra_cluster_distance', row='exp_group', col='cluster', hue='channel_exp_name', linewidth=2, aspect=3, facet_kws={'margin_titles': True})
plt.show()

#add 'taste' column to intra_dist_df by matching channel according to taste_map
taste_map = {0: 'Suc', 1: 'NaCl', 2: 'CA', 3: 'QHCl', 4: 'Spont'}
intra_dist_df['taste'] = intra_dist_df['channel'].map(taste_map)
g = sns.catplot(data=intra_dist_df, kind='box', x='cluster', y='intra_cluster_distance', row='exp_group', col='session', margin_titles=True, linewidth=2, saturation=1, aspect=0.75, color='white')
g.map_dataframe(sns.swarmplot, x='cluster', y='intra_cluster_distance', dodge=True, alpha=1, palette='colorblind')
g.set_ylabels('intra-cluster distance')
g.set_titles(row_template='{row_name}', col_template='{col_name} {col_var}')
plt.show()
#save the plots
g.savefig(PA.save_dir + '/intra_cluster_distance_violinplot.png')
g.savefig(PA.save_dir + '/intra_cluster_distance_violinplot.svg')

#it doesn't make sense to compare distances between sessions because there are different numbers of neurons in each session
all_dist_df = df[['exp_group', 'session', 'channel', 'exp_name', 'all_trial_distances']]
all_dist_df = pd.melt(all_dist_df, id_vars=['exp_group', 'session', 'channel', 'exp_name'], value_vars=['all_trial_distances'], var_name='cluster', value_name='all_trial_distance')
all_dist_df = all_dist_df.explode('all_trial_distance')
all_dist_df['all_trial_distance'] = all_dist_df['all_trial_distance'].astype(float)
all_dist_df['exp_group'] = all_dist_df['exp_group'].str.replace('naive', 'Naive')
all_dist_df['exp_group'] = all_dist_df['exp_group'].str.replace('suc_preexp', 'Suc. Pre-exposed')
#filter out exp group Suc. Pre-exposed
all_dist_df = all_dist_df.loc[all_dist_df['exp_group'] == 'Naive']
#all_dist_df = all_dist_df.groupby(['exp_group', 'session','channel','exp_name']).mean().reset_index()
g = sns.catplot(data=all_dist_df, kind='box', x='session', y='all_trial_distance', hue='exp_name', row='channel', margin_titles=True, linewidth=2, aspect=3)
plt.legend(loc='upper right')
g.map_dataframe(sns.stripplot, x='session', y='all_trial_distance', dodge=True, hue='exp_name', alpha=1, palette='colorblind')
#show the legend

plt.show()
