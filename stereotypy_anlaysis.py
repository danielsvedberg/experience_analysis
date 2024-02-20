import blechpy
import numpy as np
import blechpy.dio.h5io as h5io
import pandas as pd
from joblib import Parallel, delayed
import trialwise_analysis as ta
import analysis as ana
import matplotlib.pyplot as plt
import feather

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy' # directory where the project is
proj = blechpy.load_project(proj_dir) #load the project
rec_info = proj.rec_info.copy() #get the rec_info table
rec_dirs = rec_info['rec_dir']

PA = ana.ProjectAnalysis(proj)

def get_trial_info(dat):
    dintrials = dat.dig_in_trials
    dintrials['taste_trial'] = 1
    #groupby name and cumsum taste trial
    dintrials['taste_trial'] = dintrials.groupby('name')['taste_trial'].cumsum()
    #rename column trial_num to 'session_trial'
    dintrials = dintrials.rename(columns={'trial_num':'session_trial','name':'taste'})
    #select just the columns 'taste_trial', 'taste', 'session_trial', 'channel', and 'on_time'
    dintrials = dintrials[['taste_trial', 'taste', 'session_trial', 'channel', 'on_time']]
    return dintrials

#%% calculate and plot euclidean and cosine distances for each taste trial
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
            'channel': int(din[-1]), #get the din number from string din
            'taste_trial': np.arange(rate.shape[1])
        })
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    #add index info to df from dintrials using merge on taste_trial and channel
    df = pd.merge(df, dintrials, on=['taste_trial', 'channel'])
    #remove all rows where taste == 'Spont'
    df = df.loc[df['taste'] != 'Spont']
    #subtract the min of 'session_trial' from 'session_trial' to get the session_trial relative to the start of the recording
    df['session_trial'] = df['session_trial'] - df['session_trial'].min()
    return df


# Parallelize processing of each rec_dir
num_cores = -1  # Use all available cores
final_dfs = Parallel(n_jobs=num_cores)(delayed(process_rec_dir)(rec_dir) for rec_dir in rec_dirs)

# Concatenate all resulting data frames into one
final_df = pd.concat(final_dfs, ignore_index=True)

#merge in rec_info into final_df
final_df = pd.merge(final_df, rec_info, on='rec_dir')
final_df['session'] = final_df['rec_num']

subject_col = 'exp_name'
group_cols = ['exp_group','session','taste']
trial_col = 'session_trial'
value_col = 'euclidean_distance'
preprodf, shuffle = ta.preprocess_nonlinear_regression(final_df,subject_col,group_cols,trial_col,value_col,nIter=10000, save_dir=PA.save_dir, overwrite=False)

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

ta.plot_nonlinear_regression_stats(preprodf, shuffle, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col,value_col=value_col, save_dir=PA.save_dir, flag=flag, textsize=textsize, nIter=nIter)

pred_change_df, pred_change_shuff = ta.get_pred_change(preprodf, shuffle, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col)
ta.plot_predicted_change(pred_change_df, pred_change_shuff, group_cols, value_col=value_col, trial_col=trial_col, save_dir=PA.save_dir, flag=flag, textsize=textsize, nIter=nIter)

# make a matrix of the Euclidean distance for each taste trial for each taste and session, then take the average
def make_euc_dist_matrix(rec_dir):
    df_list = []
    dat = blechpy.load_dataset(rec_dir)
    dintrials = get_trial_info(dat)
    dintrials['taste_trial'] = dintrials['taste_trial'] - 1
    time_array, rate_array = h5io.get_rate_data(rec_dir)
    #set up a dict to store a matrix for each key in rate_array
    euc_dist_mats = {}
    for din, rate in rate_array.items():
        #make a virtual matrix of the euclidean distance between each trial for each bin
        euc_distances = []
        taste_trial_A = []
        taste_trial_B = []
        for i in range(rate.shape[1]):
            for j in range(rate.shape[1]):
                if j > i:
                    pass
                else:
                    slice_distances = np.zeros(rate.shape[2])
                    for s in range(rate.shape[2]):
                        euc_dist = np.linalg.norm(rate[:,i,s] - rate[:,j,s])
                        slice_distances[s] = euc_dist
                    euc_distances.append(np.mean(slice_distances))
                    taste_trial_A.append(i)
                    taste_trial_B.append(j)
        #make a dataframe from the euc_distances, with columns 'taste_trial_A', 'taste_trial_B', and 'euc_dist'
        euc_dist_df = pd.DataFrame({
            'taste_trial': taste_trial_A,
            'trial_B': taste_trial_B,
            'euc_dist': euc_distances
        })
        #add the rec_dir and din to the dataframe
        euc_dist_df['rec_dir'] = rec_dir
        euc_dist_df['channel'] = int(din[-1])
        #add the dataframe to df_list
        df_list.append(euc_dist_df)
    #concatenate all the dataframes in df_list
    df = pd.concat(df_list, ignore_index=True)
    #add index info to df from dintrials using merge on taste_trial and channel
    df = pd.merge(df, dintrials, on=['taste_trial', 'channel'])
    #remove all rows where taste == 'Spont'
    df = df.loc[df['taste'] != 'Spont']
    #subtract the min of 'session_trial' from 'session_trial' to get the session_trial relative to the start of the recording
    df['session_trial'] = df['session_trial'] - df['session_trial'].min()
    return df

# Parallelize processing of each rec_dir
num_cores = -1  # Use all available cores
final_dfs = Parallel(n_jobs=num_cores)(delayed(make_euc_dist_matrix)(rec_dir) for rec_dir in rec_dirs)
final_df = pd.concat(final_dfs, ignore_index=True)
final_df = pd.merge(final_df, rec_info, on='rec_dir')
final_df['session'] = final_df['rec_num']

#save final_df to feather
final_df.to_feather(PA.save_dir + '/trial_euc_dists.feather')
final_df['euc_dist'] = final_df.groupby(['exp_group'])['euc_dist'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))


#group final_df by 'taste_trial', 'trial_B', 'taste', 'exp_group', and 'session', and take the mean of euclidean distance
taste_mean_df = final_df.groupby(['taste_trial', 'trial_B', 'taste', 'channel','exp_group', 'session']).mean().reset_index()
all_mean_df = final_df.groupby(['taste_trial', 'trial_B', 'exp_group','session']).mean().reset_index()

#get a matrix of the mean euclidean distance for each taste trial for each taste and session
#instantiate a 4x3 grid of subplots

tastes = ['Suc', 'NaCl', 'CA', 'QHCl']
sessions = [1, 2, 3]
exp_groups = ["naive","suc_preexp"]
for exp_group in exp_groups:
    fig, axs = plt.subplots(4, 3, figsize=(9,10))
    for i, taste in enumerate(tastes):
        for j, session in enumerate(sessions):
            group = taste_mean_df[(taste_mean_df['taste'] == taste) & (taste_mean_df['session'] == session) & (taste_mean_df['exp_group'] == exp_group)]
            for_pivot = group[['taste_trial', 'trial_B', 'euc_dist']]
            taste_matrix = for_pivot.pivot(index='trial_B', columns='taste_trial', values='euc_dist')
            ax = axs[i, j]
            cax = ax.matshow(taste_matrix, cmap='viridis')
            if j != 0:
                ax.set_yticks([])
            if i != 0:
                ax.set_xticks([])
            if i == 0:
                ax.set_title('Session ' + str(session), pad=20, fontsize=20)
            if j == 2:
                #set a y label on the right axis with the taste
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(taste, rotation=-90, labelpad=60, fontsize=20)
                #also include the colorbar
                fig.colorbar(cax, ax=ax)
    for ax in axs[:,0]:
        ax.set_ylabel('Trial')
    #put a title over the entire figure with the exp_group
    fig.suptitle(exp_group, fontsize=20)
    #reduce the space between columns in the subplot
    plt.subplots_adjust(wspace=0, hspace=0.01)
    plt.show()
    #save the figure
    fig.savefig(PA.save_dir + '/taste_euc_dist_heatmap_' + exp_group + '.png')


fig, ax = plt.subplots(2,3, figsize=(10,5))
for k, exp_group in enumerate(exp_groups):
    for j, session in enumerate(sessions):
        group = all_mean_df[(all_mean_df['session'] == session) & (all_mean_df['exp_group'] == exp_group)]
        for_pivot = group[['taste_trial', 'trial_B', 'euc_dist']]
        all_matrix = for_pivot.pivot(index='trial_B', columns='taste_trial', values='euc_dist')
        cax = ax[k,j].matshow(all_matrix, cmap='viridis')
        if j != 0:
            ax[k,j].set_yticks([])
        if k != 0:
            ax[k,j].set_xticks([])
        if k == 0:
            ax[k,j].set_title('Session ' + str(session), pad=20, fontsize=20)
        if j == 2:
            ax[k,j].yaxis.set_label_position("right")
            ax[k,j].set_ylabel(exp_group, rotation=-90, labelpad=60, fontsize=20)
            fig.colorbar(cax, ax=ax[k,j])
for ax in ax[:,0]:
    ax.set_ylabel('Trial')
plt.subplots_adjust(wspace=0, hspace=0.01)
plt.show()
#save the figure
fig.savefig(PA.save_dir + '/all_euc_dist_heatmap.png')
#plot a correlation matrix of the mean euclidean distance for each taste trial for each taste and session




#%% heirarchical clustering analysis of euclidean distance


from scipy.spatial.distance import pdist
def make_euc_dist_matrix2(rec_dir):
    df_list = []
    dat = blechpy.load_dataset(rec_dir)
    dintrials = get_trial_info(dat)
    dintrials['taste_trial'] = dintrials['taste_trial'] - 1
    time_array, rate_array = h5io.get_rate_data(rec_dir)
    #set up a dict to store a matrix for each key in rate_array
    for din, rate in rate_array.items():
        #make a virtual matrix of the euclidean distance between each trial for each bin
        bins = np.arange(2000, 5000)
        dists = []
        for b in bins:
            euc_dist = pdist(rate[:,:,b].T, 'euclidean')
            dists.append(euc_dist)
        #turn dists into a matrix
        euc_dists = np.array(dists)
        #take the mean of euc dists along the 0 axis
        euc_dists = euc_dists.mean(axis=0)
        #make an index called dist_idx which is just the number index of the euc_dists array
        dist_idx = np.arange(len(euc_dists))
        #make a dataframe from the euc_distances, and bins
        euc_dist_df = pd.DataFrame({'euc_dist': euc_dists, 'dist_idx': dist_idx})
        #add the rec_dir and din to the dataframe
        euc_dist_df['rec_dir'] = rec_dir
        euc_dist_df['channel'] = int(din[-1])
        euc_dist_df['n_trials'] = rate.shape[1]
        #add the dataframe to df_list
        df_list.append(euc_dist_df)
    #concatenate all the dataframes in df_list
    df = pd.concat(df_list, ignore_index=True)
    return df

#get rows of rec_info where exp_group is 'naive' and session is 1
df = rec_info.loc[(rec_info['exp_group'] == 'naive') & (rec_info['rec_num'] == 1)]
din = ['dig_in_0']

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def cluster_data(sample, t):
    distance_matrix = pdist(sample)
    linkage_matrix = linkage(distance_matrix, method='average')
    cluster_labels = fcluster(linkage_matrix, t=t, criterion='distance')
    return cluster_labels
def opti_t_bin(X):
    t_range = np.linspace(0.1, 100, 100)
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
            #downsample by averaging every 10 bins
            rate = rate.reshape(rate.shape[0], rate.shape[1], -1, 10).mean(axis=3)
            # z-score the rate
            #rate = (rate - rate.mean()) / rate.std()
            #average rate across bins
            X = rate[:, :, bins].mean(axis=2).T
            optimized = opti_t_bin(X)
            #optimized = Parallel(n_jobs=-1)(delayed(optimize_bin)(rate, b) for b in bins)
            #replace all entries in list optimal_ts that are None with Nan
            if optimized is None:
                optimized = np.nan
            #optimized = #[np.nan if x is None else x for x in optimized]
            optimal_ts.append(np.nanmean(optimized))
    optimal_ts = np.nanmean(optimal_ts)
    return optimal_ts

rec_dirs = rec_info['rec_dir']
ts = []
for rec_dir in rec_dirs:
    print(rec_dir)
    ts.append(optimize_t(rec_dir))

#make a dataframe with rec_dir and ts
ts_df = pd.DataFrame({'rec_dir': rec_dirs, 'best_t_val': ts})
rec_info = proj.rec_info.copy()
#merge ts_df with rec_info
rec_info = pd.merge(rec_info, ts_df, on='rec_dir')

def make_consensus_matrix(rec_info):
    bins = np.arange(200, 500)
    matrices = []
    names = []
    for name, group in rec_info.groupby(['exp_group', 'rec_num']):
        names.append(name)
        consensus_matrix = np.zeros((30,30))
        for _, row in group.iterrows():
            rec_dir = row['rec_dir']
            time_array, rate_array = h5io.get_rate_data(rec_dir)
            for din, rate in rate_array.items():
                #downsample rate from 7000 bins to 700 by averaging every 10 bins
                rate = rate.reshape(rate.shape[0], rate.shape[1], -1, 10).mean(axis=3)
                #z-score the rate
                #rate = (rate - rate.mean()) / rate.std()
                if din != 'dig_in_4':
                    n_trials = rate.shape[1]
                    for b in bins:
                        X = rate[:, :, b].T
                        cluster_labels = cluster_data(X, t=row['best_t_val'])
                        for i in range(n_trials):
                            for j in range(n_trials):
                                if j > i:
                                    if cluster_labels[i] == cluster_labels[j]:
                                        consensus_matrix[i, j] += 1

        div_factor = len(group) * (len(rate_array.keys()) - 1) * len(bins)
        consensus_matrix /= div_factor
        matrices.append(consensus_matrix)
    return matrices, names

matrices, names = make_consensus_matrix(rec_info)

for i, mat in enumerate(matrices):
    #fold mat to make it symmetrical
    mat = squareform((mat + mat.T) / 2)
    consensus_linkage_matrix = linkage(mat, method='average')
    plt.figure(figsize=(10, 7))
    label = str(list(names[i]))
    dendrogram(consensus_linkage_matrix)
    #make the title label
    plt.title(label)
    plt.show()



# Parallelize processing of each rec_dir
num_cores = -1  # Use all available cores
final_dfs = Parallel(n_jobs=num_cores)(delayed(make_euc_dist_matrix2)(rec_dir) for rec_dir in rec_dirs)
final_df = pd.concat(final_dfs, ignore_index=True)
final_df = pd.merge(final_df, rec_info, on='rec_dir')
final_df['session'] = final_df['rec_num']
#save final_df to feather
final_df.to_feather(PA.save_dir + '/trial_euc_dists2.feather')

#scale the euc_dist column for each grouping of exp_group
final_df['euc_dist'] = final_df.groupby(['exp_group'])['euc_dist'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

channel_taste_map = {0: 'Suc', 1: 'NaCl', 2: 'CA', 3: 'QHCl', 4: 'Spont'}
#make channel int
final_df['channel'] = final_df['channel'].astype(int)
#add tastes to final_df using the taste_map
final_df['taste'] = final_df['channel'].map(channel_taste_map)
#filter out spont trials
final_df = final_df.loc[final_df['taste'] != 'Spont']
#group final_df by 'taste_trial', 'trial_B', 'taste', 'exp_group', and 'session', and take the mean of euclidean distance
taste_mean_df = final_df.groupby(['dist_idx', 'taste', 'channel','exp_group', 'session']).mean().reset_index()
all_mean_df = final_df.groupby(['dist_idx', 'exp_group','session']).mean().reset_index()

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score
#make a matrix of the mean euclidean distance for each taste trial for each taste and session
for nm, group in all_mean_df.groupby(['exp_group', 'session']):
    Z = linkage(group['euc_dist'], 'ward')
    fig, ax = plt.subplots()
    dn = dendrogram(Z, ax=ax)
    ax.set_title(nm)
    plt.show()

