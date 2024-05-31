import blechpy
import stereotypy_clustering_functions as scf
import blechpy.dio.h5io as h5io
import pandas as pd
from joblib import Parallel, delayed
import trialwise_analysis as ta
import analysis as ana
import numpy as np
import os


proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
rec_info = proj.rec_info.copy()  # get the rec_info table
rec_dirs = rec_info['rec_dir']

PA = ana.ProjectAnalysis(proj)

all_units, held_df = PA.get_unit_info(overwrite=False)

#%% consensus clustering (second attempt) with averaging distances for each trial and then performing consensus clustering

save_dir = PA.save_dir + os.sep + 'clustering_analysis'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
matrices, names, df = scf.make_consensus_matrix2(rec_info)

shuff_dir = PA.save_dir + os.sep + 'clustering_analysis/shuffles'
shuff_matrices,shuff_names_df,shuff_dfs = scf.make_consensus_matrix_shuffle(rec_info, n_iter=100, overwrite=False, save_dir=shuff_dir)

scf.plot_correlation_matrices(matrices, names, save=True, save_dir=save_dir)
scf.plot_heirarchical_clustering(matrices, names, save=True, save_dir=save_dir)


naive_mats = matrices[:3]
naive_names = names[:3]
scf.plot_correlation_matrices_single(naive_mats,naive_names, save=True, save_dir=save_dir)
scf.plot_heirarchical_clustering_single(naive_mats, naive_names, save=True, save_dir=save_dir)

shuff_matrices = shuff_matrices.mean(axis=0)

shuff_matrices_naive = shuff_matrices[:3]
shuff_names_naive = shuff_names_df[['exp_group','session']].drop_duplicates()
shuff_names_naive = shuff_names_naive[shuff_names_naive['exp_group'] == 'naive']
#turn shuff_names_naive from a df to a list of tuples
shuff_names_naive = [tuple(row) for i, row in shuff_names_naive.iterrows()]
scf.plot_correlation_matrices_single(shuff_matrices_naive, shuff_names_naive, save=True, save_dir=shuff_dir, flag='shuffle')
scf.plot_heirarchical_clustering_single(shuff_matrices_naive, shuff_names_naive, save=True, save_dir=shuff_dir, flag='shuffle')

# sweep over different values of t to get the best silhouette score
thresholds = scf.get_consensus_thresholds(matrices, names)
# plot the dendograms
fig, leaves = scf.plot_heirarchical_clustering(matrices, names, threshold=thresholds, save=True, save_dir=save_dir)
naive_thresh = thresholds[:3]
fig, leaves = scf.plot_heirarchical_clustering_single(naive_mats, naive_names, threshold=thresholds, save=True, save_dir=save_dir)

#merge df with rec_info
df = pd.merge(df, rec_info, on=(['rec_dir', 'exp_group']))
shuff_df = pd.merge(shuff_dfs, rec_info, on=(['rec_dir', 'exp_group']))

#group df by rec_dir and scale top_branch_dist from 0 to 1 in each group
df['top_branch_dist'] = df.groupby('rec_dir')['top_branch_dist'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

df = scf.get_AB_clustering(df)
shuff_df = scf.get_AB_clustering(shuff_df)

#%% plot the cluster sizes
df_AB_long = scf.longform_AB_clustering(df)
#scf.plot_cluster_sizes(df_AB_long, save_dir=save_dir)
shuff_AB_long = scf.longform_AB_clustering(shuff_df, shuffle=True)

scf.plot_cluster_sizes_w_shuff(df_AB_long, shuff_AB_long, save_dir=save_dir)

#%% plot the average trial of the two largest clusters
import pingouin as pg

df_AB_labels = scf.get_AB_cluster_labels(df)
newdf_naive = df_AB_labels[df_AB_labels['exp_group'] == 'naive']
scf.plot_cluster_avg_trial_naive(newdf_naive, save_dir=save_dir)

shuff_df_AB_labels = scf.get_AB_cluster_labels(shuff_df, shuffle=True)
shuff_df_AB_labs_naive = shuff_df_AB_labels[shuff_df_AB_labels['exp_group'] == 'naive']

aov = scf.plot_cluster_avg_trial_naive_w_shuff(newdf_naive, shuff_df_AB_labs_naive, save_dir=save_dir, flag='with_shuffle')

#%% plot the intra inter and null cluster distances
#compute intra and inter-cluster distances
intra_inter_df = scf.get_intra_inter_distances(df)

scf.plot_cluster_distances(intra_inter_df, save_dir=save_dir)
scf.plot_cluster_distances_naive(intra_inter_df, save_dir=save_dir)

#%% calculate and plot Euclidean and cosine distances for each taste trial
import blechpy
import stereotypy_clustering_functions as scf
import blechpy.dio.h5io as h5io
import pandas as pd
from joblib import Parallel, delayed
import trialwise_analysis as ta
import analysis as ana
import numpy as np
import os


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
        #euc_dist_mat = (euc_dist_mat - np.mean(euc_dist_mat)) / np.std(euc_dist_mat)

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
    df['euclidean_distance'] = df['euclidean_distance'].transform(lambda x: (x - x.mean()) / x.std())
    # subtract the min of 'session_trial' from 'session_trial' to get the session_trial relative to the start of the recording
    df['session_trial'] = df['session_trial'] - df['session_trial'].min()
    return df



proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
rec_info = proj.rec_info.copy()  # get the rec_info table
rec_dirs = rec_info['rec_dir']
PA = ana.ProjectAnalysis(proj)
all_units, held_df = PA.get_unit_info(overwrite=False)

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
trial_col = 'taste_trial'
value_col = 'euclidean_distance'
preprodf, shuffle = ta.preprocess_nonlinear_regression(final_df, subject_col, group_cols, trial_col, value_col,
                                                       nIter=10000, save_dir=PA.save_dir, overwrite=False)

ta.plot_fits(preprodf, trial_col='taste_trial', dat_col='euclidean_distance', save_dir=PA.save_dir, flag='euclidean_distance_demo', time_col='session')

flag = 'stereotypy_euclidean_distance'
nIter = 10000
textsize = 20
parallel = True
yMin = preprodf[value_col].min()
yMax = preprodf[value_col].max()

ta.plotting_pipeline(preprodf, shuffle, trial_col, value_col, nIter=nIter, save_dir=PA.save_dir, flag=flag)


ta.plot_fits_summary_avg(preprodf, shuff_df=shuffle, dat_col=value_col, trial_col=trial_col, save_dir=PA.save_dir,
                         use_alpha_pos=False, textsize=textsize, dotalpha=0.15, flag=flag, nIter=nIter,
                         parallel=parallel, ymin=yMin, ymax=yMax)

for exp_group, group in preprodf.groupby(['exp_group']):
    group_shuff = shuffle.groupby('exp_group').get_group(exp_group)
    if flag is not None:
        save_flag = exp_group + '_' + flag
    else:
        save_flag = exp_group
    ta.plot_fits_summary_avg(group, shuff_df=group_shuff, dat_col=value_col, trial_col=trial_col,
                             save_dir=PA.save_dir, use_alpha_pos=False, textsize=textsize, dotalpha=0.15,
                             flag=save_flag, nIter=nIter, parallel=parallel, ymin=yMin, ymax=yMax)

ta.plot_fits_summary(preprodf, dat_col=value_col, trial_col=trial_col, save_dir=PA.save_dir, time_col='session',
                     use_alpha_pos=False, dotalpha=0.15, flag=flag)

ta.plot_nonlinear_regression_stats(preprodf, shuffle, subject_col=subject_col, group_cols=group_cols,
                                   trial_col=trial_col, value_col=value_col, save_dir=PA.save_dir, flag=flag,
                                   textsize=textsize, nIter=nIter)

pred_change_df, pred_change_shuff = ta.get_pred_change(preprodf, shuffle, subject_col=subject_col,
                                                       group_cols=group_cols, trial_col=trial_col)
ta.plot_predicted_change(pred_change_df, pred_change_shuff, group_cols, value_col=value_col, trial_col=trial_col,
                         save_dir=PA.save_dir, flag=flag, textsize=textsize, nIter=nIter)

ta.plot_session_differences(pred_change_df, pred_change_shuff, subject_col, group_cols,
                            trial_col=trial_col, value_col=value_col, stat_col='pred. change', save_dir=PA.save_dir,
                            flag=flag, textsize=textsize, nIter=nIter)

#%% calculate and model the average inter-trial distances for each taste trial
import blechpy
import stereotypy_clustering_functions as scf
import blechpy.dio.h5io as h5io
import pandas as pd
from joblib import Parallel, delayed
import trialwise_analysis as ta
import analysis as ana
import numpy as np
import os

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is

proj = blechpy.load_project(proj_dir)  # load the project
PA = ana.ProjectAnalysis(proj)
proj_save_dir = PA.save_dir
folder = 'intertrial_distances'
save_dir = proj_save_dir + os.sep + folder
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print('Created directory:', save_dir)

rec_info = proj.rec_info.copy()  # get the rec_info table
rec_dirs = rec_info['rec_dir']
PA = ana.ProjectAnalysis(proj)
all_units, held_df = PA.get_unit_info(overwrite=False)

# Parallelize processing of each rec_dir
num_cores = -1  # Use all available cores
final_dfs = Parallel(n_jobs=num_cores)(delayed(scf.get_avg_intertrial_distances)(rec_dir) for rec_dir in rec_dirs)

# Concatenate all resulting data frames into one
final_df = pd.concat(final_dfs, ignore_index=True)

# merge in rec_info into final_df
final_df = pd.merge(final_df, rec_info, on='rec_dir')
final_df['session'] = final_df['rec_num']


subject_col = 'exp_name'
group_cols = ['exp_group', 'session', 'taste']
trial_col = 'taste_trial'
value_col = 'euclidean_distance'
preprodf, shuffle = ta.preprocess_nonlinear_regression(final_df, subject_col, group_cols, trial_col, value_col,
                                                       nIter=1000, save_dir=save_dir, overwrite=True)

ta.plot_fits(preprodf, trial_col='taste_trial', dat_col='euclidean_distance', save_dir=save_dir, flag='intertrial_distance_demo', time_col='session')

flag = 'intertrial_euc_dist'
nIter = 10000
textsize = 20
parallel = True
yMin = preprodf[value_col].min()
yMax = preprodf[value_col].max()

ta.plotting_pipeline(preprodf, shuffle, trial_col, value_col, nIter=nIter, save_dir=save_dir, flag=flag)


ta.plot_fits_summary_avg(preprodf, shuff_df=shuffle, dat_col=value_col, trial_col=trial_col, save_dir=save_dir,
                         use_alpha_pos=False, textsize=textsize, dotalpha=0.15, flag=flag, nIter=nIter,
                         parallel=parallel, ymin=yMin, ymax=yMax)

for exp_group, group in preprodf.groupby(['exp_group']):
    group_shuff = shuffle.groupby('exp_group').get_group(exp_group)
    if flag is not None:
        save_flag = exp_group + '_' + flag
    else:
        save_flag = exp_group
    ta.plot_fits_summary_avg(group, shuff_df=group_shuff, dat_col=value_col, trial_col=trial_col,
                             save_dir=save_dir, use_alpha_pos=False, textsize=textsize, dotalpha=0.15,
                             flag=save_flag, nIter=nIter, parallel=parallel, ymin=yMin, ymax=yMax)

ta.plot_fits_summary(preprodf, dat_col=value_col, trial_col=trial_col, save_dir=save_dir, time_col='session',
                     use_alpha_pos=False, dotalpha=0.15, flag=flag)

ta.plot_nonlinear_regression_stats(preprodf, shuffle, subject_col=subject_col, group_cols=group_cols,
                                   trial_col=trial_col, value_col=value_col, save_dir=save_dir, flag=flag,
                                   textsize=textsize, nIter=nIter)

pred_change_df, pred_change_shuff = ta.get_pred_change(preprodf, shuffle, subject_col=subject_col,
                                                       group_cols=group_cols, trial_col=trial_col)
ta.plot_predicted_change(pred_change_df, pred_change_shuff, group_cols, value_col=value_col, trial_col=trial_col,
                         save_dir=save_dir, flag=flag, textsize=textsize, nIter=nIter)

ta.plot_session_differences(pred_change_df, pred_change_shuff, subject_col, group_cols,
                            trial_col=trial_col, value_col=value_col, stat_col='pred. change', save_dir=save_dir,
                            flag=flag, textsize=textsize, nIter=nIter)

