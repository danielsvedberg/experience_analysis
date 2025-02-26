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

def process_rec_dir(rec_dir, bsln_sub=False):
    df_list = []
    dat = blechpy.load_dataset(rec_dir)
    dintrials = get_trial_info(dat)
    time_array, rate_array = h5io.get_rate_data(rec_dir)
    for din, rate in rate_array.items():
        if bsln_sub:
            bsln_rate = np.mean(rate[:, :, 0:2000], axis=2)
            rate = rate - bsln_rate[:, :, np.newaxis]

        avg_firing_rate = np.mean(rate, axis=1)  # avg across trials, Neurons x Bins

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

        avg_cos_sim = np.mean(cos_sim_mat[:, 2100:5000], axis=1)
        avg_euc_dist = np.mean(euc_dist_mat[:, 2100:5000], axis=1)

        df = pd.DataFrame({
            'cosine_similarity': avg_cos_sim,
            'euclidean_distance': avg_euc_dist,
            'rec_dir': rec_dir,
            'channel': int(din[-1]),  # get the din number from string din
            'taste_trial': np.arange(rate.shape[1])
        })
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    df['taste_trial'] = df['taste_trial'] + 1  # add 1 to taste_trial to match the taste_trial in dintrials
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

sub_cols = ['exp_name']
gr_cols = ['exp_group', 'session', 'taste']
trial_col = 'taste_trial'
value_col = 'euclidean_distance'
folder = 'dist_to_avg_stereotypy'
proj_save_dir = PA.save_dir
save_dir = proj_save_dir + os.sep + folder
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print('Created directory:', save_dir)
nIter = 10000

preprodf, shuffle = ta.preprocess_nonlinear_regression(final_df, sub_cols, gr_cols, trial_col, value_col,
                                                       nIter=nIter, save_dir=save_dir, overwrite=False)
#ta.plotting_pipeline(preprodf, shuffle, trial_col, value_col, gr_cols, sub_cols, nIter=nIter, save_dir=save_dir)

#ta.plot_fits(preprodf, trial_col='taste_trial', dat_col='euclidean_distance', save_dir=save_dir,
#             time_col='session')

preproD13 = preprodf.loc[preprodf['session'] != 2]
shufflD13 = shuffle.loc[shuffle['session'] != 2]
ta.plotting_pipeline(preproD13, shufflD13, trial_col, value_col, gr_cols, sub_cols, nIter=nIter, save_dir=save_dir, flag = '_D_1_3_')

###baseline sub version
folder = 'dist_to_avg_stereotypy_bsln_sub'
proj_save_dir = PA.save_dir
save_dir = proj_save_dir + os.sep + folder
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print('Created directory:', save_dir)
num_cores = -1  # Use all available cores
final_dfs = Parallel(n_jobs=num_cores)(delayed(process_rec_dir)(rec_dir, bsln_sub=True) for rec_dir in rec_dirs)
# Concatenate all resulting data frames into one
final_df = pd.concat(final_dfs, ignore_index=True)
# merge in rec_info into final_df
final_df = pd.merge(final_df, rec_info, on='rec_dir')
final_df['session'] = final_df['rec_num']
subject_cols = ['exp_name']
group_cols = ['exp_group', 'session', 'taste']
trial_col = 'taste_trial'
value_col = 'euclidean_distance'
nIter = 10000
#preprodf, shuffle = ta.preprocess_nonlinear_regression(final_df, subject_cols, group_cols, trial_col, value_col,
#                                                       nIter=nIter, save_dir=save_dir, overwrite=False)
ta.plotting_pipeline(preprodf, shuffle, trial_col, value_col, group_cols, subject_cols, nIter=nIter, save_dir=save_dir)



# ta.plot_fits_summary_avg(preprodf, shuff_df=shuffle, dat_col=value_col, trial_col=trial_col, save_dir=PA.save_dir,
#                          use_alpha_pos=False, textsize=textsize, dotalpha=0.15, flag=flag, nIter=nIter,
#                          parallel=parallel, ymin=yMin, ymax=yMax)
#
# for exp_group, group in preprodf.groupby(['exp_group']):
#     group_shuff = shuffle.groupby('exp_group').get_group(exp_group)
#     if flag is not None:
#         save_flag = exp_group + '_' + flag
#     else:
#         save_flag = exp_group
#     ta.plot_fits_summary_avg(group, shuff_df=group_shuff, dat_col=value_col, trial_col=trial_col,
#                              save_dir=PA.save_dir, use_alpha_pos=False, textsize=textsize, dotalpha=0.15,
#                              flag=save_flag, nIter=nIter, parallel=parallel, ymin=yMin, ymax=yMax)
#
# ta.plot_fits_summary(preprodf, dat_col=value_col, trial_col=trial_col, save_dir=PA.save_dir, time_col='session',
#                      use_alpha_pos=False, dotalpha=0.15, flag=flag)
#
# ta.plot_nonlinear_regression_stats(preprodf, shuffle, subject_col=subject_col, group_cols=group_cols,
#                                    trial_col=trial_col, value_col=value_col, save_dir=PA.save_dir, flag=flag,
#                                    textsize=textsize, nIter=nIter)
#
# pred_change_df, pred_change_shuff = ta.get_pred_change(preprodf, shuffle, subject_col=subject_col,
#                                                        group_cols=group_cols, trial_col=trial_col)
# ta.plot_predicted_change(pred_change_df, pred_change_shuff, group_cols, value_col=value_col, trial_col=trial_col,
#                          save_dir=PA.save_dir, flag=flag, textsize=textsize, nIter=nIter)
#
# ta.plot_session_differences(pred_change_df, pred_change_shuff, subject_col, group_cols,
#                             trial_col=trial_col, value_col=value_col, stat_col='pred. change', save_dir=PA.save_dir,
#                             flag=flag, textsize=textsize, nIter=nIter)

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
all_units, held_df = PA.get_unit_info(overwrite=True)

# run scf.get_avg_intertrial_distances on each rec_dir non parallel
final_dfs = []
for rec_dir in rec_dirs:
    final_dfs.append(scf.get_avg_intertrial_distances(rec_dir, bsln_sub=False))
# Concatenate all resulting data frames into one
final_df = pd.concat(final_dfs, ignore_index=True)

# merge in rec_info into final_df
final_df = pd.merge(final_df, rec_info, on='rec_dir')
final_df['session'] = final_df['rec_num']

subject_col = ['exp_name']
group_cols = ['exp_group', 'session', 'taste']
trial_col = 'taste_trial'
value_col = 'euclidean_distance'
flag = 'intertrial_euc_dist'

preprodf, shuffle = ta.preprocess_nonlinear_regression(final_df, subject_col, group_cols, trial_col, value_col,
                                                       nIter=10000, save_dir=save_dir, overwrite=True, flag=flag)
nIter = 10000
textsize = 20
parallel = True

preprodf = preprodf.loc[preprodf['exp_name'] != 'DS33']
shuffle = shuffle.loc[shuffle['exp_name'] != 'DS33']
#preprodf = preprodf.loc[preprodf['exp_name'] != 'DS41']
#shuffle = shuffle.loc[shuffle['exp_name'] != 'DS41']

ta.plotting_pipeline(preprodf, shuffle, trial_col, value_col, group_cols, subject_col, nIter=nIter, save_dir=save_dir, flag=flag)

folder = 'intertrial_distances_bsln_sub'
save_dir = proj_save_dir + os.sep + folder
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print('Created directory:', save_dir)

# run scf.get_avg_intertrial_distances on each rec_dir non parallel
final_dfs = []
for rec_dir in rec_dirs:
    final_dfs.append(scf.get_avg_intertrial_distances(rec_dir, bsln_sub=True))
# Concatenate all resulting data frames into one
final_df = pd.concat(final_dfs, ignore_index=True)


# merge in rec_info into final_df
final_df = pd.merge(final_df, rec_info, on='rec_dir')
final_df['session'] = final_df['rec_num']

subject_col = ['exp_name']
group_cols = ['exp_group', 'session', 'taste']
trial_col = 'taste_trial'
value_col = 'euclidean_distance'
flag = 'intertrial_euc_dist_bsln_sub'

preprodf, shuffle = ta.preprocess_nonlinear_regression(final_df, subject_col, group_cols, trial_col, value_col,
                                                       nIter=10000, save_dir=save_dir, overwrite=False, flag=flag)
nIter = 10000
textsize = 20

preprodf = preprodf.loc[preprodf['exp_name'] != 'DS33']
shuffle = shuffle.loc[shuffle['exp_name'] != 'DS33']
preprodf = preprodf.loc[preprodf['exp_name'] != 'DS41']
shuffle = shuffle.loc[shuffle['exp_name'] != 'DS41']

ta.plotting_pipeline(preprodf, shuffle, trial_col, value_col, group_cols, subject_col, nIter=nIter, save_dir=save_dir, flag=flag)










ta.plot_fits(preprodf, trial_col='taste_trial', dat_col='euclidean_distance', save_dir=save_dir, flag=flag,
             time_col='session')



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

