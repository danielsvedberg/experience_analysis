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
import matplotlib
matplotlib.use('Agg')


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
    df['Z(distance)'] = df['euclidean_distance'].transform(lambda x: (x - x.mean()) / x.std())
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
value_col = 'Z(distance)'
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

#ta.plot_fits(preprodf, trial_col='taste_trial', dat_col='euclidean_distance', save_dir=save_dir,time_col='session')

preproD13 = preprodf.loc[preprodf['session'] != 2]
shufflD13 = shuffle.loc[shuffle['session'] != 2]
ta.plotting_pipeline(preproD13, shufflD13, trial_col, value_col, gr_cols, sub_cols, nIter=nIter, save_dir=save_dir,
                     flag = 'D_1_3', xticklabs=True)


#loop and take away trials 0 through i
ppdfs = []
sdfs = []
for i in range(0, 25):
    folder = f'dist_to_avg_stereotypy_D_minus_0-{i}'
    save_dir = proj_save_dir + os.sep + folder
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print('Created directory:', save_dir)
    fdf = final_df.loc[final_df['session_trial'] > i]
    ppdf, sdsfs = ta.preprocess_nonlinear_regression(fdf, sub_cols, gr_cols, trial_col, value_col,
                                                           nIter=1000, shuffle=True, save_dir=save_dir, overwrite=False)

    sdsfs = sdsfs[['session', 'exp_name', 'taste', 'r2', 'exp_group', 'iternum']]
    sdsfs = sdsfs.groupby(['session', 'iternum', 'taste', 'exp_group']).mean().reset_index()
    sdsfs['trials_removed'] = i

    ppdf['trials_removed'] = i
    ppdfs.append(ppdf)
    sdfs.append(sdsfs)

lesion_preprodf = pd.concat(ppdfs, ignore_index=True)
lesion_shuffledf = pd.concat(sdfs, ignore_index=True)
#save lesion_preprodf to csv
#save in save_dir
lesion_preprodf.to_csv(save_dir + os.sep + 'lesion_preprodf.csv', index=False)
lesion_shuffledf.to_csv(save_dir + os.sep + 'lesion_shuffledf.csv', index=False)

import seaborn as sns
import matplotlib.pyplot as plt
#plot boxplot of lesion_preprodf with x as trials_removed and y as mean_fit
avglesion = lesion_preprodf[lesion_preprodf['session'] == 1]
#avglesion = lesion_preprodf.copy()
avglesion = avglesion[avglesion['exp_group'] == 'naive']
avglesion = avglesion.groupby(['exp_name', 'trials_removed', 'session', 'exp_group']).mean().reset_index()

avgshuffle = lesion_shuffledf[lesion_shuffledf['session'] == 1]
#avgshuffle = lesion_shuffledf.copy()
avgshuffle = avgshuffle[avgshuffle['exp_group'] == 'naive']

#remove trials removed greater than 27
avglesion = avglesion[avglesion['trials_removed'] <= 26]
avgshuffle = avgshuffle[avgshuffle['trials_removed'] <= 26]

#get the 97.5 percentile of r2 for each trials_removed in avgshuffle
avglesionupper = avglesion.groupby('trials_removed')['r2'].quantile(0.975).reset_index()
avglesionlower = avglesion.groupby('trials_removed')['r2'].quantile(0.025).reset_index()
avglesion = avglesion.groupby(['trials_removed']).mean().reset_index()

#calculate new whis values for boxplot as 2.5 and 97.5 percentiles bonferroni corrected for number of trials_removed
n_trials_removed = 26
whis_lower = 0.025 / n_trials_removed
whis_upper = 1 - (0.025 / n_trials_removed)

whis = avgshuffle.groupby('trials_removed')['r2'].quantile([whis_lower, whis_upper]).unstack().reset_index()
whis = whis.rename(columns={whis_lower: 'whis_lower', whis_upper: 'whis_upper'})
whis['whis_upper'] = whis['whis_upper'] - whis['whis_lower']

#for each trials removed, get the quantile of avglesion r2 relative to avgshuffle r2
stars = []
pvals = []
for i in range(len(avglesion)):
    tr = avglesion.loc[i, 'trials_removed']
    r2 = avglesion.loc[i, 'r2']
    shuffle_r2 = avgshuffle.loc[avgshuffle['trials_removed'] == tr, 'r2']
    quantile = (shuffle_r2 < r2).mean()
    pval = min(quantile, 1 - quantile) * 2 * n_trials_removed  # two-tailed p-value
    print(f'Trials removed: {tr}, R²: {r2:.4f}, Quantile: {quantile:.4f}, p-value: {pval:.4f}')
    pvals.append(pval)
    stars.append(ta.get_pval_stars(pval))
pval_table = pd.DataFrame({'trials_removed': avglesion['trials_removed'], 'r2_value':avglesion['r2'], 'p_value':pvals, 'stars':stars})
#save pval_table to csv
pval_table.to_csv(save_dir + os.sep + 'lesion_pval_table.csv', index=False)

import plot_size_utils as psu
#text size is 8pt
matplotlib.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(figsize=(10, 6))
#plt.bar(avglesion['trials_removed'], avglesion['ones'], bottom=avglesion['r2'], color='tab:blue', alpha=0.5,
#        label='Mean R² (Lesioned Data)', linewidth=1.5, edgecolor='tab:blue')
plt.errorbar(avglesion['trials_removed'], avglesion['r2'], yerr=[avglesion['r2'] - avglesionlower['r2'],
                                                                   avglesionupper['r2'] - avglesion['r2']],
             fmt='o', color='tab:blue', capsize=2, label='95% CI (Lesioned Data)', markersize=2, elinewidth=1)
plt.bar(whis['trials_removed'], whis['whis_upper'], bottom=whis['whis_lower'], color='gray', alpha=0.5)
#plot the pval stars above the error bars
for i in range(len(pval_table)):
    trials_removed = pval_table.loc[i, 'trials_removed']
    upper = avglesionupper.loc[avglesionupper['trials_removed'] == trials_removed, 'r2'].values[0]
    star = pval_table.loc[i, 'stars']
    if star != '':
        plt.text(trials_removed+0.35, upper + 0.1, star, fontsize=8, ha='center', va = 'center', color='black', rotation=90)
#get the max of avglesionupper r2
max_r2 = avglesionupper['r2'].max()
plt.xticks([0,5,10,15,20,25])
plt.ylim(0, max_r2 + 0.15)
plt.ylabel('Mean R²', fontsize=8)
plt.xlabel('Trials Removed', fontsize=8)
#set x tick label size to 8
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
psu.adjust_figure_for_panel_size_auto(fig, panel_width=2.5, panel_height=1)

#save plot to save_dir
plt.savefig(save_dir + os.sep + 'lesion_preprodf_boxplot.png')
#save as svg
plt.savefig(save_dir + os.sep + 'lesion_preprodf_boxplot.svg')
plt.close()
'''

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
'''