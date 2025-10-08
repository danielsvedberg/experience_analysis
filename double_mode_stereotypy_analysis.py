import analysis as ana
import hmm_analysis as hmma
import blechpy
import new_plotting as nplt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import plot_size_utils as psu

import stereotypy_clustering_functions
import trialwise_analysis as ta
import scipy.stats as stats
import time
from joblib import Parallel, delayed
from blechpy.dio import h5io
#

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
rec_info = proj.rec_info.copy()  # get the rec_info table
rec_info = rec_info.rename(columns={'rec_num': 'session'})
rec_dirs = rec_info['rec_dir']
PA = ana.ProjectAnalysis(proj)

#all_units, held_df = PA.get_unit_info(overwrite=False)

save_dir = PA.save_dir + '/double_mode_stereotypy_analysis'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print('save dir' + save_dir + ' created')


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

def euc_dist_trials(rates):
    avg_firing_rate = np.mean(rates, axis=1)  # Neurons x Bins
    euc_dist = np.linalg.norm(rates - avg_firing_rate[:, np.newaxis, :], axis=0)
    euc_dist = euc_dist.mean(axis=1)
    return euc_dist

def process_split(rate_array, rec_dir, split=10):
    df_list = []

    for din, rate in rate_array.items():
        if din != 'dig_in_4':
            pre_rate = rate[:, :split, :]
            post_rate = rate[:, split:, :]

            pre_euc_dist = euc_dist_trials(pre_rate)

            post_euc_dist = euc_dist_trials(post_rate)

            #bind pre and post euc_dist_mat
            if len(pre_euc_dist) == 0:
                euc_dist = post_euc_dist
            elif len(post_euc_dist) == 0:
                euc_dist = pre_euc_dist
            else: #concantenate pre_euc_dist_mat and post_euc_dist_mat along axis 1
                euc_dist = np.concatenate((pre_euc_dist, post_euc_dist))

            df = pd.DataFrame({
                'euclidean_distance': euc_dist,
                'rec_dir': rec_dir,
                'channel': int(din[-1]),  # get the din number from string din
                'taste_trial': np.arange(rate.shape[1]),
                'split': split
            })
            df['splitside'] = df['taste_trial'].apply(lambda x: 'pre' if x < split else 'post')
            min_pre_trial = df.loc[df['splitside'] == 'pre', 'taste_trial'].min() + 1
            max_pre_trial = df.loc[df['splitside'] == 'pre', 'taste_trial'].max() + 1
            minlabel = str(min_pre_trial) + '-' + str(max_pre_trial)
            min_post_trial = df.loc[df['splitside'] == 'post', 'taste_trial'].min() + 1
            max_post_trial = df.loc[df['splitside'] == 'post', 'taste_trial'].max() + 1
            maxlabel = str(min_post_trial) + '-' + str(max_post_trial)
            df['trial_group'] = df['splitside'].apply(lambda x: minlabel if x == 'pre' else maxlabel)
            df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    df = df.reset_index(drop=True)
    return df

def process_rec_dir(rec_dir, shuffle=False):
    splits = np.arange(30)
    #get the 0th index, and indices 2 to -1
    splits = np.concatenate((splits[:1], splits[2:-1]))

    ts = 100
    te = 3000
    #time_array = time_array[i_e]
    time_array, rate_array = h5io.get_psths(rec_dir)
    #get indexes of time_array where time_array is greater than ts and less than te
    i_e = (time_array >= ts) & (time_array <= te)
    for din in rate_array.keys():
        #if din is not the dig_in_4 channel, shuffle the trials
        if din != 'dig_in_4':
            rate_array[din] = rate_array[din][:, :, i_e]
            if shuffle:
                n_trials = rate_array[din].shape[1]
                rate_array[din] = rate_array[din][:, np.random.choice(n_trials, n_trials, replace=True), :]

    dfs = [process_split(rate_array, rec_dir, split) for split in splits]
    dfs = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
    return dfs

#make the shuffle
def iter_shuffle(rec_dirs, niter=100):
    def sing_iter(iternum):
        res_list = []
        print('Processing iteration: ' + str(iternum))
        for rd in rec_dirs:
            #print('Processing rec_dir: ' + rd)
            res = process_rec_dir(rd, shuffle=True)
            res['iternum'] = iternum
            res_list.append(res)
        res = pd.concat(res_list, ignore_index=True).reset_index(drop=True)
        return res

    dfs = Parallel(n_jobs=-1)(delayed(sing_iter)(iternum) for iternum in range(niter))
    dfs = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
    return dfs

# Parallelize processing of each rec_dir
num_cores = -1  # Use all available cores
#final_dfs = Parallel(n_jobs=num_cores)(delayed(process_rec_dir)(rec_dir) for rec_dir in rec_dirs)
# Concatenate all resulting data frames into one
#final_df = pd.concat(final_dfs, ignore_index=True)
#save to pickle
#final_df.to_pickle(save_dir + '/double_mode_stereotypy_analysis.pkl')

#shuff_dfs = iter_shuffle(rec_dirs, niter=2000)
#shuff_dfs.to_pickle(save_dir + '/double_mode_euc_dist_stereotypy_shuffle_1.pkl')

#shuff_dfs = iter_shuffle(rec_dirs, niter=2000)
#shuff_dfs.to_pickle(save_dir + '/double_mode_euc_dist_stereotypy_shuffle_2.pkl')
'''shuff_dfs = iter_shuffle(rec_dirs, niter=2000)
shuff_dfs.to_pickle(save_dir + '/double_mode_euc_dist_stereotypy_shuffle_3.pkl')
#remove shuff dfs from memory
del shuff_dfs
shuff_dfs = iter_shuffle(rec_dirs, niter=2000)
shuff_dfs.to_pickle(save_dir + '/double_mode_euc_dist_stereotypy_shuffle_4.pkl')
del shuff_dfs
shuff_dfs = iter_shuffle(rec_dirs, niter=2000)
shuff_dfs.to_pickle(save_dir + '/double_mode_euc_dist_stereotypy_shuffle_5.pkl')'''


#l

######################################################################################################################
rec_info = proj.rec_info.copy()
rec_info = rec_info[['exp_name', 'exp_group', 'rec_num', 'rec_dir']]
#rename rec num to session
rec_info = rec_info.rename(columns={'rec_num': 'session'})
#preprocess the shuffle ################################################################################################

def preprocess_shuffle(filepath):
    shuff_df = pd.read_pickle(filepath)
    #get the max iternum, loop through subsets of shuff_df in chunks of 1000 iterations
    shuff_df['iternum'] = shuff_df['iternum'].astype(int)
    max_iternum = shuff_df['iternum'].max()
    n_loops = max_iternum // 500 + 1

    final_shuff_df_list = []
    best_split_df_list = []
    iteravg_shuff_df_list = []

    for i in range(n_loops):
        start = i * 500
        end = (i + 1) * 500
        subset = shuff_df.loc[(shuff_df['iternum'] >= start) & (shuff_df['iternum'] < end)]
        subset = pd.merge(subset, rec_info, on='rec_dir')
        max_trials = subset.groupby(['rec_dir', 'channel'])['taste_trial'].max().reset_index()
        max_trials = max_trials.rename(columns={'taste_trial': 'max_trial'})
        subset = pd.merge(subset, max_trials, on=['rec_dir', 'channel'])
        subset = subset.loc[subset['max_trial'] - subset['split'] > 1]
        subset = subset.loc[subset['split'] != 2]
        subset = subset.loc[subset['split'] != 1]
        subset['Z(Euc dist)'] = subset.groupby(['rec_dir', 'channel', 'iternum'])['euclidean_distance'].transform(
            lambda x: (x - x.mean()) / x.std())
        iteravg = subset.groupby(['exp_name', 'channel', 'session', 'exp_group', 'taste_trial', 'split']).mean().reset_index()
        subset = subset.merge(iteravg, on=['exp_name', 'channel', 'session', 'exp_group', 'taste_trial', 'split'], suffixes=('', '_shuff'))
        subset['euc_dist_shuffnorm'] = subset['euclidean_distance'] - subset['euclidean_distance_shuff']
        subset['Z(euc_dist_shuffnorm)'] = subset.groupby(['rec_dir', 'channel', 'iternum'])['euc_dist_shuffnorm'].transform(lambda x: (x - x.mean()) / x.std())

        sub_unsplit = subset.loc[subset['split'] == 0]
        subsplitzero = sub_unsplit.groupby(['exp_name', 'channel', 'session', 'exp_group', 'iternum']).mean().reset_index()
        subset = pd.merge(subset, subsplitzero, on=['exp_name', 'channel', 'session', 'exp_group', 'iternum'], suffixes=('', '_splitzero'))
        subset['euc_dist_zeronorm'] = subset['euclidean_distance'] - subset['euclidean_distance_splitzero']
        subset['Z(euc_dist_zeronorm)'] = subset.groupby(['rec_dir', 'channel', 'iternum'])['euc_dist_zeronorm'].transform(lambda x: (x - x.mean()) / x.std())

        # for each grouping of rec_dir, channel, session, exp_group, split, and iternum, get the rows where split is lowest
        trl_avg = subset.groupby(  # first, get the average euc dist across trials
            ['rec_dir', 'session', 'channel', 'exp_group', 'split', 'iternum']).mean().reset_index()
        trl_avg = trl_avg.loc[trl_avg['split'] != 0]  # take out the unsplit rows
        trl_channel_avg = trl_avg.groupby(['rec_dir', 'session', 'exp_group', 'split',
                                           'iternum']).mean().reset_index()  # get the average euc dist across channels for more compressed version
        best_split = trl_avg.loc[trl_avg.groupby(['rec_dir', 'session', 'channel', 'exp_group', 'iternum'])[
            'euclidean_distance'].idxmin()]  # get rows with split with lowest euc distance for each grouping of rec_dir, session, channel, exp_group, and iternum

        final_shuff_df_list.append(trl_channel_avg)
        best_split_df_list.append(best_split)
        iteravg_shuff_df_list.append(iteravg)

    final_shuff = pd.concat(final_shuff_df_list, ignore_index=True).reset_index(drop=True)
    best_split_shuff = pd.concat(best_split_df_list, ignore_index=True).reset_index(drop=True)
    iteravg_shuff = pd.concat(iteravg_shuff_df_list, ignore_index=True).reset_index(drop=True)
    iteravg_shuff = iteravg_shuff.groupby(['exp_name', 'channel', 'session', 'exp_group', 'taste_trial', 'split']).mean().reset_index()

    return final_shuff, best_split_shuff, iteravg_shuff



path_root = save_dir + '/double_mode_euc_dist_stereotypy_shuffle_'
final_shuff_df = []
best_split_shuff_df = []
iteravg_shuff_df = []
for i in range(1,2):
    filepath = path_root + str(i) + '.pkl'
    fsd, bsd, iad = preprocess_shuffle(filepath)
    final_shuff_df.append(fsd)
    best_split_shuff_df.append(bsd)
    iteravg_shuff_df.append(iad)

final_shuff_df = pd.concat(final_shuff_df, ignore_index=True).reset_index(drop=True)
best_split_shuff_df = pd.concat(best_split_shuff_df, ignore_index=True).reset_index(drop=True)
iteravg_shuff_df = pd.concat(iteravg_shuff_df, ignore_index=True).reset_index(drop=True)
iteravg_shuff_df = iteravg_shuff_df.groupby(['exp_name', 'channel', 'session', 'exp_group', 'taste_trial', 'split']).mean().reset_index()
#drop iternum column from iteravg_shuff_df
iteravg_shuff_df = iteravg_shuff_df.drop(columns=['iternum'])
#save the final shuff df and best split shuff df to pickle
final_shuff_df.to_pickle(save_dir + '/double_mode_euc_dist_stereotypy_shuffle.pkl')
best_split_shuff_df.to_pickle(save_dir + '/double_mode_euc_dist_stereotypy_best_split_shuffle.pkl')
iteravg_shuff_df.to_pickle(save_dir + '/double_mode_euc_dist_stereotypy_iteravg_shuffle.pkl')



#load the final shuff df and best split shuff df
final_shuff_df = pd.read_pickle(save_dir + '/double_mode_euc_dist_stereotypy_shuffle.pkl')
best_split_shuff_df = pd.read_pickle(save_dir + '/double_mode_euc_dist_stereotypy_best_split_shuffle.pkl')
iteravg_shuff_df = pd.read_pickle(save_dir + '/double_mode_euc_dist_stereotypy_iteravg_shuffle.pkl')
#replace any instance of capital E in Euc dist column names with lowercase e
iteravg_shuff_df = iteravg_shuff_df.rename(columns={'Z(Euc dist)': 'Z(euc dist)'})
best_split_shuff_df = best_split_shuff_df.rename(columns={'Z(Euc dist)': 'Z(euc dist)'})
final_shuff_df = final_shuff_df.rename(columns={'Z(Euc dist)': 'Z(euc dist)'})

#merge in rec_info to final_shuff_df and best_split_shuff_df
final_shuff_df = pd.merge(final_shuff_df, rec_info, on='rec_dir')


#get final df from pickle and preprocess###############################################################################

final_df = pd.read_pickle(save_dir + '/double_mode_stereotypy_analysis.pkl')
final_df = pd.merge(final_df, rec_info, on='rec_dir')
#for each combination of rec_dir and channel, find the maximum trial. The split cannot be greater than the maximum trial - 2
max_trials = final_df.groupby(['rec_dir', 'channel'])['taste_trial'].max().reset_index()
max_trials = max_trials.rename(columns={'taste_trial': 'max_trial'})
final_df = pd.merge(final_df, max_trials, on=['rec_dir', 'channel'])
final_df = final_df.loc[(final_df['max_trial'] - final_df['split']) > 1] #ensure there are at least 2 trials in each split side on the top end
final_df = final_df.loc[final_df['split'] != 2] #ensure there are at least 2 trials in each split side on the low end
final_df = final_df.loc[final_df['split'] != 1]
final_df['Z(euc dist)'] = final_df.groupby(['rec_dir', 'channel'])['euclidean_distance'].transform(lambda x: (x - x.mean()) / x.std())

final_df = pd.merge(final_df, iteravg_shuff_df, on=['exp_name', 'channel', 'session', 'exp_group', 'split', 'taste_trial'], suffixes=('', '_shuff'))

final_df['euc_dist_shuffnorm'] = final_df['euclidean_distance']-final_df['euclidean_distance_shuff']
final_df['Z(Euc. dist - shuff)'] = final_df.groupby(['rec_dir', 'channel'])['euc_dist_shuffnorm'].transform(lambda x: (x - x.mean()) / x.std())

unsplit = final_df.loc[final_df['split'] == 0]
splitzero = unsplit.groupby(['exp_name', 'channel', 'session', 'exp_group']).mean().reset_index()
final_df = final_df.loc[final_df['split'] != 0]
final_df = pd.merge(final_df, splitzero, on=['exp_name', 'channel', 'session', 'exp_group'], suffixes=('', '_splitzero'))
#final_df['euc_dist_zeronorm'] = final_df['euclidean_distance']-final_df['euclidean_distance_splitzero']
#final_df['Z(euc_dist_zeronorm)'] = final_df.groupby(['rec_dir', 'channel'])['euc_dist_zeronorm'].transform(lambda x: (x - x.mean()) / x.std())
#final_df['euc_dist_splitzero-shuff'] = final_df['euclidean_distance_splitzero']-final_df['euclidean_distance_shuff']

#average the euclidean distance for each animal, taste, split, and session
avg_df = final_df.groupby(['exp_name', 'split', 'session', 'exp_group']).mean().reset_index()
avg_shuff_df = final_shuff_df


best_split_df = avg_df.loc[avg_df.groupby(['exp_name', 'session', 'exp_group', 'channel'])['euclidean_distance'].idxmin()]
best_split_df['Z(Euc. dist - shuff)'] = best_split_df.groupby(['exp_name', 'session', 'channel'])['euc_dist_shuffnorm'].transform(lambda x: (x - x.mean()) / x.std())
best_shuff = best_split_shuff_df

def get_sig_stars(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return ''


sbcolors = sns.color_palette()
fontsize = 8
axticksize = 8
ci = (1-(0.05/30)/2)*100

fig, axs = plt.subplots(1, 2, figsize=(5,3), sharey=True, sharex=True)
for i, session in enumerate([1,3]):
    ax = axs[i]
    for exp_group in ['naive']:
        df = avg_df.loc[(avg_df['exp_group'] == exp_group) & (avg_df['session'] == session)]
        sz = splitzero.loc[(splitzero['exp_group'] == exp_group) & (splitzero['session'] == session)].mean()
        shuff_df = avg_shuff_df.loc[(avg_shuff_df['exp_group'] == exp_group) & (avg_shuff_df['session'] == session)]
        #make a seaborn lineplot
        sns.lineplot(x='split', y='euclidean_distance', data=df, ax=ax, ci=None)
        sns.lineplot(x='split', y='euclidean_distance', data=shuff_df, ax=ax, color='black', alpha=0.1, ci = ci)
        #plot a blue dashed horizontal line at y = splitzero, with a 95% confidence interval
        sns.lineplot(x=[3, 28], y=sz['euclidean_distance'], ax=ax, linestyle='--', color = sbcolors[0])
        ax.set_title('Session ' + str(session), fontsize=fontsize)
        ax.set_xlabel('Split-trial', fontsize=fontsize)
        ax.set_ylabel('Euc. dist. (AU)', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=axticksize)
plt.tight_layout()
plt.show()
plt.savefig(save_dir + '/double_mode_dist_vs_split.png')
plt.savefig(save_dir + '/double_mode_dist_vs_split.svg')

sbcolors = sns.color_palette()
fontsize = 14
axticksize = 14
ci = (1-(0.05/24)/2)*100
fig, axs = plt.subplots(1, 2, figsize=(5,3), sharey=True, sharex=True)
for i, session in enumerate([1,3]):
    ax = axs[i]
    for exp_group in ['naive']:
        df = final_df.loc[(final_df['exp_group'] == exp_group) & (final_df['session'] == session)]
        df = df.groupby(['rec_dir', 'channel', 'split']).mean().reset_index()
        #sz = splitzero.loc[(splitzero['exp_group'] == exp_group) & (splitzero['session'] == session)]
        shuff_df = final_shuff_df.loc[(final_shuff_df['exp_group'] == exp_group) & (final_shuff_df['session'] == session)]
        shuff_df = shuff_df.groupby(['rec_dir', 'channel', 'split']).mean().reset_index()
        #make a seaborn lineplot
        sns.lineplot(x='split', y='Z(Euc dist)', data=df, ax=ax)
        sns.lineplot(x='split', y='Z(Euc dist)', data=shuff_df, ax=ax, color='black', alpha=0.5, ci = ci)
        #sns.lineplot(x='split', y='euc_dist_splitzero-shuff', data=df, ax=ax, linestyle='--', color=sbcolors[0])
        #sns.lineplot(x='split', y='euc_dist_shuffnorm', data=df, ax=ax)
        #sns.lineplot(x='split', y='euc_dist_shuffnorm', data=shuff_df, ax=ax, color='black', alpha=0.1, ci = ci)
        ax.set_title('Session ' + str(session), fontsize=fontsize)
        ax.set_xlabel('Split-trial', fontsize=fontsize)
        ax.set_ylabel('Z(Euc dist)', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=axticksize)
        #set x ticks to 5, 10, 15, 20, 25
        ax.set_xticks(np.arange(5, 30, 5))
        #set y ticks to include zero
        ymin, ymax = ax.get_ylim()
        ymin = min(ymin, 0)
        ymax = max(ymax, 0)
        ax.set_ylim(ymin, ymax)
plt.tight_layout()
plt.show()
plt.savefig(save_dir + '/double_mode_dist_vs_split_Z.png')
plt.savefig(save_dir + '/double_mode_dist_vs_split_Z.svg')


sbcolors = sns.color_palette()
fontsize = 14
axticksize = 14
ci = (1-(0.05/24)/2)*100
fig, axs = plt.subplots(1, 2, figsize=(5,3), sharey=True, sharex=True)
for i, session in enumerate([1,3]):
    ax = axs[i]
    for exp_group in ['naive']:
        df = avg_df.loc[(avg_df['exp_group'] == exp_group) & (avg_df['session'] == session)]
        shuff_df = avg_shuff_df.loc[(avg_shuff_df['exp_group'] == exp_group) & (avg_shuff_df['session'] == session)]
        #make a seaborn lineplot
        sns.lineplot(x='split', y='Z(Euc. dist - shuff)', data=df, ax=ax)
        sns.lineplot(x='split', y='Z(euc_dist_shuffnorm)', data=shuff_df, ax=ax, color='black', alpha=0.1, ci = ci)
        #plot a blue dashed horizontal line at y = splitzero, with a 95% confidence interval
        sns.lineplot(x='split', y='Z(Euc. dist - shuff)_splitzero',data=df,  ax=ax, linestyle='--', color = sbcolors[0])
        ax.set_title('Session ' + str(session), fontsize=fontsize)
        ax.set_xlabel('Split-trial', fontsize=fontsize)
        ax.set_ylabel('Z(Euc dist - shuff)', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=axticksize)
        ax.set_xticks(np.arange(5, 30, 5))
        #set y ticks to include zero
        ymin, ymax = ax.get_ylim()
        ymin = min(ymin, 0)
        ymax = max(ymax, 0)
        ax.set_ylim(ymin, ymax)
plt.tight_layout()
plt.show()
plt.savefig(save_dir + '/double_mode_dist_vs_split_shuff_norm.png')
plt.savefig(save_dir + '/double_mode_dist_vs_split_shuff_norm.svg')

sbcolors = sns.color_palette()
fontsize = 14
axticksize = 14
ci = (1-(0.05/24)/2)*100
fig, axs = plt.subplots(1, 2, figsize=(5,3), sharey=True, sharex=True)
for i, session in enumerate([1,3]):
    ax = axs[i]
    for exp_group in ['naive']:
        df = avg_df.loc[(avg_df['exp_group'] == exp_group) & (avg_df['session'] == session)]
        sz = splitzero.loc[(splitzero['exp_group'] == exp_group) & (splitzero['session'] == session)]
        shuff_df = avg_shuff_df.loc[(avg_shuff_df['exp_group'] == exp_group) & (avg_shuff_df['session'] == session)]
        #make a seaborn lineplot
        sns.lineplot(x='split', y='Z(euc_dist) - z(euc_dist_shuff)', data=df, ax=ax)
        #sns.lineplot(x='split', y='Z(euc_dist_shuffnorm)', data=shuff_df, ax=ax, color='black', alpha=0.1, ci = ci)
        #plot a blue dashed horizontal line at y = splitzero, with a 95% confidence interval
        #sns.lineplot(x='split', y='Z(euc_dist_shuffnorm)_splitzero',data=df,  ax=ax, linestyle='--', color = sbcolors[0])
        ax.set_title('Session ' + str(session), fontsize=fontsize)
        ax.set_xlabel('Split-trial', fontsize=fontsize)
        ax.set_ylabel('Z(Euc dist - shuff)', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=axticksize)
        ax.set_xticks(np.arange(5, 30, 5))
        #set y ticks to include zero
        ymin, ymax = ax.get_ylim()
        ymin = min(ymin, 0)
        ymax = max(ymax, 0)
        ax.set_ylim(ymin, ymax)
plt.tight_layout()
plt.show()
plt.savefig(save_dir + '/double_mode_dist_vs_split_Z_shuff_norm.png')
plt.savefig(save_dir + '/double_mode_dist_vs_split_Z_shuff_norm.svg')




#plot the ecdf of the best split trials for each session, with a 95% confidence interval from the shuffle
fig, axs = plt.subplots(1, 2, figsize=(5,3), sharey=True, sharex=True)
for i, session in enumerate([1, 3]):
    for exp_group in ['naive']:
        df = best_split_df.loc[(best_split_df['exp_group'] == exp_group) &
                               (best_split_df['session']   == session)]
        df['split'] = df['split'] - 0.5
        shuffdf = best_shuff.loc[(best_shuff['exp_group'] == exp_group) &
                                 (best_shuff['session']   == session)]
        shuffdf['split'] = shuffdf['split'] - 0.5


        # ---------- ECDFs ---------------------------------------------------
        sns.ecdfplot(data=shuffdf, x='split', stat='proportion',
                     color='black', linewidth=2, ax=ax, label='shuffled', zorder=2)
        sns.ecdfplot(data=df,      x='split', stat='proportion',
                     color=sbcolors[0], linewidth=2, ax=ax, label='real', zorder=3)

        # ---------- dashed blue median line ---------------------------------
        med_split = df['split'].median()
        #mode_split = df['split'].mode()[0]
        ecdf_at_med = (df['split'] <= med_split).mean()   # height of ECDF at median
        #ecdf_at_mode = (df['split'] <= mode_split).mean()   # height of ECDF at mode
        ax.vlines(med_split, 0, ecdf_at_med, color=sbcolors[0],
                  linestyle='--', linewidth=1.5, zorder=4)

        # ---------- cosmetics ----------------------------------------------
        ax.set_title(f'Session {session}', fontsize=fontsize)
        ax.set_xlabel('best split-trial', fontsize=fontsize)
        ax.set_ylabel('fraction',         fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=axticksize)
        ax.set_xlim(2, 27)
        #see x ticks at 5, 10, 15, 20, 25
        ax.set_xticks(np.arange(5, 30, 5))

plt.tight_layout()

plt.savefig(save_dir + '/double_mode_best_split_cdf.png')
plt.savefig(save_dir + '/double_mode_best_split_cdf.svg')

#now make a histogram of the best split for each session
fig, axs = plt.subplots(1, 2, figsize=(5,3), sharey=True, sharex=True)
for i, session in enumerate([1,3]):
    for exp_group in ['naive']:
        df = best_split_df.loc[(best_split_df['exp_group'] == exp_group) & (best_split_df['session'] == session)]
        df['split'] = df['split'] - 0.5
        shuffdf = best_shuff.loc[(best_shuff['exp_group'] == exp_group) & (best_shuff['session'] == session)]
        shuffdf['split'] = shuffdf['split'] - 0.5
        #join the two dataframes #
        #df['type'] = 'real'
        #shuffdf['type'] = 'shuffled'
        # df = pd.concat([df, shuffdf], ignore_index=True)
        ax = axs[i]
        sns.histplot(ax=ax, data=df, x='split', stat='probability', alpha=0.5, color=sbcolors[0], binwidth=2, edgecolor=None)
        sns.histplot(shuffdf['split'], ax=ax, stat='probability', color='black', alpha=0.25, binwidth=2, edgecolor=None)
        #plot a vertical line at the median of the real data
        med_split = df['split'].median()
        ax.axvline(med_split, color=sbcolors[0], linestyle='--', linewidth=2, label='median')
    ax.set_title('Session ' + str(session), fontsize=fontsize)
    ax.set_xlabel('best split-trial', fontsize=fontsize)
    ax.set_ylabel('fraction', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=axticksize)
    ax.set_xticks(np.arange(5, 30, 5))
    ax.set_xlim(2, 27)
plt.tight_layout()
plt.savefig(save_dir + '/double_mode_best_split_hist.png')
plt.savefig(save_dir + '/double_mode_best_split_hist.svg')


unsplit['splitside'] = 'single\ntemplate'
unsplit['template'] = 'single\ntemplate'
#for final_df, replace all occurences of '1-30' in 'trial_group' with 'combined'
final_df['trial_group'] = final_df['trial_group'].replace('1-30', 'combined')
#refactor all entries in splitside from 'post' to 'post\nsplit' and 'pre' to 'pre\nsplit'
final_df['splitside'] = final_df['splitside'].replace('post', 'post\nsplit')
final_df['splitside'] = final_df['splitside'].replace('pre', 'pre\nsplit')
final_df['template'] = 'two\ntemplate'

def plot_split_bars(final_df, best_split_df, save_dir, yvar):
    sbcolors = sns.color_palette()
    fontsize = 13
    axticksize = 13
    splitavos = []
    templavos = []
    import pingouin as pg
    fig1, axs1 = plt.subplots(1, 2, figsize=(5,3), sharey=True)
    fig2, axs2 = plt.subplots(1, 2, figsize=(5,3), sharey=True)
    for i, session in enumerate([1,3]):
        axtop = axs1[i]
        axbot = axs2[i]
        for exp_group in ['naive']:
            #get the df with the best split for each exp_name
            best_split = best_split_df.loc[(best_split_df['exp_group'] == exp_group) & (best_split_df['session'] == session)][['exp_name','split']]
            #get the rows of final_df where the exp_name and split match the best split
            df = final_df.loc[(final_df['exp_group'] == exp_group) & (final_df['session'] == session)]
            df = pd.merge(df, best_split, on=['exp_name', 'split'])

            df['AnTaste'] = df['exp_name'] + df['channel'].astype(str)
            #for splitavodf, get all rows where splitside == 'pre\nsplit' and 'post\nsplit'
            splitavodf = df.loc[(df['splitside'] == 'pre\nsplit') | (df['splitside'] == 'post\nsplit')]
            #take the average of euclidean distance for each exp_name, channel, and splitside
            splitavodf = splitavodf.groupby(['exp_name', 'channel', 'splitside', 'session', 'exp_group']).mean().reset_index()
            splitavodf['expXchannel'] = splitavodf['exp_name'] + splitavodf['channel'].astype(str)
    

            #take the average of euclidean distance for each exp_name, channel, and template
            templdf = df.copy()
            templdf['template'] = 'single\ntemplate'
            templdf['euc_dist_shuffnorm'] =  templdf['euclidean_distance_splitzero'] - templdf['euclidean_distance_shuff']
            #concatenate df with templdf
            templdf = pd.concat([templdf, df], ignore_index=True)
            templdf = templdf.groupby(['exp_name', 'channel', 'template', 'session', 'exp_group']).mean().reset_index()

            templdf['expXchannel'] = templdf['exp_name'] + templdf['channel'].astype(str)
            #perform a repeated measures anova
            splitavo = pg.rm_anova(data=splitavodf, dv=yvar, within=['splitside'], subject='expXchannel', detailed=True)
            split_pval = splitavo['p-unc'].values[0]
            splitavos.append(splitavo)
            templavo = pg.rm_anova(data=templdf, dv=yvar, within=['template'], subject='expXchannel', detailed=True)
            templ_pval = templavo['p-unc'].values[0]
            templavos.append(templavo)
            #append dfzero to df
            sns.swarmplot(x='splitside', y=yvar, data=splitavodf, ax=axtop,
                          order=['pre\nsplit', 'post\nsplit'], color=sbcolors[0], alpha = 0.5)
            sns.barplot(x='splitside', y=yvar, data=splitavodf, ax=axtop, order=['pre\nsplit', 'post\nsplit'], fill=False, edgecolor='black', capsize = 0.4)
            sns.swarmplot(x='template', y=yvar, data=templdf, ax=axbot,
                          order=['two\ntemplate', 'single\ntemplate'], color=sbcolors[0], alpha=0.5)
            sns.barplot(x='template', y=yvar, data=templdf, ax=axbot, order=['two\ntemplate', 'single\ntemplate'], fill=False, edgecolor='black', capsize = 0.4)

            #get the height of the upper whisker in the barplot
            max_split = splitavodf[yvar].max()
            max_templ = templdf[yvar].max()
            min_split = splitavodf[yvar].min()
            min_templ = templdf[yvar].min()
            #set the y axis top limit for both plots to 1
            #axtop.set_ylim(top=1)
            #axbot.set_ylim(top=1)
            if split_pval < 0.05: #then draw a line between 0 and 1, and put a star above the line
                height = max_split * 1.1
                axtop.plot([0, 1], [height, height], lw=2, color='black')
                stars = get_sig_stars(split_pval)
                axtop.text(0.5, height*0.8, stars, fontsize=fontsize, ha='center')
                #round the p value to 3 decimal places and print it
                split_pval = round(split_pval, 4)
                axtop.text(0.5, min_split, 'p = ' + str(split_pval), fontsize=fontsize, ha='center')
            if templ_pval < 0.05:
                height = max_templ * 1.1
                axbot.plot([0,1], [height, height], lw=2, color='black')
                stars = get_sig_stars(templ_pval)
                axbot.text(0.5, height*0.8, stars, fontsize=fontsize, ha='center')
                #print the p value
                templ_pval = round(templ_pval, 4)
                axbot.text(0.5, min_templ, 'p = ' + str(templ_pval), fontsize=fontsize, ha='center')
    
            #take away the y axis label
            if session == 1:
                axtop.set_ylabel('Euc. dist. - shuff', fontsize=fontsize)
                #make the y axis labels larger
                axbot.set_ylabel('Euc. dist. - shuff', fontsize=fontsize)
    
            else:
                axtop.set_ylabel('')
                axbot.set_ylabel('')
            axtop.set_xlabel('')
            axbot.set_xlabel('')
            axtop.tick_params(axis='both', which='major', labelsize=axticksize)
            axbot.tick_params(axis='both', which='major', labelsize=axticksize)
            axtop.set_title('Session ' + str(session), fontsize=fontsize)
            axbot.set_title('Session ' + str(session), fontsize=fontsize)
    
    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()
    #save the figures
    fig1.savefig(save_dir + '/double_mode_split_prevspost.png')
    fig1.savefig(save_dir + '/double_mode_split_prevspost.svg')
    fig2.savefig(save_dir + '/double_mode_split_latevssing.svg')
    fig2.savefig(save_dir + '/double_mode_split_latevssing.png')

plot_split_bars(final_df, best_split_df, save_dir, 'Z(euc_dist_shuffnorm)')
plot_split_bars(final_df, best_split_df, save_dir, 'euc_dist_shuffnorm')


#for unsplit, make a histogram of the euclidean distances
unsplit['Z(euc_dist)'] = unsplit.groupby(['rec_dir', 'channel'])['euclidean_distance'].transform(lambda x: (x - x.mean()) / x.std())
unsplit['taste_trial'] = unsplit['taste_trial'] + 1
#for each grouping of session, exp_group, and taste trial, perform a one sample t-test against 0 for Z(euc_dist)
unsplit_ttest = unsplit.groupby(['session', 'exp_group', 'taste_trial']).apply(lambda x: stats.ttest_1samp(x['Z(euc_dist)'], 0))
unsplit_ttest = pd.DataFrame(unsplit_ttest.tolist(), index=unsplit_ttest.index, columns=['tstat', 'pval'])
unsplit_ttest = unsplit_ttest.reset_index()
#bonferroni correct for 30 comparisons
unsplit_ttest['pval'] = unsplit_ttest['pval'] * 30


sbcolors = sns.color_palette()
fontsize = 14
axticksize = 14
sessions = [1,3]
max_euc_dist = unsplit['Z(euc_dist)'].max()
min_euc_dist = unsplit['Z(euc_dist)'].min()
#plot a correlogram of the euclidean distances and trial number
fig, axs = plt.subplots(1, 2, figsize=(10,3), sharey=True, sharex=True)
for i, session in enumerate(sessions):
    ax = axs[i]
    for exp_group in ['naive']:
        df = unsplit.loc[(unsplit['exp_group'] == exp_group) & (unsplit['session'] == session)]
        pvaldf = unsplit_ttest.loc[(unsplit_ttest['exp_group'] == exp_group) & (unsplit_ttest['session'] == session)]
        sns.boxenplot(data=df, x='taste_trial', y='Z(euc_dist)', ax=ax, color = sbcolors[0], linewidth=0.5, showfliers=False, k_depth='full')
        #plot a horizontal line that is the average euclidean distance
        avg_euc_dist = df['Z(euc_dist)'].mean()
        ax.axhline(avg_euc_dist, color='black', linestyle='--')
        #if the pvalue for the trial is less than 0.05, plot a star above the line
        for idx, row in pvaldf.iterrows():
            if row['pval'] < 0.05:
                #get the number of stars to plot
                stars = get_sig_stars(row['pval'])
                #get the maximum value of df at the trial number
                med= df.loc[df['taste_trial'] == row['taste_trial'], 'Z(euc_dist)'].max()
                #print the stars turned 90 degrees
                ax.text(row['taste_trial']-0.7, max_euc_dist, stars, fontsize=fontsize, ha='center', va ='bottom', rotation=90)
                #print the p value below
                ax.text(row['taste_trial']-0.7, max_euc_dist, 'p = ' + str(round(row['pval'], 4)), fontsize=9, ha='center', va='top', rotation=-90)

        ax.set_ylim(min_euc_dist,max_euc_dist*1.3)

        ax.set_title('Session ' + str(session), fontsize=fontsize)
        ax.set_xlabel('trial number', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=axticksize)
        #make the x axis labels every other label
        xlabs = ax.get_xticklabels()
        #set every other label to ''
        for label in xlabs:
            label.set_visible(False)
        for label in xlabs[::2]:
            label.set_visible(True)

        if i == 0:
            ax.set_ylabel('Z(euc. dist.)', fontsize=fontsize)
        else:
            ax.set_ylabel('')
plt.tight_layout()
#save the figure
plt.show()
plt.savefig(save_dir + '/trialwise_boxenplot.png')
plt.savefig(save_dir + '/trialwise_boxenplot.svg')