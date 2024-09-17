import analysis as ana
import hmm_analysis as hmma
import blechpy
import new_plotting as nplt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

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
rec_dirs = rec_info['rec_dir']
PA = ana.ProjectAnalysis(proj)
all_units, held_df = PA.get_unit_info(overwrite=True)

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

def process_split(rec_dir, split=10, shuffle=False):
    df_list = []
    time_array, rate_array = h5io.get_psths(rec_dir)
    ts = 100
    te = 3000
    #get indexs of time_array where time_array is greater than ts and less than te
    i_e = (time_array >= ts) & (time_array <= te)
    time_array = time_array[i_e]
    for din, rate in rate_array.items():
        if din != 'dig_in_4':
            rate = rate[:,:,i_e]

            if shuffle:
                #if shuffle, shuffle the rate array along axis 1
                rate = rate[:, np.random.permutation(rate.shape[1]), :]

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
    dfs = [process_split(rec_dir, split, shuffle) for split in splits]
    dfs = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
    return dfs
#make the shuffle
def iter_shuffle(rec_dirs, niter=100):
    def sing_iter(iternum):
        res_list = []
        for rd in rec_dirs:
            print('Processing rec_dir: ' + rd)
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
final_dfs = Parallel(n_jobs=num_cores)(delayed(process_rec_dir)(rec_dir) for rec_dir in rec_dirs)
# Concatenate all resulting data frames into one
final_df = pd.concat(final_dfs, ignore_index=True)
#save to pickle
final_df.to_pickle(save_dir + '/double_mode_stereotypy_analysis.pkl')

shuff_dfs = iter_shuffle(rec_dirs, niter=100)
shuff_dfs.to_pickle(save_dir + '/double_mode_euc_dist_stereotypy_shuffle.pkl')

######################################################################################################################
rec_info = proj.rec_info.copy()
rec_info = rec_info[['exp_name', 'exp_group', 'rec_num', 'rec_dir']]
#rename rec num to session
rec_info = rec_info.rename(columns={'rec_num': 'session'})

#get final df from pickle and preprocess###############################################################################
final_df = pd.read_pickle(save_dir + '/double_mode_stereotypy_analysis.pkl')
final_df = pd.merge(final_df, rec_info, on='rec_dir')
#remove split == 30
final_df = final_df.loc[final_df['split'] != 30]
#for each combination of rec_dir and channel, find the maximum trial. The split cannot be greater than the maximum trial - 2
max_trials = final_df.groupby(['rec_dir', 'channel'])['taste_trial'].max().reset_index()
max_trials = max_trials.rename(columns={'taste_trial': 'max_trial'})
final_df = pd.merge(final_df, max_trials, on=['rec_dir', 'channel'])
final_df = final_df.loc[final_df['max_trial'] - final_df['split'] > 1]
final_df = final_df.loc[final_df['split'] != 2]
final_df['Z(euc dist)'] = final_df.groupby(['rec_dir','channel'])['euclidean_distance'].transform(lambda x: (x - x.mean()) / x.std())
unsplit = final_df.loc[final_df['split'] == 0]
splitzero = unsplit.groupby(['exp_name', 'channel', 'session', 'exp_group']).mean().reset_index()
final_df = pd.merge(final_df, splitzero, on=['exp_name', 'channel', 'session', 'exp_group'], suffixes=('', '_splitzero'))
final_df['net_euc_dist'] = final_df['euclidean_distance']/final_df['euclidean_distance_splitzero']


#preprocess the shuffle ################################################################################################
shuff_dfs = pd.read_pickle(save_dir + '/double_mode_euc_dist_stereotypy_shuffle.pkl')
final_shuff_df = pd.merge(shuff_dfs, rec_info, on='rec_dir')
#remove split == 30
final_shuff_df = final_shuff_df.loc[final_shuff_df['split'] != 30]
#for each combination of rec_dir and channel, find the maximum trial. The split cannot be greater than the maximum trial - 1
max_trials = final_shuff_df.groupby(['rec_dir', 'channel'])['taste_trial'].max().reset_index()
max_trials = max_trials.rename(columns={'taste_trial': 'max_trial'})
final_shuff_df = pd.merge(final_shuff_df, max_trials, on=['rec_dir', 'channel'])
final_shuff_df = final_shuff_df.loc[final_shuff_df['max_trial'] - final_shuff_df['split'] > 1]
final_shuff_df = final_shuff_df.loc[final_shuff_df['split'] != 2]
final_shuff_df['Z(euc dist)'] = final_shuff_df.groupby(['rec_dir','channel','iternum'])['euclidean_distance'].transform(lambda x: (x - x.mean()) / x.std())
unsplit_shuff = final_shuff_df.loc[final_shuff_df['split'] == 0]
splitzero_shuff = unsplit_shuff.groupby(['exp_name', 'channel', 'session', 'exp_group', 'iternum']).mean().reset_index()
final_shuff_df = pd.merge(final_shuff_df, splitzero_shuff, on=['exp_name', 'channel', 'session', 'exp_group', 'iternum'], suffixes=('', '_splitzero'))
final_shuff_df['net_euc_dist'] = final_shuff_df['euclidean_distance']/final_shuff_df['euclidean_distance_splitzero']

#for each split in avg_df, for each grouping of rec_dir and channel, get the difference between the real euclidean distance and the shuffled euclidean distance for that grouping
#first, get the average shuffled euclidean distance for each split:
iteravgshuff = final_shuff_df.groupby(['exp_name', 'channel', 'session', 'exp_group', 'split', 'taste_trial']).mean().reset_index()
#then, introduce a column to avg_df that is the shuffled euclidean distance for that split, via merge
final_df = pd.merge(final_df, iteravgshuff, on=['exp_name', 'channel', 'session', 'exp_group', 'split', 'taste_trial'], suffixes=('', '_shuff'))
final_shuff_df = pd.merge(final_shuff_df, iteravgshuff, on=['exp_name', 'channel', 'session', 'exp_group', 'split', 'taste_trial'], suffixes=('', '_shuff'))

final_df['euc_dist_shuffnorm'] = final_df['euclidean_distance']/final_df['euclidean_distance_shuff']
final_df['Z(euc_dist_shuffnorm)'] = final_df.groupby(['rec_dir','channel'])['euc_dist_shuffnorm'].transform(lambda x: (x - x.mean()) / x.std())
final_shuff_df['euc_dist_shuffnorm'] = final_shuff_df['euclidean_distance']/final_shuff_df['euclidean_distance_shuff']
final_shuff_df['Z(euc_dist_shuffnorm)'] = final_shuff_df.groupby(['rec_dir','channel','iternum'])['euc_dist_shuffnorm'].transform(lambda x: (x - x.mean()) / x.std())

unsplit = final_df.loc[final_df['split'] == 0]
splitzero = unsplit.groupby(['exp_name', 'channel', 'session', 'exp_group']).mean().reset_index()
final_df = final_df.loc[final_df['split'] != 0]
unsplit_shuff = final_shuff_df.loc[final_shuff_df['split'] == 0]
splitzero_shuff = unsplit_shuff.groupby(['exp_name', 'channel', 'session', 'exp_group', 'iternum']).mean().reset_index()
final_shuff_df = final_shuff_df.loc[final_shuff_df['split'] != 0]


#average the euclidean distance for each animal, taste, split, and session
avg_df = final_df.groupby(['exp_name', 'channel', 'split', 'session', 'exp_group']).mean().reset_index()
avg_shuff_df = final_shuff_df.groupby(['exp_name', 'channel', 'split', 'session', 'exp_group', 'iternum']).mean().reset_index()
#get the average euclidean distance for split == 0

#get the average euclidean distance for split == 0
shuffsplitzero = unsplit_shuff.groupby(['exp_name', 'channel', 'session', 'exp_group']).mean().reset_index()

best_split_df = avg_df.loc[avg_df.groupby(['exp_name', 'session', 'exp_group', 'channel'])['net_euc_dist'].idxmin()]
best_shuff = avg_shuff_df.loc[avg_shuff_df.groupby(['exp_name', 'session', 'exp_group', 'channel', 'iternum'])['euclidean_distance'].idxmin()]

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
        sns.lineplot(x='split', y='euclidean_distance', data=df, ax=ax, ci=None)
        sns.lineplot(x='split', y='euclidean_distance', data=shuff_df, ax=ax, color='black', alpha=0.1, ci = ci)
        #plot a blue dashed horizontal line at y = splitzero, with a 95% confidence interval
        sns.lineplot(x=[2, 29], y=sz['euclidean_distance'].mean(), ax=ax, linestyle='--', color = sbcolors[0])
        ax.set_title('Session ' + str(session), fontsize=fontsize)
        ax.set_xlabel('Split-trial', fontsize=fontsize)
        ax.set_ylabel('Z(Euc. Dist.)', fontsize=fontsize)
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
        df = avg_df.loc[(avg_df['exp_group'] == exp_group) & (avg_df['session'] == session)]
        sz = splitzero.loc[(splitzero['exp_group'] == exp_group) & (splitzero['session'] == session)]
        shuff_df = avg_shuff_df.loc[(avg_shuff_df['exp_group'] == exp_group) & (avg_shuff_df['session'] == session)]
        #make a seaborn lineplot
        sns.lineplot(x='split', y='Z(euc_dist_shuffnorm)', data=df, ax=ax)
        sns.lineplot(x='split', y='Z(euc_dist_shuffnorm)', data=shuff_df, ax=ax, color='black', alpha=0.1, ci = ci)
        #plot a blue dashed horizontal line at y = splitzero, with a 95% confidence interval
        #sns.lineplot(x='split', y='euclidean_distance_splitzero', ax=ax, linestyle='--', color = sbcolors[0])
        ax.set_title('Session ' + str(session), fontsize=fontsize)
        ax.set_xlabel('Split-trial', fontsize=fontsize)
        ax.set_ylabel('Z(Euc. Dist.)', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=axticksize)
plt.tight_layout()
plt.show()
plt.savefig(save_dir + '/double_mode_dist_vs_split_shuff_norm.png')
plt.savefig(save_dir + '/double_mode_dist_vs_split_shuff_norm.svg')


#now make a histogram of the best split for each session
fig, axs = plt.subplots(1, 2, figsize=(5,3), sharey=True, sharex=True)
for i, session in enumerate([1,3]):
    for exp_group in ['naive']:
        df = best_split_df.loc[(best_split_df['exp_group'] == exp_group) & (best_split_df['session'] == session)]
        shuffdf = best_shuff.loc[(best_shuff['exp_group'] == exp_group) & (best_shuff['session'] == session)]
        #join the two dataframes
        # df['type'] = 'real'
        # shuffdf['type'] = 'shuffled'
        # df = pd.concat([df, shuffdf], ignore_index=True)

        ax = axs[i]
        sns.histplot(ax=ax, data=df, x='split', stat='probability', alpha=0.5, color=sbcolors[0], binwidth=4)
        sns.histplot(shuffdf['split'], ax=ax, stat='probability', color='black', alpha=0.3, binwidth=4)

        ax.set_title('Session ' + str(session), fontsize=fontsize)
        ax.set_xlabel('best split-trial', fontsize=fontsize)
        ax.set_ylabel('fraction', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=axticksize)
        ax.set_xlim(3, 27)
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

def get_sig_stars(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return ''


splitavos = []
templavos = []
import pingouin as pg
fig1, axs1 = plt.subplots(1, 2, figsize=(10,5), sharey=True)
fig2, axs2 = plt.subplots(1, 2, figsize=(10,5), sharey=True)
for i, session in enumerate([1,3]):
    axtop = axs1[i]
    axbot = axs2[i]
    for exp_group in ['naive']:
        #get the df with the best split for each exp_name
        best_split = best_split_df.loc[(best_split_df['exp_group'] == exp_group) & (best_split_df['session'] == session)][['exp_name','split']]
        #get the rows of final_df where the exp_name and split match the best split
        df = final_df.loc[(final_df['exp_group'] == exp_group) & (final_df['session'] == session)]
        df = pd.merge(df, best_split, on=['exp_name', 'split'])

        dfzero = unsplit.loc[(unsplit['exp_group'] == exp_group) & (unsplit['session'] == session)]
        df = pd.concat([df, dfzero], ignore_index=True)
        #join exp_name and channel to make column antaste
        df['AnTaste'] = df['exp_name'] + df['channel'].astype(str)
        #for splitavodf, get all rows where splitside == 'pre\nsplit' and 'post\nsplit'
        splitavodf = df.loc[(df['splitside'] == 'pre\nsplit') | (df['splitside'] == 'post\nsplit')]
        #take the average of euclidean distance for each exp_name, channel, and splitside
        splitavodf = splitavodf.groupby(['exp_name', 'channel', 'splitside', 'session', 'exp_group']).mean().reset_index()
        splitavodf['expXchannel'] = splitavodf['exp_name'] + splitavodf['channel'].astype(str)

        #for templdf, get all rows where splitside == 'two\ntemplate' or 'single\ntemplate'
        templdf = df.loc[(df['template'] == 'two\ntemplate') | (df['template'] == 'single\ntemplate')]
        #take the average of euclidean distance for each exp_name, channel, and template
        templdf = templdf.groupby(['exp_name', 'channel', 'template', 'session', 'exp_group']).mean().reset_index()
        templdf['expXchannel'] = templdf['exp_name'] + templdf['channel'].astype(str)
        #perform a repeated measures anova
        splitavo = pg.rm_anova(data=splitavodf, dv='net_euc_dist', within=['splitside'], subject='expXchannel', detailed=True)
        split_pval = splitavo['p-unc'].values[0] * 3
        splitavos.append(splitavo)
        templavo = pg.rm_anova(data=templdf, dv='net_euc_dist', within=['template'], subject='expXchannel', detailed=True)
        templ_pval = templavo['p-unc'].values[0] * 3
        templavos.append(templavo)
        #append dfzero to df
        sns.barplot(x='splitside', y='net_euc_dist', data=splitavodf, ax=axtop, order=['pre\nsplit', 'post\nsplit'], fill=False, edgecolor='black')
        sns.swarmplot(x='splitside', y='net_euc_dist', data=splitavodf, ax=axtop, order=['pre\nsplit', 'post\nsplit'])
        sns.barplot(x='template', y='net_euc_dist', data=templdf, ax=axbot, order=['two\ntemplate', 'single\ntemplate'], fill=False, edgecolor='black')
        sns.swarmplot(x='template', y='net_euc_dist', data=templdf, ax=axbot, order=['two\ntemplate', 'single\ntemplate'])
        #set the y axis top limit for both plots to 1
        #axtop.set_ylim(top=1)
        #axbot.set_ylim(top=1)
        if split_pval < 0.05: #then draw a line between 0 and 1, and put a star above the line
            axtop.plot([0, 1], [0.8, 0.8], lw=2, color='black')
            stars = get_sig_stars(split_pval)
            axtop.text(0.5, 0.8, stars, fontsize=fontsize, ha='center')
        if templ_pval < 0.05:
            axbot.plot([0,1], [0.9, 0.9], lw=2, color='black')
            stars = get_sig_stars(templ_pval)
            axbot.text(0.5, 0.9, stars, fontsize=fontsize, ha='center')

        #take away the y axis label
        if session == 1:
            axtop.set_ylabel('shuff. normalized\nZ(Euc. Dist.)', fontsize=fontsize)
            #make the y axis labels larger
            axbot.set_ylabel('shuff. normalized\nZ(Euc. Dist.)', fontsize=fontsize)

        else:
            axtop.set_ylabel('')
            axbot.set_ylabel('')
        axtop.tick_params(axis='both', which='major', labelsize=axticksize)
        axbot.tick_params(axis='both', which='major', labelsize=axticksize)

        #axbot.set_xlabel("split trial: " + str(int(best_split+1)), fontsize=fontsize)
        #axtop.set_xlabel("split trial: " + str(int(best_split+1)), fontsize=fontsize)
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