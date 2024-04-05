import analysis as ana
import hmm_analysis as hmma
import blechpy
import new_plotting as nplt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
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
all_units, held_df = PA.get_unit_info(overwrite=False)

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
    euc_dist_mat = np.zeros((rates.shape[1], rates.shape[2]))  # Trials x Bins

    for i in range(rates.shape[1]):  # Loop over trials
        for j in range(rates.shape[2]):  # Loop over bins
            trial_rate_bin = rates[:, i, j]
            avg_firing_rate_bin = avg_firing_rate[:, j]

            # Euclidean distance
            euc_dist = np.linalg.norm(trial_rate_bin - avg_firing_rate_bin)
            euc_dist_mat[i, j] = euc_dist
    return euc_dist_mat

def process_split(rec_dir, split=10, shuffle=False):
    df_list = []
    time_array, rate_array = h5io.get_rate_data(rec_dir)
    for din, rate in rate_array.items():
        if din != 'dig_in_4':
            rate = rate[:,:,2000:5000]
            if shuffle:
                #if shuffle, shuffle the rate array along axis 1
                rate = rate[:, np.random.permutation(rate.shape[1]), :]

            #downsample rate in axis 2 by 10, by taking the average of every 10 bins
            rate = rate.reshape(rate.shape[0], rate.shape[1], -1, 10).mean(axis=3)

            pre_rate = rate[:, :split, :]
            post_rate = rate[:, split:, :]

            pre_euc_dist_mat = euc_dist_trials(pre_rate)
            post_euc_dist_mat = euc_dist_trials(post_rate)

            #bind pre and post euc_dist_mat
            if pre_euc_dist_mat.shape[0] == 0:
                euc_dist_mat = post_euc_dist_mat
            elif post_euc_dist_mat.shape[0] == 0:
                euc_dist_mat = pre_euc_dist_mat
            else: #concantenate pre_euc_dist_mat and post_euc_dist_mat along axis 1
                euc_dist_mat = np.concatenate((pre_euc_dist_mat, post_euc_dist_mat), axis=0)
            # zscore every entry of euc_dist_mat
            #euc_dist_mat = (euc_dist_mat - np.mean(euc_dist_mat)) / np.std(euc_dist_mat)
            avg_euc_dist = np.mean(euc_dist_mat, axis=1) #average across bins

            df = pd.DataFrame({
                'euclidean_distance': avg_euc_dist,
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
            all_df = df.groupby(['rec_dir', 'channel']).mean().reset_index()
            all_df['splitside'] = 'both'
            all_df['trial_group'] = '1-30'
            df = df.groupby(['rec_dir', 'channel', 'splitside']).mean().reset_index()
            df['trial_group'] = df['splitside'].apply(lambda x: minlabel if x == 'pre' else maxlabel)
            #concatenate all_df and df
            df = pd.concat([df, all_df], ignore_index=True)
            #drop taste_trial column
            df = df.drop(columns='taste_trial')
            df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    df = df.reset_index(drop=True)
    return df

def process_rec_dir(rec_dir, shuffle=False):
    splits = np.arange(30)
    dfs = [process_split(rec_dir, split, shuffle) for split in splits]
    dfs = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
    return dfs

# Parallelize processing of each rec_dir
num_cores = -1  # Use all available cores
final_dfs = Parallel(n_jobs=num_cores)(delayed(process_rec_dir)(rec_dir) for rec_dir in rec_dirs)
# Concatenate all resulting data frames into one
final_df = pd.concat(final_dfs, ignore_index=True)

save_dir = PA.save_dir + '/double_mode_stereotypy_analysis'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print('save dir' + save_dir + ' created')
#save to pickle
final_df.to_pickle(save_dir + '/double_mode_stereotypy_analysis.pkl')

#make the shuffle
def iter_shuffle(rec_dirs, niter=100):
    def sing_iter(iternum):
        for rec_dir in rec_dirs:
            res = process_rec_dir(rec_dir, shuffle=True)
            res['iternum'] = iternum
        return res

    dfs = Parallel(n_jobs=-1)(delayed(sing_iter)(iternum) for iternum in range(niter))
    dfs = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
    return dfs

shuff_dfs = iter_shuffle(rec_dirs, niter=100)
shuff_dfs.to_pickle(save_dir + '/double_mode_euc_dist_stereotypy_shuffle.pkl')

#get final df from pickle
final_df = pd.read_pickle(save_dir + '/double_mode_stereotypy_analysis.pkl')




rec_info = proj.rec_info.copy()
rec_info = rec_info[['exp_name', 'exp_group', 'rec_num', 'rec_dir']]
#rename rec num to session
rec_info = rec_info.rename(columns={'rec_num': 'session'})
final_df = pd.merge(final_df, rec_info, on='rec_dir')
final_df = final_df.loc[(final_df['split'] != 1) & (final_df['split'] != 29)]
final_df['euclidean_distance'] = final_df.groupby('rec_dir')['euclidean_distance'].transform(lambda x: (x - x.mean()) / x.std())

unsplit_df = final_df.loc[final_df['splitside'] == 'both']
split_df = final_df.loc[final_df['splitside'] != 'both']

#plot the data

fontsize = 20
axticksize = 17
fig, axs = plt.subplots(1, 3, figsize=(10,5), sharey=True, sharex=True)
for session in [1,2,3]:
    ax = axs[session - 1]
    for exp_group in ['naive']:

        df = unsplit_df.loc[(unsplit_df['exp_group'] == exp_group) & (unsplit_df['session'] == session)]
        #make a seaborn lineplot
        sns.lineplot(x='split', y='euclidean_distance', data=df, ax=ax)
        ax.set_title('Session ' + str(session), fontsize=fontsize)
        ax.set_xlabel('Split-trial', fontsize=fontsize)
        ax.set_ylabel('Z-scored Euc. Dist.', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=axticksize)
plt.tight_layout()
plt.show()
plt.savefig(save_dir + '/double_mode_dist_vs_split.png')
plt.savefig(save_dir + '/double_mode_dist_vs_split.svg')


#for each grouping of split and session, find the template with the lowest euclidean distance in unsplit_df
best_splits = unsplit_df.groupby(['session', 'split', 'exp_group'])['euclidean_distance'].mean().reset_index()
best_splits = best_splits.loc[best_splits.groupby(['session', 'exp_group'])['euclidean_distance'].idxmin()]
#make split zero, which is all rows of final_df where split == 0 and splitside == 'both'
splitzero = final_df.loc[(final_df['split'] == 0) & (final_df['splitside'] == 'both')]
splitzero['splitside'] = 'single\ntemplate'
splitzero['trial_group'] = 'single\ntemplate'
#for final_df, replace all occurences of '1-30' in 'trial_group' with 'combined'
final_df['trial_group'] = final_df['trial_group'].replace('1-30', 'combined')
#refactor all entries in splitside from 'post' to 'post\nsplit' and 'pre' to 'pre\nsplit'
final_df['splitside'] = final_df['splitside'].replace('post', 'post\nsplit')
final_df['splitside'] = final_df['splitside'].replace('pre', 'pre\nsplit')
final_df['splitside'] = final_df['splitside'].replace('both', 'two\ntemplate')

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
fig1, axs1 = plt.subplots(1, 3, figsize=(10,5), sharey=True)
fig2, axs2 = plt.subplots(1, 3, figsize=(10,5), sharey=True)
for session in [1,2,3]:
    axtop = axs1[session - 1]
    axbot = axs2[session - 1]
    for exp_group in ['naive']:
        best_split = best_splits.loc[(best_splits['exp_group'] == exp_group) & (best_splits['session'] == session)]['split'].item()
        df = final_df.loc[(final_df['exp_group'] == exp_group) & (final_df['session'] == session) & (final_df['split'] == best_split)]
        dfzero = splitzero.loc[(splitzero['exp_group'] == exp_group) & (splitzero['session'] == session)]
        df = pd.concat([df, dfzero], ignore_index=True)
        #join exp_name and channel to make column antaste
        df['AnTaste'] = df['exp_name'] + df['channel'].astype(str)
        #for splitavodf, get all rows where splitside == 'pre\nsplit' and 'post\nsplit'
        splitavodf = df.loc[(df['splitside'] == 'pre\nsplit') | (df['splitside'] == 'post\nsplit')]
        #for templdf, get all rows where splitside == 'two\ntemplate' or 'single\ntemplate'
        templdf = df.loc[(df['splitside'] == 'two\ntemplate') | (df['splitside'] == 'single\ntemplate')]


        #perform a repeated measures anova
        splitavo = pg.rm_anova(data=splitavodf, dv='euclidean_distance', within=['splitside','channel'], subject='exp_name', detailed=True)
        split_pval = splitavo['p-unc'].values[0]
        splitavos.append(splitavo)
        templavo = pg.rm_anova(data=templdf, dv='euclidean_distance', within=['splitside','channel'], subject='exp_name', detailed=True)
        templ_pval = templavo['p-unc'].values[0]
        templavos.append(templavo)

        #append dfzero to df
        sns.barplot(x='splitside', y='euclidean_distance', data=splitavodf, ax=axtop, order=['pre\nsplit', 'post\nsplit'], fill=False, edgecolor='black')
        sns.barplot(x='splitside', y='euclidean_distance', data=templdf, ax=axbot, order=['two\ntemplate', 'single\ntemplate'], fill=False, edgecolor='black')
        #set the y axis top limit for both plots to 1
        axtop.set_ylim(top=1)
        axbot.set_ylim(top=1)
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
            axtop.set_ylabel('Z(Euc. Dist.)', fontsize=fontsize)
            #make the y axis labels larger
            axbot.set_ylabel('Z(Euc. Dist.)', fontsize=fontsize)

        else:
            axtop.set_ylabel('')
            axbot.set_ylabel('')
        axtop.tick_params(axis='both', which='major', labelsize=axticksize)
        axbot.tick_params(axis='both', which='major', labelsize=axticksize)

        axbot.set_xlabel("split trial: " + str(int(best_split+1)), fontsize=fontsize)
        axtop.set_xlabel("split trial: " + str(int(best_split+1)), fontsize=fontsize)
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