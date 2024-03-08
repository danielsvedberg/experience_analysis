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
#
proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
HA = ana.HmmAnalysis(proj)  # create a hmm analysis object

best_hmms = HA.get_best_hmms(sorting = 'best_AIC')  # get the best hmms
best_hmms['session'] = best_hmms['time_group']  # rename time_group to session
#just get naive hmms
best_hmms = best_hmms.loc[best_hmms['exp_group'] == 'naive'].reset_index(drop=True)
def get_average_pr_mode(pr_mode_mat, time):
    #get indices of time greater than 100 and less than 2500
    time_idx = np.where((time >= 100) & (time <= 2500))[0]
    pr_mode_mat = pr_mode_mat[:, time_idx]
    pr_mode_mat = np.mean(pr_mode_mat, axis=1)
    return pr_mode_mat

def get_accuracy(split_mode_hmms):
    overall_avg = []
    for i, row in split_mode_hmms.iterrows():
        time = row['time']
        # get the early split p(mode)
        pre_pr_mode_mat = row['pre_pr_mode']
        pre_res = get_average_pr_mode(pre_pr_mode_mat, time)
        post_pr_mode_mat = row['post_pr_mode']
        if post_pr_mode_mat is None:
            overall_avg.append(pre_res)
        else:
            post_res = get_average_pr_mode(post_pr_mode_mat, time)
            overall_avg.append(np.concatenate((pre_res, post_res)))

    return overall_avg

def process_split(best_hmms, split, shuffle=False):
    id_cols = ['exp_name', 'exp_group', 'session', 'taste', 'hmm_id']
    split_mode_hmms = hmma.getSplitModeHMMs(best_hmms, split_trial=split, shuffle=shuffle)  # get the split mode hmms
    acc = get_accuracy(split_mode_hmms)
    df = split_mode_hmms[id_cols]
    df['split'] = split
    df['accuracy'] = acc
    return df


#make a null distribution version of this

splits = np.arange(1,29)
res_list = []
for iter in range(200):
    split_list = []
    for split in splits:
        split_list.append(process_split(best_hmms, split, shuffle=True))
    split_df = pd.concat(split_list)
    split_df['iter'] = iter
    res_list.append(split_df)
overall_acc_df_shuffle = pd.concat(res_list)
#save to .json
save_dir = HA.save_dir
folder_name = 'split_trial_hmm_analysis'
save_dir = os.path.join(save_dir, folder_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
#overall_acc_df_shuffle.to_json(os.path.join(save_dir, 'overall_acc_df_shuffle.json'))
#save to pickle
overall_acc_df_shuffle.to_pickle(os.path.join(save_dir, 'overall_acc_df_shuffle.pkl'))
overall_acc_df_shuffle_save = overall_acc_df_shuffle.copy()
overall_acc_df_shuffle['accuracy'] = overall_acc_df_shuffle['accuracy'].apply(np.mean)
overall_acc_df_shuffle['session'] = overall_acc_df_shuffle['session'].astype(int)

splits = np.arange(1,29)
res_list = []
for split in splits:
    res_list.append(process_split(best_hmms, split))
overall_acc_df = pd.concat(res_list)
overall_acc_df['accuracy'] = overall_acc_df['accuracy'].apply(np.mean)

#set rcparams font to arial
plt.rcParams['font.family'] = 'FreeSans'
plt.rcParams["svg.fonttype"] = 'none'

import seaborn as sns
#get the color code for the first color in seaborn's tab10 palette
tabBlue = sns.color_palette('tab10')[0]
#overall_acc_df = overall_acc_df.explode('accuracy')
overall_acc_df['session'] = overall_acc_df['session'].astype(int)
#make a figure with 3 columns as subplots
fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True, sharex=True)
for session in [1, 2, 3]:
    ax = axs[session-1]
    sample_group = overall_acc_df.loc[overall_acc_df['session'] == session].reset_index(drop=True)
    sample_acc = sample_group['accuracy'].to_numpy()

    shuffle_group = overall_acc_df_shuffle.loc[overall_acc_df_shuffle['session'] == session].reset_index(drop=True)
    shuffle_acc = shuffle_group['accuracy'].to_numpy()

    sample_splits = sample_group['split'].values
    shuffle_splits = shuffle_group['split'].values

    sns.lineplot(x=shuffle_splits, y=shuffle_acc, ax=ax, color='gray')
    sns.lineplot(x=sample_splits, y=sample_acc, ax=ax, color=tabBlue)

    ax.set_ylim(0.5, 1)
    ax.set_title(f'Session {session}')
    ax.set_xlabel('Split-trial')
    if session == 1:
        ax.set_ylabel('pr(template)')
plt.tight_layout()
plt.show()
#save plot
plt.savefig(os.path.join(save_dir, 'split_trial_accuracy_wshuffle.svg'))
plt.savefig(os.path.join(save_dir, 'split_trial_accuracy_wshuffle.png'))


