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

def process_split(best_hmms, split, shuffle=False):
    id_cols = ['exp_name', 'exp_group', 'session', 'taste', 'hmm_id']
    data_cols = ['split', 'pr_pre', 'pr_post']
    split_mode_hmms = hmma.getSplitModeHMMs(best_hmms, split_trial=split, shuffle=shuffle)  # get the split mode hmms
    df = split_mode_hmms[id_cols]
    df[data_cols] = split_mode_hmms[data_cols]
    return df

from joblib import Parallel, delayed
splits = np.arange(1,30)
res_list = Parallel(n_jobs=11)(delayed(process_split)(best_hmms, split) for split in splits)
# res_list = []
# for split in splits:
#     res_list.append(process_split(best_hmms, split))
overall_acc_df = pd.concat(res_list)
overall_acc_df

def get_null_dist(best_hmms, overwrite=False):
    save_dir = HA.save_dir
    folder_name = 'split_trial_hmm_analysis'
    save_dir = os.path.join(save_dir, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if overwrite:
        # splits = np.arange(1,30)
        # niter = 100
        #
        # shuffle_df = Parallel(n_jobs=8)(delayed(iter_splits)(best_hmms, splits, i) for i in range(niter))
        # shuffle_df = pd.concat(shuffle_df)
        # shuffle_df.to_pickle(os.path.join(save_dir, 'overall_acc_df_shuffle.pkl'))
    else:
        shuffle_df = pd.read_pickle(os.path.join(save_dir, 'overall_acc_df_shuffle.pkl'))
    return shuffle_df

overall_acc_df_shuffle = get_null_dist(best_hmms, overwrite=False)


def get_accuracy(df):
    pr_templates = []
    pr_opposites = []
    for i, row in df.iterrows():
        split_trial = row['split']
        pr_pre = row['pr_pre'][:split_trial]

        if row['pr_post'] is not None:
            pr_post = row['pr_post'][split_trial:]
            pr_template = np.concatenate((pr_pre, pr_post)).mean()
            pr_templates.append(pr_template)

            pr_pre_opp = row['pr_pre'][split_trial:]
            pr_post_opp = row['pr_post'][:split_trial]
            pr_opposite = np.concatenate((pr_pre_opp, pr_post_opp)).mean()
            pr_opposites.append(pr_opposite)
        else:
            pr_template = pr_pre.mean()
            pr_templates.append(pr_template)
            pr_opposite = np.nan
            pr_opposites.append(pr_opposite)

    df['pr_template'] = pr_templates
    df['pr_opposite'] = pr_opposites
    return df

overall_acc_df = get_accuracy(overall_acc_df)
overall_acc_df_shuffle = get_accuracy(overall_acc_df_shuffle)

overall_acc_df_shuffle_save = overall_acc_df_shuffle.copy()
overall_acc_df_shuffle['session'] = overall_acc_df_shuffle['session'].astype(int)
overall_acc_df['session'] = overall_acc_df['session'].astype(int)

import seaborn as sns
import matplotlib.pyplot as plt
import os  # Make sure you've imported os for os.path.join()

plt.rcParams.update({'font.size': 20})

# Get the color code for the first color in seaborn's tab10 palette
tabBlue = sns.color_palette('tab10')[0]
tabBrown = sns.color_palette('tab10')[5]

# Assuming overall_acc_df and overall_acc_df_shuffle are already defined and preprocessed as needed
overall_acc_df['session'] = overall_acc_df['session'].astype(int)

import seaborn as sns
plt.rcParams.update({'font.size': 20})
#get the color code for the first color in seaborn's tab10 palette
tabBlue = sns.color_palette('tab10')[0]
tabBrown = sns.color_palette('tab10')[5]
#overall_acc_df = overall_acc_df.explode('accuracy')
overall_acc_df['session'] = overall_acc_df['session'].astype(int)
#make a figure with 3 columns as subplots
fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True, sharex=True)
for session in [1, 2, 3]:
    ax = axs[session-1]
    sample_group = overall_acc_df.loc[overall_acc_df['session'] == session].reset_index(drop=True)
    sample_acc = sample_group['pr_template'].to_numpy()
    sample_opp = sample_group['pr_opposite'].to_numpy()

    shuffle_group = overall_acc_df_shuffle.loc[overall_acc_df_shuffle['session'] == session].reset_index(drop=True)
    shuffle_acc = shuffle_group['pr_template'].to_numpy()

    sample_splits = sample_group['split'].values
    shuffle_splits = shuffle_group['split'].values

    #make the line fatter
    sns.lineplot(x=shuffle_splits, y=shuffle_acc, ax=ax, color='gray', linewidth=2)
    sns.lineplot(x=sample_splits, y=sample_opp, ax=ax, color=tabBrown, linewidth=2)
    sns.lineplot(x=sample_splits, y=sample_acc, ax=ax, color=tabBlue, linewidth=2)
    #add a legend

    ax.set_ylim(0.25, 1)
    #shift the x axis forward by 1
    ax.set_xticklabels(ax.get_xticks() + 1)
    #set the x ticks so they are 1, 15, 30
    ax.set_xticks([0, 14, 29])
    ax.set_title(f'Session {session}')
    ax.set_xlabel('Split-trial')
    #set the y ticks to be size 20
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=18)
    if session == 1:
        ax.set_ylabel('pr(template)')

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)  # Adjust the bottom
axs[1].legend(['trial shuffle', 'opposite template', 'matching template'],
          bbox_to_anchor=(0.5, -0.2), loc='upper center', ncol=3, fontsize=20, fancybox=False, shadow=False)
plt.show()
#save plot
save_dir = HA.save_dir
folder_name = 'split_trial_hmm_analysis'
save_dir = os.path.join(save_dir, folder_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
plt.savefig(os.path.join(save_dir, 'split_trial_accuracy_wshuffle.svg'))
plt.savefig(os.path.join(save_dir, 'split_trial_accuracy_wshuffle.png'))
