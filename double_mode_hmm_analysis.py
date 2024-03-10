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

#make a null distribution version of this
splits = np.arange(1,30)

res_list = []
for iternum in range(100):
    print("iter: ", iternum)
    split_list = Parallel(n_jobs=10)(delayed(process_split)(best_hmms, split, shuffle=True) for split in splits)
    split_df = pd.concat(split_list)
    split_df['iter'] = iternum
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



#%% make bar graphs of the differences
#first, for each trial, get the rows with the best split-trial accuracy in overall_acc_df
best_splits = []
best_splits_shuffle = []
for name, group in overall_acc_df.groupby(['session']):
    group = group.reset_index(drop=True)
    group_shuffle = overall_acc_df_shuffle.loc[overall_acc_df_shuffle['session'] == name].reset_index(drop=True)
    #get the mean pr_template for each grouping of split trial
    mean_pr = group.groupby('split')['pr_template'].mean()
    best_split = mean_pr.idxmax()
    best_split_df = group.loc[group['split'] == best_split]
    best_splits.append(best_split_df)

    best_split_shuffle = group_shuffle.loc[group_shuffle['split'] == best_split]
    best_splits_shuffle.append(best_split_shuffle)

best_splits_df = pd.concat(best_splits)
best_splits_df['group'] = 'best\ntrial-split'
best_splits_df['iter'] = -1

best_splits_shuffle_df = pd.concat(best_splits_shuffle)
best_splits_shuffle_df['group'] = 'best\ntrial-split'


max_split = overall_acc_df['split'].max()
single_template = overall_acc_df.loc[overall_acc_df['split'] == max_split].reset_index(drop=True)
single_template['group'] = 'single\ntemplate'
single_template['iter'] = -1

single_template_shuffle = overall_acc_df_shuffle.loc[overall_acc_df_shuffle['split'] == max_split].reset_index(drop=True)
single_template_shuffle['group'] = 'single\ntemplate'

#concatenate all the dfs into one long one
bar_df = pd.concat([best_splits_df, single_template])
bar_df_shuffle = pd.concat([best_splits_shuffle_df, single_template_shuffle])

def get_stars(pval):
    if pval < 0.05:
        return '*'
    elif pval < 0.01:
        return '**'
    elif pval < 0.001:
        return '***'

#import ols
import pingouin as pg
#perform a repeated measures ANOVA
#between the groups, perform an ANOVA, taking into account exp_name, taste, and session
bar_df['exp_name_taste'] = bar_df['exp_name'] + '_' + bar_df['taste']

tab_blue = sns.color_palette('tab10')[0]
#set matplotlib text to larger font
plt.rcParams.update({'font.size': 20})
#get the 95th percentile highest value of bar_df['pr_template']
max_val = bar_df['pr_template'].quantile(0.95)
groups = bar_df['group'].unique().tolist()
fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharey=True, sharex=True)
for i, session in enumerate([1, 2, 3]):
    ax = axs[i]
    sample_group = bar_df.loc[(bar_df['session'] == session)].reset_index(drop=True)
    aov = pg.rm_anova(data=sample_group, dv='pr_template', within='group', subject='exp_name_taste')
    pw = pg.pairwise_ttests(data=sample_group, dv='pr_template', within='group', subject='exp_name_taste')
    print(pw)
    #between the groups, perform an ANOVA, taking into account exp_name, taste, and session

    shuffle_group = bar_df_shuffle.loc[(bar_df_shuffle['session'] == session)].reset_index(drop=True)
    shuffle_group = shuffle_group.groupby(['group', 'iter']).mean().reset_index()
    #shuffle upper 95% confidence interval
    conf_level = (0.05/3)
    shuffle_upper = shuffle_group.groupby('group')['pr_template'].quantile(1-conf_level).reset_index()
    shuffle_lower = shuffle_group.groupby('group')['pr_template'].quantile(0).reset_index()
    x = shuffle_upper['group']
    upper = shuffle_upper['pr_template']
    lower = shuffle_lower['pr_template']
    sns.barplot(x='group', y='pr_template', data=sample_group, ax=ax, ci=95, color='White', edgecolor=tab_blue, capsize=0.5, linewidth=2)
    ax.bar(x=x, height=upper-lower, bottom=lower, color='gray', alpha=1, linewidth=2)
    ax.set_ylim(0, 1)
    #if pw['p-unc'][0] < 0.05, add a horizontal line between the two groups and a star above it
    pval = pw['p-unc'][0]
    if pval < 0.05:
        #get the maximum value from sample_group
        stars = get_stars(pval)
        ax.plot([0, 1], [max_val, max_val], color='k', linewidth=2)
        ax.text(0.5, max_val+0.01, stars, fontsize=20, ha='center')
    ax.set_xlabel('')
    ax.set_title(f'Session {session}')
    ax.set_xticklabels(['best\ntrial-split', 'single\ntemplate'], fontsize=15)
    if session == 1:
        ax.set_ylabel('pr(template)')
    else:
        ax.set_ylabel('')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
plt.show()
#save
plt.savefig(os.path.join(save_dir, 'split_trial_accuracy_bars.svg'))
plt.savefig(os.path.join(save_dir, 'split_trial_accuracy_bars.png'))

best_split_d1 = best_splits_df.loc[best_splits_df['session'] == 1]

