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
#
proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
HA = ana.HmmAnalysis(proj)  # create a hmm analysis object

best_hmms = HA.get_best_hmms(sorting='best_AIC')  # get the best hmms
best_hmms['session'] = best_hmms['time_group']  # rename time_group to session
#just get naive hmms
best_hmms = best_hmms.loc[best_hmms['exp_group'] == 'naive'].reset_index(drop=True)
best_hmms = best_hmms[['rec_dir', 'exp_name', 'exp_group','session', 'taste', 'channel', 'hmm_id']]

def process_split(best_hmms_df, split, shuffle=False):
    id_cols = ['exp_name', 'exp_group', 'session', 'taste', 'hmm_id']
    data_cols = ['split', 'pr_pre', 'pr_post']
    split_mode_hmms = hmma.getSplitModeHMMs(best_hmms_df, split_trial=split, shuffle=shuffle)  # get the split mode hmms
    #df = pd.concat([split_mode_hmms[id_cols], split_mode_hmms[data_cols]], axis=1)
    df = split_mode_hmms[id_cols]
    df[data_cols] = split_mode_hmms[data_cols]
    #df = df.reset_index(drop=True)
    return df

def iter_splits(best_hmms_df, splits, iternum):
    print(f'Iteration {iternum}')
    split_df = [process_split(best_hmms_df, split, shuffle=True) for split in splits]
    split_df = pd.concat(split_df, ignore_index=True)
    split_df['iter'] = iternum
    return split_df

def get_null_dist(best_hmms, overwrite=False):
    save_dir = HA.save_dir
    folder_name = 'split_trial_hmm_analysis'
    save_dir = os.path.join(save_dir, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if overwrite:
        splits = np.arange(1,30)
        niter = 100

        shuffle_df = Parallel(n_jobs=8)(delayed(iter_splits)(best_hmms, splits, i) for i in range(niter))
        shuffle_df = pd.concat(shuffle_df)
        shuffle_df.to_pickle(os.path.join(save_dir, 'overall_acc_df_shuffle.pkl'))
    else:
        shuffle_df = pd.read_pickle(os.path.join(save_dir, 'overall_acc_df_shuffle.pkl'))
    return shuffle_df

overall_acc_df_shuffle = get_null_dist(best_hmms, overwrite=False)

def get_acc_df(best_hmms, overwrite=False):
    save_dir = HA.save_dir
    folder_name = 'split_trial_hmm_analysis'
    save_dir = os.path.join(save_dir, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if overwrite:
        splits = np.arange(0,30)
        overall_acc_df = Parallel(n_jobs=10)(delayed(process_split)(best_hmms, split) for split in splits)
        overall_acc_df = pd.concat(overall_acc_df)
        overall_acc_df.to_pickle(os.path.join(save_dir, 'overall_acc_df.pkl'))
    else:
        overall_acc_df = pd.read_pickle(os.path.join(save_dir, 'overall_acc_df.pkl'))
    return overall_acc_df

overall_acc_df = get_acc_df(best_hmms, overwrite=False)

def get_accuracy(df):
    pr_templates = []
    pr_opposites = []
    pr_pres = []
    pr_posts = []
    for i, row in df.iterrows():
        split_trial = row['split']
        if row['pr_post'] is not None:# and row['pr_pre'] is not None:
            pr_post = row['pr_post'][split_trial:]
            pr_posts.append(pr_post.mean())
            pr_post_opp = row['pr_post'][:split_trial]
        else:
            pr_post = np.nan
            pr_posts.append(pr_post)
            pr_post_opp = np.nan

        if row['pr_pre'] is not None:
            pr_pre = row['pr_pre'][:split_trial]
            pr_pres.append(pr_pre.mean())
            pr_pre_opp = row['pr_pre'][split_trial:]
        else:
            pr_pre = np.nan
            pr_pres.append(pr_pre)
            pr_pre_opp = np.nan

        if pr_pre is not np.nan and pr_post is not np.nan:
            pr_template = np.nanmean(np.concatenate((pr_pre, pr_post)))
            pr_opposite = np.nanmean(np.concatenate((pr_pre_opp, pr_post_opp)))
        elif pr_pre is not np.nan:
            pr_template = pr_pre.mean()
            pr_opposite = pr_pre_opp.mean()
        elif pr_post is not np.nan:
            pr_template = pr_post.mean()
            pr_opposite = pr_post_opp.mean()

        pr_templates.append(pr_template)
        pr_opposites.append(pr_opposite)

    df['pr_pre'] = pr_pres
    df['pr_post'] = pr_posts
    df['pr_template'] = pr_templates
    df['pr_opposite'] = pr_opposites
    return df

overall_acc_df = get_accuracy(overall_acc_df)
#overall_acc_df_shuffle = get_accuracy(overall_acc_df_shuffle)

#overall_acc_df_shuffle_save = overall_acc_df_shuffle.copy()
#overall_acc_df_shuffle['session'] = overall_acc_df_shuffle['session'].astype(int)
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

ticklab_size = 17
axlab_size = 20
subplots_margins = {'wspace': 0.05, 'hspace': 0.1, 'left': 0.1, 'right': 0.9}



#%% plot split point vs accuracy
#make a figure with 3 columns as subplots
fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True, sharex=True)
for session in [1, 2, 3]:
    ax = axs[session-1]
    sample_group = overall_acc_df.loc[overall_acc_df['session'] == session].reset_index(drop=True)
    sample_acc = sample_group['pr_template'].to_numpy()
    sample_opp = sample_group['pr_opposite'].to_numpy()

    #shuffle_group = overall_acc_df_shuffle.loc[overall_acc_df_shuffle['session'] == session].reset_index(drop=True)
    #shuffle_acc = shuffle_group['pr_template'].to_numpy()

    sample_splits = sample_group['split'].values
    #shuffle_splits = shuffle_group['split'].values

    #make the line fatter
    #sns.lineplot(x=shuffle_splits, y=shuffle_acc, ax=ax, color='gray', linewidth=2)
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
plt.subplots_adjust(bottom=0.25, **subplots_margins)  # Adjust the bottom
axs[1].legend(['opposite template', 'matching template'],
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

#%% calculate best splits
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


single_template = overall_acc_df.loc[overall_acc_df['split'] == 0].reset_index(drop=True)
single_template['group'] = 'single\ntemplate'
single_template['iter'] = -1

# single_template_shuffle = overall_acc_df_shuffle.loc[overall_acc_df_shuffle['split'] == max_split].reset_index(drop=True)
# single_template_shuffle['group'] = 'single\ntemplate'

#concatenate all the dfs into one long one
bar_df = pd.concat([best_splits_df, single_template])
# bar_df_shuffle = pd.concat([best_splits_shuffle_df, single_template_shuffle])

#%% plot bar graph of split vs single template accuracy
def get_stars(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval <= 0.05:
        return '*'


#import ols
import pingouin as pg
#perform a repeated measures ANOVA
#between the groups, perform an ANOVA, taking into account exp_name, taste, and session
bar_df['exp_name_taste'] = bar_df['exp_name'] + '_' + bar_df['taste']

tab_blue = sns.color_palette('tab10')[0]
#set matplotlib text to larger font
plt.rcParams.update({'font.size': 20})

#%% make bar graphs of the differences
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

    #shuffle_group = bar_df_shuffle.loc[(bar_df_shuffle['session'] == session)].reset_index(drop=True)
    #shuffle_group = shuffle_group.groupby(['group', 'iter']).mean().reset_index()
    #shuffle upper 95% confidence interval
    conf_level = (0.05/3)
    #shuffle_upper = shuffle_group.groupby('group')['pr_template'].quantile(1-conf_level).reset_index()
    #shuffle_lower = shuffle_group.groupby('group')['pr_template'].quantile(0).reset_index()
    # x = shuffle_upper['group']
    # upper = shuffle_upper['pr_template']
    # lower = shuffle_lower['pr_template']
    sns.barplot(x='group', y='pr_template', data=sample_group, ax=ax, ci=95, color='White', edgecolor=tab_blue, capsize=0.5, linewidth=2)
    # ax.bar(x=x, height=upper-lower, bottom=lower, color='gray', alpha=1, linewidth=2)
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
    plt.subplots_adjust(bottom=0.2, **subplots_margins)
plt.show()
#save
plt.savefig(os.path.join(save_dir, 'split_trial_accuracy_bars.svg'))
plt.savefig(os.path.join(save_dir, 'split_trial_accuracy_bars.png'))


best_split_d1 = best_splits_df.loc[best_splits_df['session'] == 1]


#%% make a bar graph of pre vs post split

#perform a repeated measures ANOVA
#between the groups, perform an ANOVA, taking into account exp_name, taste, and session
bardf2 = bar_df[['exp_name_taste','session', 'taste', 'split', 'pr_pre', 'pr_post']]
#remove all split == 29
bardf2 = bardf2.loc[bardf2['split'] != 0]
bardf2 = bardf2.melt(id_vars=['exp_name_taste', 'session', 'taste', 'split'], value_vars=['pr_pre', 'pr_post'], var_name='group', value_name='pr_template')

# bar_df_shuffle2 = bar_df_shuffle[['exp_name', 'taste', 'session', 'split', 'pr_pre', 'pr_post', 'iter']]
# bar_df_shuffle2 = bar_df_shuffle2.loc[bar_df_shuffle2['split'] != 29]
# bar_df_shuffle2 = bar_df_shuffle2.melt(id_vars=['exp_name', 'taste', 'session', 'split', 'iter'], value_vars=['pr_pre', 'pr_post'], var_name='group', value_name='pr_template')

tab_blue = sns.color_palette('tab10')[0]
#set matplotlib text to larger font
plt.rcParams.update({'font.size': 20})
#get the 95th percentile highest value of bar_df2['pr_template']
max_val = bardf2['pr_template'].quantile(0.95)
groups = bardf2['group'].unique().tolist()
fig, axs = plt.subplots(1, 3, figsize=(10, 4), sharey=True, sharex=True)
for i, session in enumerate([1, 2, 3]):
    ax = axs[i]
    sample_group = bardf2.loc[(bardf2['session'] == session)].reset_index(drop=True)
    aov = pg.rm_anova(data=sample_group, dv='pr_template', within='group', subject='exp_name_taste')
    pw = pg.pairwise_ttests(data=sample_group, dv='pr_template', within='group', subject='exp_name_taste')
    print(pw)
    #between the groups, perform an ANOVA, taking into account exp_name, taste, and session
    #
    # shuffle_group = bar_df_shuffle2.loc[(bar_df_shuffle2['session'] == session)].reset_index(drop=True)
    # shuffle_group = shuffle_group.groupby(['group', 'iter']).mean().reset_index()
    #shuffle upper 95% confidence interval
    conf_level = (0.05/3)
    # shuffle_upper = shuffle_group.groupby('group')['pr_template'].quantile(1-conf_level).reset_index()
    # shuffle_lower = shuffle_group.groupby('group')['pr_template'].quantile(0).reset_index()
    # x = shuffle_upper['group']
    # upper = shuffle_upper['pr_template']
    # lower = shuffle_lower['pr_template']
    g = sns.barplot(x='group', y='pr_template', data=sample_group, ax=ax, ci=95, color='White', edgecolor=tab_blue, capsize=0.5, linewidth=2)#, order=['pr_pre', 'pr_post'])
    #ax.bar(x=x, height=upper-lower, bottom=lower, color='gray', alpha=1, linewidth=2)
    ax.set_ylim(0, 1)
    #if pw['p-unc'][0] < 0.05, add a horizontal line between the two groups and a star above it
    pval = aov['p-unc'][0]
    if pval < 0.05:
        #get the maximum value from sample_group
        stars = get_stars(pval)
        ax.plot([0, 1], [max_val, max_val], color='k', linewidth=2)
        ax.text(0.5, max_val+0.01, stars, fontsize=20, ha='center')
    ax.set_xlabel('')
    ax.set_title(f'Session {session}')
    #ax.set_xticklabels(['pre-\nsplit', 'post-\nsplit'], fontsize=15)
    if session == 1:
        ax.set_ylabel('pr(template)')
    else:
        ax.set_ylabel('')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, **subplots_margins)
plt.show()
#save
plt.savefig(os.path.join(save_dir, 'pre_post_split_bars.svg'))
plt.savefig(os.path.join(save_dir, 'pre_post_split_bars.png'))




#%% single-mode hmm analysis with trialwise regression
import trialwise_analysis as ta
proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
HA = ana.HmmAnalysis(proj)  # create a hmm analysis object
save_dir = HA.save_dir
folder_name = 'HMM_stereotypy_nonlinear_regression'
save_dir = os.path.join(save_dir, folder_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#################### analysis of gamma mode ####################
subject_col = 'exp_name'
group_cols = ['exp_group','session','taste']

avg_gamma_mode_df = HA.get_avg_gamma_mode(overwrite=False)
#refactor exp_group column in avg_gamma_mode_df from ['naive','suc_preexp'] to ['naive','sucrose preexposed']
avg_gamma_mode_df['exp_group'] = avg_gamma_mode_df['exp_group'].replace({'suc_preexp':'sucrose preexposed'})
avg_gamma_mode_df['session trial'] = avg_gamma_mode_df['session_trial']
avg_gamma_mode_df['session'] = avg_gamma_mode_df['time_group'].astype(int)
avg_gamma_mode_df['taste trial'] = avg_gamma_mode_df['taste_trial'].astype(int)
avg_gamma_mode_df['pr(mode trial)'] = avg_gamma_mode_df['pr(mode state)']

for trial_col in ['taste_trial']:
    df3, shuff = ta.preprocess_nonlinear_regression(avg_gamma_mode_df, subject_col=subject_col, group_cols=group_cols,
                                                         trial_col=trial_col, value_col='pr(mode state)', overwrite=False,
                                                         nIter=10000, ymin=0, ymax=1, save_dir=save_dir)
    df3['stereotypy'] = df3['pr(mode state)']
    #shuff['pr(mode trial)'] = shuff['pr(mode state)']
    ta.plotting_pipeline(df3, shuff, trial_col=trial_col, value_col='stereotypy', ymin=0, ymax=1, nIter=10000, save_dir=save_dir, flag='hmm_stereotypy')