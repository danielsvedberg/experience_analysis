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
import plot_size_utils as psu

#

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
rec_info = proj.rec_info.copy()  # get the rec_info table
rec_dirs = rec_info['rec_dir']
PA = ana.ProjectAnalysis(proj)
all_units, held_df = PA.get_unit_info(overwrite=True)

save_dir = PA.save_dir + '/double_mode_stereotypy_analysis_single_neuron'
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


def avg_diff_trials(rates):
    avg_firing_rate = np.mean(rates, axis=1)  # Neurons x Bins (average across trials)
    diff = rates - avg_firing_rate[:, np.newaxis, :]
    diff = diff.mean(axis=2) #average across bins within each neuron x trial
    return diff


def process_split(rec_dir, split=10, shuffle=False):
    df_list = []
    time_array, rate_array = h5io.get_psths(rec_dir)
    ts = 100
    te = 3000
    # get indexs of time_array where time_array is greater than ts and less than te
    i_e = (time_array >= ts) & (time_array <= te)
    time_array = time_array[i_e]
    for din, rate in rate_array.items():
        if din != 'dig_in_4':
            rate = rate[:, :, i_e]

            if shuffle:
                # if shuffle, shuffle the rate array along axis 1
                rate = rate[:, np.random.permutation(rate.shape[1]), :]

            pre_rate = rate[:, :split, :]
            post_rate = rate[:, split:, :]

            pre_euc_dist = avg_diff_trials(pre_rate)

            post_euc_dist = avg_diff_trials(post_rate)

            # bind pre and post euc_dist_mat
            if len(pre_euc_dist) == 0:
                euc_dist = post_euc_dist
            elif len(post_euc_dist) == 0:
                euc_dist = pre_euc_dist
            else:  # concantenate pre_euc_dist_mat and post_euc_dist_mat along axis 1
                euc_dist = np.concatenate((pre_euc_dist, post_euc_dist), axis=1)

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
    # get the 0th index, and indices 2 to -1
    splits = np.concatenate((splits[:1], splits[2:-1]))
    dfs = [process_split(rec_dir, split, shuffle) for split in splits]
    dfs = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
    return dfs


# make the shuffle
def iter_shuffle(rec_dirs, niter=2000):
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
#final_dfs = Parallel(n_jobs=num_cores)(delayed(process_rec_dir)(rec_dir) for rec_dir in rec_dirs)
# Concatenate all resulting data frames into one
#final_df = pd.concat(final_dfs, ignore_index=True)
# save to pickle
#final_df.to_pickle(save_dir + '/double_mode_stereotypy_analysis.pkl')

#shuff_dfs = iter_shuffle(rec_dirs, niter=2000)
#shuff_dfs.to_pickle(save_dir + '/double_mode_euc_dist_stereotypy_shuffle.pkl')

######################################################################################################################
rec_info = proj.rec_info.copy()
rec_info = rec_info[['exp_name', 'exp_group', 'rec_num', 'rec_dir']]
# rename rec num to session
rec_info = rec_info.rename(columns={'rec_num': 'session'})

# get final df from pickle and preprocess###############################################################################
final_df = pd.read_pickle(save_dir + '/double_mode_stereotypy_analysis.pkl')
final_df = pd.merge(final_df, rec_info, on='rec_dir')
# remove split == 30
final_df = final_df.loc[final_df['split'] != 30]
# for each combination of rec_dir and channel, find the maximum trial. The split cannot be greater than the maximum trial - 2
max_trials = final_df.groupby(['rec_dir', 'channel'])['taste_trial'].max().reset_index()
max_trials = max_trials.rename(columns={'taste_trial': 'max_trial'})
final_df = pd.merge(final_df, max_trials, on=['rec_dir', 'channel'])
final_df = final_df.loc[final_df['max_trial'] - final_df['split'] > 1]
final_df = final_df.loc[final_df['split'] != 2]
final_df['Z(Euc dist)'] = final_df.groupby(['rec_dir', 'channel'])['euclidean_distance'].transform(
    lambda x: (x - x.mean()) / x.std())
final_df['Var Scaled Euc Dist'] = final_df.groupby(['rec_dir', 'channel'])['euclidean_distance'].transform(
    lambda x: x / x.std())


# preprocess the shuffle ################################################################################################
shuff_dfs = pd.read_pickle(save_dir + '/double_mode_euc_dist_stereotypy_shuffle.pkl')
#just take the first 1000 iterations
shuff_dfs = shuff_dfs.loc[shuff_dfs['iternum'] < 1000]

final_shuff_df = pd.merge(shuff_dfs, rec_info, on='rec_dir')
# remove split == 30
final_shuff_df = final_shuff_df.loc[final_shuff_df['split'] != 30]
# for each combination of rec_dir and channel, find the maximum trial. The split cannot be greater than the maximum trial - 1
max_trials = final_shuff_df.groupby(['rec_dir', 'channel'])['taste_trial'].max().reset_index()
max_trials = max_trials.rename(columns={'taste_trial': 'max_trial'})
final_shuff_df = pd.merge(final_shuff_df, max_trials, on=['rec_dir', 'channel'])
final_shuff_df = final_shuff_df.loc[final_shuff_df['max_trial'] - final_shuff_df['split'] > 1]
final_shuff_df = final_shuff_df.loc[final_shuff_df['split'] != 2]
final_shuff_df['Z(Euc dist)'] = final_shuff_df.groupby(['rec_dir', 'channel', 'iternum'])[
    'euclidean_distance'].transform(lambda x: (x - x.mean()) / x.std())
final_shuff_df['Var Scaled Euc Dist'] = final_shuff_df.groupby(['rec_dir', 'channel', 'iternum'])[
    'euclidean_distance'].transform(lambda x: x / x.std())

# first, get the average shuffled euclidean distance for each split:
iteravgshuff = final_shuff_df.groupby(
    ['rec_dir', 'exp_name', 'channel', 'session', 'exp_group', 'split', 'taste_trial']).mean().reset_index()
# then, introduce a column to avg_df that is the shuffled euclidean distance for that split, via merge
final_df = pd.merge(final_df, iteravgshuff, on=['rec_dir', 'exp_name', 'channel', 'session', 'exp_group', 'split', 'taste_trial'],
                    suffixes=('', '_shuff'))
final_shuff_df = pd.merge(final_shuff_df, iteravgshuff,
                          on=['rec_dir', 'exp_name', 'channel', 'session', 'exp_group', 'split', 'taste_trial'],
                          suffixes=('', '_shuff'))


final_df['euc_dist_shuffnorm'] = final_df['euclidean_distance']/final_df['euclidean_distance_shuff']
final_df['varscale_euc_dist_shuffnorm'] = final_df['Var Scaled Euc Dist']/final_df['Var Scaled Euc Dist_shuff']
final_df['Z(euc_dist)_shuffnorm)'] = final_df['Z(Euc dist)']/final_df['Z(Euc dist)_shuff']
final_shuff_df['euc_dist_shuffnorm'] = final_shuff_df['euclidean_distance']/final_shuff_df['euclidean_distance_shuff']
final_shuff_df['varscale_euc_dist_shuffnorm'] = final_shuff_df['Var Scaled Euc Dist'] / final_shuff_df['Var Scaled Euc Dist_shuff']
final_shuff_df['Z(euc_dist)_shuffnorm)'] = final_shuff_df['Z(Euc dist)'] / final_shuff_df['Z(Euc dist)_shuff']

unsplit = final_df.loc[final_df['split'] == 0]
final_df = final_df.loc[final_df['split'] != 0]

unsplit_shuff = final_shuff_df.loc[final_shuff_df['split'] == 0]
final_shuff_df = final_shuff_df.loc[final_shuff_df['split'] != 0]

# average the euclidean distance for each animal, taste, split, and session
avg_df = final_df.groupby(['rec_dir','exp_name', 'channel', 'split', 'session', 'exp_group']).mean().reset_index()
avg_shuff_df = final_shuff_df.groupby(
    ['rec_dir', 'exp_name', 'channel', 'split', 'session', 'exp_group', 'iternum']).mean().reset_index()

final_df['trial_group'] = final_df['trial_group'].replace('1-30', 'combined')
# refactor all entries in splitside from 'post' to 'post\nsplit' and 'pre' to 'pre\nsplit'
final_df['splitside'] = final_df['splitside'].replace('post', 'post\nsplit')
final_df['splitside'] = final_df['splitside'].replace('pre', 'pre\nsplit')
final_df['template'] = 'two\ntemplate'
unsplit['splitside'] = 'single\ntemplate'
unsplit['template'] = 'single\ntemplate'

final_df = final_df.reset_index(drop=True)
avg_df = avg_df.reset_index(drop=True)


def get_sig_stars(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return ''

def get_best_splits(yvar='euc_dist_shuffnorm'):
    best_splits = avg_df.loc[
        avg_df.groupby(['exp_name', 'session', 'exp_group', 'channel'])[yvar].idxmin()]
    best_splits = best_splits[['rec_dir', 'channel', 'split']]
    # get only the rows of final_df that have the corresponding best splits
    best_splits_df = pd.merge(final_df, best_splits, on=['rec_dir', 'channel', 'split'])

    best_splits_shuff = avg_shuff_df.loc[
        avg_shuff_df.groupby(['exp_name', 'session', 'exp_group', 'channel', 'iternum'])[yvar].idxmin()]
    best_splits_shuff = best_splits_shuff[['rec_dir', 'channel', 'split', 'iternum']]
    best_shuff = pd.merge(final_shuff_df, best_splits_shuff, on=['rec_dir', 'channel', 'split', 'iternum'])

    return best_splits_df, best_shuff

sbcolors = sns.color_palette()
fontsize = 8
axticksize = 8
ci = (1 - ((0.05/2) / 24))

def plot_each_split(yvar = 'euc_dist_shuffnorm', ylab = 'Euc. dist. - shuff'):

    fig, axs = plt.subplots(1, 1, figsize=(3, 3), sharey=True, sharex=True)
    for i, session in enumerate([1]):
        ax = axs
        for exp_group in ['naive']:
            df = final_df.loc[(final_df['exp_group'] == exp_group) & (final_df['session'] == session)]
            df = df.groupby(['exp_name', 'session', 'channel', 'split']).mean().reset_index()
            df['split'] = df['split'] + 0.5
            shuff_df = final_shuff_df.loc[
                (final_shuff_df['exp_group'] == exp_group) & (final_shuff_df['session'] == session)]
            shuff_df = shuff_df.groupby(
                ['session', 'split', 'iternum']).mean().reset_index()
            shuff_df['split'] = shuff_df['split'] + 0.5

            trial_min = df['split'].min()
            trial_max = df['split'].max()

            unsplit_df = unsplit.loc[(unsplit['exp_group'] == exp_group) & (unsplit['session'] == session)]
            unsplit_df = unsplit_df.groupby(['exp_name', 'session', 'channel']).mean().reset_index()
            boots = []
            nboot = 10000
            for _ in range(nboot):
                sample = unsplit_df.sample(frac=1, replace=True)
                boots.append(sample[yvar].mean())
            unsplit_yval = np.mean(boots)
            unsplit_yci_lower = np.percentile(boots, 2.5)
            unsplit_yci_upper = np.percentile(boots, 97.5)

            ax.fill_between([trial_min, trial_max], unsplit_yci_lower, unsplit_yci_upper, color=sbcolors[0], alpha=0.3)
            # add a horizontal line at the value of yvar for unsplit_df
            ax.hlines(unsplit_yval, trial_min, trial_max, color=sbcolors[0], linestyle='-.')

            # make a seaborn lineplot
            #sns.lineplot(x='split', y='euc_dist_splitzero-shuff', data=df, ax=ax, linestyle='--', color=sbcolors[0])
            shuff_lower = shuff_df.groupby('split')[yvar].quantile(1-ci).reset_index()
            shuff_upper = shuff_df.groupby('split')[yvar].quantile(ci).reset_index()
            #plot the shuff_lower and shuff_upper as a shaded region
            ax.fill_between(shuff_lower['split'], shuff_lower[yvar], shuff_upper[yvar], color='gray', alpha=0.3)

            sns.lineplot(x='split', y=yvar, data=df, ax=ax)
            sns.lineplot(x='split', y=yvar, data=shuff_df, ax=ax, color='black', alpha=0.1, linewidth=1)
            #for each split, get the 2.5th percentile


            #get the x value where avg euc_dist_shuffnorm is minimum for df
            avgdf = df.groupby('split').mean().reset_index()
            min_split = avgdf.loc[avgdf['euc_dist_shuffnorm'].idxmin()]['split']
            #get the 95% confidence interval of the minimum split in df by resampling
            boot_mins = []
            n_boot = 10000
            for _ in range(n_boot):
                sample = df.sample(frac=1, replace=True)
                avgsample = sample.groupby('split').mean().reset_index()
                boot_mins.append(avgsample.loc[avgsample[yvar].idxmin()]['split'])
            boot_ci_lower = np.percentile(boot_mins, 2.5)
            boot_ci_upper = np.percentile(boot_mins, 97.5)

            #plot the boot_ci_lower and boot_ci_upper as vertical dashed lines
            plot_height = ax.get_ylim()[0]
            ax.axvline(boot_ci_lower, color=sbcolors[0], linestyle=':')
            ax.axvline(boot_ci_upper, color=sbcolors[0], linestyle=':')
            #plot the min_split as a vertical solid line
            ax.axvline(min_split, color=sbcolors[0], linestyle='--')

            ax.set_title('Session ' + str(session), fontsize=fontsize)
            ax.set_xlabel('Split-trial', fontsize=fontsize)
            ax.set_ylabel(ylab, fontsize=fontsize)
            ax.tick_params(axis='both', which='major', labelsize=axticksize)
            ax.set_xticks([5, 10, 15, 20, 25])
    plt.tight_layout()
    fig = psu.adjust_figure_for_panel_size_auto(fig, panel_width=1, panel_height=1)
    fn = '/double_mode_dist_vs_split_' + yvar
    plt.savefig(save_dir + fn + '.png')
    plt.savefig(save_dir + fn + '.svg')

plot_each_split(yvar = 'euc_dist_shuffnorm', ylab = 'Euc. dist. / shuff')
#plot_each_split(yvar = 'Z(euc_dist)_shuffnorm)', ylab = 'Z(Euc. dist. / shuff)')
plot_each_split(yvar = 'Z(Euc dist)', ylab = 'Z(Euc. dist.)')
#plot_each_split(yvar = 'euc_dist_zeronorm', ylab = 'unsplit/split\nEuc. dist.')
#plot_each_split(yvar = 'Var Scaled Euc Dist', ylab = 'Var. scaled\nEuc. dist.')
#plot_each_split(yvar = 'varscale_euc_dist_shuffnorm', ylab = 'Var. scaled\nEuc. dist. / shuff')

#unsplit still good here

# now make a histogram of the best split for each session
def plot_split_hist(yvar, sessions=None):
    if sessions is None:
        sessions = [1, 3]
    n_sessions = len(sessions)

    best_splits_df, best_shuff = get_best_splits(yvar=yvar)

    fig, axs = plt.subplots(1, n_sessions, figsize=(5, 3), sharey=True, sharex=True)
    for i, session in enumerate(sessions):
        for exp_group in ['naive']:
            df = best_splits_df.loc[(best_splits_df['exp_group'] == exp_group) & (best_splits_df['session'] == session)]
            df = df.groupby(['exp_name', 'session', 'channel', 'split']).mean().reset_index()

            shuffdf = best_shuff.loc[(best_shuff['exp_group'] == exp_group) & (best_shuff['session'] == session)]
            shuffdf = shuffdf.groupby(['exp_name', 'session', 'channel', 'split', 'iternum']).mean().reset_index()

            # join the two dataframes
            df['split'] = df['split'] + 0.5
            shuffdf['split'] = shuffdf['split'] + 0.5

            #use bootstrapping to get the 95% confidence interval of split in df
            boot_meds = []
            n_boot = 10000
            for _ in range(n_boot):
                sample = df['split'].sample(frac=1, replace=True)
                boot_meds.append(sample.median())
            boot_ci_lower = np.percentile(boot_meds, 2.5)
            boot_ci_upper = np.percentile(boot_meds, 97.5)
            boot_ci_median = np.percentile(boot_meds, 50)

            if n_sessions == 1:
                ax = axs
            else:
                ax = axs[i]

            a = sns.histplot(ax=ax, data=df, x='split', stat='probability', alpha=0.5, color=sbcolors[0], binwidth=2, edgecolor='none')
            b = sns.histplot(shuffdf['split'], ax=ax, stat='probability', color='black', alpha=0.3, binwidth=2, edgecolor='none')

            plot_height = max(a.patches, key=lambda p: p.get_height()).get_height()

            ax.vlines(boot_ci_lower, 0, plot_height, color=sbcolors[0], linestyle='--')
            ax.vlines(boot_ci_upper, 0, plot_height, color=sbcolors[0], linestyle='--')
            ax.vlines(boot_ci_median, 0, plot_height, color=sbcolors[0], linestyle='-')

            ax.set_title('Session ' + str(session), fontsize=fontsize)
            ax.set_xlabel('best split-trial', fontsize=fontsize)
            ax.set_ylabel('fraction', fontsize=fontsize)
            ax.tick_params(axis='both', which='major', labelsize=axticksize)
            #set the x ticks to be 5, 10, 15, 20, 25
            ax.set_xticks([5, 10, 15, 20, 25])
            ax.set_xlim(3, 28)
    plt.tight_layout()
    fig = psu.adjust_figure_for_panel_size_auto(fig, panel_width=1, panel_height=1)
    filename = save_dir + '/double_mode_best_split_hist_' + yvar
    plt.savefig(filename + '.png')
    plt.savefig(filename + '.svg')

plot_split_hist(yvar='euc_dist_shuffnorm', sessions = [1])
plot_split_hist(yvar='Z(Euc dist)', sessions = [1])

#unsplit still good here

def plot_split_bars(unsplit_df, save_dir, yvar, ylab=None):

    best_split_df, best_shuff = get_best_splits(yvar=yvar)
    # for final_df, replace all occurences of '1-30' in 'trial_group' with 'combined'

    sbcolors = sns.color_palette()
    fontsize = 8
    axticksize = 8
    splitavos = []
    templavos = []
    import pingouin as pg
    fig1, axs1 = plt.subplots(1, 1, figsize=(5, 3), sharey=True)
    fig2, axs2 = plt.subplots(1, 1, figsize=(5, 3), sharey=True)
    for i, session in enumerate([1]):
        axtop = axs1
        axbot = axs2
        for exp_group in ['naive']:
            # get the df with the best split for each exp_name
            bestsplit = \
            best_split_df.loc[(best_split_df['exp_group'] == exp_group) & (best_split_df['session'] == session)]
            # get the rows of final_df where the exp_name and split match the best split
            unsplit_sub = unsplit_df.loc[(unsplit_df['exp_group'] == exp_group) & (unsplit_df['session'] == session)]

            # for splitavodf, get all rows where splitside == 'pre\nsplit' and 'post\nsplit'
            splitavodf = bestsplit
            # take the average of euclidean distance for each exp_name, channel, and splitside
            splitavodf = splitavodf.groupby(
                ['exp_name', 'channel', 'splitside', 'session', 'exp_group']).mean().reset_index()
            splitavodf['expXchannel'] = splitavodf['exp_name'] + splitavodf['channel'].astype(str)

            # perform a repeated measures anova
            splitavo = pg.rm_anova(data=splitavodf, dv=yvar, within='splitside', subject='expXchannel', detailed=True)
            #check if the 'p-unc' column exists in splitavo
            if 'p-unc' not in splitavo.columns:
                print('warning: p-unc column not found in splitavo')
                splitavos.append(np.nan)
                split_pval = np.nan
            else:
                split_pval = splitavo['p-unc'].values[0]
                splitavos.append(splitavo)

            # take the average of euclidean distance for each exp_name, channel, and template
            unsplit_sub['template'] = 'single\ntemplate'
            bestsplit['template'] = 'two\ntemplate'

            # concatenate df with templdf
            templdf = pd.concat([bestsplit, unsplit_sub], ignore_index=True)
            templdf = templdf.groupby(['exp_name', 'channel', 'template', 'session', 'exp_group']).mean().reset_index()
            templdf['expXchannel'] = templdf['exp_name'] + templdf['channel'].astype(str)

            templavo = pg.rm_anova(data=templdf, dv=yvar, within='template', subject='expXchannel', detailed=True)
            if 'p-unc' not in templavo.columns:
                print('warning: p-unc column not found in templavo')
                templavos.append('nan')
                templ_pval = np.nan
            else:
                templ_pval = templavo['p-unc'].values[0]
                templavos.append(templavo)

            sns.barplot(x='splitside', y=yvar, data=splitavodf, ax=axtop, order=['pre\nsplit', 'post\nsplit'],
                        fill=False, edgecolor='black', capsize=0.4, errwidth=1.5)
            sns.swarmplot(x='splitside', y=yvar, data=splitavodf, ax=axtop,
                          order=['pre\nsplit', 'post\nsplit'], color=sbcolors[0], alpha=0.4, size=3)


            sns.barplot(x='template', y=yvar, data=templdf, ax=axbot, order=['two\ntemplate', 'single\ntemplate'],
                        fill=False, edgecolor='black', capsize=0.4, errwidth=1.5)
            sns.swarmplot(x='template', y=yvar, data=templdf, ax=axbot,
                          order=['two\ntemplate', 'single\ntemplate'], color=sbcolors[0], alpha=0.4, size=3)


            # get the height of the upper whisker in the barplot
            max_split = splitavodf[yvar].max()
            max_templ = templdf[yvar].max()
            min_split = splitavodf[yvar].min()
            min_templ = templdf[yvar].min()
            split_range = max_split - min_split
            templ_range = max_templ - min_templ

            if session == 1:
                split_top = max_split + split_range * 0.2
                axtop.set_ylim(min_split - split_range*0.1, top=split_top)
                split_range = split_top - (min_split - split_range*0.1)

                templ_top = max_templ + templ_range * 0.2
                axbot.set_ylim(bottom=min_templ - templ_range * 0.1, top=templ_top)
                templ_range = templ_top - (min_templ - templ_range*0.1)


            if split_pval < 0.05:  # then draw a line between 0 and 1, and put a star above the line
                height = split_top - split_range*0.1
                axtop.plot([0, 1], [height, height], lw=2, color='black')
                stars = get_sig_stars(split_pval)
                axtop.text(0.5, split_top - split_range*0.15, stars, fontsize=fontsize*1.5, ha='center')
                # round the p value to 3 decimal places and print it
                #split_pval = round(split_pval, 4)
                #axtop.text(0.5, min_split, 'p = ' + str(split_pval), fontsize=fontsize, ha='center')


            if templ_pval < 0.05:
                height = templ_top - templ_range*0.1
                axbot.plot([0, 1], [height, height], lw=2, color='black')
                stars = get_sig_stars(templ_pval)
                axbot.text(0.5, templ_top - templ_range*0.15, stars, fontsize=fontsize*1.5, ha='center')
                # print the p value
                #templ_pval = round(templ_pval, 4)
                #axbot.text(0.5, min_templ, 'p = ' + str(templ_pval), fontsize=fontsize, ha='center')

            # take away the y axis label
            if session == 1:
                if ylab is None:
                    ylab = 'Euc. dist. / shuff'

                axtop.set_ylabel(ylab, fontsize=fontsize)
                # make the y axis labels larger
                axbot.set_ylabel(ylab, fontsize=fontsize)

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

    fig1 = psu.adjust_figure_for_panel_size_auto(fig1, panel_width=1, panel_height=1)
    fig2 = psu.adjust_figure_for_panel_size_auto(fig2, panel_width=1, panel_height=1)

    plt.show()
    # save the figures
    fig1.savefig(save_dir + '/' + yvar + 'double_mode_split_prevspost.png')
    fig1.savefig(save_dir + '/' + yvar + 'double_mode_split_prevspost.svg')
    fig2.savefig(save_dir + '/' + yvar + 'double_mode_split_latevssing.svg')
    fig2.savefig(save_dir + '/' + yvar + 'double_mode_split_latevssing.png')
    # save the anova results to a csv
    splitavosdf = pd.concat(splitavos, ignore_index=True)
    templavosdf = pd.concat(templavos, ignore_index=True)
    splitavosdf.to_csv(save_dir + '/' + yvar + 'double_mode_split_prevspost_anova.csv')
    templavosdf.to_csv(save_dir + '/' + yvar + 'double_mode_split_latevssing_anova.csv')

#plot_split_bars(final_df, best_split_df, save_dir, 'Z(euc_dist_zeronorm)')
#plot_split_bars(final_df, best_split_df, save_dir, 'Z(euc_dist_shuffnorm)')
plot_split_bars(final_df, save_dir, 'euc_dist_shuffnorm', ylab='Euc. dist. / shuffle') #just fucking trust me this is the right variable
plot_split_bars(final_df, save_dir, 'Z(Euc dist)', ylab='Z(Euc. dist.)')


#unsplit still good here
unsplit_save = unsplit.copy()
def plot_boxen_trialwise(unsplit_df, save_dir, ref_val = 0, yvar=None, ylab=None):
    if yvar is None:
        yvar = 'Z(Euc dist)'
    if ylab is None:
        ylab = yvar

    # for unsplit, make a histogram of the euclidean distances
    if yvar is 'Z(euc_dist)':
        unsplit_df[yvar] = unsplit_df.groupby(['rec_dir', 'channel'])['euclidean_distance'].transform(
            lambda x: (x - x.mean()) / x.std())

    unsplit['trial'] = unsplit['taste_trial'] + 1
    #require that minimum trial number is 1
    if unsplit_df['trial'].min() == 0:
        raise ValueError('taste_trial minimum is 0, expected 1')

    # for each grouping of session, exp_group, and taste trial, perform a one sample t-test against 0 for Z(euc_dist)
    unsplit_ttest = unsplit_df.groupby(['session', 'exp_group', 'trial']).apply(
        lambda x: stats.ttest_1samp(x[yvar], ref_val))
    unsplit_ttest = pd.DataFrame(unsplit_ttest.tolist(), index=unsplit_ttest.index, columns=['tstat', 'pval'])
    unsplit_ttest = unsplit_ttest.reset_index()
    # bonferroni correct for 30 comparisons
    unsplit_ttest['pval'] = unsplit_ttest['pval'] * 30

    sbcolors = sns.color_palette()
    fontsize = 14
    axticksize = 14
    sessions = [1]
    max_euc_dist = unsplit_df[yvar].max()
    min_euc_dist = unsplit_df[yvar].min()
    # plot a correlogram of the euclidean distances and trial number
    fig, axs = plt.subplots(1, 1, figsize=(10, 3), sharey=True, sharex=True)
    for i, session in enumerate(sessions):
        ax = axs
        for exp_group in ['naive']:
            df = unsplit_df.loc[(unsplit_df['exp_group'] == exp_group) & (unsplit_df['session'] == session)]
            pvaldf = unsplit_ttest.loc[(unsplit_ttest['exp_group'] == exp_group) & (unsplit_ttest['session'] == session)]
            sns.boxenplot(data=df, x='trial', y=yvar, ax=ax, color=sbcolors[0], linewidth=0.5,
                          showfliers=False, k_depth='full')
            # plot a horizontal line that is the average euclidean distance
            avg_euc_dist = df[yvar].mean()
            ax.axhline(avg_euc_dist, color='black', linestyle='--')

            yrange = max_euc_dist - min_euc_dist
            ax.set_ylim(min_euc_dist - yrange * 0.1, max_euc_dist + yrange * 0.3)

            # if the pvalue for the trial is less than 0.05, plot a star above the line
            for idx, row in pvaldf.iterrows():
                if row['pval'] < 0.05:
                    # get the number of stars to plot
                    stars = get_sig_stars(row['pval'])
                    # get the maximum value of df at the trial number
                    med = df.loc[df['trial'] == row['trial'], yvar].max()
                    # print the stars turned 90 degrees
                    ax.text(row['trial'] + 0.5, max_euc_dist, stars, fontsize=fontsize, ha='right', va='bottom',
                            rotation=90)
                    #TODO print table w pvalues
                    # print the p value below
                    #ax.text(row['trial'] - 0.7, max_euc_dist, 'p = ' + str(round(row['pval'], 4)), fontsize=9,
                    #        ha='center', va='top', rotation=-90)



            ax.set_title('Session ' + str(session), fontsize=fontsize)
            ax.set_xlabel('trial number', fontsize=fontsize)
            ax.tick_params(axis='both', which='major', labelsize=axticksize)
            # make the x axis labels every other label
            xlabs = ax.get_xticklabels()
            # set every other label to ''
            for label in xlabs:
                label.set_visible(False)
            for label in xlabs[::3]:
                label.set_visible(True)

            if i == 0:
                ax.set_ylabel(ylab, fontsize=fontsize)
            else:
                ax.set_ylabel('')
    plt.tight_layout()

    fig = psu.adjust_figure_for_panel_size_auto(fig, panel_width=3, panel_height=1)
    filename = save_dir + '/trialwise_boxenplot_' + yvar
    plt.savefig(filename + '.png')
    plt.savefig(filename + '.svg')

    #make a table of pvalues for each trial
    unsplit_ttest.to_csv(save_dir + '/trialwise_boxenplot_' + yvar + '_pvalues.csv')

    #adjust panel size

plot_boxen_trialwise(unsplit, save_dir, ref_val=1, yvar='euc_dist_shuffnorm', ylab='Euc. dist. / shuffle')
plot_boxen_trialwise(unsplit, save_dir, ref_val=0, yvar='Z(Euc dist)', ylab='Z(Euc. dist.)')