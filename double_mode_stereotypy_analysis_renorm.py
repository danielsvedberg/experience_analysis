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

            pre_euc_dist = euc_dist_trials(pre_rate)

            post_euc_dist = euc_dist_trials(post_rate)

            # bind pre and post euc_dist_mat
            if len(pre_euc_dist) == 0:
                euc_dist = post_euc_dist
            elif len(post_euc_dist) == 0:
                euc_dist = pre_euc_dist
            else:  # concantenate pre_euc_dist_mat and post_euc_dist_mat along axis 1
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
#drop all rows where session == 2
final_df = final_df[final_df['session'] != 2]
#drop all rows where exp_group is not 'naive'
final_df = final_df[final_df['exp_group'] == 'naive']
# remove split == 30
final_df = final_df.loc[final_df['split'] != 30]
# for each combination of rec_dir and channel, find the maximum trial. The split cannot be greater than the maximum trial - 2
max_trials = final_df.groupby(['rec_dir', 'channel'])['taste_trial'].max().reset_index()
max_trials = max_trials.rename(columns={'taste_trial': 'max_trial'})
final_df = pd.merge(final_df, max_trials, on=['rec_dir', 'channel'])
final_df = final_df.loc[final_df['max_trial'] - final_df['split'] > 1]
final_df = final_df.loc[final_df['split'] != 2]
final_df['iternum'] = -1

# preprocess the shuffle ################################################################################################
final_shuff_df = pd.read_pickle(save_dir + '/double_mode_euc_dist_stereotypy_shuffle.pkl')
final_shuff_df = pd.merge(final_shuff_df, rec_info, on='rec_dir')
#drop all rows where session == 2
final_shuff_df = final_shuff_df[final_shuff_df['session'] != 2]
#drop all rows where exp_group is not 'naive'
final_shuff_df = final_shuff_df[final_shuff_df['exp_group'] == 'naive']
# remove split == 30
final_shuff_df = final_shuff_df.loc[final_shuff_df['split'] != 30]
# for each combination of rec_dir and channel, find the maximum trial. The split cannot be greater than the maximum trial - 1
max_trials = final_shuff_df.groupby(['rec_dir', 'channel'])['taste_trial'].max().reset_index()
max_trials = max_trials.rename(columns={'taste_trial': 'max_trial'})
final_shuff_df = pd.merge(final_shuff_df, max_trials, on=['rec_dir', 'channel'])
final_shuff_df = final_shuff_df.loc[final_shuff_df['max_trial'] - final_shuff_df['split'] > 1]
final_shuff_df = final_shuff_df.loc[final_shuff_df['split'] != 2]
#rearrange the columns of final_shuff_df to match final_df
final_shuff_df = final_shuff_df[final_df.columns]

#vertically stack final_df and final_shuff_df
final_shuff_df = pd.concat([final_shuff_df, final_df], ignore_index=True)

#calculate the z score of euclidean distance - shuffled euclidean distance for each rec_dir and channel
final_shuff_df['Z(Euc dist)'] = final_shuff_df.groupby(['rec_dir', 'channel'])['euclidean_distance'].transform(
    lambda x: (x - x.mean()) / x.std())
#for each rec_dir and channel, get the chi-squared distribution of euclidean distance
#chi_sq_df = final_shuff_df.groupby(['rec_dir', 'channel', 'split'])['euclidean_distance'].apply(
#    lambda x: stats.chi2(df=len(x) - 1, scale=np.var(x)).mean()).reset_index()

unsplit = final_shuff_df.loc[final_shuff_df['split'] == 0]
splitzero = unsplit.loc[unsplit['iternum'] == -1]
splitzero = splitzero.groupby(['exp_name', 'channel', 'session', 'exp_group']).mean().reset_index()

final_df = final_shuff_df.loc[final_shuff_df['iternum'] == -1]
#final_df = final_df.loc[final_df['split'] != 0]

unsplit_shuff = unsplit.loc[unsplit['iternum'] != -1]
splitzero_shuff = unsplit_shuff.groupby(['exp_name', 'channel', 'session', 'exp_group', 'iternum']).mean().reset_index()

final_shuff_df = final_shuff_df.loc[final_shuff_df['iternum'] != -1]
#final_shuff_df = final_shuff_df.loc[final_shuff_df['split'] != 0]

# average the euclidean distance for each animal, taste, split, and session
avg_df = final_df.groupby(['exp_name', 'split', 'session', 'channel', 'exp_group']).mean().reset_index()
avg_shuff_df = final_shuff_df.groupby(
    ['exp_name', 'split', 'session', 'channel', 'exp_group', 'iternum']).mean().reset_index()

#redefine save_dir to be save_dir + '/double_mode_stereotypy_analysis_renorm'
save_dir = save_dir + '/double_mode_stereotypy_analysis_renorm'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print('save dir' + save_dir + ' created')

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
ci = (1 - (0.05 / 25)) * 100
fig, axs = plt.subplots(1, 2, figsize=(5, 3), sharey=True, sharex=True)
for i, session in enumerate([1, 3]):
    ax = axs[i]
    for exp_group in ['naive']:
        df = avg_df.loc[(avg_df['exp_group'] == exp_group) & (avg_df['session'] == session)]
        shuff_df = final_shuff_df.loc[(final_shuff_df['exp_group'] == exp_group) & (final_shuff_df['session'] == session)]

        #calculate the ci confidence interval of the shuffled data for each split
        ci_lower = []
        ci_upper = []
        splits = df['split'].unique()
        for split in splits:
            split_data = shuff_df.loc[shuff_df['split'] == split]
            #bootstrap the confidence interval, resample 1000 times and average by iternum
            bootstrap_split_data = []
            n_bootstrap = 5
            for _ in range(n_bootstrap):
                sample = split_data.sample(frac=1, replace=True)
                sample = sample.groupby('iternum').mean().reset_index()
                bootstrap_split_data.append(sample['Z(Euc dist)'].values)
            lower = np.percentile(bootstrap_split_data, (100 - ci) / 2)
            upper = np.percentile(bootstrap_split_data, 100 - (100 - ci) / 2)
            ci_lower.append(lower)
            ci_upper.append(upper)
        ci_lower = np.array(ci_lower)
        ci_upper = np.array(ci_upper)

        # make a seaborn lineplot
        ax.fill_between(splits, ci_lower, ci_upper, color='grey', alpha=0.3)
        sns.lineplot(x='split', y='Z(Euc dist)', data=df, ax=ax, ci=95)
        sns.lineplot(x='split', y='Z(Euc dist)', data=shuff_df, ax=ax, color='black', alpha=0.1, ci=None)


        ax.set_title('Session ' + str(session), fontsize=fontsize)
        ax.set_xlabel('Split-trial', fontsize=fontsize)
        ax.set_ylabel('Euc. dist. (AU)', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=axticksize)
plt.tight_layout()
fig = psu.adjust_figure_for_panel_size_auto(fig, panel_width=1, panel_height=1)
plt.savefig(save_dir + '/double_mode_dist_vs_split.png')
plt.savefig(save_dir + '/double_mode_dist_vs_split.svg')


best_split_df = avg_df.loc[
    avg_df.groupby(['exp_name', 'session', 'exp_group', 'channel'])['euclidean_distance'].idxmin()]

best_shuff = avg_shuff_df.loc[
    avg_shuff_df.groupby(['exp_name', 'session', 'exp_group', 'channel', 'iternum'])['euclidean_distance'].idxmin()]

# now make a histogram of the best split for each session
fig, axs = plt.subplots(1, 2, figsize=(5, 3), sharey=True, sharex=True)
for i, session in enumerate([1, 3]):
    for exp_group in ['naive']:
        df = best_split_df.loc[(best_split_df['exp_group'] == exp_group) & (best_split_df['session'] == session)]
        shuffdf = best_shuff.loc[(best_shuff['exp_group'] == exp_group) & (best_shuff['session'] == session)]
        # join the two dataframes
        # df['type'] = 'real'
        # shuffdf['type'] = 'shuffled'
        # df = pd.concat([df, shuffdf], ignore_index=True)
        df['split'] = df['split'] + 0.5
        shuffdf['split'] = shuffdf['split'] + 0.5

        #get the 95% confidence interval of split in df and shuffdf

        shuff_ci_lower = np.percentile(shuffdf['split'], 2.5)
        shuff_ci_upper = np.percentile(shuffdf['split'], 97.5)
        shuff_ci_median = np.percentile(shuffdf['split'], 50)
        #use bootstrapping to get the 95% confidence interval of split in df
        boot_meds = []
        n_boot = 10000
        for _ in range(n_boot):
            sample = df['split'].sample(frac=1, replace=True)
            boot_meds.append(sample.median())
        boot_ci_lower = np.percentile(boot_meds, 2.5)
        boot_ci_upper = np.percentile(boot_meds, 97.5)
        boot_ci_median = np.percentile(boot_meds, 50)

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
        ax.set_xlim(0, 28)
plt.tight_layout()
fig = psu.adjust_figure_for_panel_size_auto(fig, panel_width=1, panel_height=1)
plt.savefig(save_dir + '/double_mode_best_split_hist.png')
plt.savefig(save_dir + '/double_mode_best_split_hist.svg')


unsplit['splitside'] = 'single\ntemplate'
unsplit['template'] = 'single\ntemplate'
# for final_df, replace all occurences of '1-30' in 'trial_group' with 'combined'
final_df['trial_group'] = final_df['trial_group'].replace('1-30', 'combined')
# refactor all entries in splitside from 'post' to 'post\nsplit' and 'pre' to 'pre\nsplit'
final_df['splitside'] = final_df['splitside'].replace('post', 'post\nsplit')
final_df['splitside'] = final_df['splitside'].replace('pre', 'pre\nsplit')
final_df['template'] = 'two\ntemplate'

final_shuff_df['trial_group'] = final_shuff_df['trial_group'].replace('1-30', 'combined')
# refactor all entries in splitside from 'post' to 'post\nsplit' and 'pre' to 'pre\nsplit'
final_shuff_df['splitside'] = final_shuff_df['splitside'].replace('post', 'post\nsplit')
final_shuff_df['splitside'] = final_shuff_df['splitside'].replace('pre', 'pre\nsplit')
final_shuff_df['template'] = 'two\ntemplate'

def plot_split_bars(final_df, shuff_df, best_split_df, save_dir, yvar):
    sbcolors = sns.color_palette()
    fontsize = 8
    axticksize = 8
    splitavos = []
    templavos = []
    import pingouin as pg
    fig1, axs1 = plt.subplots(1, 2, figsize=(5, 3), sharey=True)
    fig2, axs2 = plt.subplots(1, 2, figsize=(5, 3), sharey=True)
    for i, session in enumerate([1, 3]):
        axtop = axs1[i]
        axbot = axs2[i]
        for exp_group in ['naive']:
            # get the df with the best split for each exp_name
            best_split = \
            best_split_df.loc[(best_split_df['exp_group'] == exp_group) & (best_split_df['session'] == session)][
                ['exp_name', 'split']]
            # get the rows of final_df where the exp_name and split match the best split
            df = final_df.loc[(final_df['exp_group'] == exp_group) & (final_df['session'] == session)]
            df = pd.merge(df, best_split, on=['exp_name', 'split'])

            bsdf = \
            best_shuff.loc[(best_shuff['exp_group'] == exp_group) & (best_shuff['session'] == session)][
                ['exp_name', 'split', 'iternum']]

            sdf = final_shuff_df.loc[(final_shuff_df['exp_group'] == exp_group) & (final_shuff_df['session'] == session)]
            sdf = pd.merge(sdf, bsdf, on=['exp_name', 'split', 'iternum'])

            df['AnTaste'] = df['exp_name'] + df['channel'].astype(str)
            sdf['AnTaste'] = sdf['AnTaste'] + sdf['channel'].astype(str)

            # for splitavodf, get all rows where splitside == 'pre\nsplit' and 'post\nsplit'
            splitavodf = df.loc[(df['splitside'] == 'pre\nsplit') | (df['splitside'] == 'post\nsplit')]

            # take the average of euclidean distance for each exp_name, channel, and splitside
            splitavodf = splitavodf.groupby(
                ['exp_name', 'channel', 'splitside', 'session', 'exp_group']).mean().reset_index()
            splitavodf['expXchannel'] = splitavodf['exp_name'] + splitavodf['channel'].astype(str)

            # take the average of euclidean distance for each exp_name, channel, and template
            templdf = df.copy()
            templdf['template'] = 'single\ntemplate'
            templdf['euc_dist_shuffnorm'] = templdf['euclidean_distance_splitzero'] - templdf[
                'euclidean_distance_shuff']
            # concatenate df with templdf
            templdf = pd.concat([templdf, df], ignore_index=True)
            templdf = templdf.groupby(['exp_name', 'channel', 'template', 'session', 'exp_group']).mean().reset_index()

            templdf['expXchannel'] = templdf['exp_name'] + templdf['channel'].astype(str)
            # perform a repeated measures anova
            splitavo = pg.rm_anova(data=splitavodf, dv=yvar, within=['splitside'], subject='expXchannel', detailed=True)
            #check if the 'p-unc' column exists in splitavo
            if 'p-unc' not in splitavo.columns:
                print('warning: p-unc column not found in splitavo')
                splitavos.append(np.nan)
                templ_pval = np.nan
            else:
                split_pval = splitavo['p-unc'].values[0]
                splitavos.append(splitavo)

            templavo = pg.rm_anova(data=templdf, dv=yvar, within='template', subject='expXchannel', detailed=True)
            if 'p-unc' not in templavo.columns:
                print('warning: p-unc column not found in templavo')
                templavos.append(np.nan)
                templ_pval = np.nan
            else:
                templ_pval = templavo['p-unc'].values[0]
                templavos.append(templavo)

            ci_lower = []
            ci_upper = []
            for phase in ['pre\nsplit', 'post\nsplit']:
                dat = sdf.loc[sdf['splitside' == phase]]
                dat = dat.groupby('iternum').mean()
                #get the 95% confidence interval
                lower = np.percentile(dat[yvar], (100 - ci) / 2)
                upper = np.percentile(dat[yvar], 100 - (100 - ci) / 2)
                plt.plot(x=phase, y=)



            sns.swarmplot(x='splitside', y=yvar, data=splitavodf, ax=axtop,
                          order=['pre\nsplit', 'post\nsplit'], color=sbcolors[0], alpha=0.5)
            sns.barplot(x='splitside', y=yvar, data=splitavodf, ax=axtop, order=['pre\nsplit', 'post\nsplit'],
                        fill=False, edgecolor='black', capsize=0.4)


            sns.swarmplot(x='template', y=yvar, data=templdf, ax=axbot,
                          order=['two\ntemplate', 'single\ntemplate'], color=sbcolors[0], alpha=0.5)
            sns.barplot(x='template', y=yvar, data=templdf, ax=axbot, order=['two\ntemplate', 'single\ntemplate'],
                        fill=False, edgecolor='black', capsize=0.4)

            # get the height of the upper whisker in the barplot
            max_split = splitavodf[yvar].max()
            max_templ = templdf[yvar].max()
            min_split = splitavodf[yvar].min()
            min_templ = templdf[yvar].min()
            # set the y axis top limit for both plots to 1
            # axtop.set_ylim(top=1)
            # axbot.set_ylim(top=1)
            if split_pval < 0.05:  # then draw a line between 0 and 1, and put a star above the line
                height = max_split * 1.1
                axtop.plot([0, 1], [height, height], lw=2, color='black')
                stars = get_sig_stars(split_pval)
                axtop.text(0.5, height * 0.8, stars, fontsize=fontsize, ha='center')
                # round the p value to 3 decimal places and print it
                split_pval = round(split_pval, 4)
                axtop.text(0.5, min_split, 'p = ' + str(split_pval), fontsize=fontsize, ha='center')
            if templ_pval < 0.05:
                height = max_templ * 1.1
                axbot.plot([0, 1], [height, height], lw=2, color='black')
                stars = get_sig_stars(templ_pval)
                axbot.text(0.5, height * 0.8, stars, fontsize=fontsize, ha='center')
                # print the p value
                templ_pval = round(templ_pval, 4)
                axbot.text(0.5, min_templ, 'p = ' + str(templ_pval), fontsize=fontsize, ha='center')

            # take away the y axis label
            if session == 1:
                axtop.set_ylabel('Euc. dist. - shuff', fontsize=fontsize)
                # make the y axis labels larger
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

plot_split_bars(final_df, best_split_df, save_dir, 'Z(Euc dist)')
#plot_split_bars(final_df, best_split_df, save_dir, 'Z(euc_dist_shuffnorm)')
#plot_split_bars(final_df, best_split_df, save_dir, 'euc_dist_shuffnorm') #just fucking trust me this is the right variable


# for unsplit, make a histogram of the euclidean distances
unsplit['Z(euc_dist)'] = unsplit.groupby(['rec_dir', 'channel'])['euclidean_distance'].transform(
    lambda x: (x - x.mean()) / x.std())
unsplit['taste_trial'] = unsplit['taste_trial'] + 1
# for each grouping of session, exp_group, and taste trial, perform a one sample t-test against 0 for Z(euc_dist)
unsplit_ttest = unsplit.groupby(['session', 'exp_group', 'taste_trial']).apply(
    lambda x: stats.ttest_1samp(x['Z(euc_dist)'], 0))
unsplit_ttest = pd.DataFrame(unsplit_ttest.tolist(), index=unsplit_ttest.index, columns=['tstat', 'pval'])
unsplit_ttest = unsplit_ttest.reset_index()
# bonferroni correct for 30 comparisons
unsplit_ttest['pval'] = unsplit_ttest['pval'] * 30

sbcolors = sns.color_palette()
fontsize = 14
axticksize = 14
sessions = [1, 3]
max_euc_dist = unsplit['Z(euc_dist)'].max()
min_euc_dist = unsplit['Z(euc_dist)'].min()
# plot a correlogram of the euclidean distances and trial number
fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharey=True, sharex=True)
for i, session in enumerate(sessions):
    ax = axs[i]
    for exp_group in ['naive']:
        df = unsplit.loc[(unsplit['exp_group'] == exp_group) & (unsplit['session'] == session)]
        pvaldf = unsplit_ttest.loc[(unsplit_ttest['exp_group'] == exp_group) & (unsplit_ttest['session'] == session)]
        sns.boxenplot(data=df, x='taste_trial', y='Z(euc_dist)', ax=ax, color=sbcolors[0], linewidth=0.5,
                      showfliers=False, k_depth='full')
        # plot a horizontal line that is the average euclidean distance
        avg_euc_dist = df['Z(euc_dist)'].mean()
        ax.axhline(avg_euc_dist, color='black', linestyle='--')
        # if the pvalue for the trial is less than 0.05, plot a star above the line
        for idx, row in pvaldf.iterrows():
            if row['pval'] < 0.05:
                # get the number of stars to plot
                stars = get_sig_stars(row['pval'])
                # get the maximum value of df at the trial number
                med = df.loc[df['taste_trial'] == row['taste_trial'], 'Z(euc_dist)'].max()
                # print the stars turned 90 degrees
                ax.text(row['taste_trial'] - 0.7, max_euc_dist, stars, fontsize=fontsize, ha='center', va='bottom',
                        rotation=90)
                # print the p value below
                ax.text(row['taste_trial'] - 0.7, max_euc_dist, 'p = ' + str(round(row['pval'], 4)), fontsize=9,
                        ha='center', va='top', rotation=-90)

        ax.set_ylim(min_euc_dist, max_euc_dist * 1.3)

        ax.set_title('Session ' + str(session), fontsize=fontsize)
        ax.set_xlabel('trial number', fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=axticksize)
        # make the x axis labels every other label
        xlabs = ax.get_xticklabels()
        # set every other label to ''
        for label in xlabs:
            label.set_visible(False)
        for label in xlabs[::2]:
            label.set_visible(True)

        if i == 0:
            ax.set_ylabel('Z(euc. dist.)', fontsize=fontsize)
        else:
            ax.set_ylabel('')
plt.tight_layout()
# save the figure
plt.show()
plt.savefig(save_dir + '/trialwise_boxenplot.png')
plt.savefig(save_dir + '/trialwise_boxenplot.svg')