#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:17:30 2022

@author: dsvedberg
"""

# get into the directory
import analysis as ana
import blechpy
import new_plotting as nplt
import hmm_analysis as hmma
import seaborn as sns
# import os
import pandas as pd
import os

# you need to make a project analysis using blechpy.project() first
# rec_dir =  '/media/dsvedberg/Ubuntu Disk/taste_experience'
rec_dir = '/media/dsvedber g/Ubuntu Disk/taste_experience_resorts'
proj = blechpy.load_project(rec_dir)
PA = ana.ProjectAnalysis(proj)
[all_units, held_df] = PA.get_unit_info()  # overwrite = True) #run and check for correct area then run get best hmm
HA = ana.HmmAnalysis(proj)
HA.get_hmm_overview()  # overwrite = True) #use overwrrite = True to debug
# HA.sort_hmms_by_params()#overwrite = True)
HA.sort_hmms_by_BIC(overwrite=True)
srt_df = HA.get_sorted_hmms()
# HA.mark_early_and_late_states() #this is good, is writing in early and late states I think
best_hmms = HA.get_best_hmms(sorting='best_BIC',
                             overwrite=True)  # sorting = 'params #4') #HA has no attribute project analysis #this is not getting the early and late states
# heads up, params #5 doesn't cover all animals, you want params #4
# play around with state timings in mark_early_and_late_states()

sns.set(style='white', font_scale=1.5, rc={"figure.figsize": (10, 10)})
###############################################################################
###NAIVE BAYES CLASSIFICATION##################################################
_, NB_decode, _, _ = HA.analyze_NB_ID(overwrite=False)  # run with no overwrite, make sure to not overwrite NB_meta and NB_decode if you have them
NB_decode[['Y', 'epoch']] = NB_decode.Y.str.split('_', expand=True)
NB_decode['taste'] = NB_decode.trial_ID
NB_decode['state_num'] = NB_decode['hmm_state'].astype(int)

NB_decode = ana.add_session_trial(NB_decode, proj, trial_col='trial_num', trial_id='trial_ID')

aov_groups = ['time_group', 'exp_group', 'epoch']
within = ['taste']
subject = 'exp_name'
dvcols = ['p_correct']
NB_aov, NB_ph = ana.iter_trial_group_anova(NB_decode, aov_groups, dvcols, within, subject,
                                           trial_cols=['trial_num', 'session_trial'], save_dir=HA.save_dir,
                                           save_suffix='NB_decode')

NB_plot_aov = NB_aov.loc[(NB_aov.Source=='trial_group')].reset_index(drop=True)

nplt.plot_NB_decoding(NB_decode, plotdir=HA.save_dir, trial_group_size=20)

NB_summary = NB_decode.groupby(['exp_name', 'time_group', 'trial_ID', 'prestim_state', 'rec_dir', 'hmm_state']).agg(
    'mean').reset_index()

nplt.plot_NB_decoding(NB_decode, plotdir=HA.save_dir, trial_group_size=20, trial_col='session_trial')
nplt.plot_NB_decoding(NB_decode, plotdir=HA.save_dir, trial_group_size=5, trial_col='trial_num')


###############################################################################
###Naive Bayes timing##########################################################
NB_timings = hmma.getNBTimings(HA)

###Naive Bayes timing ANOVA###
aov_groups = ['time_group', 'exp_group', 'epoch']
within = ['taste']
subject = 'exp_name'
dvcols = ['t_start', 't_end', 't_med', 'duration']
time_aov, time_ph = ana.iter_trial_group_anova(NB_timings, aov_groups, dvcols, within, subject,
                                           trial_cols=['trial_num', 'session_trial'], save_dir=HA.save_dir,
                                           save_suffix='NB_timings')

time_plot_aov = time_aov.loc[(time_aov.Source == 'trial_group')].reset_index(drop=True)
time_plot_aov['F'] = time_plot_aov['F'].fillna(0)
sns.set(font_scale=1, style='white')
sns.catplot(data=time_plot_aov, x='n_trial_groups', y='F', hue='exp_group', col='time_group', row='trial_type', kind='boxen', margin_titles=True, aspect=0.5)

sns.catplot(data=time_plot_aov, x='n_trial_groups', y='p-GG-corr', hue='exp_group', col='time_group', row='trial_type', kind='boxen', margin_titles=True, aspect=0.5)

###Naive Bayes timing plotting##########################################################
# plot timing correlations grouped by session:
nplt.plot_NB_timing(NB_timings, HA.save_dir, trial_group_size=20, trial_col='session_trial')
nplt.plot_NB_timing(NB_timings, HA.save_dir, trial_group_size=5, trial_col='trial_num')

early_timings = NB_timings.loc[NB_timings.epoch == 'early']
early_timings = early_timings.groupby(['taste', 'exp_name', 'time_group']).filter(lambda group: len(group) >= 10)
'''
for name, group in early_timings.groupby(['exp_name']):
    print('running')
    nplt.plot_trialwise_lm(group, x_col='session_trial', y_facs=['t_end'] \
                           , hue=None, col='time_group', row='taste', save_dir=HA.save_dir, save_prefix=name)
'''
'''
##############################################################################
###Naive Bayes timing changepoints############################################
NB_meta, NB_decode, NB_best, NB_timings = HA.analyze_NB_ID(overwrite=False, parallel=True)  # run with overwrite

NB_decode['state_num'] = NB_decode['hmm_state'].astype('int64')
NB_decode['taste'] = NB_decode['trial_ID']
grcols = ['rec_dir', 'trial_num', 'taste', 'state_num']
NB_decsub = NB_decode[grcols + ['p_correct']].drop_duplicates()

NB_timings = NB_timings.merge(NB_decsub, on=grcols, how='left')
NB_timings = NB_timings.drop_duplicates()
NB_timings[['Y', 'epoch']] = NB_timings.state_group.str.split('_', expand=True)
avg_timing = NB_timings.groupby(['exp_name', 'taste', 'state_group']).mean()[['t_start', 't_end', 't_med', 'duration']]
avg_timing = avg_timing.rename(columns=lambda x: 'avg_' + x)

NB_timings = pd.merge(NB_timings, avg_timing, on=['exp_name', 'taste', 'state_group'], how='left').drop_duplicates()
NB_timings = NB_timings.reset_index()
idxcols1 = list(NB_timings.loc[:, 'exp_name':'state_num'].columns)
idxcols2 = list(NB_timings.loc[:, 'pos_in_trial':].columns)
idxcols = idxcols1 + idxcols2
NB_timings = NB_timings.set_index(idxcols)
NB_timings = NB_timings.reset_index()

early_timings = NB_timings.loc[NB_timings.epoch == 'early']
cngpts = ana.detect_changepoints(early_timings, ['exp_name', 'time_group', 'exp_group', 'taste'], 't_start',
                                 'trial_num')
nplt.plot_changepoint_histograms(cngpts, save_dir=HA.save_dir)

##############################################################################
###mixed LM trial split analysis #############################################
NB_timings = hmma.getNBtimings(HA)
fit_groups = ['exp_group', 'time_group']
model_groups = ['exp_name', 'taste']
grouping = fit_groups + model_groups
early_timings = NB_timings.loc[NB_timings.epoch == 'early']
early_timings = early_timings.groupby(grouping).filter(lambda group: len(group) >= 10)
early_timings['exp_name'] = early_timings['exp_name'].astype('str')
early_timings['taste'] = early_timings['taste'].astype('category')
early_timings['exp_group'] = early_timings['exp_group'].astype('category')

trial_col = 'trial_num'
response_col = 't_end_zscore'
model_cols = ['exp_group', 'time_group', 'taste']
fit_groups = ['exp_name']
subject_col = 'exp_name'
taste_col = 'taste'
session_col = 'time_group'
condition_col = 'exp_group'
model_groups = ['exp_group', 'time_group', 'taste', 'exp_name']

results_df = ana.find_changepoints_individual(early_timings, model_groups, fit_groups, trial_col, response_col)
nplt.plot_piecewise_individual_models(results_df, session_col, taste_col, condition_col, trial_col, response_col,
                                      subject_col, HA.save_dir)

###trying to use the piecewise-regression library
results_df = ana.fit_piecewise_regression(early_timings, model_groups, 't_end', 'trial_num')
results_df = results_df.sort_values(['time_group'])
results_df = results_df.sort_values(['converged'], ascending=False)
nplt.plot_multiple_piecewise_regression(results_df, save_dir=HA.save_dir)

results_df = ana.fit_piecewise_regression(early_timings, model_groups, 't_end', 'trial_num')
###############################################################################
###Analysis of trial vs NB factors correlations################################
groupings = ['time_group', 'exp_group', 'state_group']
features = ['t_start', 't_end', 't_med', 'duration', 'p_correct', 'palatability']
yfacs = ['t_end', 't_med', 't_start']
df = NB_timings
df = df.loc[df.p_correct > 0.5]

# plot timing correlations grouped by session:
NB_corrs = hmma.analyze_state_correlations(df, groupings, 'trial_num', features)
nplt.plotRsquared(NB_corrs, yfacs, save_dir=HA.save_dir, save_prefix='NB_timing')

# plot timing correlations grouped by session x taste:
groupings = ['time_group', 'exp_group', 'state_group', 'taste']
NB_corrs_tst = hmma.analyze_state_correlations(df, groupings, 'trial_num', features)
nplt.plotRsquared(NB_corrs_tst, yfacs, row='taste', save_dir=HA.save_dir, save_prefix='NB_timing')
'''
###############################################################################
###LR ANALYSIS
'''
LR_timings, LR_trials, LR_res, LR_meta = HA.analyze_pal_linear_regression()
LR_timings['err'] = LR_timings.Y_pred - LR_timings.palatability
LR_timings.err = LR_timings.err.astype(float)
LR_timings['SquaredError'] = LR_timings.err ** 2
features = ['t_start', 't_end', 't_med', 'err', 'SquaredError', 'Y_pred']
groupings = ['time_group', 'exp_group']
LR_corrs = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
groupings = ['time_group', 'exp_group']
LR_corrs_alltst = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)

##Plot accuracy & timing of linear regression##################################

nplt.plot_LR(LR_timings, plotdir=HA.save_dir, trial_group_size=5)
nplt.plot_pal_data(LR_timings, save_dir=HA.save_dir)

###############################################################################
HA.analyze_hmms(overwrite=True)  # comment out/adapt some analyses
HA.plot_hmm_timing()
'''
###############################################################################
###mode gamma probability analysis

### is mean gamma probability of mode state correlated with accuracy? #########
# get gamma probability of mode state for each bin
gamma_mode_df = hmma.binstate(best_hmms)  # TODO get session trial number in here
gamma_mode_df = ana.add_session_trial(gamma_mode_df, proj)# rebin every 50ms into a bin
gamma_mode_df['time_bin'] = gamma_mode_df.time.astype(int) / 20
gamma_mode_df['time_bin'] = gamma_mode_df['time_bin'].astype(int)
gamma_mode_df['binned_time'] = gamma_mode_df.time_bin * 20
gamma_mode_df['session'] = gamma_mode_df.time_group  # rename time_group column to session
gamma_mode_df['session_trial'] = gamma_mode_df.session_trial.astype(int)

gamma_mode_pct_df = gamma_mode_df.groupby(['exp_name', 'exp_group', 'time_group', 'taste', 'trial', 'trial_group'])[
    'gamma_mode'].apply(lambda c: (c > 0.9).sum() / len(c)).reset_index()
avg_gamma_mode_df = gamma_mode_df.groupby(
    ['exp_name', 'exp_group', 'time_group', 'taste', 'trial']).mean().reset_index()
avg_model_groups = ['exp_group', 'time_group', 'taste']
### plot gamma mode probability over time #####################################
avg_gamma_mode_df['pr(mode state)'] = avg_gamma_mode_df.gamma_mode
avg_gamma_mode_df['session_trial'] = avg_gamma_mode_df.session_trial.astype(int)

trial_cols = ['trial', 'session_trial']
trial_groups = [3, 4, 5, 6]
for i in trial_groups:
    for j in trial_cols:
        nplt.plot_trialwise_rel2(avg_gamma_mode_df, y_facs=['pr(mode state)'], save_dir=HA.save_dir,
                                 save_prefix='gamma_mode_AIC', trial_col=j, n_trial_groups=i)

### gamma mode ANOVAS
aov_groups = ['time_group', 'exp_group']
within = ['taste']
subject = 'exp_name'
dvcols = 'gamma_mode'

gm_aov, gm_ph = ana.iter_trial_group_anova(avg_gamma_mode_df, groups=aov_groups, dep_vars=dvcols, within=within,
                                           save_dir=HA.save_dir)

gm_plt_aov = gm_aov.loc[gm_aov.Source == 'trial_group']

gm_plt_aov['session'] = gm_plt_aov.time_group
gm_plt_aov['trial_type'] = gm_plt_aov['trial_type'].replace('trial', 'taste_trial')
gm_plt_aov = gm_plt_aov.reset_index(drop=True)
g = sns.relplot(data=gm_plt_aov, x='n_trial_groups', y='F', hue='exp_group', col='session', row='exp_group', style='trial_type', facet_kws={'margin_titles':True}, aspect=0.8, kind='line')
g.savefig(HA.save_dir + '/gamma_mode_FvalComparison.png')

g = sns.relplot(data=gm_plt_aov, x='n_trial_groups', y='MS', hue='exp_group', col='session', row='exp_group', style='trial_type', facet_kws={'margin_titles':True}, aspect=0.8, kind='line')
g.savefig(HA.save_dir + '/gamma_mode_MSvalComparison.png')

sns.catplot(data=gm_plt_aov, x='n_trial_groups', y='p-GG-corr', hue='exp_group', col='time_group', row='trial_type', margin_titles=True, aspect = 0.5)
plt.show()

### Aggregating all ANOVAS for plotting #######################################
NB_decode_AOV_file = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts/taste_experience_resorts_analysis/hmm_analysis/NB_decode_all_ANOVA.csv'
gamma_mode_AOV_file = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts/taste_experience_resorts_analysis/hmm_analysis/gamma_mode_all_ANOVA.csv'
timing_AOV_file = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts/taste_experience_resorts_analysis/hmm_analysis/NB_timings_all_ANOVA.csv'

NB_aov = pd.read_csv(NB_decode_AOV_file)

time_aov = pd.read_csv(timing_AOV_file)

gm_aov = pd.read_csv(gamma_mode_AOV_file)
gm_aov['epoch'] = 'all'

super_aov = pd.concat([gm_aov, NB_aov, time_aov])
super_aov = super_aov.reset_index(drop=True)
super_aov = super_aov.loc[super_aov['Source'] == 'trial_group']
super_aov = super_aov.loc[super_aov['epoch'] != 'prestim']
super_aov['trial_type'] = super_aov['trial_type'].replace('trial_num', 'trial')
super_aov['trial_type'] = super_aov['trial_type'].replace('trial', 'taste_trial')
super_aov[['time_group', 'n_trial_groups']] = super_aov[['time_group', 'n_trial_groups']].astype(int)
super_aov['session'] = super_aov['time_group']
super_aov['significant'] = super_aov['p-GG-corr'] < 0.05

p = sns.catplot(data=super_aov, x='n_trial_groups', y='p-GG-corr', hue='exp_group', col='session', row='trial_type', margin_titles=True, aspect=1, height=5, kind='swarm', dodge=True)
for ax in p.axes.flat:
    ax.axhline(0.05, ls='--', color='gray')
p.savefig(os.path.join(HA.save_dir, 'all_measures_Pval_vs_splitnum.png'))

sns.set(font_scale=1, style='white')
g = sns.catplot(data=super_aov, x='n_trial_groups', y='F', hue='exp_group', col='session', row = 'trial_type', margin_titles=True, aspect=0.5, kind='boxen')
g.savefig(os.path.join(HA.save_dir, 'all_measures_Fval_vs_splitnum.png'))

p = sns.catplot(data=super_aov, x='n_trial_groups', y='ng2', hue='exp_group', col='session', row='trial_type', margin_titles=True, aspect=0.5, height=5, kind='boxen')
p.savefig(os.path.join(HA.save_dir, 'all_measures_eta_vs_splitnum.png'))

p = sns.catplot(data=super_aov, x='n_trial_groups', y='MS', hue='exp_group', col='session', row='trial_type', margin_titles=True, aspect=0.5, height=5, kind='boxen')
p.savefig(os.path.join(HA.save_dir, 'all_measures_MS_vs_splitnum.png'))

super_aov['metric_significant'] = super_aov.groupby(['epoch','dependent_var','n_trial_groups','trial_type'])['p-GG-corr'].transform(lambda x: (x < 0.05).any())
sig_aov = super_aov.loc[super_aov['metric_significant']]
sig_aov['variable'] = '\n' + sig_aov['dependent_var'] + '|' + sig_aov['epoch'] + '_epoch'

sig_eta_aov = sig_aov.loc[sig_aov['p-GG-corr'] < 0.05]
g = sns.relplot(data=sig_eta_aov, x='n_trial_groups', y='ng2',style='trial_type', markers = True, col='session', facet_kws={'margin_titles':True}, aspect=0.8, height=6, kind='line')
g.savefig(os.path.join(HA.save_dir, 'sig_measures_eta_vs_session.png'))

g = sns.relplot(data=sig_aov, x='session', y='p-GG-corr', hue='exp_group',style='trial_type', col='variable', row='n_trial_groups', facet_kws={'margin_titles':True}, aspect=1, height=4, kind='line')
for ax in g.axes.flat:
    ax.axhline(0.05, ls='--', color='gray')
g.savefig(os.path.join(HA.save_dir, 'sig_measures_Pval_vs_session.png'))

g = sns.relplot(data=sig_aov, x='n_trial_groups', y='F', hue='exp_group',style='trial_type',marker=True, col='variable', row='session', facet_kws={'margin_titles':True}, linewidth=2, aspect=1, height=4, kind='line')
g.set(ylim=(0,20))
g.savefig(os.path.join(HA.save_dir, 'sig_measures_F_vs_session.png'))


# TODO: create plotting function for ANOVAs

### perform piecewise regression of gamma mode and plot over trials ############
model_groups = ['exp_name', 'exp_group', 'time_group', 'taste']
gam_pw = ana.fit_piecewise_regression(gamma_mode_df, model_groups, 'gamma_mode', 'trial')
gam_pw = gam_pw.sort_values(['time_group'])
gam_pw = gam_pw.sort_values(['converged'], ascending=False)
nplt.plot_multiple_piecewise_regression2(gam_pw, save_dir=HA.save_dir)

gamavg_pw = ana.fit_piecewise_regression(avg_gamma_mode_df, avg_model_groups, 'gamma_mode', 'trial')
gamavg_pw = gamavg_pw.sort_values(['time_group'])
gamavg_pw = gamavg_pw.sort_values(['converged'], ascending=False)
grvars = ['exp_group']
nplt.plot_multiple_piecewise_regression(gamavg_pw, grvars=grvars, save_dir=HA.save_dir)

### is max gamma probability of mode state correlated with accuracy? ##########
maxgammadf = hmma.getMaxGamma(best_hmms, trial_group=5)
maxgammadf2 = maxgammadf.loc[maxgammadf.max_gamma < 0.9]

nplt.plotGammaPar(maxgammadf, save_dir=HA.save_dir, yax="max_gamma")
nplt.plotGammaBarPar(maxgammadf2, save_dir=HA.save_dir, yax="max_gamma")

maxgammadf['time_bin'] = maxgammadf.time.astype(int) / 10
maxgammadf['time_bin'] = maxgammadf['time_bin'].astype(int)
maxgammadf['trial_bin'] = maxgammadf.trial_group  # rename trial_group column to trial_bin
maxgammadf['session'] = maxgammadf.time_group  # rename time_group column to session
nplt.plotGammaPar(maxgammadf, save_dir=HA.save_dir, yax="max_gamma")

### is gamma probability of mode state, averaged across the trial, correlated with accuracy? ##########

gmdf_poststim = gamma_mode_df.loc[gamma_mode_df.time > 0]  # filter away prestim bins
# average gamma probability of mode state for trial
gmdf_avg = gmdf_poststim.groupby(['exp_name', 'taste', 'time_group', 'trial', 'exp_group']).mean().reset_index()
gmdf_avg['trial'] = gmdf_avg.trial.astype(int)  # make trial number an int
gmdf_avg['mean pr(mode state)'] = gmdf_avg['gamma_mode']  # rename gamma_mode column to mean pr(mode state)
gmdf_avg['trial_bin'] = gmdf_avg.trial_group  # rename trial_group column to trial_bin
gmdf_avg['session'] = gmdf_avg.time_group  # rename time_group column to session
# plot scatterplot of mean gamma probability of mode state vs trial, with linear regression
nplt.plot_trialwise_lm(df=gmdf_avg, x_col='trial', y_facs=['mean pr(mode state)'], h='exp_group', c='time_group',
                       save_dir=HA.save_dir, save_prefix='gamma_mode')

### is trial accuracy rank according to gamma probability of mode state correlated with accuracy? ######
# rank each trial by accuracy
gmdf_avg['accuracy_rank'] = gmdf_avg.groupby(['exp_name', 'taste', 'time_group', 'exp_group'])['gamma_mode'].rank(
    ascending=False)

# plot accuracy rank vs trial with linear regression
nplt.plot_trialwise_lm(df=gmdf_avg, x_col='trial', y_facs=['accuracy_rank'], hue='exp_group', col='time_group',
                       save_dir=HA.save_dir, save_prefix='gamma_mode')

# plot 2d distribution of accuracy rank vs trial
g = sns.displot(data=gmdf_avg, x='trial', y='accuracy_rank', color='0', bins=5, row='exp_group', col='time_group',
                palette='colorblind', facet_kws={'margin_titles': True})
sf = os.path.join(HA.save_dir, 'accuracy_rank_vs_trial_histogram.png')
g.savefig(sf)

pr_modegamma_poststim = gamma_mode_df.loc[gamma_mode_df.time > 0]
pr_modegamma_poststim['trial'] = pr_modegamma_poststim.trial.astype(int)
binavg_pr_mode_gamma = pr_modegamma_poststim.groupby(
    ['taste', 'trial', 'exp_name', 'time_group', 'exp_group', 'trial_group']).mean().reset_index()
binavg_pr_mode_gamma['taste'] = pd.Categorical(binavg_pr_mode_gamma['taste'], ['Suc', 'NaCl', 'CA', 'QHCl'])
g = sns.relplot(data=binavg_pr_mode_gamma, x="trial", y="gamma_mode", row="time_group", col="taste", hue="exp_group",
                kind="line",
                facet_kws={'margin_titles': True})
g.tight_layout()
g.set_ylabels("p(gamma) of mode state")
sf = os.path.join(HA.save_dir, 'test.png')
g.savefig(sf)


def getgroupidx(grouped_df, idx):
    name = list(grouped_df.groups.keys())[idx]
    group = grouped_df.get_group(name)
    return group, name
