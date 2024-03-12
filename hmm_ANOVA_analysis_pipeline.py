import analysis as ana
import blechpy
import new_plotting as nplt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import trialwise_analysis as ta
import scipy.stats as stats

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
HA = ana.HmmAnalysis(proj)  # create a hmm analysis object

###############################################################################
#%%NAIVE BAYES CLASSIFICATION
###############################################################################
NB_decode = HA.get_NB_decode()  # get the decode dataframe with some post-processing
#
# ### plot NB accuracy over time #
# trial_cols = ['taste_trial', 'session_trial']
# trial_groups = [3, 4, 5, 6]
# for i in trial_groups:
#     for j in trial_cols:
#         nplt.plot_trialwise_rel2(NB_decode, y_facs=['p_correct'], save_dir=HA.save_dir,
#                                  save_prefix='NB_accuracy', trial_col=j, n_trial_groups=i)
#
# ### Naive bayes trial-group ANVOA ###
# test = NB_decode.groupby(['exp_name','time_group', 'epoch']).mean()
# aov_groups = ['time_group', 'exp_group', 'epoch']
# within = ['taste']
# subject = 'exp_name'
# dvcols = ['p_correct']
# NB_aov, NB_ph = ana.iter_trial_group_anova(NB_decode, aov_groups, dvcols, within, subject,
#                                            trial_cols=['taste_trial', 'session_trial'], save_dir=HA.save_dir,
#                                            save_suffix='NB_decode')
#
# NB_plot_aov = NB_aov.loc[(NB_aov.Source == 'trial_group')].reset_index(drop=True)

#%% NB accuracy ANOVA, split trials by single point###
aov_groups = ['time_group', 'exp_group', 'epoch']
within = ['taste']
subject = 'exp_name'
dvcols = ['p_correct']
NB_split_aov, NB_split_ph = ana.iter_trial_split_anova(NB_decode, aov_groups, dvcols, within, subject,
                                                           trial_cols=['taste_trial'],#, 'session_trial'],
                                                           n_splits=30, save_dir=HA.save_dir, save_suffix='NB_decode')

#%%
NB_split_aov_sig = NB_split_aov.loc[NB_split_aov['p-GG-corr'] < 0.05]
NB_split_aov_sig = NB_split_aov_sig.loc[NB_split_aov_sig['Source'] == 'trial_group']
for nm, group in NB_split_aov_sig.groupby(['trial_type']):
    g = sns.relplot(data=group, x='trial_split', y='p-GG-corr', hue='exp_group', row='epoch', style='Source', col='time_group', facet_kws={'margin_titles':True}, s=100, aspect=1.25, height=4)
    g.fig.suptitle(nm + ' trial split ANOVA: accuracy')
    g.fig.subplots_adjust(top=0.9)
    plt.show()

###############################################################################
#%%aive Bayes timing
###############################################################################
NB_timings = HA.get_NB_timing()

#%% plot NB timing over time #####################################
#trial_cols = ['taste_trial', 'session_trial']
#trial_groups = [3, 4, 5, 6]
trial_cols = ['taste_trial', 'session_trial']
trial_groups = [4]
epochs = ['prestim', 'early', 'late']
y_cols = ['t_start', 't_end', 't_med', 'duration']

for i in trial_groups:
    for j in trial_cols:
        for nm, group in NB_timings.groupby(['epoch']):
            prefix = 'NB_timing_' + nm
            nplt.plot_trialwise_rel2(group, y_facs=y_cols, save_dir=HA.save_dir,
                                 save_prefix=prefix, trial_col=j, n_trial_groups=i)


for i in y_cols:
    for j in ['session_trial']:#trial_cols:
        for nm, group in NB_timings.groupby(['epoch','exp_group']):
            prefix = 'NB_timing_' + nm[0] + '_' + nm[1]
            g = sns.lmplot(data=group, x=j, y=i, hue="exp_name", row="taste", col="time_group",
                           facet_kws={'margin_titles': True}, height=4, aspect=1.25, scatter_kws={'alpha': 0.5})
            g.fig.suptitle(nm)
            g.fig.subplots_adjust(top=0.9)
            g.set(ylim=(0,3000))
            plt.show()

#%%Naive Bayes timing ANOVA###
aov_groups = ['time_group', 'exp_group', 'epoch']
within = ['taste']
subject = 'exp_name'
dvcols = ['t_start', 't_end', 't_med', 'duration']

time_aov, time_ph = ana.iter_trial_group_anova(NB_timings, aov_groups, dvcols, within, subject,
                                               trial_cols=['taste_trial', 'session_trial'], save_dir=HA.save_dir,
                                               save_suffix='NB_timings')

time_plot_aov = time_aov.loc[(time_aov.Source == 'trial_group')].reset_index(drop=True)
time_plot_aov['F'] = time_plot_aov['F'].fillna(0)
sns.set(font_scale=1, style='white')
sns.catplot(data=time_plot_aov, x='n_trial_groups', y='F', hue='exp_group', col='time_group', row='trial_type',
            kind='boxen', margin_titles=True, aspect=0.5)

sns.catplot(data=time_plot_aov, x='n_trial_groups', y='p-GG-corr', hue='exp_group', col='time_group', row='trial_type',
            kind='boxen', margin_titles=True, aspect=0.5)

#%%split trials by single point ANOVA###
aov_groups = ['time_group', 'exp_group', 'epoch']
within = ['taste']
subject = 'exp_name'
dvcols = ['t_start', 't_end', 't_med', 'duration']

#timing = NB_timings.loc[NB_timings['p_correct'] > 0.5]
time_split_aov, time_split_ph = ana.iter_trial_split_anova(NB_timings, aov_groups, dvcols, within, subject,
                                                           trial_cols=['taste_trial', 'session_trial'],
                                                           n_splits=30, save_dir=HA.save_dir, save_suffix='NB_timings')
time_split_aov_sig = time_split_aov.loc[time_split_aov['p-GG-corr'] < 0.05]
time_split_aov_sig = time_split_aov_sig.loc[time_split_aov_sig['Source'] == 'trial_group']

#%%plot results of trial-split ANOVA on timing
for nm, group in time_split_aov_sig.groupby(['trial_type', 'epoch']):
    g = sns.relplot(data=group, x='trial_split', y='p-GG-corr', hue='exp_group', row='dependent_var', style='Source', col='time_group', facet_kws={'margin_titles':True}, s=100, aspect=1.25, height=4)
    g.fig.suptitle(nm[0] + ' ' + nm[1] + ' timing split')
    g.fig.subplots_adjust(top=0.9)
    plt.show()

#%% snip trials ANOVA timing
snips_aov = []
aov_groups = ['time_group', 'exp_group', 'epoch']
within = ['taste']
subject = 'exp_name'
dvcols = ['t_start', 't_end', 't_med', 'duration']
for i in range(0,20):
    time_snip = NB_timings.loc[NB_timings['taste_trial'] >= i]
    ns = 30-i
    time_snip_aov, time_snip_ph = ana.iter_trial_split_anova(time_snip, aov_groups, dvcols, within, subject,
                                                        trial_cols=['taste_trial'],
                                                        n_splits=ns, save_dir=HA.save_dir, save_suffix='timings_ANOVA_snip')
    time_snip_aov['start_trial'] = i
    time_snip_sig = time_snip_aov.loc[time_snip_aov['p-GG-corr'] < 0.05]
    snips_aov.append(time_snip_sig)

snips_aov = pd.concat(snips_aov)
snips_trials_aov = snips_aov.loc[snips_aov['Source'] == 'trial_group'].reset_index()
snips_trials_aov['session'] = snips_trials_aov['time_group']
snips_trials_aov = snips_trials_aov.loc[snips_trials_aov['start_trial'] <= 20]


for nm, group in snips_trials_aov.groupby(['dependent_var', 'trial_type']):
    group = group.sort_values(by=['exp_group'])
    sns.set(font_scale=1, style='white')
    g = sns.relplot(data=group, x='trial_split', y='p-GG-corr', hue='exp_group', style='epoch', col='start_trial', row='session', facet_kws={'margin_titles':True}, aspect=0.4, height=3.5)
    g.set_titles(col_template='{col_name}')
    g.set_ylabels('p-value (adj.)')
    g.fig.suptitle(nm[0] + ' ' + nm[1] +" trials cut off from start of session")
    g.fig.subplots_adjust(top=0.9)
    plt.show()
    sf = os.path.join(HA.save_dir, nm[0] + nm[1] + '_Pval_vs_splitnum_vs_trialsnip.png')
    g.savefig(os.path.join(HA.save_dir,sf))


###############################################################################
#%%mode gamma probability analysis

avg_gamma_mode_df = HA.get_avg_gamma_mode()
#%% plot gamma mode probability over time #####################################
trial_cols = ['trial', 'session_trial']
trial_groups = [3, 4, 5, 6]
for i in trial_groups:
    for j in trial_cols:
        nplt.plot_trialwise_rel2(avg_gamma_mode_df, y_facs=['pr(mode state)'], save_dir=HA.save_dir,
                                 save_prefix='gamma_mode_AIC', trial_col=j, n_trial_groups=i)

### gamma mode ANOVAS
#grouping factors
aov_groups = ['time_group', 'exp_group']
within = ['taste']
subject = 'exp_name'
dvcols = 'gamma_mode'

#run ANOVA on binned trials
gm_aov, gm_ph = ana.iter_trial_group_anova(avg_gamma_mode_df, groups=aov_groups, dep_vars=dvcols, within=within,
                                           save_dir=HA.save_dir)

gm_plt_aov = gm_aov.loc[gm_aov.Source == 'trial_group']

gm_plt_aov['session'] = gm_plt_aov.time_group
gm_plt_aov['trial_type'] = gm_plt_aov['trial_type'].replace('trial', 'taste_trial')
gm_plt_aov = gm_plt_aov.reset_index(drop=True)
g = sns.relplot(data=gm_plt_aov, x='n_trial_groups', y='F', hue='exp_group', col='session', row='exp_group',
                style='trial_type', facet_kws={'margin_titles': True}, aspect=0.8, kind='line')
g.savefig(HA.save_dir + '/gamma_mode_FvalComparison.png')

g = sns.relplot(data=gm_plt_aov, x='n_trial_groups', y='MS', hue='exp_group', col='session', row='exp_group',
                style='trial_type', facet_kws={'margin_titles': True}, aspect=0.8, kind='line')
g.savefig(HA.save_dir + '/gamma_mode_MSvalComparison.png')

sns.catplot(data=gm_plt_aov, x='n_trial_groups', y='p-GG-corr', hue='exp_group', col='time_group', row='trial_type',
            margin_titles=True, aspect=0.5)
plt.show()

#%%run ANOVA on gamma mode splitting trials by a single point
aov_groups = ['time_group', 'exp_group']
within = ['taste']
subject = 'exp_name'
dvcols = 'gamma_mode'

gm_split_aov, gm_split_ph = ana.iter_trial_split_anova(avg_gamma_mode_df, aov_groups, dvcols, within, subject,
                                                              trial_cols=['taste_trial', 'session_trial'],
                                                                n_splits=30, save_dir=HA.save_dir, save_suffix='gamma_mode')
gm_split_aov['session'] = gm_split_aov['time_group']

gm_split_aov_sig = gm_split_aov.loc[gm_split_aov['p-GG-corr'] < 0.05].reset_index(drop=True)
gm_split_aov_sig['session'] = gm_split_aov_sig['time_group']
gm_split_aov_sig = gm_split_aov_sig.loc[gm_split_aov_sig['Source'] == 'trial_group']

#%%plot a relplot of the ANOVA significance levels for each significant trial split
sns.set(font_scale=1.25, style='white')
for nm, group in gm_split_aov_sig.groupby(['trial_type']):
    g = sns.relplot(data=group, x='trial_split', y='p-GG-corr', hue='exp_group', row='trial_type', style='Source',
                    col='session', facet_kws={'margin_titles':True}, s=100, aspect=1.25, height=4)
    g.set_titles(row_template='{row_name}')
    plt.show()
    sn = nm + '_Pval_vs_splitnum_gamma_mode.png'
    sf = os.path.join(HA.save_dir, sn)
    g.savefig(sf)


#run ANOVA on gamma mode splitting, but remove the first 11 trials. this should abolish significance
snips_aov = []
for i in range(0,20):
    gm_snip = avg_gamma_mode_df.loc[avg_gamma_mode_df['taste_trial'] >= i]
    ns = 30-i
    gm_snip_aov, gm_snip_ph = ana.iter_trial_split_anova(gm_snip, aov_groups, dvcols, within, subject,
                                                        trial_cols=['taste_trial'],
                                                        n_splits=ns, save_dir=HA.save_dir, save_suffix='gamma_mode_snip')
    gm_snip_aov['start_trial'] = i
    gm_snip_sig = gm_snip_aov.loc[gm_snip_aov['p-GG-corr'] < 0.05]
    snips_aov.append(gm_snip_sig)

snips_aov = pd.concat(snips_aov)
snips_trials_aov = snips_aov.loc[snips_aov['Source'] == 'trial_group']
snips_trials_aov['session'] = snips_trials_aov['time_group']
snips_trials_aov = snips_trials_aov.loc[snips_trials_aov['start_trial'] < 16]

sns.set(font_scale=1, style='white')
g = sns.relplot(data=snips_trials_aov, x='trial_split', y='p-GG-corr', hue='exp_group', col='start_trial', row='session', facet_kws={'margin_titles':True}, aspect=0.4, height=3.5)
g.set_titles(col_template='{col_name}')
g.set_ylabels('p-value (adj.)')
g.fig.suptitle("trials cut off from start of session")
g.fig.subplots_adjust(top=0.9)
plt.show()
g.savefig(os.path.join(HA.save_dir, 'gamma_mode_Pval_vs_splitnum_vs_trialsnip.png'))

#%%

gm_split_aov_sig = gm_split_aov_sig.loc[gm_split_aov_sig['Source'] == 'trial_group']
dfs = []
for nm, group in gm_split_aov.groupby(['trial_type']):
    trial_splits = group['trial_split'].unique()
    for i in trial_splits:
        df = avg_gamma_mode_df.copy()
        df['trial_type'] = nm
        df['split_trial'] = i
        df['half'] = df[nm] > i
        df['half'] = df['half'].astype(int)
        df = df.drop(nm, axis=1)
        dfs.append(df)
dfs = pd.concat(dfs)

for nm, group in dfs.groupby(['trial_type']):
    p = sns.catplot(data=group, x='split_trial', y='gamma_mode', hue='half', row='exp_group', col='session', kind='boxen', sharex=False, margin_titles=True, aspect=1.75, height=4.5)
    p.set_titles(row_template='{row_name}')
    p.fig.suptitle(nm + ' gamma mode split by half')
    p.fig.subplots_adjust(top=0.9)
    p.savefig(os.path.join(HA.save_dir, nm + '_gamma_mode_split_by_half.png'))
plt.show()

trldiff = dfs.groupby(['half', 'exp_name','exp_group', 'time_group','split_trial','trial_type']).mean().reset_index()
def grpdiff(group, col):
    return group[col].diff()

trldiff = trldiff.groupby(['exp_name', 'exp_group', 'time_group', 'split_trial','trial_type']).apply(grpdiff, 'pr(mode state)').reset_index().dropna()
trldiff['pr(mode state) change'] = trldiff['pr(mode state)']
trldiff['session'] = trldiff['time_group']
for nm, group in trldiff.groupby(['trial_type']):
    p = sns.catplot(data=group, x='split_trial', y='pr(mode state) change', row='trial_type', col='session', kind='boxen', hue='exp_group', sharex=False, margin_titles=True, aspect=1.75, height=4.5)
    sf = os.path.join(HA.save_dir, nm + '_gamma_mode_split_by_half_change.png')
    p.savefig(sf)
plt.show()


for nm, group in trldiff.groupby(['trial_type']):
    p = sns.catplot(data=dfs, x='split_trial', y='gamma_mode', hue='exp_group', col='session', kind='boxen', margin_titles=True, aspect=2, height=4, sharex=False)


###############################################################################
#%% combine all ANOVAs into one dataframe

NB_decode_split_AOV_file = proj_dir + '/taste_experience_resorts_analysis/hmm_analysis/NB_decode_all_trial_split_ANOVA.csv'
gamma_mode_split_AOV_file = proj_dir + '/taste_experience_resorts_analysis/hmm_analysis/gamma_mode_all_trial_split_ANOVA.csv'
timing_split_AOV_file = proj_dir + '/taste_experience_resorts_analysis/hmm_analysis/NB_timings_all_trial_split_ANOVA.csv'

NB_split_aov = pd.read_csv(NB_decode_split_AOV_file)
time_split_aov = pd.read_csv(timing_split_AOV_file)
gm_split_aov = pd.read_csv(gamma_mode_split_AOV_file)

supersplit_aov = pd.concat([gm_split_aov, NB_split_aov, time_split_aov])
supersplit_aov = supersplit_aov.reset_index(drop=True)
supersplit_aov = supersplit_aov.loc[supersplit_aov['Source'] != 'taste']
supersplit_aov = supersplit_aov.loc[supersplit_aov['epoch'] != 'prestim']
supersplit_aov[['time_group', 'n_trial_groups']] = supersplit_aov[['time_group', 'n_trial_groups']].astype(int)
supersplit_aov['session'] = supersplit_aov['time_group']
supersplit_aov['significant'] = supersplit_aov['p-GG-corr'] < 0.05
supersplit_sigaov = supersplit_aov.loc[supersplit_aov['significant']].reset_index(drop=True)
supersplit_sigaov['trial_split'] = supersplit_sigaov['trial_split'].astype(int)
supersplit_sigaov['log_F'] = np.log(supersplit_sigaov['F'])
supersplit_sigaov = supersplit_sigaov.loc[supersplit_sigaov['Source'] == 'trial_group']
supersplit_sigaov['epoch'] = supersplit_sigaov['epoch'].fillna('all')
supersplit_sigaov['dependent_var'] = supersplit_sigaov['dependent_var'].replace('gamma_mode', 'pr(mode state)')
supersplit_sigaov['dependent_var'] = supersplit_sigaov['dependent_var'].replace('p_correct', 'accuracy')


sns.set(font_scale=1, style='white')
for nm,group in supersplit_sigaov.groupby(['trial_type']):
    p = sns.relplot(data=group, x='trial_split', y='p-GG-corr', hue='exp_group', style='epoch', col='session',
                    row='dependent_var', aspect=1.5, height=2.5, s=100, facet_kws={"margin_titles":True})
    p.set_titles(row_template='{row_name}')
    p.set_ylabels('p-value (adj.)')
    p.fig.suptitle("p-value vs trial split: " + nm +"s", fontsize=20)
    p.fig.subplots_adjust(top=0.93)
    plt.show()
    fn = nm + '_Pval_vs_splitnum.png'
    sf = os.path.join(HA.save_dir, fn)
    p.savefig(sf)

avg_supersplit_sigaov = supersplit_sigaov.groupby(['trial_type', 'exp_group', 'session', 'trial_split']).mean().reset_index()
for nm,group in supersplit_sigaov.groupby(['trial_type']):
    p = sns.relplot(data=group, x='trial_split', y='p-GG-corr', hue='exp_group', col='session',
                    aspect=1.25, height=4, facet_kws={"margin_titles":True}, kind='line')
    p.set_titles(row_template='{row_name}')
    p.set_ylabels('p-value (adj.)')
    p.fig.suptitle("p-value vs trial split average: " + nm, fontsize=20)
    p.fig.subplots_adjust(top=0.8)
    plt.show()
    fn = nm + '_Pval_vs_splitnum_average.png'
    sf = os.path.join(HA.save_dir, fn)
    p.savefig(sf)


for nm,group in supersplit_sigaov.groupby(['trial_type']):
    p = sns.relplot(data=group, x='trial_split', y='log_F', hue='exp_group', style='epoch', col='session',
                    row='dependent_var', aspect=1.5, height=2.5, s=100, facet_kws={"margin_titles":True})
    p.set_titles(row_template='{row_name}')
    p.fig.suptitle("log(F-value) vs trial split: " + nm)
    p.fig.subplots_adjust(top=0.93)
    plt.show()
    fn = nm + '_logFval_vs_splitnum.png'
    sf = os.path.join(HA.save_dir, fn)
    p.savefig(sf)

for nm,group in supersplit_sigaov.groupby(['trial_type']):
    p = sns.relplot(data=group, x='trial_split', y='log_F', hue='exp_group', col='session',
                    aspect=1.5, height=4.5, facet_kws={"margin_titles":True}, kind='line')
    p.set_titles(row_template='{row_name}')
    p.fig.suptitle("p-value vs trial split average: " + nm)
    p.fig.subplots_adjust(top=0.8)
    plt.show()
    fn = nm + '_logFval_vs_splitnum_average.png'
    sf = os.path.join(HA.save_dir, fn)
    p.savefig(sf)


NB_decode_AOV_file = proj_dir + '/hmm_analysis/NB_decode_all_ANOVA.csv'
gamma_mode_AOV_file = proj_dir + '/hmm_analysis/gamma_mode_all_ANOVA.csv'
timing_AOV_file = proj_dir + '/hmm_analysis/NB_timings_all_ANOVA.csv'

NB_aov = pd.read_csv(NB_decode_AOV_file)

time_aov = pd.read_csv(timing_AOV_file)

gm_aov = pd.read_csv(gamma_mode_AOV_file)
gm_aov['epoch'] = 'all'

super_aov = pd.concat([gm_aov, NB_aov, time_aov])
super_aov = super_aov.reset_index(drop=True)
super_aov = super_aov.loc[super_aov['Source'] != 'taste']
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
g = sns.catplot(data=super_aov, x='n_trial_groups', y='F', hue='exp_group', col='session', row='trial_type', margin_titles=True, aspect=0.5, kind='boxen')
g.savefig(os.path.join(HA.save_dir, 'all_measures_Fval_vs_splitnum.png'))

p = sns.catplot(data=super_aov, x='n_trial_groups', y='ng2', hue='exp_group', col='session', row='trial_type', margin_titles=True, aspect=0.5, height=5, kind='boxen')
p.savefig(os.path.join(HA.save_dir, 'all_measures_eta_vs_splitnum.png'))

p = sns.catplot(data=super_aov, x='n_trial_groups', y='MS', hue='exp_group', col='session', row='trial_type', margin_titles=True, aspect=0.5, height=5, kind='boxen')
p.savefig(os.path.join(HA.save_dir, 'all_measures_MS_vs_splitnum.png'))

super_aov['metric_significant'] = super_aov.groupby(['epoch','dependent_var','n_trial_groups','trial_type', 'Source'])['p-GG-corr'].transform(lambda x: (x < 0.05).any())
sig_group_aov = super_aov.loc[super_aov['metric_significant']]
sig_group_aov['variable'] = '\n' + sig_group_aov['dependent_var'] + '|' + sig_group_aov['epoch'] + '_epoch'

sig_aov = super_aov.loc[super_aov['p-GG-corr'] < 0.05]
sig_aov['variable'] = '\n' + sig_aov['dependent_var'] + '|' + sig_aov['epoch'] + '_epoch'
sig_aov['log_F'] = np.log10(sig_aov['F'])

g = sns.relplot(data=sig_aov, x='n_trial_groups', y='ng2',style='trial_type', markers = True, col='session', facet_kws={'margin_titles':True}, aspect=0.8, height=6, kind='line')
g.savefig(os.path.join(HA.save_dir, 'sig_measures_eta_vs_session.png'))

g = sns.relplot(data=sig_aov, x='session', y='p-GG-corr', hue='exp_group',style='trial_type', col='variable', row='n_trial_groups', facet_kws={'margin_titles':True}, aspect=1, height=4, kind='line')
for ax in g.axes.flat:
    ax.axhline(0.05, ls='--', color='gray')
g.savefig(os.path.join(HA.save_dir, 'sig_measures_Pval_vs_session.png'))



g = sns.relplot(data=sig_aov, x='n_trial_groups', y='log_F', hue='exp_group',style='trial_type', marker=True, s=100, col='variable', row='session', facet_kws={'margin_titles':True}, aspect=0.75, height=4)
g.savefig(os.path.join(HA.save_dir, 'sig_measures_F_vs_session.png'))

sigsum_aov = sig_aov.groupby(['n_trial_groups','trial_type', 'exp_group', 'session']).sum().reset_index()
g = sns.relplot(data=sigsum_aov, x='n_trial_groups', y='log_F', hue='exp_group',style='trial_type', marker=True, row='session', facet_kws={'margin_titles':True}, linewidth=2, aspect=2, height=4, kind='line')
g.set_ylabels('sum of log F values')
g.savefig(os.path.join(HA.save_dir, 'sig_measures_F_vs_session_all.png'))

####
#idea one: cross correlogram of euclidean distances between states in each animal, nested along levels:
##level 1: states within an HMM
##level 2: states within a session/across tastes
##level 3: states within an animal/across sessions


#idea two: "histogram" of states as function of trials--maybe z score them?

