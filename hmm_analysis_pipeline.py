import analysis as ana
import blechpy
import new_plotting as nplt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
HA = ana.HmmAnalysis(proj)  # create a hmm analysis object

###############################################################################
#%%NAIVE BAYES CLASSIFICATION
###############################################################################
NB_decode = HA.get_NB_decode()  # get the decode dataframe with some post-processing

test = NB_decode.groupby(['exp_name','time_group', 'epoch']).mean()
aov_groups = ['time_group', 'exp_group', 'epoch']
within = ['taste']
subject = 'exp_name'
dvcols = ['p_correct']
NB_aov, NB_ph = ana.iter_trial_group_anova(NB_decode, aov_groups, dvcols, within, subject,
                                           trial_cols=['taste_trial', 'session_trial'], save_dir=HA.save_dir,
                                           save_suffix='NB_decode')

NB_plot_aov = NB_aov.loc[(NB_aov.Source == 'trial_group')].reset_index(drop=True)

### plot NB accuracy over time #####################################
trial_cols = ['taste_trial', 'session_trial']
trial_groups = [3, 4, 5, 6]
for i in trial_groups:
    for j in trial_cols:
        nplt.plot_trialwise_rel2(NB_decode, y_facs=['p_correct'], save_dir=HA.save_dir,
                                 save_prefix='NB_accuracy', trial_col=j, n_trial_groups=i)

#split trials by single point
NB_split_aov, NB_split_ph = ana.iter_trial_split_anova(NB_decode, aov_groups, dvcols, within, subject,
                                                           trial_cols=['taste_trial', 'session_trial'],
                                                           n_splits=30, save_dir=HA.save_dir, save_suffix='NB_decode')
NB_split_aov_sig = NB_split_aov.loc[NB_split_aov['p-GG-corr'] < 0.05]

###############################################################################
#%%aive Bayes timing
###############################################################################
NB_timings = HA.get_NB_timing()

###Naive Bayes timing ANOVA###
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

### plot NB timing over time #####################################
#trial_cols = ['taste_trial', 'session_trial']
#trial_groups = [3, 4, 5, 6]
trial_cols = ['session_trial']
trial_groups = [4]

epochs = ['prestim', 'early', 'late']
y_cols = ['t_start', 't_end', 't_med', 'duration']
for i in trial_groups:
    for j in trial_cols:
        for nm, group in NB_timings.groupby(['epoch']):
            prefix = 'NB_timing_' + nm
            nplt.plot_trialwise_rel2(group, y_facs=y_cols, save_dir=HA.save_dir,
                                 save_prefix=prefix, trial_col=j, n_trial_groups=i)

#split trials by single point
time_split_aov, time_split_ph = ana.iter_trial_split_anova(NB_timings, aov_groups, dvcols, within, subject,
                                                           trial_cols=['taste_trial', 'session_trial'],
                                                           n_splits=30, save_dir=HA.save_dir, save_suffix='NB_timings')
time_split_aov_sig = time_split_aov.loc[time_split_aov['p-GG-corr'] < 0.05]



###############################################################################
#%%mode gamma probability analysis
###############################################################################
### is mean gamma probability of mode state correlated with accuracy? #########
# get gamma probability of mode state for each bin
avg_gamma_mode_df = HA.get_avg_gamma_mode()

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
g = sns.relplot(data=gm_plt_aov, x='n_trial_groups', y='F', hue='exp_group', col='session', row='exp_group',
                style='trial_type', facet_kws={'margin_titles': True}, aspect=0.8, kind='line')
g.savefig(HA.save_dir + '/gamma_mode_FvalComparison.png')

g = sns.relplot(data=gm_plt_aov, x='n_trial_groups', y='MS', hue='exp_group', col='session', row='exp_group',
                style='trial_type', facet_kws={'margin_titles': True}, aspect=0.8, kind='line')
g.savefig(HA.save_dir + '/gamma_mode_MSvalComparison.png')

sns.catplot(data=gm_plt_aov, x='n_trial_groups', y='p-GG-corr', hue='exp_group', col='time_group', row='trial_type',
            margin_titles=True, aspect=0.5)
plt.show()

### plot gamma mode probability over time #####################################
trial_cols = ['trial', 'session_trial']
trial_groups = [3, 4, 5, 6]
for i in trial_groups:
    for j in trial_cols:
        nplt.plot_trialwise_rel2(avg_gamma_mode_df, y_facs=['pr(mode state)'], save_dir=HA.save_dir,
                                 save_prefix='gamma_mode_AIC', trial_col=j, n_trial_groups=i)

#split trials by a single point
gm_split_aov, gm_split_ph = ana.iter_trial_split_anova(avg_gamma_mode_df, aov_groups, dvcols, within, subject,
                                                              trial_cols=['taste_trial', 'session_trial'],
                                                                n_splits=30, save_dir=HA.save_dir, save_suffix='gamma_mode')
gm_split_aov_sig = gm_split_aov.loc[gm_split_aov['p-GG-corr'] < 0.05]




###############################################################################
#%% combine all ANOVAs into one dataframe
NB_decode_split_AOV_file = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts/taste_experience_resorts_analysis/hmm_analysis/NB_decode_all_trial_split_ANOVA.csv'
gamma_mode_split_AOV_file = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts/taste_experience_resorts_analysis/hmm_analysis/gamma_mode_all_trial_split_ANOVA.csv'
timing_split_AOV_file = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts/taste_experience_resorts_analysis/hmm_analysis/NB_timings_all_trial_split_ANOVA.csv'

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

for nm,group in supersplit_sigaov.groupby(['trial_type']):
    p = sns.relplot(data=group, x='trial_split', y='p-GG-corr', hue='exp_group', style='epoch', col='session',
                    row='dependent_var', aspect=1, height=5, facet_kws={"margin_titles":True})
    p.set_titles(row_template='{row_name}')
    plt.show()

for nm,group in supersplit_sigaov.groupby(['trial_type']):
    p = sns.relplot(data=group, x='trial_split', y='log_F', hue='exp_group', style='epoch', col='session',
                    row='dependent_var', aspect=1, height=5, facet_kws={"margin_titles":True})
    p.set_titles(row_template='{row_name}')
    plt.show()




NB_decode_AOV_file = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts/taste_experience_resorts_analysis/hmm_analysis/NB_decode_all_ANOVA.csv'
gamma_mode_AOV_file = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts/taste_experience_resorts_analysis/hmm_analysis/gamma_mode_all_ANOVA.csv'
timing_AOV_file = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts/taste_experience_resorts_analysis/hmm_analysis/NB_timings_all_ANOVA.csv'

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

