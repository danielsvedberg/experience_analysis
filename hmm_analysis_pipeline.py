import analysis as ana
import blechpy
import new_plotting as nplt
import hmm_analysis as hmma
import seaborn as sns

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
HA = ana.HmmAnalysis(proj)  # create a hmm analysis object

###############################################################################
#%%NAIVE BAYES CLASSIFICATION
###############################################################################
HA = ana.HmmAnalysis(proj)  # create a hmm analysis object
NB_decode = HA.get_NB_decode()  # get the decode dataframe with some post-processing

aov_groups = ['time_group', 'exp_group', 'epoch']
within = ['taste']
subject = 'exp_name'
dvcols = ['p_correct']
NB_aov, NB_ph = ana.iter_trial_group_anova(NB_decode, aov_groups, dvcols, within, subject,
                                           trial_cols=['taste_trial', 'session_trial'], save_dir=HA.save_dir,
                                           save_suffix='NB_decode')

NB_plot_aov = NB_aov.loc[(NB_aov.Source == 'trial_group')].reset_index(drop=True)

nplt.plot_NB_decoding(NB_decode, plotdir=HA.save_dir, trial_group_size=20)

NB_summary = NB_decode.groupby(['exp_name', 'time_group', 'trial_ID', 'prestim_state', 'rec_dir', 'hmm_state']).agg(
    'mean').reset_index()

nplt.plot_NB_decoding(NB_decode, plotdir=HA.save_dir, trial_group_size=20, trial_col='session_trial')
nplt.plot_NB_decoding(NB_decode, plotdir=HA.save_dir, trial_group_size=5, trial_col='trial_num')

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
                                               trial_cols=['trial_num', 'session_trial'], save_dir=HA.save_dir,
                                               save_suffix='NB_timings')

time_plot_aov = time_aov.loc[(time_aov.Source == 'trial_group')].reset_index(drop=True)
time_plot_aov['F'] = time_plot_aov['F'].fillna(0)
sns.set(font_scale=1, style='white')
sns.catplot(data=time_plot_aov, x='n_trial_groups', y='F', hue='exp_group', col='time_group', row='trial_type',
            kind='boxen', margin_titles=True, aspect=0.5)

sns.catplot(data=time_plot_aov, x='n_trial_groups', y='p-GG-corr', hue='exp_group', col='time_group', row='trial_type',
            kind='boxen', margin_titles=True, aspect=0.5)


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
