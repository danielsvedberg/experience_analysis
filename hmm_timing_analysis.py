#%% held unit based analysis
import numpy as np
import pandas as pd
import analysis as ana
import blechpy
import blechpy.dio.h5io as h5io
import held_unit_analysis as hua
import matplotlib.pyplot as plt
import seaborn as sns
import aggregation as agg
import os
from joblib import Parallel, delayed
import scipy.stats as stats
import hmm_analysis as hmma
from blechpy.analysis import poissonHMM as ph

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
HA = ana.HmmAnalysis(proj)  # create a hmm analysis object
timing = hmma.get_NB_states_and_probs(HA, get_best=True)

timing['t(early-late)'] = timing['t(end)']
#mark if epoch in each row == 'early'
timing['early-late-trans'] = timing['epoch'] == 'early'
#for each group of exp_name, session, and taste, identify if there is an early epoch
#if there is no early epoch, then set timing['early-late-trans'] to True and timing['t(early-late)'] to t(start)

#%%
naive_dat = timing[timing['exp_group'] == 'naive']
#naive_dat = naive_dat.loc[naive_dat['epoch'] == 'early']
naive_dat['anTaste'] = naive_dat['exp_name'] + '_' + naive_dat['taste']
#set format as str
naive_dat['anTaste'] = naive_dat['anTaste'].astype(str)
aov_groups = ['session', 'epoch']
within = ['trial_group', 'taste']
subject = 'exp_name' #'anTaste'
dvcols = ['t(start)','t(end)']#early-late)']


NB_split_aov, NB_split_ph, NB_split_diff = ana.iter_trial_split_anova(naive_dat, aov_groups, dvcols, within, subject,
                                                           trial_cols=['taste_trial'],#, 'session_trial'],
                                                           n_splits=30, save_dir=HA.save_dir, save_suffix='NB_decode_new')
NB_split_aov = NB_split_aov.loc[NB_split_aov['Source'] == 'trial_group']
NB_split_aov['significant'] = NB_split_aov['p-unc'] < (0.05/24)

#%% Now with average
NB_split_aov = NB_split_aov.loc[(NB_split_aov['trial_split'] > 2) & (NB_split_aov['trial_split'] < 27)]
NB_split_diff = NB_split_diff.loc[(NB_split_diff['trial_split'] > 2) & (NB_split_diff['trial_split'] < 27)]

epochs = ['late']#, 'late']#, 'average']
full_df = NB_split_aov #pd.concat([NB_split_aov, avg_df], ignore_index=True)
import matplotlib
tabblue = matplotlib.colors.to_rgba('tab:blue')
fig, axs = plt.subplots(1,3, figsize=(10, 4), sharex=True, sharey=True)
for i, epoch in enumerate(epochs):#(epoch, group) in enumerate(full_df.groupby('epoch')):
    group = full_df.loc[full_df['epoch'] == epoch]
    for j in [1,2,3]:
        ax = axs[j-1]

        diff_group = NB_split_diff.loc[(NB_split_diff['session'] == j) & (NB_split_diff['epoch'] == epoch)]
        sns.lineplot(x='trial_split', y='t(start)', data=diff_group, ax=ax, color='black')

        for k, (sig, grp) in enumerate(group.groupby('significant')):
            dat = grp.loc[group['session'] == j]
            if sig:
                sig_splits = dat['trial_split'].values
                sig_vals = diff_group.loc[diff_group['trial_split'].isin(sig_splits)]
                mean_val = sig_vals.groupby('trial_split')['t(start)'].mean().reset_index()
                ax.scatter(mean_val['trial_split'], mean_val['t(start)'], color=tabblue, s=50)
                # color = tabblue
                # ax2.scatter(dat['trial_split'], dat['p-GG-corr'], color=color, s=50)
            else:
                continue

        ax.axhline(0, color='black', linestyle='--')
        ax. set_xticks([3, 14, 26])
        if i == 0:
            ax.set_title('session ' + str(j))
        if i == 0:
            ax.set_xlabel('trial split')
            ax.tick_params(axis='x', labelsize=17)
            ax.set_xticklabels(ax.get_xticks()+1, fontsize=17)
        if j == 1:
            ax.set_ylabel('t(transition)\n late - early trials', fontsize=20)
            ax.tick_params(axis='y', labelsize=17)
        if j == 3:
            ax2 = ax.twinx()
            label = "early-late\ntransition"
            #ax2.set_ylabel(label, rotation=270, labelpad=50)
            ax2.set_yticks([])
            ax2.set_yticklabels([])
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.9, bottom=0.2, left=0.15, right=0.95)
plt.show()
save_dir = HA.save_dir
folder = 'NB_timing_split_trial_ANOVA'
if not os.path.exists(save_dir + os.sep + folder):
    os.makedirs(save_dir + os.sep + folder)
save_file = save_dir + os.sep + folder + os.sep + 'NB_timing_split_trial_ANOVA_wavg.svg'
fig.savefig(save_file)
save_file = save_dir + os.sep + folder + os.sep + 'NB_timing_split_trial_ANOVA_wavg.png'
fig.savefig(save_file)

#%%

epochs = ['best_acc']#, 'late']#, 'average']
full_df = NB_split_aov #pd.concat([NB_split_aov, avg_df], ignore_index=True)
import matplotlib
tabblue = matplotlib.colors.to_rgba('tab:blue')
fig, axs = plt.subplots(1,3, figsize=(10, 4), sharex=True, sharey=True)
for i, epoch in enumerate(epochs):#(epoch, group) in enumerate(full_df.groupby('epoch')):
    group = full_df.loc[full_df['epoch'] == epoch]
    for j in [1,2,3]:
        ax = axs[j-1]

        diff_group = NB_split_diff.loc[(NB_split_diff['session'] == j) & (NB_split_diff['epoch'] == epoch)]
        sns.lineplot(x='trial_split', y='t(end)', data=diff_group, ax=ax, color='black')

        for k, (sig, grp) in enumerate(group.groupby('significant')):
            dat = grp.loc[group['session'] == j]
            if sig:
                sig_splits = dat['trial_split'].values
                sig_vals = diff_group.loc[diff_group['trial_split'].isin(sig_splits)]
                mean_val = sig_vals.groupby('trial_split')['t(end)'].mean().reset_index()
                ax.scatter(mean_val['trial_split'], mean_val['t(end)'], color=tabblue, s=50)
                # color = tabblue
                # ax2.scatter(dat['trial_split'], dat['p-GG-corr'], color=color, s=50)
            else:
                continue

        ax.axhline(0, color='black', linestyle='--')
        ax. set_xticks([3, 14, 26])
        if i == 0:
            ax.set_title('session ' + str(j))
        if i == 0:
            ax.set_xlabel('trial split')
            ax.tick_params(axis='x', labelsize=17)
            ax.set_xticklabels(ax.get_xticks()+1, fontsize=17)
        if j == 1:
            ax.set_ylabel('t(transition)\n late - early trials', fontsize=20)
            ax.tick_params(axis='y', labelsize=17)
        if j == 3:
            ax2 = ax.twinx()
            label = "early-late\ntransition"
            #ax2.set_ylabel(label, rotation=270, labelpad=50)
            ax2.set_yticks([])
            ax2.set_yticklabels([])
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.9, bottom=0.2, left=0.15, right=0.95)
plt.show()
save_dir = HA.save_dir
folder = 'NB_timing_split_trial_ANOVA'
if not os.path.exists(save_dir + os.sep + folder):
    os.makedirs(save_dir + os.sep + folder)
save_file = save_dir + os.sep + folder + os.sep + 'NB_timing_split_trial_ANOVA_bestacc.svg'
fig.savefig(save_file)
save_file = save_dir + os.sep + folder + os.sep + 'NB_timing_split_trial_ANOVA_bestacc.png'
fig.savefig(save_file)

#%%
def get_pval_stars(pval):
    if pval < (0.001/3):
        return '***'
    elif pval < (0.01/3):
        return '**'
    elif pval < (0.05/3):
        return '*'
    else:
        return ''

#for each day, identify the best split by finding the split with the lowest p-GG-corr
best_aovs = []
best_dat = []
for (session, epoch), group in full_df.groupby(['session', 'epoch']):
    group = group.reset_index(drop=True)
    best_split = group.loc[group['p-GG-corr'] == group['p-GG-corr'].min()]['trial_split'].item()
    best_aovs.append(group.loc[group['p-GG-corr'] == group['p-GG-corr'].min()])

    naive_sess = naive_dat.loc[naive_dat['session'] == session].reset_index(drop=True)
    if epoch != 'average':
        naive_sess = naive_sess.loc[naive_sess['epoch'] == epoch].reset_index(drop=True)
    else:
        naive_sess['epoch'] = 'average'

    naive_sess['trial group'] = naive_sess['taste_trial'] > best_split
    min_trial = naive_sess['taste_trial'].min()
    max_trial = naive_sess['taste_trial'].max()
    pre_split_label = str(min_trial+1) + '-' + str(int(best_split+1))
    post_split_label = str(int(best_split)+2) + '-' + str(max_trial+1)
    print(pre_split_label, post_split_label)
    label_dict = {False: pre_split_label, True: post_split_label}
    naive_sess['trial group'] = naive_sess['trial group'].map(label_dict)
    best_dat.append(naive_sess)
best_dat = pd.concat(best_dat).reset_index(drop=True)
best_aovs = pd.concat(best_aovs).reset_index(drop=True)


chance_level = 1/(3*3)

fig, axs = plt.subplots(1,3, figsize=(10, 4), sharex=False, sharey=True)
for session in [1,2,3]:
    for e, epoch in enumerate(['early']):#, 'average']):
        ax = axs[session-1]
        stats_dat = best_aovs.loc[best_aovs['session'] == session]
        stats_dat = stats_dat.loc[stats_dat['epoch'] == epoch]
        session_dat = best_dat.loc[best_dat['session'] == session]
        session_dat = session_dat.loc[session_dat['epoch'] == epoch]

        sns.barplot(x='trial group', y='t(end)', data=session_dat, ax=ax, color='lightgray')
        if stats_dat['significant'].item():
            #get the 95th percentile of session_dat['t(end)']
            max_val = session_dat['t(end)'].mean() * 1.2
            #make a horizontal line at 1.1 times the max value of session_dat['t(end)'], spanning the midpoints of the bars
            ax.hlines(max_val, 0,1, color='black')
            #put a star at the top of the line
            ax.text(0.5, max_val, get_pval_stars(stats_dat['p-GG-corr'].item()), ha='center', va='bottom', fontsize=20)
            #set the y limit to 1.1 times the max value of session_dat['t(end)']
            ax.set_ylim(0, max_val * 1.2)

        if e == 0:
            ax.set_title('session ' + str(session))

        if session != 1:
            ax.set_ylabel('', fontsize=17)
        else:
            ax.set_ylabel('time (ms)', fontsize=20)
        if session == 3:
            ax2 = ax.twinx()
            label = 'early-late\ntransition'
            ax2.set_ylabel(label, rotation=270, labelpad=50, fontsize=20)
            ax2.set_yticks([])

plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.15, top=0.9, bottom=0.175)
plt.show()

#save
save_dir = HA.save_dir
folder = 'NB_timing_split_trial_ANOVA'
if not os.path.exists(save_dir + os.sep + folder):
    os.makedirs(save_dir + os.sep + folder)
save_file = save_dir + os.sep + folder + os.sep + 'NB_timing_split_trial_ANOVA_bar.svg'
fig.savefig(save_file)
save_file = save_dir + os.sep + folder + os.sep + 'NB_timing_split_trial_ANOVA_bar.png'
fig.savefig(save_file)

#%%
timing['trial_group'] = timing['taste_trial'] > 10
timing['trial_group'] = timing['trial_group'].map({False: 'early', True: 'late'})
timing = timing.loc[timing['exp_group'] == 'naive']

fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey=True, sharex=True)
for i, session in enumerate([1,2,3]):
    for j, trial_group in enumerate(['early', 'late']):
        ax = axs[j, i]
        session_timing = timing[timing['session'] == session]
        session_timing = session_timing[session_timing['trial_group'] == trial_group].reset_index(drop=True)
        #order session_timing by state_num
        #session_timing = session_timing.sort_values(by='state_num')
        #turn state num into a string
        #session_timing['state_num'] = session_timing['state_num'].astype(str)
        sns.histplot(data=session_timing, x="t(end)", multiple="stack", stat="density", ax=ax, legend=False)
        #plot the median of each state
        medians = session_timing['t(end)'].median()
        ax.axvline(medians)#, color=f'C{statenum}', linestyle='--')
plt.show()

indices = []
cdfs = []
for name, group in timing.groupby(['exp_name','session','exp_group','taste','taste_trial', 'trial_group']):
    group = group.reset_index(drop=True)
    #make a cumulative distribution array of the start times
    n_states = len(group)
    cdf = np.zeros((n_states,2000))
    for i, row in group.iterrows():
        #make an array of zeros 2000 indices long
        t_start = int(row['t_start'])
        cdf[i, t_start:] = 1
    #sum cdf
    cdf = cdf.sum(axis=0)
    indices.append(name)
    cdfs.append(cdf)

#make a dataframe:
#turn names into separate lists of exp_name, session, exp_group, taste, trial
exp_name = [i[0] for i in indices]
session = [i[1] for i in indices]
exp_group = [i[2] for i in indices]
taste = [i[3] for i in indices]
trial = [i[4] for i in indices]
trial_group = [i[5] for i in indices]
cdf_df = pd.DataFrame({'exp_name': exp_name, 'session': session, 'exp_group': exp_group, 'taste': taste, 'taste_trial': trial, 'trial_group':trial_group, 'cdf': cdfs})

fig, axs = plt.subplots(4, 3, figsize=(15, 20), sharey=True, sharex=True)
for i, session in enumerate([1,2,3]):
    for j, taste in enumerate(['Suc', 'NaCl','CA','QHCl']):
        for k, tg in enumerate(['early', 'late']):
            ax = axs[j,i]
            session_cdf = cdf_df[cdf_df['session'] == session]
            session_cdf = session_cdf[session_cdf['taste'] == taste]
            session_cdf = session_cdf[session_cdf['trial_group'] == tg].reset_index(drop=True)
            cdfs = np.array(session_cdf['cdf'].tolist())
            cdf = cdfs.mean(axis=0)
            ax.plot(cdf)
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
for i, session in enumerate([1,2,3]):
    for k, tg in enumerate(['early', 'late']):
        ax = axs[i]
        session_cdf = cdf_df[cdf_df['session'] == session]
        session_cdf = session_cdf[session_cdf['trial_group'] == tg].reset_index(drop=True)
        cdfs = np.array(session_cdf['cdf'].tolist())
        cdf = cdfs.mean(axis=0)
        ax.plot(cdf)
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharey=True, sharex=True)
for i, session in enumerate([1,2,3]):
    ax = axs
    session_cdf = cdf_df[cdf_df['session'] == session]
    session_cdf = session_cdf[session_cdf['trial_group'] == tg].reset_index(drop=True)
    cdfs = np.array(session_cdf['cdf'].tolist())
    cdf = cdfs.mean(axis=0)
    ax.plot(cdf)
plt.show()


#%%
import os
import analysis as ana
import hmm_analysis as hmma
import blechpy
import trialwise_analysis as ta
group_cols = ['exp_group', 'session', 'taste']
trial_col = 'taste_trial'
nIter = 1000

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
HA = ana.HmmAnalysis(proj)  # create a hmm analysis object
timing = hmma.get_NB_states_and_probs(HA, get_best=True)

save_dir = HA.save_dir
#make a new folder called nonlinear_regression in save_dir
folder = 'nonlinear_regression_timing'
save_dir = save_dir + os.sep + folder
if not os.path.exists(save_dir):
    os.makedirs(save_dir + os.sep + folder)

timing_early = timing[timing['epoch'] == 'early']
value_col = 't(end)'
flag = 'early_epoch'
df3, shuff = ta.preprocess_nonlinear_regression(timing_early, subject_col='exp_name', group_cols=group_cols,
                                                trial_col=trial_col, value_col=value_col, overwrite=True,
                                                nIter=nIter, save_dir=save_dir, flag=flag)
ta.plotting_pipeline(df3, shuff, trial_col, value_col, nIter=nIter, save_dir=save_dir, flag=flag)

value_col = 't(start)'
flag = 'early_epoch'
df3, shuff = ta.preprocess_nonlinear_regression(timing_early, subject_col='exp_name', group_cols=group_cols,
                                                trial_col=trial_col, value_col=value_col, overwrite=True,
                                                nIter=nIter, save_dir=save_dir, flag=flag)
ta.plotting_pipeline(df3, shuff, trial_col, value_col, nIter=nIter, save_dir=save_dir, flag=flag)



timing_late = timing[timing['epoch'] == 'late']
value_col = 't(start)'
flag = 'late_epoch'
df3, shuff = ta.preprocess_nonlinear_regression(timing_late, subject_col='exp_name', group_cols=group_cols,
                                                trial_col=trial_col, value_col=value_col, overwrite=True,
                                                nIter=nIter, save_dir=save_dir, flag=flag)
ta.plotting_pipeline(df3, shuff, trial_col, value_col, nIter=nIter, save_dir=save_dir, flag=flag)


timing_best_acc = timing[timing['epoch'] == 'best_acc']
value_col = 't(start)'
flag = 'best_acc_epoch'
df3, shuff = ta.preprocess_nonlinear_regression(timing_best_acc, subject_col='exp_name', group_cols=group_cols,
                                                trial_col=trial_col, value_col=value_col, overwrite=True,
                                                nIter=nIter, save_dir=save_dir, flag=flag)
ta.plotting_pipeline(df3, shuff, trial_col, value_col, nIter=nIter, save_dir=save_dir, flag=flag)

