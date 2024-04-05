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
import hmm_analysis as hmma

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
HA = ana.HmmAnalysis(proj)  # create a hmm analysis object

NB_states = hmma.get_NB_states_and_probs(HA, get_best=True)

#%%
naive_dat = NB_states[NB_states['exp_group'] == 'naive']
naive_dat['anTaste'] = naive_dat['taste'] + '_' + naive_dat['epoch']
aov_groups = ['session', 'epoch']
within = ['trial_group', 'taste']
subject = 'exp_name'
dvcols = ['pr(correct state)']
NB_split_aov, NB_split_ph, NB_split_diff = ana.iter_trial_split_anova(naive_dat, aov_groups, dvcols, within, subject,
                                                           trial_cols=['taste_trial'],#, 'session_trial'],
                                                           n_splits=30, save_dir=HA.save_dir, save_suffix='NB_decode_new')
NB_split_aov = NB_split_aov.loc[NB_split_aov['Source'] == 'trial_group']
NB_split_aov['significant'] = NB_split_aov['p-unc'] < (0.05/24)

#%% Now with average

epochs = ['early', 'late', 'best_acc']#, 'average']
full_df = NB_split_aov #pd.concat([NB_split_aov, avg_df], ignore_index=True)

NB_split_diff = NB_split_diff.loc[(NB_split_diff['trial_split'] > 2) & (NB_split_diff['trial_split'] < 27)]
full_df = full_df.loc[(full_df['trial_split'] > 2) & (full_df['trial_split'] < 27)]
import matplotlib
tabblue = matplotlib.colors.to_rgba('tab:blue')
fig, axs = plt.subplots(3,3, figsize=(10, 10), sharex=True, sharey=True)
for i, epoch in enumerate(epochs):#(epoch, group) in enumerate(full_df.groupby('epoch')):
    group = full_df.loc[full_df['epoch'] == epoch]
    for j in [1,2,3]:
        diff_group = NB_split_diff.loc[(NB_split_diff['session'] == j) & (NB_split_diff['epoch'] == epoch)]
        ax = axs[i, j-1]
        sns.lineplot(x='trial_split', y='pr(correct state)', data=diff_group, ax=ax, color='black')
        ax.axhline(0, color='black', linestyle='--')
        ax2 = ax.twinx()
        for k, (sig, grp) in enumerate(group.groupby('significant')):
            dat = grp.loc[group['session'] == j]
            if sig:
                sig_splits = dat['trial_split'].values
                sig_vals = diff_group.loc[diff_group['trial_split'].isin(sig_splits)]
                mean_val = sig_vals.groupby('trial_split')['pr(correct state)'].mean().reset_index()
                ax.scatter(mean_val['trial_split'], mean_val['pr(correct state)'], color=tabblue, s=50)
                #color = tabblue
                #ax2.scatter(dat['trial_split'], dat['p-GG-corr'], color=color, s=50)
            else:
                continue

        if i == 0:
            ax.set_title('session ' + str(j))
        ax.set_xticks([3, 14, 26])
        if i == 1:
            ax.set_xlabel('trial split')
            ax.set_xticklabels(ax.get_xticks()+1, fontsize=17)
            ax.tick_params(axis='x', labelsize=17)
        if j == 1:
            ax.set_ylabel('late - early trials\npr(correct state)', fontsize=20)
            ax.tick_params(axis='y', labelsize=17)

        if j == 3:
            label = epoch + ' epoch'
            ax2.set_ylabel(label, fontsize=20)#, rotation=270)#, labelpad=50)
        else:
            ax2.set_ylabel('')
        ax2.set_yticklabels([])
        ax2.set_yticks([])

plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.9, bottom=0.125, left=0.15, right=0.95)
plt.show()
save_dir = HA.save_dir
folder = 'NB_accuracy_split_trial_ANOVA'
if not os.path.exists(save_dir + os.sep + folder):
    os.makedirs(save_dir + os.sep + folder)
save_file = save_dir + os.sep + folder + os.sep + 'NB_accuracy_split_trial_ANOVA_wavg.svg'
fig.savefig(save_file)
save_file = save_dir + os.sep + folder + os.sep + 'NB_accuracy_split_trial_ANOVA_wavg.png'
fig.savefig(save_file)

#%%
def get_pval_stars(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
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

fig, axs = plt.subplots(2,3, figsize=(10, 7), sharex=False, sharey=True)
for session in [1,2,3]:
    for e, epoch in enumerate(['early','late']):#, 'average']):
        ax = axs[e, session-1]
        stats_dat = best_aovs.loc[best_aovs['session'] == session]
        stats_dat = stats_dat.loc[stats_dat['epoch'] == epoch]
        session_dat = best_dat.loc[best_dat['session'] == session]
        session_dat = session_dat.loc[session_dat['epoch'] == epoch]

        sns.barplot(x='trial group', y='pr(correct state)', data=session_dat, ax=ax, color='lightgray')
        if stats_dat['significant'].item():
            #get the 95th percentile of session_dat['pr(correct state)']
            max_val = session_dat['pr(correct state)'].mean() * 1.2
            #make a horizontal line at 1.1 times the max value of session_dat['pr(correct state)'], spanning the midpoints of the bars
            ax.hlines(max_val, 0,1, color='black')
            #put a star at the top of the line
            ax.text(0.5, max_val, get_pval_stars(stats_dat['p-GG-corr'].item()), ha='center', va='bottom', fontsize=20)
            #set the y limit to 1.1 times the max value of session_dat['pr(correct state)']
            ax.set_ylim(0, max_val * 1.2)
        ax.hlines(chance_level, -0.5, 1.5, color='black', linestyle='--')

        if e != 1:
            ax.set_xlabel('')

        if e == 0:
            ax.set_title('session ' + str(session))

        if session != 1:
            ax.set_ylabel('', fontsize=16)
        else:
            ax.set_ylabel('pr(correct state)', fontsize=17)

        if session == 3:
            ax2 = ax.twinx()
            ax2.set_ylabel(epoch + '\nepoch', rotation=270, labelpad=50, fontsize=20)
            ax2.set_yticks([])


plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.2, top=0.925, bottom=0.1, left=0.1, right=0.9)
plt.show()

#save
save_dir = HA.save_dir
folder = 'NB_accuracy_split_trial_ANOVA'
if not os.path.exists(save_dir + os.sep + folder):
    os.makedirs(save_dir + os.sep + folder)
save_file = save_dir + os.sep + folder + os.sep + 'NB_accuracy_split_trial_ANOVA_bar.svg'
fig.savefig(save_file)
save_file = save_dir + os.sep + folder + os.sep + 'NB_accuracy_split_trial_ANOVA_bar.png'
fig.savefig(save_file)

#%%
import trialwise_analysis as ta
import blechpy
import hmm_analysis as hmma
import analysis as ana
import os

proj = blechpy.load_project('/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy')
HA = ana.HmmAnalysis(proj)
save_dir = HA.save_dir
#make a new folder called nonlinear_regression in save_dir
folder = 'nonlinear_regression'
save_dir = save_dir + os.sep + folder
if not os.path.exists(save_dir):
    os.makedirs(save_dir + os.sep + folder)

NB_df_accuracy = NB_states

late = NB_df_accuracy.loc[NB_df_accuracy['epoch'] == 'late']
group_cols = ['exp_group', 'session', 'taste']
value_col = 'pr(correct state)'
trial_col = 'taste_trial'
nIter = 10000
flag = 'late_epoch'
df3, shuff = ta.preprocess_nonlinear_regression(late, subject_col='exp_name', group_cols=group_cols,
                                                trial_col=trial_col, value_col=value_col, overwrite=False,
                                                nIter=nIter, save_dir=save_dir, flag=flag)
ta.plotting_pipeline(df3, shuff, trial_col, value_col, nIter=nIter, save_dir=save_dir, flag=flag)

early = NB_df_accuracy.loc[NB_df_accuracy['epoch'] == 'early']
df3, shuff = ta.preprocess_nonlinear_regression(early, subject_col='exp_name', group_cols=group_cols,
                                                trial_col=trial_col, value_col=value_col, overwrite=False,
                                                nIter=nIter, save_dir=save_dir, flag='early_epoch')
ta.plotting_pipeline(df3, shuff, trial_col, value_col, nIter=nIter, save_dir=save_dir, flag='early_epoch')


best = NB_df_accuracy.loc[NB_df_accuracy['epoch'] == 'best_acc']
df3, shuff = ta.preprocess_nonlinear_regression(best, subject_col='exp_name', group_cols=group_cols,
                                                trial_col=trial_col, value_col=value_col, overwrite=False,
                                                nIter=nIter, save_dir=save_dir, flag='best_acc_epoch')
ta.plotting_pipeline(df3, shuff, trial_col, value_col, nIter=nIter, save_dir=save_dir, flag='best_acc_epoch')
