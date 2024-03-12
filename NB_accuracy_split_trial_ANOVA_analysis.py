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

NB_df = HA.analyze_NB_ID2(overwrite=False)
NB_df['duration'] = NB_df['t_end'] - NB_df['t_start']
NB_df['t_med'] = (NB_df['t_end'] + NB_df['t_start']) / 2

#for each rec_dir, subtract the min of off_time from all off_times
NB_df['off_time'] = NB_df.groupby('rec_dir')['off_time'].apply(lambda x: x - x.min())

#for each grouping of taste and rec_dir, make a new column called 'length_rank' ranking the states' length
NB_df['avg_t_start'] = NB_df.groupby(['taste', 'rec_dir','state'])['t_start'].transform('mean')
NB_df['avg_duration'] = NB_df.groupby(['taste', 'rec_dir','state'])['duration'].transform('mean')
NB_df = NB_df.loc[:, 't_start':]

NB_df['pr(correct state)'] = NB_df['p_correct']
NB_df['session time'] = NB_df['off_time']
NB_df['t(median)'] = NB_df['t_med']
NB_df['t(start)'] = NB_df['t_start']
#get rid of all columns before t_start

NB_df_accuracy = NB_df.reset_index(drop=True)
#get rid of all rows where avg_t_start is 0
#NB_df_accuracy = NB_df_accuracy.loc[NB_df_accuracy['avg_t_start'] != 0]
#get rid of all rows where avg_t_start is less than 1500
NB_df_accuracy = NB_df_accuracy.loc[NB_df_accuracy['avg_t_start'] < 1500]
NB_df_accuracy = NB_df_accuracy.loc[NB_df_accuracy['avg_t_start'] > 0]
NB_df_accuracy['trial_duration_rank'] = NB_df_accuracy.groupby(['taste', 'rec_dir','taste_trial'])['duration'].rank(ascending=False)
NB_df_accuracy['trial_accuracy_rank'] = NB_df_accuracy.groupby(['taste', 'rec_dir','taste_trial'])['pr(correct state)'].rank(ascending=False)
NB_df_accuracy = NB_df_accuracy.loc[NB_df_accuracy['trial_accuracy_rank'] <= 2]
NB_df_accuracy['trial_order_rank'] = NB_df_accuracy.groupby(['taste', 'rec_dir','taste_trial'])['t_start'].rank(ascending=True, method='first')

order_map = {1:'early', 2:'late'}
NB_df_accuracy['epoch'] = NB_df_accuracy['trial_order_rank'].map(order_map)

NB_df_naive = NB_df_accuracy.loc[NB_df_accuracy['exp_group'] == 'naive'].reset_index(drop=True)
NB_df_naive = NB_df_naive[['exp_name','session','taste','taste_trial','epoch','pr(correct state)']]


trial_info = proj.get_dig_in_trial_df(reformat=True)
trial_info = trial_info.loc[trial_info['exp_group'] == 'naive'].reset_index(drop=True)
trial_info = trial_info.loc[trial_info['taste'] != 'Spont'].reset_index(drop=True)
#make a copy of trial_info with an epoch column filled with 'early'
trial_info['epoch'] = 'early'
#make a copy of trial_info with an epoch column filled with 'late'
trial_info_late = trial_info.copy()
trial_info_late['epoch'] = 'late'
#join the two
trial_info = pd.concat([trial_info, trial_info_late], ignore_index=True).reset_index(drop=True)
trial_info = trial_info[['exp_name','session','taste','taste_trial','epoch']]


#merge to fill in the missing trials from taste_trial and session_trial using trial_info
merge = trial_info.merge(NB_df_naive, on=['exp_name','session','taste','taste_trial','epoch'], how='left').fillna(0)

aov_groups = ['session', 'epoch']
within = ['taste']
subject = 'exp_name'
dvcols = ['pr(correct state)']
NB_split_aov, NB_split_ph = ana.iter_trial_split_anova(merge, aov_groups, dvcols, within, subject,
                                                           trial_cols=['taste_trial'],#, 'session_trial'],
                                                           n_splits=30, save_dir=HA.save_dir, save_suffix='NB_decode_new')
NB_split_aov = NB_split_aov.loc[NB_split_aov['Source'] == 'trial_group']
NB_split_aov['significant'] = NB_split_aov['p-GG-corr'] < 0.05

import matplotlib
tabblue = matplotlib.colors.to_rgba('tab:blue')
fig, axs = plt.subplots(2,3, figsize=(10, 5), sharex=True, sharey=True)
for i, (epoch, group) in enumerate(NB_split_aov.groupby('epoch')):
    for j in [1,2,3]:
        ax = axs[i, j-1]
        for k, (sig, grp) in enumerate(group.groupby('significant')):
            dat = grp.loc[group['session'] == j]
            if sig:
                color = 'r'
            else:
                color = tabblue
            #plot with enlarged points
            ax.scatter(dat['trial_split'], dat['p-GG-corr'], color=color, s=50)
        if i == 0:
            ax.set_title('session ' + str(j))
        if i == 1:
            ax.set_xlabel('trial split')
            ax.set_xticklabels(ax.get_xticks(), fontsize=17)
        if j == 1:
            ax.set_ylabel('p-value')
            ax.set_yticklabels(ax.get_yticks(), fontsize=17)
        if j == 3:
            ax2 = ax.twinx()
            ax2.set_ylabel(epoch + '\nepoch', rotation=270, labelpad=50)
            ax2.set_yticks([])
            ax2.set_yticklabels([])
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.9, bottom=0.2, left=0.1, right=0.9)
plt.show()
save_dir = HA.save_dir
folder = 'NB_accuracy_split_trial_ANOVA'
if not os.path.exists(save_dir + os.sep + folder):
    os.makedirs(save_dir + os.sep + folder)
save_file = save_dir + os.sep + folder + os.sep + 'NB_accuracy_split_trial_ANOVA.svg'
fig.savefig(save_file)
save_file = save_dir + os.sep + folder + os.sep + 'NB_accuracy_split_trial_ANOVA.png'
fig.savefig(save_file)


#%% Now with average
avg_df = []
for name, group in NB_split_aov.groupby(['trial_split', 'session']):
    #take the mean of all columns except p-GG-corr and significant
    avg = group.mean()
    combined_pval = stats.combine_pvalues(group['p-GG-corr'].values, method='fisher')
    avg['p-GG-corr'] = combined_pval[1]
    avg['significant'] = combined_pval[1] < 0.05
    avg_df.append(avg)

avg_df = pd.concat(avg_df, axis=1).T
avg_df['epoch'] = 'average'
epochs = ['early', 'late', 'average']
full_df = pd.concat([NB_split_aov, avg_df], ignore_index=True)
tabblue = matplotlib.colors.to_rgba('tab:blue')
fig, axs = plt.subplots(3,3, figsize=(10, 8), sharex=True, sharey=True)
for i, epoch in enumerate(epochs):#(epoch, group) in enumerate(full_df.groupby('epoch')):
    group = full_df.loc[full_df['epoch'] == epoch]
    for j in [1,2,3]:
        ax = axs[i, j-1]
        for k, (sig, grp) in enumerate(group.groupby('significant')):
            dat = grp.loc[group['session'] == j]
            if sig:
                color = tabblue
            else:
                color = 'gray'
            #plot with enlarged points
            ax.scatter(dat['trial_split'], dat['p-GG-corr'], color=color, s=50)
            #put a blue line at p = 0.05
            ax.axhline(0.05, color='b', linestyle='--')
        if i == 0:
            ax.set_title('session ' + str(j))
        if i == 2:
            ax.set_xlabel('trial split')
            ax.tick_params(axis='x', labelsize=17)
        if j == 1:
            ax.set_ylabel('p-value')
            ax.tick_params(axis='y', labelsize=17)
        if j == 3:
            ax2 = ax.twinx()
            if epoch =='average':
                label = 'combined'
            else:
                label = epoch + '\nepoch'
            ax2.set_ylabel(label, rotation=270, labelpad=50)
            ax2.set_yticks([])
            ax2.set_yticklabels([])
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.9, bottom=0.1, left=0.1, right=0.9)
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

best_split = avg_df.loc[avg_df['p-GG-corr'] == avg_df['p-GG-corr'].min()]['trial_split'].item()
# for each grouping of session, find the value of trial_split where p-GG-corr is the lowest
avg_df['best_split'] = avg_df.groupby('session').apply(lambda x: x.loc[x['p-GG-corr'] == x['p-GG-corr'].min()]['trial_split'].item())
merge['trial group'] = merge['taste_trial'] > best_split

trial_split_labels = ['1-' + str(int(best_split)), str(int(best_split+1)) + '-30']
merge['trial group'] = merge['trial group'].map({False: trial_split_labels[0], True: trial_split_labels[1]})
merge['pr(correct state)'] = merge['pr(correct state)']

best_split_stats = full_df.loc[full_df['trial_split'] == best_split]

chance_level = 1/(3*3)

fig, axs = plt.subplots(3,3, figsize=(10, 8), sharex=True, sharey=True)
for session in [1,2,3]:
    for e, epoch in enumerate(['early','late', 'average']):
        ax = axs[e, session-1]
        stats_dat = best_split_stats.loc[best_split_stats['session'] == session]
        stats_dat = stats_dat.loc[stats_dat['epoch'] == epoch]
        session_dat = merge.loc[merge['session'] == session]
        if e != 2:
            session_dat = merge.loc[merge['epoch'] == epoch]
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
        if e != 2:
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
plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.925, bottom=0.1, left=0.1, right=0.9)
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
