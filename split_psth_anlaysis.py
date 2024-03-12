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

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
PA = ana.ProjectAnalysis(proj)
# get the held responses
held_resp = hua.get_held_resp(PA)

rec_info = proj.rec_info.copy()  # get the rec_info table

# filter out held_resp to include only exp_name == 'DS46', held==True, and taste_responsive==True
filtered = held_resp.query('held==True')#'exp_name == "DS46"')# and held==True')# and taste_responsive==True')
#remove taste == 'Spont'
filtered = filtered[filtered['taste'] != 'Spont']
#get unique rows
filtered = filtered.drop_duplicates().reset_index(drop=True)
held_units = filtered['held_unit_name'].to_numpy()
dfs = []
for name, group in filtered.groupby(['held_unit_name']):
    taste_resp = group['taste_responsive'].tolist()
    group['any_responsive'] = any(taste_resp)
    group['all_responsive'] = all(taste_resp)
    dfs.append(group)
filtered = pd.concat(dfs)

#remove all rows where any_responsive == False
filtered = filtered[filtered['any_responsive'] == True]

taste_map = {'Suc': 0, 'NaCl': 1, 'CA': 2, 'QHCl': 3, 0: 'Suc', 1: 'NaCl', 2: 'CA', 3: 'QHCl', 'Spont': 4, 4: 'Spont'}
#loop through each row in filtered, and calculate the magnitude of each response
mean_responses = []
response_sds = []
for i, row in filtered.iterrows():
    rec_dir = row['rec_dir']
    unit_name = row['unit_name']
    taste = row['taste']
    din = taste_map[taste]

    time_array, spike_array = h5io.get_psths(rec_dir, din=din, units=[unit_name])
    #time_array, spike_array = h5io.get_spike_data(rec_dir, din=din, units=[unit_name])

    prestim = spike_array[:,0:1900]
    poststim = spike_array[:,2000:4000]
    prestim = prestim.sum(axis=1)
    poststim = poststim.sum(axis=1)
    response = poststim - prestim
    mean_response = response.mean()
    response_sd = response.std()
    mean_responses.append(mean_response)
    response_sds.append(response_sd)

filtered['mean_response'] = mean_responses
filtered['response_sd'] = response_sds

dfs = []
for name, group in filtered.groupby(['held_unit_name']):
    max_response = group['mean_response'].max()
    max_sd = group['response_sd'].max()
    group['max_response'] = max_response
    group['max_sd'] = max_sd
    dfs.append(group)
filtered = pd.concat(dfs)

#for each group of held_unit_name, calulate column 'two_thirds_responsive' which means that at least 2/3 of the rows contain a taste_responsive==True
#filtered['half_responsive'] = filtered.groupby('held_unit_name')['taste_responsive'].transform(lambda x: x.sum() > 6)
#filter out the rows where two_thirds_responsive == False
#filtered = filtered[filtered['half_responsive']==True]

#create a column called 'response_rank' and 'sd_rank' that ranks the mean_response and response_sd for each held_unit_name
filtered['response_rank'] = filtered.groupby(['taste', 'rec_dir'])['max_response'].rank(ascending=False)
filtered['sd_rank'] = filtered.groupby(['taste', 'rec_dir'])['max_sd'].rank(ascending=False)

#get the rows with the maximum response according to response_rank
max_response = filtered

max_response = max_response.merge(rec_info, on=['rec_dir','exp_name', 'exp_group'])
#relabel rec_num to session
max_response = max_response.rename(columns={'rec_num': 'session'})
#just get the rows of max_response with held_unit_numbers 25 and 86
max_response = max_response[max_response['held_unit_name'].isin([25.0, 86.0])]
#%% plot the PSTH for each taste and session for each held unit
#make a folder in the save_dir called 'held_unit_psth'
save_dir = PA.save_dir + '/split_held_unit_psth'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

from scipy.ndimage import gaussian_filter1d
def plot_split_held_psth(group, split_trial=12):
    taste_map = {'Suc': 0, 'NaCl': 1, 'CA': 2, 'QHCl': 3, 'Spont':4, 0: 'Suc', 1: 'NaCl', 2: 'CA', 3: 'QHCl', 4:'Spont'}
    session_map = {1: 0, 2: 1, 3: 2}
    exp_group_map = {'naive': 0, 'suc_preexp': 1, 0: 'naive', 1: 'suc_preexp'}
    dins = ["dig_in_0", "dig_in_1", "dig_in_2", "dig_in_3"]#, "dig_in_4"]
    tastes = ['Suc', 'NaCl', 'CA', 'QHCl', 'Spont']
    sessions = [1, 2, 3]
    split_groups = ['trials 1-' + str(split_trial), 'trials ' + str(split_trial+1) + '-30']
    #make a subplot with 3 columns and 1 row
    fig, axs = plt.subplots(2, 3, figsize=(10, 7), sharey=True, sharex=True)
    for i, session in enumerate(sessions):
        for j, split_group in enumerate(split_groups):
            subset = group.query('session == @session')
            if len(subset) == 0:
                continue
            time_array, rate_array = h5io.get_rate_data(subset['rec_dir'].iloc[0], units=[subset['unit_name'].iloc[0]])
            spike_array = rate_array
            time_array = time_array[::50]
            #get the indices of time array that are greater than -500 and less than 3000
            tidxs = np.where((time_array >= -500) & (time_array < 2500))[0]
            time_array = time_array[tidxs]
            dflist = []
            #loop throug the dictionary spike_array
            for channel, din in enumerate(dins):
                spikes = spike_array[din]
                n_trials = spikes.shape[0]
                reshaped_matrix = spikes.reshape(n_trials, -1, 50)
                spikes = reshaped_matrix.mean(axis=2)
                if j == 0:
                    spikes = spikes[:split_trial, tidxs]
                    n_trials = spikes.shape[0]
                    trial_array = np.arange(n_trials)
                else:
                    spikes = spikes[split_trial:, tidxs]
                    n_trials = spikes.shape[0]
                    trial_array = np.arange(split_trial, split_trial+n_trials)

                time_mat = np.tile(time_array, (n_trials, 1))

                spikes = list(spikes)
                time_mat = list(time_mat)
                trial_array = trial_array.tolist()
                df = pd.DataFrame({'spikes': spikes, 'timebin':time_mat, 'trial': trial_array})
                df = df.explode(['spikes', 'timebin']).reset_index(drop=True)
                taste = taste_map[channel]
                df['taste'] = taste
                dflist.append(df)
            df = pd.concat(dflist)
            df = df.reset_index(drop=True)
            ax = axs[j,i]
            if i == 2:
                #legend = True
                #on the right side axis of the plot, add a y label that is split_group
                ax2 = ax.twinx()
                ax2.set_ylabel(split_group, fontsize=20, rotation=-90, labelpad=20)
                #eliminate the y ticks and labels
                ax2.set_yticklabels([])
                ax2.set_yticks([])
            else:
                legend = False
            #if len of df is 0, raise an error
            if len(df) == 0:
                raise ValueError('DataFrame is empty')
            sns.lineplot(data=df, x='timebin', y='spikes', hue='taste', ax=ax, legend=legend)
            if j == 0:
                ax.set_title('Session ' + str(session))
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Firing rate')
    plt.suptitle("held unit#: " + str(group['held_unit_name'].iloc[0]))
    #pad the subplots so the title doesn't overlap with the plots
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.2, wspace=0.01, hspace=0.05, left=0.1, right=0.9)
    axs[1, 1].legend(['Sucrose', 'NaCl', 'CA', 'QHCl'], bbox_to_anchor=(0.5, -0.3), loc='upper center', ncol=4,
                     fontsize=20, fancybox=False, shadow=False)
    plt.show()
    en = str(group['exp_name'].iloc[0])
    hua = str(int(group['held_unit_name'].iloc[0]))
    save_name = en + '_' + hua
    save_path = save_dir + '/' + save_name
    plt.savefig(save_path + '.png')
    plt.savefig(save_path + '.svg')
    plt.close()

def plot_an(an_group):
    for name, group in an_group.groupby('held_unit_name'):
        plot_split_held_psth(group)
for name, group in max_response.groupby('exp_name'):
    plot_an(group)

Parallel(n_jobs=-1)(delayed(plot_an)(group) for name, group in max_response.groupby('exp_name'))