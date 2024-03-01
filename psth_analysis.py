import blechpy
import numpy as np
import blechpy.dio.h5io as h5io
import pandas as pd
from joblib import Parallel, delayed
import trialwise_analysis as ta
import analysis as ana
import matplotlib.pyplot as plt
import feather
import seaborn as sns
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score
import matplotlib.gridspec as gridspec


proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
rec_info = proj.rec_info.copy()  # get the rec_info table
rec_dirs = rec_info['rec_dir']

PA = ana.ProjectAnalysis(proj)

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

#loop through each grouping of exp_group and rec_num, and for each grouping, calculate the average PSTH for each bin,
#as well as the 95% confidence interval for each bin, and save the results to a dataframe

taste_map = {'Suc': 0, 'NaCl': 1, 'CA': 2, 'QHCl': 3, 0: 'Suc', 1: 'NaCl', 2: 'CA', 3: 'QHCl'}
exp_group_map= {'naive': 0, 'suc_preexp': 1, 0: 'naive', 1: 'suc_preexp'}
session_map = {1: 0, 2: 1, 3: 2}
neuron_list = []
psth_list = []
for din in [0,1,2,3]:
    psth_mat = np.zeros((2, 3, 30, 500))
    n_neurons = np.zeros((2, 3))
    for name, group in rec_info.groupby(['exp_group', 'rec_num']):
        exp_group = name[0]
        session = name[1]
        exp_group_idx = exp_group_map[exp_group]
        session_idx = session_map[session]
        psth = np.zeros((30, 500))
        neurons = 0

        for i, row in group.iterrows():
            rec_dir = row['rec_dir']
            time_array, spike_array = h5io.get_spike_data(rec_dir, din=din)
            #rebin axis 2 by summing every 10ms
            spike_array = spike_array.reshape(spike_array.shape[0], spike_array.shape[1], -1, 10).sum(axis=3)
            spike_array = spike_array[:, :, 100:600]
            psth_array = spike_array.sum(axis=1)
            n_trials = spike_array.shape[0]

            neurons += spike_array.shape[1]
            n_neurons[exp_group_idx, session_idx] += spike_array.shape[1]
            psth[0:n_trials,:] += psth_array
        psth = psth / neurons
        psth_mat[exp_group_idx, session_idx, :, :] = psth
        n_neurons[exp_group_idx, session_idx] = neurons

    neuron_list.append(n_neurons)
    psth_list.append(psth_mat)

all_taste_psth = np.array(psth_list)
all_taste_psth = np.mean(all_taste_psth, axis=0)
all_taste_neurons = np.array(neuron_list)
all_taste_neurons = np.mean(all_taste_neurons, axis=0)

#get the max value of all_taste_psth
max_val = np.max(all_taste_psth)

#plot a 2x3 grid of heatmaps of the average PSTH for each taste and session
def plot_psth_heatmaps(matrices):
    exp_group_colors = ['Blues', 'Oranges']
    # get the maximum and minimum for every 3 entries of matrices

    # Adjust the figure layout
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.1])  # Extra column for the color bars on the left

    # Create axes for the plots, adjusting for the extra column for the color bars
    axs = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(2)]

    # Create separate color bars for each row in the first column
    cbar_axes = [fig.add_subplot(gs[i, 3]) for i in range(2)]

    for i in range(3):
        for j in range(2):
            mat = matrices[j, i, :, :]
            #scale mat from 0 to 1
            mat = mat / mat.max()
            mat = mat[:,50:400]
            exp_group = exp_group_map[j]
            session = i + 1

            ax = axs[j][i]  # Adjust for the shifted indexing due to color bar column
            cmap = plt.get_cmap(exp_group_colors[j])

            cax = ax.matshow(mat, vmin=0, vmax=1, origin='lower', cmap=cmap, aspect='auto')
            if i != 0:
                ax.set_yticks([])
            else:
                ax.set_ylabel('Trial', fontsize=20)
                ax.set_yticks([0, 9, 19, 29])
                ax.set_yticklabels(ax.get_yticks()+1, fontsize=14)
            if j != 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel('Trial', fontsize=20)
                #set the x ticks so that for every 100 indices, there is a tick, and set the tick labels to go from -2000 to 4000
                ax.set_xticks([50, 150, 250])
                ax.set_xticklabels(ax.get_xticks()*10-500, fontsize=14)
                ax.set_xlabel('Time (ms)', fontsize=20)
                ax.xaxis.set_ticks_position('bottom')
            if j == 0:
                ax.set_title('Session ' + str(session), pad=-6, fontsize=20)
            # Set the ylabel on the right side of the last column of the data plots
            if i == 2:  # Corrected condition to match the last column of the data plots
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(exp_group, rotation=270, labelpad=120, fontsize=20)
    
            # Add one color bar per row in the first column
            if i == 1:  # This ensures color bars are added once per row
                cb = fig.colorbar(cax, cax=cbar_axes[j], orientation='vertical')
                cbar_axes[j].yaxis.set_ticks_position('right')
                cbar_axes[j].yaxis.set_label_position('right')
                cbar_axes[j].set_ylabel('rate index', fontsize=20, rotation=270, labelpad=20)
                cb.set_ticks([0.2, 0.4, 0.6, 0.8])
    plt.subplots_adjust(right=0.85, left=0.1)
    plt.show()
    plt.savefig(PA.save_dir + 'psth_heatmaps.png')

plot_psth_heatmaps(all_taste_psth)

def plot_psth_heatmap2(matrices):
    exp_group_colors = ['Blues', 'Oranges']
    # get the maximum and minimum for every 3 entries of matrices

    # Adjust the figure layout
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.1])  # Extra column for the color bars on the left

    # Create axes for the plots, adjusting for the extra column for the color bars
    axs = [fig.add_subplot(gs[i]) for i in range(3)]

    # Create separate color bars for each row in the first column
    cbar_axes = fig.add_subplot(gs[3])

    for i in range(3):
        mat = matrices[i, :, :]
        # scale mat from 0 to 1
        mat = mat / mat.max()
        mat = mat[:, 50:400]
        session = i + 1

        ax = axs[i]  # Adjust for the shifted indexing due to color bar column

        cax = ax.matshow(mat, vmin=0, vmax=1, origin='lower', cmap='Blues', aspect='auto')

        if i == 0:
            ax.set_ylabel('Trial', fontsize=20)
            ax.set_yticks([0, 9, 19, 29])
            ax.set_yticklabels(ax.get_yticks() + 1, fontsize=14)
        else:
            ax.set_yticks([])

        ax.set_xlabel('Trial', fontsize=20)
        # set the x ticks so that for every 100 indices, there is a tick, and set the tick labels to go from -2000 to 4000
        ax.set_xticks([50, 150, 250])
        ax.set_xticklabels(ax.get_xticks() * 10 - 500, fontsize=14)
        ax.set_xlabel('Time (ms)', fontsize=20)
        ax.xaxis.set_ticks_position('bottom')

        ax.set_title('Session ' + str(session), pad=-6, fontsize=20)

        # Add one color bar per row in the first column
        if i == 1:  # This ensures color bars are added once per row
            cb = fig.colorbar(cax, cax=cbar_axes, orientation='vertical')
            cbar_axes.yaxis.set_ticks_position('right')
            cbar_axes.yaxis.set_label_position('right')
            cbar_axes.set_ylabel('rate index', fontsize=20, rotation=270, labelpad=20)
            cb.set_ticks([0.2, 0.4, 0.6, 0.8])

    plt.subplots_adjust(right=0.85, left=0.1)
    plt.tight_layout()
    plt.show()
    plt.savefig(PA.save_dir + 'naive_psth_heatmaps.png')


naive_mat = all_taste_psth[0, :, :, :]
plot_psth_heatmap2(naive_mat)


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

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
PA = ana.ProjectAnalysis(proj)
# get the held responses
held_resp = hua.get_held_resp(PA)

rec_info = proj.rec_info.copy()  # get the rec_info table

# filter out held_resp to include only exp_name == 'DS46', held==True, and taste_responsive==True
filtered = held_resp.query('exp_name == "DS46" and held==True')# and taste_responsive==True')

#get unique rows
filtered = filtered.drop_duplicates().reset_index(drop=True)
held_units = filtered['held_unit_name'].to_numpy()
#identify which numbers in held_units are repeated at least 3 times
unique, counts = np.unique(held_units, return_counts=True)
#extract the unit names that are repeated at least 3 times
repeated = unique[counts >= 12]

#filter out the rows in filtered that have held_unit_name in repeated
filtered = filtered[filtered['held_unit_name'].isin(repeated)]

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
filtered['half_responsive'] = filtered.groupby('held_unit_name')['taste_responsive'].transform(lambda x: x.sum() > 6)
#filter out the rows where two_thirds_responsive == False
filtered = filtered[filtered['half_responsive']==True]

#create a column called 'response_rank' and 'sd_rank' that ranks the mean_response and response_sd for each held_unit_name
filtered['response_rank'] = filtered.groupby(['taste', 'rec_dir'])['max_response'].rank(ascending=False)
filtered['sd_rank'] = filtered.groupby(['taste', 'rec_dir'])['max_sd'].rank(ascending=False)

#get the rows with the maximum response according to response_rank
max_response = filtered.query('response_rank < 15')

max_response = max_response.merge(rec_info, on=['rec_dir','exp_name', 'exp_group'])
#relabel rec_num to session
max_response = max_response.rename(columns={'rec_num': 'session'})

def plot_held_psth(group):
    taste_map = {'Suc': 0, 'NaCl': 1, 'CA': 2, 'QHCl': 3, 'Spont':4, 0: 'Suc', 1: 'NaCl', 2: 'CA', 3: 'QHCl', 4:'Spont'}
    session_map = {1: 0, 2: 1, 3: 2}
    exp_group_map = {'naive': 0, 'suc_preexp': 1, 0: 'naive', 1: 'suc_preexp'}
    dins = ["dig_in_0", "dig_in_1", "dig_in_2", "dig_in_3", "dig_in_4"]
    tastes = ['Suc', 'NaCl', 'CA', 'QHCl', 'Spont']
    sessions = [1, 2, 3]
    #make a subplot with 3 columns and 1 row
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i, session in enumerate(sessions):
        #for j, taste in enumerate(tastes):
        subset = group.query('session == @session')
        #time_array, spike_array = h5io.get_spike_data(subset['rec_dir'].iloc[0], units=[subset['unit_name'].iloc[0]])
        #time_array, spike_array = h5io.get_psths(subset['rec_dir'].iloc[0], units=[subset['unit_name'].iloc[0]])
        time_array, rate_arrray = h5io.get_rate_data(subset['rec_dir'].iloc[0], units=[subset['unit_name'].iloc[0]])
        spike_array = rate_arrray
        time_array = time_array[::250]
        dflist = []
        #loop throug the dictionary spike_array
        for channel, din in enumerate(dins):
            taste = tastes[channel]
            spikes = spike_array[din]
            #downsample
            n_trials = spikes.shape[0]
            reshaped_matrix = spikes.reshape(n_trials, -1, 250)
            spikes = reshaped_matrix.mean(axis=2)
            #repeat time_array so it has the same shape as spike_array
            time_mat = np.tile(time_array, (n_trials, 1))
            trial_array = np.arange(n_trials)
            spikes = list(spikes)
            time_mat = list(time_mat)
            trial_array = trial_array.tolist()
            df = pd.DataFrame({'spikes': spikes, 'timebin':time_mat, 'trial': trial_array})
            df = df.explode(['spikes', 'timebin']).reset_index(drop=True)
            df['taste'] = taste
            dflist.append(df)
        df = pd.concat(dflist)
        df = df.reset_index(drop=True)
        ax = axs[i]
        sns.lineplot(data=df, x='timebin', y='spikes', hue='taste', ax=ax)
        ax.set_title('Session ' + str(session))
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Firing rate')
    plt.suptitle("held unit#: " + str(group['held_unit_name'].iloc[0]))
    plt.show()

for name, group in max_response.groupby('held_unit_name'):
    plot_held_psth(group)