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

#%% plot the PSTH for each taste and session for each held unit
#make a folder in the save_dir called 'held_unit_psth'
save_dir = PA.save_dir + '/held_unit_psth'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

from scipy.ndimage import gaussian_filter1d
def plot_held_psth(group):
    taste_map = {'Suc': 0, 'NaCl': 1, 'CA': 2, 'QHCl': 3, 'Spont':4, 0: 'Suc', 1: 'NaCl', 2: 'CA', 3: 'QHCl', 4:'Spont'}
    session_map = {1: 0, 2: 1, 3: 2}
    exp_group_map = {'naive': 0, 'suc_preexp': 1, 0: 'naive', 1: 'suc_preexp'}
    dins = ["dig_in_0", "dig_in_1", "dig_in_2", "dig_in_3"]#, "dig_in_4"]
    tastes = ['Suc', 'NaCl', 'CA', 'QHCl', 'Spont']
    sessions = [1, 2, 3]
    #make a subplot with 3 columns and 1 row
    fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True, sharex=True)
    for i, session in enumerate(sessions):
        #for j, taste in enumerate(tastes):
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
            #spikes = gaussian_filter1d(spikes, 2, axis=1)
            spikes = spikes[:, tidxs]
            time_mat = np.tile(time_array, (n_trials, 1))
            trial_array = np.arange(n_trials)
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
        ax = axs[i]
        if i == 2:
            legend = True
        else:
            legend = False
        #if len of df is 0, raise an error
        if len(df) == 0:
            raise ValueError('DataFrame is empty')
        sns.lineplot(data=df, x='timebin', y='spikes', hue='taste', ax=ax, legend=legend)
        ax.set_title('Session ' + str(session), fontsize=20)
        ax.set_xlabel('Time (ms)', fontsize=20)
        ax.set_ylabel('Firing rate', fontsize=20)
        #set the x and y font size
        ax.tick_params(axis='both', which='major', labelsize=17)
    plt.suptitle("held unit#: " + str(group['held_unit_name'].iloc[0]))
    #pad the subplots so the title doesn't overlap with the plots
    plt.tight_layout()
    plt.subplots_adjust(top=0.825)#, right=0.95)
    en = str(group['exp_name'].iloc[0])
    hua = str(group['held_unit_name'].iloc[0])
    save_name = en + '_' + hua
    save_path = save_dir + '/' + save_name
    plt.savefig(save_path + '.png')
    plt.savefig(save_path + '.svg')
    plt.close()

def plot_an(an_group):
    for name, group in an_group.groupby('held_unit_name'):
        plot_held_psth(group)

Parallel(n_jobs=-1)(delayed(plot_an)(group) for name, group in max_response.groupby('exp_name'))

#%%
import scipy.stats as stats
#now, for every unit in max_response, calculate the taste discrimination index for each unit
# Parameters
window_length = 250  # ms
step_size = 25
num_time_bins = 7000
num_windows = (num_time_bins - window_length) // step_size + 1
tastes = ['Suc', 'NaCl', 'CA', 'QHCl']
time_idx = np.arange(-2000, 4775, 25)

# Function to calculate firing rates with rolling window
def calculate_firing_rates(spike_counts, window_length, step_size):
    num_trials = spike_counts.shape[0]
    num_windows = (num_time_bins - window_length) // step_size + 1
    firing_rates = np.zeros((num_trials, num_windows))
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_length
        firing_rates[:, i] = spike_counts[:, start_idx:end_idx].sum(axis=1) / (window_length / 1000.0)  # Convert to Hz
    return firing_rates

target_df = max_response
#isolate the columns  'rec_dir', 'exp_name', 'exp_group', 'session', 'unit_num', 'held_unit_name'
target_df = target_df[['rec_dir', 'exp_name', 'exp_group', 'session', 'unit_num', 'held_unit_name']]
#get unique rows
target_df = target_df.drop_duplicates().reset_index(drop=True)
pval_list = []
significant_windows_list = []
dins = ["dig_in_0", "dig_in_1", "dig_in_2", "dig_in_3"]#, "dig_in_4"]
tastes = ['Suc', 'NaCl', 'CA', 'QHCl']#, 'Spont']
for i, row in target_df.iterrows():
    rec_dir = row['rec_dir']
    unit_num = row['unit_num']
    #get the spike data
    time_array, responses = h5io.get_spike_data(rec_dir, units=[unit_num])
    #dins = list(responses.keys())
    #replace the keys of the dictionary with tastes
    responses = {tastes[i]: responses[dins[i]] for i in range(len(dins))}

    # Calculate firing rates for each taste
    firing_rates_dict = {taste: calculate_firing_rates(responses[taste], window_length, step_size) for taste in tastes}

    # Perform ANOVA for each window
    p_values = []
    for i in range(firing_rates_dict[tastes[0]].shape[1]):
        current_window_data = [firing_rates_dict[taste][:, i] for taste in tastes]
        f_val, p_val = stats.f_oneway(*current_window_data)
        p_values.append(p_val)
    pval_list.append(p_values)
    # Identify significant windows (part of three consecutive windows with p < 0.05)
    significant_windows = []
    for i in range(1, len(p_values) - 1):
        if p_values[i - 1] < 0.05 and p_values[i] < 0.05 and p_values[i + 1] < 0.05:
            significant_windows.append(i)
    significant_windows_list.append(significant_windows)
    print("Significant windows (0-indexed):", significant_windows)

target_df['p_values'] = pval_list
target_df['significant_windows'] = significant_windows_list

#make a new directory called "held_unit_disrim_plots"
save_dir = PA.save_dir + '/held_unit_discrim_plots'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#get indexes of scaled_time_array greater than -500 and less than 3000
tidxs = np.where((time_idx >= -500) & (time_idx < 2500))[0]
#determine how many indices were cut off the beginning
cut_off = np.where(time_idx >= -500)[0][0]

trim_time = time_idx[tidxs]
#need to fix minimum of sifnificant windows since it gets pushed forward
for name, group in target_df.groupby('held_unit_name'):
    #order group by session
    group = group.sort_values(by='session')
    group = group.reset_index(drop=True)
    fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharey=True, sharex=True)
    for i, row in group.iterrows():
        session = row['session']
        axs = ax[session-1]
        p_values = np.array(row['p_values'])
        #get p_values that are in tidxs
        p_values = p_values[tidxs]
        significant_windows = row['significant_windows']
        #get the significant windows that are in tidxs
        significant_windows = [i for i in significant_windows if i in tidxs]
        significant_windows = significant_windows - cut_off
        #shade the significant windows
        axs.fill_between(trim_time, 0, 1, where=[i in significant_windows for i in range(len(p_values))], color='lightgrey')
        axs.plot(trim_time, p_values)
        axs.set_title('Session ' + str(row['session']), fontsize=20)
        axs.set_xlabel('Time (s)', fontsize=20)
        if i == 0:
            axs.set_ylabel('p-value', fontsize=20)
        #set tick label size
        axs.tick_params(axis='both', which='major', labelsize=17)
    plt.suptitle('Taste Discrimination Index for ' + str(name))
    plt.tight_layout()
    plt.subplots_adjust(top=0.825)#, right=0.95)

    plt.savefig(save_dir + '/' + 'unit_' + str(name) + '.png')
    plt.savefig(save_dir + '/' + 'unit_' + str(name) + '.svg')
    plt.close()


#%% Spearman palatability analysis
palatatability_ranks = {'Suc': 1, 'NaCl': 2, 'CA': 3, 'QHCl': 4}

import scipy.stats as stats
#now, for every unit in max_response, calculate the taste discrimination index for each unit
# Parameters
window_length = 250  # ms
step_size = 25
num_time_bins = 7000
num_windows = (num_time_bins - window_length) // step_size + 1
tastes = ['Suc', 'NaCl', 'CA', 'QHCl']
time_idx = np.arange(-2000, 4775, 25)

# Function to calculate firing rates with rolling window
def calculate_firing_rates(spike_counts, window_length, step_size):
    num_trials = spike_counts.shape[0]
    num_windows = (num_time_bins - window_length) // step_size + 1
    firing_rates = np.zeros((num_trials, num_windows))
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_length
        firing_rates[:, i] = spike_counts[:, start_idx:end_idx].sum(axis=1) / (window_length / 1000.0)  # Convert to Hz
    return firing_rates

target_df = max_response
#isolate the columns  'rec_dir', 'exp_name', 'exp_group', 'session', 'unit_num', 'held_unit_name'
target_df = target_df[['rec_dir', 'exp_name', 'exp_group', 'session', 'unit_num', 'held_unit_name']]
#get unique rows
target_df = target_df.drop_duplicates().reset_index(drop=True)
dins = ["dig_in_0", "dig_in_1", "dig_in_2", "dig_in_3"]#, "dig_in_4"]
tastes = ['Suc', 'NaCl', 'CA', 'QHCl']#, 'Spont']
palatability_ranks = {'Suc': 1, 'NaCl': 2, 'CA': 3, 'QHCl': 4}
din_palatability_map = {'dig_in_0': 1, 'dig_in_1': 2, 'dig_in_2': 3, 'dig_in_3': 4}

def calc_pal_corr(row):
    rec_dir = row['rec_dir']
    unit_num = row['unit_num']
    #get the spike data
    time_array, responses = h5io.get_psths(rec_dir, units=[unit_num])
    rates_mats = []
    pal_mats = []
    for din in dins:
        rates = responses[din]
        # make an array the same shape as rates, but repeating the palatability rank for each bin
        pal_mat = np.zeros_like(rates)
        pal = din_palatability_map[din]
        pal_mat[:] = pal
        rates_mats.append(rates)
        pal_mats.append(pal_mat)

    # stack the rates and palatability matrices
    rates = np.vstack(rates_mats)
    pal = np.vstack(pal_mats)

    # Calculate Spearman's rank correlation for each window
    correlation_coefficients = []
    p_values = []
    for j in range(rates.shape[1]):
        rho, p_val = stats.spearmanr(rates[:, j], pal[:, j])
        correlation_coefficients.append(abs(rho))
        p_values.append(p_val)

    significant_windows = []
    significant_times = []
    for j in range(1, len(p_values) - 1):
        if p_values[j - 1] < 0.05 and p_values[j] < 0.05 and p_values[j + 1] < 0.05:
            significant_windows.append(j)
            significant_times.append(time_array[j])

    return p_values, significant_windows, significant_times, correlation_coefficients, time_array

def calc_pal_group(group):
    p_val_list = []
    significant_windows_list = []
    significant_times_list = []
    correlation_coefficients_list = []
    time_array_list = []
    for i, row in group.iterrows():
        p_values, significant_windows, significant_times, correlation_coefficients, time_array = calc_pal_corr(row)
        p_val_list.append(p_values)
        significant_windows_list.append(significant_windows)
        significant_times_list.append(significant_times)
        correlation_coefficients_list.append(correlation_coefficients)
        time_array_list.append(time_array)
    group['p_values'] = p_val_list
    group['significant_windows'] = significant_windows_list
    group['significant_times'] = significant_times_list
    group['correlation_coefficients'] = correlation_coefficients_list
    group['time_array'] = time_array_list
    return group

group_vars = ['rec_dir']
reslist = Parallel(n_jobs=-1)(delayed(calc_pal_group)(group) for name, group in target_df.groupby(group_vars))
target_df = pd.concat(reslist)
target_df = target_df.reset_index(drop=True)

#make a new directory called "held_unit_disrim_plots"
save_dir = PA.save_dir + '/held_unit_palatability_plots'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

sessions = [1, 2, 3]
#need to fix minimum of sifnificant windows since it gets pushed forward
for name, group in target_df.groupby('held_unit_name'):
    #order group by session
    group = group.sort_values(by='session')
    group = group.reset_index(drop=True)
    fig, ax = plt.subplots(1, 3, figsize=(10, 5), sharey=True, sharex=True)
    for session in sessions:
        row = group[group['session'] == session]
        axs = ax[session - 1]
        if len(row) == 0:
            continue
        else:
            time_array = np.array(row['time_array'].iloc[0])
            coeffs = np.array(row['correlation_coefficients'].iloc[0])
            pvals = np.array(row['p_values'].iloc[0])
            significant_windows = np.array(row['significant_windows'].iloc[0])
            sig_array = np.zeros_like(time_array)
            if len(significant_windows) > 0:
                sig_array[significant_windows] = 1

            trim_time = np.where((time_array >= -500) & (time_array < 2500))[0]
            time = time_array[trim_time]
            coeffs = coeffs[trim_time]
            pvals = pvals[trim_time]
            sig_array = sig_array[trim_time]

            axs.fill_between(time, sig_array, alpha=0.7, color='lightgrey')
            # #plot pvals on the right axis
            # axs2 = axs.twinx()
            # axs2.plot(time, pvals, color='black', line)
            axs.plot(time, coeffs)

        axs.set_title('Session ' + str(session), fontsize=20)
        axs.set_xlabel('Time (s)', fontsize=20)
        if session == 1:
            axs.set_ylabel('correlation coefficient', fontsize=20)
        #set tick label size
        axs.tick_params(axis='both', which='major', labelsize=17)
    plt.suptitle('Spearman correlation coefficient for ' + str(name))
    plt.tight_layout()
    plt.subplots_adjust(top=0.825)

    plt.savefig(save_dir + '/' + 'unit_' + str(name) + '.png')
    plt.savefig(save_dir + '/' + 'unit_' + str(name) + '.svg')
    plt.close()

