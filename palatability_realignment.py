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

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
PA = ana.ProjectAnalysis(proj)
#get the all_unit_table
all_units, held_df = PA.get_unit_info(overwrite=False)
all_units = all_units[all_units['exp_group'] == 'naive']

HA = ana.HmmAnalysis(proj)  # create a hmm analysis object
seq_df = hmma.get_seq_df(HA)
seq_df = seq_df[seq_df['exp_group'] == 'naive']

rows = []
#for i, row in seq_df.iterrows():
for name, group in seq_df.groupby(['rec_dir','taste']):
    group = group.reset_index(drop=True)
    seqs = np.vstack(group['sequences'])
    #get the portions of seq that line up with time = 1000:1500
    time = np.array(group['time'].iloc[0])
    tidxs = np.where((time >= 800))[0]

    stds = np.std(seqs, axis=1)
    multi_state_tr = np.where(stds > 0)[0]
    filt_seqs = seqs[multi_state_tr, :]
    lf_seqs = filt_seqs[:, tidxs]
    lf_seqs = lf_seqs.flatten()
    mode = int(stats.mode(lf_seqs)[0])

    for i, row in group.iterrows():
        seq = np.array(row['sequences'])
        #get indices of seq that are equal to mode
        mode_loc = (seq == mode)
        mode_loc = np.array(mode_loc, dtype=int)

        if sum(mode_loc) == 0 or sum(mode_loc) == 2000 or mode_loc[0] == 1:
            t_start = np.nan
        else:
            diff = np.diff(mode_loc)
            on_edges = np.where(diff == 1)[0] + 1
            off_edges = np.where(diff == -1)[0]
            if len(on_edges) > 1:
                t_start = time[on_edges[0]]
                # if len(off_edges) < len(on_edges):
                #     off_edges = np.append(off_edges, len(mode_loc) - 1)
                # valid_edges = (on_edges != 0)
                # on_edges = on_edges[valid_edges]
                # off_edges = off_edges[valid_edges]
                # epoch_lengths = off_edges - on_edges
                # longest_on_edge_idx = int(on_edges[np.argmax(epoch_lengths)])
                # t_start = time[longest_on_edge_idx]
            else:
                t_start = int(time[on_edges])

        if t_start > 0:
            row['t_start'] = t_start
        else:
            row['t_start'] = np.nan
        row['late_state'] = mode
        rows.append(row)

seq_df = pd.concat(rows, axis=1).T

#%% attempt 040224: trying something different to get late state
seq_df = hmma.get_NB_states_and_probs(HA)
seq_df = seq_df[seq_df['exp_group'] == 'naive']
seq_df = seq_df[seq_df['taste'] != 'Spont']
seq_df = seq_df[seq_df['epoch'] == 'late']
seq_df['t_start'] = seq_df['t(start)']

#eliminate all rows from seq_df where t_start is nan
#seq_df = seq_df.dropna(subset=['t_start'])

avg_t_start = seq_df['t_start'].mean()

palatatability_ranks = {'Suc': 1, 'NaCl': 2, 'CA': 3, 'QHCl': 4}
din_palatability_map = {'dig_in_0':1, 'dig_in_1':2, 'dig_in_2':3, 'dig_in_3':4}
dins = ['dig_in_0', 'dig_in_1', 'dig_in_2', 'dig_in_3']
din_taste_map = {'dig_in_0':'Suc', 'dig_in_1':'NaCl', 'dig_in_2':'CA', 'dig_in_3':'QHCl', 'Suc':'dig_in_0', 'NaCl':'dig_in_1', 'CA':'dig_in_2', 'QHCl':'dig_in_3'}

# first, let's calculate the average palatability for each unit
all_coeffs = []
peak_coeffs = []
all_pvals = []
all_significant_windows = []
sig_pals = []
time_arrays = []
all_coeffs_re = []
peak_coeffs_re = []
all_pvals_re = []
all_significant_windows_re = []
sig_pals_re = []

realign_window_start = -1850
realign_window_end = 2850
realign_window = np.arange(realign_window_start, realign_window_end, 25)
n_realign_bins = len(realign_window)

for i, row in all_units.iterrows():
    rec_dir = row['rec_dir']
    unit_name = row['unit_name']
    time_array, psth = h5io.get_psths(rec_dir, units=[unit_name])
    time_arrays.append(time_array)
    #get indices of time_array greater than 100 and less than 2500
    ROI = np.where((time_array > 100) & (time_array < 2500))[0]
    rates_mats = []
    realigned_rates_mats = []
    pal_mats = []
    pal_realigned_mats = []
    for din in dins:
        rates = psth[din]
        # make an array the same shape as rates, but repeating the palatability rank for each bin
        pal_mat = np.zeros_like(rates)
        pal = din_palatability_map[din]
        pal_mat[:] = pal
        rates_mats.append(rates)
        pal_mats.append(pal_mat)

        taste = din_taste_map[din]
        group_hmm = seq_df[seq_df['rec_dir'] == rec_dir]
        group_hmm = group_hmm[group_hmm['taste'] == taste]
        if group_hmm.empty:
            print(f'No HMM data for {unit_name} in {rec_dir} for {taste}')
        else:
            group_hmm = group_hmm.reset_index(drop=True)
            group_hmm = group_hmm.sort_values(by='taste_trial')
            pal_realigned_mat = np.zeros((len(group_hmm), n_realign_bins))
            pal_realigned_mat[:] = pal
            pal_realigned_mats.append(pal_realigned_mat)
            taste_trials = group_hmm['taste_trial'].to_numpy()
            t_starts = group_hmm['t_start'].to_numpy()

            realigned_responses = []
            for _, (taste_trial, t_start) in enumerate(zip(taste_trials, t_starts)):
                t0 = int(t_start - 1850)
                ti = int(t_start + 2850)
                # for idx0, get the closest value in time_array to t0
                diffstart = np.abs(time_array - t_start)
                idxstart = np.argmin(diffstart)
                idx0 = int(idxstart-(1850/25))
                idxi = int(idxstart+(2850/25))
                realigned = rates[taste_trial, idx0:idxi]
                realigned_responses.append(realigned)
                if len(realigned) != 188:
                    raise ValueError(f'Unit {unit_name} in {rec_dir} for {taste} has {len(realigned)} bins')
                else:
                    print('succ')
            realigned_responses = np.array(realigned_responses)
            realigned_rates_mats.append(realigned_responses)

    # stack the rates and palatability matrices
    rates = np.vstack(rates_mats)
    pal = np.vstack(pal_mats)
    realigned_rates = np.vstack(realigned_rates_mats)
    pal_realigned = np.vstack(pal_realigned_mats)

    # calculate the correlation between the rates and palatability
    coeffs = []
    pvals = []
    for j in range(rates.shape[1]):
        nbins = rates.shape[1]
        ntrials = int(rates.shape[0]/4)
        # meanrates = rates.reshape((len(dins), ntrials, nbins))
        # meanrates = np.nanmean(meanrates, axis=1)
        # meanpal = pal.reshape((len(dins), ntrials, nbins))
        # meanpal = np.nanmean(meanpal, axis=1)
        correlation, pvalue = stats.spearmanr(rates[:,j], pal[:,j])
        coeffs.append(abs(correlation))
        pvals.append(pvalue)
    all_pvals.append(pvals)
    all_coeffs.append(np.abs(coeffs))
    peak_coeff = np.nanmax(np.abs(coeffs))
    peak_coeffs.append(peak_coeff)

    re_coeffs = []
    re_pvals = []
    for j in range(realigned_rates.shape[1]):
        ntrials= int(realigned_rates.shape[0]/4)

        #calculate the average of realigned rates and palatability along axis 0 for every 30 indices of axis 0
        # meanrealigned_rates = realigned_rates.reshape((len(dins), ntrials, n_realign_bins))
        # meanrealigned_rates = np.nanmean(meanrealigned_rates, axis=0)
        # meanpal_realigned = pal_realigned.reshape((len(dins), ntrials, n_realign_bins))
        # meanpal_realigned = np.nanmean(meanpal_realigned, axis=0)
        correlation, pvalue = stats.spearmanr(realigned_rates[:,j], pal_realigned[:,j])
        if np.isnan(correlation):
            correlation = 0
            pvalue = 1
        re_coeffs.append(abs(correlation))
        re_pvals.append(pvalue)
    all_pvals_re.append(re_pvals)
    all_coeffs_re.append(np.abs(re_coeffs))
    #throw an exception if any of all_coeffs_re are empty or nan)

    peak_coeff_re = np.nanmax(np.abs(re_coeffs))
    peak_coeffs_re.append(peak_coeff_re)

    significant_windows = []
    for j in range(1, len(pvals) - 1):
        if pvals[j - 1] < 0.05 and pvals[j] < 0.05 and pvals[j + 1] < 0.05:
            significant_windows.append(j)
    all_significant_windows.append(significant_windows)
    #remove all windows significant windows that are not in ROI
    ROIwindows = [x for x in significant_windows if x in ROI]

    significant_windows_re = []
    for j in range(1, len(re_pvals) - 1):
        if re_pvals[j - 1] < 0.05 and re_pvals[j] < 0.05 and re_pvals[j + 1] < 0.05:
            significant_windows_re.append(j)
    all_significant_windows_re.append(significant_windows_re)

    pal_unit = len(ROIwindows) > 0
    sig_pals.append(pal_unit)

all_units['peak_coeff'] = peak_coeffs
all_units['sig_pal'] = sig_pals
all_units['pal_coeffs'] = all_coeffs
all_units['pvals'] = all_pvals
all_units['sig_windows'] = all_significant_windows
all_units['time_array'] = time_arrays
all_units['peak_coeff_realigned'] = peak_coeffs_re
all_units['pal_coeffs_realigned'] = all_coeffs_re
all_units['pvals_realigned'] = all_pvals_re


#next, filter only the rows with significant palatability
sig_units = all_units.copy()
sig_units = sig_units[sig_units['sig_pal']]
sig_units['session'] = sig_units['time_group']

naive_sig = sig_units[sig_units['exp_group'] == 'naive'].reset_index(drop=True)
naive_sig['session'] = naive_sig['time_group'].astype(int)

naive_sig_summary = []
for name, group in naive_sig.groupby('session'):
    #first, bootstrap the null distribution for the proportion of significant windows for the group
    group_pvals = group['pvals'].tolist()
    group_pvals = np.array(group_pvals)

    group_pvals_re = group['pvals_realigned'].tolist()
    group_pvals_re = np.array(group_pvals_re)

    group_sig = np.nanmean(group_pvals < 0.05, axis=0)
    group_sig_re = np.nanmean(group_pvals_re < 0.05, axis=0)
    n_rows = len(group)
    row_indices = np.arange(n_rows)
    boot = []
    boot_re = []
    for i in range(10000):
        #take a random sampling of coeffs of size group_slice along axis 0
        sample = np.random.choice(n_rows, n_rows, replace=True)
        sample_pvals = group_pvals[sample]
        sample_pvals_re = group_pvals_re[sample]
        #calculate the mean of the sample
        sample_sig = np.nanmean(sample_pvals < 0.05, axis=0)
        sample_sig_re = np.nanmean(sample_pvals_re < 0.05, axis=0)
        boot.append(sample_sig)
        boot_re.append(sample_sig_re)

    boot = np.array(boot).flatten()
    boot_re = np.array(boot_re).flatten()
    #
    time_array = group['time_array'].iloc[0]
    group_coeffs = group['pal_coeffs'].tolist()
    group_coeffs_re = group['pal_coeffs_realigned'].tolist()
    group_coeffs = np.array(group_coeffs)
    group_coeffs_re = np.array(group_coeffs_re)
    mean_coeffs = np.nanmean(group_coeffs, axis=0)
    mean_coeffs_re = np.nanmean(group_coeffs_re, axis=0)
    med_coeffs = np.nanmedian(group_coeffs, axis=0)
    med_coeffs_re = np.nanmedian(group_coeffs_re, axis=0)

    group_pvals = group['pvals'].tolist()
    group_pvals_re = group['pvals_realigned'].tolist()
    group_pvals = np.array(group_pvals)
    group_pvals_re = np.array(group_pvals_re)
    avg_pvals = np.nanmean(group_pvals, axis=0)
    avg_pvals_re = np.nanmean(group_pvals_re, axis=0)
    #for each index in axis 1, calculate the p value of the mean using boots

    prop_sig = np.nanmean(group_pvals < 0.05, axis=0)
    prop_sig_re = np.nanmean(group_pvals_re < 0.05, axis=0)

    agg_pvals = []
    for i in prop_sig:
        agg_pval = np.nanmean(boot > i)
        agg_pvals.append(agg_pval)

    agg_pvals_re = []
    for i in prop_sig_re:
        agg_pval = np.nanmean(boot_re > i)
        agg_pvals_re.append(agg_pval)

    significant_windows = []
    significant_times = []
    for j in range(1, len(agg_pvals) - 1):
        if agg_pvals[j - 1] < 0.05 and agg_pvals[j] < 0.05 and agg_pvals[j + 1] < 0.05:
            significant_windows.append(j)
            significant_times.append(time_array[j])
    sig_array = np.zeros_like(time_array)
    if len(significant_windows) > 0:
        sig_array[significant_windows] = 1

    significant_windows_re = []
    for j in range(1, len(agg_pvals_re) - 1):
        if agg_pvals_re[j - 1] < 0.05 and agg_pvals_re[j] < 0.05 and agg_pvals_re[j + 1] < 0.05:
            significant_windows_re.append(j)

    sig_array_re = np.zeros_like(agg_pvals_re)
    if len(significant_windows_re) > 0:
        sig_array_re[significant_windows_re] = 1

    #make a dataframe:
    df = pd.DataFrame({'exp_group':['naive'], 'session':[int(name)]})
    df['pal_coeffs'] = [mean_coeffs]
    df['med_coeffs'] = [med_coeffs]
    df['pvals'] = [agg_pvals]
    df['significant_windows'] = [significant_windows]
    df['significant_times'] = [significant_times]
    df['signfiicant_array'] = [sig_array]
    df['time'] = [time_array]
    df['avg_pvals'] = [avg_pvals]
    df['prop_sig'] = [prop_sig]

    df['pal_coeffs_realigned'] = [mean_coeffs_re]
    df['med_coeffs_realigned'] = [med_coeffs_re]
    df['pvals_realigned'] = [agg_pvals_re]
    df['significant_windows_realigned'] = [significant_windows_re]
    df['signfiicant_array_realigned'] = [sig_array_re]
    df['avg_pvals_realigned'] = [avg_pvals_re]
    df['prop_sig_realigned'] = [prop_sig_re]
    naive_sig_summary.append(df)
naive_sig_summary = pd.concat(naive_sig_summary)

time = time_arrays[0]
diff = np.abs(time - avg_t_start)
idx = np.argmin(diff)
midpoint = time[idx]
idx0 = np.where(time == 0)[0][0]
idx2000 = np.where(time == 2000)[0][0]
idxmidpoint = np.where(time == midpoint)[0][0]
prebins = idxmidpoint - idx0
postbins = idx2000 - idxmidpoint
idxmidpoint = 1850/25

sessions = [1,2,3]
fig, axs = plt.subplots(2, 3, figsize=(10,7), sharey=False, sharex=True)
for i in sessions:
    session_sum = naive_sig_summary[naive_sig_summary['session'] == int(i)]

    time = session_sum['time'][0]
    # get indices of time greater than -500 and less than 2500
    tidxs = np.where((time >= 0) & (time < 2000))[0]
    time = time[tidxs]

    corr = session_sum['med_coeffs'][0]
    corr = corr[tidxs]
    corr_re = session_sum['med_coeffs_realigned'][0]
    idx0 = int(idxmidpoint - prebins)
    idxi = int(idxmidpoint + postbins)
    corr_re = corr_re[idx0:idxi]
    t0 = 0
    re_time = session_sum['time'][0]
    tidx0 = np.where(re_time == t0)[0][0]
    tidxi = tidx0 + len(corr_re)
    re_time = re_time[tidx0:tidxi]

    # pvals = session_sum['pvals'][0]
    pvals = session_sum['prop_sig'][0]
    pvals = pvals[tidxs]
    sig_array = session_sum['signfiicant_array'][0]
    sig_array = sig_array[tidxs]

    pvals_re = session_sum['prop_sig_realigned'][0]
    pvals_re = pvals_re[idx0:idxi]
    sig_array_re = session_sum['signfiicant_array_realigned'][0]
    sig_array_re = sig_array_re[idx0:idxi]

    for j in range(2):
        ax = axs[j, i-1]
        ax2 = ax.twinx()
        ax.set_ylim(0.07, 0.17)
        ax2.set_ylim(0, 0.5)
        if j == 0:
            ax2.plot(time, pvals, color='black', linestyle='--')
            ax.fill_between(time, sig_array, color='lightgrey', alpha=0.9)
            ax.plot(time, corr, 'tab:blue')
            ax.set_title(f'Session {i}')
            yax_flag = 'stimulus-aligned'
        if j == 1:
            ax2.plot(re_time, pvals_re, color='black', linestyle='--')

            ax.fill_between(re_time, sig_array_re, color='lightgrey', alpha=0.9)
            ax.plot(re_time, corr_re, 'tab:blue')
            yax_flag = 'state-aligned'
        if i == 3:
            ax2.set_ylabel('% units\nsignificant', fontsize=18)
            #multiply the y axis tick labels by 100
            yticks = ax2.get_yticks()
            yticks = yticks*100
            #convert the y ticks to int
            yticks = yticks.astype(int)
            ax2.set_yticklabels(yticks, fontsize=14)
        else:
            ax2.set_yticks([])
        if i == 1:
            ax.set_ylabel('Mean Spearman\nCorrelation\n'+yax_flag, fontsize=19)
            # set the text color to the same blue as in the line plot
            ax.yaxis.label.set_color('tab:blue')
        else:
            ax.set_yticks([])

plt.tight_layout()
plt.show()
save_dir = PA.save_dir
pal_dir = os.path.join(save_dir, 'average_palatability')
if not os.path.exists(pal_dir):
    os.mkdir(pal_dir)
#save the figure
fig.savefig(os.path.join(pal_dir, 'naive_palatability.png'))
fig.savefig(os.path.join(pal_dir, 'naive_palatability.svg'))

plt.show()