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

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
PA = ana.ProjectAnalysis(proj)
#get the all_unit_table
all_units, held_df = PA.get_unit_info(overwrite=False)

palatatability_ranks = {'Suc': 1, 'NaCl': 2, 'CA': 3, 'QHCl': 4}
din_palatability_map = {'dig_in_0':1, 'dig_in_1':2, 'dig_in_2':3, 'dig_in_3':4}
dins = ['dig_in_0', 'dig_in_1', 'dig_in_2', 'dig_in_3']

# first, let's calculate the average palatability for each unit
all_coeffs = []
peak_coeffs = []
all_pvals = []
all_significant_windows = []
sig_pals = []
time_arrays = []
for i, row in all_units.iterrows():
    rec_dir = row['rec_dir']
    unit_name = row['unit_name']
    time_array, psth = h5io.get_psths(rec_dir, units=[unit_name])
    time_arrays.append(time_array)
    #get indices of time_array greater than 100 and less than 2500
    ROI = np.where((time_array > 100) & (time_array < 2500))[0]

    rates_mats = []
    pal_mats = []
    for din in dins:
        rates = psth[din]
        # make an array the same shape as rates, but repeating the palatability rank for each bin
        pal_mat = np.zeros_like(rates)
        pal = din_palatability_map[din]
        pal_mat[:] = pal
        rates_mats.append(rates)
        pal_mats.append(pal_mat)

    # stack the rates and palatability matrices
    rates = np.vstack(rates_mats)
    pal = np.vstack(pal_mats)

    # calculate the correlation between the rates and palatability
    coeffs = []
    pvals = []
    for j in range(rates.shape[1]):
        correlation, pvalue = stats.spearmanr(rates[:,j], pal[:,j])
        coeffs.append(abs(correlation))
        pvals.append(pvalue)
    all_pvals.append(pvals)
    all_coeffs.append(np.abs(coeffs))
    peak_coeff = np.max(np.abs(coeffs))
    peak_coeffs.append(peak_coeff)

    significant_windows = []
    for j in range(1, len(pvals) - 1):
        if pvals[j - 1] < 0.01 and pvals[j] < 0.01 and pvals[j + 1] < 0.01:
            significant_windows.append(j)
    all_significant_windows.append(significant_windows)
    #remove all windows significant windows that are not in ROI
    ROIwindows = [x for x in significant_windows if x in ROI]


    pal_unit = len(ROIwindows) > 0
    sig_pals.append(pal_unit)

all_units['peak_coeff'] = peak_coeffs
all_units['sig_pal'] = sig_pals
all_units['pal_coeffs'] = all_coeffs
all_units['pvals'] = all_pvals
all_units['sig_windows'] = all_significant_windows
all_units['time_array'] = time_arrays


#next, filter only the rows with significant palatability
sig_units = all_units.copy()
sig_units = sig_units[sig_units['sig_pal']]
sig_units['session'] = sig_units['time_group']

naive_sig = sig_units[sig_units['exp_group'] == 'naive']
naive_sig['session'] = naive_sig['time_group'].astype(int)

naive_sig_summary = []
for name, group in naive_sig.groupby('session'):
    #first, bootstrap the null distribution for the proportion of significant windows for the group
    group_pvals = group['pvals'].tolist()
    group_pvals = np.array(group_pvals)
    group_sig = np.mean(group_pvals < 0.05, axis=0)
    n_rows = len(group)
    row_indices = np.arange(n_rows)
    boot = []
    for i in range(10000):
        #take a random sampling of coeffs of size group_slice along axis 0
        sample = np.random.choice(n_rows, n_rows, replace=True)
        sample_pvals = group_pvals[sample]
        #calculate the mean of the sample
        sample_sig = np.mean(sample_pvals < 0.05, axis=0)
        boot.append(sample_sig)
    boot = np.array(boot).flatten()

    #
    time_array = group['time_array'].iloc[0]
    group_coeffs = group['pal_coeffs'].tolist()
    group_coeffs = np.array(group_coeffs)
    mean_coeffs = np.mean(group_coeffs, axis=0)
    med_coeffs = np.median(group_coeffs, axis=0)

    group_pvals = group['pvals'].tolist()
    group_pvals = np.array(group_pvals)
    avg_pvals = np.mean(group_pvals, axis=0)
    #for each index in axis 1, calculate the p value of the mean using boots

    prop_sig = np.mean(group_pvals < 0.05, axis=0)

    agg_pvals = []
    for i in prop_sig:
        agg_pval = np.mean(boot > i)
        agg_pvals.append(agg_pval)

    significant_windows = []
    significant_times = []
    for j in range(1, len(agg_pvals) - 1):
        if agg_pvals[j - 1] < 0.05 and agg_pvals[j] < 0.05 and agg_pvals[j + 1] < 0.05:
            significant_windows.append(j)
            significant_times.append(time_array[j])
    sig_array = np.zeros_like(time_array)
    if len(significant_windows) > 0:
        sig_array[significant_windows] = 1
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
    naive_sig_summary.append(df)
naive_sig_summary = pd.concat(naive_sig_summary)

sessions = [1,2,3]
fig, axs = plt.subplots(1, 3, figsize=(10,5), sharey=True, sharex=True)
for i in sessions:
    session_sum = naive_sig_summary[naive_sig_summary['session'] == int(i)]

    time = session_sum['time'][0]
    #get indices of time greater than -500 and less than 2500
    tidxs = np.where((time > -500) & (time < 2500))[0]
    time = time[tidxs]

    #corr = session_sum['pal_coeffs'][0]
    corr = session_sum['med_coeffs'][0]
    corr = corr[tidxs]
    #pvals = session_sum['pvals'][0]
    pvals = session_sum['prop_sig'][0]
    pvals = pvals[tidxs]
    sig_array = session_sum['signfiicant_array'][0]
    sig_array = sig_array[tidxs]

    ax = axs[i-1]
    ax.fill_between(time, sig_array, color='lightgrey', alpha=0.9)
    #make a second axis on the left
    ax2 = ax.twinx()
    ax2.plot(time, pvals, color='black', linestyle='--')
    if i == 3:
        ax2.set_ylabel('% units significant')
    else:
        ax2.set_yticks([])

    ax.plot(time, corr)


    #sns.lineplot(x='time', y='pal_coeffs', data=session_dat, ax=ax)
    ax.set_title(f'Session {i}')
    ax.set_xlabel('Time (ms)')
    #set the y axis limits to be from 0 to 0.5
    ax.set_ylim(0, 0.25)
    if i == 1:
        ax.set_ylabel('Mean\nSpearman Correlation')
        #set the text color to the same blue as in the line plot
        ax.yaxis.label.set_color('tab:blue')
    plt.tight_layout()

save_dir = PA.save_dir
pal_dir = os.path.join(save_dir, 'average_palatability')
if not os.path.exists(pal_dir):
    os.mkdir(pal_dir)
#save the figure
fig.savefig(os.path.join(pal_dir, 'naive_palatability.png'))
fig.savefig(os.path.join(pal_dir, 'naive_palatability.svg'))

#%% next we get the HMMs and get the transition times for each trial

HA = ana.HmmAnalysis(proj)  # create a hmm analysis object
