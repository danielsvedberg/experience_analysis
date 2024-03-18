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
PA = ana.ProjectAnalysis(proj)
#get the all_unit_table
all_units, held_df = PA.get_unit_info(overwrite=False)
all_units = all_units[all_units['exp_group'] == 'naive']

HA = ana.HmmAnalysis(proj)  # create a hmm analysis object
seq_df = hmma.get_seq_df(HA)
seq_df = seq_df[seq_df['exp_group'] == 'naive'].reset_index(drop=True)
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
            else:
                t_start = int(time[on_edges])

        if t_start > 0:
            row['t_start'] = t_start
        else:
            row['t_start'] = np.nan
        row['late_state'] = mode
        rows.append(row)

seq_df = pd.concat(rows, axis=1).T
#eliminate all rows from seq_df where t_start is nan
seq_df = seq_df.loc[seq_df['exp_name'] == 'DS46']

dfs = []
for name, group in seq_df.groupby(['rec_dir', 'taste', 'hmm_id', 'late_state']):
    #order group by taste_trial
    group = group.sort_values(by='taste_trial').reset_index(drop=True)
    rec_dir = name[0]
    taste = name[1]
    hmm_id = name[2]
    late_state = name[3]
    h5_file = hmma.get_hmm_h5(rec_dir)
    hmm, time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
    gamma = hmm.stat_arrays['gamma_probabilities']
    pr_late_state = gamma[:, late_state, :]
    n_trials = pr_late_state.shape[0]
    time = np.tile(time, (n_trials,1))
    #turn axis 0 of pr_late_state into a list
    pr_late_state = pr_late_state.tolist()
    time = time.tolist()
    group['pr_late_state'] = pr_late_state
    group['time'] = time
    dfs.append(group)
seq_df = pd.concat(dfs, axis=0).reset_index(drop=True)

fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharey=True, sharex=True)
for i, session in enumerate([1,2,3]):
    for j, taste in enumerate(['Suc']):#, 'NaCl', 'CA', 'QHCl']):
        group = seq_df[seq_df['session'] == session]
        group = group[group['taste'] == taste]
        #group = group.explode(['pr_late_state', 'time'])
        ax = axs[i]
        for _, row in group.iterrows():
            pr_late_state = row['pr_late_state']
            time = row['time']
            trial_num = row['taste_trial']
            ax.plot(time, pr_late_state, color='tab:blue', alpha=0.2)

        #get the average of [pr_late_state] for group
        pr_late_state = np.array(group['pr_late_state'].tolist())
        pr_late_state = np.mean(pr_late_state, axis=0)
        ax.plot(time, pr_late_state, color='black', linewidth=2)
        ax.set_title(f'Session {session}')
        ax.set_xlabel('Time (ms)')
        if i == 0:
            ax.set_ylabel('P(late state)')

plt.tight_layout()
plt.show()
#save
save_dir = HA.save_dir
save_path = os.path.join(save_dir, 'demo_hmms')
if not os.path.exists(save_path):
    os.mkdir(save_path)
fig.savefig(os.path.join(save_path, 'late_state_demo.png'))
fig.savefig(os.path.join(save_path, 'late_state_demo.svg'))


