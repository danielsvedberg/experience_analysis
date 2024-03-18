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
