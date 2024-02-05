import blechpy
from blechpy.analysis import poissonHMM as phmm
import analysis as ana
import hmm_analysis as hmma
import glob
import os

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts' # directory where the project is
proj = blechpy.load_project(proj_dir) #load the project
rec_info = proj.rec_info.copy() #get the rec_info table
#loop through each recording, get the hmm, run plot_saved models
rec_dirs = rec_info['rec_dir'].tolist()
def plot_hmm(rec_dir):
    handler = phmm.HmmHandler(rec_dir)
    handler.plot_saved_models(file_ext='png')
#run plot_hmm in parallel across every item in rec_dirs
from joblib import Parallel, delayed
Parallel(n_jobs=4)(delayed(plot_hmm)(rec_dir) for rec_dir in rec_dirs)


def get_hmm_h5(rec_dir):
    tmp = glob.glob(rec_dir + os.sep + '**' + os.sep + '*HMM_Analysis.hdf5', recursive=True)
    if len(tmp)>1:
        raise ValueError(str(tmp))

    if len(tmp) == 0:
        return None

    return tmp[0]

HA = ana.HmmAnalysis(proj) #create a hmm analysis object
best_hmms = HA.get_best_hmms(sorting='best_AIC', overwrite=False) #get rows of hmm_overview where sorting column==sorting arugument
DS39_hmms = best_hmms.loc[best_hmms['exp_name'] == 'DS39']
DS39_hmms = DS39_hmms.loc[DS39_hmms['taste'] == 'Suc']
DS39_hmms['time_group'] = DS39_hmms['time_group'].astype(int)
DS39_rd = rec_info.loc[rec_info.exp_name =='DS39']

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dfs = []
counter = 0
for i in [1,3]:
    hmmid = DS39_hmms.loc[DS39_hmms['time_group'] == i]['hmm_id'].item()
    rd = DS39_rd.loc[DS39_rd['rec_num'] == i]['rec_dir'].item()
    hmm, time, params = phmm.load_hmm_from_hdf5(get_hmm_h5(rd), hmmid)
    mode_seqs, mode_gamma, best_seqs = hmma.getModeHmm(hmm)
    time_array = np.arange(len(mode_seqs)) - 250
    ones = np.ones(len(mode_seqs))
    data_dict = {'time': time_array, 'state': mode_seqs, 'emission': ones}
    df = pd.DataFrame(data_dict)
    df['rec_num'] = i
    df['rec_order'] = counter
    dfs.append(df)
    counter += 1
dfs = pd.concat(dfs)
def get_hmm_plot_colors(n_states):
    colors = [plt.cm.tab10(x) for x in np.linspace(0, 1, n_states)]
    return colors

#generate 2 matplotlb subplots side by side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,1.5))
for nm, group in dfs.groupby(['state', 'rec_order']):
    emission = group['emission'].values
    time = group['time'].values
    ax = axes[nm[1]]
    ax.fill_between(time, 0, emission, alpha=0.5, color=get_hmm_plot_colors(5)[nm[0]])
    #remove y axis labels
    ax.set_yticklabels([])
    #make the x axis tick labels size 20
    ax.tick_params(axis='x', labelsize=20)
    #put a red vertical line at time = 0
    ax.axvline(0, color='red')
#tight layout
fig.tight_layout()
#save as svg in HA.save_dir
fig.savefig(HA.save_dir + '/mode hmm demo.svg')
#fig.subplots_adjust(wspace=0.05)