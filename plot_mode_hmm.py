import blechpy
from blechpy.analysis import poissonHMM as phmm
import analysis as ana
import hmm_analysis as hmma
import glob
import os

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy' # directory where the project is
proj = blechpy.load_project(proj_dir) #load the project
rec_info = proj.rec_info.copy() #get the rec_info table
#loop through each recording, get the hmm, run plot_saved models
rec_dirs = rec_info['rec_dir'].tolist()

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
    # mode_seqs, mode_gamma, best_seqs = hmma.getModeHmm(hmm)
    pre_mode, pr_pre, post_mode, pr_post, seq = hmma.getSplitMode(hmm, split_trial=12, shuffle=False, output='classic')
    time_array = np.arange(len(pre_mode))
    for j, mode in enumerate([pre_mode, post_mode]):
        ones = np.ones(len(mode))
        data_dict = {'time': time_array, 'state': mode, 'emission': ones}
        df = pd.DataFrame(data_dict)
        df['rec_num'] = i
        df['rec_order'] = counter
        df['split_group'] = ['pre', 'post'][j]
        df['split_order'] = j
        dfs.append(df)
    counter += 1
dfs = pd.concat(dfs)
def get_hmm_plot_colors(n_states):
    colors = [plt.cm.tab10(x) for x in np.linspace(0, 1, n_states)]
    return colors

#generate 2 matplotlb subplots side by side
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,2.25), sharex=True, sharey=True)
for nm, group in dfs.groupby(['rec_order', 'split_group', 'split_order']):
    splt = nm[-1]
    ax = axes[nm[-1], nm[0]]
    for i, grp in group.groupby('state'):
        emission = grp['emission'].values
        time = grp['time'].values
        ax.fill_between(time, 0, emission, alpha=0.5, color=get_hmm_plot_colors(6)[i])
    #remove y axis labels
    ax.set_yticklabels([])
    if nm[0] == 0:
        ax.set_ylabel(['trials\n1-11', 'trials\n12-20'][splt], fontsize=20)
    #make the x axis tick labels size 20
    if nm[-1] == 1:
        ax.tick_params(axis='x', labelsize=20)
    #put a red vertical line at time = 0
    ax.axvline(0, color='red')
#tight layout
plt.tight_layout()
#reduce padding between subplots
plt.subplots_adjust(wspace=0.05, hspace=0.1, top=0.95, bottom=0.2, left=0.1, right=0.95)
plt.show()
#save as svg in HA.save_dir
fig.savefig(HA.save_dir + '/double mode hmm demo.svg')
fig.savefig(HA.save_dir + '/double mode hmm demo.png')
#fig.subplots_adjust(wspace=0.05)