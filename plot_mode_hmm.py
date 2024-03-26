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
save_dir = HA.save_dir + os.sep + 'hmm_mode_demos'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print('save dir' + save_dir + ' created')


best_hmms = HA.get_best_hmms(sorting='best_AIC', overwrite=False) #get rows of hmm_overview where sorting column==sorting arugument
DS39_hmms = best_hmms.loc[best_hmms['exp_name'] == 'DS39']
DS39_hmms = DS39_hmms.loc[DS39_hmms['taste'] == 'Suc']
DS39_hmms['time_group'] = DS39_hmms['time_group'].astype(int)
DS39_rd = rec_info.loc[rec_info.exp_name =='DS39']

ticklab_size = 17
axlab_size = 20
subplots_margins = {'wspace': 0.05, 'hspace': 0.25, 'left': 0.1, 'right': 0.9}

#%% plot double mode
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
    pre_mode, pr_pre, pre_gamma, post_mode, pr_post, post_gamma = hmma.getSplitMode(hmm, split_trial=12, shuffle=False, output='classic')
    time_array = np.arange(len(pre_mode))
    mode_seqs = [pre_mode, post_mode]
    mode_gamma = [pre_gamma, post_gamma]
    for j, (mode, gamma) in enumerate(zip(mode_seqs, mode_gamma)):
        emission = gamma
        data_dict = {'time': time_array, 'state': mode, 'emission': emission}
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
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,3), sharex=True, sharey=True)
for nm, group in dfs.groupby(['rec_order', 'split_group', 'split_order']):
    splt = nm[-1]
    ax = axes[nm[-1], nm[0]]
    for i, grp in group.groupby('state'):
        emission = grp['emission'].values
        time = grp['time'].values
        ax.fill_between(time, 0, emission, alpha=0.5, color=get_hmm_plot_colors(6)[i])
        ax.plot(time, emission, color=get_hmm_plot_colors(6)[i], lw=3)
    if nm[0] == 0:
        ax.set_ylabel('avg\np(state)', fontsize=axlab_size)
        ax.set_yticks([0,1])
        ax.set_yticklabels([0,1])
    else:
        ax2 = ax.twinx()
        ax2.set_yticklabels([])
        ax2.set_ylabel(["trials\n1-11", "trials\n12-20"][splt], fontsize=axlab_size)
    #make the x axis tick labels size 20
    if nm[-1] == 1:
        ax.set_xlabel('time (ms)', fontsize=axlab_size)
    ax.tick_params(axis='both', labelsize=ticklab_size)
    #put a red vertical line at time = 0
    ax.axvline(0, color='red', lw=3)
#tight layout
plt.tight_layout()
#reduce padding between subplots
plt.subplots_adjust(top=0.95, bottom=0.25, **subplots_margins)
plt.show()
#save as svg in HA.save_dir
fig.savefig(save_dir + '/double mode hmm demo.svg')
fig.savefig(save_dir + '/double mode hmm demo.png')
#fig.subplots_adjust(wspace=0.05)

# %% plot single mode
dfs = []
counter = 0
for i in [1,3]:
    hmmid = DS39_hmms.loc[DS39_hmms['time_group'] == i]['hmm_id'].item()
    rd = DS39_rd.loc[DS39_rd['rec_num'] == i]['rec_dir'].item()
    hmm, time, params = phmm.load_hmm_from_hdf5(get_hmm_h5(rd), hmmid)
    mode_seqs, mode_gamma, best_seqs = hmma.getModeHmm(hmm)
    mode_gamma = mode_gamma.mean(axis=0)

    data_dict = {'time': time, 'state': mode_seqs, 'emission': mode_gamma}
    df = pd.DataFrame(data_dict)
    df['rec_num'] = i
    df['rec_order'] = counter
    dfs.append(df)
    counter += 1
dfs = pd.concat(dfs)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,1.75), sharex=True, sharey=True)
for nm, group in dfs.groupby(['rec_order']):
    splt = nm
    ax = axes[nm]
    for i, grp in group.groupby('state'):
        emission = grp['emission'].values
        time = grp['time'].values
        ax.plot(time, emission, color=get_hmm_plot_colors(6)[i], lw=3)
        ax.fill_between(time, 0, emission, alpha=0.5, color=get_hmm_plot_colors(6)[i])
    #remove y axis labels

    if nm == 0:
        ax.set_ylabel('avg\np(state)', fontsize=20)
        ax.set_yticks([0,1])
        ax.set_yticklabels([0,1])

    ax.set_xlabel('time (ms)', fontsize=20)
    #make the x axis tick labels size 20
    ax.tick_params(axis='both', labelsize=17)
    #put a red vertical line at time = 0
    ax.axvline(0, color='red')
    if nm == 1:
        ax2 = ax.twinx()
        ax2.set_yticklabels([])
        ax2.set_ylabel('trials\n1-30', fontsize=20)

#tight layout
plt.tight_layout()
#reduce padding between subplots
plt.subplots_adjust(wspace=0.05, hspace=0.1, top=0.9, bottom=0.4, left=0.1, right=0.9)
plt.show()
#save as svg in HA.save_dir
fig.savefig(HA.save_dir + '/mode hmm demo.svg')
fig.savefig(HA.save_dir + '/mode hmm demo.png')