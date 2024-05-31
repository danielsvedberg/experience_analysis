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
filtered = filtered.loc[filtered['taste'] != 'Spont']
#get unique rows
filtered = filtered.drop_duplicates(ignore_index=True)
held_units = filtered['held_unit_name'].to_numpy()

dfs = []
for name, group in filtered.groupby(['held_unit_name']):
    taste_resp = group['taste_responsive'].tolist()
    group['any_responsive'] = any(taste_resp)
    group['all_responsive'] = all(taste_resp)
    dfs.append(group)
filtered = pd.concat(dfs, ignore_index=True)

#remove all rows where any_responsive == False
filtered = filtered.loc[filtered['any_responsive'] == True]
filtered = filtered.reset_index(drop=True)

taste_map = {'Suc': 0, 'NaCl': 1, 'CA': 2, 'QHCl': 3, 0: 'Suc', 1: 'NaCl', 2: 'CA', 3: 'QHCl', 'Spont': 4, 4: 'Spont'}
#loop through each row in filtered, and calculate the magnitude of each response
prestim_start = -2000
prestim_end = -100
poststim_start = 100
poststim_end = 2000

dflist = []
for i, row in filtered.iterrows():
    rec_dir = row['rec_dir']
    unit_name = row['unit_name']
    taste = row['taste']
    din = taste_map[taste]
    dat = blechpy.load_dataset(rec_dir)
    din_trials = dat.dig_in_trials.query('channel == @din')
    session_trials = din_trials['trial_num']
    #make an array same size as session_trials, just ascending count
    taste_trials = np.arange(len(session_trials))
    trial_times = din_trials['on_time']
    time_array, psth_array = h5io.get_psths(rec_dir, din=din, units=[unit_name])
    prestim_idxs = np.where((time_array >= prestim_start) & (time_array < prestim_end))[0]
    poststim_idxs = np.where((time_array >= poststim_start) & (time_array < poststim_end))[0]
    #zscore the entire matrix
    z_array = (psth_array - psth_array.mean())/psth_array.std()
    prestim = z_array[:,prestim_idxs]
    poststim = z_array[:,poststim_idxs]
    prestim = prestim.mean(axis=1)
    poststim = poststim.mean(axis=1)
    response = poststim - prestim
    data_dict = {'taste_trial': taste_trials, 'session_trial': session_trials, 'trial_time': trial_times, 'prestim': prestim, 'poststim': poststim, 'response': response}
    resp_df = pd.DataFrame(data_dict)
    resp_df['unit_name'] = unit_name
    resp_df['taste'] = taste
    resp_df['rec_dir'] = rec_dir
    dflist.append(resp_df)
df = pd.concat(dflist, ignore_index=True)

#merge in rec_info
rec_info = rec_info[['rec_dir', 'exp_group', 'exp_name', 'rec_num']]
df = df.merge(rec_info, on='rec_dir')
#rename 'rec_num' to 'session'
df = df.rename(columns={'rec_num': 'session'})

#%%process through the trialwise analysis
import trialwise_analysis as ta
import hmm_analysis as hmma

group_cols = ['exp_group', 'session', 'taste']
trial_col = 'taste_trial'
nIter = 10

save_dir = PA.save_dir
#make a new folder called nonlinear_regression in save_dir
folder = 'held_unit_nonlinear_regression'
save_dir = save_dir + os.sep + folder
if not os.path.exists(save_dir):
    os.makedirs(save_dir + os.sep + folder)

value_col = 'prestim'
flag = 'response'
df3, shuff = ta.preprocess_nonlinear_regression(df, subject_col=['exp_name', 'unit_name'], group_cols=group_cols,
                                                trial_col=trial_col, value_col=value_col, overwrite=False,
                                                nIter=nIter, save_dir=save_dir, flag=flag)
ta.plotting_pipeline(df3, shuff, trial_col, value_col, nIter=nIter, save_dir=save_dir, flag=flag)
