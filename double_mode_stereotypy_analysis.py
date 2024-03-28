import analysis as ana
import hmm_analysis as hmma
import blechpy
import new_plotting as nplt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import trialwise_analysis as ta
import scipy.stats as stats
import time
from joblib import Parallel, delayed
from blechpy.dio import h5io
#

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
rec_info = proj.rec_info.copy()  # get the rec_info table
rec_dirs = rec_info['rec_dir']

PA = ana.ProjectAnalysis(proj)
all_units, held_df = PA.get_unit_info(overwrite=False)

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

def euc_dist_trials(rates):
    avg_firing_rate = np.mean(rates, axis=1)  # Neurons x Bins
    euc_dist_mat = np.zeros((rates.shape[1], rates.shape[2]))  # Trials x Bins

    for i in range(rates.shape[1]):  # Loop over trials
        for j in range(rates.shape[2]):  # Loop over bins
            trial_rate_bin = rates[:, i, j]
            avg_firing_rate_bin = avg_firing_rate[:, j]

            # Euclidean distance
            euc_dist = np.linalg.norm(trial_rate_bin - avg_firing_rate_bin)
            euc_dist_mat[i, j] = euc_dist
    return euc_dist_mat

def process_split(rec_dir, split=10):
    df_list = []
    dat = blechpy.load_dataset(rec_dir)
    dintrials = get_trial_info(dat)
    time_array, rate_array = h5io.get_rate_data(rec_dir)
    for din, rate in rate_array.items():
        pre_rate = rate[:, :split, :]
        post_rate = rate[:, split:, :]

        pre_euc_dist_mat = euc_dist_trials(pre_rate)
        post_euc_dist_mat = euc_dist_trials(post_rate)

        #bind pre and post euc_dist_mat
        if pre_euc_dist_mat.shape[0] == 0:
            euc_dist_mat = post_euc_dist_mat
        elif post_euc_dist_mat.shape[0] == 0:
            euc_dist_mat = pre_euc_dist_mat
        else: #concantenate pre_euc_dist_mat and post_euc_dist_mat along axis 1
            euc_dist_mat = np.concatenate((pre_euc_dist_mat, post_euc_dist_mat), axis=0)
        # zscore every entry of euc_dist_mat
        euc_dist_mat = (euc_dist_mat - np.mean(euc_dist_mat)) / np.std(euc_dist_mat)
        avg_euc_dist = np.mean(euc_dist_mat[:, 2000:5000], axis=1)

        df = pd.DataFrame({
            'euclidean_distance': avg_euc_dist,
            'rec_dir': rec_dir,
            'channel': int(din[-1]),  # get the din number from string din
            'taste_trial': np.arange(rate.shape[1]),
            'split': split
        })
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    # add index info to df from dintrials using merge on taste_trial and channel
    df = pd.merge(df, dintrials, on=['taste_trial', 'channel'])
    # remove all rows where taste == 'Spont'
    df = df.loc[df['taste'] != 'Spont']
    # subtract the min of 'session_trial' from 'session_trial' to get the session_trial relative to the start of the recording
    df['session_trial'] = df['session_trial'] - df['session_trial'].min()
    df = df.reset_index(drop=True)
    return df

def process_rec_dir(rec_dir):
    splits = np.arange(30)
    dfs = [process_split(rec_dir, split) for split in splits]
    dfs = pd.concat(dfs, ignore_index=True).reset_index(drop=True)
    return dfs

# Parallelize processing of each rec_dir
num_cores = -1  # Use all available cores
final_dfs = Parallel(n_jobs=num_cores)(delayed(process_rec_dir)(rec_dir) for rec_dir in rec_dirs)

# Concatenate all resulting data frames into one
final_df = pd.concat(final_dfs, ignore_index=True)

rec_dir = rec_dirs[0]