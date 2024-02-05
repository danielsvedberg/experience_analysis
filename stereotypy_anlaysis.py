#first I need to loop through each recording
#then I need to load the h5 file of each recording
#then I need to loop through the unique dins for each recording
#then I need to load the rate arrays for each din
#then I need to get the average firing rate for each din across all the trials
#then I need to get the cosine distance between the average firing rate and the firing rate of each trial
#then I need to calculate the euclidean distance between the average firing rate and the firing rate of each trial

import blechpy
import numpy as np
import blechpy.dio.h5io as h5io
import pandas as pd
from joblib import Parallel, delayed

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy' # directory where the project is
proj = blechpy.load_project(proj_dir) #load the project
rec_info = proj.rec_info.copy() #get the rec_info table
rec_dirs = rec_info['rec_dir']

def get_trial_info(dat):
    dintrials = dat.dig_in_trials
    dintrials['taste_trial'] = 1
    #groupby name and cumsum taste trial
    dintrials['taste_trial'] = dintrials.groupby('name')['taste_trial'].cumsum()
    #rename column trial_num to 'session_trial'
    dintrials = dintrials.rename(columns={'trial_num':'session_trial','name':'taste'})
    #select just the columns 'taste_trial', 'taste', 'session_trial', 'channel', and 'on_time'
    dintrials = dintrials[['taste_trial', 'taste', 'session_trial', 'channel', 'on_time']]
    return dintrials

def process_rec_dir(rec_dir):
    df_list = []
    dat = blechpy.load_dataset(rec_dir)
    dintrials = get_trial_info(dat)
    time_array, rate_array = h5io.get_rate_data(rec_dir)
    for din, rate in rate_array.items():
        avg_firing_rate = np.mean(rate, axis=1)  # Neurons x Bins
        cos_dist_mat = np.zeros((rate.shape[1], rate.shape[2]))  # Trials x Bins
        euc_dist_mat = np.zeros((rate.shape[1], rate.shape[2]))  # Trials x Bins

        for i in range(rate.shape[1]):  # Loop over trials
            for j in range(rate.shape[2]):  # Loop over bins
                trial_rate_bin = rate[:, i, j]
                avg_firing_rate_bin = avg_firing_rate[:, j]

                # Cosine distance
                cos_sim = np.dot(trial_rate_bin, avg_firing_rate_bin) / (
                            np.linalg.norm(trial_rate_bin) * np.linalg.norm(avg_firing_rate_bin))
                cos_dist = 1 - cos_sim
                cos_dist_mat[i, j] = cos_dist

                # Euclidean distance
                euc_dist = np.linalg.norm(trial_rate_bin - avg_firing_rate_bin)
                euc_dist_mat[i, j] = euc_dist

        avg_cos_dist = np.mean(cos_dist_mat[:, 2000:5000], axis=1)
        avg_euc_dist = np.mean(euc_dist_mat[:, 2000:5000], axis=1)

        df = pd.DataFrame({
            'cosine_distance': avg_cos_dist,
            'euclidean_distance': avg_euc_dist,
            'rec_dir': rec_dir,
            'channel': int(din[-1]), #get the din number from string din
            'taste_trial': np.arange(rate.shape[1])
        })
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    #add index info to df from dintrials using merge
    df = pd.merge(df, dintrials, on='taste_trial')
    return df


# Parallelize processing of each rec_dir
num_cores = -1  # Use all available cores
final_dfs = Parallel(n_jobs=num_cores)(delayed(process_rec_dir)(rec_dir) for rec_dir in rec_dirs)

# Concatenate all resulting data frames into one
final_df = pd.concat(final_dfs, ignore_index=True)

#merge in rec_info into final_df
final_df = pd.merge(final_df, rec_info, on='rec_dir')
