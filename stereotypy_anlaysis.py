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

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy' # directory where the project is
proj = blechpy.load_project(proj_dir) #load the project
rec_info = proj.rec_info.copy() #get the rec_info table
rec_dirs = rec_info['rec_dir']

df_list = []
for rec_dir in rec_dirs:
    time_array, rate_array = h5io.get_rate_data(rec_dir)
    dins = rate_array.keys()
    for din in dins:
        rate = rate_array[din]
        avg_firing_rate = np.mean(rate, axis=1)
        n_trials = rate.shape[1]
        cos_dist_mat = np.zeros((rate.shape[1], rate.shape[2]))
        euc_dist_mat = np.zeros((rate.shape[1], rate.shape[2]))
        for i in range(n_trials):
            trial_rate = rate[:,i,:]
            #calculate cosine distance and euclidean distance for each bin in trial_rate, which is in dim 1
            n_bins = trial_rate.shape[1]
            for j in range(n_bins):
                trial_rate_bin = trial_rate[:,j]
                avg_firing_rate_bin = avg_firing_rate[:,j]
                cosine_distance = np.dot(avg_firing_rate_bin, trial_rate_bin) / (np.linalg.norm(avg_firing_rate_bin) * np.linalg.norm(trial_rate_bin))
                euclidean_distance = np.linalg.norm(avg_firing_rate_bin - trial_rate_bin)

                cos_dist_mat[i,j] = cosine_distance
                euc_dist_mat[i,j] = euclidean_distance
                #get the average of the cosine and euclidean distances for each trial from 2000 to 5000ms
        avg_cos_dist = np.mean(cos_dist_mat[:, 2000:5000], axis=1)
        avg_euc_dist = np.mean(euc_dist_mat[:, 2000:5000], axis=1)
        #make a dataframe
        data_dict = {'cosine_distance': avg_cos_dist, 'euclidean_distance': avg_euc_dist}
        df = pd.DataFrame(data_dict)
        df['rec_dir'] = rec_dir
        df['din'] = din
        df['trial'] = np.arange(n_trials)
        df_list.append(df)
df = pd.concat(df_list)

import numpy as np
import pandas as pd

# Assuming rec_dirs, h5io.get_rate_data are defined as in your context
df_list = []
for rec_dir in rec_dirs:
    time_array, rate_array = h5io.get_rate_data(rec_dir)
    for din, rate in rate_array.items():
        # Pre-compute average firing rates across trials for each bin
        avg_firing_rate = np.mean(rate, axis=1)  # Neurons x Bins

        # Initialize matrices for distances
        cos_dist_mat = np.zeros((rate.shape[1], rate.shape[2]))  # Trials x Bins
        euc_dist_mat = np.zeros((rate.shape[1], rate.shape[2]))  # Trials x Bins

        for i in range(rate.shape[1]):  # Loop over trials
            for j in range(rate.shape[2]):  # Loop over bins
                trial_rate_bin = rate[:, i, j]
                avg_firing_rate_bin = avg_firing_rate[:, j]

                # Compute cosine distance
                cos_sim = np.dot(trial_rate_bin, avg_firing_rate_bin) / (
                            np.linalg.norm(trial_rate_bin) * np.linalg.norm(avg_firing_rate_bin))
                cos_dist = 1 - cos_sim
                cos_dist_mat[i, j] = cos_dist

                # Compute Euclidean distance
                euc_dist = np.linalg.norm(trial_rate_bin - avg_firing_rate_bin)
                euc_dist_mat[i, j] = euc_dist

        # Calculate averages over specified bins
        avg_cos_dist = np.mean(cos_dist_mat[:, 2000:5000], axis=1)
        avg_euc_dist = np.mean(euc_dist_mat[:, 2000:5000], axis=1)

        # Create and append DataFrame
        df = pd.DataFrame({
            'cosine_distance': avg_cos_dist,
            'euclidean_distance': avg_euc_dist,
            'rec_dir': rec_dir,
            'din': din,
            'trial': np.arange(rate.shape[1])
        })
        df_list.append(df)

# Concatenate all data frames into one
final_df = pd.concat(df_list, ignore_index=True)


