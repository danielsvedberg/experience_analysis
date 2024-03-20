import analysis as ana
import blechpy
import new_plotting as nplt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import trialwise_analysis as ta
import scipy.stats as stats

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
HA = ana.HmmAnalysis(proj)  # create a hmm analysis object

NB_df = HA.analyze_NB_ID2(overwrite=False)
NB_df['duration'] = NB_df['t_end'] - NB_df['t_start']
NB_df['t_med'] = (NB_df['t_end'] + NB_df['t_start']) / 2

#for each rec_dir, subtract the min of off_time from all off_times
NB_df['off_time'] = NB_df.groupby('rec_dir')['off_time'].apply(lambda x: x - x.min())

#for each grouping of taste and rec_dir, make a new column called 'length_rank' ranking the states' length
NB_df['avg_t_start'] = NB_df.groupby(['taste', 'rec_dir','state'])['t_start'].transform('mean')
NB_df['avg_duration'] = NB_df.groupby(['taste', 'rec_dir','state'])['duration'].transform('mean')
NB_df = NB_df.loc[:, 't_start':]

NB_df['pr(correct state)'] = NB_df['p_taste_correct']
NB_df['session time'] = NB_df['off_time']
NB_df['t(median)'] = NB_df['t_med']
NB_df['t(start)'] = NB_df['t_start']
NB_df['t(end)'] = NB_df['t_end']
#get rid of all columns before t_start

NB_df_accuracy = NB_df.reset_index(drop=True)
#get rid of all rows where avg_t_start is 0
NB_df_accuracy = NB_df_accuracy.loc[NB_df_accuracy['avg_t_start'] != 0]
#get rid of all rows where avg_t_start is less than 1500 and greater than 100
NB_df_accuracy = NB_df_accuracy.loc[NB_df_accuracy['avg_t_start'] > 100]
NB_df_accuracy = NB_df_accuracy.loc[NB_df_accuracy['t_start'] > 0]
NB_df_accuracy = NB_df_accuracy.loc[NB_df_accuracy['duration'] > 50]

#rank states in trials by duration
NB_df_accuracy['trial_duration_rank'] = NB_df_accuracy.groupby(['taste', 'rec_dir','taste_trial'])['duration'].rank(ascending=False)
#rank states in trials by accuracy
NB_df_accuracy['trial_accuracy_rank'] = NB_df_accuracy.groupby(['taste', 'rec_dir','taste_trial'])['pr(correct state)'].rank(ascending=False)
NB_df_accuracy = NB_df_accuracy.loc[NB_df_accuracy['trial_accuracy_rank'] <= 2]

NB_df_accuracy['trial_order_rank'] = NB_df_accuracy.groupby(['taste', 'rec_dir','taste_trial'])['t_start'].rank(ascending=True, method='first')
NB_df_accuracy['avg_trial_order_rank'] = NB_df_accuracy.groupby(['taste', 'rec_dir','state'])['trial_order_rank'].transform('mean')
#round the average trial order rank to the nearest integer
NB_df_accuracy['avg_trial_order_rank'] = NB_df_accuracy['avg_trial_order_rank'].round()

#add a column called 'len_group' that is the number of rows in each grouping of rec_dir, taste, and taste_trial
NB_df_accuracy['len_group'] = NB_df_accuracy.groupby(['rec_dir', 'taste', 'taste_trial'])['t_start'].transform('count')
NB_df_accuracy['epoch'] = NB_df_accuracy['trial_order_rank']
#for each row where len_group == 1, replace the value of epoch with avg_trial_order_rank
#get indexes of rows where len_group == 1
idx = NB_df_accuracy.loc[NB_df_accuracy['len_group'] == 1].index
#replace the value of epoch with avg_trial_order_rank for each row in idx
NB_df_accuracy.loc[idx, 'epoch'] = NB_df_accuracy.loc[idx, 'avg_trial_order_rank']

order_map = {1:'early', 2:'late'}
NB_df_accuracy['epoch'] = NB_df_accuracy['epoch'].map(order_map)

NB_df = NB_df_accuracy[['exp_group','exp_name','session','taste','taste_trial','epoch','pr(correct state)', 't(start)',
                       't(median)', 't(end)', 'duration']]

trial_info = proj.get_dig_in_trial_df(reformat=True)
trial_info = trial_info.loc[trial_info['taste'] != 'Spont'].reset_index(drop=True)
#make a copy of trial_info with an epoch column filled with 'early'
trial_info['epoch'] = 'early'
#make a copy of trial_info with an epoch column filled with 'late'
trial_info_late = trial_info.copy()
trial_info_late['epoch'] = 'late'
#join the two
trial_info = pd.concat([trial_info, trial_info_late], ignore_index=True).reset_index(drop=True)
trial_info = trial_info[['exp_group','exp_name','session','taste','taste_trial','epoch']]


#merge to fill in the missing trials from taste_trial and session_trial using trial_info
merge = trial_info.merge(NB_df, on=['exp_group','exp_name','session','taste','taste_trial','epoch'], how='left')
#fill na in pr(correct state) with 0
merge['pr(correct state)'] = merge['pr(correct state)'].fillna(0)

