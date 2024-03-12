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

NB_df['pr(correct state)'] = NB_df['p_correct']
NB_df['session time'] = NB_df['off_time']
NB_df['t(median)'] = NB_df['t_med']
NB_df['t(start)'] = NB_df['t_start']
#get rid of all columns before t_start

NB_df_accuracy = NB_df.reset_index(drop=True)
#get rid of all rows where avg_t_start is 0
#get rid of all rows where avg_t_start is less than 1500
NB_df_accuracy = NB_df_accuracy.loc[NB_df_accuracy['avg_t_start'] < 1500]
NB_df_accuracy = NB_df_accuracy.loc[NB_df_accuracy['avg_t_start'] > 0]
NB_df_accuracy['trial_duration_rank'] = NB_df_accuracy.groupby(['taste', 'rec_dir','taste_trial'])['duration'].rank(ascending=False)
NB_df_accuracy['trial_accuracy_rank'] = NB_df_accuracy.groupby(['taste', 'rec_dir','taste_trial'])['pr(correct state)'].rank(ascending=False)
NB_df_accuracy = NB_df_accuracy.loc[NB_df_accuracy['trial_accuracy_rank'] <= 2]
NB_df_accuracy['trial_order_rank'] = NB_df_accuracy.groupby(['taste', 'rec_dir','taste_trial'])['t_start'].rank(ascending=True, method='first')

order_map = {1:'early', 2:'late'}
NB_df_accuracy['epoch'] = NB_df_accuracy['trial_order_rank'].map(order_map)

NB_df_naive = NB_df_accuracy.loc[NB_df_accuracy['exp_group'] == 'naive'].reset_index(drop=True)
NB_df_naive = NB_df_naive[['rec_dir','exp_name','session','taste','taste_trial','epoch','pr(correct state)']]

#get rid of all rows with early epoch
NB_df_naive = NB_df_naive.loc[NB_df_naive['epoch'] == 'late']

for rec_dir, group in NB_df_naive.groupby('rec_dir'):
