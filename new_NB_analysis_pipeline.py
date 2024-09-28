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
from joblib import Parallel, delayed
import multiprocessing

def preprocess_NB(NB_df):
    valence_map = {'Suc':'pos', 'NaCl':'pos', 'CA': 'neg', 'QHCl':'neg'}
    NB_df['valence'] = NB_df['taste'].map(valence_map)
    NB_df['valence_pred'] = NB_df['taste_pred'].map(valence_map)

    NB_df['duration'] = NB_df['t_end'] - NB_df['t_start']
    NB_df['t_med'] = (NB_df['t_end'] + NB_df['t_start']) / 2

    #for each rec_dir, subtract the min of off_time from all off_times
    NB_df['off_time'] = NB_df.groupby('rec_dir')['off_time'].apply(lambda x: x - x.min())

    #for each grouping of taste and rec_dir, make a new column called 'length_rank' ranking the states' length
    NB_df['order_in_seq'] = NB_df.groupby(['taste', 'rec_dir','taste_trial'])['t_med'].rank(ascending=True, method='first')
    NB_df['length_rank'] = NB_df.groupby(['taste', 'rec_dir','taste_trial'])['duration'].rank(ascending=False)
    NB_df['state_accuracy_rank'] = NB_df.groupby(['taste', 'rec_dir','taste_trial'])['p_state_correct'].rank(ascending=False)
    NB_df['taste_accuracy_rank'] = NB_df.groupby(['taste', 'rec_dir','taste_trial'])['p_taste_correct'].rank(ascending=False)
    NB_df['valence_accuracy_rank'] = NB_df.groupby(['taste', 'rec_dir','taste_trial'])['p_valence_correct'].rank(ascending=False)

    NB_df['p(correct state)'] = NB_df['p_state_correct']
    NB_df['p(correct taste)'] = NB_df['p_taste_correct']
    NB_df['p(correct valence)'] = NB_df['p_valence_correct']

    NB_df['session time'] = NB_df['off_time']
    NB_df['t(median)'] = NB_df['t_med']
    NB_df['t(start)'] = NB_df['t_start']
    NB_df['t(end)'] = NB_df['t_end']
    NB_df['change in t(end)'] = NB_df['t(end)'].sub(NB_df.groupby(['taste', 'rec_dir'])['t(end)'].transform('mean'))
    NB_df['Δ t(early-late trans.)'] = NB_df['t(end)'].sub(NB_df.groupby(['taste', 'rec_dir'])['t(end)'].transform('mean'))
    NB_df['|Δ t(early-late trans.)|'] = NB_df['Δ t(early-late trans.)'].abs()
    return NB_df

def get_relevant_states(df, state_determinant, exclude_epoch=None):
    #remove all rows where Y == 'prestim'
    df = df.loc[df['Y'] != 'prestim']
    #remove all rows where duration is less than 100
    df = df.loc[df['duration'] >= 100]
    #remove all rows where t_start is greater than 2500
    df = df.loc[df['t_start'] <= 2000]
    df = df.loc[df['t_end'] >= 100] #remove all rows where t_end is less than 100
    epoch_idx = {1: 'early', 2: 'late'}
    #rerank early_df by state_determinant for each taste, rec_dir, and taste_trial
    #remove '_rank' from state_determinant
    #det = state_determinant.split('_rank')[0]
    sd_rank = state_determinant + '_rank'
    df[sd_rank] = df.groupby(['taste', 'rec_dir','taste_trial'])[state_determinant].rank(ascending=False)

    df = df.loc[df[sd_rank] <= 2]

    df['order_in_seq'] = df.groupby(['taste', 'rec_dir','taste_trial'])['t_med'].rank(ascending=True, method='first')
    df = df.loc[df['order_in_seq'] <= 2]
    # apply epoch index according to order_in_seq
    df['epoch'] = df['order_in_seq'].map(epoch_idx)
    #for each grouping of rec_dir, taste, and taste_trial, check if there is just one state
    #if the group just has one row, check if t_start is greater than 1000
    #if it is, reassign 'epoch' to 'late'
    for nm, group in df.groupby(['taste', 'rec_dir', 'taste_trial']):
        if len(group) == 1:
            if group['t(start)'].values[0] > 1000:
                df.loc[(df['taste'] == nm[0]) & (df['rec_dir'] == nm[1]) & (df['taste_trial'] == nm[2]), 'epoch'] = 'late'

    if exclude_epoch is not None:
        if exclude_epoch == 'early':
            df = df.loc[df['epoch'] != 'early']
        elif exclude_epoch == 'late':
            df = df.loc[df['epoch'] != 'late']
        else:
            raise ValueError('exclude_epoch must be either "early" or "late"')

    return df

def pipeline(df, value_col, trial_col, state_determinant, exclude_epoch=None):
    print('running pipeline for ' + value_col + ' and ' + trial_col + ' and ' + state_determinant)
    analysis_folder = value_col + '_' + trial_col + '_' + state_determinant + '_nonlinear_regression'
    #check if dir exists, if not, make it
    save_dir = os.path.join(HA.save_dir, analysis_folder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    nIter = 100
    df = get_relevant_states(df, state_determinant, exclude_epoch=exclude_epoch)

    subject_col = 'exp_name'
    group_cols=['exp_group', 'session', 'taste']

    for nm, group in df.groupby(['epoch']):
        #get rid of any rows with nans value col of group
        group = group.dropna(subset=[value_col])
        epoch = nm
        print(epoch)


        save_flag = state_determinant + '_determine_' + epoch

        df3, shuff = ta.preprocess_nonlinear_regression(group, subject_cols=subject_col, group_cols=group_cols,
                                                                 trial_col=trial_col, value_col=value_col, overwrite=True,
                                                                 nIter=nIter, save_dir=save_dir, flag=save_flag)

        ta.plotting_pipeline(df3, shuff, trial_col, value_col, group_cols, [subject_col], nIter=nIter, save_dir=save_dir, flag=save_flag)

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
HA = ana.HmmAnalysis(proj)  # create a hmm analysis object

###Process state coding
HA = ana.HmmAnalysis(proj) #create a hmm analysis object
ov = HA.get_hmm_overview(overwrite=True) #extract all the hmms into a dataframe and saves it
HA.sort_hmms_by_AIC(overwrite=True) #label hmms in hmm_overview with lowest AIC for each recording, save to sorted hmms
best_hmms = HA.get_best_hmms(sorting='best_AIC', overwrite=True) #get rows of hmm_overview where sorting column==sorting arugument

NB_df = HA.analyze_NB_ID2(overwrite=True)
NB_df = preprocess_NB(NB_df)

trial_col = 'taste_trial'
state_determinant = 'p(correct taste)'
value_col = 'p(correct taste)'
pipeline(NB_df, value_col, trial_col, state_determinant, exclude_epoch='early')

trial_col = 'taste_trial'
state_determinant = 'p(correct taste)'
value_col = 't(start)'
pipeline(NB_df, value_col, trial_col, state_determinant)

trial_col = 'taste_trial'
state_determinant = 'p(correct taste)'
value_col = 'p(correct valence)'
pipeline(NB_df, value_col, trial_col, state_determinant)
plt.close('all')

trial_col = 'taste_trial'
state_determinant = 'p(correct taste)'
value_col = 't(end)'
pipeline(NB_df, value_col, trial_col, state_determinant)

#
# trial_col = 'taste_trial'
# state_determinant = 'taste_accuracy_rank'
# value_col = '|t(early:late)-x̄(t(early:late)|'
# pipeline(NB_df, value_col, trial_col, state_determinant, exclude_epoch='late')

#
# trial_col = ['session_trial', 'taste_trial', 'session time']
# value_col = ['p(correct taste)', 't(median)', 't(start)', 'duration']
# state_determinant = ['p(correct taste)']
# #get each unique combination of trial col, value col, and state determinant
# trial_combos = [(t, v, s) for t in trial_col for v in value_col for s in state_determinant]
#
# #for each trial_combo, run the pipeline in parallel using joblib
#
# num_cores = multiprocessing.cpu_count()
# #get rid of the first
#
# Parallel(n_jobs=-1)(delayed(pipeline)(NB_df, value_col, trial_col, state_determinant) for trial_col, value_col, state_determinant in trial_combos)