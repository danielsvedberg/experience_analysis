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
    #NB_df['length_rank'] = NB_df.groupby(['taste', 'rec_dir','taste_trial'])['duration'].rank(ascending=False)
    #NB_df['state_accuracy_rank'] = NB_df.groupby(['taste', 'rec_dir','taste_trial'])['p_state_correct'].rank(ascending=False)
    #NB_df['taste_accuracy_rank'] = NB_df.groupby(['taste', 'rec_dir','taste_trial'])['p_taste_correct'].rank(ascending=False)
    #NB_df['valence_accuracy_rank'] = NB_df.groupby(['taste', 'rec_dir','taste_trial'])['p_valence_correct'].rank(ascending=False)

    NB_df['p(correct state)'] = NB_df['p_state_correct']
    NB_df['p(correct taste)'] = NB_df['p_taste_correct']
    NB_df['p(correct valence)'] = NB_df['p_valence_correct']

    NB_df['session time'] = NB_df['off_time']
    NB_df['t(median)'] = NB_df['t_med']
    NB_df['t(start)'] = NB_df['t_start']
    NB_df['t(end)'] = NB_df['t_end']
    return NB_df

def get_relevant_states(df, state_determinant, exclude_epoch=None):
    epoch_idx = {'early': 1, 'late': 2}
    #remove all rows where Y == 'prestim'
    df = df.loc[df['Y'] != 'prestim']
    #remove all rows where duration is less than 50
    df = df.loc[df['duration'] >= 100]
    #remove all rows where duration is greater than 3000
    df = df.loc[df['duration'] <= 3000]
    #remove all rows where t_start is greater than 2000
    #df = df.loc[df['t_start'] <= 2500]
    df = df.loc[df['t_end'] >= 200] #remove all rows where t_end is less than 200
    #round p(correct taste) and p(correct valence) 3 decimal places
    df['p(correct taste)'] = df['p(correct taste)'].round(2)
    df['p(correct valence)'] = df['p(correct valence)'].round(2)
    df = df.sort_values(by=['rec_dir', 'session_trial', 't_start'])
    early_df = df.loc[df['t_start'] <= 750]
    early_df = early_df.sort_values(by=['rec_dir', 'session_trial', 't_start'])
    #rerank early_df by state_determinant for each taste, rec_dir, and taste_trial
    #remove '_rank' from state_determinant
    #det = state_determinant.split('_rank')[0]
    sd_rank = state_determinant + '_rank'
    early_df[sd_rank] = early_df.groupby(['taste', 'rec_dir','taste_trial'])[state_determinant].rank(ascending=False)
    early_df['p(correct valence) rank'] = early_df.groupby(['taste', 'rec_dir','taste_trial'])['p(correct valence)'].rank(ascending=False)
    early_df['order_rank'] = early_df.groupby(['taste', 'rec_dir','taste_trial'])['t_start'].rank(ascending=True, method='first')
    early_df['order X correct rank'] = early_df['order_rank'] * (1-early_df['p(correct taste)'])
    early_df['order X correct rank'] = early_df.groupby(['taste', 'rec_dir','taste_trial'])['order X correct rank'].rank(ascending=True, method='first')

    early_df = early_df.loc[early_df['order X correct rank'] == 1]
    early_df['epoch'] = 'early'
    #now, get the rows of df that are not contained in early_df
    late_df = df.loc[~df.index.isin(early_df.index)]
    late_df = late_df.loc[late_df['t_start'] >= 100]
    #rerank late_df by state_determinant for each taste, rec_dir, and taste_trial
    late_df['p(correct valence) rank'] = late_df.groupby(['taste', 'rec_dir','taste_trial'])['p(correct valence)'].rank(ascending=False)
    late_df['order_rank'] = late_df.groupby(['taste', 'rec_dir','taste_trial'])['t_start'].rank(ascending=True, method='first')
    late_df['order X correct rank'] = late_df['order_rank'] * (1-late_df['p(correct valence)'])
    late_df['order X correct rank'] = late_df.groupby(['taste', 'rec_dir','taste_trial'])['order X correct rank'].rank(ascending=True, method='first')
    late_df = late_df.loc[late_df['order X correct rank'] == 1]
    late_df['epoch'] = 'late'
    df = pd.concat([early_df, late_df])
    df = df.sort_values(by=['rec_dir', 'session_trial', 't_start'])

    df['order_in_seq'] = df.groupby(['rec_dir','session_trial'])['t_med'].rank(ascending=True, method='first')

    #for each grouping of rec_dir, taste, and taste_trial, check if there is just one state
    #if the group just has one row, check if t_start is greater than 1000
    #if it is, reassign 'epoch' to 'late'
    for nm, group in df.groupby(['taste', 'rec_dir', 'taste_trial']):
        if len(group) == 1:
            if group['t(start)'].values[0] >= 400: #saddacca 2016 shows gaping can start as early as 400ms
                df.loc[(df['taste'] == nm[0]) & (df['rec_dir'] == nm[1]) & (df['taste_trial'] == nm[2]), 'epoch'] = 'late'

    df['p(correct taste)-avg'] = df['p(correct taste)'].sub(df.groupby(['taste', 'rec_dir','epoch'])['p(correct taste)'].transform('mean'))
    df['p(correct valence)-avg'] = df['p(correct valence)'].sub(df.groupby(['taste', 'rec_dir','epoch'])['p(correct valence)'].transform('mean'))
    df['t(end)-avg'] = df['t(end)'].sub(df.groupby(['taste', 'rec_dir','epoch'])['t(end)'].transform('mean'))
    df['t(start)-avg'] = df['t(start)'].sub(df.groupby(['taste', 'rec_dir','epoch'])['t(start)'].transform('mean'))

    if exclude_epoch is not None:
        if exclude_epoch == 'early':
            df = df.loc[df['epoch'] != 'early']
        elif exclude_epoch == 'late':
            df = df.loc[df['epoch'] != 'late']
        else:
            raise ValueError('exclude_epoch must be either "early" or "late"')

    return df

def prepipeline(df, value_col, trial_col, state_determinant, exclude_epoch=None, nIter=1000):
    print('running pipeline for ' + value_col + ' and ' + trial_col + ' and ' + state_determinant)
    analysis_folder = value_col + '_' + trial_col + '_' + state_determinant + '_nonlinear_regression'
    #check if dir exists, if not, make it
    save_dir = os.path.join(HA.save_dir, analysis_folder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df = get_relevant_states(df, state_determinant, exclude_epoch=exclude_epoch)

    subject_col = 'exp_name'
    group_cols=['exp_group', 'session', 'taste']

    for nm, group in df.groupby(['epoch']):
        #get rid of any rows with nans value col of group
        group = group.dropna(subset=[value_col])
        epoch = nm
        print(epoch)


        save_flag = state_determinant + '_determine_' + epoch

        ta.preprocess_nonlinear_regression(group, subject_cols=subject_col, group_cols=group_cols,
                                           trial_col=trial_col, value_col=value_col, overwrite=True,
                                           nIter=nIter, save_dir=save_dir, flag=save_flag)

def plottingpipe(df, value_col, trial_col, state_determinant, exclude_epoch=None, nIter=10000):
    print('plotting pipeline for ' + value_col + ' and ' + trial_col + ' and ' + state_determinant)
    analysis_folder = value_col + '_' + trial_col + '_' + state_determinant + '_nonlinear_regression'
    #check if dir exists, if not, make it
    save_dir = os.path.join(HA.save_dir, analysis_folder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
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
                                                                 trial_col=trial_col, value_col=value_col, overwrite=False,
                                                                 nIter=nIter, save_dir=save_dir, flag=save_flag)

        ta.plotting_pipeline(df3, shuff, trial_col, value_col, group_cols, [subject_col], nIter=nIter, save_dir=save_dir, flag=save_flag)
        plt.close('all')


proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
HA = ana.HmmAnalysis(proj)  # create a hmm analysis object

NB_df = HA.analyze_NB_ID2(overwrite=True)

NB_df = preprocess_NB(NB_df)

trial_col = 'taste_trial'
state_determinant = 'p(correct taste)'

value_cols = ['p(correct taste)-avg', 'p(correct valence)-avg', 't(end)-avg', 't(start)-avg']
exclude_epochs = [None, None, 'late', 'early']

for value_col, exclude_epoch in zip(value_cols, exclude_epochs):
    prepipeline(NB_df, value_col, trial_col, state_determinant, exclude_epoch=exclude_epoch)

#run plotting pipeline in parallel using joblib
Parallel(n_jobs=-1)(delayed(plottingpipe)(NB_df, value_col, trial_col, state_determinant, exclude_epoch=exclude_epoch) for value_col, exclude_epoch in zip(value_cols, exclude_epochs))





# value_col = 'p(correct taste)-avg'
# pipeline(NB_df, value_col, trial_col, state_determinant)
#
# value_col = 'p(correct valence)-avg'
# pipeline(NB_df, value_col, trial_col, state_determinant)

# value_col = 't(end)-avg'
# pipeline(NB_df, value_col, trial_col, state_determinant, exclude_epoch = 'late')
# plt.close('all')

#
# trial_col = 'taste_trial'
# state_determinant = 'taste_accuracy_rank'
# value_col = '|t(early:late)-x?(t(early:late)|'
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