import analysis as ana
import blechpy
import new_plotting as nplt
import matplotlib.pyplot as plt
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
    df = df.loc[df['duration'] >= 50]
    #remove all rows where duration is greater than 3000
    df = df.loc[df['duration'] <= 3000]
    #remove all rows where t_start is greater than 2000
    #df = df.loc[df['t_start'] <= 2500]
    df = df.loc[df['t_end'] >= 150] #remove all rows where t_end is less than 200
    #round p(correct taste) and p(correct valence) 3 decimal places
    df['p(correct taste)'] = df['p(correct taste)'].round(2)
    df['p(correct valence)'] = df['p(correct valence)'].round(2)
    df = df.sort_values(by=['rec_dir', 'session_trial', 't_start'])
    sd_rank = state_determinant + '_rank'
    sd_pctile = state_determinant + '_pctile'
    df[sd_rank] = df.groupby(['taste', 'rec_dir','taste_trial'])[state_determinant].rank(ascending=False, method='first') #higher accuracy = lower rank
    df[sd_pctile] = df.groupby(['taste', 'rec_dir','taste_trial'])[state_determinant].rank(ascending=False, pct=True, method='first') #higher accuracy = lower pctile
    df['p(correct valence) rank'] = df.groupby(['taste', 'rec_dir','taste_trial'])['p(correct valence)'].rank(ascending=False, method='first') 
    df['p(correct valence) pctile'] = df.groupby(['taste', 'rec_dir','taste_trial'])['p(correct valence)'].rank(ascending=False, pct=True, method='first') 
    df['order_rank'] = df.groupby(['taste', 'rec_dir','taste_trial'])['t_start'].rank(ascending=True) #sooner = lower rank
    df['order_pctile'] = df.groupby(['taste', 'rec_dir','taste_trial'])['t_start'].rank(ascending=True, pct=True) #sooner = lower pctile
    df['order X correct pctile'] = df['order_pctile'] * df[sd_pctile] #calculate composite score of timing and accuracy, sooner x more accurate = lower score
    df['avg_t_start'] = df.groupby(['taste', 'rec_dir','state'])['t_start'].transform('mean') #sooner = lower rank
    df['avg_t_start'] = df['avg_t_start'] > -249
    
    early_df = df.loc[df['avg_t_start'] <= 1000]
    #rank early_df by state_determinant for each taste, rec_dir, and taste_trial
    #early_df['order X correct rank'] = early_df.groupby(['taste', 'rec_dir','taste_trial'])['order X correct pctile'].rank(ascending=True, method='first') #lower score = better/smaller rank
    early_df[sd_rank] = early_df.groupby(['taste', 'rec_dir','taste_trial'])[state_determinant].rank(ascending=False, method='first') #higher accuracy = lower rank
    #early_df = early_df.loc[early_df['order X correct rank'] == 1] 
    early_df = early_df.loc[early_df[sd_rank] == 1]
    early_df['epoch'] = 'early'
    
    #now, get the rows of df that are not contained in early_df
    late_df = df.loc[~df.index.isin(early_df.index)]
    late_df = late_df.loc[late_df['t_start'] >= 100] 
    late_df = late_df.loc[late_df['avg_t_start'] <= 2500]
    #rank late_df by state_determinant for each taste, rec_dir, and taste_trial
    #late_df['order X correct rank'] = late_df.groupby(['taste', 'rec_dir','taste_trial'])['order X correct pctile'].rank(ascending=True, method='first')
    #late_df = late_df.loc[late_df['order X correct rank'] == 1]
    late_df[sd_rank] = late_df.groupby(['taste', 'rec_dir','taste_trial'])[state_determinant].rank(ascending=False, method='first') #higher accuracy = lower rank
    #early_df = early_df.loc[early_df['order X correct rank'] == 1] 
    late_df = late_df.loc[late_df[sd_rank] == 1]
    late_df['epoch'] = 'late'
    df = pd.concat([early_df, late_df])
    df = df.sort_values(by=['rec_dir', 'session_trial', 't_start'])

    df['order_in_seq'] = df.groupby(['rec_dir','session_trial'])['t_med'].rank(ascending=True, method='first')
    df['epoch'] = df['order_in_seq'].map({1:'early', 2:'late'})

    #for each grouping of rec_dir, taste, and taste_trial, check if there is just one state
    #if the group just has one row, check if t_start is greater than 1000
    #if it is, reassign 'epoch' to 'late'
    for nm, group in df.groupby(['taste', 'rec_dir', 'taste_trial']):
        if len(group) == 1:
            if group['avg_t_start'].values[0] >= 750: #saddacca 2016 shows gaping can start as early as 400ms
                df.loc[(df['taste'] == nm[0]) & (df['rec_dir'] == nm[1]) & (df['taste_trial'] == nm[2]), 'epoch'] = 'late'
    

    df['p(correct taste)-avg'] = df['p(correct taste)'].sub(df.groupby(['taste', 'rec_dir','epoch'])['p(correct taste)'].transform('mean'))
    df['p(correct valence)-avg'] = df['p(correct valence)'].sub(df.groupby(['taste', 'rec_dir','epoch'])['p(correct valence)'].transform('mean'))
    df['t(end)-avg'] = df['t(end)'].sub(df.groupby(['taste', 'rec_dir','epoch'])['t(end)'].transform('mean'))
    df['t(start)-avg'] = df['t(start)'].sub(df.groupby(['taste', 'rec_dir','epoch'])['t(start)'].transform('mean'))
    df['duration-avg'] = df['duration'].sub(df.groupby(['taste', 'rec_dir','epoch'])['duration'].transform('mean'))
    df['|t(start)-avg|'] = df['t(start)-avg'].abs()
    df['|t(end)-avg|'] = df['t(end)-avg'].abs()

    if exclude_epoch is not None:
        if exclude_epoch == 'early':
            df = df.loc[df['epoch'] != 'early']
        elif exclude_epoch == 'late':
            df = df.loc[df['epoch'] != 'late']
        else:
            raise ValueError('exclude_epoch must be either "early" or "late"')

    return df

def prepipeline(df, value_col, trial_col, state_determinant, exclude_epoch=None, nIter=10000):
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

def plottingpipe(df, value_col, trial_col, state_determinant, exclude_epoch=None, nIter=10000, exclude_d2=True):
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
        df3list = []
        for n, g in shuff.groupby(['exp_name', 'session','taste']):
            exp_name = n[0]
            session = n[1]
            taste = n[2]
            r2vals = g.r2.to_numpy()
            df3row = df3.query('exp_name==@exp_name and session==@session and taste==@taste')
            r2 = df3row.r2.to_numpy()[0]
            pval = np.nanmean(r2vals >= r2)
            df3row['p_val'] = pval
            df3list.append(df3row)

        df3 = pd.concat(df3list)
        if exclude_d2:
            df3 = df3.loc[df3['session'] != 2]
            shuff = shuff.loc[shuff['session'] != 2]
            save_flag = 'd1_d3' + save_flag


        ta.plotting_pipeline(df3, shuff, trial_col, value_col, group_cols, [subject_col], nIter=nIter,
                             save_dir=save_dir, flag=save_flag, xticklabs=False)
        plt.close('all')


proj_dir = '/media/volume/sdb/taste_experience_resorts_copy_forward_hmms'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
HA = ana.HmmAnalysis(proj)  # create a hmm analysis object
HA.sort_hmms_by_AIC(overwrite=False) #label hmms in hmm_overview with lowest AIC for each recording, save to sorted hmms
best_hmms = HA.get_best_hmms(sorting='best_AIC', overwrite=False) #get rows of hmm_overview where sorting column==sorting arugument

NB_df = HA.analyze_NB_ID2(overwrite=False)

NB_df = preprocess_NB(NB_df)

trial_col = 'taste_trial'
state_determinant = 'p(correct taste)'

#value_cols = ['t(start)', 't(start)-avg', 't(end)', 't(end)-avg', 'p(correct taste)', 'p(correct valence)', 'p(correct taste)-avg', 'p(correct valence)-avg']
#exclude_epochs = [None, None, 'late', 'late', None, None, None, None]

#value_cols = ['t(start)','t(start)-avg', 'p(correct taste)', 'p(correct valence)', 'p(correct taste)-avg', 'p(correct valence)-avg']
#exclude_epochs = ['early', 'early', 'early', 'early', 'early', 'early']
#value_cols = ['p(correct taste)', 'p(correct taste)']
#exclude_epochs = ['early', 'late']
#value_cols = ['|t(start)-avg|', '|t(end)-avg|']
#exclude_epochs= ['early', 'late']

value_cols = ['t(end)', '|t(end)-avg|','p(correct taste)', 'p(correct taste)', 'p(correct valence)', 'p(correct valence)']
exclude_epochs= ['late', 'late', 'early', 'late', 'early', 'late']
#value_cols = ['p(correct state)', 'p(correct state)']
#exclude_epochs=['late','early']

#for value_col, exclude_epoch in zip(value_cols, exclude_epochs):
#    prepipeline(NB_df, value_col, trial_col, state_determinant, exclude_epoch=exclude_epoch)

#run plotting pipeline in parallel using joblib
Parallel(n_jobs=-1)(delayed(plottingpipe)(NB_df, value_col, trial_col, state_determinant, exclude_epoch=exclude_epoch) for value_col, exclude_epoch in zip(value_cols, exclude_epochs))

