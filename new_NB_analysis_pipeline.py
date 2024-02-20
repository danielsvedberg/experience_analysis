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
NB_df['order_in_seq'] = NB_df.groupby(['taste', 'rec_dir','taste_trial'])['t_med'].rank(ascending=True, method='first')
NB_df['length_rank'] = NB_df.groupby(['taste', 'rec_dir','taste_trial'])['duration'].rank(ascending=False)
NB_df['accuracy_rank'] = NB_df.groupby(['taste', 'rec_dir','taste_trial'])['p_correct'].rank(ascending=False)

NB_df['p(correct state)'] = NB_df['p_correct']
NB_df['p(correct taste)'] = NB_df['p_taste_correct']
NB_df['session time'] = NB_df['off_time']
NB_df['t(median)'] = NB_df['t_med']
NB_df['t(start)'] = NB_df['t_start']

def plot_nonlinear_regression_comparison(df3, shuff, stat_col, subject_col, group_cols, trial_col, value_col, flag=None, nIter=100, ymin=None, ymax=None, textsize=20):
    groups = [subject_col] + group_cols
    avg_shuff = shuff.groupby(group_cols + ['iternum']).mean().reset_index()
    avg_df3 = df3.groupby(groups).mean().reset_index() #trial average df3
    # plot the r2 values for each session with the null distribution
    if flag is not None:
        save_flag = trial_col + '_' + value_col + '_' + flag
    else:
        save_flag = trial_col + '_' + value_col
    ta.plot_r2_pval_diffs_summary(avg_shuff, avg_df3, stat_col, save_flag=save_flag, save_dir=HA.save_dir, textsize=textsize, nIter=nIter, n_comp=3, ymin=ymin, ymax=ymax)#re-run and replot with nIter=nIter

#for each group, plot the bar graphs quantifying the r2 values
def plot_nonlinear_regression_stats(df3, shuff, subject_col, group_cols, trial_col, value_col, flag=None, nIter=100, textsize=20, ymin=None, ymax=None):
    groups = [subject_col] + group_cols
    avg_shuff = shuff.groupby(groups + ['iternum']).mean().reset_index() #average across exp_names
    avg_df3 = df3.groupby(groups).mean().reset_index() #trial average df3

    for exp_group, group in avg_df3.groupby(['exp_group']):
        group_shuff = shuff.groupby('exp_group').get_group(exp_group)
        avg_group_shuff = avg_shuff.groupby('exp_group').get_group(exp_group)

        if flag is not None:
            save_flag = trial_col + '_' + value_col + '_' + flag
        else:
            save_flag = trial_col + '_' + value_col + '_' + exp_group + '_only'
        ta.plot_r2_pval_summary(avg_group_shuff, group, save_flag=save_flag, save_dir=HA.save_dir, textsize=textsize, nIter=nIter, n_comp=3, ymin=ymin, ymax=ymax)
def plot_nonlinear_regression_comparison(df3, shuff, stat_col, subject_col, group_cols, trial_col, value_col, flag=None, nIter=100, ymin=None, ymax=None, textsize=20):
    groups = [subject_col] + group_cols
    avg_shuff = shuff.groupby(group_cols + ['iternum']).mean().reset_index()
    avg_df3 = df3.groupby(groups).mean().reset_index() #trial average df3
    # plot the r2 values for each session with the null distribution
    if flag is not None:
        save_flag = trial_col + '_' + value_col + '_' + flag
    else:
        save_flag = trial_col + '_' + value_col
    ta.plot_r2_pval_diffs_summary(avg_shuff, avg_df3, stat_col, save_flag=save_flag, save_dir=HA.save_dir, textsize=textsize, nIter=nIter, n_comp=3, ymin=ymin, ymax=ymax)#re-run and replot with nIter=nIter

#for each group, plot the bar graphs quantifying the r2 values
def plot_nonlinear_regression_stats(df3, shuff, subject_col, group_cols, trial_col, value_col, flag=None, nIter=100, textsize=20, ymin=None, ymax=None):
    groups = [subject_col] + group_cols
    avg_shuff = shuff.groupby(groups + ['iternum']).mean().reset_index() #average across exp_names
    avg_df3 = df3.groupby(groups).mean().reset_index() #trial average df3

    for exp_group, group in avg_df3.groupby(['exp_group']):
        group_shuff = shuff.groupby('exp_group').get_group(exp_group)
        avg_group_shuff = avg_shuff.groupby('exp_group').get_group(exp_group)

        if flag is not None:
            save_flag = trial_col + '_' + value_col + '_' + flag
        else:
            save_flag = trial_col + '_' + value_col + '_' + exp_group + '_only'
        ta.plot_r2_pval_summary(avg_group_shuff, group, save_flag=save_flag, save_dir=HA.save_dir, textsize=textsize, nIter=nIter, n_comp=3, ymin=ymin, ymax=ymax)
def plot_nonlinear_line_graphs(df3, shuff, subject_col, group_cols, trial_col, value_col, flag=None, nIter=100, parallel=True, yMin=None, yMax=None, textsize=20):
    groups = [subject_col] + group_cols
    if yMin is None:
        yMin = min(df3[value_col])
    if yMax is None:
        yMax = max(df3[value_col])

    ta.plot_fits_summary_avg(df3, shuff_df=shuff, dat_col=value_col, trial_col=trial_col, save_dir=HA.save_dir,
                             use_alpha_pos=False, textsize=textsize, dotalpha=0.15, flag=flag, nIter=nIter,
                             parallel=parallel, yMin=yMin, yMax=yMax)
    for exp_group, group in df3.groupby(['exp_group']):
        group_shuff = shuff.groupby('exp_group').get_group(exp_group)
        if flag is not None:
            save_flag = exp_group + '_' + flag
        else:
            save_flag = exp_group
        ta.plot_fits_summary_avg(group, shuff_df=group_shuff, dat_col=value_col, trial_col=trial_col,
                                 save_dir=HA.save_dir, use_alpha_pos=False, textsize=textsize, dotalpha=0.15,
                                 flag=flag, nIter=nIter, parallel=parallel, yMin=yMin, yMax=yMax)


trial_col = ['session_trial', 'taste_trial', 'session time']
value_col = ['p(correct state)', 'p(correct taste)', 't(median)', 't(start)', 'duration']
state_determinant = ['duration', 'p_correct']
#get each unique combination of trial col, value col, and state determinant
trial_combos = [(t, v, s) for t in trial_col for v in value_col for s in state_determinant]

def plotting_pipeline(df3, shuff, trial_col, value_col, ymin=None, ymax=None, nIter=10000, flag=None):
    subject_col = 'exp_name'
    group_cols=['exp_group', 'session', 'taste']
    plot_nonlinear_line_graphs(df3, shuff, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col, value_col=value_col, nIter=nIter, textsize=20, yMin=ymin, yMax=ymax, flag=flag)
    #plot the stats quantificaiton of the r2 values
    plot_nonlinear_regression_stats(df3, shuff, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col, value_col=value_col, nIter=nIter, textsize=20, flag=flag, ymin=ymin, ymax=ymax)
    # #plot the stats quantificaation of the r2 values with head to head of naive vs sucrose preexposed
    plot_nonlinear_regression_comparison(df3, shuff, stat_col='r2', subject_col=subject_col, group_cols=group_cols, trial_col=trial_col, value_col=value_col, nIter=nIter, textsize=20, flag=flag, ymin=ymin, ymax=ymax)
    # #plot the sessionwise differences in the r2 values
    ta.plot_session_differences(df3, shuff, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col,

                             value_col=value_col, stat_col='r2', nIter=nIter, textsize=20, flag=flag, ymin=-ymax, ymax=ymax)
    r2_pred_change, shuff_r2_pred_change = ta.get_pred_change(df3, shuff, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col)
    # #plot the predicted change in value col over the course of the session, with stats
    ta.plot_predicted_change(r2_pred_change, shuff_r2_pred_change, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col,
                          value_col=value_col, nIter=nIter, textsize=20, flag=flag, ymin=-ymax, ymax=ymax)
    # # #plot the session differences in the predicted change of value col
    ta.plot_session_differences(r2_pred_change, shuff_r2_pred_change, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col,
                             value_col=value_col, stat_col='pred. change', nIter=nIter, textsize=20, flag=flag, ymin=-ymax, ymax=ymax)

    # plot the within session difference between naive and preexp for pred. change
    ymin = min(r2_pred_change['pred. change'].min(), shuff_r2_pred_change['pred. change'].min())
    ymax = max(r2_pred_change['pred. change'].max(), shuff_r2_pred_change['pred. change'].max())
    ta.plot_nonlinear_regression_comparison(r2_pred_change, shuff_r2_pred_change, stat_col='pred. change', subject_col=subject_col, group_cols=group_cols, trial_col=trial_col, value_col='pred. change', nIter=nIter, textsize=20, flag=flag, ymin=ymin, ymax=ymax)

def pipeline(df, value_col, trial_col, state_determinant):
    print('running pipeline for ' + value_col + ' and ' + trial_col + ' and ' + state_determinant)
    nIter = 10000
    epoch_idx = {1: 'early', 2: 'late'}

    subject_col = 'exp_name'
    group_cols=['exp_group', 'session', 'taste']

    df = df[df[state_determinant] < 2]
    df['order_in_seq'] = df.groupby(['taste', 'rec_dir','taste_trial'])['t_med'].rank(ascending=True, method='first')
    for nm, group in df.groupby(['order_in_seq']):
        #get rid of any rows with nans value col of group
        group = group.dropna(subset=[value_col])
        epoch = epoch_idx[nm]
        flag = state_determinant + '_determine_' + epoch

        df3, shuff = ta.preprocess_nonlinear_regression(group, subject_col=subject_col, group_cols=group_cols,
                                                                 trial_col=trial_col, value_col=value_col, overwrite=False,
                                                                 nIter=nIter, save_dir=HA.save_dir, flag=flag)

        plotting_pipeline(df3, shuff, trial_col, value_col, nIter=nIter, flag=flag)


#for each trial_combo, run the pipeline in parallel using joblib
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
#get rid of the first

Parallel(n_jobs=-1)(delayed(pipeline)(NB_df, value_col, trial_col, state_determinant) for trial_col, value_col, state_determinant in trial_combos)

# for i in trial_combos:
#     print('running pipeline for ' + i[0] + ' and ' + i[1] + ' and ' + i[2])
#     pipeline(NB_df, i[0], i[1], i[2])