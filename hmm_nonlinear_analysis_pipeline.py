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
import time
#
proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
HA = ana.HmmAnalysis(proj)  # create a hmm analysis object

def preprocess_nonlinear_regression(df, subject_col, group_cols, trial_col, value_col, yMin=None, yMax=None, parallel=True, overwrite=False, nIter=10000, flag=None):
    groupings = [subject_col] + group_cols
    params, r2, y_pred = ta.nonlinear_regression(df, subject_cols=groupings, trial_col=trial_col, value_col=value_col,
                                                 parallel=parallel, yMin=yMin, yMax=yMax)
    r2_df = pd.Series(r2).reset_index()  # turn dict into a series with multi-index
    r2_df = r2_df.reset_index(drop=True)  # reset the index so it becomes a dataframe
    colnames = groupings + ['r2']
    column_mapping = dict(zip(r2_df.columns, colnames))
    r2_df = r2_df.rename(columns=column_mapping)

    #r2_df_groupmean = r2_df.groupby(group_cols).mean().reset_index()
    # %% plot the nonlinear regression fits with the raw data
    identifiers = []
    alpha = []
    beta = []
    c = []
    for i, group in df.groupby(groupings):
        pr = params[i]
        identifiers.append(i)
        alpha.append(pr[0])
        beta.append(pr[1])
        c.append(pr[2])

    ids_df = pd.DataFrame(identifiers, columns=groupings)
    params_df = ids_df.copy()
    params_df['alpha'] = alpha
    params_df['beta'] = beta
    params_df['c'] = c

    # merge the params with df
    df2 = df.merge(params_df, on=groupings)
    df3 = df2.merge(r2_df, on=groupings)
    # for each row of df3, get the modeled value for that row, by passing the trial number and alpha, beta, c values to the model function
    modeled = []
    for i, row in df3.iterrows():
        pr = row[['alpha', 'beta', 'c']]
        mod = ta.model(row[trial_col], *pr)
        modeled.append(mod)
    df3['modeled'] = modeled
    # get the null distribution of r2 values
    shuffle = ta.iter_shuffle(df3, nIter=nIter, subject_cols=groupings, trial_col=trial_col, value_col=value_col,
                            yMin=yMin, yMax=yMax,
                            save_dir=HA.save_dir, overwrite=overwrite, parallel=parallel, flag=flag)

    if shuffle is [] or shuffle is None:
        raise ValueError('shuffle is None')
    return df3, shuffle

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
        flag = exp_group + '_' + flag
        ta.plot_fits_summary_avg(group, shuff_df=group_shuff, dat_col=value_col, trial_col=trial_col,
                                 save_dir=HA.save_dir, use_alpha_pos=False, textsize=textsize, dotalpha=0.15,
                                 flag=flag, nIter=nIter, parallel=parallel, yMin=yMin, yMax=yMax)

#this function plots the bar graphs of the differences in r2 between exp groups
def plot_nonlinear_regression_comparison(df3, shuff, subject_col, group_cols, trial_col, value_col, flag= None, nIter=100, ymin=None, ymax=None, textsize=20):
    groups = [subject_col] + group_cols
    avg_shuff = shuff.groupby(group_cols + ['iternum']).mean().reset_index()
    avg_df3 = df3.groupby(groups).mean().reset_index() #trial average df3
    # plot the r2 values for each session with the null distribution
    if flag is not None:
        save_flag = trial_col + '_' + value_col + '_' + flag
    else:
        save_flag = trial_col + '_' + value_col
    ta.plot_r2_pval_diffs_summary(avg_shuff, avg_df3, save_flag=save_flag, save_dir=HA.save_dir, textsize=textsize, nIter=nIter, n_comp=3, ymin=ymin, ymax=ymax)#re-run and replot with nIter=nIter

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

#import joblib parallel
from joblib import Parallel, delayed
def get_session_differences(df3, shuff, stat_col='r2'):
    diff_col = stat_col + ' difference'
    #make a new dataframe called day_diffs that contains the difference in r2 between day 1 and day 2, and day 1 and day 3, and day 2 and day 3
    def calculate_differences(group_name, group_df):
        results = []
        for i in range(len(group_df) - 1):
            for j in range(i + 1, len(group_df)):
                diff = group_df.iloc[i][stat_col] - group_df.iloc[j][stat_col]
                session_diff = f"{group_df.iloc[i]['session']}-{group_df.iloc[j]['session']}"
                results.append({'Group': group_name, 'Session Difference': session_diff, diff_col: diff})
        return results

    group_columns = ['exp_group', 'exp_name', 'taste']
    grouped = df3.groupby(group_columns)
    results = Parallel(n_jobs=-1)(delayed(calculate_differences)(group_name, group_df.sort_values('session')) for group_name, group_df in grouped)
    flat_results = [item for sublist in results for item in sublist]
    r2_diffs = pd.DataFrame(flat_results)
    expanded_groups = pd.DataFrame(r2_diffs['Group'].tolist(), columns=group_columns)
    # Concatenate the new columns with the original DataFrame
    r2_diffs = pd.concat([expanded_groups, r2_diffs.drop('Group', axis=1)], axis=1)

    # Apply function to shuff_r2_df
    group_columns = ['exp_group', 'exp_name', 'taste', 'iternum']
    grouped = shuff.groupby(group_columns)
    results = Parallel(n_jobs=-1)(delayed(calculate_differences)(group_name, group_df.sort_values('session')) for group_name, group_df in grouped)
    flat_results = [item for sublist in results for item in sublist]
    shuff_r2_diffs = pd.DataFrame(flat_results)
    expanded_groups = pd.DataFrame(shuff_r2_diffs['Group'].tolist(), columns=group_columns)
    # Concatenate the new columns with the original DataFrame
    shuff_r2_diffs = pd.concat([expanded_groups, shuff_r2_diffs.drop('Group', axis=1)], axis=1)
    shuff_r2_diffs = shuff_r2_diffs.groupby(['Session Difference', 'exp_group', 'taste', 'iternum']).mean().reset_index()
    return r2_diffs, shuff_r2_diffs

def plot_session_differences(df3, shuff, subject_col, group_cols, trial_col, value_col, stat_col=None, flag=None, nIter=100, textsize=20, ymin=None, ymax=None):
    if stat_col is None:
        stat_col = 'r2'

    r2_diffs, shuff_r2_diffs = get_session_differences(df3, shuff, stat_col=stat_col)
    groups = [subject_col] + group_cols
    #avg_df3 = df3.groupby(groups).mean().reset_index()

    if flag is not None:
        save_flag = trial_col + '_' + value_col + '_' + flag
    else:
        save_flag = trial_col + '_' + value_col
    ta.plot_daywise_r2_pval_diffs(shuff_r2_diffs, r2_diffs, stat_col=stat_col, save_flag=save_flag, save_dir=HA.save_dir, textsize=textsize, nIter=nIter, n_comp=3, ymin=ymin, ymax=ymax)

def get_pred_change(df3, shuff, subject_col, group_cols, trial_col):
    groups = [subject_col] + group_cols
    trials = df3[trial_col].unique()

    avg_df3 = df3.groupby(groups).mean().reset_index() #trial average df3
    def add_pred_change(df):
        pred_change = []
        for i, row in df.iterrows():
            params = row[['alpha', 'beta', 'c']]
            pred_change.append(ta.calc_pred_change(trials, params))
        df['pred. change'] = pred_change
        return df

    pred_change_df = add_pred_change(avg_df3)

    pred_change_shuff = []
    for i, row in shuff.iterrows():
        params = row['params']
        pred_change_shuff.append(ta.calc_pred_change(trials, params))
    shuff['pred. change'] = pred_change_shuff
    return pred_change_df, shuff
def plot_predicted_change(pred_change_df, pred_change_shuff, subject_col, group_cols, trial_col, value_col, flag=None, nIter=100, textsize=20, ymin=None, ymax=None):

    avg_shuff = pred_change_shuff.groupby(group_cols + ['iternum']).mean().reset_index()

    if flag is not None:
        save_flag = trial_col + '_' + value_col + '_' + flag
    else:
        save_flag = trial_col + '_' + value_col
    ta.plot_r2_pval_summary(avg_shuff, pred_change_df, stat_col='pred. change', save_flag=save_flag, save_dir=HA.save_dir, two_tailed=True, textsize=textsize, nIter=nIter, n_comp=3, ymin=ymin, ymax=ymax)

subject_col = 'exp_name'
group_cols = ['exp_group','session','taste']
def plotting_pipeline(df3, shuff, trial_col, value_col, ymin=None, ymax=None, nIter=10000, flag=None):
    subject_col = 'exp_name'
    plot_nonlinear_line_graphs(df3, shuff, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col, value_col=value_col, nIter=nIter, textsize=20, yMin=ymin, yMax=ymax, flag=flag)
    #plot the stats quantificaiton of the r2 values
    plot_nonlinear_regression_stats(df3, shuff, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col, value_col=value_col, nIter=nIter, textsize=20, flag=flag, ymin=ymin, ymax=ymax)
    # #plot the stats quantificaation of the r2 values with head to head of naive vs sucrose preexposed
    plot_nonlinear_regression_comparison(df3, shuff, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col, value_col=value_col, nIter=nIter, textsize=20, flag=flag, ymin=ymin, ymax=ymax)
    # #plot the sessionwise differences in the r2 values
    plot_session_differences(df3, shuff, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col,
                             value_col=value_col, stat_col='r2', nIter=nIter, textsize=20, flag=flag, ymin=-ymax, ymax=ymax)
    r2_pred_change, shuff_r2_pred_change = get_pred_change(df3, shuff, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col)
    # # #plot the predicted change in value col over the course of the session, with stats
    plot_predicted_change(r2_pred_change, shuff_r2_pred_change, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col,
                          value_col=value_col, nIter=nIter, textsize=20, flag=flag, ymin=-ymax, ymax=ymax)
    # # #plot the session differences in the predicted change of value col
    plot_session_differences(r2_pred_change, shuff_r2_pred_change, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col,
                             value_col=value_col, stat_col='pred. change', nIter=nIter, textsize=20, flag=flag, ymin=-ymax, ymax=ymax)


########################################################################################################################

#################### analysis of gamma mode ####################
avg_gamma_mode_df = HA.get_avg_gamma_mode(overwrite=False)
#refactor exp_group column in avg_gamma_mode_df from ['naive','suc_preexp'] to ['naive','sucrose preexposed']
avg_gamma_mode_df['exp_group'] = avg_gamma_mode_df['exp_group'].replace({'suc_preexp':'sucrose preexposed'})
avg_gamma_mode_df['session trial'] = avg_gamma_mode_df['session_trial']
avg_gamma_mode_df['session'] = avg_gamma_mode_df['time_group'].astype(int)
avg_gamma_mode_df['taste trial'] = avg_gamma_mode_df['taste_trial'].astype(int)
avg_gamma_mode_df['pr(mode trial)'] = avg_gamma_mode_df['pr(mode state)']

for trial_col in ['session trial']:#, 'taste trial']:
    df3, shuff = preprocess_nonlinear_regression(avg_gamma_mode_df, subject_col=subject_col, group_cols=group_cols,
                                                         trial_col=trial_col, value_col='pr(mode state)', overwrite=False,
                                                         nIter=10000, yMin=0, yMax=1)
    df3['pr(mode trial)'] = df3['pr(mode state)']
    #shuff['pr(mode trial)'] = shuff['pr(mode state)']
    plotting_pipeline(df3, shuff, trial_col=trial_col, value_col='pr(mode trial)', ymin=0, ymax=1, nIter=10000)

# ta.plot_fits(df3, dat_col='pr(mode trial)', trial_col='session trial', save_dir=HA.save_dir, use_alpha_pos=False)
# naive_only = df3[df3['exp_group']=='naive']
# shuff_naive = shuff[shuff['exp_group']=='naive']
# ta.plot_fits_summary(naive_only, dat_col='pr(mode trial)', trial_col='session trial', save_dir=HA.save_dir, use_alpha_pos=False, dotalpha=0.15, flag='gamma_mode', nIter=1000)

##########################################################################################33

#################### analysis of accuracy ###################
NB_decode = HA.get_NB_decode()  # get the decode dataframe with some post-processing
NB_decode['pr(correct)'] = NB_decode['p_correct'].astype(float)
NB_decode['session trial'] = NB_decode['session_trial'].astype(int)
NB_decode['taste trial'] = NB_decode['taste_trial'].astype(int)
NB_decode['session'] = NB_decode['time_group'].astype(int)
epochs = ['early']
def model_id(epoch): #zip in is a tuple of (groupby, trial_col) where groupby is a tuple of (epoch, group) and trial_col is a string of 'session trial' or 'taste trial'
    group = NB_decode[NB_decode['epoch']==epoch]
    trial_col = 'session trial'
    flag = str(epoch) + '_epoch_accuracy'
    for trial_col in ['session trial']:#, 'taste trial']:
        df3, shuff = preprocess_nonlinear_regression(group, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col, value_col='pr(correct)', overwrite=True, nIter=10000, yMin=0, yMax=1, flag=flag)
        plotting_pipeline(df3, shuff, trial_col=trial_col, value_col='pr(correct)', nIter=10000, ymin=0, ymax=1, flag=flag)
for epoch in epochs:
    model_id(epoch)

#################### analysis of timing ####################

NB_timings = HA.get_NB_timing()  # get the timings dataframe with some post-processing
NB_timings['session trial'] = NB_timings['session_trial'].astype(int)
NB_timings['session'] = NB_timings['time_group'].astype(int)
NB_timings['taste trial'] = NB_timings['taste_trial'].astype(int)
NB_timings['t(start)'] = NB_timings['t_start']
NB_timings['t(end)'] = NB_timings['t_end']
NB_timings['t(med)'] = NB_timings['t_med']
group_cols = ['exp_group','session','taste']
timing_cols = ['t(start)', 't(end)', 't(med)']#, 't_med', 'duration']
trial_cols = ['session trial']
epochs = ['early', 'late']
import itertools
iterlist = itertools.product(epochs, timing_cols, trial_cols)

for epoch, value_col, trial_col in iterlist:
    if value_col == 't(end)' and epoch == 'early':
        continue
    if value_col == 't(start)' and epoch == 'early':
        continue
    print(epoch, value_col, trial_col)
    group = NB_timings[NB_timings['epoch']==epoch]
    flag = str(epoch)
    yMin = 0
    yMax = max(group[value_col])
    df3, shuff = preprocess_nonlinear_regression(group, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col, value_col=value_col, overwrite=True, nIter=10000, yMin=yMin, yMax=yMax, flag=flag)
    plotting_pipeline(df3, shuff, trial_col=trial_col, value_col=value_col, ymin=yMin, ymax=yMax, nIter=10000, flag=flag)


subset = NB_timings.loc[NB_timings['epoch']=='late']
df3, shuff = preprocess_nonlinear_regression(subset, subject_col=subject_col, group_cols=group_cols, trial_col='session trial', value_col='t(start)', overwrite=False, nIter=10000, yMin=0, yMax=1, flag='early')
ta.plot_fits(df3, dat_col='t(start)', trial_col='session trial', save_dir=HA.save_dir, use_alpha_pos=False, flag='test')
naive_only = df3[df3['exp_group']=='naive']
shuff_naive = shuff[shuff['exp_group']=='naive']
ta.plot_fits_summary(naive_only, dat_col='pr(mode trial)', trial_col='session trial', save_dir=HA.save_dir, use_alpha_pos=False, dotalpha=0.15, flag='gamma_mode', nIter=1000)