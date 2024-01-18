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

def model_nonlinear_regression(df, subject_col, group_cols, trial_col, value_col, flag= None, yMin=None, yMax=None, nIter=100, parallel=True, overwrite=False):
    groupings = [subject_col] + group_cols
    params, r2, y_pred = ta.nonlinear_regression(df, subject_cols=groupings, trial_col=trial_col, value_col=value_col, parallel=parallel, yMin=yMin, yMax=yMax)
    r2_df = pd.Series(r2).reset_index()  # turn dict into a series with multi-index
    r2_df = r2_df.reset_index(drop=True)  # reset the index so it becomes a dataframe
    colnames = groupings + ['r2']
    column_mapping = dict(zip(r2_df.columns, colnames))
    r2_df = r2_df.rename(columns=column_mapping)

    r2_df_groupmean = r2_df.groupby(group_cols).mean().reset_index()

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

    #for each row of df3, get the modeled value for that row, by passing the trial number and alpha, beta, c values to the model function
    modeled = []
    for i, row in df3.iterrows():
        pr = row[['alpha','beta','c']]
        mod = ta.model(row[trial_col], *pr)
        modeled.append(mod)
    df3['modeled'] = modeled
    # get the null distribution of r2 values
    shuff = ta.iter_shuffle(df3, nIter=nIter, subject_cols=groupings, trial_col=trial_col, value_col=value_col, yMin=yMin, yMax=yMax,
                            save_dir=HA.save_dir, overwrite=overwrite, parallel=parallel)

    shuff_pval_df = ta.get_shuff_pvals(shuff, r2_df) # TODO figure out why r2_df/ df2 does not have column r2

    avg_shuff = shuff.groupby(group_cols + ['iternum']).mean().reset_index()

    #ta.plot_fits(df3, dat_col=value_col, trial_col=trial_col, save_dir=HA.save_dir, use_alpha_pos=False)
    #for nm, group in df3.groupby(['exp_name','exp_group','session','taste']):
    #    ta.plot_fits(group, dat_col=value_col, trial_col=trial_col, save_dir=HA.save_dir, use_alpha_pos=False, textsize=20, dotalpha=0.15, flag=flag, nIter=nIter, parallel=parallel)

    ta.plot_fits_summary_avg(df3, shuff_df=shuff, dat_col=value_col, trial_col=trial_col, save_dir=HA.save_dir,
                             use_alpha_pos=False, textsize=20, dotalpha=0.15, flag=flag, nIter=nIter, parallel=False)

    # plot the r2 values for each session with the null distribution
    if flag is not None:
        save_flag = value_col + '_' + flag
    else:
        save_flag = value_col
    ta.plot_r2_pval_summary(avg_shuff, r2_df, save_flag=save_flag, save_dir=HA.save_dir, textsize=20, nIter=nIter)#re-run and replot with nIter=nIter
    #ta.plot_null_dist(avg_shuff, r2_df_groupmean, save_flag=save_flag, save_dir=HA.save_dir)
########################################################################################################################
subject_col = 'exp_name'
group_cols = ['exp_group','session','taste']

#################### analysis of gamma mode ####################
avg_gamma_mode_df = HA.get_avg_gamma_mode(overwrite=False)
#refactor exp_group column in avg_gamma_mode_df from ['naive','suc_preexp'] to ['naive','sucrose preexposed']
avg_gamma_mode_df['exp_group'] = avg_gamma_mode_df['exp_group'].replace({'suc_preexp':'sucrose preexposed'})
avg_gamma_mode_df['session trial'] = avg_gamma_mode_df['session_trial']
avg_gamma_mode_df['session'] = avg_gamma_mode_df['time_group'].astype(int)
avg_gamma_mode_df['taste trial'] = avg_gamma_mode_df['taste_trial'].astype(int)
#%% trialwise nonlinear regression
for trial_col in ['session trial', 'taste trial']:
    model_nonlinear_regression(avg_gamma_mode_df, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col, value_col='pr(mode state)', overwrite=False, nIter=10000, yMin=0, yMax=1)

#################### analysis of accuracy ####################
NB_decode = HA.get_NB_decode()  # get the decode dataframe with some post-processing
NB_decode['pr(correct)'] = NB_decode['p_correct'].astype(float)
NB_decode['session trial'] = NB_decode['session_trial'].astype(int)
NB_decode['session'] = NB_decode['time_group'].astype(int)
#
#
# for i, group in NB_decode.groupby('epoch'):
#     flag = str(i) + '_epoch' + '_accuracy'
#     model_nonlinear_regression(group, subject_col=subject_col, group_cols=group_cols, trial_col='session trial', value_col='pr(correct)', flag=flag, yMin=0, yMax=1, overwrite=False, nIter=100)
# #parallelize the above:
def model_id(zip_in): #zip in is a tuple of (groupby, trial_col) where groupby is a tuple of (epoch, group) and trial_col is a string of 'session trial' or 'taste trial'
    (epoch, group) = zip_in[0]
    trial_col = zip_in[1]
    flag = str(epoch) + '_epoch_accuracy'
    model_nonlinear_regression(group, subject_col=subject_col, group_cols=group_cols, trial_col='trial_col', value_col='pr(correct)', flag=flag, yMin=0, yMax=1, overwrite=False, nIter=10000)

from joblib import Parallel, delayed
Parallel(n_jobs=-1)(delayed(model_id)(zip_in) for zip_in in zip(NB_decode.groupby('epoch'), ['session trial', 'taste trial']))

#################### analysis of timing ####################

NB_timings = HA.get_NB_timing()  # get the timings dataframe with some post-processing
NB_timings['session trial'] = NB_timings['session_trial'].astype(int)
NB_timings['session'] = NB_timings['time_group'].astype(int)
NB_timings['taste trial'] = NB_timings['taste_trial'].astype(int)

timing_cols = ['t_start', 't_end', 't_med', 'duration']
# for i, group in NB_timings.groupby('epoch'):
#     for col in timing_cols:
#         for trial_col in ['session trial', 'taste trial']:
#             flag = str(i) + '_epoch_' + col + '_' + trial_col
#             model_nonlinear_regression(group, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col, value_col=col, flag=flag, yMin=None, yMax=None, nIter=10000, overwrite=False)

timing_zip = zip(NB_timings.groupby('epoch'), timing_cols, ['session trial', 'taste trial'])
def model_timing(zip_in):
    (epoch,group) = zip_in[0]
    value_col = zip_in[1]
    trial_col = zip_in[2]
    #flag = str(epoch) + '_epoch_' + value_col + '_' + trial_col
    flag = str(epoch)
    yMin = min(group[value_col])
    yMax = max(group[value_col])
    model_nonlinear_regression(group, subject_col=subject_col, group_cols=group_cols, trial_col=trial_col, value_col=value_col, flag=flag, yMin=yMin, yMax=yMax, nIter=10000, overwrite=False)

Parallel(n_jobs=-1)(delayed(model_timing)(zip_in) for zip_in in timing_zip)