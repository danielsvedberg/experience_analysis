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

#
proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
HA = ana.HmmAnalysis(proj)  # create a hmm analysis object

def model_nonlinear_regression(df, subject_col, group_cols, trial_col, value_col, flag= None):
    groupings = [subject_col] + group_cols
    params, r2, y_pred = ta.nonlinear_regression(df, subject_cols=groupings, trial_col=trial_col, value_col=value_col)
    r2_df = pd.Series(r2).reset_index()  # turn dict into a series with multi-index
    r2_df = r2_df.reset_index()  # reset the index so it becomes a dataframe
    r2_df = r2_df.rename(
        columns={'level_0': 'exp_name', 'level_1': 'exp_group', 'level_2': 'session', 'level_3': 'taste', 0: 'r2'})

    cols = ['exp_group', 'taste', 'session']
    r2_df_groupmean = r2_df.groupby(cols).mean().reset_index()

    # %% plot the nonlinear regression fits with the raw data
    modeled = []
    identifiers = []
    alpha = []
    beta = []
    c = []
    for i, group in df.groupby(groupings):
        pr = params[i]
        modeled.append(ta.model(group[trial_col], *pr))
        identifiers.append(i)
        alpha.append(pr[0])
        beta.append(pr[1])
        c.append(pr[2])

    # make a dataframe with the params
    params_df = pd.DataFrame({'exp_name': [i[0] for i in identifiers], 'exp_group': [i[1] for i in identifiers],
                              'time_group': [i[2] for i in identifiers], 'taste': [i[3] for i in identifiers],
                              'alpha': alpha, 'beta': beta, 'c': c})
    # merge the params with df
    df2 = df.merge(params_df, on=groupings)

    df2['modeled'] = np.concatenate(modeled)

    ta.plot_fits_summary(df2, dat_col=value_col, trial_col=trial_col, save_dir=HA.save_dir, use_alpha_pos=False, dotalpha=0.2, flag=flag)
    # %% get the null distribution of r2 values
    shuff = ta.iter_shuffle(df2, niter=100, subject_cols=groupings, trial_col=trial_col, value_col=value_col,
                            save_dir=HA.save_dir, overwrite=True)  # TODO break this down by group
    # save shuff as feather datafr
    avg_shuff = shuff.groupby(['exp_group', 'time_group', 'taste', 'iternum']).mean().reset_index()
    avg_shuff['session'] = avg_shuff['time_group']

    # %% plot the r2 values for each session with the null distribution
    if flag is not None:
        save_flag = trial_col + '_' + flag
    else:
        save_flag = trial_col
    ta.plot_null_dist(avg_shuff, r2_df_groupmean, save_flag=save_flag, save_dir=HA.save_dir)

subject_col = 'exp_name'
group_cols = ['exp_group','time_group','taste']

#################### analysis of gamma mode ####################
avg_gamma_mode_df = HA.get_avg_gamma_mode(overwrite=False)
#%% trialwise nonlinear regression
model_nonlinear_regression(avg_gamma_mode_df, subject_col=subject_col, group_cols=group_cols, trial_col='session_trial', value_col='pr(mode state)')

#################### analysis of accuracy ####################
NB_decode = HA.get_NB_decode()  # get the decode dataframe with some post-processing
for i, group in NB_decode.groupby('epoch'):
    flag = str(i) + '_epoch'
    model_nonlinear_regression(group, subject_col=subject_col, group_cols=groupings, trial_col='session_trial', value_col='p_correct', flag=flag)
#################### analysis of timing ####################