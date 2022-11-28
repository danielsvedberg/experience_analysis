#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:05:04 2022

@author: dsvedberg
"""
"""
Created on Sun Mar  7 12:32:58 2021

@author: dsvedberg
"""

import numpy as np
import matplotlib.pyplot as plt
import datashader
from blechpy.analysis import poissonHMM as phmm
import glob
import re
import os
import pandas as pd
#get into the directory
import analysis as ana
import blechpy
import new_plotting as nplt
import hmm_analysis as hmma 
import scipy
import seaborn as sns

#you need to make a project analysis using blechpy.project() first
#rec_dir =  '/media/dsvedberg/Ubuntu Disk/taste_experience'
rec_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts'
proj = blechpy.load_project(rec_dir)
proj.make_rec_info_table() #run this in case you changed around project stuff

PA = ana.ProjectAnalysis(proj)

PA.detect_held_units()#overwrite = True) #this part also gets the all units file
[all_units, held_df] = PA.get_unit_info()#overwrite = True) #run and check for correct area then run get best hmm

#PA.process_single_units(overwrite = True) #maybe run
#PA.run(overwrite = True)

HA = ana.HmmAnalysis(proj)
HA.get_hmm_overview()#overwrite = True) #use overwrrite = True to debug
#HA.sort_hmms_by_params()#overwrite = True)
HA.sort_hmms_by_BIC()#overwrite = True)
srt_df = HA.get_sorted_hmms()
HA.mark_early_and_late_states() #this is good, is writing in early and late states I think
best_hmms = HA.get_best_hmms(sorting = 'best_BIC')#, overwrite = True)# sorting = 'params #4') #HA has no attribute project analysis #this is not getting the early and late states 
#heads up, params #5 doesn't cover all animals, you want params #4
#play around with state timings in mark_early_and_late_states()

LR_timings,LR_trials,LR_res, LR_meta = HA.analyze_pal_linear_regression()
LR_timings['err'] = LR_timings.Y_pred - LR_timings.palatability
LR_timings.err= LR_timings.err.astype(float)
LR_timings['SquaredError'] = LR_timings.err**2
features = ['t_start', 't_end', 't_med', 'err', 'SquaredError','Y_pred']
groupings = ['time_group','exp_group']
LR_corrs = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
groupings = ['time_group', 'exp_group']
LR_corrs_alltst = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 5)
sns.set(font_scale = 1)
nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 5)
sns.set(font_scale = 1.25)
nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 5)
features = ['t_start', 't_end', 'duration','t_med', 'err', 'SquaredError','Y_pred']
groupings = ['time_group','exp_group']
LR_corrs = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
groupings = ['time_group', 'exp_group']
LR_corrs_alltst = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
groupings = ['time_group','exp_group']
LR_corrs2 = hmma.analyze_state_correlations(LR_timings, groupings, 'palatability', features)
LR_corrs2 = hmma.analyze_state_correlations(LR_timings, groupings, 'palatability', 'Y_pred')
nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 5)
groupings = ['time_group','exp_group','palatability']
LR_corrs = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
LR_corrs = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
groupings = ['time_group','exp_group','palatability']
LR_corrs = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
groupings = ['time_group', 'exp_group']
LR_corrs_alltst = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)

groupings = ['time_group','exp_group']
LR_corrs2 = hmma.analyze_state_correlations(LR_timings, groupings, 'palatability', 'Y_pred')
groupings = ['time_group','exp_group','palatability']
LR_corrs = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
groupings = ['time_group', 'exp_group']
LR_corrs_alltst = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)

groupings = ['time_group','exp_group']
LR_corrs2 = hmma.analyze_state_correlations(LR_timings, groupings, 'palatability', ['Y_pred'])
%debug
groupings = ['time_group','exp_group','palatability']
LR_corrs = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
groupings = ['time_group', 'exp_group']
LR_corrs_alltst = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)

groupings = ['time_group','exp_group']
LR_corrs2 = hmma.analyze_state_correlations(LR_timings, groupings, 'palatability', ['Y_pred'])
LR_corrs = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
groupings = ['time_group', 'exp_group']
LR_corrs_alltst = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)

groupings = ['time_group','exp_group']
LR_corrs2 = hmma.analyze_state_correlations(LR_timings, groupings, 'palatability', ['Y_pred'])
sns.set(font_scale = 1.25)
nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 5)
nplt.plot_pal_data(LR_timings,save_dir = HA.save_dir)
nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 5)
sns.set(font_scale = 1.2)
nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 5)
features = ['t_start', 't_end', 'duration','t_med', 'err', 'SquaredError','Y_pred']
groupings = ['time_group', 'exp_group', 'exp_name']
LR_corrs_plt = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
nplt.plot_daywise_data(LR_corrs_plt, ['duration', 'SquaredError'], save_dir = HA.save_dir, save_prefix = "Rsquared")

features = ['t_start', 't_end', 'duration','t_med', 'err', 'SquaredError','Y_pred']
groupings = ['time_group', 'exp_group', 'exp_name']
LR_corrs_plt = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
LR_corrs_plt = LR_corrs_plt.rename(columns = {'trial_num':'trial','time_group':'session', 'exp_group':'condition', 't_start': 't(start)', 't_end':'t(end)', 't_med':'t(median)','err':'error', 'Y_pred':'pred-pal'})
nplt.plot_daywise_data(LR_corrs_plt, ['duration', 'SquaredError'], save_dir = HA.save_dir, save_prefix = "Rsquared")
features = ['t_start', 't_end', 'duration','t_med', 'err', 'SquaredError','Y_pred']
groupings = ['time_group', 'exp_group', 'exp_name']
LR_corrs_plt = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
LR_corrs_plt = LR_corrs_plt.rename(columns = {'trial_num':'trial','time_group':'session', 'exp_group':'condition', 't_start': 't(start)', 't_end':'t(end)', 't_med':'t(median)','err':'error', 'Y_pred':'pred-pal'})
for i in features:
    LR_corrs_sub = LR_corrs_plt.loc[LR_corrs_plt.Feature == 'i']
    nplt.plot_daywise_data(LR_corrs_sub, ['Correlation'], save_dir = HA.save_dir, save_prefix = i+"Rsquared")
%debug
for i in ['duration', 'SquaredError']:
    LR_corrs_sub = LR_corrs_plt.loc[LR_corrs_plt.Feature == 'i']
    nplt.plot_daywise_data(LR_corrs_sub, ['Correlation'], save_dir = HA.save_dir, save_prefix = i+"Rsquared")
i
for i in ['duration', 'SquaredError']:
    LR_corrs_sub = LR_corrs_plt.loc[LR_corrs_plt.Feature == i]
    nplt.plot_daywise_data(LR_corrs_sub, ['Correlation'], save_dir = HA.save_dir, save_prefix = i+"Rsquared")
for i in ['duration', 'SquaredError']:
    df = LR_timings
    df = df.loc[df.Feature == i]    
    nplt.plotRsquared(df,i, save_dir = HA.save_dir, save_prefix = i+'Rsquared')
for i in ['duration', 'SquaredError']:
    df = LR_corrs
    df = df.loc[df.Feature == i]    
    nplt.plotRsquared(df,i, save_dir = HA.save_dir, save_prefix = i+'Rsquared')
for i in ['duration', 'SquaredError']:
    df = LR_corrs
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df = df.loc[df.Feature == i]    
    nplt.plotRsquared(df,i, save_dir = HA.save_dir, save_prefix = i+'Rsquared')
plt.close('all')
for i in ['duration', 'SquaredError']:
    df = LR_corrs
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df = df.loc[df.Feature == i]    
    nplt.plotRsquared(df,i, save_dir = HA.save_dir, save_prefix = i+'Rsquared')
plt.close('all')
for i in ['duration', 'SquaredError']:
    df = LR_corrs
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df = df.loc[df.Feature == i]    
    nplt.plotRsquared(df,i, save_dir = HA.save_dir, save_prefix = i+'Rsquared')
plt.close('all')
LR_timings['err'] = LR_timings.Y_pred - LR_timings.palatability
LR_timings.err= LR_timings.err.astype(float)
LR_timings['SquaredError'] = LR_timings.err**2
features = ['t_start', 't_end', 'duration','t_med', 'err', 'SquaredError','Y_pred']
groupings = ['time_group','exp_group','palatability','taste']
LR_corrs = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)

for i in ['duration', 'SquaredError']:
    df = LR_corrs_alltst
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df = df.loc[df.Feature == i]    
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'Rsquared')
for i in ['duration', 'SquaredError']:
    df = LR_corrs
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df = df.loc[df.Feature == i]    
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'Rsquared')
plt.close('all')
for i in ['duration', 'SquaredError']:
    df = LR_corrs
    df = df.sort_values(by = ['palatability'], ascending = False)
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df = df.loc[df.Feature == i]    
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'Rsquared')
for i in ['duration', 'SquaredError']:
    df = LR_corrs
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df = df.sort_values(by = ['palatability','session'], ascending = False)
    df = df.loc[df.Feature == i]    
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'Rsquared')
plt.close('all')
for i in ['duration', 'SquaredError']:
    df = LR_corrs
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df = df.sort_values(by = ['palatability','session'], ascending = False)
    df = df.sort_values(by = ['session'], ascending = True)
    df = df.loc[df.Feature == i]    
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'Rsquared')
plt.close('all')
for i in ['duration', 'SquaredError']:
    df = LR_corrs
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df = df.sort_values(by = ['palatability'], ascending = False)
    df = df.sort_values(by = ['session'], ascending = True)
    df = df.loc[df.Feature == i]    
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'Rsquared')
plt.close('all')
for i in ['duration', 'SquaredError']:
    df = LR_corrs
    df = df.loc[df.Feature == i] 
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df = df.sort_values(by = ['session'], ascending = True)
    df = df.sort_values(by = ['palatability'], ascending = False)
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'Rsquared')
plt.close('all')
for i in ['duration', 'SquaredError']:
    df = LR_corrs
    df = df.loc[df.Feature == i] 
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df['session'] = df['session'].astype(int)
    #df = df.sort_values(by = ['session'], ascending = True)
    df = df.sort_values(by = ['palatability'], ascending = False)
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'Rsquared')
plt.close('all')
LR_timings['err'] = LR_timings.Y_pred - LR_timings.palatability
LR_timings.err= LR_timings.err.astype(float)
LR_timings['SquaredError'] = LR_timings.err**2
features = ['t_start', 't_end', 'duration','t_med', 'err', 'SquaredError','Y_pred']
groupings = ['time_group','exp_group','palatability','taste']

LR_corrs = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
for i in ['duration', 'SquaredError']:
    df = LR_corrs
    df = df.loc[df.Feature == i] 
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df['session'] = df['session'].astype(int)
    df = df.sort_values(by = ['condition'], ascending = True)
    df = df.sort_values(by = ['palatability'], ascending = False)
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'Rsquared')
plt.close('all')

for i in ['duration', 'SquaredError']:
    df = LR_corrs
    df = df.loc[df.Feature == i] 
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df['session'] = df['session'].astype(int)
    df = df.sort_values(by = ['condition'], ascending = False)
    df = df.sort_values(by = ['palatability'], ascending = False)
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'Rsquared')


