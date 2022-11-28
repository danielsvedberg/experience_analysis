#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:06:36 2022

@author: dsvedberg
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

#HA.plot_grouped_BIC()
#HA.plot_best_BIC()
#check df for nan in HMMID or early or late state

###############################################################################
###NAIVE BAYES CLASSIFICATION##################################################
[NB_res,NB_meta,NB_decode,NB_best,NB_timings] = HA.analyze_NB_ID(overwrite = True)
grcols = ['rec_dir','trial_num','taste','state_num']
NB_decode['state_num'] = NB_decode['hmm_state'].astype(int)
NB_decsub = NB_decode[grcols+['p_correct']]
NB_timings = NB_timings.merge(NB_decsub, on = grcols)

groupings = ['time_group','exp_group','state_group']
features = ['t_start','t_end','t_med','duration','p_correct']
df = NB_timings
df = df.loc[df.p_correct > 0.5]
NB_corrs = hmma.analyze_state_correlations(df, groupings,'trial_num', features)
NB_decode['taste'] = NB_decode.trial_ID
groupings = ['time_group','exp_group','state_group']
features = ['t_start','t_end','t_med','duration','p_correct']
df = NB_timings
df = df.loc[df.p_correct > 0.5]
NB_corrs = hmma.analyze_state_correlations(df, groupings,'trial_num', features)
grcols = ['rec_dir','trial_num','taste','state_num']
NB_decode['state_num'] = NB_decode['hmm_state'].astype(int)
NB_decsub = NB_decode[grcols+['p_correct']]
NB_timings = NB_timings.merge(NB_decsub, on = grcols)

groupings = ['time_group','exp_group','state_group']
features = ['t_start','t_end','t_med','duration','p_correct']
df = NB_timings
df = df.loc[df.p_correct > 0.5]
NB_corrs = hmma.analyze_state_correlations(df, groupings,'trial_num', features)

groupings = ['time_group','exp_group','state_group','taste']
features = ['t_start','t_end','t_med','duration','p_correct']
df = NB_timings
df = df.loc[df.p_correct > 0.5]
NB_corrs_tst = hmma.analyze_state_correlations(df, groupings,'trial_num', features)
for i in ['duration', 'SquaredError']:
    df = NB_corrs
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df = df.loc[df.Feature == i]    
    nplt.plotRsquared(df,i, save_dir = HA.save_dir, save_prefix = i+'tstRsquared')


for i in ['duration', 'SquaredError']:
    df = NB_corrs_tst
    df = df.loc[df.Feature == i] 
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df['session'] = df['session'].astype(int)
    df = df.sort_values(by = ['condition'], ascending = False)
    df = df.sort_values(by = ['palatability'], ascending = False)
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'tstRsquared')
plt.close('all')
for i in ['duration', 'SquaredError']:
    df = NB_corrs
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df = df.loc[df.Feature == i]    
    nplt.plotRsquared(df,i, save_dir = HA.save_dir, save_prefix = i+'tstRsquared')
plt.close('all')
for i in ['t_end']:
    df = NB_corrs
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df = df.loc[df.Feature == i]    
    nplt.plotRsquared(df,i, save_dir = HA.save_dir, save_prefix = i+'tstRsquared')


for i in ['t_end']:
    df = NB_corrs_tst
    df = df.loc[df.Feature == i] 
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df['session'] = df['session'].astype(int)
    df = df.sort_values(by = ['condition'], ascending = False)
    df = df.sort_values(by = ['palatability'], ascending = False)
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'tstRsquared')

for i in ['t_end']:
    df = NB_corrs_tst
    df = df.loc[df.Feature == i] 
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df['session'] = df['session'].astype(int)
    df = df.sort_values(by = ['condition'], ascending = False)
    #df = df.sort_values(by = ['palatability'], ascending = False)
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'tstRsquared')
plt.close('all')

for i in ['t_end']:
    df = NB_corrs_tst
    df = df.loc[df.Feature == i] 
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df['session'] = df['session'].astype(int)
    df = df.sort_values(by = ['condition'], ascending = False)
    #df = df.sort_values(by = ['palatability'], ascending = False)
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'tstRsquared')


for i in ['t_end']:
    df = NB_corrs_tst
    df = df.loc[df.Feature == i] 
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df['session'] = df['session'].astype(int)
    df = df.sort_values(by = ['condition'], ascending = False)
    #df = df.sort_values(by = ['palatability'], ascending = False)
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'tstRsquared')
plt.close('all')


for i in ['t_end']:
    df = NB_corrs_tst
    df = df.loc[df.Feature == i] 
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df['session'] = df['session'].astype(int)
    df = df.sort_values(by = ['condition'], ascending = False)
    #df = df.sort_values(by = ['palatability'], ascending = False)
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'tstRsquared')
for i in ['t_end']:
    df = NB_corrs
    df = df.loc[df.state_group  != 0]
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df = df.loc[df.Feature == i]    
    nplt.plotRsquared(df,i, save_dir = HA.save_dir, save_prefix = i+'tstRsquared')


for i in ['t_end']:
    df = NB_corrs_tst
    df = df.loc[df.state_group  != 0]
    df = df.loc[df.Feature == i] 
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df['session'] = df['session'].astype(int)
    df = df.sort_values(by = ['condition'], ascending = False)
    #df = df.sort_values(by = ['palatability'], ascending = False)
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'tstRsquared')
for i in ['t_end']:
    df = NB_corrs
    df = df.loc[df.state_group  != 'prestim']
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df = df.loc[df.Feature == i]    
    nplt.plotRsquared(df,i, save_dir = HA.save_dir, save_prefix = i+'tstRsquared')


for i in ['t_end']:
    df = NB_corrs_tst
    df = df.loc[df.state_group  != 'prestim']
    df = df.loc[df.Feature == i] 
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df['session'] = df['session'].astype(int)
    df = df.sort_values(by = ['condition'], ascending = False)
    #df = df.sort_values(by = ['palatability'], ascending = False)
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'tstRsquared')
plt.close('all')
for i in ['t_end']:
    df = NB_corrs
    df = df.loc[df.state_group  != 'prestim']
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df = df.loc[df.Feature == i]    
    nplt.plotRsquared(df,i, save_dir = HA.save_dir, save_prefix = i+'tstRsquared')

for i in ['t_end']:
    df = NB_corrs_tst
    df = df.loc[df.state_group  != 'prestim']
    df = df.loc[df.Feature == i] 
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df['session'] = df['session'].astype(int)
    df = df.sort_values(by = ['condition'], ascending = False)
    #df = df.sort_values(by = ['palatability'], ascending = False)
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'tstRsquared')
nplt.plot_NB_timing(NB_timings, plotdir = HA.save_dir, trial_group_size = 5)

for i in ['t_end']:
    df = NB_corrs
    df = df.loc[df.state_group  != 'prestim']
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df = df.loc[df.Feature == i]    
    df = df.sort_values(by = ['condition'], ascending = False)
    nplt.plotRsquared(df,i, save_dir = HA.save_dir, save_prefix = i+'tstRsquared')


for i in ['t_end']:
    df = NB_corrs_tst
    df = df.loc[df.state_group  != 'prestim']
    df = df.loc[df.Feature == i] 
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df['session'] = df['session'].astype(int)
    df = df.sort_values(by = ['condition'], ascending = False)
    #df = df.sort_values(by = ['palatability'], ascending = False)
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'tstRsquared')
plt.close('all')

for i in ['t_end']:
    df = NB_corrs
    df = df.loc[df.state_group  != 'prestim']
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df = df.loc[df.Feature == i]    
    df = df.sort_values(by = ['condition'], ascending = True)
    nplt.plotRsquared(df,i, save_dir = HA.save_dir, save_prefix = i+'tstRsquared')


for i in ['t_end']:
    df = NB_corrs_tst
    df = df.loc[df.state_group  != 'prestim']
    df = df.loc[df.Feature == i] 
    df = df.rename(columns = {'time_group':'session', 'exp_group':'condition'})
    df['session'] = df['session'].astype(int)
    df = df.sort_values(by = ['condition'], ascending = True)
    #df = df.sort_values(by = ['palatability'], ascending = False)
    nplt.plotRsquared(df,i,row = 'taste', save_dir = HA.save_dir, save_prefix = i+'tstRsquared')
plt.close('all')
import blechpy
dat = blechpy.load_dataset('/media/dsvedberg/Ubuntu Disk/taste_experience_resorts/DS39/DS39_spont_taste_201029_154308')
ut = dat.get_unit_table()
ut
din = dat.dig_in_mapping
din = dat.dig_in_mapping()
dat.make_psth_plots()
dat = blechpy.load_dataset('/media/dsvedberg/Ubuntu Disk/taste_experience_resorts/DS39/DS39_spont_taste_201029_154308')
import blechpy
dat = blechpy.load_dataset('/media/dsvedberg/Ubuntu Disk/taste_experience_resorts/DS39/DS39_spont_taste_201029_154308')
dat.make_psth_plots()
ut = dat.get_unit_table()
dat.make_psth_plots()
