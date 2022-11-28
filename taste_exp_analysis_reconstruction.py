#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:17:30 2022

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
import matplotlib
matplotlib.use('TkAgg')

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
#HA.mark_early_and_late_states() #this is good, is writing in early and late states I think
best_hmms = HA.get_best_hmms(sorting = 'best_BIC')#, overwrite = True)# sorting = 'params #4') #HA has no attribute project analysis #this is not getting the early and late states 
#heads up, params #5 doesn't cover all animals, you want params #4
#play around with state timings in mark_early_and_late_states()

#HA.plot_grouped_BIC()
#HA.plot_best_BIC()
#check df for nan in HMMID or early or late state
sns.set(style = 'white', font_scale = 1.5, rc={"figure.figsize":(10, 10)})
###############################################################################
###NAIVE BAYES CLASSIFICATION##################################################
#[NB_meta,NB_decode,NB_best,NB_timings] = HA.analyze_NB_ID(overwrite = True)
[NB_decode,NB_timings] = HA.analyze_NB_ID(overwrite = False)
NB_decode[['Y','epoch']] = NB_decode.Y.str.split('_',expand=True)
NB_decode['taste'] = NB_decode.trial_ID
NB_decode['state_num'] = NB_decode['hmm_state'].astype(int)

nplt.plot_NB_decoding(NB_decode,plotdir = HA.save_dir, trial_group_size = 10)

NB_summary = NB_decode.groupby(['exp_name','time_group','trial_ID','prestim_state','rec_dir','hmm_state']).agg('mean').reset_index()


# [_,NB_meta_late,NB_decode_late,NB_best_hmms_late,NB_timings_late] = HA.analyze_NB_ID(overwrite = True, epoch = 'late')
# NB_decode_late['taste'] = NB_decode.trial_ID
# NB_decode_late['state_num'] = NB_decode['hmm_state'].astype(int)
# nplt.plot_NB_decoding(NB_decode_late, plotdir = HA.save_dir, epoch = 'late', trial_group_size = 5)
# NB_late_summary = NB_decode_late.groupby(['exp_name','time_group','trial_ID','prestim_state','rec_dir','hmm_state']).agg('mean').reset_index()


###############################################################################
###Naive Bayes timing##########################################################

grcols = ['rec_dir','trial_num','taste','state_num']
NB_decsub = NB_decode[grcols+['p_correct']]
NB_timings = NB_timings.merge(NB_decsub, on = grcols)
NB_timings[['Y','epoch']] = NB_timings.state_group.str.split('_',expand=True)
nplt.plot_NB_timing(NB_timings,HA.save_dir,trial_group_size = 10)

###############################################################################
###Analysis of trial vs NB factors correlations################################
groupings = ['time_group','exp_group','state_group']
features = ['t_start','t_end','t_med','duration','p_correct','palatability']
yfacs = ['t_end', 't_med', 't_start']
df = NB_timings
df = df.loc[df.p_correct > 0.5]


#plot timing correlations grouped by session:
NB_corrs = hmma.analyze_state_correlations(df, groupings,'trial_num', features)
nplt.plotRsquared(NB_corrs,yfacs,save_dir = HA.save_dir, save_prefix = 'NB_timing')

#plot timing correlations grouped by session x taste:
groupings = ['time_group','exp_group','state_group','taste']
NB_corrs_tst = hmma.analyze_state_correlations(df, groupings,'trial_num', features)
nplt.plotRsquared(NB_corrs_tst,yfacs, row = 'taste', save_dir = HA.save_dir, save_prefix = 'NB_timing')


###############################################################################
###LR ANALYSIS
LR_timings,LR_trials,LR_res, LR_meta = HA.analyze_pal_linear_regression()
LR_timings['err'] = LR_timings.Y_pred - LR_timings.palatability
LR_timings.err= LR_timings.err.astype(float)
LR_timings['SquaredError'] = LR_timings.err**2
features = ['t_start', 't_end', 't_med', 'err', 'SquaredError','Y_pred']
groupings = ['time_group','exp_group']
LR_corrs = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
groupings = ['time_group', 'exp_group']
LR_corrs_alltst = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)

##Plot accuracy & timing of linear regression##################################

nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 5)
nplt.plot_pal_data(LR_timings,save_dir = HA.save_dir)
###############################################################################
HA.analyze_hmms(overwrite = True) #comment out/adapt some analyses
HA.plot_hmm_timing()








