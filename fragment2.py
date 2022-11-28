#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:04:11 2022

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
nplt.plot_pal_data(LR_timings,save_dir = HA.save_dir)
nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 5)
nplt.plot_pal_data(LR_timings,save_dir = HA.save_dir)
nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 5)
nplt.plot_pal_data(LR_timings,save_dir = HA.save_dir)
nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 5)
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


