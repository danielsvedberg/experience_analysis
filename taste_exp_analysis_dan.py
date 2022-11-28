
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

#you need to make a project analysis using blechpy.project() first
#rec_dir =  '/media/dsvedberg/Ubuntu Disk/taste_experience'
rec_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts'
proj = blechpy.load_project(rec_dir)
proj.make_rec_info_table() #run this in case you changed around project stuff

PA = ana.ProjectAnalysis(proj)

PA.detect_held_units()#overwrite = True) #this part also gets the all units file
[all_units, held_df] = PA.get_unit_info()#overwrite = True) #run and check for correct area then run get best hmm

PA.process_single_units()#overwrite = True) #maybe run
#PA.run()#overwrite = True)

HA = ana.HmmAnalysis(proj)
HA.get_hmm_overview()#overwrite = True) #use overwrrite = True to debug
#HA.sort_hmms_by_params()#overwrite = True)
HA.sort_hmms_by_BIC()#overwrite = True)
srt_df = HA.get_sorted_hmms()
HA.mark_early_and_late_states() #this is good, is writing in early and late states I think
best_hmms = HA.get_best_hmms(sorting = 'best_BIC')#overwrite = True, sorting = 'params #4') #HA has no attribute project analysis #this is not getting the early and late states 
#heads up, params #5 doesn't cover all animals, you want params #4
#play around with state timings in mark_early_and_late_states()

HA.plot_grouped_BIC()
HA.plot_best_BIC()
#check df for nan in HMMID or early or late state


# NB_res, NB_meta = hmma.analyze_NB_state_classification(best_hmms, all_units, prestim = True)
# decode_data = hmma.process_NB_classification(NB_meta,NB_res)
[NB_res,NB_meta,decode_data,best_hmms,timings] = HA.analyze_NB_ID(overwrite=True)
timings = timings.loc[timings.state_group == 'ID_state']

import seaborn as sns
timingsub = timings.loc[timings.exp_name != 'DS36']
timingsub = timingsub.rename(columns = {'trial_num': 'trial'})
timingsub = timingsub[['exp_name','time_group','exp_group','taste','trial','t_start','t_end','duration']]
timingsub = timingsub.loc[timingsub.duration > 5]
save_dir = HA.save_dir

yfacs = ['duration','t_start','t_end']
jfacs = ['Suc','NaCl','CA','QHCl']
for i in yfacs:
    for j in jfacs:
        sf = os.path.join(save_dir, 'trialno_v_'+i+'_'+j+'.svg')
        tastesub = timingsub.loc[timingsub.taste == j]
        g = sns.FacetGrid(tastesub, col = 'time_group', row = 'exp_group', hue = 'exp_name', margin_titles = True, aspect = 1, height = 5)
        g.map(sns.scatterplot, 'trial',i, s = 50)
        g.fig.subplots_adjust(top = 0.9)
        g.fig.suptitle('trial number vs ID '+i+': '+j)
        g.add_legend()
        g.savefig(sf)
        
df1 = pd.DataFrame()

ts = timingsub.loc[:,timingsub.columns != 'exp_name']
grouped_timings = ts.groupby(['time_group','exp_group','taste'])
for nm, grp in grouped_timings:
    print(grp)
    df = pd.DataFrame()
    gr = grp.loc[:,grp.columns != '']
    feat1s = []
    feat2s = []
    corrs = []
    p_values = []

    for feat1 in grp.columns:
        for feat2 in grp.columns:
            if feat1 != feat2:
                feat1s.append(feat1)
                feat2s.append(feat2)
                corr, p_value = scipy.stats.spearmanr(grp[feat1], grp[feat2])
                corrs.append(corr)
                p_values.append(p_value)

df['Feature_1'] = feat1s
df['Feature_2'] = feat2s
df['Correlation'] = corrs
df['p_value'] = p_values


lin_timings,lin_trials,lin_res,lin_meta = HA.analyze_pal_linear_regression()



# grouped_timings = timingsub.groupby(['time_group','exp_group','taste'])
# test = grouped_timings.corr().reset_index()
# test = test.loc[test.level_3 == 'trial']
# test2 = pd.calculate_pvalues(grouped_timings)

# nplt.plot_trialwise_bayesian_decoding(decode_data,plotdir = HA.save_dir)

# test = decode_data
# test = test.reset_index()
# test = test.loc[test.prestim_state != True]
# test = test[['exp_name','rec_group','trial_ID','hmm_state','hmm_id']].drop_duplicates()

# dat_file = '/media/dsvedberg/Ubuntu Disk/taste_experience/NB_results.dat'
# test = np.fromfile(dat_file)


# HA.analyze_hmms(overwrite = True) #comment out/adapt some analyses
# HA.plot_hmm_timing()


import new_plotting as nplt
import os
import plotting as plt
best_hmms = HA.get_best_hmms()
save_dir = HA.save_dir
seq_fn = os.path.join(save_dir, 'All_Sequences.png')
plt.plot_hmm_sequence_heatmap(best_hmms, 'time_group', seq_fn)
#anyting with confusion should be commented out because big error
#plot hmm timing should be run