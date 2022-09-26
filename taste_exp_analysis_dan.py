
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

#you need to make a project analysis using blechpy.project() first
#rec_dir =  '/media/dsvedberg/Ubuntu Disk/taste_experience'
rec_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts'
proj = blechpy.load_project(rec_dir)
proj.make_rec_info_table() #run this in case you changed around project stuff

PA = ana.ProjectAnalysis(proj)

PA.detect_held_units(overwrite = True) #this part also gets the all units file
[all_units, held_df] = PA.get_unit_info(overwrite = True) #run and check for correct area then run get best hmm

#PA.process_single_units(overwrite = True) #maybe run
#PA.run()#overwrite = True)

HA = ana.HmmAnalysis(proj)
HA.get_hmm_overview(overwrite = True) #use overwrrite = True to debug
#HA.sort_hmms_by_params()#overwrite = True)
HA.sort_hmms_by_BIC(overwrite = True)
srt_df = HA.get_sorted_hmms()
HA.mark_early_and_late_states() #this is good, is writing in early and late states I think
best_hmms = HA.get_best_hmms(overwrite = True, sorting = 'best_BIC')#overwrite = True, sorting = 'params #4') #HA has no attribute project analysis #this is not getting the early and late states 
#heads up, params #5 doesn't cover all animals, you want params #4
#play around with state timings in mark_early_and_late_states()

#check df for nan in HMMID or early or late state


# NB_res, NB_meta = hmma.analyze_NB_state_classification(best_hmms, all_units, prestim = True)
# decode_data = hmma.process_NB_classification(NB_meta,NB_res)
[NB_res,NB_meta,decode_data,best_hmms,timings] = HA.analyze_NB_ID(save=True)
timings = timings.loc[timings.state_group == 'ID']

import seaborn as sns
timingsub = timings.loc[timings.exp_name != 'DS36']
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

for i, group in timingsub.groupby('rec_group'):
    g = sns.pairplot(data=group, hue = 'taste')
    sf = os.path.join(save_dir,i + '_pairplots.svg')
    g.savefig(sf)
#TODO 080722: 2 state hmm solutions fucking life up. Abu said just cut out 2 state models, gotta redo this shit. 



nplt.plot_trialwise_bayesian_decoding(decode_data,plotdir = HA.save_dir)

test = decode_data
test = test.reset_index()
test = test.loc[test.prestim_state != True]
test = test[['exp_name','rec_group','trial_ID','hmm_state','hmm_id']].drop_duplicates()

dat_file = '/media/dsvedberg/Ubuntu Disk/taste_experience/NB_results.dat'
test = np.fromfile(dat_file)


HA.analyze_hmms(overwrite = True) #comment out/adapt some analyses
HA.plot_hmm_timing()




import new_plotting as nplt
import os
import plotting as plt
best_hmms = HA.get_best_hmms()
save_dir = HA.save_dir
seq_fn = os.path.join(save_dir, 'All_Sequences.png')
plt.plot_hmm_sequence_heatmap(best_hmms, 'time_group', seq_fn)
#anyting with confusion should be commented out because big error
#plot hmm timing should be run