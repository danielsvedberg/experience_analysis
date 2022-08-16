
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

#you need to make a project analysis using blechpy.project() first
proj = blechpy.load_project('/media/dsvedberg/Ubuntu Disk/taste_experience')

PA = ana.ProjectAnalysis(proj)
PA.detect_held_units()#overwrite = True) #this part also gets the all units file
[all_units, held_df] = PA.get_unit_info()#overwrite = True) #run and check for correct area then run get best hmm
#PA.process_single_units() #maybe run

HA = ana.HmmAnalysis(proj)
HA.get_hmm_overview()#overwrite = True) #use overwrrite = True to debug
HA.sort_hmms_by_params()#overwrite = True)
srt_df = HA.get_sorted_hmms()
HA.mark_early_and_late_states() #this is good, is writing in early and late states I think
best_hmms = HA.get_best_hmms()#overwrite = True, sorting = 'params #4') #HA has no attribute project analysis #this is not getting the early and late states 
#heads up, params #5 doesn't cover all animals, you want params #4
#play around with state timings in mark_early_and_late_states()

#check df for nan in HMMID or early or late state
import hmm_analysis as hmma 

NB_res, NB_meta = hmma.analyze_NB_state_classification(best_hmms, all_units, prestim = True)
decode_data = hmma.process_NB_classification(NB_meta,NB_res)



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