#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 12:32:58 2021

@author: dsvedberg
"""

# import numpy as np
# import matplotlib.pyplot as plt
# import blechpy as bp
# from blechpy.analysis import poissonHMM as phmm

# d1 = ['/media/dsvedberg/T7/DS36/DS36_spont_taste_200924_145722',
#       '/media/dsvedberg/T7/DS39/DS39_spont_taste_201029_154308',
#       '/media/dsvedberg/T7/DS40/DS40_spont_taste_201103_150101']
# d3 = ['/media/dsvedberg/T7/DS36/DS36_spont_taste_200926_152041',
#       '/media/dsvedberg/T7/DS39/DS39_spont_taste_201031_101959',
#       '/media/dsvedberg/T7/DS40/DS40_spont_taste_201105_140926']

# handler = phmm.HmmHandler('/media/dsvedberg/T7/DS39/DS39_spont_taste_201029_154308')
# hmm, dt, params = handler.get_hmm(1)
# sequences = hmm.stat_arrays['best_sequences']

# first_trans = np.argmax(sequences==1, axis = 1)-500
# second_trans = np.argmax(sequences==2,axis = 1)-500
# third_trans = np.argmax(sequences==3, axis = 1)-500

# n_bins = len(first_trans)
# fig,ax = plt.subplots(figsize=(8,4))
# n, bins, patches = ax.hist(first_trans, n_bins,  density = True, histtype = 'step', cumulative = True, label = 'Empirical')

# #aim 1: examine baseline delay

# ##
#get into the directory
import analysis as ana
import blechpy as bp
proj = bp.load_project('/media/dsvedberg/T7/')

PA = ana.ProjectAnalysis(proj)
PA.detect_held_units() #step 1
PA.get_unit_info() #run and check for correct area then run get best hmm
PA.process_single_units() #maybe run

HA = ana.HmmAnalysis(proj)
HA.get_hmm_overview()
HA.sort_hmms_by_params(overwrite = True)
HA.mark_early_and_late_states()
HA.get_best_hmms(overwrite = True, sorting = 'params #3') #HA has no attribute project analysis

#play around with state timings in mark_early_and_late_states()

#check df for nan in HMMID or early or late state
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