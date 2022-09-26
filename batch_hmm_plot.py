#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 16:37:13 2022

@author: dsvedberg
"""

import blechpy
from blechpy.analysis import poissonHMM as phmm
from joblib import Parallel, delayed

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts'
proj = blechpy.load_project(proj_dir)

rec_info = proj.get_rec_info()
rec_dirs = rec_info.rec_dir

def load_plot_hmm(rec_dir):
    handler = phmm.HmmHandler(rec_dir)
    handler.plot_saved_models()
    
Parallel(n_jobs = 6)(delayed(load_plot_hmm)(i) for i in rec_dirs)