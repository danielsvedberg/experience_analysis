#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:29:45 2022

@author: dsvedberg
"""

from blechpy.analysis import poissonHMM as phmm

rec_dir = '/media/dsvedberg/Ubuntu Disk/TE_sandbox/DS39_spont_taste_201029_154308'
handler = phmm.HmmHandler(rec_dir)

params = [{'threshold':1e-1, 'max_iter':500, 'n_repeats':5,'time_start':-500, 'time_end': 2500, 'taste': ['Suc','NaCl','CA','QHCl'],'n_states': x} for x in [10,11]]
handler.add_params(params)
handler.run()