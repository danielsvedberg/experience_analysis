#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 18:55:17 2023

@author: dsvedberg
"""
import blechpy
pd = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts'
proj = blechpy.load_project(pd)

ri = proj.get_rec_info()

for i in ri['rec_dir']:
    dat = blechpy.load_dataset(i)
    dat.export_TrialSpikeArrays2Mat()