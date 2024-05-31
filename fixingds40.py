#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 15:15:37 2022

@author: dsvedberg
"""
import blechpy
import pandas as pd
import blechpy.dio.h5io as h5io
import tables
import numpy as np

dat = blechpy.load_dataset('/media/dsvedberg/Ubuntu Disk/taste_experience_resorts/DS40/DS40_spont_taste_201104_150441')
trials= dat.dig_in_trials
badtrials =trials#trials.loc[trials.channel == 1]
badidxs = badtrials['on_index'].array
badidxs = badidxs[::-1]

h5_file = dat.h5_file

#fixing the sorted unit arrays
h5 = tables.open_file(h5_file, mode = 'r+')
for unit in h5.root.sorted_units:
    unm = unit._v_name
    for bidx in badidxs: 
        times = h5.root.sorted_units[unm].times[:]
        waves = h5.root.sorted_units[unm].waveforms[:]
        t = times[(times > bidx - 30) & (times < bidx + 30)]
        if len(t) > 0:
            idx = np.where(times==t)
            idx = idx[0][0]
            print(len(h5.root.sorted_units[unm].times[:]))
            newtimes = np.delete(times,idx)
            newwaveforms = np.delete(waves,idx,0)
            h5.remove_node(h5.root.sorted_units[unm].times)
            h5.create_array(h5.root.sorted_units[unm],"times", newtimes)
            h5.remove_node(h5.root.sorted_units[unm].waveforms)
            h5.create_array(h5.root.sorted_units[unm],"waveforms", newwaveforms)

#fixing the spike arrays
h5 = tables.open_file(h5_file, mode = 'r+')
new_array = h5.root.spike_trains.dig_in_1.spike_array[:]
new_array[:,:,2000] = 0
h5.remove_node(h5.root.spike_trains.dig_in_1.spike_array)
h5.create_array(h5.root.spike_trains.dig_in_1, "spike_array", new_array)

h5.flush()
h5.close()