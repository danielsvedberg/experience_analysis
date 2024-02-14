import analysis as ana
import blechpy
import new_plotting as nplt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
import trialwise_analysis as ta
import scipy.stats as stats

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
HA = ana.HmmAnalysis(proj)  # create a hmm analysis object

NB_df = HA.analyze_NB_ID2(overwrite=False)
NB_df['duration'] = NB_df['t_end'] - NB_df['t_start']

#select rows where t_start is less than 500
early_df = NB_df[NB_df['t_start'] < 500]
#select rows where t_end is less than 1500
early_df = early_df[early_df['t_end'] < 1500]
#then, for each grouping of taste and rec_dir, select the row with the longest duration
early_df = early_df.loc[early_df.groupby(['taste', 'rec_dir'])['duration'].idxmax()]
