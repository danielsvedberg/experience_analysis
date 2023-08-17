import analysis as ana
import blechpy
import pandas as pd

#%% Load analysis from files
proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts' # directory where the project is
proj = blechpy.load_project(proj_dir) #load the project
HA = ana.HmmAnalysis(proj) #create a hmm analysis object
decode = HA.get_NB_decode() #get the decode dataframe with some post-processing
timing = HA.get_NB_timing() #get the timing dataframe with some post-processing