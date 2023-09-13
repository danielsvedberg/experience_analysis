#%% Start analysis from scratch

import analysis as ana
import blechpy
### load the project
#### requires having set up experiment and project objects beforehand
proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts' # directory where the project is
proj = blechpy.load_project(proj_dir) #load the project
proj.make_rec_info_table() #run this in case you changed around project stuff

### run the single-unit analyses, also a pre-req for the hmm analysis
PA = ana.ProjectAnalysis(proj) #create a project analysis object
PA.detect_held_units(overwrite=True) #establish the units, also gets the all units file
[all_units, held_df] = PA.get_unit_info(overwrite=True) #run and check for correct area then run get best hmm
PA.process_single_units(overwrite=True) #run the single-unit analysis, check function to see if all parts are working
PA.run(overwrite=True) #run several single-unit analyses

### run the hmm analysis
HA = ana.HmmAnalysis(proj) #create a hmm analysis object
HA.get_hmm_overview(overwrite=True) #extract all the hmms into a dataframe and saves it
HA.sort_hmms_by_AIC(overwrite=True) #label hmms in hmm_overview with lowest AIC for each recording, save to sorted hmms
best_hmms = HA.get_best_hmms(sorting='best_AIC', overwrite=True) #get rows of hmm_overview where sorting column==sorting arugument

### perform the naive bayes analysis of the hmm
NB_meta,NB_decode,NB_best,NB_timings = HA.analyze_NB_ID(overwrite=True, multi_process=True) #run with overwrite

#%% important calls to make checks
ov = HA.get_hmm_overview(overwrite=False) #get the hmm_overview dataframe
sorted = HA.sort_hmms_by_AIC(overwrite=False) #get hmm_overview sorted by best AIC
best = HA.get_best_hmms(sorting='best_AIC', overwrite=False) #get rows of hmm_overview where sorting column==sorting arugument


