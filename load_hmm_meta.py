import analysis as ana
import blechpy
import new_plotting as nplt
import hmm_analysis as hmma 
import seaborn as sns
import os
import pandas as pd
import pylab as plt

rec_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts'
proj = blechpy.load_project(rec_dir)
proj.make_rec_info_table() #run this in case you changed around project stuff

PA = ana.ProjectAnalysis(proj)

PA.detect_held_units()#overwrite = True) #this part also gets the all units file
[all_units, held_df] = PA.get_unit_info()#overwrite = True) #run and check for correct area then run get best hmm

PA.process_single_units()#overwrite = True) #maybe run
#PA.run(overwrite = True)

HA = ana.HmmAnalysis(proj)
HA.get_hmm_overview()#overwrite = True) #use overwrrite = True to debug
#HA.sort_hmms_by_params()#overwrite = True)
HA.sort_hmms_by_BIC(overwrite = True)
srt_df = HA.get_sorted_hmms()
#HA.mark_early_and_late_states() #this is good, is writing in early and late states I think
best_hmms = HA.get_best_hmms(sorting = 'best_BIC', overwrite=False)

NB_meta,NB_decode,NB_best,NB_timings = HA.analyze_NB_ID(overwrite = False, parallel = True) #run with overwrite
NB_decode[['Y','epoch']] = NB_decode.Y.str.split('_',expand=True)
NB_decode['taste'] = NB_decode.trial_ID
NB_decode['state_num'] = NB_decode['hmm_state'].astype(int)

NB_decode['state_num'] = NB_decode['hmm_state'].astype('int64')
NB_decode['taste'] = NB_decode['trial_ID']
grcols = ['rec_dir','trial_num','taste','state_num']
NB_decsub = NB_decode[grcols+['p_correct']].drop_duplicates()

NB_timings = NB_timings.merge(NB_decsub, on = grcols, how = 'left')
NB_timings = NB_timings.drop_duplicates()
NB_timings[['Y','epoch']] = NB_timings.state_group.str.split('_',expand=True)
avg_timing = NB_timings.groupby(['exp_name','taste', 'state_group']).mean()[['t_start','t_end','t_med','duration']]
avg_timing = avg_timing.rename(columns = lambda x : 'avg_'+x)


NB_timings = pd.merge(NB_timings, avg_timing, on = ['exp_name', 'taste','state_group'], how = 'left').drop_duplicates()
NB_timings = NB_timings.reset_index()
idxcols1 = list(NB_timings.loc[:,'exp_name':'state_num'].columns)
idxcols2 = list(NB_timings.loc[:, 'pos_in_trial':].columns)
idxcols = idxcols1 + idxcols2
NB_timings = NB_timings.set_index(idxcols)
NB_timings = NB_timings.reset_index()
NB_timings = NB_timings.set_index(['exp_name','taste', 'state_group','session_trial','time_group'])
operating_columns = ['t_start','t_end','t_med', 'duration']

#remove all trials with less than 3 states
NB_timings = NB_timings.groupby(['rec_dir','taste','session_trial']).filter(lambda x: len(x) >= 3)

from scipy.stats import zscore
for i in operating_columns:
    zscorename = i+'_zscore'
    abszscorename = i+'_absZscore'
    NB_timings[zscorename] = NB_timings.groupby(['exp_name', 'state_group'])[i].transform(lambda x: zscore(x))
    NB_timings[zscorename] = NB_timings[zscorename].fillna(0)
    NB_timings[abszscorename] = abs(NB_timings[zscorename])
    
NB_timings = NB_timings.reset_index()
NB_timings['trial-group'] = NB_timings['trial_group']
NB_timings['session_trial'] = NB_timings.session_trial.astype(int) #make trial number an int
NB_timings['time_group'] = NB_timings.time_group.astype(int) #make trial number an int
