import blechpy
from blechpy.analysis import poissonHMM as phmm
import analysis as ana
import pandas as pd
proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts' # directory where the project is
proj = blechpy.load_project(proj_dir) #load the project
rec_info = proj.get_rec_info()

overviews = []
for i, row in rec_info.iterrows():
    rec_dir = row['rec_dir']
    handler = phmm.HmmHandler(rec_dir)
    ov = handler.get_data_overview()
    max_rates = []

    for j, r in ov.iterrows():
        hmm_id = r['hmm_id']
        hmm, _, _ = handler.get_hmm(hmm_id)
        emissions = hmm.emission
        max_rate = emissions.max()
        max_rates.append(max_rate)
    ov['max_rate'] = max_rates
    ov['rec_dir'] = rec_dir
    overviews.append(ov)
overviews = pd.concat(overviews)

test = overviews.loc[overviews.max_rate > 100]
