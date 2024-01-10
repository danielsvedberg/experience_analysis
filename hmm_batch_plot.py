import blechpy
from blechpy.analysis import poissonHMM as phmm

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts' # directory where the project is
proj = blechpy.load_project(proj_dir) #load the project
rec_info = proj.rec_info.copy() #get the rec_info table
#loop through each recording, get the hmm, run plot_saved models
rec_dirs = rec_info['rec_dir'].tolist()
def plot_hmm(rec_dir):
    handler = phmm.HmmHandler(rec_dir)
    handler.plot_saved_models(file_ext='png')
#run plot_hmm in parallel across every item in rec_dirs
from joblib import Parallel, delayed
Parallel(n_jobs=4)(delayed(plot_hmm)(rec_dir) for rec_dir in rec_dirs)
