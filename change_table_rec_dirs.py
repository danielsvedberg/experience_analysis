import blechpy
import analysis as ana
import hmm_analysis as hmma

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy' # directory where the project is

# Load the project
proj = blechpy.load_project(proj_dir) #load the project
PA = ana.ProjectAnalysis(proj) #create a project analysis object
HA = ana.HmmAnalysis(proj) #create a hmm analysis object

