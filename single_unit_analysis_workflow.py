import analysis as ana
import blechpy
### load the project
#### requires having set up experiment and project objects beforehand
proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts' # directory where the project is
proj = blechpy.load_project(proj_dir) #load the project
proj.make_rec_info_table() #run this in case you changed around project stuff

### run the single-unit analyses, also a pre-req for the hmm analysis
PA = ana.ProjectAnalysis(proj) #create a project analysis object
[all_units, held_df] = PA.get_unit_info() #run and check for correct area then run get best hmm