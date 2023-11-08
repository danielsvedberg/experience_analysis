import blechpy
proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy' # directory where the project is
proj = blechpy.load_project(proj_dir) #load the project
proj._change_root(proj_dir)
proj.save()

proj.make_rec_info_table()
rec_info = proj.get_rec_info()
for nm, group in rec_info.groupby('exp_dir'):
    print(nm)
    exp = blechpy.load_experiment(nm)
    exp._change_root(nm)
    exp.save()

for nm, group in rec_info.groupby('rec_dir'):
    print(nm)
    dat = blechpy.load_dataset(nm)
    dat._change_root(nm)
    dat.save()