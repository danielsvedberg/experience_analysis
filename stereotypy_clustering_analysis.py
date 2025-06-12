import blechpy
import stereotypy_clustering_functions as scf
import blechpy.dio.h5io as h5io
import pandas as pd
from joblib import Parallel, delayed
import trialwise_analysis as ta
import analysis as ana
import numpy as np
import os


proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
rec_info = proj.rec_info.copy()  # get the rec_info table
rec_dirs = rec_info['rec_dir']

PA = ana.ProjectAnalysis(proj)

all_units, held_df = PA.get_unit_info(overwrite=False)

#%% consensus clustering (second attempt) with averaging distances for each trial and then performing consensus clustering

save_dir = PA.save_dir + os.sep + 'clustering_analysis'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
matrices, names, df = scf.make_consensus_matrix2(rec_info)

shuff_dir = PA.save_dir + os.sep + 'clustering_analysis/shuffles'
shuff_matrices,shuff_names_df,shuff_dfs = scf.make_consensus_matrix_shuffle(rec_info, n_iter=1000, overwrite=False, save_dir=shuff_dir)

scf.plot_correlation_matrices(matrices, names, save=True, save_dir=save_dir)
scf.plot_heirarchical_clustering(matrices, names, save=True, save_dir=save_dir)


naive_mats = matrices[:3]
naive_names = names[:3]
scf.plot_correlation_matrices_single(naive_mats,naive_names, save=True, save_dir=save_dir)
scf.plot_heirarchical_clustering_single(naive_mats, naive_names, save=True, save_dir=save_dir)

shuff_matrices = shuff_matrices.mean(axis=0)

shuff_matrices_naive = shuff_matrices[:3]
shuff_names_naive = shuff_names_df[['exp_group','session']].drop_duplicates()
shuff_names_naive = shuff_names_naive[shuff_names_naive['exp_group'] == 'naive']
#turn shuff_names_naive from a df to a list of tuples
shuff_names_naive = [tuple(row) for i, row in shuff_names_naive.iterrows()]
scf.plot_correlation_matrices_single(shuff_matrices_naive, shuff_names_naive, save=True, save_dir=shuff_dir, flag='shuffle')
scf.plot_heirarchical_clustering_single(shuff_matrices_naive, shuff_names_naive, save=True, save_dir=shuff_dir, flag='shuffle')

# sweep over different values of t to get the best silhouette score
thresholds = scf.get_consensus_thresholds(matrices, names)
# plot the dendograms
fig, leaves = scf.plot_heirarchical_clustering(matrices, names, threshold=thresholds, save=True, save_dir=save_dir)
naive_thresh = thresholds[:3]
fig, leaves = scf.plot_heirarchical_clustering_single(naive_mats, naive_names, threshold=thresholds, save=True, save_dir=save_dir)

#merge df with rec_info
df = pd.merge(df, rec_info, on=(['rec_dir', 'exp_group']))
shuff_df = pd.merge(shuff_dfs, rec_info, on=(['rec_dir', 'exp_group']))

#group df by rec_dir and scale top_branch_dist from 0 to 1 in each group
df['top_branch_dist'] = df.groupby('rec_dir')['top_branch_dist'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

df = scf.get_AB_clustering(df)
shuff_df = scf.get_AB_clustering(shuff_df)

#%% plot the cluster sizes
df_AB_long = scf.longform_AB_clustering(df)
#scf.plot_cluster_sizes(df_AB_long, save_dir=save_dir)
shuff_AB_long = scf.longform_AB_clustering(shuff_df, shuffle=True)

scf.plot_cluster_sizes_w_shuff(df_AB_long, shuff_AB_long, save_dir=save_dir)
scf.plot_cluster_sizes_w_shuff_single(df_AB_long, shuff_AB_long, save_dir=save_dir)
#%% plot the average trial of the two largest clusters
import pingouin as pg

df_AB_labels = scf.get_AB_cluster_labels(df)
newdf_naive = df_AB_labels[df_AB_labels['exp_group'] == 'naive']
scf.plot_cluster_avg_trial_naive(newdf_naive, save_dir=save_dir)

shuff_df_AB_labels = scf.get_AB_cluster_labels(shuff_df, shuffle=True)
shuff_df_AB_labs_naive = shuff_df_AB_labels[shuff_df_AB_labels['exp_group'] == 'naive']

aov = scf.plot_cluster_avg_trial_naive_w_shuff(newdf_naive, shuff_df_AB_labs_naive, save_dir=save_dir, flag='with_shuffle')
#%% plot the intra inter and null cluster distances
#compute intra and inter-cluster distances
intra_inter_df = scf.get_intra_inter_distances(df)

scf.plot_cluster_distances(intra_inter_df, save_dir=save_dir)
scf.plot_cluster_distances_naive(intra_inter_df, save_dir=save_dir)
