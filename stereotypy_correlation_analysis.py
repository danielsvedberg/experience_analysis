import blechpy
import stereotypy_clustering_functions as scf
import blechpy.dio.h5io as h5io
import pandas as pd
from joblib import Parallel, delayed
import trialwise_analysis as ta
import analysis as ana
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
rec_info = proj.rec_info.copy()  # get the rec_info table
rec_dirs = rec_info['rec_dir']

dflist = []
for rec in rec_dirs:
    df = scf.get_neuron_intertrial_correlations(rec)
    dflist.append(df)

#paralleling the above code
dflist = Parallel(n_jobs=-1)(delayed(scf.get_neuron_intertrial_correlations)(rec) for rec in rec_dirs)
df = pd.concat(dflist, axis=0)
#merge in rec_info along rec_dir
df = pd.merge(df, rec_info, on='rec_dir')
#remove all rows with na in corr
df = df.dropna(subset=['correlation'])

#create column called trial_group where trials 1-10 are '1-10', 11-20 are '11-20', etc
df['trial_group'] = df['taste_trial'].apply(lambda x: str(10*(x//10)+1) + '-' + str(10*(x//10)+10))
df['comp_trial_group'] = df['comp_trial'].apply(lambda x: str(10*(x//10)+1) + '-' + str(10*(x//10)+10))
df['trial_group_comp'] = df['comp_trial_group'] + ' vs ' +  df['trial_group']

#create a copy of df where rows in trial_group_comp with '1-10 vs 1-10' etc are removed
df2 = df[df['trial_group'] != df['comp_trial_group']]
#order by trial_group column so '1

df3 = df2.groupby(['trial_group_comp', 'rec_dir', 'taste', 'neuron']).mean().reset_index()

#plot a barplot of the correlation values for each trial_group_comp, with a stripplot overlaid
ax = sns.stripplot(x='trial_group_comp', y='correlation', data=df3, jitter=0.3, color='black', alpha=0.05)

sns.barplot(x='trial_group_comp', y='correlation', data=df3)
plt.show()


#paralleling the above code
dflist = Parallel(n_jobs=-1)(delayed(scf.get_neuron_intertrialgroup_correlations)(rec) for rec in rec_dirs)
df = pd.concat(dflist, axis=0)
#merge in rec_info along rec_dir
rec_info = rec_info[['rec_dir', 'exp_group', 'exp_name', 'rec_num']]
df = pd.merge(df, rec_info, on='rec_dir')
df['session'] = df['rec_num']
df['an_neuron'] = str(df['neuron']) + '_' + df['exp_name']

df['trial_group_comp'] = df['trial_group'] + ' vs ' + df['comp_trial_group']
#remove DS33 and DS41 using query
df = df.query('exp_name != "DS33" and exp_name != "DS41"')
#filter only rows where exp group is 'naive'
naive = df[df['exp_group'] == 'naive']
naiveS1 = naive[naive['session'] == 1]

#perform an anova on the correlation values for each trial_group_comp
import pingouin as pg
aov = pg.rm_anova(data=naiveS1, dv='correlation', within='trial_group_comp', subject='an_neuron')
pw = pg.pairwise_ttests(data=naiveS1, dv='correlation', within='trial_group_comp', subject='an_neuron', padjust='bonf')
#use the p values from p-corr column in pw to add significance bars and stars
sig_comps = pw[pw['p-corr'] < 0.05]


#ax = sns.violinplot(x='trial_group_comp', y='correlation', data=naiveS1, inner=None, color='lightgray')
ax = sns.barplot(x='trial_group_comp', y='correlation', data=naiveS1, facecolor=(0,0,0,0), edgecolor='black', capsize=0.1)
#make the x axis label "trials"
ax.set_xlabel('Trials')
#make the y axis label "correlation (Pearson's R)
ax.set_ylabel('Mean Correlation')
#increase the y max to 0.5
ax.set_ylim(0, 0.5)
#increase the axis tick font size to 18
ax.tick_params(axis='both', which='major', labelsize=15)

comparisons = [('1-10 vs 11-20', '11-20 vs 21-30', 0.047152),  # (group A, group B, corrected p-value)
               ('1-10 vs 21-30', '11-20 vs 21-30', 0.003134)]
# Add significance bars
def add_sig_bars(ax, comparisons, y_offset=0.01, height_increase_per_group=0.05, min_height=0.1):
    used_heights = {}

    for i, (group_A, group_B, p_val) in enumerate(comparisons):
        # Define the position for each group on the x-axis
        group_A_pos = list(naiveS1['trial_group_comp']).index(group_A)
        group_B_pos = list(naiveS1['trial_group_comp']).index(group_B)

        # Get the max y-value between the two bars
        y_max = max(ax.patches[group_A_pos].get_height(), ax.patches[group_B_pos].get_height())

        # Calculate the comparison width (difference in x-axis positions)
        comparison_width = abs(group_B_pos - group_A_pos)

        # Ensure that wider comparisons are plotted higher
        if comparison_width not in used_heights:
            # First time we're seeing this comparison width, so set the height
            bar_height = y_max + y_offset + (comparison_width * height_increase_per_group)
            used_heights[comparison_width] = bar_height
        else:
            # If the width is already used, use a slightly higher height for subsequent comparisons
            bar_height = used_heights[comparison_width] + height_increase_per_group
            used_heights[comparison_width] = bar_height

        # Draw the horizontal significance line between the two bars
        ax.plot([group_A_pos, group_B_pos], [bar_height, bar_height], color='black')

        # Draw the vertical tick lines at the ends
        ax.plot([group_A_pos, group_A_pos], [bar_height, bar_height - 0.01], color='black')
        ax.plot([group_B_pos, group_B_pos], [bar_height, bar_height - 0.01], color='black')

        # Add the stars for significance
        if p_val < 0.001:
            sig_text = '***'
        elif p_val < 0.01:
            sig_text = '**'
        elif p_val < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'  # not significant

        # Add the significance stars
        ax.text((group_A_pos + group_B_pos) * 0.5, bar_height, sig_text, ha='center', size=20)
        # print the p value in clear text above the significance stars
        ax.text((group_A_pos + group_B_pos) * 0.5 + 0.5, bar_height + 0.005, f'p={p_val:.3f}', ha='right')

# Apply the function to add significance bars to the plot
add_sig_bars(ax, comparisons)
plt.tight_layout()
#save the plot to svg
PA = ana.ProjectAnalysis(proj)
save_dir = PA.save_dir + os.sep + 'trial_group_correlations'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = save_dir + os.sep + 'trial_group_correlations.svg'
plt.savefig(save_path)

HA = ana.HmmAnalysis(proj)
ov = HA.get_hmm_overview(overwrite=False)
#get rows in ov where exp_name is 'DS39'
DS39 = ov[ov['exp_name'] == 'DS39']
best_hmms = HA.get_best_hmms(sorting='best_AIC', overwrite=False)
best_hmms_DS39 = best_hmms[best_hmms['exp_name'] == 'DS39']