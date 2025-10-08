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
import matplotlib
import math


proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
rec_info = proj.rec_info.copy()  # get the rec_info table
rec_dirs = rec_info['rec_dir']


#for all matplotlib plots, set the default font to Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['xtick.major.width'] = 0.72
plt.rcParams['ytick.major.width'] = 0.72
plt.rcParams['xtick.minor.width'] = 0.72
plt.rcParams['ytick.minor.width'] = 0.72
#set font size to 8
dfontsize = 8
plt.rcParams.update({'font.size': dfontsize})
default_margins = {'left': 0.1, 'right': 0.95, 'top': 0.9, 'bottom': 0.1, 'wspace': 0.05, 'hspace': 0.05}
summary_margins = {'left': 0.1, 'right': 0.95, 'top': 0.9, 'bottom': 0.3, 'wspace': 0.01, 'hspace': 0.01}
dPanW = 1.2
dPanH = 0.8
dPad = round(0.2/(8*(1/72)), 2)
dh_pad = round(0.09/(8*(1/72)), 2)

def detect_subplot_grid_shape(fig):
    """
    Attempt to detect the number of rows and columns of 'main' subplots
    in a figure by examining their positions in normalized figure coordinates.

    This assumes a rectangular grid of subplots laid out by something
    like plt.subplots(nrows, ncols) (i.e. uniform sized Axes).

    Returns
    -------
    (nrows, ncols) : tuple of int
    """
    # Get all Axes
    axes = fig.get_axes()

    # Filter to only "real" subplot Axes. We skip things like colorbars or inset_axes
    # A simple heuristic: check if it's an instance of SubplotBase.
    # Alternatively, you could skip any axes that appear to be legends or colorbars.
    real_subplots = []
    for ax in axes:
        # We also skip invisible or twin axes (to avoid duplicates).
        if (ax.get_visible() and
                isinstance(ax, matplotlib.axes.SubplotBase)):
            real_subplots.append(ax)

    if not real_subplots:
        return 0, 0  # No valid subplots found

    # Extract bounding boxes in normalized figure coords
    # We'll look at the center or the lower-left corner.
    # For consistent subplots, either approach works if they're arranged in a grid.
    x_positions = []
    y_positions = []
    for ax in real_subplots:
        bbox = ax.get_position()  # returns Bbox(x0, y0, x1, y1) in [0..1]
        # Let's use the center to avoid minor floating offsets on edges
        x_center = 0.5 * (bbox.x0 + bbox.x1)
        y_center = 0.5 * (bbox.y0 + bbox.y1)
        x_positions.append(x_center)
        y_positions.append(y_center)

    # Group unique centers for x and y, which correspond to columns and rows respectively
    # We'll define a small tolerance to group subplots in the same row/column
    # if their centers differ by less than some tiny threshold.
    def unique_positions(vals, tol=1e-3):
        # Sort them
        vals_sorted = sorted(vals)
        unique_vals = []
        current_group = None
        for v in vals_sorted:
            if current_group is None:
                current_group = v
                unique_vals.append(v)
            else:
                # if difference is large enough, we treat as a new group
                if abs(v - current_group) > tol:
                    unique_vals.append(v)
                    current_group = v
        return unique_vals

    unique_x = unique_positions(x_positions, tol=1e-3)
    unique_y = unique_positions(y_positions, tol=1e-3)

    ncols = len(unique_x)
    nrows = len(unique_y)

    return (nrows, ncols)

# get the size of a single panel in inches
def get_panel_frac(fig):
    # 4. Collect bounding boxes of all "real" subplots
    ax = fig.get_axes()[0]
    bb = ax.get_position()
    xW = bb.x1 - bb.x0
    yH = bb.y1 - bb.y0

    return xW, yH

def check_panel_size(fig, panel_width, panel_height):
    init_width, init_height = fig.get_size_inches()
    xW, yH = get_panel_frac(fig)
    subplot_width = xW * init_width
    subplot_height = yH * init_height
    panel_width_error = abs(panel_width - subplot_width)
    panel_height_error = abs(panel_height - subplot_height)

    if panel_width_error > 0.01 or panel_height_error > 0.01:
        print('Panel width error: ' + str(panel_width_error))
        print('Panel height error: ' + str(panel_height_error))
        return False
    else:
        return True

def round_up_1d(num):
    return math.ceil(num * 10) / 10

def adjust_figure_for_panel_size_auto(fig, panel_width=dPanW, panel_height=dPanH, do_second_tight=True):
    """
    Adjust an existing figure so that after tight_layout() the full grid of subplots
    has a total dimension of (ncols * panel_width) x (nrows * panel_height) inches,
    where nrows and ncols are automatically detected.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object (already containing subplots).
    panel_width : float
        Desired width (in inches) of each subplot *panel*.
    panel_height : float
        Desired height (in inches) of each subplot *panel*.
    do_second_tight : bool, optional
        If True, applies tight_layout() again after resizing the figure.
        This can slightly refine the result.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The same figure object, but resized so that the total subplot grid
        matches the target dimension (ncols * panel_width x nrows * panel_height).
    """
    # 1. round up panel width and height to 1 decimal place
    panel_width = round_up_1d(panel_width)
    panel_height = round_up_1d(panel_height)


    # 2. Detect the grid shape from the Axes
    nrows, ncols = detect_subplot_grid_shape(fig)
    if nrows == 0 or ncols == 0:
        # Could not detect a valid grid - return unchanged
        return fig

    #desired margins in inches:
    left = 0.6
    right = 0.1
    bottom = 0.3
    top = 0.2
    wspace = 0.1
    hspace = 0.1

    h_padding = top + bottom + (hspace * (nrows - 1))
    v_padding = left + right + (wspace * (ncols - 1))

    target_fig_size = (ncols * panel_width + v_padding, nrows * panel_height + h_padding)

    #now get the fraction of the figure that will be each padding
    left_frac = left / target_fig_size[0]
    right_frac = right / target_fig_size[0]
    bottom_frac = bottom / target_fig_size[1]
    top_frac = top / target_fig_size[1]
    wspace_frac = wspace/panel_width
    hspace_frac = hspace/panel_height

    #now, set the figure size to the target size
    fig.set_size_inches(target_fig_size)

    #and adjust the margins to the desired values
    fig.subplots_adjust(left=left_frac, right=1-right_frac, top=1-top_frac, bottom=bottom_frac, wspace=wspace_frac, hspace=hspace_frac)

    check = check_panel_size(fig, panel_width, panel_height)
    if check:
        print('Fig size adjusted to: ' + str(fig.get_size_inches()))
        return fig
    else:
        raise ValueError('Panel size not correct after adjustment')

'''
#dflist = []
#for rec in rec_dirs:
#    df = scf.get_neuron_intertrial_correlations(rec)
#    dflist.append(df)

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
sns.stripplot(x='trial_group_comp', y='correlation', data=df3, jitter=0.3, color='black', alpha=0.05)
sns.barplot(x='trial_group_comp', y='correlation', data=df3)
plt.show()
'''


#paralleling the above code
dflist = Parallel(n_jobs=-1)(delayed(scf.get_neuron_intertrialgroup_correlations)(rec) for rec in rec_dirs)
df = pd.concat(dflist, axis=0)

#merge in rec_info along rec_dir
rec_info = rec_info[['rec_dir', 'exp_group', 'exp_name', 'rec_num']]
df = pd.merge(df, rec_info, on='rec_dir')
df['session'] = df['rec_num']
df['an_neuron'] = str(df['neuron']) + '_' + df['exp_name']

df['trial_group_comp'] = df['trial_group'] + ' vs ' + df['comp_trial_group']
#replace spaces in the column trial group comp with \n
df['trial_group_comp'] = df['trial_group_comp'].str.replace(' ', '\n')

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

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig.set_size_inches(2,1.5)

sns.barplot(x='trial_group_comp', y='correlation',
                 data=naiveS1, facecolor=(0,0,0,0), edgecolor='black', capsize=0.05, linewidth=0.5, errcolor='black',
            ax=ax)
#make the x axis label "trials"
ax.set_xlabel('Trials', fontsize=8)
#make the y axis label "correlation (Pearson's R)
ax.set_ylabel('Mean Correlation', fontsize=8)
#increase the y max to 0.5
ax.set_ylim(0, 0.5)
#increase the axis tick font size to 18
ax.tick_params(axis='both', which='major', labelsize=8)

comparisons = [('1-10\nvs\n11-20', '11-20\nvs\n21-30', 0.047152),  # (group A, group B, corrected p-value)
               ('1-10\nvs\n21-30', '11-20\nvs\n21-30', 0.003134)]
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
        ax.text((group_A_pos + group_B_pos) * 0.5, bar_height, sig_text, ha='center', size=8)
        # print the p value in clear text above the significance stars
        ax.text((group_A_pos + group_B_pos) * 0.5 + 0.5, bar_height + 0.005, f'p={p_val:.3f}', ha='right', size=8)

# Apply the function to add significance bars to the plot
add_sig_bars(ax, comparisons)
# Adjust the figure to fit the panel size
fig = adjust_figure_for_panel_size_auto(fig, panel_width=dPanW, panel_height=dPanH, do_second_tight=True)

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