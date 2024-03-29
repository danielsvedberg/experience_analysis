import aggregation as agg
import os
import numpy as np
import pandas as pd
import pylab as plt
from scipy.stats import sem, norm, chi2_contingency, rankdata, spearmanr
import itertools as it
from plotting import add_suplabels, change_hue
import analysis_stats as stats
from scipy.ndimage.filters import gaussian_filter1d
from blechpy import load_dataset
import seaborn as sns
from joblib import Parallel, delayed

# I might need to replace a bunch of x['time_group'] under df['grouping']

TASTE_COLORS = {'Suc': 'tab:blue', 'NaCl': 'tab:red', 'QHCl': 'tab:purple', 'CA': 'tab:yellow', 'Spont': 'tab:green'}
EXP_COLORS = {'suc_preexp': 'tab:blue', 'naive': 'tab:green'}
ORDERS = {'exp_group': ['naive', 'suc_preexp'],
          # 'cta_group': ['CTA', 'No CTA'],
          'taste': ['Suc', 'NaCl', 'CA', 'QHCl'],
          # 'time_group': ['Exposure_1','Exposure_2','Exposure_3'],
          'time_group': ['1', '2', '3'],
          'state_group': ['early', 'late'],
          'MDS_time': ['Early (0-750ms)', 'Late (750-1500ms)'],
          'unit_type': ['pyramidal', 'interneuron'],
          'state_presence': ['both', 'early_only', 'late_only', 'neither'],
          'trial_group': [1, 2, 3, 4, 5, 6]}


def plot_multiple_piecewise_regression(results_df, grvars=['exp_group', 'exp_name'], save_dir=None):
    # Get unique values for 'exp_group' and 'time_group'
    time_groups = results_df['time_group'].unique()

    # Specify the order of 'taste'
    tastes = ['Suc', 'NaCl', 'CA', 'QHCl']

    # Loop over each 'exp_group'
    for name, df_exp_group in results_df.groupby(grvars):
        if len(grvars) >= 2:
            grnames = '_'.join(name)
        else:
            grnames = name

        trial_col = str(df_exp_group['trial_col'].iloc[1])
        response_col = str(df_exp_group['response_col'].iloc[1])

        # Get the limits of the data
        x_min = df_exp_group['pw_fit'].apply(lambda fit: min(fit.xx)).min()
        x_max = df_exp_group['pw_fit'].apply(lambda fit: max(fit.xx)).max()
        y_min = df_exp_group['pw_fit'].apply(lambda fit: min(fit.yy)).min()
        y_max = df_exp_group['pw_fit'].apply(lambda fit: max(fit.yy)).max()

        # Calculate the buffer (10% of the range of the data)
        x_buffer = (x_max - x_min) * 0.1
        y_buffer = (y_max - y_min) * 0.1

        # Adjust the limits with the buffer
        x_min -= x_buffer
        x_max += x_buffer
        y_min -= y_buffer
        y_max += y_buffer

        # Create a subplot for each combination of 'taste' and 'time_group'
        fig, axs = plt.subplots(len(tastes), len(time_groups), figsize=(15, 10))
        for i, taste in enumerate(tastes):
            for j, time_group in enumerate(time_groups):
                ax = axs[i, j]
                fits = df_exp_group[(df_exp_group['taste'] == taste) & (df_exp_group['time_group'] == time_group)][
                    'pw_fit'].values
                plt.sca(axs[i, j])
                for fit in fits:
                    fit.plot()
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                # Add x-axis label for the bottom row of subplots
                if i == len(tastes) - 1:
                    ax.set_xlabel(trial_col)
                # Add y-axis label for the leftmost subplots
                if j == 0:
                    ax.set_ylabel(response_col, fontsize = 14)
                # Add taste label for the rightmost subplots
                if j == len(time_groups) - 1:
                    ax.text(1.05, 0.5, taste, rotation=-90, size='xx-large', weight='bold', transform=ax.transAxes)

        # Set the subplot titles at the margins of the subplot grid
        for ax, time_group in zip(axs[0], time_groups):
            ax.set_title(f"Session: {time_group}")

        plt.suptitle(f"Experiment Group: {grnames}", y=0.98)
        plt.subplots_adjust(top=0.9)

        if save_dir is not None:
            fn = f'{grnames}_{trial_col}_{response_col}_piecewise_regression.png'
            plt.savefig(os.path.join(save_dir, fn))
            plt.close()
        else:
            plt.show()

def plot_multiple_piecewise_regression2(results_df, save_dir=None):
    # Get unique values for 'exp_group' and 'time_group'
    time_groups = results_df['time_group'].unique()

    # Specify the order of 'taste'
    tastes = ['Suc', 'NaCl', 'CA', 'QHCl']

    # Loop over each 'exp_group'
    for name, df_exp_group in results_df.groupby(['exp_group', 'exp_name']):
        exp_group = name[0]
        exp_name = name[1]

        trial_col = str(df_exp_group['trial_col'].iloc[1])
        response_col = str(df_exp_group['response_col'].iloc[1])

        # Get the limits of the data
        x_min = df_exp_group['pw_fit'].apply(lambda fit: min(fit.xx)).min()
        x_max = df_exp_group['pw_fit'].apply(lambda fit: max(fit.xx)).max()
        y_min = df_exp_group['pw_fit'].apply(lambda fit: min(fit.yy)).min()
        y_max = df_exp_group['pw_fit'].apply(lambda fit: max(fit.yy)).max()

        # Calculate the buffer (10% of the range of the data)
        x_buffer = (x_max - x_min) * 0.1
        y_buffer = (y_max - y_min) * 0.1

        # Adjust the limits with the buffer
        x_min -= x_buffer
        x_max += x_buffer
        y_min -= y_buffer
        y_max += y_buffer

        # Create a subplot for each combination of 'taste' and 'time_group'
        fig, axs = plt.subplots(len(tastes), len(time_groups), figsize=(15, 10))
        for i, taste in enumerate(tastes):
            for j, time_group in enumerate(time_groups):
                ax = axs[i, j]
                fits = df_exp_group[(df_exp_group['taste'] == taste) & (df_exp_group['time_group'] == time_group)][
                    'pw_fit'].values
                plt.sca(axs[i, j])
                for fit in fits:
                    fit.plot_data(color='gray', alpha=0.005)
                    fit.plot_breakpoint_confidence_intervals()
                    fit.plot_breakpoints()
                    fit.plot_fit(color='black')
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                # Add x-axis label for the bottom row of subplots
                if i == len(tastes) - 1:
                    ax.set_xlabel(trial_col)
                # Add y-axis label for the leftmost subplots
                if j == 0:
                    ax.set_ylabel(response_col, fontsize=14)
                # Add taste label for the rightmost subplots
                if j == len(time_groups) - 1:
                    ax.text(1.05, 0.5, taste, rotation=-90, size='xx-large', weight='bold', transform=ax.transAxes)

        # Set the subplot titles at the margins of the subplot grid
        for ax, time_group in zip(axs[0], time_groups):
            ax.set_title(f"Session: {time_group}")
        plt.suptitle(f"Experiment Group: {exp_group} , {exp_name}", y=0.98)
        plt.subplots_adjust(top=0.9)

        if save_dir is not None:
            fn = f'{exp_group}_{exp_name}_{trial_col}_{response_col}_piecewise_regression.png'
            plt.savefig(os.path.join(save_dir, fn))
            plt.close()
        else:
            plt.show()

def plot_changepoint_histograms(changepoints_df, save_dir = None):
    """
    Plot the histogram of the changepoints for each group.

    Parameters
    ----------
    changepoints_df : pandas.DataFrame
        Output from the detect_changepoints function.
    """
    # Unpack the list of changepoints into separate rows
    df_long = changepoints_df.explode('changepoints').reset_index(drop=True)
    df_long['session'] = df_long.time_group
    for name, group in df_long.groupby('exp_group'):
        # Create a FacetGrid with 'taste' dictating the rows and 'time_group' the columns
        grid = sns.FacetGrid(group, row='taste', col='session', palette='Set1', margin_titles=True, legend_out=True)

        # Plot the histograms
        grid.map_dataframe(sns.histplot, x='changepoints', bins=6, hue='exp_name', stat="count", multiple="stack",
                           edgecolor='k', legend=True)
        if save_dir:
            sf = f'{name}_changepoint_histograms.png'
            save_path = os.path.join(save_dir, sf)
            plt.savefig(save_path)
            plt.close()
        else:
            return grid


def plot_piecewise_individual_models(results_df, session_col, taste_col, condition_col, trial_col, response_col,
                                     subject_col, save_dir):
    results_df = results_df.copy()
    results_df['session'] = results_df[session_col].astype(str)
    results_df['trial'] = results_df[trial_col]
    # Creating a FacetGrid
    results_df.sort_values(by=[taste_col, condition_col, subject_col, 'trial'], inplace=True)
    g = sns.FacetGrid(results_df, row=taste_col, col='session', hue=condition_col,
                      row_order=['Suc', 'NaCl', 'CA', 'QHCl'], height=5, margin_titles=True)

    # Plotting raw data points
    g.map_dataframe(sns.scatterplot, x='trial', y=response_col, alpha=0.25)

    # Plotting the models for each subject
    g.map_dataframe(sns.lineplot, x=trial_col, y='prediction', style=subject_col, ci=None, linewidth=2)

    g.set_axis_labels(y_var=response_col)
    # Add a legend
    g.add_legend(title_fontsize='x-large', fontsize='x-large')
    plt.show()
    sf = os.path.join(save_dir, 'piecewise_model_individual.png')
    g.savefig(sf)


def plot_piecewise_model(df, model, changepoint, trial_col, response_col, condition_col, taste_col, session_col,
                         save_dir):
    row_order = ORDERS['taste']
    # Create a new variable representing trials after the changepoint
    df['trial_after'] = np.where(df[trial_col] > changepoint, df[trial_col] - changepoint, 0)

    # Generate the predicted values from the model
    df['predicted'] = model.predict(df)

    # Initialize a FacetGrid object
    g = sns.FacetGrid(df, col=session_col, row=taste_col, hue=condition_col, row_order=row_order, height=5, aspect=1,
                      margin_titles=True)

    # Map the scatter plot for observed values
    g.map(sns.scatterplot, trial_col, response_col, alpha=0.5)

    # Map the line plot for predicted values
    g.map(sns.lineplot, trial_col, 'predicted')

    # Add a vertical line at the changepoint
    for ax in g.axes.flat:
        ax.axvline(x=changepoint, color='green', linestyle='--')

    # Add legend
    g.add_legend()

    # Show the plot
    plt.show()
    sf = os.path.join(save_dir, 'piecewise_model.png')
    g.savefig(sf)


def plotGamma(group, name, save_dir, xax='time', yax="gamma_mode", row="session", col="trial_bin", hue="exp_group"):
    group = group.reset_index()
    p = sns.relplot(data=group, x=xax, y=yax, row="session", col=col, hue=hue, kind="line",
                    facet_kws={'margin_titles': True})
    p.tight_layout()
    p.set_ylabels("p(gamma) of mode state")
    # p.move_legend("lower center", bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False)
    if len(name) > 1:
        name = '_'.join(name)
    p.fig.suptitle(name)
    p.fig.subplots_adjust(top=0.9)
    plt.ylim(0, 1)
    fn = name + "_" + yax + "X" + col + ".png"
    sf = os.path.join(save_dir, fn)
    p.savefig(sf)


def plotGammaPar(gamma_df, save_dir, groups=['taste'], xax='time', yax="gamma_mode", row="session", col="trial_bin",
                 hue="exp_group"):
    Parallel(n_jobs=8)(
        delayed(plotGamma)(group, name, save_dir, xax=xax, yax=yax, row=row, col=col, hue=hue) for name, group in
        gamma_df.groupby(groups))


def plotGammaBar(group, name, save_dir, yax="gamma_mode"):
    sns.set(style="whitegrid")
    group = group.reset_index()

    # Create a FacetGrid with the desired structure
    g = sns.FacetGrid(group, col="time_group", row="exp_group", height=4, aspect=1, margin_titles=True)

    # Add a bar plot to the FacetGrid for each panel
    g.map(sns.violinplot, "trial_group", yax, order=None, ci=None, color="lightblue")

    # Add a swarm plot to the FacetGrid for each panel
    # g.map(sns.stripplot, "trial_group", yax, order=None, color="black")

    # Set the axis labels
    g.set_axis_labels("trial_group", yax)

    # Add a title for each panel
    # g.set_titles("time_group = {session #}, exp_group = {experiment group}")

    # Adjust the space between panels
    # g.fig.subplots_adjust(wspace=.05, hspace=.15)

    sn = '_'.join(name)
    fn = sn + "_" + yax + "_barplot.png"
    sf = os.path.join(save_dir, fn)
    g.savefig(sf)


def plotGammaBarPar(gamma_df, save_dir, yax="gamma_mode"):
    Parallel(n_jobs=8)(delayed(plotGammaBar)(group, name, save_dir, yax) for name, group in gamma_df.groupby(['taste']))


def plot_NB_decoding(df, plotdir=None, trial_group_size=5, trial_col='trial_num'):
    # df[['Y','epoch']] = df.Y.str.split('_', expand = True)
    df = df.copy()
    df['trial-group'] = df[trial_col] / trial_group_size
    df['trial-group'] = df['trial-group'].astype(int)
    df['tg_start'] = df.groupby(['trial-group'])[trial_col].transform('min') + 1
    df['tg_end'] = df.groupby(['trial-group'])[trial_col].transform('max') + 1
    df['trials'] = df.tg_start.astype(str) + '-' + df.tg_end.astype(str)

    df = df.loc[df.single_state == False]
    df = df.loc[df.prestim_state == False]

    # df['p_taste'] = df.CA + df.CA + df.QHCl+ df.Suc
    df = df.loc[df.prestim_state != True]
    df = df.rename(
        columns={'p_correct': 'p(correct)', 'time_group': 'session', 'exp_group': 'condition'})

    for i in ['early', 'late']:
        sp = 'NB_' + i
        sub_df = df.loc[df.epoch == i].copy()
        plot_trialwise_lm(sub_df, x_col=trial_col, y_facs=['p(correct)'], save_dir=plotdir, save_prefix=sp)
        plot_trialwise_rel(sub_df, 'trial-group', ['p(correct)'], save_dir=plotdir, save_prefix=sp, trial_col=trial_col)
        plot_daywise_data(sub_df, ['p(correct)'], save_dir=plotdir, save_prefix=sp)


def plot_NB_timing(df, plotdir=None, trial_group_size=5, trial_col='trial_num'):
    # df['trial-group'] = df.trial_num/trial_group_size

    df['trial-group'] = df[trial_col] / trial_group_size
    df['trial-group'] = df['trial-group'].astype(int)
    df['time_group'] = df['time_group'].astype(int)
    df['tg_start'] = df.groupby(['trial-group'])[trial_col].transform('min') + 1
    df['tg_end'] = df.groupby(['trial-group'])[trial_col].transform('max') + 1
    df['trials'] = df.tg_start.astype(str) + '-' + df.tg_end.astype(str)

    df = df.loc[df.single_state == False]
    df = df.loc[df.state_num != 0]
    df = df.loc[df.p_correct > 0.5]
    df = df.rename(columns={'time_group': 'session', 'exp_group': 'condition', 't_start': 't(start)', 't_end': 't(end)',
                            't_med': 't(median)', 'err': 'error', 'Y_pred': 'predicted-taste'})
    y_facs = list(df.loc[:, 't(start)':'duration_zscore'].columns)
    # ['duration','t(start)', 't(end)', 't(median)']

    for i in ['early', 'late']:
        sp = 'NB_' + i + '_'
        sub_df = df.loc[df.epoch == i]
        # plot_trialwise_lm(sub_df, x_col = trial_col, y_facs = y_facs, save_dir = plotdir, save_prefix = sp)
        plot_trialwise_rel(sub_df, x_col='trial-group', y_facs=y_facs, save_dir=plotdir, save_prefix=sp,
                           trial_col=trial_col)
        # plot_daywise_data(sub_df, y_facs, save_dir = plotdir, save_prefix = 'NB')


def plot_LR(df, plotdir=None, trial_group_size=5):
    df['trial-group'] = df.trial_num / trial_group_size
    df['trial-group'] = df['trial-group'].astype(int)
    df = df.loc[df.single_state == False]
    df = df.loc[df.state_group != 'Prestim']

    df = df.rename(
        columns={'trial_num': 'trial', 'time_group': 'session', 'exp_group': 'condition', 't_start': 't(start)',
                 't_end': 't(end)', 't_med': 't(median)', 'err': 'error', 'Y_pred': 'pred-pal'})
    y_facs = ['duration', 'error', 'SquaredError', 'pred-pal']

    #plot_trialwise_lm(df, x_col='trial', y_facs=y_facs, save_dir=plotdir, save_prefix='LR')
    plot_trialwise_rel(df, x_col='trial-group', y_facs=y_facs, save_dir=plotdir, save_prefix='LR')
    plot_daywise_data(df, y_facs, save_dir=plotdir, save_prefix='LR')


def plot_trialwise_rel(df, x_col, y_facs, sess_col='time_group', cond_col='exp_group', save_dir=None, save_prefix=None, trial_col='trial'):
    df = df.copy()
    df['tg_start'] = df.groupby([x_col])[trial_col].transform('min') + 1
    df['tg_end'] = df.groupby([x_col])[trial_col].transform('max') + 1
    df['trials'] = df.tg_start.astype(str) + '-' + df.tg_end.astype(str)
    df['session'] = df[sess_col]
    df['condition'] = df[cond_col]

    df = df.copy()
    sns.set_theme(style='ticks', font_scale=1.5)
    cols = [x_col, 'trials']
    triallabs = df[cols].reset_index(drop=True).drop_duplicates().sort_values(by=[x_col])

    for y_col in y_facs:
        df['taste'] = pd.Categorical(df['taste'], ['Suc', 'NaCl', 'CA', 'QHCl'])
        # plot with raste on each row
        xline_dat = df.groupby(['taste', 'session', 'condition'])[y_col].agg('mean').reset_index()
        g = sns.relplot(kind='line', data=df,
                        x=x_col, y=y_col, row='taste', col='session', hue='condition', style='condition',
                        markers=True, err_style='band', ci=99.72, height=4, aspect=1,
                        linewidth=3,
                        facet_kws={"margin_titles": True})
        g.set_titles(row_template='{row_name}')

        n_groups = int(max(df[x_col]))
        xt = np.arange(0, n_groups + 1)
        xend = n_groups + 0.25
        g.set(xlim=(-0.25, xend), xticks=xt)  # , xlabel=trial_col)
        g.set_xticklabels(triallabs['trials'], rotation=45)
        if y_col == 'p(correct)':
            g.set(ylim=(-0.1, 1.1), yticks=[0, 0.25, 0.5, 0.75, 1])

        axes = g.axes.flatten()
        counter = 0
        for i, ax in enumerate(axes):
            print(i)
            ax.axhline(xline_dat[y_col].iloc[counter])
            counter = counter + 1
            ax.axhline(xline_dat[y_col].iloc[counter], c='orange')
            counter = counter + 1

        if save_dir:
            nm1 = save_prefix + '_' + trial_col + '_VS_' + y_col + '_Rel.svg'
            sf = os.path.join(save_dir, nm1)
            g.savefig(sf)
            print(sf)
        plt.close('all')

        # plot with tastes aggregated into one row
        df2 = df.copy()
        df2['taste'] = 'All Tastes'
        h = sns.relplot(kind='line', data=df2,
                        x=x_col, y=y_col, col='session', row='taste', hue='condition', style='condition',
                        markers=True, err_style='band', ci=99.72,
                        linewidth=3, height=5, aspect=.8,
                        facet_kws={"margin_titles": True})
        h.set_titles(row_template='{row_name}')
        h.set(xlim=(-0.25, xend), xticks=xt)  # , xlabel=trial_col)
        h.set_xticklabels(triallabs['trials'], rotation=45)

        if y_col == 'p(correct)':
            h.set(ylim=(-0.1, 1.1), yticks=[0, 0.25, 0.5, 0.75, 1])

        xline_dat = df2.groupby(['session', 'condition'])[y_col].agg('mean').reset_index()
        axes = h.axes.flatten()
        counter = 0
        for i, ax in enumerate(axes):
            print(i)
            ax.axhline(xline_dat[y_col].iloc[counter])
            counter = counter + 1
            ax.axhline(xline_dat[y_col].iloc[counter], c='orange')
            counter = counter + 1

        if save_dir:
            nm2 = save_prefix + '_' + trial_col + '_VS_' + y_col + '_all_tsts_Rel.svg'
            sf2 = os.path.join(save_dir, nm2)
            h.savefig(sf2)
            print(sf2)
        plt.close('all')

def plot_trialwise_rel2(df, y_facs, sess_col='time_group', cond_col='exp_group', save_dir=None, save_prefix=None, trial_col='trial', n_trial_groups = None):
    df = df.copy()
    if n_trial_groups is None:
        if min(df[trial_col]) == 0:
            n_trial_groups = max(df[trial_col]) + 1
        else:
            n_trial_groups = max(df[trial_col])

    df['trial group'] = pd.cut(df[trial_col], n_trial_groups, labels=False).astype('int') #create x axis index
    df['tg_start'] = df.groupby(['trial group'])[trial_col].transform('min') + 1 #create label of trial # starting each trial group
    df['tg_end'] = df.groupby(['trial group'])[trial_col].transform('max') + 1 #create label of trial # ending each trial group
    df['trials'] = df.tg_start.astype(str) + '-' + df.tg_end.astype(str) #create label of trial # range for each trial group
    df['session'] = df[sess_col]
    df['condition'] = df[cond_col]

    df = df.copy()
    sns.set_theme(style='ticks', font_scale=1.5)
    cols = ['trial group', 'trials']
    triallabs = df[cols].reset_index(drop=True).drop_duplicates().sort_values(by=['trial group'])

    for y_col in y_facs:
        df['taste'] = pd.Categorical(df['taste'], ['Suc', 'NaCl', 'CA', 'QHCl'])
        g = sns.relplot(kind='line', data=df,
                        x='trial group', y=y_col, col='session', row='taste', hue='condition', style='condition',
                        markers=True, err_style='band', ci=95, height=4, aspect=1,
                        linewidth=3,
                        facet_kws={"margin_titles": True})
        g.set_titles(row_template='{row_name}')

        n_groups = int(max(df['trial group']))
        xt = np.arange(0, n_groups + 1)
        xend = n_groups + 0.25
        g.set(xlim=(-0.25, xend), xticks=xt)  # , xlabel=trial_col)
        g.set_xticklabels(triallabs['trials'], rotation=45)
        if y_col == 'p(correct)' or y_col == 'pr(mode state)':
            g.set(ylim=(-0.1, 1.1), yticks=[0, 0.25, 0.5, 0.75, 1])

        if save_dir:
            nm1 = save_prefix + '_' + str(n_trial_groups) + '_' + trial_col + '_VS_' + y_col + '_Rel.png'
            sf = os.path.join(save_dir, nm1)
            g.savefig(sf)
            print(sf)
        plt.close('all')

        # plot with tastes aggregated into one row
        df2 = df.copy()
        df2['taste'] = 'All Tastes'
        h = sns.relplot(kind='line', data=df2,
                        x='trial group', y=y_col, row='taste', col='session', hue='condition', style='condition',
                        markers=True, err_style='band', ci=95,
                        linewidth=3, height=5, aspect=0.8,
                        facet_kws={"margin_titles": True})
        h.set_titles(row_template='{row_name}')
        h.set(xlim=(-0.25, xend), xticks=xt)  # , xlabel=trial_col)
        h.set_xticklabels(triallabs['trials'], rotation=45)

        if y_col == 'p(correct)' or y_col == 'pr(mode state)':
            h.set(ylim=(-0.1, 1.1), yticks=[0, 0.25, 0.5, 0.75, 1])

        if save_dir:
            nm2 = save_prefix + '_' + str(n_trial_groups) + '_' + trial_col + '_VS_' + y_col + '_all_tsts_Rel.png'
            sf2 = os.path.join(save_dir, nm2)
            h.savefig(sf2)
            print(sf2)
        plt.close('all')


def plot_trialwise_lm(df, x_col, y_facs, hue='condition', col='session', row='taste', save_dir=None, save_prefix=None):
    row_order = ['Suc', 'NaCl', 'CA', 'QHCl']
    df['all'] = 'all tastes'
    for ycol in y_facs:

        g = sns.lmplot(data=df, x=x_col, y=ycol, hue=hue, col=col, row=row, row_order=row_order, aspect=1, height=4,
                       facet_kws={"margin_titles": True})
        g.set_titles(row_template='{row_name}')
        g.set(ylim=(0, 3000))
        h = sns.lmplot(data=df, x=x_col, y=ycol, hue=hue, col=col, row='all', aspect=0.75, height=6,
                       facet_kws={"margin_titles": True}, scatter_kws={"alpha": 0.25})
        h.set_titles(row_template='{row_name}')
        h.set(ylim=(0, 3000))

        if x_col == 'trial-group':
            g.set(xlim=(-0.25, 5.25), xticks=[0, 1, 2, 3, 4, 5])
            g.set_xticklabels(['1-5', '6-10', '11-15', '16-20', '21-25', '26-30'], rotation=60)
            h.set(xlim=(-0.25, 5.25), xticks=[0, 1, 2, 3, 4, 5])
            h.set_xticklabels(['1-5', '6-10', '11-15', '16-20', '21-25', '26-30'], rotation=60)

        if ycol == 'p(correct)':
            g.set(ylim=(-0.1, 1.1), yticks=[0, 0.25, 0.5, 0.75, 1])
            h.set(ylim=(-0.1, 1.1), yticks=[0, 0.25, 0.5, 0.75, 1])

        if ycol == 'SquaredError':
            g.set(ylim=(-0.1, 3))
            h.set(ylim=(-0.1, 3))

        if ycol == 'error':
            g.set(ylim=(-2, 2))
            h.set(ylim=(-2, 2))

        if save_dir is not None:
            nm = save_prefix + x_col + '_VS_' + ycol + '_LMP.png'
            sf = os.path.join(save_dir, nm)
            g.savefig(sf)
            print(sf)
            nm2 = save_prefix + x_col + '_VS_' + ycol + '_alltsts_LMP.png'
            sf2 = os.path.join(save_dir, nm2)
            h.savefig(sf2)
            print(sf2)
        plt.close('all')


# def plotRsquared(df,y,row = None,save_dir = None,save_prefix = None):
#     g = sns.catplot(kind = 'bar', data = df,
#                     x = 'session', y = 'Correlation', hue = 'condition', row = row,
#                     margin_titles=True, aspect = 0.25, height = 20, capsize = 0.2, errwidth = 1, row_order = ['Suc','NaCl','CA','QHCl'])

#     g.tight_layout()
#     g.set_titles(row_template = '{row_name}')
#     #g.fig.suptitle(y)
#     if save_dir:
#         nm1 = save_prefix + y+ '.svg'
#         sf = os.path.join(save_dir, nm1)
#         g.savefig(sf)
#         print(sf)

def plotRsquared(df, yfacs, row=None, save_dir=None, save_prefix=None):
    df = df.rename(columns={'time_group': 'session', 'exp_group': 'condition'})
    df['session'] = df.session.astype(int)
    df = df.loc[df.state_group != 'prestim']
    df = df.sort_values(by=['condition'], ascending=True)
    for i in yfacs:
        data = df.loc[df.Feature == i]

        g = sns.catplot(kind='bar', data=data,
                        x='session', y='Correlation', hue='condition', row=row,
                        margin_titles=True, aspect=0.25, height=20, capsize=0.2, errwidth=1,
                        row_order=['Suc', 'NaCl', 'CA', 'QHCl'])

        g.tight_layout()
        g.set_titles(row_template='{row_name}')
        g.fig.suptitle(i)
        if save_dir:
            nm1 = save_prefix + i + '.svg'
            sf = os.path.join(save_dir, nm1)
            g.savefig(sf)
            print(sf)
    plt.close('all')


def plot_daywise_data(df, yfacs, save_dir=None, save_prefix=None):
    for ycol in yfacs:
        g = sns.catplot(kind='bar', data=df,
                        x='session', y=ycol, hue='condition', row='taste',
                        ci=95, margin_titles=True, aspect=0.25, height=20, capsize=0.2, errwidth=1)
        g.tight_layout()
        g.set_titles(row_template='{row_name}')
        # sns.move_legend(g, "lower center", bbox_to_anchor=(.4, 1), ncol=2, title=None, frameon=False)

        h = sns.catplot(kind='bar', data=df,
                        x='session', y=ycol, hue='condition',
                        ci=95, margin_titles=True, aspect=1, height=10, capsize=0.2, errwidth=1)
        h.tight_layout()
        h.set_titles(row_template='{row_name}')
        # sns.move_legend(h, "lower center", bbox_to_anchor=(.4, 1), ncol=2, title=None, frameon=False)

        if ycol == 'p(correct)':
            g.set(ylim=(-0.1, 1.1), yticks=[0, 0.25, 0.5, 0.75, 1])
            h.set(ylim=(-0.1, 1.1), yticks=[0, 0.25, 0.5, 0.75, 1])

        if save_dir:
            nm1 = save_prefix + ycol + '.svg'
            sf = os.path.join(save_dir, nm1)
            g.savefig(sf)
            print(sf)
            nm2 = save_prefix + ycol + '_all_tsts.svg'
            sf2 = os.path.join(save_dir, nm2)
            h.savefig(sf2)
            plt.close('all')
            print(sf2)
        else:
            return g, h


def plot_pal_data(df, save_dir=None):
    df = df.loc[df.single_state == False]
    df = df.loc[df.state_group != 'Prestim']
    df = df.rename(columns={'palatability': 'pal-rating', 'Y_pred': 'predicted-pal', 'exp_group': 'condition',
                            'time_group': 'session', 'trial_num': 'trial'})

    g = sns.lmplot(data=df, x='pal-rating', y='predicted-pal', row='condition',
                   col='session', hue='condition', legend=False, aspect=1, height=6,
                   facet_kws={"margin_titles": True})
    g.set_titles(row_template='{row_name}')
    g.set(ylim=(-1, 6))

    if save_dir:
        nm1 = 'pal_xyplot.svg'
        sf = os.path.join(save_dir, nm1)
        g.savefig(sf)
        print(sf)


def plot_confusion_differences(df, save_file=None):
    pal_df = stats.get_diff_df(df, ['exp_group', 'state_group'],
                               'time_group', 'pal_confusion')
    id_df = stats.get_diff_df(df, ['exp_group', 'state_group'],
                              'time_group', 'ID_confusion')
    pal_df['grouping'] = pal_df.apply(lambda x: '%s\n%s' % (x['exp_group']), axis=1)
    id_df['grouping'] = id_df.apply(lambda x: '%s\n%s' % (x['exp_group']), axis=1)
    o1 = ORDERS['exp_group']
    # o2 = ORDERS['cta_group']
    o2 = ORDERS['state_group']
    # x_order = ['GFP\nCTA', 'Cre\nNo CTA', 'GFP\nNo CTA']
    x_order = ['Exposure_1', 'Exposure_3']
    cond_order = list(it.product(x_order, o2))

    fig, axes = plt.subplots(ncols=2, figsize=(15, 7), sharey=False)
    sns.barplot(data=id_df, ax=axes[0], x='grouping', y='mean_diff',
                hue='state_group', order=x_order, hue_order=o2)
    sns.barplot(data=pal_df, ax=axes[1], x='grouping', y='mean_diff',
                hue='state_group', order=x_order, hue_order=o2)
    xdata = [x.get_x() + x.get_width() / 2 for x in axes[0].patches]
    xdata.sort()
    tmp_pal = pal_df.set_index(['grouping', 'state_group'])[['mean_diff', 'sem_diff']].to_dict()
    tmp_id = id_df.set_index(['grouping', 'state_group'])[['mean_diff', 'sem_diff']].to_dict()
    for x, grp in zip(xdata, cond_order):
        ym = tmp_id['mean_diff'][grp]
        yd = tmp_id['sem_diff'][grp]
        axes[0].plot([x, x], [ym - yd, ym + yd], color='k', linewidth=3)
        ym = tmp_pal['mean_diff'][grp]
        yd = tmp_pal['sem_diff'][grp]
        axes[1].plot([x, x], [ym - yd, ym + yd], color='k', linewidth=3)

    for ax in axes:
        ymax = np.max(np.abs(ax.get_ylim()))
        ax.set_ylim([-ymax, ymax])
        ax.set_xlabel('')
        ax.set_ylabel('')
        # ax.axhline(0, linestyle='--', linewidth=1, alpha=0.6, color='k')
        ax.grid(True, axis='y', linestyle=':')
        if ax.is_first_col():
            ax.set_ylabel(r'$\Delta$ % classified as NaCl')
            ax.get_legend().remove()
        else:
            ax.get_legend().set_title('HMM State')

    axes[0].set_title('ID Confusion')
    axes[1].set_title('Pal Confusion')
    fig.subplots_adjust(top=0.85)
    fig.suptitle('Change in saccharin classification over learning')
    if save_file:
        fig.savefig(save_file)
        plt.close(fig)
        return
    else:
        return fig, axes


def plot_confusion_correlations(df, save_file=None):
    """look at ID and pal confusion vs n_cells and # of trials or each tastant"""
    data_cols = ['ID_confusion', 'pal_confusion']
    comparison_vars = ['n_cells', 'nacl_trials', 'ca_trials',
                       'quinine_trials', 'sacc_trials']

    # Actually just convert string cols to numbers and look at correlation matrix
    convert_vars = ['exp_name', 'exp_group', 'time_group', 'state_group']
    df2 = df.copy()
    for col in convert_vars:
        grps = df[col].unique()
        mapping = {x: i for i, x in enumerate(grps)}
        df2[col] = df[col].map(mapping)

    df2 = df2[[*convert_vars, *data_cols, *comparison_vars]]

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    cbar_ax = fig.add_axes([.9, 0.1, .05, .7])

    cm = df2.rcorr(method='spearman', stars=False, padjust='bonf', decimals=15)
    cm[cm == '-'] = 1
    cm = cm.astype('float')
    m1 = np.triu(cm)
    m2 = np.tril(cm)
    g = sns.heatmap(cm, annot=True, fmt='.2g', vmin=-1, vmax=1, center=0,
                    square=True, cmap='coolwarm', ax=ax, cbar_ax=cbar_ax,
                    mask=m1)
    sns.heatmap(cm, annot=True, fmt='.1g', mask=m2, cbar=False, square=True, ax=ax)
    ax.text(0.05, .9, 'corr')
    ax.text(0.5, 0.25, 'p')
    ax.plot([0, 1], [0, 1], color='k', linewidth=2)
    statistics = df2.pairwise_corr(padjust='bonf', method='spearman')
    # g = sns.heatmap(df2.corr(method='spearman'), annot=True, vmin=-1, vmax=1, center=0,
    #                 square=True, cmap='coolwarm', ax=ax, cbar_ax=cbar_ax)
    fig.set_size_inches(12, 8)
    g.set_title('Confusion Correlation Matrix', pad=20)
    cbar_ax.set_position([0.75, 0.20, 0.04, .71])
    plt.tight_layout()
    if save_file is None:
        return fig, ax
    else:
        fig.savefig(save_file)
        plt.close(fig)
        fn, ext = os.path.splitext(save_file)
        fn += '.txt'
        agg.write_dict_to_txt({'Confusion Correlation Statistics': statistics}, fn)


def plot_coding_correlations(df, save_file=None):
    """look at ID and pal confusion vs n_cells and # of trials or each tastant"""
    df = fix_coding_df(df)
    data_cols = ['id_acc', 'pal_acc']
    comparison_vars = ['n_cells', 'n_held_cells']

    # Actually just convert string cols to numbers and look at correlation matrix
    convert_vars = ['exp_name', 'exp_group', 'time_group',
                    'state_group']
    df2 = df.copy()
    for col in convert_vars:
        grps = df[col].unique()
        mapping = {x: i for i, x in enumerate(grps)}
        df2[col] = df[col].map(mapping)

    df2 = df2[[*convert_vars, *data_cols, *comparison_vars]]

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    cbar_ax = fig.add_axes([.9, 0.1, .05, .7])

    cm = df2.rcorr(method='spearman', stars=False, padjust='bonf', decimals=15)
    cm[cm == '-'] = 1
    cm = cm.astype('float')
    m1 = np.triu(cm)
    m2 = np.tril(cm)
    g = sns.heatmap(cm, annot=True, fmt='.2g', vmin=-1, vmax=1, center=0,
                    square=True, cmap='coolwarm', ax=ax, cbar_ax=cbar_ax,
                    mask=m1)
    sns.heatmap(cm, annot=True, fmt='.1g', mask=m2, cbar=False, square=True, ax=ax)
    ax.text(0.05, .9, 'corr')
    ax.text(0.5, 0.25, 'p')
    ax.plot([0, 1], [0, 1], color='k', linewidth=2)
    statistics = df2.pairwise_corr(padjust='bonf', method='spearman')
    # g = sns.heatmap(df2.corr(method='spearman'), annot=True, vmin=-1, vmax=1, center=0,
    #                 square=True, cmap='coolwarm', ax=ax, cbar_ax=cbar_ax)
    fig.set_size_inches(12, 8)
    g.set_title('Coding Correlation Matrix', pad=20)
    cbar_ax.set_position([0.75, 0.20, 0.04, .71])
    plt.tight_layout()
    if save_file is None:
        return fig, ax
    else:
        fig.savefig(save_file)
        plt.close(fig)
        fn, ext = os.path.splitext(save_file)
        fn += '.txt'
        agg.write_dict_to_txt({'Coding Correlation Statistics': statistics}, fn)


def plot_timing_correlations(df, save_file=None):
    df = df.loc[df.duration > 5]
    df = df[['exp_name', 'time_group', 'exp_group', 'taste', 'trial', 't_start', 't_end', 't_med', 'duration']]
    data_cols = ['t_start', 't_end', 'duration']
    convert_vars = ['exp_name', 'exp_group', 'time_group', 'taste']
    comparison_vars = ['palatability', 'n_cells']

    df2 = df.copy()
    for col in convert_vars:
        grps = df[col].unique()
        mapping = {x: i for i, x in enumerate(grps)}
        df2[col] = df[col].map(mapping)

    df2 = df2[[*convert_vars, *data_cols, *comparison_vars]].dropna()

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    cbar_ax = fig.add_axes([.9, 0.1, .05, .7])

    cm = df2.rcorr(method='spearman', stars=False, padjust='bonf', decimals=15)
    cm[cm == '-'] = 1
    cm = cm.astype('float')
    m1 = np.triu(cm)
    m2 = np.tril(cm)
    g = sns.heatmap(cm, annot=True, fmt='.2g', vmin=-1, vmax=1, center=0,
                    square=True, cmap='coolwarm', ax=ax, cbar_ax=cbar_ax,
                    mask=m1)
    sns.heatmap(cm, annot=True, fmt='.1g', mask=m2, cbar=False, square=True, ax=ax)
    ax.text(0.05, .9, 'corr')
    ax.text(0.5, 0.25, 'p')
    ax.plot([0, 1], [0, 1], color='k', linewidth=2)
    statistics = df2.pairwise_corr(padjust='bonf', method='spearman')
    # g = sns.heatmap(df2.corr(method='spearman'), annot=True, vmin=-1, vmax=1, center=0,
    #                 square=True, cmap='coolwarm', ax=ax, cbar_ax=cbar_ax)
    fig.set_size_inches(12, 8)
    g.set_title('Timing Correlation Matrix', pad=20)
    cbar_ax.set_position([0.75, 0.20, 0.04, .71])
    plt.tight_layout()
    if save_file is None:
        return fig, ax
    else:
        fig.savefig(save_file)
        plt.close(fig)
        fn, ext = os.path.splitext(save_file)
        fn += '.txt'
        agg.write_dict_to_txt({'Timing Correlation Statistics': statistics}, fn)


def plot_trialwise_timing_deprecated(df, save_prefix=None):
    df = df.loc[df.state_group == 'ID']
    df = df.loc[df.p_ID > 0.5]
    df = df.loc[df.t_start > -200]

    yfacs = ['duration', 't_start', 't_end', 't_med']
    jfacs = ['Suc', 'NaCl', 'CA', 'QHCl']
    for i in yfacs:
        for j in jfacs:

            sf = save_prefix + 'trlnoVS' + i + '_' + j + '.png'

            tastesub = df.loc[df.taste == j]
            g = sns.FacetGrid(tastesub, col='time_group', row='exp_group', hue='exp_name', margin_titles=True, aspect=1,
                              height=5)
            g.map(sns.scatterplot, 'trial', i, s=50)
            g.fig.subplots_adjust(top=0.9)
            g.fig.suptitle('trial number vs ID ' + i + ': ' + j)
            g.add_legend()

            if save_prefix is not None:
                g.savefig(sf)

            plt.close('all')


def plot_confusion_data(df, save_file=None, group_col='exp_group', kind='bar',
                        plot_points=False):
    df = df.copy()
    # Make extra column for composite grouping
    df['grouping'] = df.apply(lambda x: '%s_%s' % (x[group_col], x['time_group']), axis=1)

    states = df['state_group'].unique()
    groups = df[group_col].unique()
    hues = df['time_group'].unique()
    group_order = ORDERS[group_col]
    hue_order = ORDERS['time_group']
    state_order = ORDERS['state_group']
    cond_order = []
    for g, h in it.product(group_order, hue_order):
        cond_order.append('%s_%s' % (g, h))

    fig = plt.figure(figsize=(14, 10))
    outer_ax = add_suplabels(fig, 'Bootstrapped Confusion Analysis', '', '% '
                                                                         'Sacc trials classified as NaCl')
    nrows = len(states)
    ncols = 3
    axes = np.array([[fig.add_subplot(nrows, ncols, j + ncols * i + 1)
                      for j in range(ncols)] for i in range(nrows)])
    axes[0, 0].get_shared_y_axes().join(axes[0, 0], axes[1, 0])
    axes[0, 1].get_shared_y_axes().join(axes[0, 1], axes[1, 1])
    axes[0, 2].get_shared_y_axes().join(axes[0, 2], axes[1, 2])
    statistics = {}
    for sg, (id_ax, pal_ax, psc_ax) in zip(state_order, axes):
        grp = df.query('state_group == @sg')
        id_kw_stat, id_kw_p, id_gh_df = stats.kw_and_gh(grp, 'grouping',
                                                        'ID_confusion')
        pal_kw_stat, pal_kw_p, pal_gh_df = stats.kw_and_gh(grp, 'grouping',
                                                           'pal_confusion')
        psc_kw_stat, psc_kw_p, psc_gh_df = stats.kw_and_gh(grp, 'grouping',
                                                           'pal_confusion_score')
        id_sum = grp.groupby(['exp_group', 'time_group'])['ID_confusion'].describe()
        pal_sum = grp.groupby(['exp_group', 'time_group'])['pal_confusion'].describe()
        psc_sum = grp.groupby(['exp_group', 'time_group'])['pal_confusion_score'].describe()
        statistics[sg] = {'id': {'kw_stat': id_kw_stat, 'kw_p': id_kw_p,
                                 'posthoc': id_gh_df, 'summary': id_sum},
                          'pal': {'kw_stat': pal_kw_stat, 'kw_p': pal_kw_p,
                                  'posthoc': pal_gh_df, 'summary': pal_sum},
                          'pal_score': {'kw_stat': psc_kw_stat, 'kw_p': psc_kw_p,
                                        'posthoc': psc_gh_df, 'summary': psc_sum}}

        g1 = plot_box_and_paired_points(grp, group_col, 'ID_confusion',
                                        'time_group', order=group_order,
                                        hue_order=hue_order,
                                        subjects='exp_name',
                                        ax=id_ax, kind=kind,
                                        plot_points=plot_points)

        g2 = plot_box_and_paired_points(grp, group_col, 'pal_confusion',
                                        'time_group', order=group_order,
                                        hue_order=hue_order,
                                        subjects='exp_name',
                                        ax=pal_ax, kind=kind,
                                        plot_points=plot_points)

        g3 = plot_box_and_paired_points(grp, group_col, 'pal_confusion_score',
                                        'time_group', order=group_order,
                                        hue_order=hue_order,
                                        subjects='exp_name',
                                        ax=psc_ax, kind=kind,
                                        plot_points=plot_points)

        # g1.axhline(50, linestyle='--', alpha=0.5, color='k')
        # g2.axhline(33.3, linestyle='--', alpha=0.5, color='k')

        g1.set_ylabel(sg.replace('_', ' '))
        g1.legend_.remove()
        g1.set_xlabel('')
        g2.set_xlabel('')
        g2.set_ylabel('')
        g2.legend_.remove()
        g3.set_xlabel('')
        g3.set_ylabel('')
        if id_ax.is_first_row():
            g1.set_title('ID Confusion')
            g2.set_title('Pal Confusion')
            g3.set_title('Pal Confusion Score')

        if not pal_ax.is_last_row():
            g3.legend_.remove()
        else:
            g3.legend(bbox_to_anchor=[1.2, 1.2, 0, 0])

        # n_cells = grp.groupby('grouping').size().to_dict()
        n_cells = None
        g1_y = plot_sig_stars(g1, id_gh_df, cond_order, n_cells=n_cells)
        g2_y = plot_sig_stars(g2, pal_gh_df, cond_order, n_cells=n_cells)
        g3_y = plot_sig_stars(g3, psc_gh_df, cond_order, n_cells=n_cells)

    if save_file:
        fn, ext = os.path.splitext(save_file)
        fn2 = fn + '.txt'
        fig.savefig(save_file)
        plt.close(fig)
        agg.write_dict_to_txt(statistics, save_file=fn2)
    else:
        return fig, axes, statistics


def plot_timing_data(df, save_file=None, group_col='exp_group', kind='bar', plot_points=False):
    df = df[df['valid']]
    # assert len(df.taste.unique()) == 1, 'Please run one taste at a time'
    df = df.copy().dropna(subset=[group_col, 'state_group', 'time_group',
                                  't_start', 't_end'])
    df = df.query('valid == True')

    # Make extra column for composite grouping
    # df['grouping'] = df.apply(lambda x: '%s_%s' % (x[group_col], x['time_group']), axis=1)
    df['grouping'] = df['exp_group']
    df = df.copy().dropna(subset=['grouping'])
    if len(df.taste.unique()) == 1:
        taste = df['taste'].unique()[0]
    else:
        taste = "All Tastes"
    # 031320: found that state group has a lot of 0 and 1 states marked as
    # early and late check analyze_hmm_state_timing in hmma
    states = df['state_group'].unique()
    groups = df[group_col].unique()
    hues = df['time_group'].unique()
    group_order = ORDERS[group_col]
    hue_order = ORDERS['time_group']
    state_order = ORDERS['state_group']
    cond_order = []
    # for g,h in it.product(group_order, hue_order):
    #     cond_order.append('%s_%s' % (g,h))
    cond_order = ['naive', 'suc_preexp']

    fig = plt.figure(figsize=(11, 7))
    outer_ax = add_suplabels(fig, f'{taste} HMM Timing Analysis', '',
                             'transition time (ms)')
    plot_grps = [('early', 't_end'), ('late', 't_start')]
    titles = {'t_end': 'End Times', 't_start': 'Start Times'}
    axes = np.array([fig.add_subplot(1, len(plot_grps), i + 1)
                     for i in range(len(plot_grps))])
    statistics = {}
    for (sg, vg), ax in zip(plot_grps, axes):
        grp = df.query('state_group == @sg')
        kw_stat, kw_p, gh_df = stats.kw_and_gh(grp, 'grouping', vg)
        summary = grp.groupby(['exp_group', 'time_group'])[vg].describe()
        statistics[sg] = {titles[vg]: {'kw_stat': kw_stat, 'kw_p': kw_p,
                                       'posthoc': gh_df, 'summary': summary}}

        g1 = plot_box_and_paired_points(grp, group_col, vg,
                                        'time_group', order=group_order,
                                        hue_order=hue_order,
                                        subjects='exp_name',
                                        ax=ax, kind=kind,
                                        plot_points=plot_points)

        g1.set_ylabel('')
        g1.set_xlabel('')
        g1.set_title('%s %s' % (sg, titles[vg]))

        if ax.is_last_col():
            g1.legend(bbox_to_anchor=[0.9, 1, 0, 0])
        else:
            g1.legend_.remove()

        n_cells = grp.groupby('grouping').size().to_dict()
        g1_y = plot_sig_stars(g1, gh_df, cond_order, n_cells=n_cells)

    if save_file:
        fn, ext = os.path.splitext(save_file)
        fn2 = fn + '.txt'
        fig.savefig(save_file)
        plt.close(fig)
        agg.write_dict_to_txt(statistics, save_file=fn2)
    else:
        return fig, axes, statistics


# fix this entirely so we get day data as well
def plot_intraday_timing(df, save_file=None, group_col='time_group', kind='bar', plot_points=False):
    df = df[df['valid']]
    # assert len(df.taste.unique()) == 1, 'Please run one taste at a time'
    df = df.copy().dropna(subset=[group_col, 'state_group', 'time_group',
                                  't_start', 't_end'])
    df = df.query('valid == True')

    # Make extra column for composite grouping
    df['grouping'] = df.apply(lambda x: '%s_%s' % (x['exp_group'], x[group_col]), axis=1)
    # df['grouping'] = df.apply(lambda x: '%s' % (x['exp_group']), axis=1)
    df = df.copy().dropna(subset=['grouping'])
    if len(df.taste.unique()) == 1:
        taste = df['taste'].unique()[0]
    else:
        taste = "All Tastes"
    # 031320: found that state group has a lot of 0 and 1 states marked as
    # early and late check analyze_hmm_state_timing in hmma
    states = df['state_group'].unique()
    groups = df[group_col].unique()
    hues = df['trial_group'].unique()
    group_order = ORDERS['exp_group']
    hue_order = ORDERS['trial_group']
    state_order = ORDERS['state_group']
    cond_order = []
    for g, h in it.product(group_order, hue_order):
        cond_order.append('%s_%s' % (g, h))
    # cond_order = ['Naive','Preexp']
    fig = plt.figure(figsize=(11, 7))
    outer_ax = add_suplabels(fig, f'{taste} HMM Timing Analysis', '',
                             'transition time (ms)')
    plot_grps = [('early', 't_end'), ('late', 't_start')]
    titles = {'t_end': 'End Times', 't_start': 'Start Times'}
    axes = np.array([fig.add_subplot(1, len(plot_grps), i + 1)
                     for i in range(len(plot_grps))])
    statistics = {}
    for (sg, vg), ax in zip(plot_grps, axes):
        print(sg, "_", vg)
        grp = df.query('state_group == @sg')
        print(grp)
        kw_stat, kw_p, gh_df = stats.kw_and_gh(grp, 'grouping', vg)
        summary = grp.groupby(['time_group', 'exp_group'])[vg].describe()
        # summary = grp.groupby(['', group_col])[vg].describe()
        statistics[sg] = {titles[vg]: {'kw_stat': kw_stat, 'kw_p': kw_p,
                                       'posthoc': gh_df, 'summary': summary}}

        g1 = plot_box_and_paired_points(grp, 'exp_group', vg,
                                        group_col, order=group_order,
                                        hue_order=hue_order,
                                        subjects='exp_name',
                                        ax=ax, kind=kind,
                                        plot_points=plot_points)

        g1.set_ylabel('')
        g1.set_xlabel('')
        g1.set_title('%s %s' % (sg, titles[vg]))

        if ax.is_last_col():
            g1.legend(bbox_to_anchor=[0.9, 1, 0, 0])
        else:
            g1.legend_.remove()

        n_cells = grp.groupby('grouping').size().to_dict()
        g1_y = plot_sig_stars(g1, gh_df, cond_order, n_cells=n_cells)

    if save_file:
        fn, ext = os.path.splitext(save_file)
        fn2 = fn + '.txt'
        fig.savefig(save_file)
        plt.close(fig)
        agg.write_dict_to_txt(statistics, save_file=fn2)
    else:
        return fig, axes, statistics


# TODO plot state DENSITY distributions
def plot_timing_distributions(df, state='early', value_col='t_end', save_file=None):
    df = df[df['valid']]
    df = df.query('state_group == @state').copy()
    # df['grouping'] = df.apply(lambda x: '%s_%s' % (x['exp_group'],x['time_group']), axis=1) # %s_%s --means query string_string
    df['grouping'] = df['exp_group']  # df.apply(lambda x: '%s' % (x['exp_group']), axis=1)
    # df['comp_group'] = df['grouping']
    df['comp_group'] = df.apply(lambda x: '%s_%s' % (x['grouping'], x['time_group']), axis=1)
    # df['comp_group'] = df['time_group']
    groups = df.grouping.unique()
    groups = groups.tolist()
    time_groups = df.time_group.unique()
    time_groups = time_groups.tolist()

    colors = sns.color_palette()[:len(time_groups)]

    bins = np.linspace(0, 2500, 15)
    bin_centers = bins[:-1] + np.diff(bins) / 2
    labels = []
    dists = []

    def drop_zero_cols(x):
        return x[:, np.any(x != 0, axis=0)]

    fig, axes = plt.subplots(nrows=len(groups), figsize=(8, 10))
    i = 0
    for n1, group in df.groupby('grouping'):
        idx = groups.index(n1)
        ax = axes[idx]
        for i, n2 in enumerate(time_groups):

            grp = group.query('time_group == @n2')  # change grp to group
            # cghange wherever it uses n2->n1,
            data = grp[value_col]
            mu, sig = norm.fit(data)
            x = np.linspace(0, 2500, 100)
            y = norm.pdf(x, mu, sig)
            counts, _ = np.histogram(data, bins=bins)
            density, _ = np.histogram(data, bins=bins, density=True)
            y_fit = norm.pdf(bin_centers, mu, sig)
            labels.append('%s_%s' % (n1, n2))
            dists.append(counts)
            if ax.is_last_row():
                l1 = n2
            else:
                l1 = None

            ax.hist(data, density=True, fc=(*colors[i], 0.4), label=l1, bins=bins, edgecolor='k')
            ss_res = np.sum((density - y_fit) ** 2)
            ss_tot = np.sum((density - np.mean(density)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            fit_str = r'$\mu$=%3.3g, $\sigma$=%3.3g, $r^{2}$=%0.3g' % (mu, sig, r2)
            ax.plot(x, y, color=colors[i], label=fit_str)
            i = i + 1

        ax.set_ylabel(n1)
        ax.legend()
        sns.despine(ax=ax)

        if not ax.is_last_row():
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Transition Time (ms)')

    # g = sns.displot(data=df.query('state_group == @state'), x=value_col,
    #                hue='time_group', row='grouping', hue_order = time_groups,
    #                kde=False, row_order=groups)
    # for l, ax in zip(groups, g.axes):
    #    ax.set_title('')
    #    ax.set_ylabel(l)
    #    if ax.is_last_row():
    #        ax.set_xlabel('Transition Time (ms)')

    ##g.set_titles('{row_name}')
    # g.fig.set_size_inches([7.9, 8.2])
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    title = '%s state %s times' % (state, value_col.split('_')[-1])
    fig.suptitle(title)

    # Get distributions and do stats and fits
    dists = np.vstack(dists)

    s, p, dof, exp = chi2_contingency(drop_zero_cols(dists))
    out_stats = {'omnibus': {'A': 'all', 'B': 'all', 'chi2 stat': s, 'pval': p, 'dof': dof}}
    pairs = [['%s_%s' % (x, y) for y in time_groups] for x in groups]
    # pairs = groups
    for i, (A, B) in enumerate(pairs):
        # A = pairs[0]
        # B = pairs[1]
        i1 = labels.index(A)
        i2 = labels.index(B)
        x = drop_zero_cols(dists[[i1, i2]])
        s, p, dof, exp = chi2_contingency(x)
        out_stats = {f'{i}': {'A': A, 'B': B, 'chi2 stat': s, 'pval': p, 'dof': dof}}
        if p <= 0.001:
            ss = '***'
        elif p <= 0.01:
            ss = '**'
        elif p <= 0.05:
            ss = '*'
        else:
            ss = ''

        grp = [x for x in groups if x in A][0]
        idx = groups.index(grp)
        axes[idx].set_title(ss)

    tmp = ['%s:  %s' % (x, y) for x, y in zip(labels, dists)]
    tmp = '\n' + '\n'.join(tmp) + '\n'
    out_stats['counts'] = tmp

    if save_file:
        fig.savefig(save_file)
        plt.close(fig)
        fn, ext = os.path.splitext(save_file)
        agg.write_dict_to_txt(out_stats, save_file=fn + '.txt')
        return None
    else:
        return fig, axes


def fix_coding_df(df):
    # Melt df to get state_group column
    id_cols = ['exp_name', 'exp_group', 'time_group']
    other_cols = ['n_cells', 'n_held_cells']
    data_cols = {'id_acc': ('early_ID_acc', 'late_ID_acc'),
                 'pal_acc': ('early_pal_acc', 'late_pal_acc')}
    df2 = None
    for k, v in data_cols.items():
        tmp = df.melt(id_vars=[*id_cols, *other_cols],
                      value_vars=v,
                      var_name='state_group', value_name=k)
        tmp['state_group'] = tmp['state_group'].apply(lambda x: x.split('_')[0])
        if df2 is None:
            df2 = tmp
        else:
            df2 = pd.merge(df2, tmp, on=[*id_cols, *other_cols, 'state_group'],
                           validate='1:1')

    # NaN in n_cells, means not enough cells were present to fit hmms (<3)
    df = df2.dropna().copy()
    return df


def plot_coding_data(df, save_file=None, group_col='exp_group',
                     plot_points=False, kind='bar'):
    df = fix_coding_df(df)
    # Make extra column for composite grouping
    df['grouping'] = df.apply(lambda x: '%s_%s' % (x[group_col], x['time_group']), axis=1)

    states = df['state_group'].unique()
    groups = df[group_col].unique()
    hues = df['time_group'].unique()
    group_order = ORDERS[group_col]
    hue_order = ORDERS['time_group']
    state_order = ORDERS['state_group']
    cond_order = []
    for g, h in it.product(group_order, hue_order):
        cond_order.append('%s_%s' % (g, h))

    fig = plt.figure(figsize=(14, 10))
    outer_ax = add_suplabels(fig, 'HMM Coding Analysis', '', 'Classification Accuracy (%)')
    nrows = len(states)
    ncols = 2
    axes = np.array([[fig.add_subplot(nrows, ncols, j + ncols * i + 1)
                      for j in range(ncols)] for i in range(nrows)])
    axes[0, 0].get_shared_y_axes().join(axes[0, 0], axes[1, 0])
    axes[0, 0].get_shared_y_axes().join(axes[0, 0], axes[0, 1])
    axes[0, 1].get_shared_y_axes().join(axes[0, 1], axes[1, 1])
    statistics = {}
    for sg, (id_ax, pal_ax) in zip(state_order, axes):
        grp = df.query('state_group == @sg')
        id_kw_stat, id_kw_p, id_gh_df = stats.kw_and_gh(grp, 'grouping', 'id_acc')
        pal_kw_stat, pal_kw_p, pal_gh_df = stats.kw_and_gh(grp, 'grouping', 'pal_acc')
        id_sum = grp.groupby(['exp_group', 'time_group'])['id_acc'].describe()
        pal_sum = grp.groupby(['exp_group', 'time_group'])['pal_acc'].describe()
        statistics[sg] = {'id': {'kw_stat': id_kw_stat, 'kw_p': id_kw_p,
                                 'posthoc': id_gh_df, 'summary': id_sum},
                          'pal': {'kw_stat': pal_kw_stat, 'kw_p': pal_kw_p,
                                  'posthoc': pal_gh_df, 'summary': pal_sum}}

        g1 = plot_box_and_paired_points(grp, group_col, 'id_acc',
                                        'time_group', order=group_order,
                                        hue_order=hue_order,
                                        subjects='exp_name',
                                        ax=id_ax, kind=kind,
                                        plot_points=plot_points)

        g2 = plot_box_and_paired_points(grp, group_col, 'pal_acc',
                                        'time_group', order=group_order,
                                        hue_order=hue_order,
                                        subjects='exp_name',
                                        ax=pal_ax, kind=kind,
                                        plot_points=plot_points)

        g1.axhline(100 / 3, linestyle='--', alpha=0.5, color='k')
        g2.axhline(100 / 3, linestyle='--', alpha=0.5, color='k')

        g1.set_ylabel(sg.replace('_', ' '))
        g1.legend_.remove()
        g1.set_xlabel('')
        g2.set_xlabel('')
        g2.set_ylabel('')
        if id_ax.is_first_row():
            g1.set_title('ID Coding Accuracy')
            g2.set_title('Pal Coding Accuracy')

        if not pal_ax.is_last_row():
            g2.legend_.remove()
        else:
            g2.legend(bbox_to_anchor=[1.2, 1.2, 0, 0])

        # n_cells = grp.groupby('grouping').size().to_dict()
        n_cells = None
        g1_y = plot_sig_stars(g1, id_gh_df, cond_order, n_cells=n_cells)
        g2_y = plot_sig_stars(g2, pal_gh_df, cond_order, n_cells=n_cells)

    if save_file:
        fn, ext = os.path.splitext(save_file)
        fn2 = fn + '.txt'
        fig.savefig(save_file)
        plt.close(fig)
        agg.write_dict_to_txt(statistics, save_file=fn2)
    else:
        return fig, axes, statistics


def plot_box_and_paired_points(df, x, y, hue, order=None, hue_order=None,
                               subjects=None, estimator=np.mean,
                               error_func=sem, kind='box', plot_points=True, **kwargs):
    groups = df[x].unique()
    hues = df[hue].unique()
    if order is None:
        order = groups

    if hue_order is None:
        hue_order = hues

    # early state end time
    if kind == 'box':
        ax = sns.boxplot(data=df, x=x, hue=hue, y=y, order=order,
                         hue_order=hue_order, **kwargs)
    elif kind == 'bar':
        ax = sns.barplot(data=df, x=x, hue=hue, y=y, order=order,
                         hue_order=hue_order, **kwargs)
    elif kind == 'violin':
        ax = sns.violinplot(data=df, x=x, hue=hue, y=y, order=order,
                            hue_order=hue_order, **kwargs)
    else:
        raise ValueError('kind must be bar or box or violin')

    if not plot_points:
        return ax

    xpts = []
    for p in ax.patches:
        x1 = p.get_x() + p.get_width() / 2
        xpts.append(x1)

    xpts.sort()
    max_jitter = np.min(np.diff(xpts))
    plot_pts = []
    xmap = {}
    for (g, h), xp in zip(it.product(order, hue_order), xpts):
        xmap[(g, h)] = xp

    for subj, grp in df.groupby(subjects):
        for g in grp[x].unique():
            xvals = []
            yvals = []
            yerr = []
            for h in hue_order:
                if h not in grp[hue].values:
                    continue

                tmp = grp[(grp[hue] == h) & (grp[x] == g)]
                r = (np.random.rand(1)[0] - 0.5) * max_jitter / 4
                yvals.append(estimator(tmp[y]))
                xvals.append(xmap[(g, h)] + r)
                yerr.append(error_func(tmp[y]))

            ax.errorbar(xvals, yvals, yerr=yerr, alpha=0.4, marker='.',
                        markersize=10, color='grey', linewidth=2)

    return ax


def plot_sig_stars(ax, posthoc_df, cond_order, n_cells=None):
    if posthoc_df is None:
        return

    truedf = posthoc_df[posthoc_df['reject']]

    xpts = []
    ypts = []
    if len(ax.patches) < len(cond_order):
        labels = ax.get_xticklabels()
        # xpts = {x.get_text(): x.get_position()[0] for x in labels}
        xpts = {x: i for i, x in enumerate(cond_order)}
        ypts = {}
        pts = {}
        for l in ax.lines:
            xd = l.get_xdata()
            yd = l.get_ydata()
            for x, y in zip(xd, yd):
                if x in ypts.keys():
                    ypts[x] = np.max((ypts[x], y))
                else:
                    ypts[x] = y

        for k, x in xpts.items():
            pts[k] = (x, ypts[x])

    else:
        for p, cond in zip(ax.patches, cond_order):
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            xpts.append(x)
            ypts.append(y)

        idx = np.argsort(xpts)
        xpts = [xpts[i] for i in idx]
        ypts = [ypts[i] for i in idx]
        pts = {cond: (x, y) for cond, x, y in zip(cond_order, xpts, ypts)}

    slines = []  # x1, x2, y1, y2
    sdists = []
    max_y = 0

    for i, row in truedf.iterrows():
        g1 = row['A']
        g2 = row['B']
        p = row['pval']
        if p <= 0.001:
            ss = '***'
        elif p <= 0.01:
            ss = '**'
        elif p <= 0.05:
            ss = '*'
        else:
            continue

        x1, y1 = pts[g1]
        x2, y2 = pts[g2]
        y1 = 1.2 * y1
        y2 = 1.2 * y2
        dist = abs(x2 - x1)
        sdists.append(dist)
        slines.append((x1, x2, y1, y2, ss))
        if y1 > max_y:
            max_y = y1

        if y2 > max_y:
            max_y = y2

    if n_cells:
        for k, v in n_cells.items():
            x1, y1 = pts[k]
            ax.text(x1, .1 * max_y, f'N={v}', horizontalalignment='center', color='white')

    if truedf.empty:
        return

    sdists = np.array(sdists)
    idx = list(np.argsort(sdists))
    idx.reverse()
    scaling = max_y * 8 / 100
    ytop = max_y + scaling * len(truedf)
    maxy = ytop
    for i in idx:
        x1, x2, y1, y2, ss = slines[i]
        mid = (x1 + x2) / 2
        ax.plot([x1, x1, x2, x2], [y1, ytop, ytop, y2], linewidth=1, color='k')
        ax.text(mid, ytop, ss, horizontalalignment='center', fontsize=14, fontweight='bold')
        ytop -= scaling

    return maxy + 5


# TODO: break down by groups
def plot_BIC(ho, proj, save_file=None):
    ho = fix_hmm_overview(ho, proj)
    # ho = ho.query('notes == "sequential - BIC test"')
    ho = ho[ho['n_cells'] >= 3]
    ho = ho[ho['taste'] != 'Spont']
    fig, ax = plt.subplots()
    g = sns.barplot(data=ho, x='n_states', y='BIC', ax=ax)
    cond_order = sorted(ho['n_states'].unique())
    kw_s, kw_p, gh = stats.kw_and_gh(ho, 'n_states', 'BIC')
    # plot_sig_stars(g, gh, cond_order)
    statistics = {'BIC Stats': {'KW stat': kw_s, 'KW p': kw_p, 'games_howell': gh}}
    if save_file:
        fig.savefig(save_file)
        fn, ext = os.path.splitext(save_file)
        agg.write_dict_to_txt(statistics, fn + '.txt')
        plt.close(fig)
    else:
        return fig, ax, statistics


def plot_best_BIC(srt_df, save_file=None):
    sns.set(font_scale=1)
    g = sns.FacetGrid(data=srt_df, row="exp_group", height=6, aspect=1.25)
    g.map_dataframe(sns.barplot, x="taste", y="n_states", hue="time_group",
                    order=(["Suc", "NaCl", "CA", "QHCl", "Spont"]), dodge=True,
                    palette="Set2")
    g.map_dataframe(sns.swarmplot, x="taste", y="n_states", hue="time_group",
                    order=(["Suc", "NaCl", "CA", "QHCl", "Spont"]), dodge=True,
                    palette=["#404040"])
    g.set_axis_labels("", "best # states")
    plt.legend(title="# session exposure", loc="upper left")

    if save_file:
        g.savefig(save_file)
        plt.close('all')
    else:
        return g


def plot_grouped_BIC(srt_df, save_file=None):
    sns.set(font_scale=1)
    g = sns.catplot(data=srt_df,
                    kind="bar",
                    hue="time_group",
                    col="taste",
                    row='exp_group',
                    x="n_states",
                    y="BIC",
                    col_order=(["Suc", "NaCl", "CA", "QHCl", "Spont"]),
                    aspect=.7,
                    margin_titles=True)
    g.map_dataframe(sns.swarmplot,
                    x="n_states",
                    y="BIC",
                    hue="time_group",
                    dodge=True,
                    palette=["#404040"])
    g.set_axis_labels("number of HMM states", "BIC")
    if save_file and g.fig:
        g.savefig(save_file)
        plt.close('all')
    else:
        return g


def fix_hmm_overview(ho, proj):
    ho = ho.copy()
    df = proj._exp_info
    ho['exp_group'] = ho['exp_name'].map(df.set_index('exp_name')['exp_group'].to_dict())
    # df['cta_group'] = df['CTA_learned'].apply(lambda x: 'CTA' if x else 'No CTA')
    # ho['cta_group'] = ho['exp_name'].map(df.set_index('exp_name')['cta_group'].to_dict())
    # ho['time_group'] = ho['rec_dir'].apply(lambda x: 'Exposure_1'
    #                                        if ('pre' in x or 'Train' in x)
    #                                        else 'Exposure_1') #may change later?
    return ho


def plot_hmm_trial_breakdown(df, proj, save_file=None):
    df = fix_hmm_overview(df, proj)
    # df = df.query('exp_group != "Cre" or cta_group != "CTA"')
    df = df.query('taste != "Spont" and n_cells >= 3').copy()
    # df = df.query('exclude == False')
    df['grouping'] = df.apply(lambda x: '%s_%s\n%s' % (x['exp_group'],
                                                       # x['cta_group'],
                                                       x['time_group']), axis=1)
    id_cols = ['exp_group', 'time_group', 'grouping', 'taste']
    df2 = df.groupby([*id_cols, 'state_presence']).size().reset_index()
    df2 = df2.rename(columns={0: 'count'})
    df2['percent'] = df2.groupby(['grouping', 'taste'])['count'].apply(lambda x: 100 * x / sum(x))

    o1 = ORDERS['exp_group']
    # o2 = ORDERS['cta_group']
    o2 = ORDERS['time_group']
    row_order = ORDERS['taste'].copy()
    if 'Water' in row_order:
        row_order.pop(row_order.index('Water'))

    hue_order = ORDERS['state_presence']

    # cond_order = ['%s_%s\n%s' % x for x in it.product(o1, o2)
    #               if (x[0] != "Cre" or x[1] != "CTA")]
    cond_order = o1
    statistics = {}
    g = sns.catplot(data=df2, x='grouping', y='percent', row='taste',
                    hue='state_presence', kind='bar', order=cond_order,
                    row_order=row_order, hue_order=hue_order)
    g.fig.set_size_inches([14, 20])
    g.set_xlabels('')

    df3 = df.groupby([*id_cols, 'state_presence']).size()
    df3 = df3.unstack('state_presence', fill_value=0).reset_index()
    cond_y = [df2[df2.grouping == x].percent.max() + 5 for x in cond_order]

    for taste, group in df3.groupby('taste'):
        row = row_order.index(taste)
        ax = g.axes[row, 0]
        ax.set_ylabel(taste)
        ax.set_title('')
        tmp_stats = stats.chi2_contingency_with_posthoc(group,
                                                        [  # 'exp_group',
                                                            # 'cta_group',
                                                            'time_group'],
                                                        hue_order,
                                                        label_formatter='%s_%s\n%s')

        sdf = pd.DataFrame.from_dict(tmp_stats, orient='index')
        statistics[taste] = sdf
        tsdf = sdf[sdf['reject']]
        if len(tsdf) == 0:
            continue

        ytop = max(cond_y) + 5
        for i, row in tsdf.iterrows():
            if i == 'omnibus':
                continue

            x1 = cond_order.index(row['A'])
            x2 = cond_order.index(row['B'])
            ax.plot([x1, x1, x2, x2], [cond_y[x1], ytop, ytop, cond_y[x2]], color='k')
            ss = '***' if row['pval'] < 0.001 else '**' if row['pval'] < 0.01 else '*'
            mid = (x1 + x2) / 2
            ax.text(mid, ytop + 1, ss, fontsize=14)
            ytop += 8

    g.fig.suptitle('% trials containing HMM states')
    # plt.tight_layout()
    if save_file:
        g.fig.savefig(save_file)
        plt.close(g.fig)
        fn, ext = os.path.splitext(save_file)
        agg.write_dict_to_txt(statistics, fn + '.txt')
    else:
        return g, statistics


def plot_taste_responsive_units(tasty_df, save_file=None):
    tasty_df = tasty_df[tasty_df['single_unit']].copy()
    order = ORDERS['exp_group']
    hue_order = ORDERS['time_group']
    df = tasty_df.groupby(['exp_group', 'time_group',
                           'taste', 'taste_responsive']).size()
    df = df.unstack('taste_responsive', fill_value=0).reset_index()
    df = df.rename(columns={True: 'responsive', False: 'non-responsive'})

    def percent(x):
        return 100 * x['responsive'] / (x['responsive'] + x['non-responsive'])

    df['percent_responsive'] = df.apply(percent, axis=1)

    tastes = list(df.taste.unique())
    groups = list(it.product(df.exp_group.unique(), df.time_group.unique()))
    fig, axes = plt.subplots(nrows=len(tastes), ncols=len(groups), figsize=(12, 12))
    for (eg, tg, tst), grp in df.groupby(['exp_group', 'time_group', 'taste']):
        row = tastes.index(tst)
        col = groups.index((eg, tg))
        if len(tastes) == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]

        labels = ['responsive', 'non-responsive']
        values = [grp[x].sum() for x in labels]
        ax.pie(values, autopct='%1.1f%%')
        if ax.is_first_col():
            ax.set_ylabel(tst)

        if ax.is_first_row():
            ax.set_title(f'{eg}\n{tg}')

        if ax.is_last_row() and ax.is_last_col():
            ax.legend(labels, bbox_to_anchor=[1.6, 2.5, 0, 0])

    plt.subplots_adjust(top=0.85)
    fig.suptitle('% Taste Responsive Units')

    if save_file:
        fig.savefig(save_file)
        plt.close(fig)

    df2 = tasty_df.groupby(['exp_name', 'exp_group',
                            'time_group', 'rec_group',
                            'unit_num'])['taste_responsive'].any()
    df2 = df2.reset_index().groupby(['exp_group', 'time_group', 'taste_responsive']).size()
    df2 = df2.unstack('taste_responsive', fill_value=0)
    df2 = df2.rename(columns={True: 'responsive', False: 'non-responsive'}).reset_index()
    df2['percent_responsive'] = df2.apply(percent, axis=1)
    df2['n_cells'] = df2.apply(lambda x: x['responsive'] + x['non-responsive'], axis=1)
    df2['labels'] = df2.apply(lambda x: '%s_%s' % (x['exp_group'], x['time_group']), axis=1)
    n_cells = df2.set_index('labels')['n_cells'].to_dict()

    groups = list(it.product(df.exp_group.unique(), df.time_group.unique()))
    fig2, axes2 = plt.subplots(figsize=(12, 9))
    statistics = stats.chi2_contingency_for_taste_responsive_cells(df2)
    g = sns.barplot(data=df2, x='exp_group', hue='time_group',
                    y='percent_responsive', order=order,
                    hue_order=hue_order, ax=axes2)
    cond_order = ['%s_%s' % x for x in list(it.product(order, hue_order))]
    ph_df = pd.DataFrame.from_dict(statistics, orient='index')
    ph_df = ph_df.iloc[1:]
    plot_sig_stars(axes2, ph_df, cond_order, n_cells=n_cells)
    axes2.set_title('% Taste Responsive Units')

    if save_file:
        fn, ext = os.path.splitext(save_file)
        fn = fn + '-stats'
        statistics['counts'] = df2
        agg.write_dict_to_txt(statistics, fn + '.txt')
        fig2.savefig(fn + '.svg')
        plt.close(fig2)
        return
    else:
        return fig, axes, fig2, axes2


def plot_pal_responsive_units(pal_df, save_dir=None):
    pal_df = pal_df[pal_df['single_unit'] & (pal_df['area'] == 'GC')].copy()
    order = ORDERS['exp_group']
    hue_order = ORDERS['time_group']
    cond_order = ['%s_%s' % x for x in list(it.product(order, hue_order))]
    pal_df['abs_corr'] = pal_df['spearman_r'].apply(np.abs)
    pal_df['group_col'] = pal_df.apply(lambda x: '%s_%s' % (x['exp_group'], x['time_group']), axis=1)

    # Plot 1
    fig, ax = plt.subplots(figsize=(10, 9))
    g = sns.barplot(data=pal_df, x='exp_group', y='abs_corr', hue='time_group',
                    order=order, hue_order=hue_order, ax=ax)
    g.set_title('Peak Spearman Correlation to Palatability')
    g.set_ylabel('|Spearman R|')
    g.set_xlabel('')
    kw_stat, kw_p, gh_df = stats.kw_and_gh(pal_df, 'group_col', 'spearman_r')
    n_cells = pal_df.groupby('group_col')['spearman_r'].size().to_dict()
    plot_sig_stars(ax, gh_df, cond_order, n_cells=n_cells)
    if save_dir:
        fn = os.path.join(save_dir, 'palatability_spearman_corr.svg')
        fig.savefig(fn)
        fn2 = fn.replace('.svg', '.txt')
        out = {'kw_stat': kw_stat, 'kw_p': kw_p, 'Games-Howell posthoc': gh_df}
        agg.write_dict_to_txt(out, fn2)
        plt.close(fig)

    # taste disrcrim plot
    df = pal_df.groupby(['exp_group', 'time_group',
                         'taste_discriminative']).size()
    df = df.unstack('taste_discriminative', fill_value=0).reset_index()
    df = df.rename(columns={True: 'discriminative', False: 'non-discriminative'})

    def percent(x):
        return 100 * x['discriminative'] / (x['discriminative'] + x['non-discriminative'])

    df['percent_discriminative'] = df.apply(percent, axis=1)

    groups = list(it.product(df.exp_group.unique(), df.time_group.unique()))
    fig, axes = plt.subplots(ncols=len(groups), figsize=(12, 12))
    for (eg, tg), grp in df.groupby(['exp_group', 'time_group']):
        col = groups.index((eg, tg))
        ax = axes[col]

        labels = ['discriminative', 'non-discriminative']
        values = [grp[x].sum() for x in labels]
        ax.pie(values, autopct='%1.1f%%')

        if ax.is_first_row():
            ax.set_title(f'{eg}\n{tg}')

        if ax.is_last_row() and ax.is_last_col():
            ax.legend(labels, bbox_to_anchor=[1.6, 2.5, 0, 0])

    plt.subplots_adjust(top=0.85)
    fig.suptitle('% Taste Discriminative Units')

    if save_dir:
        fn = os.path.join(save_dir, 'taste_discriminative.svg')
        fig.savefig(fn)
        plt.close(fig)

    df2 = pal_df.groupby(['exp_name', 'exp_group',
                          'time_group', 'rec_group',
                          'unit_num'])['taste_discriminative'].any()
    df2 = df2.reset_index().groupby(['exp_group', 'time_group', 'taste_discriminative']).size()
    df2 = df2.unstack('taste_discriminative', fill_value=0)
    df2 = df2.rename(columns={True: 'discriminative', False: 'non-discriminative'}).reset_index()
    df2['percent_discriminative'] = df2.apply(percent, axis=1)
    df2['n_cells'] = df2.apply(lambda x: x['discriminative'] + x['non-discriminative'], axis=1)
    df2['labels'] = df2.apply(lambda x: '%s_%s' % (x['exp_group'], x['time_group']), axis=1)
    n_cells = df2.set_index('labels')['n_cells'].to_dict()

    fig2, axes2 = plt.subplots(figsize=(12, 9))
    statistics = stats.chi2_contingency_for_taste_responsive_cells(df2,
                                                                   value_cols=['discriminative',
                                                                               'non-discriminative'])
    g = sns.barplot(data=df2, x='exp_group', hue='time_group',
                    y='percent_discriminative', order=order,
                    hue_order=hue_order, ax=axes2)
    cond_order = ['%s_%s' % x for x in list(it.product(order, hue_order))]
    ph_df = pd.DataFrame.from_dict(statistics, orient='index')
    ph_df = ph_df.iloc[1:]
    plot_sig_stars(axes2, ph_df, cond_order, n_cells=n_cells)
    axes2.set_title('% Taste Discriminative Units')

    if save_dir:
        fn = os.path.join(save_dir, 'taste_discriminative_comparison')
        statistics['counts'] = df2
        agg.write_dict_to_txt(statistics, fn + '.txt')
        fig2.savefig(fn + '.svg')
        plt.close(fig2)
        return
    else:
        return fig, axes, fig2, axes2


# def plot_mean_spearman_correlation(df, save_file=None):
def plot_mean_spearman_correlation(pal_file, proj, save_file=None):
    data = np.load(pal_file)
    # labels are : exp_group, time_group, rec_dir, unit_num
    l = list(data['labels'])
    sr = data['spearman_r']
    t = data['time']
    index = pd.MultiIndex.from_tuples(l, names=['exp_group', 'time_group',
                                                'rec_dir', 'unit_num'])
    df = pd.DataFrame(sr, columns=t, index=index)
    df = df.reset_index().melt(id_vars=['exp_group', 'time_group',
                                        'rec_dir', 'unit_num'],
                               value_vars=t,
                               var_name='time_bin',
                               value_name='spearman_r')

    df = agg.apply_grouping_cols(df, proj)
    # Drop Cre - No CTA
    # df = df.query('exp_group != "Cre" or cta_group != "CTA"')
    df = df.copy()
    df['grouping'] = df.apply(lambda x: '%s_%s' % (x['exp_group']), axis=1)
    df['abs_r'] = df['spearman_r'].abs()
    df['r2'] = df['spearman_r'] ** 2
    df['resp_time'] = df['time_bin'].apply(lambda x: 'Early (0-750ms)' if x < 750
    else 'Late (750-2000ms)')

    diff_df = stats.get_diff_df(df, ['exp_group', 'time_group'],
                                'resp_time', 'r2')
    diff_df['grouping'] = diff_df.apply(lambda x: '%s\n%s' % (x['exp_group']),
                                        axis=1)
    diff_df['mean_diff'] = -diff_df['mean_diff']

    col_order = list(df.grouping.unique())
    style_order = ORDERS['time_group']
    colors = sns.color_palette()[:len(col_order)]
    styles = ['-', '--', '-.', '.']
    styles = styles[:len(style_order)]
    markers = ['.', 'D', 'x', 'v']
    markers = markers[:len(style_order)]
    hues = [1, 0.4, 1.4]
    fig = plt.figure(figsize=(15, 8))
    _ = add_suplabels(fig, 'Single Unit Mean Correlation to Palatability',
                      'Time (ms)', "Spearman's R^2")
    axes = []
    for i, grouping in enumerate(col_order):
        grp = df.query('grouping == @grouping')
        ax = fig.add_subplot(1, len(col_order), i + 1)
        axes.append(ax)

        for j, tg in enumerate(style_order):
            tmp = grp.query('time_group == @tg').groupby('time_bin')['r2'].agg([np.mean, sem])
            x = np.array(tmp.index)
            x = x - x[0]
            y = tmp['mean'].to_numpy()
            y = gaussian_filter1d(y, 3)
            err = tmp['sem'].to_numpy()
            c = colors[j]
            ax.fill_between(x, y + err, y - err, color=c, alpha=0.4)
            ax.plot(x, y, color=c, linewidth=2, label=tg)
            ax.set_xlim([x[0], x[-1]])

        ax.set_title(grouping.replace('_', ' '))
        if ax.is_last_col():
            ax.legend(style_order, bbox_to_anchor=[1.2, 1.2, 0, 0])

    plt.tight_layout()

    df['simple_groups'] = df.apply(lambda x: '%s_%s' % (x['exp_group']),
                                   axis=1)
    df['grouping'] = df.apply(lambda x: '%s_%s\n%s' % (x['exp_group'],
                                                       x['time_group']), axis=1)
    df['comp_grouping'] = df.apply(lambda x: '%s_%s' % (x['grouping'], x['resp_time']), axis=1)
    o1 = ORDERS['exp_group']
    o2 = ORDERS['time_group']
    o3 = ['Early (0-750ms)', 'Late (750-2000ms)']
    o4 = ORDERS['taste']
    s_order = [f'{x}_{y}' for x, y in it.product(o1, o2)]
    s_order = [x for x in s_order if x in df.simple_groups.unique()]
    g_order = [f'{x}_{y}\n{z}' for x, y, z in it.product(o1, o2)]
    g_order = [x for x in g_order if x in df.grouping.unique()]
    cond_order = [f'{x}_{y}' for x, y in it.product(g_order, o3)]
    cond_order = [x for x in cond_order if x in df.comp_grouping.unique()]
    fig2, ax = plt.subplots(figsize=(14, 8))
    g = sns.barplot(data=df, x='grouping', y='r2', hue='resp_time',
                    order=g_order, hue_order=o4, ax=ax)
    kw_s, kw_p, gh_df = stats.kw_and_gh(df, 'comp_grouping', 'r2')
    statistics = {'KW Stat': kw_s, 'KW p-val': kw_p, 'Games-Howell posthoc': gh_df}
    # Slim down gh_df to only comparisons I care about
    valid_gh = []
    for i, row in gh_df.iterrows():
        a = row['A']
        b = row['B']
        s1 = a.split('\n')
        s1 = np.array([s1[0], *s1[1].split('_')])
        s2 = b.split('\n')
        s2 = np.array([s2[0], *s2[1].split('_')])
        if sum(s1 == s2) == 2 and s1[0] == s2[0]:
            valid_gh.append(row)

    valid_gh = pd.DataFrame(valid_gh)
    plot_sig_stars(ax, valid_gh, cond_order)
    ax.set_xlabel('')
    ax.set_ylabel("Spearman's R^2")
    ax.set_title('Mean Palatability Correlation\n'
                 'only showing small subset of significant differences')
    tmp = df.groupby(['exp_name', 'exp_group', 'time_group'])['unit_num']
    tmp = tmp.agg(lambda x: len(np.unique(x)))
    tmp = tmp.groupby(['exp_group', 'time_group']).sum().reset_index()
    tmp = tmp.rename(columns={'unit_num': 'n_cells'})
    statistics['n_cells'] = tmp

    # Plot with simplified grouping
    fig4, ax = plt.subplots(figsize=(10, 8))
    # s_order = ['Exposure_1', 'Exposure_3']
    s_order = ['naive', 'suc_preexp']
    g2_order = [f'{x}\n{y}' for x, y in it.product(s_order, o3)]
    g = sns.barplot(data=df, x='simple_groups', y='r2', hue='time_group',
                    order=s_order, hue_order=o3, ax=ax)
    ax.set_ylim([0, 0.11])
    kw_s, kw_p, gh_df = stats.kw_and_gh(df.query('exclude==False'), 'grouping', 'r2')
    other_stats = {'KW Stat': kw_s, 'KW p-val': kw_p, 'Games-Howell psthoc': gh_df}
    # Everything is significant, just add in illustrator
    # plot_sig_stars(ax, gh_df, g2_order)

    # Plot differences
    # o1 = ['GFP\nCTA', 'Cre\nNo CTA', 'GFP\nNo CTA']
    # o1 = ['Exposure_1', 'Exposure_3']
    o1 = ['naive', 'suc_preexp']
    cond_order = list(it.product(o1))
    fig3, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=diff_df, ax=ax, x='grouping', y='mean_diff',
                hue='time_group', order=o1, hue_order=o2)
    xdata = [x.get_x() + x.get_width() / 2 for x in ax.patches]
    xdata.sort()
    tmp = diff_df.set_index(['grouping', 'time_group'])[['mean_diff', 'sem_diff']].to_dict()
    for x, grp in zip(xdata, cond_order):
        ym = tmp['mean_diff'][grp]
        yd = tmp['sem_diff'][grp]
        ax.plot([x, x], [ym - yd, ym + yd], color='k', linewidth=3)

    ax.set_xlabel('')
    ax.grid(True, axis='y', linestyle=':')
    ax.set_ylabel(r"$\Delta$ Spearman's R^2")
    ax.get_legend().set_title('Epoch')
    ax.set_title('Change in correlation to palatability\nbetween early and late halves of response')

    if save_file:
        fig.savefig(save_file)
        plt.close(fig)
        fn, ext = os.path.splitext(save_file)
        fn = fn + '-comparison'
        fig2.savefig(fn + '.svg')
        agg.write_dict_to_txt(statistics, fn + '.txt')
        plt.close(fig2)

        fn2 = fn.replace('comparison', 'simple_comparison')
        fig4.savefig(fn2 + '.svg')
        agg.write_dict_to_txt(other_stats, fn2 + '.txt')
        plt.close(fig4)

        fn = fn.replace('comparison', 'differences.svg')
        fig3.savefig(fn)
        plt.close(fig3)
    else:
        return fig, fig2, fig3, statistics


def plot_MDS(df, value_col='MDS_dQ_v_dN', group_col='exp_group',
             ylabel='dQ/dN', save_file=None, kind='bar'):
    order = ORDERS[group_col]
    hue_order = ORDERS['time_group']
    col_order = ORDERS['MDS_time']
    cond_order = ['%s_%s' % x for x in list(it.product(order, hue_order))]
    df['group_col'] = df.apply(lambda x: '%s_%s' % (x[group_col], x['time_group']), axis=1)

    g = sns.catplot(data=df, y=value_col, x=group_col,
                    hue='time_group', col='time', kind=kind, order=order,
                    hue_order=hue_order, col_order=col_order)

    axes = g.axes[0]
    statistics = {}
    for ax, (tg, grp) in zip(axes, df.groupby('time')):
        kw_s, kw_p, gh_df = stats.kw_and_gh(grp, 'group_col', value_col)
        statistics[tg] = {'kw-stat': kw_s, 'kw-p': kw_p,
                          'Games-Howell posthoc': gh_df}
        n_cells = grp.groupby('group_col').size().to_dict()
        plot_sig_stars(ax, gh_df, cond_order, n_cells=n_cells)
        ax.set_title(tg)
        if ax.is_first_col():
            ax.set_ylabel(ylabel)

        ax.set_xlabel('')

    plt.tight_layout()
    g.fig.set_size_inches(12, 10)
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Relative MDS Distances of Saccharin Trials')
    if save_file:
        g.fig.savefig(save_file)
        plt.close(g.fig)
        fn, ext = os.path.splitext(save_file)
        agg.write_dict_to_txt(statistics, fn + '.txt')
    else:
        return g, statistics


def plot_full_dim_MDS(df, save_file=None):
    df = df.copy()
    df['time'] = df.time.apply(lambda x: x.split(' ')[0])
    df['grouping'] = df.apply(lambda x: '%s_%s' % (x['exp_group']),
                              axis=1)
    df['plot_group'] = df.apply(lambda x: '%s\n%s' % (x['grouping'],
                                                      x['time']),
                                axis=1)
    df['comp_group'] = df.apply(lambda x: '%s_%s' % (x['plot_group'],
                                                     x['time_group']),
                                axis=1)

    df = df.query('(exp_group != "Cre" or cta_group != "CTA") '
                  'and taste != "Spont"')

    o1 = ORDERS['exp_group']
    # o2 = ORDERS['cta_group']
    o3 = ['Early', 'Late']
    o4 = ORDERS['time_group']
    plot_order = ['%s_%s\n%s' % (x, y, z) for z, x, y in it.product(o3, o1)]
    plot_order = [x for x in plot_order if x in df['plot_group'].unique()]
    comp_order = ['%s_%s' % (x, y) for x, y in it.product(plot_order, o4)]
    row_order = ORDERS['taste'].copy()
    if 'Water' in row_order:
        _ = row_order.pop(row_order.index('Water'))

    g = sns.catplot(data=df, row='taste', x='plot_group',
                    y='dQ_v_dN_fullMDS', kind='bar', hue='time_group',
                    order=plot_order,
                    hue_order=o4,
                    row_order=row_order,
                    sharey=False)
    statistics = {}
    for tst, group in df.groupby('taste'):
        row = row_order.index(tst)
        ax = g.axes[row, 0]
        ax.set_title('')
        ax.set_ylabel(tst)
        ax.set_xlabel('')
        kw_s, kw_p, gh_df = stats.kw_and_gh(group, 'comp_group', 'dQ_v_dN_fullMDS')
        tmp = {'Kruskal-Wallis stat': kw_s, 'Kruskal-Wallis p-val': kw_p,
               'Games-Howell posthoc': gh_df}
        statistics[tst] = tmp
        valid_comp = []
        for i, row in gh_df.iterrows():
            A = row['A'].split('\n')
            B = row['B'].split('\n')
            A = [y for x in A for y in x.split('_')]
            B = [y for x in B for y in x.split('_')]
            # if (A[0] == B[0] and A[1] == B[1]) or (A[3] == B[3]):
            if A[2] == B[2]:
                valid_comp.append(row)

        valid_comp = pd.DataFrame(valid_comp)
        plot_sig_stars(ax, valid_comp, comp_order)

    g.fig.set_size_inches(16, 18)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Relative MDS distances (dQ/dN)\nFull Dimension Solution')
    # g.legend.set_bbox_to_anchor([1.2,1.2,0,0])
    # plt.tight_layout()
    if save_file:
        g.fig.savefig(save_file)
        plt.close(g.fig)
        fn, ext = os.path.splitext(save_file)
        agg.write_dict_to_txt(statistics, fn + '.txt')
    else:
        return g, statistics


def plot_unit_firing_rates(all_units, group_col='exp_group', save_file=None):
    df = all_units.query('single_unit == True and exclude==False').copy()
    ups = df.groupby(['exp_name', 'rec_group']).size().mean()
    ups_sem = df.groupby(['exp_name', 'rec_group']).size().sem()
    statistics = {'units per session': '%1.2f ± %1.2f' % (ups, ups_sem),
                  'units per group': df.groupby('exp_group').size().to_dict()}
    value_cols = ['baseline_firing', 'response_firing', 'norm_response_firing']
    id_cols = ['exp_name', 'exp_group', 'rec_group', 'area', 'unit_type', 'time_group']
    df = df.melt(id_vars=id_cols, value_vars=value_cols,
                 var_name='firing_type', value_name='firing_rate')

    order = ORDERS[group_col]
    hue_order = ORDERS['time_group']
    col_order = ORDERS['unit_type']
    row_order = ['baseline_firing', 'response_firing', 'norm_response_firing']
    cond_order = ['%s_%s' % x for x in list(it.product(order, hue_order))]
    df['group_col'] = df.apply(lambda x: '%s_%s' % (x[group_col], x['time_group']), axis=1)

    # plot baseline and response firing rates 
    g = sns.catplot(data=df, x=group_col, y='firing_rate', kind='bar', hue='time_group',col='unit_type', row='firing_type', margin_titles=True,
                    order=order,
                    hue_order=hue_order, col_order=col_order,
                    row_order=row_order, sharey=False)
    g.set_titles(row_template='{row_name}', col_template='{col_name}')
    g.fig.set_size_inches((15, 12))
    for (ft, ut), group in df.groupby(['firing_type', 'unit_type']):
        row = row_order.index(ft)
        col = col_order.index(ut)
        ax = g.axes[row, col]
        if ax.is_first_row():
            ax.set_title(ut)
        else:
            ax.set_title('')

        if ax.is_first_col():
            ax.set_ylabel(' '.join(ft.split('_')[:-1]))
        else:
            ax.set_ylabel('')

        n_cells = group.groupby('group_col').size().to_dict()
        kw_s, kw_p, gh_df = stats.kw_and_gh(group, 'group_col', 'firing_rate')
        tmp = {'KW Stat': kw_s, 'KW p': kw_p, 'Games-Howell posthoc': gh_df}
        if ft in statistics.keys():
            statistics[ft][ut] = tmp
        else:
            statistics[ft] = {ut: tmp}

        plot_sig_stars(ax, gh_df, cond_order, n_cells=n_cells)

    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Unit Firing Rates (Hz)')
    if save_file:
        fn, ext = os.path.splitext(save_file)
        fn = fn + '.txt'
        g.fig.savefig(save_file)
        agg.write_dict_to_txt(statistics, fn)
        plt.close(g.fig)
    else:
        return g, statistics


def plot_saccharin_consumption(proj, save_file=None):
    df = proj._exp_info.copy()
    df['saccharin_consumption'] = df.saccharin_consumption.apply(lambda x: 100 * x)
    df['grouping'] = df.apply(lambda x: '%s_%s' % (x['exp_group']), axis=1)
    o1 = ORDERS['exp_group']
    # o2 = ORDERS['cta_group']
    order = ['%s_%s' % x for x in it.product(o1)]
    _ = order.pop(order.index('Cre_CTA'))
    df = df.query('grouping != "Cre_CTA"')
    # df = df.query('exclude == False')
    order = [x for x in order if x in df.grouping.unique()]
    order = ['GFP_CTA', 'Cre_No CTA', 'GFP_No CTA']
    df = df.dropna()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    g = sns.boxplot(data=df, x='grouping', y='saccharin_consumption', order=order, ax=ax)
    n_cells = df.groupby('grouping').size().to_dict()
    kw_s, kw_p, gh_df = stats.kw_and_gh(df, 'grouping', 'saccharin_consumption')
    out_stats = {'counts': n_cells, 'Kruskal-Wallis Stat': kw_s, 'Kruskal-Wallis p-val': kw_p,
                 'Games-Howell posthoc': gh_df}
    g.set_xlabel('')
    g.set_ylabel('% Saccharin Consumption')
    g.set_title('Saccharin Consumption\nrelative to mean water consumption')
    plot_sig_stars(g, gh_df, cond_order=order, n_cells=n_cells)
    g.axhline(80, linestyle='--', color='k', alpha=0.6)
    g.set_yscale('log')
    g.set_yticklabels(g.get_yticks(minor=True), minor=True)

    if save_file:
        fig.savefig(save_file)
        fn, ext = os.path.splitext(save_file)
        agg.write_dict_to_txt(out_stats, fn + '.txt')
        plt.close(fig)
    else:
        return g, out_stats


def plot_held_unit_comparison(rec1, unit1, rec2, unit2, pvals, params,
                              held_unit_name, exp_name, exp_group, taste,
                              save_file=None):
    dig1 = load_dataset(rec1).dig_in_mapping.copy().set_index('name')
    dig2 = load_dataset(rec2).dig_in_mapping.copy().set_index('name')
    ch1 = dig1.loc[taste, 'channel']
    ch2 = dig2.loc[taste, 'channel']

    bin_size = params['response_comparison']['win_size']
    step_size = params['response_comparison']['step_size']
    time_start = params['response_comparison']['time_win'][0]
    time_end = params['response_comparison']['time_win'][1]
    alpha = params['response_comparison']['alpha']
    baseline_win = params['baseline_comparison']['win_size']
    smoothing = params['psth']['smoothing_win']

    t1, fr1, _ = agg.get_firing_rate_trace(rec1, unit1, ch1, bin_size=bin_size,
                                           step_size=step_size,
                                           t_start=time_start, t_end=time_end,
                                           baseline_win=baseline_win,
                                           remove_baseline=True)

    t2, fr2, _ = agg.get_firing_rate_trace(rec2, unit2, ch2, bin_size=bin_size,
                                           step_size=step_size,
                                           t_start=time_start, t_end=time_end,
                                           baseline_win=baseline_win,
                                           remove_baseline=True)

    pt1, psth1, _ = agg.get_psth(rec1, unit1, ch1, params, remove_baseline=True)

    pt2, psth2, _ = agg.get_psth(rec2, unit2, ch2, params, remove_baseline=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    # --------------------------------------------------------------------------------
    # Overlayed PSTH plot
    # --------------------------------------------------------------------------------
    mp1 = np.mean(psth1, axis=0)
    sp1 = sem(psth1, axis=0)
    mp2 = np.mean(psth2, axis=0)
    sp2 = sem(psth2, axis=0)
    mp1 = gaussian_filter1d(mp1, sigma=smoothing)  # smooth PSTH
    mp2 = gaussian_filter1d(mp2, sigma=smoothing)  # smooth PSTH
    line1 = ax.plot(pt1, mp1, linewidth=3, label='preCTA')
    ax.fill_between(pt1, mp1 - sp1, mp1 + sp1, alpha=0.4)
    line2 = ax.plot(pt2, mp2, linewidth=3, label='postCTA')
    ax.fill_between(pt2, mp2 - sp2, mp2 + sp2, alpha=0.4)
    ax.axvline(0, linewidth=2, linestyle='--', color='k')
    top = np.max((mp1 + sp1, mp2 + sp2), axis=0)
    sig_y = 1.25 * np.max(top)
    p_y = 1.1 * np.max(top)
    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0], 1.75 * np.max(top)])
    ax.set_xlim([pt1[0], pt1[-1]])
    intervals = []
    int_ps = []
    for t, p in zip(t1, pvals):
        if p > alpha:
            continue

        start = t - bin_size / 2
        end = t + bin_size / 2
        if len(intervals) > 0 and intervals[-1][1] == start:
            intervals[-1][1] = end
            int_ps[-1].append(p)
        else:
            intervals.append([start, end])
            int_ps.append([p])

        ax.plot([start, end], [sig_y, sig_y], linewidth=2, color='k')
        p_str = '%0.3g' % p
        # ax.text(t, p_y, p_str, horizontalalignment='center', fontsize=12)

    for it, ip in zip(intervals, int_ps):
        mid = np.mean(it)
        max_p = np.max(ip)
        if max_p <= 0.001:
            ss = '***'
        elif max_p <= 0.01:
            ss = '**'
        elif max_p <= 0.05:
            ss = '*'
        else:
            continue

        ax.text(mid, sig_y + 0.1, ss, horizontalalignment='center')

    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title('Held Unit %s : %s : %s : %s\nFiring rate relative to '
                 'baseline' % (held_unit_name, exp_name, exp_group, taste))
    ax.legend()

    if save_file:
        fig.savefig(save_file)
        plt.close(fig)
    else:
        return fig, ax


def plot_PSTHs(rec, unit, params, save_file=None, ax=None):
    dat = load_dataset(rec)
    dim = dat.dig_in_mapping.set_index('name')
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))
    else:
        fig = ax.figure

    bin_size = params['psth']['win_size']
    step_size = params['psth']['step_size']
    p_win = params['psth']['plot_window']
    smoothing = params['psth']['smoothing_win']

    rates = []
    labels = []
    time = None
    for taste, row in dim.iterrows():
        ch = row['channel']
        t, fr, _ = agg.get_firing_rate_trace(rec, unit, ch, bin_size,
                                             step_size=step_size,
                                             t_start=p_win[0], t_end=p_win[1])
        if time is None:
            time = t

        rank = agg.PAL_MAP[taste]
        # Ignore Water
        if rank > 0:
            pal = np.ones((fr.shape[0],)) * agg.PAL_MAP[taste]
            rates.append(fr)
            labels.append(pal)

        if taste != 'Water':
            pt, psth, _ = agg.get_psth(rec, unit, ch, params, remove_baseline=False)
            mp = np.mean(psth, axis=0)
            mp = gaussian_filter1d(mp, smoothing)
            sp = sem(psth, axis=0)
            ax.plot(pt, mp, linewidth=3, label=taste)
            ax.fill_between(pt, mp - sp, mp + sp, alpha=0.4)

    # Compute and plot spearman corr R^2
    if len(rates) > 0:
        rates = np.vstack(rates)
        labels = np.concatenate(labels)
        n_bins = len(time)
        s_rs = np.zeros((n_bins,))
        s_ps = np.ones((n_bins,))
        for i, t in enumerate(time):
            if all(rates[:, i] == 0):
                continue
            else:
                response_ranks = rankdata(rates[:, i])
                s_rs[i], s_ps[i] = spearmanr(response_ranks, labels)

        s_rs = s_rs ** 2
        s_rs = gaussian_filter1d(s_rs, smoothing)
        ax2 = ax.twinx()
        ax2.plot(time, s_rs, color='k', linestyle=':', linewidth=3)
        ax2.set_ylabel(r'Spearman $\rho^{2}')

    ax.axvline(0, color='k', linewidth=1, linestyle='--')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_xlim([pt[0], pt[-1]])
    if isinstance(unit, int):
        unit_name = 'unit%03d' % unit
    else:
        unit_name = 'unit'

    ax.set_title('%s %s' % (os.path.basename(rec), unit_name))
    ax.legend()
    if not os.path.isdir(os.path.dirname(save_file)):
        os.mkdir(os.path.dirname(save_file))

    if save_file is not None:
        fig.savefig(save_file)
        plt.close(fig)
        return
    else:
        return fig, ax
