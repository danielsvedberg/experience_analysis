import itertools
import os
import shutil
import pdb
import pandas as pd
import numpy as np
import pyarrow.feather as feather
import pickle
import aggregation as agg
import plotting as plt
import new_plotting as nplt
import analysis_stats as stats
import helper_funcs as hf
import population_analysis as pop
import hmm_analysis as hmma
import statsmodels.formula.api as smf
import patsy
import glob
from scipy.stats import mannwhitneyu, spearmanr, sem, f_oneway, rankdata, pearsonr, ttest_ind
from blechpy import load_project, load_dataset, load_experiment
from blechpy.plotting import data_plot as dplt
from copy import deepcopy
from blechpy.analysis import spike_analysis as sas, poissonHMM as phmm
from blechpy.dio import h5io
from scipy.stats import sem
from blechpy.utils import write_tools as wt
import pylab as pyplt
from datetime import datetime
from tqdm import tqdm
from joblib import parallel_backend
from joblib import Parallel, delayed

# PAL_MAP = {'Water': -1, 'Saccharin': -1, 'Quinine': 1,
#            'Citric Acid': 2, 'NaCl': 3}


PAL_MAP = {'Spont': -1, 'Suc': 1, 'QHCl': 4,
           'CA': 3, 'NaCl': 2}

ELECTRODES_IN_GC = {'DS31': 'both', 'DS33': 'both', 'DS36': 'both', 'DS39': 'both', 'DS40': 'both', 'DS41': 'both',
                    'DS41': 'both', 'DS42': 'both', 'DS44': 'both', 'DS45': 'both', 'DS46': 'both', 'DS47': 'both'}

ANALYSIS_PARAMS = {'taste_responsive': {'win_size': 750, 'alpha': 0.05},
                   'pal_responsive': {'win_size': 250, 'step_size': 25,
                                      'time_win': [-250, 2000], 'alpha': 0.05},
                   'baseline_comparison': {'win_size': 1500, 'alpha': 0.01},
                   'response_comparison': {'win_size': 250, 'step_size': 250,
                                           'time_win': [0, 1500], 'alpha': 0.05,
                                           'n_boot': 10000},
                   'psth': {'win_size': 250, 'step_size': 25, 'smoothing_win': 3,
                            'plot_window': [-1000, 2000]},
                   'pca': {'win_size': 750, 'step_size': 750,
                           'smoothing_win': 3,
                           'plot_window': [-500, 2000], 'time_win': [0, 1500]}}

ELF_DIR = '/data/Katz_Data/Stk11_Project/'
MONO_DIR = '/media/roshan/Gizmo/Katz_Data/Stk11_Project/'
LOCAL_MACHINE = os.uname()[1]
if LOCAL_MACHINE == 'Mononoke':
    DATA_DIR = MONO_DIR
elif LOCAL_MACHINE == 'StealthElf':
    DATA_DIR = ELF_DIR

DATA_DIR = '/media/dsvedberg/T7'


def get_file_dirs(animID):
    anim_dir = os.path.join(DATA_DIR, animID)
    fd = [os.path.join(anim_dir, x) for x in os.listdir(anim_dir)]
    file_dirs = [x for x in fd if os.path.isdir(x)]
    out = []
    for f in file_dirs:
        fl = os.listdir(f)
        if any([x.endswith('.dat') for x in fl]):
            out.append(f)

    return out, anim_dir


def update_params(new, old):
    out = deepcopy(old)
    for k, v in new.items():
        if isinstance(v, dict) and k in out:
            out[k].update(v)
        else:
            out[k] = v

    return out


class ProjectAnalysis(object):
    def __init__(self, proj):
        self.root_dir = os.path.join(proj.root_dir, proj.data_name + '_analysis')
        self.project = proj
        save_dir = os.path.join(self.root_dir, 'single_unit_analysis')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        self.save_dir = save_dir
        self.files = {'all_units': os.path.join(save_dir, 'all_units.feather'),
                      'held_units': os.path.join(save_dir, 'held_units.feather'),
                      'params': os.path.join(save_dir, 'analysis_params.json')}

    def detect_held_units(self, percent_criterion=95, raw_waves=True, overwrite=False):
        save_dir = self.save_dir
        all_units_file = self.files['all_units']
        held_units_file = self.files['held_units']
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        if os.path.isfile(all_units_file) and os.path.isfile(held_units_file) and not overwrite:
            all_units = feather.read_feather(all_units_file)
            held_df = feather.read_feather(held_units_file)
        else:
            all_units, held_df = agg.find_held_units(self.project, percent_criterion, raw_waves)
            feather.write_feather(all_units, all_units_file)
            feather.write_feather(held_df, held_units_file)

            # Plot waveforms and J3 distribution
            plot_dir = os.path.join(save_dir, 'held_unit_waveforms')
            if not os.path.isdir(plot_dir):
                os.makedirs(plot_dir)

            plt.plot_held_units(all_units, plot_dir)
            dplt.plot_J3s(all_units['intra_J3'].dropna().to_numpy(),
                          held_df['inter_J3'].dropna().to_numpy(),
                          save_dir, percent_criterion)

        return all_units, held_df

    def plot_held_units(self):
        save_dir = self.save_dir
        all_units_file = self.files['all_units']
        held_units_file = self.files['held_units']
        if not os.path.isdir(save_dir):
            raise ValueError('Please run get_held_units first')

        if os.path.isfile(all_units_file) and os.path.isfile(held_units_file):
            all_units = feather.read_feather(all_units_file)
            held_df = feather.read_feather(held_units_file)
            # Plot waveforms and J3 distribution
            plot_dir = os.path.join(save_dir, 'held_unit_waveforms')
            if not os.path.isdir(plot_dir):
                os.makedirs(plot_dir)

            plt.plot_held_units(all_units, plot_dir)
        else:
            raise ValueError('Please run get_held_units first')
        return all_units, held_df

    def get_unit_info(self, overwrite=False, change_dirs=False):
        save_dir = self.save_dir
        all_units_file = self.files['all_units']
        held_units_file = self.files['held_units']
        if not os.path.isfile(all_units_file) or not os.path.isfile(held_units_file):
            raise ValueError('Please run get_held_units first')

        all_units = feather.read_feather(all_units_file)
        held_df = feather.read_feather(held_units_file)

        if change_dirs:
            proj = self.project
            rec_info = proj.get_rec_info()
            rec_info = rec_info[['rec_name', 'rec_dir']]
            old_rec_info = all_units[['rec_name', 'rec_dir']]
            #rename rec_dir in old_rec_info to rec_dir_old
            old_rec_info = old_rec_info.rename(columns={'rec_dir': 'old_rec'})
            #merge rec_info with old_rec_info on rec_name
            old_rec_info = pd.merge(rec_info, old_rec_info, on='rec_name', how='left')
            #replace rec_dir in all_units and held_df with rec_dir in rec_info along rec_name
            #first remove rec_dir from all_units and held_df
            all_units = all_units.drop(columns=['rec_dir'])
            all_units = pd.merge(all_units, rec_info, on='rec_name', how='left')

            #drop rec_name from old_rec_info
            old_rec_info = old_rec_info.drop(columns=['rec_name'])
            #create new df from old_rec_info where rec_dir is renamed to rec1
            r1merge = old_rec_info.rename(columns={'rec_dir': 'rec1'})
            #also make one for rec2
            r2merge = old_rec_info.rename(columns={'rec_dir': 'rec2'})
            #rename rec1 column in held_df to old_rec
            held_df = held_df.rename(columns={'rec1': 'old_rec'})
            #merge held_df with r1merge on old_rec
            held_df = pd.merge(held_df, r1merge, on='old_rec', how='left')
            #drop old_rec from held_df
            held_df = held_df.drop(columns=['old_rec'])
            #rename rec2 column in held_df to old_rec
            held_df = held_df.rename(columns={'rec2': 'old_rec'})
            #merge held_df with r2merge on old_rec
            held_df = pd.merge(held_df, r2merge, on='old_rec', how='left')
            #drop old_rec from held_df
            held_df = held_df.drop(columns=['old_rec'])
            self.write_unit_info(all_units=all_units, held_df=held_df)

        if 'time_group' not in all_units.columns or overwrite == True:
            all_units = apply_groups_from_proj(all_units, self.project)
            self.write_unit_info(all_units=all_units)

        if 'time_group' not in held_df.columns or overwrite == True:
            held_df = apply_info_from_all_units(held_df, all_units)
            self.write_unit_info(held_df=held_df)

        if 'area' not in held_df.columns:
            for i, row in held_df.iterrows():
                r1 = row['rec1']
                u1 = row['unit1']
                held_df.loc[i, 'area'] = all_units.query('rec_dir == @r1 and unit_name == @u1').iloc[0]['area']
            self.write_unit_info(held_df=held_df)

        if 'exclude' not in all_units.columns:
            all_units = agg.apply_grouping_cols(all_units, self.project)
            self.write_unit_info(all_units=all_units)

        if 'response_firing' not in all_units.columns:
            all_units = apply_unit_firing_rates(all_units)
            self.write_unit_info(all_units=all_units)

        if 'unit_type' not in all_units.columns:
            def foo(x):
                if not x['single_unit']:
                    return 'multi'
                elif x['regular_spiking']:
                    return 'pyramidal'
                else:
                    return 'interneuron'

            all_units['unit_type'] = all_units.apply(foo, axis=1)
            self.write_unit_info(all_units=all_units)

        return all_units, held_df

    def write_unit_info(self, all_units=None, held_df=None):
        save_dir = self.save_dir
        all_units_file = self.files['all_units']
        held_units_file = self.files['held_units']
        if all_units is not None:
            feather.write_feather(all_units, all_units_file)

        if held_df is not None:
            feather.write_feather(held_df, held_units_file)

    def get_params(self, params=None):
        params_file = self.files['params']
        if os.path.isfile(params_file):
            base_params = wt.read_dict_from_json(params_file)
        else:
            base_params = deepcopy(ANALYSIS_PARAMS)

        if params is not None:
            params = update_params(params, base_params)
        else:
            params = base_params

        wt.write_dict_to_json(params, params_file)
        return params

    def analyze_response_changes(self, params=None, overwrite=False):
        all_units, held_df = self.get_unit_info()
        save_dir = os.path.join(self.save_dir, 'held_unit_response_changes')
        save_file = os.path.join(save_dir, 'response_change_data.npz')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        params = self.get_params(params)

        if os.path.isfile(save_file) and not overwrite:
            data = np.load(save_file)
            return data
        else:
            # rec_map = {'preCTA': 'postCTA', 'ctaTrain': 'ctaTest'}
            all_units = all_units.dropna(subset=['held_unit_name'])
            all_units = all_units[all_units['area'] == 'GC']
            # learn_map = self.project._exp_info.set_index('exp_name')['CTA_learned']
            # learn_map = learn_map.apply(lambda x: 'CTA' if x else 'No CTA').to_dict()

            # Output structures
            alpha = params['response_comparison']['alpha']
            labels = []  # List of tuples: (exp_group, exp_name, cta_group, held_unit_name, taste)
            pvals = []
            differences = []
            sem_diffs = []
            test_stats = []
            comp_time = None
            diff_time = None

            for held_unit_name, group in all_units.groupby('held_unit_name'):
                if any(group.rec_num == 1):
                    pre_grp = group.query('rec_num == 1')
                    for unit_id1, row in pre_grp.iterrows():
                        post_row = group.loc[group.time_group != "1"]
                        if post_row.empty:
                            continue
                        else:
                            unit_id2 = post_row.index[0]
                            post_row = post_row.to_dict(orient='records')[0]

                        rec1 = row['rec_dir']
                        rec2 = post_row['rec_dir']
                        unit1 = row['unit_num']
                        unit2 = post_row['unit_num']
                        exp_group = row['exp_group']
                        exp_name = row['exp_name']
                        print('Comparing Held Unit %s, %s vs %s' % (
                            held_unit_name, row['rec_name'], post_row['rec_name']))

                        tastes, pvs, tstat, md, md_sem, ctime, dtime = \
                            compare_taste_responses(rec1, unit1, rec2, unit2,
                                                    params, method='anova')
                        l = [(exp_group, exp_name,
                              held_unit_name, t) for t in tastes]
                        labels.extend(l)
                        pvals.extend(pvs)
                        test_stats.extend(tstat)
                        differences.extend(md)
                        sem_diffs.extend(md_sem)
                        if comp_time is None:
                            comp_time = ctime
                        elif not np.array_equal(ctime, comp_time):
                            raise ValueError('Times dont match')

                        if diff_time is None:
                            diff_time = dtime
                        elif not np.array_equal(dtime, diff_time):
                            raise ValueError('Times dont match')

                        for i, tst in enumerate(tastes):
                            plot_dir = os.path.join(save_dir, 'Held_Unit_Plots', tst)
                            new_plot_dir = os.path.join(plot_dir, 'clean_plots')
                            if not os.path.isdir(plot_dir):
                                os.makedirs(plot_dir)

                            if not os.path.isdir(new_plot_dir):
                                os.makedirs(new_plot_dir)

                            fig_file = os.path.join(plot_dir, 'Held_Unit_%s-%s.svg' % (held_unit_name, tst))
                            new_fn = os.path.join(new_plot_dir,
                                                  'Held_Unit_%s-%s.svg' %
                                                  (held_unit_name, tst))

                            plt.plot_held_unit_comparison(rec1, unit1, rec2, unit2,
                                                          pvs[i], params, held_unit_name,
                                                          exp_name, exp_group, tst,
                                                          save_file=fig_file)
                            nplt.plot_held_unit_comparison(rec1, unit1, rec2, unit2,
                                                           pvs[i], params, held_unit_name,
                                                           exp_name, exp_group, tst,
                                                           save_file=new_fn)

            labels = np.vstack(labels)  # exp_group, exp_name, cta_group, held_unit_name, taste
            pvals = np.vstack(pvals)
            test_stats = np.vstack(test_stats)
            differences = np.vstack(differences)
            sem_diffs = np.vstack(sem_diffs)

            # Stick it all in an npz file
            np.savez(save_file, labels=labels, pvals=pvals, test_stats=test_stats,
                     mean_diff=differences, sem_diff=sem_diffs,
                     comp_time=comp_time, diff_time=diff_time)
            data = np.load(save_file)
            return data

    def make_aggregate_held_unit_plots(self):
        save_dir = os.path.join(self.root_dir, 'single_unit_analysis',
                                'held_unit_response_changes')
        save_file = os.path.join(save_dir, 'response_change_data.npz')
        params_file = os.path.join(save_dir, 'analysis_params.json')

        params = self.get_params()

        if os.path.isfile(save_file):
            data = np.load(save_file)
        else:
            raise FileNotFoundError('No data file found')

        alpha = params['response_comparison']['alpha']
        n_boot = params['response_comparison']['n_boot']
        labels = data['labels']  # exp_group, exp_name, held_unit_name, taste
        pvals = data['pvals']
        comp_time = data['comp_time']
        mean_diff = data['mean_diff']
        sem_diff = data['sem_diff']
        diff_time = data['diff_time']
        tastes = np.unique(labels[:, -1])
        all_df = hf.make_tidy_response_change_data(labels, pvals, comp_time, alpha=alpha)
        all_df.to_csv(os.path.join(save_dir,
                                   'held_unit_response_change_data.csv'),
                      index=False)

        # Make aggregate plots
        plot_dir = os.path.join(save_dir, 'Held_Unit_Plots')
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)

        heatmap_file = os.path.join(plot_dir, 'Held_unit_response_changes.svg')
        plt.plot_mean_differences_heatmap(labels, diff_time, mean_diff,
                                          save_file=heatmap_file, t_start=0)
        for tst in tastes:
            df = all_df[all_df['taste'] == tst]
            idx = np.where(labels[:, -1] == tst)[0]
            l = labels[idx, :]
            p = pvals[idx, :]
            md = mean_diff[idx, :]
            sd = sem_diff[idx, :]

            hf.compare_response_changes(df, tst, plot_dir, save_dir,
                                        f'{tst}_responses_changed',
                                        group_col='exp_group', alpha=alpha,
                                        exp_group='Cre', ctrl_group='GFP')

            hf.compare_response_changes(df, tst, plot_dir, save_dir,
                                        f'{tst}_responses_changed-CTA_groups',
                                        group_col='cta_group', alpha=alpha,
                                        exp_group='No CTA', ctrl_group='CTA')

            sub_df = df.query('(exp_group == "Cre" & cta_group == "No CTA") '
                              '| (exp_group == "GFP" & cta_group == "CTA")')
            hf.compare_response_changes(sub_df, tst, plot_dir, save_dir,
                                        f'{tst}_responses_changed-exclude',
                                        group_col='exp_group', alpha=alpha,
                                        exp_group='Cre', ctrl_group='GFP')

            sub_df = df.query('(exp_group == "Cre" & cta_group == "No CTA") '
                              '| (exp_group == "GFP" & cta_group == "No CTA")')
            hf.compare_response_changes(sub_df, tst, plot_dir, save_dir,
                                        f'{tst}_responses_changed-Cre_v_BadGFP',
                                        group_col='exp_group', alpha=alpha,
                                        exp_group='Cre', ctrl_group='GFP')

            sub_df = df.query('(exp_group == "GFP" & cta_group == "No CTA") '
                              '| (exp_group == "GFP" & cta_group == "CTA")')
            hf.compare_response_changes(sub_df, tst, plot_dir, save_dir,
                                        f'{tst}_responses_changed-GFP_v_GFP',
                                        group_col='cta_group', alpha=alpha,
                                        exp_group='No CTA', ctrl_group='CTA')

            tmp_data = (p <= alpha).astype('int')
            # Compare Cre vs GFP
            save_file = os.path.join(plot_dir, '%s_responses_changed-old.svg' % tst)
            plt.plot_held_percent_changed(l, comp_time, p, diff_time, md, sd,
                                          alpha, tst, save_file=save_file)

            # Compare CTA vs No CTA
            learn_file = os.path.join(plot_dir,
                                      '%s_responses_changed-CTA_groups-old.svg'
                                      % tst)
            plt.plot_held_percent_changed(l, comp_time, p, diff_time, md, sd,
                                          alpha, tst,
                                          group_col=2, save_file=learn_file)

            anim_dir = os.path.join(plot_dir, 'Per_Animal', tst)
            if not os.path.isdir(anim_dir):
                os.makedirs(anim_dir)

            animals = np.unique(l[:, 1])
            for anim in animals:
                fn = os.path.join(anim_dir, '%s_%s_responses_changed.svg' % (anim, tst))
                a_idx = np.where(l[:, 1] == anim)[0]
                a_l = l[a_idx, :]
                a_p = p[a_idx, :]
                a_md = md[a_idx, :]
                a_sd = sd[a_idx, :]
                plt.plot_held_percent_changed(a_l, comp_time, a_p, diff_time,
                                              a_md, a_sd, alpha, anim + ': ' + tst,
                                              save_file=fn)

        return

    def process_single_units(self, params=None, overwrite=False):
        save_dir = os.path.join(self.save_dir, 'single_unit_responses')
        pal_file = os.path.join(save_dir, 'palatability_data.npz')
        resp_file = os.path.join(save_dir, 'taste_responsive_pvals.npz')
        tasty_unit_file = os.path.join(save_dir, 'unit_taste_responsivity.feather')
        pal_unit_file = os.path.join(save_dir, 'unit_pal_discrim.feather')
        params = self.get_params(params)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        if os.path.isfile(tasty_unit_file) and os.path.isfile(pal_unit_file) and not overwrite:
            resp_units = feather.read_feather(tasty_unit_file)
            pal_units = feather.read_feather(pal_unit_file)
            return resp_units, pal_units

        if overwrite and os.path.isfile(pal_file):
            os.remove(pal_file)

        if overwrite and os.path.isfile(resp_file):
            os.remove(resp_file)

        fn = os.path.join(save_dir, 'unit_firing_rates.svg')
        [all_units, held_df] = self.get_unit_info()
        nplt.plot_unit_firing_rates(all_units, save_file=fn)

        all_units = all_units[all_units['single_unit']]
        resp_units = all_units.groupby('rec_dir', group_keys=False).apply(apply_tastes)
        print('-' * 80)
        print('Processing taste resposiveness')
        print('-' * 80)
        resp_units = resp_units.apply(lambda x: apply_taste_responsive(x, params, resp_file), axis=1)
        resp_units['exclude'] = resp_units.apply(agg.apply_exclude, axis=1)
        feather.write_feather(resp_units, tasty_unit_file)
        fn = os.path.join(save_dir, 'taste_responsive.svg')
        df = resp_units.query('exclude == False and taste != "Spont"')
        nplt.plot_taste_responsive_units(df, fn)

        pal_units = all_units  # [all_units.rec_name.str.contains('4taste')].copy()
        plot_dir = os.path.join(save_dir, 'single_unit_plots')
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)

        def foo(x):
            return apply_discrim_and_pal(x, params, pal_file, plot_dir)

        print('-' * 80)
        print('Processing taste discrimination and palatability')
        print('-' * 80)
        pal_units = pal_units.apply(foo, axis=1)  # causes problem
        pal_units['taste_discriminative'] = pal_units['taste_discriminative'].astype('bool')
        pal_units['exclude'] = pal_units.apply(agg.apply_exclude, axis=1)
        feather.write_feather(pal_units, pal_unit_file)
        df = pal_units[pal_units['exclude'] == False]
        nplt.plot_pal_responsive_units(df, save_dir)

        return resp_units, pal_units

    def make_aggregate_single_unit_plots(self):
        resp_units, pal_units = self.process_single_units()
        save_dir = os.path.join(self.save_dir, 'single_unit_responses')
        resp_file = os.path.join(save_dir, 'taste_responsive.svg')
        discrim_file = os.path.join(save_dir, 'taste_discriminative.svg')
        spearman_file = os.path.join(save_dir, 'palatability_spearman.svg')
        pearson_file = os.path.join(save_dir, 'palatability_pearson.svg')
        params = self.get_params()
        # For responsive, plot
        if 'time_group' not in resp_units.columns or 'time_group' not in pal_units.columns:
            time_map = {'preCTA': 'preCTA', 'ctaTrain': 'preCTA', 'ctaTest':
                'postCTA', 'postCTA': 'postCTA'}
            resp_units['time_group'] = resp_units.rec_group.map(time_map)
            pal_units['time_group'] = pal_units.rec_group.map(time_map)

        tmp_grp = resp_units.groupby(['exp_group', 'time_group', 'taste'])['taste_responsive']
        resp_df = tmp_grp.apply(lambda x: 100 * np.sum(x) / len(x)).reset_index()
        # plt.plot_taste_responsive(resp_df, resp_file)
        # plt.plot_taste_discriminative(pal_units, discrim_file)
        plt.plot_aggregate_spearman(pal_units, spearman_file)
        plt.plot_aggregate_pearson(pal_units, pearson_file)

        spear_mean = os.path.join(save_dir, 'Mean_Spearman.svg')
        pear_mean = os.path.join(save_dir, 'Mean_Pearson.svg')
        resp_time = os.path.join(save_dir, 'Taste_responsive_over_time.svg')
        resp_data = os.path.join(save_dir, 'taste_responsive_pvals.npz')
        pal_data = os.path.join(save_dir, 'palatability_data.npz')
        # plt.plot_mean_spearman(pal_data, spear_mean)
        nplt.plot_mean_spearman_correlation(pal_data, self.project, spear_mean)
        plt.plot_mean_pearson(pal_data, pear_mean)
        alpha = params['taste_responsive']['alpha']
        plt.plot_taste_response_over_time(resp_data, resp_time, alpha)

    def pca_analysis(self, overwrite=False):
        '''Grab units held across pre OR post. For each animal do pca on firing
        rate traces, then plot for individual traces and mean trace for each
        taste. 1 plot per animal, pre & post subplot
        '''
        save_dir = os.path.join(self.save_dir, 'pca_analysis')
        pc_data_file = os.path.join(save_dir, 'pc_data.feather')
        dist_data_file = os.path.join(save_dir, 'pc_dist_data.feather')
        other_dist_file = os.path.join(save_dir, 'pc_dQ_v_dN_data.feather')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        if os.path.isfile(pc_data_file) and os.path.isfile(dist_data_file) and os.path.isfile(
                other_dist_file) and not overwrite:
            pc_data = feather.read_feather(pc_data_file)
            dist_data = feather.read_feather(dist_data_file)
            other_dist_data = feather.read_feather(other_dist_file)
            return pc_data, dist_data, other_dist_data

        params = self.get_params()
        all_units, held_units = self.get_unit_info()
        all_units = all_units.dropna(subset=['held_unit_name'])
        all_units = all_units.query('area == "GC"')
        unit_names = all_units.held_unit_name.unique()
        held_units = held_units.dropna(subset=['held_unit_name'])
        held_units = held_units[held_units['held_unit_name'].isin(unit_names)]
        held_units = held_units.apply(apply_info_from_all_units, axis=1)
        if 'exp_group' not in held_units.columns:
            exp_map = self.project._exp_info.set_index('exp_name')['exp_group'].to_dict()
            held_units['exp_group'] = held_units.exp_name.map(exp_map)

        held_units = held_units.dropna(subset=['time_group'])
        unit_names = held_units['held_unit_name'].unique()
        all_units = all_units[all_units.held_unit_name.isin(unit_names)]
        # Now all_units and held_units have only units that are held over one
        # half of the experiment and are in GC
        grp = held_units.groupby(['exp_name', 'exp_group', 'time_group'])
        pc_data = grp.apply(lambda x: pop.apply_pca_analysis(x, params)).reset_index().drop(columns=['level_3'])
        pc_data['time'] = pc_data['time'].astype('int')
        pc_data['time'] = pc_data.time.apply(lambda x: 'Early (0-750ms)' if x <= 750 else 'Late (750-1500ms)')
        grp = pc_data.groupby(['exp_name', 'exp_group', 'time_group', 'time'])
        dist_data = grp.apply(pop.apply_pc_distances).reset_index()
        pc_dist_metrics = grp.apply(pop.apply_pc_dist_metric).reset_index(drop=True)
        mds_dist_metrics = grp.apply(pop.apply_mds_dist_metric).reset_index(drop=True)
        dist_metrics = pd.merge(pc_dist_metrics, mds_dist_metrics,
                                on=['exp_name', 'exp_group', 'time_group',
                                    'time', 'taste', 'trial', 'n_cells', 'PC1',
                                    'PC2', 'MDS1', 'MDS2'])
        dist_metrics = agg.apply_grouping_cols(dist_metrics, self.project)
        pc_data = agg.apply_grouping_cols(pc_data, self.project)
        feather.write_feather(pc_data, pc_data_file)
        feather.write_feather(dist_data, dist_data_file)
        feather.write_feather(dist_metrics, other_dist_file)
        return pc_data, dist_data, dist_metrics

    def plot_pca_data(self):
        pc_data, dist_data, metric_data = self.pca_analysis()
        save_dir = os.path.join(self.save_dir, 'pca_analysis')
        mds_dir = os.path.join(self.save_dir, 'mds_analysis')
        if not os.path.isdir(mds_dir):
            os.mkdir(mds_dir)

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        plt.plot_pca_distances(dist_data, os.path.join(save_dir, 'distances'))
        plt.plot_pca_metric(metric_data, os.path.join(save_dir, 'relative_PCA_distances.svg'))
        plt.plot_mds_metric(metric_data, os.path.join(mds_dir, 'relative_MDS_distances.svg'))
        plt.plot_animal_pca(pc_data, os.path.join(save_dir, 'animal_pca'))
        plt.plot_animal_mds(pc_data, os.path.join(mds_dir, 'animal_mds'))

        metric_data = agg.apply_grouping_cols(metric_data, self.project)
        metric_data = metric_data.query('exclude == False')
        fn = os.path.join(mds_dir, 'Saccharin_MDS_distances.svg')
        nplt.plot_MDS(metric_data, save_file=fn)
        fn = os.path.join(mds_dir, 'Saccharin_MDS_distances-alternate.svg')
        nplt.plot_MDS(metric_data, save_file=fn, ylabel='dQ-dN',
                      value_col='MDS_dQ_minus_dN', kind='bar')

        fn = os.path.join(mds_dir, 'FullDim_MDS_distances.svg')
        nplt.plot_full_dim_MDS(pc_data, save_file=fn)

        # Change exp group to CTA learning and re-plot
        # learn_map = self.project._exp_info.set_index('exp_name')
        # def foo(x):
        #     if learn_map.loc[x]['CTA_learned']:
        #         return 'CTA'
        #     else:
        #         return 'No CTA'

        # metric_data['exp_group'] = metric_data['exp_name'].apply(foo)
        plt.plot_pca_metric(metric_data, os.path.join(save_dir, 'relative_PCA_distances-CTA.svg'))
        plt.plot_mds_metric(metric_data, os.path.join(mds_dir, 'relative_MDS_distances-CTA.svg'))

    def fix_palatability(self):
        agg.fix_palatability(self.project, pal_map=agg.PAL_MAP)

    def fix_areas(self):
        agg.set_electrode_areas(self.project, el_in_gc=ELECTRODES_IN_GC)

    def plot_saccharin_consumption(self):
        save_dir = self.save_dir
        fn = os.path.join(save_dir, 'Saccharin_consumption.svg')
        nplt.plot_saccharin_consumption(self.project, fn)

    def run(self, overwrite=False):
        # self.fix_areas()
        # self.fix_palatability()
        # self.detect_held_units(overwrite=overwrite, raw_waves=True)
        self.analyze_response_changes(overwrite=overwrite)
        self.make_aggregate_held_unit_plots()
        self.process_single_units(overwrite=overwrite)
        self.make_aggregate_single_unit_plots()
        self.pca_analysis(overwrite=overwrite)
        self.plot_pca_data()
        # self.plot_saccharin_consumption()


def apply_info_from_rec_dir(row):
    rd1 = row['rec1']
    rd2 = row['rec2']

    try:
        row['held_over'] = 'd_' + rd1['rec_num'] + '_' + rd2['rec_num']
        row['time_group'] = None
    except:
        raise ValueError('Doesnt fit into group')

    rec = os.path.basename(rd1).split('_')
    row['exp_name'] = rec[0]
    row['rec_group'] = 'None'
    return row


def apply_groups_from_proj(df, proj):
    cols = ['exp_name', 'exp_group', 'rec_num', 'rec_group', 'rec_day', 'time_group']
    for i in cols:
        try:
            df.pop(i)
        except:
            pass
    rec_info = proj.get_rec_info()
    sel = rec_info[['rec_dir', 'exp_name', 'exp_group', 'rec_num']]
    sel['time_group'] = sel['rec_num'].astype('str')
    sel['rec_group'] = sel['exp_group'] + '_' + sel['time_group']

    df = df.merge(sel, how='inner', on=['rec_dir'])
    df = df.drop_duplicates()
    return df


# TODO: fix merge so it can handle the overwrite or whatever
def apply_info_from_all_units(held_df, all_units):
    try:
        sel = all_units[['rec_dir', 'rec_num']]
        sel1 = sel.rename(columns={'rec_dir': 'rec1', 'rec_num': 'r1num'})
        sel1 = sel1.drop_duplicates()
        sel2 = sel.rename(columns={'rec_dir': 'rec2', 'rec_num': 'r2num'})
        sel2 = sel2.drop_duplicates()
        try:
            held_df.pop('r1num')
            held_df.pop('r2num')
        except:
            pass
        held_df = pd.merge(held_df, sel1, how="inner", on=['rec1'])
        held_df = pd.merge(held_df, sel2, how="inner", on=['rec2'])
        held_df['held_over'] = 'd_' + held_df['r1num'].astype(str) + '_' + held_df['r2num'].astype(str)

        held_df['time_group'] = None

        # rd1 = row['rec1']
        # rd2 = row['rec2']
        # rn1 = all_units['rec_num'].loc[all_units.rec_dir == rd1]
        # rn2 = all_units['rec_num'].loc[all_units.rec_dir == rd2]
    except:
        raise ValueError('Doesnt fit into group')

    return held_df


def apply_discrim_and_pal(row, params, save_file, plot_dir):
    d_win = params['taste_responsive']['win_size']
    d_alpha = params['taste_responsive']['alpha']
    p_bin_size = params['pal_responsive']['win_size']
    p_step = params['pal_responsive']['step_size']
    p_alpha = params['pal_responsive']['alpha']
    p_win = params['pal_responsive']['time_win']

    rec = row['rec_dir']
    rec_name = row['rec_name']
    unit = row['unit_num']
    unit_name = row['unit_name']
    exp_group = row['exp_group']
    time_group = row['time_group']
    print('Analyzing %s %s...' % (rec_name, unit_name))

    # Check taste dicriminability
    # Grab taste responses for each taste
    dat = load_dataset(rec)
    dim = dat.dig_in_mapping.set_index('channel')
    dim = dim[dim.exclude == False]
    channels = dim.index.tolist()
    tastes = dim.name.tolist()
    responses = []
    for ch in channels:
        t, fr, _ = agg.get_firing_rate_trace(rec, unit, ch, d_win, t_start=0,
                                             t_end=d_win)
        responses.append(fr)

    if len(responses) > 1:
        f, p = f_oneway(*responses)
        row['taste_discriminative'] = p <= d_alpha
        row['discrim_p'] = p
        row['discrim_f'] = f
    else:
        row['taste_discriminative'] = False
        row['discrim_p'] = np.NaN
        row['discrim_f'] = np.NaN

    # Check palatability
    responses = []
    palatability = []
    time = None
    for ch in channels:
        rank = dim.loc[ch, 'palatability_rank']
        if rank <= 0:
            continue

        t, fr, _ = agg.get_firing_rate_trace(rec, unit, ch, p_bin_size,
                                             step_size=p_step,
                                             t_start=p_win[0], t_end=p_win[1])
        if time is None:
            time = t
        elif not np.array_equal(time, t):
            raise ValueError('Time vectors dont match')

        pal = np.ones((fr.shape[0],)) * rank
        responses.append(fr)
        palatability.append(pal)

    if len(responses) < 3:
        row['spearman_r'] = np.NaN
        row['spearman_p'] = np.NaN
        row['spearman_peak'] = np.NaN
        row['pearson_r'] = np.NaN
        row['pearson_p'] = np.NaN
        row['pearson_peak'] = np.NaN
    else:
        responses = np.vstack(responses)
        palatability = np.concatenate(palatability)
        n_bins = len(time)
        s_rs = np.zeros((n_bins,))
        s_ps = np.ones((n_bins,))
        p_rs = np.zeros((n_bins,))
        p_ps = np.ones((n_bins,))
        for i, t in enumerate(time):
            if all(responses[:, i] == 0):
                continue
            else:
                response_ranks = rankdata(responses[:, i])
                s_rs[i], s_ps[i] = spearmanr(response_ranks, palatability)
                p_rs[i], p_ps[i] = pearsonr(responses[:, i], palatability)

        sidx = np.where(s_ps <= p_alpha)[0]
        pidx = np.where(p_ps <= p_alpha)[0]
        if len(sidx) == 0:
            sidx = np.arange(0, n_bins)

        if len(pidx) == 0:
            pidx = np.arange(0, n_bins)

        smax = np.argmax(np.abs(s_rs[sidx]))
        smax = sidx[smax]
        pmax = np.argmax(np.abs(p_rs[pidx]))
        pmax = pidx[pmax]

        row['spearman_r'] = s_rs[smax]
        row['spearman_p'] = p_rs[smax]
        row['spearman_peak'] = time[smax]
        row['pearson_r'] = p_rs[pmax]
        row['pearson_p'] = p_ps[pmax]
        row['pearson_peak'] = time[pmax]

        # Save data array
        label = (exp_group, time_group, rec, unit)
        if not os.path.isfile(save_file):
            np.savez(save_file, labels=np.array([label]), time=time,
                     spearman_r=s_rs, spearman_p=s_ps, pearson_r=p_rs,
                     pearson_p=p_ps)
        else:
            data = np.load(save_file)
            labels = np.vstack((data['labels'], label))
            if not np.array_equal(time, data['time']):
                raise ValueError('Time doesnt match')

            spearman_r = np.vstack((data['spearman_r'], s_rs))
            spearman_p = np.vstack((data['spearman_p'], s_ps))
            pearson_r = np.vstack((data['pearson_r'], p_rs))
            pearson_p = np.vstack((data['pearson_p'], p_ps))
            np.savez(save_file, labels=labels, time=time,
                     spearman_r=spearman_r, spearman_p=spearman_p, pearson_r=pearson_r,
                     pearson_p=pearson_p)

        # Plot PSTHs
        # Plot spearman r & p
        # Plot pearson r & p
        psth_fn = '%s_%s_psth.svg' % (rec_name, unit_name)
        psth_file = os.path.join(plot_dir, 'PSTHs', psth_fn)
        corr_file = os.path.join(plot_dir, 'Palatability', psth_fn.replace('psth', 'corr'))
        nplt.plot_PSTHs(rec, unit, params, save_file=psth_file)  # causes problem
        plt.plot_palatability_correlation(rec_name, unit_name, time, s_rs, s_ps,
                                          p_rs, p_ps, corr_file)

    return row


def apply_taste_responsive(row, params, data_file):
    bin_size = params['taste_responsive']['win_size']
    alpha = params['taste_responsive']['alpha']
    rec = row['rec_dir']
    dat = load_dataset(rec)
    dim = dat.dig_in_mapping.set_index('name')
    taste = row['taste']
    ch = dim.loc[taste, 'channel']
    unit = row['unit_num']
    unit_name = row['unit_name']
    print('Analyzing %s %s...' % (row['rec_name'], unit_name))

    t, fr, _ = agg.get_firing_rate_trace(rec, unit, ch, bin_size,
                                         t_start=-bin_size, t_end=bin_size)
    baseline = fr[:, 0]
    response = fr[:, 1]
    if all(baseline == 0) and all(response == 0):
        f = 0
        p = 1
    else:
        f, p = ttest_ind(baseline, response)

    row['taste_responsive'] = (p <= alpha)
    row['reponse_p'] = p
    row['response_f'] = f

    # Break it up by time and save array
    # Use one way anova and dunnett's post hoc to compare all time bins to baseline
    bin_size = params['response_comparison']['win_size']
    step_size = params['response_comparison']['step_size']
    t_end = params['response_comparison']['time_win'][1]
    time, fr, _ = agg.get_firing_rate_trace(rec, unit, ch, bin_size,
                                            t_start=-bin_size, t_end=t_end)
    f, p = f_oneway(*fr.T)
    if p > alpha:
        return row

    baseline = fr[:, 0]
    fr = fr[:, 1:]
    fr = [fr[:, i] for i in range(fr.shape[1])]
    time = time[1:]
    n_bins = len(time)
    # Now use Dunnett's to compare each time bin to baseline
    CIs, pvals = stats.dunnetts_post_hoc(baseline, fr, alpha)
    # Open npz file and append to arrays
    pvals = np.array(pvals)
    this_label = (row['exp_group'], row['rec_dir'], row['unit_num'],
                  row['time_group'], row['taste'])

    if os.path.isfile(data_file):
        data = np.load(data_file, allow_pickle=True)
        labels = data['labels']
        PV = data['pvals']
        labels = np.vstack((labels, this_label))
        PV = np.vstack((PV, pvals))
        if not np.array_equal(data['time'], time):
            raise ValueError('Time vectors dont match')

    else:
        labels = np.array(this_label)
        PV = pvals

    np.savez(data_file, labels=labels, pvals=PV, time=time)
    return row


def _deprecated_compare_taste_responses(rec1, unit1, rec2, unit2, params):
    bin_size = params['response_comparison']['win_size']
    step_size = params['response_comparison']['step_size']
    time_start = params['response_comparison']['time_win'][0]
    time_end = params['response_comparison']['time_win'][1]
    baseline_win = params['baseline_comparison']['win_size']

    dat1 = load_dataset(rec1)
    dat2 = load_dataset(rec2)

    dig1 = dat1.dig_in_mapping.copy().set_index('name')
    dig2 = dat2.dig_in_mapping.copy().set_index('name')
    out_labels = []
    out_pvals = []
    out_ustats = []
    out_diff = []
    out_diff_sem = []
    bin_time = None
    for taste, row in dig1.iterrows():
        ch1 = row['channel']
        ch2 = dig2.loc[taste, 'channel']
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

        if bin_time is None:
            bin_time = t1

        if not np.array_equal(bin_time, t1) or not np.array_equal(t1, t2):
            raise ValueError('Unqueal time vectors')

        nt = len(t1)
        pvals = np.ones((nt,))
        ustats = np.zeros((nt,))
        for i, y in enumerate(zip(fr1.T, fr2.T)):
            # u, p = mann_whitney_u(y[0], y[1])
            # Mann-Whitney U gave odd results, trying anova
            u, p = f_oneway(y[0], y[1])
            pvals[i] = p
            ustats[i] = u

        # Apply bonferroni correction
        pvals = pvals * nt

        # Compute mean difference using psth parameters
        pt1, psth1, baseline1 = agg.get_psth(rec1, unit1, ch1, params)
        pt2, psth2, baseline2 = agg.get_psth(rec2, unit2, ch2, params)
        diff, sem_diff = sas.get_mean_difference(psth1, psth2)
        # Mag diff plot looked odd, trying using same binning as comparison
        # diff, sem_diff = sas.get_mean_difference(fr1, fr2)
        # ^This looked worse, going back

        # Store stuff
        out_pvals.append(pvals)
        out_ustats.append(ustats)
        out_diff.append(diff)
        out_diff_sem.append(sem_diff)
        out_labels.append(taste)

    return out_labels, out_pvals, out_ustats, out_diff, out_diff_sem, bin_time, pt1


def mann_whitney_u(resp1, resp2):
    try:
        u, p = mannwhitneyu(resp1, resp2, alternative='two-sided')
    except ValueError:
        u = 0
        p = 1

    return u, p


def apply_tastes(rec_group):
    rec_dir = rec_group.rec_dir.unique()[0]
    dat = load_dataset(rec_dir)
    tastes = dat.dig_in_mapping.name.tolist()
    tmp = rec_group.to_dict(orient='records')
    out = []
    for t in tastes:
        for item in tmp:
            j = item.copy()
            j['taste'] = t
            out.append(j)

    return pd.DataFrame(out)


class HmmAnalysis(object):
    def __init__(self, proj):
        self.root_dir = os.path.join(proj.root_dir, proj.data_name + '_analysis')
        self.project = proj
        save_dir = os.path.join(self.root_dir, 'hmm_analysis')
        self.save_dir = save_dir
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        self.files = {'params': os.path.join(save_dir, 'hmm_params.json'),
                      'hmm_overview': os.path.join(save_dir, 'hmm_overview.feather'),
                      'sorted_hmms': os.path.join(save_dir, 'sorted_hmms.feather'),
                      'best_hmms': os.path.join(save_dir, 'best_hmms.feather'),
                      'hmm_coding': os.path.join(save_dir, 'hmm_coding.feather'),
                      'hmm_confusion': os.path.join(save_dir, 'hmm_confusion.feather'),
                      'hmm_timings': os.path.join(save_dir, 'hmm_timings.feather'),
                      'pr_mode_state': os.path.join(save_dir, 'pr_mode_state.feather')}

        self.base_params = {'unit_type': 'single', 'dt': 0.001,
                            'max_iter': 500, 'n_repeats': 20, 'time_start': -500,
                            'time_end': 2500, 'n_states': 4, 'area': 'GC',
                            'hmm_class': 'PoissonHMM', 'threshold': 1e-15}
        # base params changed 8/4/20: max_iter 1000 -> 500, time_end 2000 -> 1500, n_states=2
        # changed 8/6/20: n_states->3
        # Changed 8/10/20: time_end-> 1750 & n_states -> 2

    def fit(self):
        params = self.base_params
        # params = [{'n_states': i+2, **tmp.copy()} for i in range(2)]
        save_file = self.files['hmm_overview']
        fit_df = None
        for i, row in self.project._exp_info.iterrows():
            exp = load_experiment(row['exp_dir'])
            for rec_dir in exp.recording_dirs:
                units = phmm.query_units(rec_dir, params['unit_type'], area=params['area'])
                dat = load_dataset(rec_dir)
                if len(units) < 3:
                    continue
                else:
                    handler = phmm.HmmHandler(rec_dir)
                    handler.add_params(params)
                    handler.run()
                    df = handler.get_data_overview().copy()
                    df['rec_dir'] = rec_dir
                    if fit_df is None:
                        fit_df = df
                    else:
                        fit_df = fit_df.append(df, ignore_index=True)

                feather.write_feather(fit_df, save_file)
                pyplt.close('all')

    def check_hmm_fitting(self):
        df = self.get_sorted_hmms()
        if df is None:
            return None

        return df.groupby(['rec_dir', 'taste'])['sorting'].apply(lambda x: any(x == 'best'))

    def refit_rejected(self, common_log=None):
        base_params = self.base_params
        sorted_df = self.get_sorted_hmms()
        PA = ProjectAnalysis(self.project)
        all_units, held_units = PA.get_unit_info()
        refit_hmms(sorted_df, base_params, all_units, log_file=common_log)
        # ho = self.get_hmm_overview(overwrite=True)
        return

    def get_hmm_overview(self, overwrite=False):
        if not os.path.isfile(self.files['hmm_overview']):
            overwrite = True

        if not overwrite:
            ho = feather.read_feather(self.files['hmm_overview'])
        else:
            ho = None
            print('aggregating hmm data...')

            for i, row in self.project._exp_info.iterrows():
                exp = load_experiment(row['exp_dir'])
                for rec_dir in exp.recording_dirs:
                    print('processing %s' % os.path.basename(rec_dir))
                    h5_file = hmma.get_hmm_h5(rec_dir)
                    if h5_file is None:
                        continue

                    handler = phmm.HmmHandler(rec_dir)
                    # df = handler.get_data_overview().copy()
                    df = handler.get_overview_w_AIC().copy()
                    df['rec_dir'] = rec_dir
                    if ho is None:
                        ho = df
                    else:
                        ho = ho.append(df, ignore_index=True)

                    # pbar.update(1)

            # pbar.close()
            feather.write_feather(ho, self.files['hmm_overview'])

        if not 'exp_name' in ho.columns:
            try:
                ho = apply_groups_from_proj(ho, self.project)
                feather.write_feather(ho, self.files['hmm_overview'])
            except:
                raise Exception("need to apply rec_group")

        if not 'single_state_trials' in ho.columns:
            print('counting single state trials...')
            ho['single_state_trials'] = ho.apply(hmma.check_single_state_trials, axis=1)
            feather.write_feather(ho, self.files['hmm_overview'])

        return ho

    def sort_hmms(self, overwrite=False, plot_rejected=False):
        sorted_hmms = self.get_sorted_hmms()
        if sorted_hmms is not None and not overwrite:
            return sorted_hmms

        ho = self.get_hmm_overview()
        new_sorting = hmma.sort_hmms_by_rec(ho)
        if sorted_hmms is not None:
            for i, row in sorted_hmms.iterrows():
                j = ((new_sorting['rec_dir'] == row['rec_dir']) &
                     (new_sorting['hmm_id'] == row['hmm_id']))
                k = ((new_sorting['rec_dir'] == row['rec_dir']) &
                     (new_sorting['taste'] == row['taste']) &
                     (new_sorting['hmm_id'] != row['hmm_id']))
                if row['sort_method'] == 'manual' and row['sorting'] == 'best':
                    new_sorting.loc[j, 'sorting'] = row['sorting']
                    new_sorting.loc[j, 'sort_method'] = row['sort_method']
                    new_sorting.loc[k, 'sorting'] = 'rejected'
                    new_sorting.loc[k, 'sort_method'] = 'manual'

                if not np.isnan(row['early_state']):
                    new_sorting.loc[j, 'early_state'] = row['early_state']

                if not np.isnan(row['late_state']):
                    new_sorting.loc[j, 'late_state'] = row['late_state']

        self.write_sorted_hmms(new_sorting)
        return new_sorting

    def write_sorted_hmms(self, sorted_hmms):
        sorted_file = self.files['sorted_hmms']
        feather.write_feather(sorted_hmms, sorted_file)

    def get_sorted_hmms(self):
        sorted_file = self.files['sorted_hmms']
        if os.path.isfile(sorted_file):
            sorted_hmms = feather.read_feather(sorted_file)
            if 'early_state' not in sorted_hmms.columns:
                sorted_hmms['early_state'] = np.nan

            if 'late_state' not in sorted_hmms.columns:
                sorted_hmms['late_state'] = np.nan

            if 'sorting' not in sorted_hmms.columns:
                sorted_hmms['sorting'] = 'rejected'

            if 'sort_method' not in sorted_hmms.columns:
                sorted_hmms['sort_method'] = 'auto'

            if 'palatability' not in sorted_hmms.columns:
                pal_map = {'Spont': -1, 'Suc': 1, 'QHCl': 4,
                           'CA': 3, 'NaCl': 2}
                sorted_hmms['palatability'] = sorted_hmms.taste.map(pal_map)
                self.write_sorted_hmms(sorted_hmms)

            if 'exclude' not in sorted_hmms.columns:
                sorted_hmms = agg.apply_grouping_cols(sorted_hmms, self.project)
                self.write_sorted_hmms(sorted_hmms)

            return sorted_hmms
        else:
            return None

    def input_manual_sorting(self, fn, wipe_manual=False):
        '''assumes tab-seperated variables in text file with columns: exp_name,
        rec_group, hmm_id, taste, early_state, late_state, sorting_notes
        '''
        sorted_df = self.get_sorted_hmms()
        if 'sorting_notes' not in sorted_df.columns:
            sorted_df['sorting_notes'] = ''

        if wipe_manual:
            sorted_df['sorting'] = 'rejected'
            sorted_df['sort_method'] = 'auto'
            sorted_df['sorting_notes'] = ''
        else:
            idx = (sorted_df['sort_method'] == 'auto')
            sorted_df.loc[idx, 'sorting'] = 'rejected'
            sorted_df.loc[idx, 'sorting_notes'] = ''

        sorted_df['hmm_id'] = sorted_df['hmm_id'].astype('int')
        with open(fn, 'r') as f:
            lines = f.readlines()

        columns = ['exp_name', 'rec_group', 'hmm_id', 'taste', 'early_state',
                   'late_state', 'sorting_notes']
        for l in lines:
            if l == '\n' or l == '':
                continue

            vals = l.replace('\n', '').split('\t')
            exp_name = vals[0]
            rec_group = vals[1]
            hmm_id = int(vals[2])
            early_state = int(vals[4])
            late_state = int(vals[5])
            if len(vals) == 7:
                notes = vals[6]
            else:
                notes = ''

            tmp_idx = ((sorted_df['exp_name'] == exp_name) &
                       (sorted_df['rec_group'] == rec_group) &
                       (sorted_df['hmm_id'] == hmm_id))
            sorted_df.loc[tmp_idx, 'sorting'] = 'best'
            sorted_df.loc[tmp_idx, 'sort_method'] = 'manual'
            sorted_df.loc[tmp_idx, 'early_state'] = early_state
            sorted_df.loc[tmp_idx, 'late_state'] = late_state
            sorted_df.loc[tmp_idx, 'sorting_notes'] = notes

        self.write_sorted_hmms(sorted_df)
        return sorted_df

    def get_best_hmms(self, overwrite=False, sorting='best', save_dir=None):
        best_file = self.files['best_hmms']
        if save_dir is not None:
            best_file = os.path.join(save_dir, os.path.basename(best_file))

        bf = os.path.isfile(best_file)
        if bf and not overwrite:
            best_hmms = feather.read_feather(best_file)
            return best_hmms

        df = self.get_sorted_hmms()
        all_units, _ = ProjectAnalysis(self.project).get_unit_info()  # get unit info fails to pull DS41
        out_df = hmma.make_best_hmm_list(all_units, df, sorting=sorting)
        out_df = agg.apply_grouping_cols(out_df, self.project)
        feather.write_feather(out_df, best_file)
        return out_df

    def mark_hmm_as(self, sorting, **kwargs):
        '''kwargs should be column & value and will be used to manually re-sort
        HMMs and mark then as "best", "rejected" or "refit"
        '''
        qry = ' and '.join(['{} == "{}"'.format(k, v) for k, v in kwargs.items()])
        nqry = ' or '.join(['{} != "{}"'.format(k, v) for k, v in kwargs.items()])
        sorted_hmms = self.get_sorted_hmms()
        j = None
        for k, v in kwargs.items():
            tmp = (sorted_hmms[k] == v)
            if j is None:
                j = tmp
            else:
                j = j & tmp

        old_sorting = sorted_hmms.loc[j, 'sorting'].unique()
        print('-' * 80)
        print('Marking HMMs as %s' % sorting)
        print(sorted_hmms.loc[j][['exp_name', 'rec_group', 'taste', 'hmm_id',
                                  'sorting', 'sort_method']])
        print('-' * 80)
        sorted_hmms.loc[j, 'sorting'] = sorting
        sorted_hmms.loc[j, 'sort_method'] = 'manual'
        # make sure there is only 1 best for each rec_dir & taste
        best_df = sorted_hmms[sorted_hmms['sorting'] == 'best']
        multiple_bests = []
        for name, group in best_df.groupby(['rec_dir', 'taste']):
            if len(group) != 1:
                multiple_bests.append(group.copy())

        self.write_sorted_hmms(sorted_hmms)

    def mark_hmm_state(self, exp_name, rec_group, hmm_id, early_state=None, late_state=None):
        '''Set the HMM state number that identifies the stat to be used as the
        early state or late state in analysis
        '''
        if early_state is None and late_state is None:
            return

        hmm_df = self.get_sorted_hmms()
        i = ((hmm_df['exp_name'] == exp_name) &
             (hmm_df['rec_group'] == rec_group) &
             (hmm_df['hmm_id'] == hmm_id))

        print('-' * 80)
        print('Setting state for HMMs: %s %s %s' % (exp_name, rec_group, hmm_id))
        if early_state is not None:
            print('    - setting early_state to state #%i' % early_state)
            hmm_df.loc[i, 'early_state'] = early_state

        if late_state is not None:
            print('    - setting late_state to state #%i' % late_state)
            hmm_df.loc[i, 'late_state'] = late_state

        print('Saving dataframe...')
        self.write_sorted_hmms(hmm_df)
        print('-' * 80)

    def plot_sorted_hmms(self, overwrite=False, skip_tags=[],
                         sorting_tag=None, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir
            if sorting_tag is not None:
                save_dir = os.path.join(save_dir, sorting_tag)

        plot_dirs = {'best': os.path.join(self.save_dir, 'Best_HMMs'),
                     'rejected': os.path.join(self.save_dir, 'Rejected_HMMs'),
                     'refit': os.path.join(self.save_dir, 'Refit_HMMs')}
        sorted_hmms = self.get_sorted_hmms()
        if sorting_tag is not None:
            sorted_hmms = sorted_hmms.query('sorting == @sorting_tag')
            tags = [sorting_tag]
            plot_dirs = {sorting_tag: save_dir}
        else:
            tags = sorted_hmms.sorting.unique()
            plot_dirs = {x: os.path.join(save_dir, x) for x in tags}

        for k, v in plot_dirs.items():
            if os.path.isdir(v) and overwrite:
                shutil.rmtree(v)

            if not os.path.isdir(v):
                os.mkdir(v)

        pbar = tqdm(total=sorted_hmms.shape[0])
        for i, row in sorted_hmms.iterrows():
            fn = '%s_%s_HMM%i-%s.svg' % (row['exp_name'], row['rec_group'],
                                         row['hmm_id'], row['taste'])

            # print(f'plotting {fn}...')
            if row['sorting'] not in plot_dirs.keys():
                pbar.update()
                continue

            if row['sorting'] in skip_tags:
                pbar.update()
                continue

            fn = os.path.join(plot_dirs[row['sorting']], fn)
            if os.path.isfile(fn) and not overwrite:
                pbar.update()
                continue

            es = row['early_state']
            ls = row['late_state']
            title_extra = f'Early: {es}, Late: {ls}'
            plt.plot_hmm(row['rec_dir'], row['hmm_id'], save_file=fn,
                         title_extra=title_extra)
            pbar.update()

        pbar.close()

    def plot_hmms_deprecated(self, overwrite=False):
        hmm_df = self.get_hmm_overview()
        grp_keys = list(self.base_params.keys())
        if 'n_states' not in grp_keys:
            grp_keys.append('n_states')

        exp_map = self.project._exp_info.set_index('exp_name')['exp_group'].to_dict()

        for grp_i, (name, group) in enumerate(hmm_df.groupby(grp_keys)):
            if len(group) < 3:
                continue

            save_dir = os.path.join(self.save_dir, 'All_HMMs', 'Paramters_%i' % grp_i)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            params = group[grp_keys].to_dict(orient='records')[0]
            wt.write_dict_to_json(params, os.path.join(save_dir, 'params.json'))
            plot_dir = os.path.join(save_dir, 'Plots')
            if not os.path.isdir(plot_dir):
                os.mkdir(plot_dir)

            state_breakdown = None
            for row_i, (_, row) in enumerate(group.iterrows()):
                rec_dir = row['rec_dir']
                hmm_id = row['hmm_id']
                fn = os.path.join(plot_dir, '%s_%s_HMM%i-%s.svg' % (row['exp_name'],
                                                                    row['rec_group'],
                                                                    hmm_id,
                                                                    row['taste']))

                if os.path.isfile(fn) and not overwrite:
                    continue

                plt.plot_hmm(rec_dir, hmm_id, save_file=fn)

    def plot_all_hmms(self, overwrite=False):
        sorted_df = self.get_sorted_hmms()
        plot_dir = os.path.join(self.save_dir, 'Fitted_HMMs')
        grpby = sorted_df.groupby(['exp_name', 'rec_group', 'taste'])
        counter = 0
        total = sorted_df.shape[0]
        # pbar = tqdm(total=sorted_df.shape[0])
        for name, group in grpby:
            anim_dir = os.path.join(plot_dir, '_'.join(name))
            if not os.path.isdir(anim_dir):
                os.makedirs(anim_dir)

            for i, row in group.iterrows():
                counter += 1
                print('Plotting %i of %i' % (counter, total))
                fn = '_'.join(name) + '_HMM#%s.svg' % row['hmm_id']
                fn = os.path.join(anim_dir, fn)
                if os.path.isfile(fn) and not overwrite:
                    # pbar.update()
                    continue

                # print('Plotting HMMs for %s' % ('_'.join(name)))
                plt.plot_hmm(row['rec_dir'], row['hmm_id'], save_file=fn)
                # pbar.update()

        # pbar.close()

    def plot_grouped_BIC(self):  # TODO: get rid of hard coding
        # params: list of params
        srt_df = self.get_sorted_hmms()
        srt_df = srt_df.query('time_start==-250')
        srt_df = srt_df.loc[
            srt_df.groupby(['taste', 'exp_group', 'time_group', 'n_states', 'exp_name'])['threshold'].idxmin()]

        BIC_grp_file = os.path.join(self.save_dir, 'grouped_BIC_comparison.svg')
        nplt.plot_grouped_BIC(srt_df, save_file=BIC_grp_file)

    def plot_best_BIC(self):
        srt_df = self.get_sorted_hmms()
        srt_df = srt_df.query('time_start==-250')
        srt_df = srt_df.loc[
            srt_df.groupby(['taste', 'exp_group', 'time_group', 'n_states', 'exp_name'])['threshold'].idxmin()]
        srt_df = srt_df.loc[srt_df.groupby(['taste', 'exp_group', 'time_group', 'exp_name'])['BIC'].idxmin()]

        file = os.path.join(self.save_dir, "best_BIC_comparison.svg")
        nplt.plot_best_BIC(srt_df, save_file=file)

    def plot_BIC_comparison(self):
        ho = self.get_hmm_overview()
        BIC_file = os.path.join(self.save_dir, 'bic_comparison.svg')
        nplt.plot_BIC(ho, self.project, save_file=BIC_file)

    def analyze_NB_ID2(self, overwrite=False):
        save_dir = self.save_dir
        ID_decode_sf = os.path.join(save_dir, 'all_states_decode.feather')

        if os.path.isfile(ID_decode_sf) and not overwrite:
            NB_res = feather.read_feather(ID_decode_sf)
            return NB_res
        else:
            best_hmms = self.get_best_hmms(sorting='AIC')
            proj = self.project
            PA = ProjectAnalysis(proj)
            din_trial_df = proj.get_dig_in_trial_df(reformat=True)
            all_units, held_units = PA.get_unit_info()
            NB_res = hmma.NB_state_classification(best_hmms, all_units)
            #rename trial to 'taste_trial'
            NB_res = NB_res.rename(columns = {'trial': 'taste_trial'})
            #merge with dig_in_trial_df along 'rec_dir' and 'taste_trial' and 'taste'
            NB_res = pd.merge(NB_res, din_trial_df, on=['rec_dir', 'taste_trial', 'taste', 'channel'], how='left')
            #drop the 'label' column
            NB_res = NB_res.drop(columns=['label'])
            #save NB_res to feather
            feather.write_feather(NB_res, ID_decode_sf)
            return NB_res

    def analyze_NB_val(self, overwrite=False):
        save_dir = self.save_dir
        ID_decode_sf = os.path.join(save_dir, 'pal_states_decode.feather')
        valence_map = {'Suc':'pos', 'NaCl':'pos', 'CA':'neg', 'QHCl':'neg'}


        if os.path.isfile(ID_decode_sf) and not overwrite:
            NB_res = feather.read_feather(ID_decode_sf)
            return NB_res
        else:
            best_hmms = self.get_best_hmms(sorting='AIC')
            # apply a column to best_hmms called valence, using the taste column to map to pos or neg
            best_hmms['valence'] = best_hmms['taste'].map(valence_map)
            proj = self.project
            PA = ProjectAnalysis(proj)
            din_trial_df = proj.get_dig_in_trial_df(reformat=True)
            #apply a
            all_units, held_units = PA.get_unit_info()
            NB_res = hmma.NB_state_classification(best_hmms, all_units, label_col='valence')
            #rename trial to 'taste_trial'
            NB_res = NB_res.rename(columns={'trial': 'taste_trial'})
            #merge with dig_in_trial_df along 'rec_dir' and 'taste_trial' and 'taste'
            NB_res = pd.merge(NB_res, din_trial_df, on=['rec_dir', 'taste_trial', 'channel'], how='left')
            #drop the 'label' column
            NB_res = NB_res.drop(columns=['label'])
            #save NB_res to feather
            feather.write_feather(NB_res, ID_decode_sf)
            return NB_res

    def analyze_NB_ID(self, overwrite=True, multi_process=False, sorting="best_AIC"):
        '''
        critical function for pizza talk decoding analysis of HMM states
        '''
        if overwrite is True:
            best_hmms = self.get_best_hmms(sorting=sorting)
            all_units, _ = ProjectAnalysis(self.project).get_unit_info()

            if multi_process==True:
                NB_res, NB_meta = hmma.analyze_NB_state_classification_parallel(best_hmms, all_units)
            else:
                NB_res, NB_meta = hmma.analyze_NB_state_classification_parallel(best_hmms, all_units, run_parallel=False)
                #NB_res, NB_meta = hmma.analyze_NB_state_classification(best_hmms, all_units)  # , epoch = epoch, prestim = True)

            decode_data = hmma.process_NB_classification(NB_meta, NB_res)

            best_cop = best_hmms
            decode_data = decode_data.reset_index(drop=True)

            fm = decode_data[['exp_name', 'time_group', 'trial_ID', 'Y', 'hmm_state', 'hmm_id']].drop_duplicates()
            fm = fm.pivot(index=['exp_name', 'time_group', 'trial_ID', 'hmm_id'], columns='Y', values='hmm_state')
            fm = fm.fillna(0)
            ecols = fm.loc[:, fm.columns.str.endswith('early')].astype(int)
            fm['early'] = ecols.sum(skipna=False, axis=1)

            lcols = fm.loc[:, fm.columns.str.endswith('late')].astype(int)
            fm['late'] = lcols.sum(skipna=False, axis=1)
            fm = fm[['prestim', 'early', 'late']].reset_index()

            try:
                best_cop = best_cop.drop(columns=['ID_state', 'trial_ID'])
            except:
                print('good')

            best_cop['time_group'] = best_cop.time_group.astype(int)
            best_cop['hmm_id'] = best_cop.hmm_id.astype(int)
            fm['time_group'] = fm.time_group.astype(int)
            fm['hmm_id'] = fm.hmm_id.astype(int)
            best_hmms = best_cop.merge(fm, on=["exp_name", "time_group", "hmm_id"])
            best_hmms = best_hmms.drop_duplicates()

            timings = hmma.analyze_classified_hmm_state_timing(best_hmms, decode_data, min_dur=50)
            proj = self.project
            trial_info = proj.get_trial_info()
            trial_info = trial_info.rename(
                columns={'name': 'taste', 'trial_num': 'session_trial', 'taste_trial': 'trial_num'})
            trial_info = trial_info[['trial_num', 'taste', 'off_time', 'session_trial', 'rec_dir']]
            trial_info['trial_num'] = trial_info['trial_num'] - 1
            timings = pd.merge(timings, trial_info, on=['rec_dir', 'taste', 'trial_num'], how='left')
            timings = timings.drop_duplicates()
            timings['session_trial'] = timings.groupby(['rec_dir'])['session_trial'].transform(lambda x: x - x.min())

            save_dir = self.save_dir
            ID_timing_sf = os.path.join(save_dir, 'ID_timing.feather')
            self.files['early_ID_timing'] = ID_timing_sf
            feather.write_feather(timings, ID_timing_sf)
            print('saving ID_timing to %s' % ID_timing_sf)

            ID_decode_sf = os.path.join(save_dir, 'ID_decode.feather')
            self.files['decode'] = ID_decode_sf
            feather.write_feather(decode_data, ID_decode_sf)
            print('saving ID_decode to %s' % ID_decode_sf)

            # TODO: consolidate output of this function so it's the same when saved or not
            # NB_meta.drop(columns = ['hmm_state'])
            best_file = self.files['best_hmms']
            feather.write_feather(best_hmms, best_file)
            print('saving best_hmms to %s' % best_file)

            return NB_meta, decode_data, best_hmms, timings

        else:
            save_dir = self.save_dir

            ID_decode_sf = os.path.join(save_dir, 'ID_decode.feather')
            decode_data = feather.read_feather(ID_decode_sf)

            ID_timing_sf = os.path.join(save_dir, 'ID_timing.feather')
            timings = feather.read_feather(ID_timing_sf)
            best_hmms = ['need to run with overwrite to get best_hmms']
            NB_meta = ['need to run with overwrite to get NB_meta']

            return NB_meta, decode_data, best_hmms, timings

    def analyze_pal_linear_regression(self):
        best_hmms = self.get_best_hmms(sorting='best_BIC')
        all_units, _ = ProjectAnalysis(self.project).get_unit_info()
        pal_map = {'taste': ['QHCl', 'CA', 'NaCl', 'Suc'],
                   'palatability': [1, 2, 3, 4]}
        pal_map = pd.DataFrame(pal_map)
        best_hmms = best_hmms.drop(columns=['palatability'])
        best_hmms = best_hmms.merge(pal_map, on='taste')

        results = hmma.LinReg_pal_classification(best_hmms, all_units)
        linreg_meta, all_trls = hmma.process_LinReg_pal_classification(results)

        pal_states = all_trls
        pal_states = pal_states[['rec_dir', 'Y', 'hmm_state', 'hmm_id']].drop_duplicates()
        pal_states.hmm_id = pal_states.hmm_id.astype(int)
        pal_states = pal_states.rename(columns={"hmm_state": "pal_state"})
        best_hmms = pd.merge(best_hmms, pal_states, on=["rec_dir", "hmm_id"])
        best_hmms = best_hmms.rename(columns={"Y": "pal_rating"})

        all_trls['hmm_id'] = all_trls['hmm_id'].astype(int)
        mod = best_hmms[['rec_dir', 'exp_name', 'exp_group', 'time_group', 'taste', 'hmm_id']].drop_duplicates()
        all_trls = all_trls.merge(mod, on=['rec_dir', 'hmm_id'])

        all_trls = all_trls.rename(columns={'trial': 'trial_num', 'Y': 'trial_ID'})
        all_trls['prestim_state'] = False
        timings = hmma.analyze_classified_hmm_state_timing(best_hmms, all_trls, 'pal_state', min_dur=50)
        timings = timings.loc[timings.state_group == 'pal_state']

        mf = all_trls[['rec_dir', 'trial_num', 'hmm_id', 'Y_pred']]
        timings = timings.merge(mf, on=['rec_dir', 'trial_num', 'hmm_id'])
        timings = timings.loc[timings.pos_in_trial != 0]
        timings = timings.loc[timings.t_start > 0]

        return timings, all_trls, results, linreg_meta

    def analyze_hmms(self, overwrite=False, save_dir=None):
        all_units, _ = ProjectAnalysis(self.project).get_unit_info()
        coding_file = self.files['hmm_coding']
        timing_file = self.files['hmm_timings']
        confusion_file = self.files['hmm_confusion']
        if save_dir is not None:
            cfn = os.path.basename(coding_file)
            cnfn = os.path.basename(confusion_file)
            tfn = os.path.basename(timing_file)
            coding_file = os.path.join(save_dir, cfn)
            timing_file = os.path.join(save_dir, tfn)
            confusion_file = os.path.join(save_dir, cnfn)
        else:
            save_dir = self.save_dir

        best_hmms = self.get_best_hmms(save_dir=save_dir)
        timing_stats = timing_file.replace('feather', 'txt')
        # coding analyses
        if os.path.isfile(coding_file) and not overwrite:
            coding = feather.read_feather(coding_file)
        else:
            coding = hmma.analyze_hmm_state_coding(best_hmms, all_units)
        # end of coding analysis

        if os.path.isfile(timing_file) and not overwrite:
            timings = feather.read_feather(timing_file)
        else:
            timings = hmma.analyze_hmm_state_timing(best_hmms)
            feather.write_feather(timings, timing_file)
        # if os.path.isfile(confusion_file) and not overwrite:
        #     confusion = feather.read_feather(confusion_file)
        # else:
        #     confusion = hmma.saccharin_confusion_analysis(best_hmms, all_units,
        #                                                   area='GC',
        #                                                   single_unit=True,
        #                                                   repeats=50)
        #     feather.write_feather(confusion, confusion_file)
        coding, confusion = None, None

        if not os.path.isfile(timing_stats) or overwrite:
            descrip = hmma.describe_hmm_state_timings(timings)
            with open(timing_stats, 'w') as f:
                print(descrip, file=f)

        return coding, timings, confusion
        # after this do plot hmm timing 03/10/20

    def plot_hmm_timing(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir

        _, timing, _ = self.analyze_hmms(save_dir=save_dir)
        df = timing.copy()
        # df['exclude'] = df.apply(lambda x: True if
        #                          (x['exp_group'] == 'GFP' and
        #                           x['cta_group'] == 'No CTA')
        #                          else False, axis=1)

        plot_dir = os.path.join(save_dir, 'timing_analysis')
        if os.path.isdir(plot_dir):
            shutil.rmtree(plot_dir)

        os.mkdir(plot_dir)

        corr_file = os.path.join(plot_dir, 'timing_correlations.svg')  # looks good 03/05/22
        nplt.plot_timing_correlations(df, save_file=corr_file)
        corr_file = os.path.join(plot_dir, 'timing_correlations-exclude.svg')
        nplt.plot_timing_correlations(df, save_file=corr_file)

        dist_file = os.path.join(plot_dir, 'early_end_distributions.svg')
        nplt.plot_timing_distributions(df, state='early', value_col='t_end', save_file=dist_file)

        dist_file = os.path.join(plot_dir, 'late_start_distributions.svg')
        nplt.plot_timing_distributions(df, state='late', value_col='t_start', save_file=dist_file)

        comp_file = os.path.join(plot_dir, 'timing_comparison.svg')
        nplt.plot_timing_data(df, save_file=comp_file, group_col='exp_group')

        tb_file = os.path.join(plot_dir, 'trial_group_comparison.svg')
        nplt.plot_intraday_timing(df, save_file=tb_file, group_col='trial_group')

        # dist_file = os.path.join(plot_dir, 'Suc_late_start_distributions.svg')
        # nplt.plot_timing_distributions(df.query('taste == "Suc"'),
        #                                state='late', value_col='t_start',
        #                                save_file=dist_file)

        for taste, grp in df.groupby('taste'):
            # plot distributions
            fn1 = os.path.join(plot_dir, f'{taste}_early_end_distributions.svg')
            nplt.plot_timing_distributions(grp, state='early',
                                           value_col='t_end', save_file=fn1)

            fn1 = os.path.join(plot_dir, f'{taste}_late_start_distributions.svg')
            nplt.plot_timing_distributions(grp, state='late',
                                           value_col='t_start', save_file=fn1)

            # exp_group
            comp_file = os.path.join(plot_dir, f'{taste}_timing_comparison.svg')
            nplt.plot_timing_data(grp, save_file=comp_file, group_col='exp_group')
            tb_file = os.path.join(plot_dir, f'{taste}_trial_group_comparision.svg')
            nplt.plot_intraday_timing(grp, save_file=tb_file, group_col='trial_group')

    def plot_hmm_confusion(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir

        _, _, confusion = self.analyze_hmms(save_dir=save_dir)
        confusion['exclude'] = confusion.apply(lambda x: True if
        (x['exp_group'] == 'GFP' and
         x['cta_group'] == 'No CTA')
        else False, axis=1)
        plot_dir = os.path.join(save_dir, 'confusion_analysis')
        if os.path.isdir(plot_dir):
            shutil.rmtree(plot_dir)

        os.mkdir(plot_dir)

        exc_df = confusion[confusion['exclude'] == False]

        # Make confusion plots using exp_group
        corr_file = os.path.join(plot_dir, 'confusion_correlations.svg')
        comp_file = os.path.join(plot_dir, 'confusion_comparison.svg')
        diff_file = os.path.join(plot_dir, 'confusion_differences.svg')
        nplt.plot_confusion_correlations(confusion, save_file=corr_file)
        nplt.plot_confusion_data(confusion, save_file=comp_file, group_col='exp_group')
        nplt.plot_confusion_differences(confusion, save_file=diff_file)

        # Make confusion plots using exp_group and excluding GFP-NoCTA
        corr_file = os.path.join(plot_dir, 'confusion_correlations-exclude.svg')
        comp_file = os.path.join(plot_dir, 'confusion_comparison-exclude.svg')
        nplt.plot_confusion_correlations(exc_df, save_file=corr_file)
        nplt.plot_confusion_data(exc_df, save_file=comp_file, group_col='exp_group')

        # Make confusion plots using cta_group
        comp_file = os.path.join(plot_dir, 'confusion_comparison-CTA.svg')
        nplt.plot_confusion_data(confusion, save_file=comp_file, group_col='cta_group')

        # Make confusion plots using cta_group and excluding GFP-NoCTA
        comp_file = os.path.join(plot_dir, 'confusion_comparison-CTA-exclude.svg')
        nplt.plot_confusion_data(exc_df, save_file=comp_file, group_col='cta_group')

    def plot_hmm_coding(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir

        coding, _, _ = self.analyze_hmms(save_dir=save_dir)
        coding['exclude'] = coding.apply(lambda x: True if
        (x['exp_group'] == 'GFP' and
         x['cta_group'] == 'No CTA')
        else False, axis=1)
        plot_dir = os.path.join(save_dir, 'coding_analysis')
        if os.path.isdir(plot_dir):
            shutil.rmtree(plot_dir)

        os.mkdir(plot_dir)

        exc_df = coding[coding['exclude'] == False]

        # Make coding plots using exp_group
        corr_file = os.path.join(plot_dir, 'coding_correlations.svg')
        comp_file = os.path.join(plot_dir, 'coding_comparison.svg')
        nplt.plot_coding_correlations(coding, save_file=corr_file)
        nplt.plot_coding_data(coding, save_file=comp_file, group_col='exp_group')

        # Make coding plots using exp_group and excluding GFP-NoCTA
        corr_file = os.path.join(plot_dir, 'coding_correlations-exclude.svg')
        comp_file = os.path.join(plot_dir, 'coding_comparison-exclude.svg')
        nplt.plot_coding_correlations(exc_df, save_file=corr_file)
        nplt.plot_coding_data(exc_df, save_file=comp_file, group_col='exp_group')

        # Make coding plots using cta_group
        comp_file = os.path.join(plot_dir, 'coding_comparison-CTA.svg')
        nplt.plot_coding_data(coding, save_file=comp_file, group_col='cta_group')

        # Make coding plots using cta_group and excluding GFP-NoCTA
        comp_file = os.path.join(plot_dir, 'coding_comparison-CTA-exclude.svg')
        nplt.plot_coding_data(exc_df, save_file=comp_file, group_col='cta_group')

    def plot_hmm_coding_and_timing(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir

        best_hmms = self.get_best_hmms(save_dir=save_dir)
        coding, timings, confusion = self.analyze_hmms(save_dir=save_dir)
        coding_fn = os.path.join(save_dir, 'HMM_coding.png')
        timing_fn = os.path.join(save_dir, 'HMM_timing.png')
        prob_fn = os.path.join(save_dir, 'HMM_Median_Gamma_Probs.png')
        mean_prob_fn = os.path.join(save_dir, 'HMM_Mean_Gamma_Probs.png')
        seq_fn = os.path.join(save_dir, 'All_Sequences.png')
        sacc_seq_fn = os.path.join(save_dir, 'Saccharin_Sequences.svg')
        seq_cta_fn = os.path.join(save_dir, 'All_Sequences-CTA.png')
        sacc_seq_cta_fn = os.path.join(save_dir, 'Saccharin_Sequences-CTA.png')
        plt.plot_hmm_coding_accuracy(coding, coding_fn)
        plt.plot_hmm_timings(timings, timing_fn)
        plt.plot_median_gamma_probs(best_hmms.query('taste == "Saccharin"'), prob_fn)
        plt.plot_mean_gamma_probs(best_hmms.query('taste == "Saccharin"'), mean_prob_fn)

        df = best_hmms.query('taste == "Saccharin" and exclude == False')
        plt.plot_median_gamma_probs(df, prob_fn.replace('.png', '-exclude.svg'))

        plt.plot_hmm_sequence_heatmap(best_hmms, 'exp_group', seq_fn)
        plt.plot_hmm_sequence_heatmap(best_hmms.query('taste=="Saccharin" and exclude == False'), 'exp_group',
                                      sacc_seq_fn)
        plt.plot_hmm_sequence_heatmap(best_hmms, 'cta_group', seq_cta_fn)
        plt.plot_hmm_sequence_heatmap(best_hmms.query('taste=="Saccharin"'), 'cta_group', sacc_seq_cta_fn)
        unit_fn = os.path.join(save_dir, 'Unit_Info.txt')
        with open(unit_fn, 'w') as f:
            all_units, held_units = ProjectAnalysis(self.project).get_unit_info()
            all_units = all_units.query('single_unit == True and area == "GC"')
            held_units = held_units.query('area == "GC"')
            out = ['All Unit Counts']
            out.append('-' * 80)
            a_count = all_units.groupby(['exp_group', 'exp_name',
                                         'rec_group'])['unit_num'].count()
            held_units = held_units.dropna(subset=['held_unit_name'])
            h_count = held_units.groupby(['exp_group', 'exp_name', 'held_over'])['held'].count()
            out.append(repr(a_count))
            out.append('=' * 80)
            out.append('')
            out.append('Held Unit Counts')
            out.append('-' * 80)
            out.append(repr(h_count))
            print('\n'.join(out), file=f)

    # def process_fitted_hmms(self, overwrite=False, hmm_plots=False):
    #     with pm.push_alert(success_msg='HMM Processing Complete! :D'):
    #         # ho = self.get_hmm_overview(overwrite=overwrite)
    #         ho = self.get_hmm_overview(overwrite=False)
    #         self.sort_hmms_by_params(overwrite=overwrite)
    #         self.mark_early_and_late_states()
    #         sorted_df = self.get_sorted_hmms()
    #         param_sets = sorted_df.sorting.unique()
    #
    #         self.plot_BIC_comparison()
    #
    #         for set_name in param_sets:
    #             if set_name == 'rejected':
    #                 continue
    #
    #             save_dir = os.path.join(self.save_dir, set_name)
    #             if os.path.isdir(save_dir) and overwrite:
    #                 shutil.rmtree(save_dir)
    #
    #             if not os.path.isdir(save_dir):
    #                 os.mkdir(save_dir)
    #
    #             self.write_hmm_sorting_params(sorting=set_name, save_dir=save_dir)
    #             best_hmms = self.get_best_hmms(overwrite=True, save_dir=save_dir,
    #                                            sorting=set_name)
    #             notes = best_hmms.dropna().notes.unique()[0]
    #             if 'BIC test' in notes:
    #                 continue
    #
    #             if sum(sorted_df['sorting'] == set_name) > 30:
    #                 try:
    #                     self.analyze_hmms(overwrite=True, save_dir=save_dir)
    #                     self.plot_hmm_coding_and_timing(save_dir=save_dir)
    #                     self.plot_hmm_coding(save_dir=save_dir)
    #                     self.plot_hmm_confusion(save_dir=save_dir)
    #                     self.plot_hmm_timing(save_dir=save_dir)
    #                     self.plot_hmm_trial_breakdown(save_dir=save_dir)
    #                 except:
    #                     print(f'Failed to analyze {set_name}')
    #
    #             if hmm_plots:
    #                 plot_dir = os.path.join(save_dir, 'HMM Plots')
    #                 if os.path.isdir(plot_dir) and overwrite:
    #                     shutil.rmtree(plot_dir)
    #
    #                 if not os.path.isdir(plot_dir):
    #                     os.mkdir(plot_dir)
    #
    #                 self.plot_sorted_hmms(overwrite=overwrite, save_dir=plot_dir,
    #                                       sorting_tag=set_name)
    #
    #     return sorted_df

    def plot_hmm_trial_breakdown(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir

        best_hmms = self.get_best_hmms(save_dir=save_dir)
        sorting = best_hmms.sorting.unique()
        if len(sorting) != 1:
            raise ValueError('Multiple or Zero sortings found in best_hmms')
        else:
            sorting = sorting[0]

        print('Gathering hmm trial data ... ')
        trial_df = get_hmm_trial_info(self, sorting=sorting)
        feather.write_feather(trial_df,
                                os.path.join(save_dir, 'hmm_trial_breakdown.feather'))

        fn = os.path.join(save_dir, 'hmm_trial_breakdown.svg')
        nplt.plot_hmm_trial_breakdown(trial_df, self.project, save_file=fn)

   # @pm.push_alert(success_msg='HMM processing complete!')
    def process_sorted_hmms(self, sorting='params #5', save_dir=None,
                            overwrite=False, hmm_plots=False):
        if save_dir is None:
            save_dir = self.save_dir

        sorted_df = self.get_sorted_hmms()
        if not sorting in sorted_df.sorting.unique():
            raise ValueError(f'Sorting not found: {sorting}')

        self.write_hmm_sorting_params(sorting=sorting, save_dir=save_dir)
        self.plot_BIC_comparison()
        best_hmms = self.get_best_hmms(overwrite=overwrite, sorting=sorting, save_dir=save_dir)
        self.analyze_hmms(overwrite=overwrite, save_dir=save_dir)
        self.plot_hmm_coding_and_timing(save_dir=save_dir)
        self.plot_hmm_coding(save_dir=save_dir)
        self.plot_hmm_confusion(save_dir=save_dir)
        self.plot_hmm_timing(save_dir=save_dir)
        self.plot_hmm_trial_breakdown(save_dir=save_dir)
        if hmm_plots:
            plot_dir = os.path.join(save_dir, 'HMM Plots')
            if os.path.isdir(plot_dir):
                shutil.rmtree(plot_dir)

            os.mkdir(plot_dir)

            self.plot_sorted_hmms(overwrite=True, save_dir=plot_dir,
                                  sorting_tag=sorting)

    def write_hmm_sorting_params(self, sorting='params #5', save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir

        param_cols = ['n_states', 'time_start', 'time_end', 'area',
                      'unit_type', 'dt', 'n_trials', 'notes']
        sorted_df = self.get_sorted_hmms()
        df = sorted_df.query('sorting == @sorting')
        name, group = next(iter(df.groupby(param_cols)))
        summary = ['-' * 80,
                   'parameter set: %s' % sorting,
                   '# of states: %i' % name[0],
                   'time: %i to %i' % (name[1], name[2]),
                   'area: %s' % name[3],
                   'unit type: %s' % name[4],
                   'dt: %g' % name[5],
                   'n_trials: %i' % name[6],
                   'notes: %s' % name[7],
                   '# of HMMs: %i' % len(group),
                   '-' * 80]
        with open(os.path.join(save_dir, 'HMM_parameters.txt'), 'w') as f:
            f.write('\n'.join(summary))

    def plot_hmms_for_comp(self):
        sorted_df = self.get_sorted_hmms()
        plot_dir = os.path.join(self.save_dir, 'HMMs_for_Don')
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        # For each animal, rec_group, taste plot hmms labelled by prominent params
        # Important parameters: n_states, unit_type
        # If any best in rec_group has pyramidal unit type, then also plot pyramidal solutions
        req_params = {'area': 'GC', 'dt': 0.001, 'hmm_class': 'PoissonHMM'}
        param_keys = ['dt', 'n_states', 'n_trials', 'time_start', 'time_end',
                      'unit_type', 'notes']
        df = sorted_df[sorted_df.notes.str.contains('sequential')]
        qstr = ' and '.join(['{} == "{}"'.format(k, v) for k, v in req_params.items()])
        df = df.query(qstr)
        for (exp_name, rec_group, taste), group in df.groupby(['exp_name',
                                                               'rec_group',
                                                               'taste']):
            p_grp = group.query('unit_type == "pyramidal"')
            s_grp = group.query('unit_type == "single"')

            if len(s_grp) > 0:
                all_cells = s_grp['n_cells'].unique()[0]
                tmp_grp = group.query('unit_type == "single"')
                tmp_grp = tmp_grp[tmp_grp.notes.str.contains('fixed 2')]
            else:
                print('Skipping %s %s %s' % (exp_name, rec_group, taste))
                continue

            if len(p_grp) > 0:
                pyr_cells = p_grp['n_cells'].unique()[0]
            else:
                pyr_cells = all_cells

            if pyr_cells != all_cells:
                print('Plotting pyramidal for %s %s %s' % (exp_name, rec_group, taste))
                pyr_grp = group.query('unit_type == "pyramidal"')
                if len(pyr_grp) > 0 and len(pyr_grp) < 3:
                    tmp_grp = tmp_grp.copy()
                    tmp_grp = tmp_grp.append(pyr_grp.copy())
                elif len(pyr_grp) > 3:
                    g2 = pyr_grp.query('n_states == 2')['log_likelihood'].idxmax()
                    g3 = pyr_grp.query('n_states == 3')['log_likelihood'].idxmax()
                    pyr_grp = pyr_grp.loc[[g2, g3]]
                    tmp_grp = tmp_grp.copy()
                    tmp_grp = tmp_grp.append(pyr_grp.copy())
                else:
                    print('No Pyramidal found')

            for i, row in tmp_grp.iterrows():
                fd = os.path.join(plot_dir, '%s_%s_%s' % (exp_name, rec_group, taste))
                fn = '%i-states_%s-cells.png' % (row['n_states'], row['unit_type'])
                if not os.path.isdir(fd):
                    os.makedirs(fd)

                fn = os.path.join(fd, fn)
                plt.plot_hmm(row['rec_dir'], row['hmm_id'], save_file=fn)

    def reset_hmm_sorting(self):
        sorted_df = self.get_sorted_hmms()
        sorted_df['sorting'] = 'rejected'
        sorted_df['sort_method'] = 'auto'
        self.write_sorted_hmms(sorted_df)

    def mark_early_and_late_states(self):
        ho = self.get_sorted_hmms()

        def apply_states(row):
            h5 = hmma.get_hmm_h5(row['rec_dir'])
            hmm, _, _ = phmm.load_hmm_from_hdf5(h5, row['hmm_id'])
            tmp = hmma.choose_bsln_early_late_states(hmm)
            return pd.Series({'bsln_state': tmp[0], 'early_state': tmp[1], 'late_state': tmp[2]})

        ho[['bsln_state', 'early_state', 'late_state']] = ho.apply(apply_states, axis=1)
        self.write_sorted_hmms(ho)

    def mark_poss_epochs(self):
        ho = self.get_sorted_hmms()

        def apply_states(row):
            if row.n_states > 2:
                h5 = hmma.get_hmm_h5(row['rec_dir'])
                hmm, _, _ = phmm.load_hmm_from_hdf5(h5, row['hmm_id'])
                tmp = hmma.find_poss_epoch_states(hmm)
                return pd.Series({'bsln_state': tmp[0], 'early_state': tmp[1], 'late_state': tmp[2]})
            else:
                return pd.Series({'bsln_state': 'nan', 'early_state': 'nan', 'late_state': 'nan'})

        ho[['bsln_state', 'early_state', 'late_state']] = ho.apply(apply_states, axis=1)
        self.write_sorted_hmms(ho)

    def mark_4_states(self):
        ho = self.get_sorted_hmms()

        def apply_states(row):
            h5 = hmma.get_hmm_h5(row['rec_dir'])
            hmm, _, _ = phmm.load_hmm_from_hdf5(h5, row['hmm_id'])
            tmp = hmma.choose_early_late_states(hmm)
            return pd.Series({'early_state': tmp[0], 'late_state': tmp[1]})

        ho[['early_state', 'late_state']] = ho.apply(apply_states, axis=1)
        self.write_sorted_hmms(ho)

    def iterhmms(self, **kwargs):
        ho = self.get_sorted_hmms()
        qstr = ' and '.join([f'{k} == "{v}"' for k, v in kwargs.items()])
        if qstr != '':
            ho = ho.query(qstr)

        for i, row in ho.iterrows():
            h5 = hmma.get_hmm_h5(row['rec_dir'])
            hmm, _, params = phmm.load_hmm_from_hdf5(h5, row['hmm_id'])
            yield i, hmm, params, row

    def sort_hmms_by_params(self,
                            overwrite=False):  # after, look at df and figure out parameter set label I want to analyze
        # then I can use the function process_sorted_hmms() and then it should work
        sorted_df = self.get_sorted_hmms()
        print(sorted_df)
        if sorted_df is None or overwrite:
            sorted_df = self.get_hmm_overview()

        sorted_df['sorting'] = 'rejected'
        sorted_df['sort_method'] = 'auto'
        met_params = sorted_df.query('time_start==-250')
        # 'taste != "Spont"')
        # '(notes == "sequential - final") |' #required params must edit
        #      ' (notes == "sequential - low thresh") |'
        #     ' (notes == "sequential - BIC test")')
        param_cols = ['n_states', 'time_start', 'time_end', 'area',
                      'unit_type', 'dt', 'notes']
        param_num = 0
        for name, group in met_params.groupby(param_cols):
            param_label = ('%i states, %i to %i ms, %s %s units, '
                           'dt = %g, notes: %s' % name)
            summary = ['-' * 80,
                       'parameter #: %i' % param_num,
                       '# of states: %i' % name[0],
                       'time: %i to %i' % (name[1], name[2]),
                       # 'area: %s' % name[3],
                       'unit type: %s' % name[4],
                       'dt: %g' % name[5],
                       # 'n_trials: %i' % name[6],
                       'notes: %s' % name[6],
                       '# of HMMs: %i' % len(group),
                       '-' * 80]
            print('\n'.join(summary))
            # if len(group) < 30:
            #     param_num += 1
            #     continue

            idx = np.array((group.index))
            sorted_df.loc[idx, 'sorting'] = 'params #%i' % param_num
            sorted_df.loc[idx, 'sorting_notes'] = param_label
            param_num += 1

        self.write_sorted_hmms(sorted_df)

    def sort_hmms_by_BIC(self, overwrite=False):
        sorted_df = self.get_sorted_hmms()
        if sorted_df is None or overwrite is True:
            sorted_df = self.get_hmm_overview()
            sorted_df['sorting'] = 'rejected'
            sorted_df['sort_method'] = 'auto'

        df = sorted_df.query('n_states > 2')
        df = df.loc[((df.exp_name != 'DS33') | (df.rec_num != 2) | (df.n_states != 3) | (df.taste != 'CA'))]

        df = df.loc[((df.exp_name != 'DS40') | (df.rec_num != 2) | (
                df.n_states != 3))]  # TODO: get rid of this bullshit hardcoding
        df = df.loc[((df.exp_name != 'DS40') | (df.rec_num != 3) | (df.n_states != 3) | (df.taste != 'NaCl'))]
        df = df.loc[((df.exp_name != 'DS36') | (df.rec_num != 1) | (df.n_states != 3))]
        df = df.loc[((df.exp_name != 'DS36') | (df.rec_num != 2) | (df.n_states != 3))]

        minima = []
        for name, group in df.groupby(['exp_name', 'rec_group', 'taste']):
            minima.append(group.BIC.idxmin())

        sorted_df.loc[minima, 'sorting'] = 'best_BIC'
        self.write_sorted_hmms(sorted_df)

    def sort_hmms_by_AIC(self, overwrite=False):
        sorted_df = self.get_sorted_hmms()
        if sorted_df is None or overwrite is True:
            sorted_df = self.get_hmm_overview()
            sorted_df['sorting'] = 'rejected'
            sorted_df['sort_method'] = 'auto'

        df = sorted_df.query('n_states > 2')

        minima = []
        for name, group in df.groupby(['exp_name', 'rec_group', 'taste']):
            minima.append(group.AIC.idxmin())

        sorted_df.loc[minima, 'sorting'] = 'best_AIC'

        self.write_sorted_hmms(sorted_df)

    def get_NB_decode(self):
        try:
            _, NB_decode, _, _ = self.analyze_NB_ID(overwrite=False)
        except FileNotFoundError:
            print("decode file not found, need to run analyze_NB_ID with overwrite=True first")
            return None

        NB_decode[['Y', 'epoch']] = NB_decode.Y.str.split('_', expand=True)
        NB_decode['taste'] = NB_decode['trial_ID']
        NB_decode['state_num'] = NB_decode['hmm_state'].astype(int)

        NB_decode['state_num'] = NB_decode['hmm_state'].astype('int64')
        NB_decode['epoch'] = NB_decode['epoch'].fillna('prestim')
        NB_decode = add_session_trial(NB_decode, self.project, trial_col='trial_num', trial_id='trial_ID')
        NB_decode['taste_trial'] = NB_decode['trial_num']
        NB_decode = NB_decode.drop(columns=['trial_ID', 'trial_num'])
        return NB_decode

    def get_NB_timing(self):
        try:
            _, _, _, NB_timings = self.analyze_NB_ID(overwrite=False)  # run with overwrite
        except FileNotFoundError:
            print("decode file not found, need to run analyze_NB_ID with overwrite=True first")
            return None

        test = NB_timings.groupby(['exp_name', 'exp_group','time_group', 'taste', 'trial_num', 'state_num']).size().astype('int')

        NB_decode = self.get_NB_decode()
        grcols = ['rec_dir', 'taste_trial', 'taste', 'state_num']
        NB_decsub = NB_decode[grcols + ['p_correct']].drop_duplicates()

        NB_timings['taste_trial'] = NB_timings['trial_num']
        NB_timings = NB_timings.drop(columns=['trial_num'])
        NB_timings = NB_timings.groupby(['rec_dir', 'taste', 'taste_trial']).filter(lambda x: len(x) >= 3)

        NB_timings = NB_timings.merge(NB_decsub, on=grcols, how='left')

        NB_timings = NB_timings.drop_duplicates()
        NB_timings[['Y', 'epoch']] = NB_timings.state_group.str.split('_', expand=True)
        avg_timing = NB_timings.groupby(['exp_name', 'taste', 'state_group']).mean()[
            ['t_start', 't_end', 't_med', 'duration']]
        avg_timing = avg_timing.rename(columns=lambda x: 'avg_' + x).reset_index()

        NB_timings = pd.merge(NB_timings, avg_timing, on=['exp_name', 'taste', 'state_group'],
                              how='left').drop_duplicates()
        NB_timings = NB_timings.reset_index(drop=True)
        # idxcols1 = list(NB_timings.loc[:, 'exp_name':'state_num'].columns)
        # idxcols2 = list(NB_timings.loc[:, 'pos_in_trial':].columns)
        # idxcols = idxcols1 + idxcols2
        # NB_timings = NB_timings.set_index(idxcols)
        # NB_timings = NB_timings.reset_index()
        # NB_timings = NB_timings.set_index(['exp_name', 'taste', 'state_group', 'session_trial', 'time_group'])
        # operating_columns = ['t_start', 't_end', 't_med', 'duration']

        NB_timings = NB_timings.reset_index()
        NB_timings['session_trial'] = NB_timings.session_trial.astype(int)  # make trial number an int
        NB_timings['time_group'] = NB_timings.time_group.astype(int)  # make trial number an int

        # remove all trials with less than 3 states
        return NB_timings

    def get_gamma_sequences(self, sorting='best_AIC'):
        def getmaxgammaprob(row):
            h5_file = get_hmm_h5(row['rec_dir'])
            hmm, time, params = phmm.load_hmm_from_hdf5(h5_file, row["hmm_id"])
            gamma = hmm.stat_arrays['gamma_probabilities']
            gamma_seqs = np.argmax(gamma, axis=1)
            rowids = hmm.stat_arrays['row_id']
            time = hmm.stat_arrays['time']
            colnames = ['hmm_id', 'dig_in', 'taste', 'trial']
            outdf = pd.DataFrame(rowids, columns=colnames)
            nmcols = ['exp_name', 'exp_group', 'rec_group', 'time_group', 'rec_dir']
            outdf[nmcols] = row[nmcols]
            colnames = outdf.columns
            gammadf = pd.DataFrame(gamma_seqs, columns=time)
            outdf = pd.concat([outdf, gammadf], axis=1)
            outdf = pd.melt(outdf, id_vars=colnames, value_vars=list(time), var_name='time', value_name='gamma_state')

            return outdf

        best_hmms = self.get_best_hmms(sorting=sorting)
        out = best_hmms.apply(lambda x: getmaxgammaprob(x), axis=1).tolist()
        out = pd.concat(out)

        out = add_session_trial(out, self.project)
        out['session'] = out.time_group
        out['session_trial'] = out.session_trial.astype(int)

        return out

    def get_gamma_mode(self, sorting='best_AIC', overwrite=False):
        if overwrite is True:
            def getbinstateprob(row):
                h5_file = get_hmm_h5(row['rec_dir'])  # get hmm file name
                hmm, time, params = phmm.load_hmm_from_hdf5(h5_file, row["hmm_id"])  # load hmm
                mode_seqs, mode_gamma, best_seqs = hmma.getModeHmm(hmm)  # get mode state # and gamma prob
                rowids = hmm.stat_arrays['row_id']
                time = hmm.stat_arrays['time']
                colnames = ['hmm_id', 'dig_in', 'taste', 'trial']
                outdf = pd.DataFrame(rowids, columns=colnames)
                nmcols = ['exp_name', 'exp_group', 'rec_group', 'time_group', 'rec_dir']
                outdf[nmcols] = row[nmcols]
                colnames = outdf.columns
                gammadf = pd.DataFrame(mode_gamma, columns=time)
                seqdf = pd.DataFrame(best_seqs, columns=time)
                gdf= pd.concat([outdf, gammadf], axis=1)
                sdf = pd.concat([outdf, seqdf], axis=1)
                gdf = pd.melt(gdf, id_vars=colnames, value_vars=list(time), var_name='time', value_name='gamma_mode')
                sdf = pd.melt(sdf, id_vars=colnames, value_vars=list(time), var_name='time', value_name='state_sequence')
                #merge gdf and sdf by common variables into outdf
                outdf = pd.merge(gdf, sdf, on=['hmm_id', 'dig_in', 'taste', 'trial', 'exp_name', 'exp_group', 'rec_group', 'time_group', 'rec_dir', 'time'], how='outer')
                return outdf

            best_hmms = self.get_best_hmms(sorting=sorting)
            out = best_hmms.apply(lambda x: getbinstateprob(x), axis=1).tolist()
            out = pd.concat(out)
            gamma_mode_df = add_session_trial(out, self.project)  # rebin every 50ms into a bin
            gamma_mode_df['time_bin'] = gamma_mode_df.time.astype(int) / 20
            gamma_mode_df['time_bin'] = gamma_mode_df['time_bin'].astype(int)
            gamma_mode_df['binned_time'] = gamma_mode_df.time_bin * 20
            gamma_mode_df['session'] = gamma_mode_df.time_group  # rename time_group column to session
            gamma_mode_df['session_trial'] = gamma_mode_df.session_trial.astype(int)

            #save the dataframe to a feather file
            mode_state_sf = self.files['pr_mode_state']
            feather.write_feather(gamma_mode_df, mode_state_sf) # save dataframe to feather file

        else:
            mode_state_sf = self.files['pr_mode_state']
            gamma_mode_df = feather.read_feather(mode_state_sf)

        return gamma_mode_df

    def get_avg_gamma_mode(self, sorting='best_AIC', overwrite=False):
        gmdf = self.get_gamma_mode(sorting, overwrite)

        trial_groupings = ['hmm_id','dig_in','trial','exp_name','exp_group','time_group']

        group_variances = gmdf.groupby(trial_groupings)['state_sequence'].var()
        # Get groups where variance is not zero
        groups_with_variance = group_variances[group_variances != 0].index
        # Filter the original dataframe to keep only those groups
        gmdf = gmdf[gmdf.set_index(trial_groupings).index.isin(groups_with_variance)].reset_index()

        gmdf['time'] = gmdf.time.astype(float)
        gmdf = gmdf.loc[gmdf.time > 0].reset_index(drop=True)
        avg_gamma_mode_df = gmdf.groupby(
            ['exp_name', 'exp_group', 'time_group', 'taste', 'trial']).mean().reset_index()
        avg_gamma_mode_df['pr(mode state)'] = avg_gamma_mode_df.gamma_mode
        avg_gamma_mode_df['taste_trial'] = avg_gamma_mode_df.trial.astype(int)
        avg_gamma_mode_df['session_trial'] = avg_gamma_mode_df.session_trial.astype(int)
        return avg_gamma_mode_df

def get_hmm_h5(rec_dir):
    tmp = glob.glob(rec_dir + os.sep + '**' + os.sep + '*HMM_Analysis.hdf5', recursive=True)
    if len(tmp)>1:
        raise ValueError(str(tmp))

    if len(tmp) == 0:
        return None

    return tmp[0]

def organize_hmms(sorted_df, plot_dir):
    req_params = {'dt': 0.001, 'unit_type': 'single', 'time_end': 2000}
    qstr = ' and '.join(['{} == "{}"'.format(k, v) for k, v in req_params.items()])
    sorted_df = sorted_df.query(qstr)
    sorted_df = sorted_df[(sorted_df.notes.str.contains('sequential - final') |
                           sorted_df.notes.str.contains('sequential - fixed 2'))]
    for n_states in np.arange(2, 4):
        txt_fn = os.path.join(plot_dir, '%i-state_hmm_table.txt' % n_states)
        save_dir = os.path.join(plot_dir, '%i-state_HMMs' % n_states)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        df = sorted_df.query('n_states == @n_states')
        # write exp_name, rec_group, hmm_id, taste into TSV file so that I can
        # go through put early and late states in. Split rec_groups by empty
        # space
        with open(txt_fn, 'w') as f:
            for i, row in df.iterrows():
                out_str = '\t'.join([row['exp_name'], row['rec_group'],
                                     str(row['hmm_id']), row['taste'],
                                     str(n_states - 2), str(n_states - 1)])
                print(out_str, file=f)

                fn = '%s_%s_%s_HMM%i.png' % (row['exp_name'], row['rec_group'],
                                             row['taste'], row['hmm_id'])
                fn = os.path.join(save_dir, fn)
                plt.plot_hmm(row['rec_dir'], row['hmm_id'], save_file=fn)


def get_saccharin_consumption(anim_dir):
    '''greabs animal metadata from anim-dir and returns
    mean_saccharin_consumption/mean_water_consumption
    drops CTA Training day.
    '''
    ld = [os.path.join(anim_dir, x) for x in os.listdir(anim_dir) if 'metadata.p' in x]
    if len(ld) == 0:
        return None
    if len(ld) != 1:
        raise ValueError('%i metadata files found. Expected 1.' % len(ld))

    ld = ld[0]
    with open(ld, 'rb') as f:
        dat = pickle.load(f)

    def fix(x):
        if 'Saccharin' in x:
            return 'Saccharin'
        else:
            return 'Water'

    drinks = dat.bottle_tests.iloc[1:].copy()  # Drop first day of water dep
    drinks['Substance'] = drinks['Substance'].apply(fix)
    if not any(drinks.Substance == 'Saccharin'):
        return None

    ctaTrain = [x for x in dat.ioc_tests if 'Train' in x['Test Type']]
    if len(ctaTrain) == 0:
        mean_water = drinks[drinks.Substance == 'Water']['Change (g)'].astype('float').mean()
        mean_sacc = drinks[drinks.Substance == 'Saccharin']['Change (g)'].astype('float').mean()
        return mean_sacc / mean_water

    ctaDay = ctaTrain[0]['Test Time']
    a = ctaDay.replace(hour=0, minute=0, second=0)
    b = ctaDay.replace(hour=23, minute=59, second=59)
    tmp2 = drinks.truncate(after=a).append(drinks.truncate(before=b))
    mean_water = tmp2[tmp2.Substance == 'Water']['Change (g)'].astype('float').mean()
    mean_sacc = tmp2[tmp2.Substance == 'Saccharin']['Change (g)'].astype('float').mean()
    # tmp2['Norm Change'] = (tmp2['Change (g)'] / mean(water))
    return mean_sacc / mean_water


# def apply_consumption_to_project(proj):
#     #if 'saccharin_consumption' not in proj._exp_info.columns:
#     tmp = proj._exp_info['exp_dir'].apply(get_saccharin_consumption)
#     print(tmp)
#     proj._exp_info['saccharin_consumption'] = tmp
#     #else:
#     #    tmp = proj._exp_info['saccharin_consumption']

#     proj._exp_info['CTA_learned'] = (tmp < 0.8)
#     proj._exp_info['cta_group'] = proj._exp_info.CTA_learned.map({True: 'CTA', False: 'No CTA'})
#     proj.save()


class Analysis(object):
    def __init__(self, data_dir, analysis_name, analysis_dir=None):
        pass

    def _check_files(self):
        pass

    def _load_params(self):
        pass

    def _write_params(self):
        pass

    def _update_params(self):
        pass

    def run(self):
        pass


def compare_taste_responses(rec1, unit1, rec2, unit2, params, method='bootstrap'):
    bin_size = params['response_comparison']['win_size']
    step_size = params['response_comparison']['step_size']
    time_start = params['response_comparison']['time_win'][0]
    time_end = params['response_comparison']['time_win'][1]
    baseline_win = params['baseline_comparison']['win_size']
    n_boot = params['response_comparison']['n_boot']
    alpha = params['response_comparison']['alpha']

    dat1 = load_dataset(rec1)
    dat2 = load_dataset(rec2)

    dig1 = dat1.dig_in_mapping.copy().set_index('name')
    dig2 = dat2.dig_in_mapping.copy().set_index('name')
    out_labels = []
    out_pvals = []
    out_ustats = []
    out_diff = []
    out_diff_sem = []
    bin_time = None
    for taste, row in dig1.iterrows():
        ch1 = row['channel']
        ch2 = dig2.loc[taste, 'channel']
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

        if bin_time is None:
            bin_time = t1

        if not np.array_equal(bin_time, t1) or not np.array_equal(t1, t2):
            raise ValueError('Unqueal time vectors')

        if method.lower() == 'anova':
            nt = len(t1)
            pvals = np.ones((nt,))
            ustats = np.zeros((nt,))
            for i, y in enumerate(zip(fr1.T, fr2.T)):
                # u, p = mann_whitney_u(y[0], y[1])
                # Mann-Whitney U gave odd results, trying anova
                u, p = f_oneway(y[0], y[1])
                pvals[i] = p
                ustats[i] = u
        elif method.lower() == 'bootstrap':
            nt = len(t1)
            tmp_lbls = np.vstack(fr1.shape[0] * ['u1'] + fr2.shape[0] * ['u2'])
            tmp_data = np.vstack((fr1, fr2))
            pvals, ustats, _ = stats.permutation_test(tmp_lbls, tmp_data,
                                                      alpha)

        # apply bonferroni correction
        pvals = pvals * nt

        # Compute mean difference using psth parameters
        pt1, psth1, base1 = agg.get_psth(rec1, unit1, ch1, params, remove_baseline=True)
        pt2, psth2, base2 = agg.get_psth(rec2, unit2, ch2, params, remove_baseline=True)
        # Original
        # diff, sem_diff = sas.get_mean_difference(psth1, psth2)
        # Mag diff plot looked odd, trying using same binning as comparison
        # diff, sem_diff = sas.get_mean_difference(fr1, fr2)
        # ^This looked worse, going back
        # Now computing and plotting the difference in zscored firing rates
        # zpsth1 = (psth1-base1[0])/base1[1]
        # zpsth2 = (psth2-base2[0])/base2[1]
        diff, sem_diff = sas.get_mean_difference(psth1, psth2)

        # Store stuff
        out_pvals.append(pvals)
        out_ustats.append(ustats)
        out_diff.append(diff)
        out_diff_sem.append(sem_diff)
        out_labels.append(taste)

    return out_labels, out_pvals, out_ustats, out_diff, out_diff_sem, bin_time, pt1


# get_sorted_hmms eventually gets rec group from here
# def parse_rec(rd): #this gave me the most problems

#     if rd[-1] == os.sep:
#         rd = rd[:-1]
#     parsed = os.path.basename(rd).split('_')
#     exp_name = parsed[0]
#     nm = os.path.basename(rd)    
#     if nm in E1:
#         rec_group = 'Exposure_1'
#     # elif nm in E2:
#     #     rec_group = 'Exposure_2'
#     elif nm in E3:
#         rec_group = 'Exposure_3'

#     # rec_group = parsed[-3]
#     # if rec_group == 'SaccTest':
#     #     rec_group = 'ctaTest'

#     return pd.Series([exp_name, rec_group], index=['exp_name', 'rec_group'])


# def refit_hmms(refit_df, base_params, log_file=None):
#     if not os.path.isfile(log_file):
#         f = open(log_file, 'w')
#         f.close()
# 
#     id_cols = ['taste', 'n_states', 'dt', 'time_start', 'time_end']
#     df = refit_df[['rec_dir', 'hmm_id', *id_cols]]
#     for rec_dir, group in df.groupby('rec_dir'):
#         if log_file is not None:
#             with open(log_file, 'r+') as f:
#                 processed = f.read().split('\n')
#                 if rec_dir in processed:
#                     print('Skipping %s\nAlready Processed' % rec_dir)
#                     continue
#                 else:
#                     f.write(rec_dir + '\n')
# 
#         if ELF_DIR in rec_dir and not os.path.isdir(rec_dir) and os.path.isdir(MONO_DIR):
#             rd = rec_dir.replace(ELF_DIR, MONO_DIR)
#         else:
#             rd = rec_dir
# 
#         print('Processing HMMs for %s' % rd)
#         handler = phmm.HmmHandler(rd)
#         for i, row in group.iterrows():
#             if row['taste'] == 'Water':
#                 continue
# 
#             handler.delete_hmm(**row[id_cols])
#             params = {'n_states': row['n_states'], 'taste': row['taste'],
#                       **base_params.copy()}
#             handler.add_params(params)
# 
#         handler.run()
#
from piecewise_regression import Fit
from multiprocessing import Pool, cpu_count


def fit_group(args):
    try:
        name, group, response_col, trial_col = args
        if len(group) >= 10:
            pw_fit = Fit(group[trial_col].to_list(), group[response_col].to_list(), n_breakpoints=1, tolerance=10 ** -4,
                         max_iterations=100, n_boot=20)
            # Check if the fit has converged
            converged = pw_fit.best_muggeo.converged if pw_fit.best_muggeo else False
            return list(name) + [pw_fit, converged, response_col, trial_col]
        else:
            return None
    except Exception as e:
        print(f"Error in fit_group: {e}")
        return None


def fit_piecewise_regression(df, groups, response_col, trial_col):
    # Create a multiprocessing pool
    with Pool(cpu_count()) as pool:
        # Prepare the arguments for fit_group
        args = [(name, group, response_col, trial_col) for name, group in df.groupby(groups)]
        # Use the pool to run fit_group in parallel
        results = pool.map(fit_group, args)
        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()

    # Filter out None results
    results = [result for result in results if result is not None]

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results, columns=groups + ['pw_fit', 'converged', 'response_col', 'trial_col'])

    return results_df


def find_changepoints_individual(df, model_groups, fit_groups, trial_col, response_col):
    all_results = []
    df = df.copy()
    df['combined_fit_group'] = df[fit_groups].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    df[trial_col] = df[trial_col] + 1
    for combo, sub_df in df.groupby(model_groups):
        print(combo)
        sub_df = sub_df.copy()
        sub_df.sort_values(trial_col, inplace=True)
        best_aic = np.inf
        best_changepoint = None
        best_result = None

        unique_trials = np.unique(sub_df[trial_col])

        best_subdf = []
        for changepoint in unique_trials[1:-1]:
            print(changepoint)
            sub_df["after_split"] = sub_df[trial_col] >= changepoint
            formula = f"{response_col} ~ {trial_col} * after_split"
            model = smf.mixedlm(formula, data=sub_df, groups=sub_df['combined_fit_group'])
            try:
                result = model.fit()
            except:
                continue

            num_params = len(result.params)  # number of parameters
            log_likelihood = result.llf  # log-likelihood of the model
            aic = -2 * log_likelihood + 2 * num_params
            print(aic)

            if aic < best_aic:
                best_aic = aic
                best_changepoint = changepoint
                best_result = result

        best_subdf = sub_df
        try:
            best_subdf['prediction'] = best_result.fittedvalues
        except:
            Exception("warning: no changepoint found for " + str(combo))
        best_subdf['best_changepoint'] = best_changepoint
        all_results.append(best_subdf)

    results_df = pd.concat(all_results)
    return results_df


def find_changepoint(df, trial_col, response_col, model_cols, fit_groups):
    min_aic = np.inf
    best_model = None
    best_changepoint = None

    unique_trials = df[trial_col].unique()
    # Exclude first and last trials, to ensure each group contains at least 2 trials
    models = []
    changepoints = []
    log_likelihoods = []
    aics = []
    for changepoint in unique_trials[1:-1]:
        print(changepoint)

        # Create a new variable representing trials after the changepoint
        df["trial_before"] = np.where(df[trial_col] <= changepoint, df[trial_col], 0)
        df['trial_after'] = np.where(df[trial_col] > changepoint, df[trial_col] - changepoint, 0)

        # Define the model formula
        formula = f"{response_col} ~ {' + '.join(model_cols)} + trial_before + trial_after"

        # Combine fit groups for the mixedlm 'groups' argument
        df['combined_fit_group'] = df[fit_groups].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

        # Fit a mixed model with this new variable
        model = smf.mixedlm(formula, df, groups=df['combined_fit_group'])
        result = model.fit()

        # Calculate AIC and BIC manually
        num_params = len(result.params)  # number of parameters
        num_obs = len(df)  # number of observations
        log_likelihood = result.llf  # log-likelihood of the model

        aic = -2 * log_likelihood + 2 * num_params

        aics.append(aic)
        models.append(result)
        changepoints.append(changepoint)
        log_likelihoods.append(log_likelihood)

        # Compare the AIC of this model with the best one so far
        if aic < min_aic:
            min_aic = aic
            best_model = result
            best_changepoint = changepoint

    print(f"Best changepoint: {best_changepoint}")
    print(best_model.summary())

    return best_model, best_changepoint


import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests


def split_test(df, fit_groups, model_groups, trial_col, value_col):
    results = []

    # sort the dataframe by group columns and trial index, for consistency
    df.sort_values(fit_groups + [trial_col], inplace=True)

    # create a new combined group column for the mixedlm model
    df['combined_model_group'] = df[model_groups].apply(lambda x: '_'.join(x.astype(str)), axis=1)

    # group the DataFrame by fit_groups and apply the test to each group
    for name, group in df.groupby(fit_groups):
        # find the maximum trial index to determine the range of possible splits
        max_trial = group[trial_col].max()

        # Test every possible split
        for i in range(1, max_trial):  # splits at i
            # categorize trials into two groups
            group['group'] = np.where(group[trial_col] <= i, 'first', 'remaining')

            # perform ANOVA
            model = ols(f"{value_col} ~ group", group).fit()
            anova_results = anova_lm(model)

            # create a row dict with result and group info
            row = {fg: n for fg, n in zip(fit_groups, name if isinstance(name, tuple) else (name,))}
            row.update({
                'split': i,
                'p_value': anova_results.loc['group', 'PR(>F)'],  # get p-value of the group effect
                'intercept': model.params['Intercept'],
                'coefficient': model.params['group[T.remaining]']
            })

            results.append(row)

    # create results dataframe
    results_df = pd.DataFrame(results)

    # apply Bonferroni correction
    results_df['reject'], results_df['p_value_corrected'], _, _ = multipletests(results_df['p_value'],
                                                                                method='bonferroni')

    return results_df


# Usage example:
# results = split_test(df, ['subject_group', 'date'], ['subject'], 'trial_index', 'measurement')
# print(results)

import ruptures as rpt
def detect_changepoints(df, group_list, data_col, time_col):
    """
    Detect changepoints in time series data for each grouping and compute the BIC.
    Skip any groups with fewer than 10 observations. Return a DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    group_list : list
        List of column names for grouping.
    data_col : str
        Column name for the data of interest.
    time_col : str
        Column name for the time.
        
    Returns
    -------
    pandas.DataFrame
        A DataFrame where the index is reset and turned into separate columns representing the groups,
        and the other columns are 'changepoints' and 'bic'.
    """
    all_results = []  # Will hold DataFrames for each group

    for key, group in df.groupby(group_list):
        if len(group) < 10:
            continue  # Skip groups with fewer than 10 observations

        model = "l2"  # Cost model, "l2" for signal noise represented by a Gaussian distribution
        algo = rpt.Pelt(model=model).fit(group[data_col].values)  # Fit model to data
        result = algo.predict(pen=1)  # Predict the changepoints. "pen" is the penalty value.
        # Associate changepoints with their corresponding time values
        changepoints_time = [group[time_col].values[i - 1] for i in result[:-1]]

        # Calculate BIC
        n = len(group)  # Number of observations
        k = len(result) - 1  # Number of parameters in the model (number of changepoints)
        residuals = np.concatenate(([group[data_col].values[result[i - 1]:result[i]] -
                                     np.mean(group[data_col].values[result[i - 1]:result[i]])
                                     for i in range(1, len(result))]))
        sigma_sq = np.sum(residuals ** 2) / n  # Estimate of the error variance
        bic = n * np.log(sigma_sq) + k * np.log(n)  # BIC formula

        # Create a DataFrame for this group
        group_df = pd.DataFrame({
            'changepoints': [changepoints_time],
            'bic': [bic]
        }, index=[key])

        all_results.append(group_df)

    # Concatenate all group DataFrames
    results_df = pd.concat(all_results)

    # Reset index and rename columns
    results_df.reset_index(inplace=True)

    # If the index was a MultiIndex, it will be a tuple now
    if isinstance(key, tuple):
        for i, col_name in enumerate(group_list):
            results_df[col_name] = results_df['index'].apply(lambda x: x[i])
        results_df.drop(columns='index', inplace=True)

    return results_df


def add_session_trial(df, proj, trial_col='trial', trial_id='taste'):
    '''Adds session trial number to any df with columns containing rec_dir, trial number (of the taste),
    and taste or channel
    '''
    df = df.copy()
    trial_info = proj.get_trial_info().copy()
    trial_info['taste'] = trial_info['name']
    trial_info['trial'] = trial_info['taste_trial'] - 1
    trial_info['dig_in'] = trial_info['channel']
    trial_info['din'] = trial_info['channel']
    trial_info['session_trial'] = trial_info['trial_num'].astype(int)

    trial_info[trial_col] = trial_info['trial'].astype(int)
    trial_info[trial_id] = trial_info['taste'].astype(str)
    df[trial_col] = df[trial_col].astype(int)
    df[trial_id] = df[trial_id].astype(str)
    trial_info = trial_info.loc[(trial_info.taste != 'Spont')]
    trial_info = trial_info.loc[(trial_info.channel >= 0)]
    trial_info['session_trial'] = trial_info.groupby(['rec_dir'])['session_trial'].transform(lambda x: x - x.min())

    for_merge = trial_info[['rec_dir', trial_col, trial_id, 'session_trial']]

    out = df.merge(for_merge, on=['rec_dir', trial_col, trial_id], how='left')
    out['session_trial'] = out.groupby(['rec_dir'])['session_trial'].transform(lambda x: x - x.min())
    return (out)


def refit_hmms(sorted_df, base_params, all_units, log_file=None, rec_params={}):
    '''re-wrtitten 8/4/20
    edited 8/10/20
    '''
    if not os.path.isfile(log_file):
        f = open(log_file, 'w')
        f.close()

    all_units['rec_dir'] = all_units['rec_dir'].apply(get_local_path)
    needed_df = hmma.make_necessary_hmm_list(all_units)
    best_hmms = hmma.make_best_hmm_list(all_units, sorted_df)
    base_params = {'n_trials': 15, 'unit_type': 'single', 'dt': 0.001,
                   'max_iter': 200, 'n_repeats': 50, 'time_start': -250,
                   'time_end': 2000, 'n_states': 3, 'area': 'GC',
                   'hmm_class': 'PoissonHMM', 'threshold': 1e-6, 'notes': 'sequential - fixed'}
    id_params = ['taste', 'n_trials', 'unit_type', 'dt', 'time_start',
                 'time_end', 'n_states', 'area']
    # for rec, group in needed_df.groupby(['rec_dir']):
    for rec, group in best_hmms.groupby(['rec_dir']):
        if log_file is not None:
            with open(log_file, 'r+') as f:
                processed = f.read().split('\n')
                if rec in processed:
                    print('Skipping %s\nAlready Processed' % rec)
                    continue
                else:
                    f.write(rec + '\n')
                    f.write(LOCAL_MACHINE + '\n')
                    f.write(datetime.now().strftime('%m/%d/%Y %H:%M') + '\n\n')

        print('Processing %s' % rec)
        params = base_params.copy()
        if rec in rec_params.keys():
            for k, v in rec_params[rec].items():
                params[k] = v

        handler = phmm.HmmHandler(rec)
        n_cells = group.n_cells.max()
        if n_cells > 10:
            params['unit_type'] = 'pyramidal'

        plist = [params]
        # for i, row in group.iterrows():
        #     tmp_i = ((sorted_df['rec_dir'] == rec) &
        #              (sorted_df['taste'] == row['taste']))
        #     sortings = sorted_df.loc[tmp_i, 'sorting']
        #     if any(sortings == 'best'):
        #         continue

        #     p = params.copy()
        #     p['taste'] = row['taste']
        #     p['channel'] = row['channel']

        #     handler.delete_hmm(**{x:p[x] for x in id_params})
        #     handler.add_params(p)
        # tastes = group['taste'].to_list()
        # params['taste'] = tastes
        # for i, row in group.iterrows():
        #    if np.isnan(row['hmm_id']):
        #        p = params.copy()
        #        p['taste'] = row['taste']
        #        p['channel'] = row['channel']
        #        plist.append(p)

        # p = params.copy()
        # p['time_end'] = 1750
        # plist.append(p)
        # handler.run(constraint_func=hmma.PI_A_constrained)
        handler.add_params(plist)
        handler.run(constraint_func=hmma.sequential_constrained)

        # for p in plist:
        #     p['notes'] = 'PI & A constrained - fixed'

        # handler.add_params(plist)
        # handler.run(constraint_func=hmma.PI_A_constrained)
        # handler.add_params(params)
        # handler.run(constraint_func=hmma.A_contrained)


import pingouin as pg

def trial_group_anova(df, groups, dv, within, subject='exp_name', trial_col='trial', n_trial_groups=5, trial_split=False, save_dir=None):
    df = df.copy()
    if trial_split:
        df['trial_group'] = df[trial_col] > trial_split
        df['trial_group'] = df['trial_group'].astype(int)
        n_trial_groups = 2
    else:
        df['trial_group'] = pd.cut(df[trial_col], n_trial_groups, labels=False)

    if 'trial_group' not in within:
        within.append('trial_group')

    df['taste_sub'] = df['exp_name'] + '_' + df['taste']

    aovs = []
    pws = []
    diffs = []
    for name, group in df.groupby(groups):
        if len(group) < 2:
            continue
        else:
            group = group.reset_index(drop=True)
            aov = pg.rm_anova(dv=dv, within=within, subject=subject, data=group)
            #pw = pg.pairwise_ttests(dv=dv, within=['trial_group'], subject=['taste_sub'], padjust='holm', data=group)
            pw = pg.pairwise_ttests(dv=dv, within=within, subject=subject, padjust='holm', data=group)

            diff = group.groupby([subject] + within)[dv].mean()
            diff = diff.groupby([subject] + ['taste']).diff().reset_index().dropna()

            aov[groups] = name
            pw[groups] = name
            diff[groups] = name
            extra_cols = ['trial_type', 'n_trial_groups', 'dependent_var', 'trial_split']
            extra_col_data = [trial_col, str(n_trial_groups), dv, trial_split]
            aov[extra_cols] = extra_col_data
            pw[extra_cols] = extra_col_data
            diff[extra_cols] = extra_col_data


            aovs.append(aov)
            pws.append(pw)
            diffs.append(diff)

    aovs = pd.concat(aovs)
    pws = pd.concat(pws)
    diffs = pd.concat(diffs)

    if save_dir is not None:
        if trial_split:
            save_suffix = '%s_%s_%s_%s_%s_%s.csv' % (dv, '_'.join(groups), '_'.join(within), 'trialsplit', str(trial_split), trial_col)
        else:
            save_suffix = '%s_%s_%s_%s_%s.csv' % (dv, '_'.join(groups), '_'.join(within), str(n_trial_groups), trial_col)
        aovname = 'aov_%s' % save_suffix
        posthocname = 'posthoc_%s' % save_suffix
        aovpath = os.path.join(save_dir, aovname)
        posthocpath = os.path.join(save_dir, posthocname)

        aov_cols = ['Source', 'F', 'p-unc', 'trial_type', 'n_trial_groups']
        pw_cols = ['Contrast', 'A', 'B', 'T', 'p-unc', 'trial_type', 'n_trial_groups']

        if len(groups) > 1:
            aov_cols = aov_cols + groups
            pw_cols = pw_cols + groups
        else:
            aov_cols.append(groups[0])
            pw_cols.append(groups[0])

        special_aov = aovs[aov_cols]
        special_aov = special_aov.round(3)
        special_ph = pws[pw_cols]
        special_ph = special_ph.round(3)
        special_aov.to_csv(aovpath)
        special_ph.to_csv(posthocpath)

    return aovs, pws, diffs


def iter_trial_group_anova(df, groups, dep_vars, within, subject='exp_name', trial_cols=None,
                           n_trial_groups=None, save_dir=None, save_suffix=None):
    if type(dep_vars) is not list: # if only one dependent variable is passed, make it a list
        dep_vars = [dep_vars]

    if trial_cols is None: # if no trial columns are passed, use the default
        trial_cols = ['trial', 'session_trial']
    if n_trial_groups is None: # if no n_trial_groups are passed, use the default
        n_trial_groups = [3, 4, 5, 6]

    aov_df = []
    ph_df = []
    diff_df = []
    for k in dep_vars:
        for i in trial_cols:
            for j in n_trial_groups:
                aov, ph, diff = trial_group_anova(df, groups, k, within, subject=subject, trial_col=i, n_trial_groups=j,
                                            save_dir=save_dir)
                aov_df.append(aov)
                ph_df.append(ph)
                diff_df.append(diff)
    aov_df = pd.concat(aov_df, ignore_index=True)
    ph_df = pd.concat(ph_df, ignore_index=True)
    diff_df = pd.concat(diff_df, ignore_index=True)

    if save_suffix is None:
        if len(dep_vars) > 1:
            save_suffix = '_'.join(dep_vars)
        else:
            save_suffix = dep_vars[0]

    if save_dir is not None:
        aov_filename = '%s_all_ANOVA.csv' % save_suffix
        ph_filename = '%s_all_posthoc.csv' % save_suffix
        aov_df.to_csv(os.path.join(save_dir, aov_filename))
        ph_df.to_csv(os.path.join(save_dir, ph_filename))

    return aov_df, ph_df, diff_df

def iter_trial_split_anova(df, groups, dep_vars, within, subject='exp_name', trial_cols=None,
                           n_splits=None, save_dir=None, save_suffix=None):
    if type(dep_vars) is not list: # if only one dependent variable is passed, make it a list
        dep_vars = [dep_vars]

    if trial_cols is None: # if no trial columns are passed, use the default
        trial_cols = ['trial', 'session_trial']

    if n_splits is None: n_splits = 30

    aov_df = []
    ph_df = []
    diff_df = []
    for i in trial_cols:
        min_trial = df[i].min()
        max_trial = df[i].max()
        ntrls = max_trial - min_trial + 1
        dtrls = ntrls / n_splits
        trials = np.arange(min_trial, max_trial, dtrls).astype(int)
        print(trials)
        splits = trials[2:-2]
        print(splits)
        for j in dep_vars:
            for k in splits:
                aov, ph, diff = trial_group_anova(df, groups, j, within, subject=subject, trial_col=i, trial_split=k,
                                            save_dir=save_dir)
                aov_df.append(aov)
                ph_df.append(ph)
                diff_df.append(diff)

    aov_df = pd.concat(aov_df, ignore_index=True)
    ph_df = pd.concat(ph_df, ignore_index=True)
    diff_df = pd.concat(diff_df, ignore_index=True)

    if save_suffix is None:
        if len(dep_vars) > 1:
            save_suffix = '_'.join(dep_vars)
        else:
            save_suffix = dep_vars[0]

    if save_dir is not None:
        aov_filename = '%s_all_trial_split_ANOVA.csv' % save_suffix
        ph_filename = '%s_all_trial_split_posthoc.csv' % save_suffix
        diff_filename = '%s_all_trial_split_diff.csv' % save_suffix
        aov_df.to_csv(os.path.join(save_dir, aov_filename))
        ph_df.to_csv(os.path.join(save_dir, ph_filename))
        diff_df.to_csv(os.path.join(save_dir, diff_filename))

    return aov_df, ph_df, diff_df

def get_local_path(path):
    if ELF_DIR in path and not os.path.isdir(path) and os.path.isdir(MONO_DIR):
        out = path.replace(ELF_DIR, MONO_DIR)
    elif MONO_DIR in path and not os.path.isdir(path) and os.path.isdir(ELF_DIR):
        out = path.replace(MONO_DIR, ELF_DIR)
    else:
        out = path

    return out


def refit_anim(needed_hmms, anim, rec_group=None, custom_params=None):
    df = needed_hmms.query('exp_name == @anim')
    base_params = {'n_trials': 15, 'unit_type': 'single', 'dt': 0.001,
                   'max_iter': 200, 'n_repeats': 50, 'time_start': -250,
                   'time_end': 2000, 'n_states': 3, 'area': 'GC',
                   'hmm_class': 'PoissonHMM', 'threshold': 1e-6,
                   'notes': 'sequential - final'}
    if custom_params is not None:
        for k, v in custom_params.items():
            base_params[k] = v

    for rec_dir, group in df.groupby(['rec_dir']):
        if rec_group is not None and rec_group not in rec_dir:
            continue

        rd = get_local_path(rec_dir)
        handler = phmm.HmmHandler(rd)
        for i, row in group.iterrows():
            p = base_params.copy()
            p['channel'] = row['channel']
            p['taste'] = row['taste']
            handler.add_params(p)
            p2 = p.copy()
            p2['n_states'] = 2
            p2['time_start'] = 0
            handler.add_params(p2)
            p3 = p.copy()
            p3['n_states'] = 2
            p3['time_start'] = 200
            handler.add_params(p3)
            p4 = p.copy()
            p4['time_start'] = -1000
            p4['n_states'] = 4
            handler.add_params(p4)

        print('Fitting %s' % os.path.basename(rec_dir))
        handler.run(constraint_func=hmma.sequential_constrained)


    # def fit_anim_hmms(anim):
    # if anim == 'RN5':
    #     anim = 'RN5b'
    #
    # file_dirs, anim_dir = get_file_dirs(anim)
    # base_params = {'n_trials': 15, 'unit_type': 'single', 'dt': 0.001,
    #                'max_iter': 200, 'n_repeats': 50, 'time_start': -250,
    #                'time_end': 2000, 'n_states': 3, 'area': 'GC',
    #                'hmm_class': 'PoissonHMM', 'threshold': 1e-10,
    #                'notes': 'sequential - low thresh'}
    #
    # for rec_dir in file_dirs:
    #     units = phmm.query_units(rec_dir, 'single', area='GC')
    #     if len(units) < 2:
    #         continue
    #
    #     handler = phmm.HmmHandler(rec_dir)
    #     p = base_params.copy()
    #     handler.add_params(p)
    #     # p2 = p.copy()
    #     # p2['n_states'] = 2
    #     # p2['time_start'] = 0
    #     # handler.add_params(p2)
    #     # p3 = p.copy()
    #     # p3['n_states'] = 2
    #     # p3['time_start'] = 200
    #     # handler.add_params(p3)
    #     # p4 = p.copy()
    #     # p4['time_start'] = -1000
    #     # p4['n_states'] = 4
    #     # handler.add_params(p4)
    #     # if 'ctaTrain' in rec_dir:
    #     #     p5 = p.copy()
    #     #     _ = p5.pop('n_trials')
    #     #     handler.add_params(p5)
    #
    #     dataname = os.path.basename(rec_dir)
    #     with pm.push_alert(success_msg=f'Done fitting for {dataname}'):
    #         print('Fitting %s' % os.path.basename(rec_dir))
    #         if LOCAL_MACHINE == 'StealthElf':
    #             handler.run(constraint_func=hmma.sequential_constrained)
    #         elif LOCAL_MACHINE == 'Mononoke':
    #             with parallel_backend('multiprocessing'):
    #                 handler.run(constraint_func=hmma.sequential_constrained)


#@pm.push_alert(success_msg='Finished fitting HMMs')
# def fit_hmms_for_BIC_test(proj, all_units, min_states=2, max_states=6):
#     base_params = {'n_trials': 15, 'unit_type': 'single', 'dt': 0.001,
#                    'max_iter': 200, 'n_repeats': 50, 'time_start': -250,
#                    'time_end': 2000, 'area': 'GC',
#                    'hmm_class': 'PoissonHMM', 'threshold': 1e-6,
#                    'notes': 'sequential - BIC test'}
#
#     for i, row in proj._exp_info.iterrows():
#         anim = row['exp_name']
#         with pm.push_alert(success_msg=f'Done fiiting for {anim}'):
#             if anim == 'RN5':
#                 anim = 'RN5b'
#
#             file_dirs, anim_dir = get_file_dirs(anim)
#             for rec_dir in file_dirs:
#                 units = all_units.query('area == "GC" and rec_dir == @rec_dir and single_unit == True')
#                 if len(units) < 3:
#                     print('Only %i units for %s. Skipping...' % (len(units), os.path.basename(rec_dir)))
#                     continue
#
#                 handler = phmm.HmmHandler(rec_dir)
#                 for N in range(min_states, max_states):
#                     p = base_params.copy()
#                     p['n_states'] = N
#                     handler.add_params(p)
#
#                 print(f'Fitting {rec_dir}')
#                 if LOCAL_MACHINE == 'StealthElf':
#                     handler.run(constraint_func=hmma.sequential_constrained)
#                 elif LOCAL_MACHINE == 'Mononoke':
#                     with parallel_backend('multiprocessing'):
#                         handler.run(constraint_func=hmma.sequential_constrained)


def final_refits(needed_hmms):
    # TODO: RN24 preCTA Quinine A constrained 2-state, single unit

    refit_anim(needed_hmms, 'RN25', rec_group='ctaTest')
    refit_anim(needed_hmms, 'RN25', rec_group='ctaTrain')
    refit_anim(needed_hmms, 'RN5', rec_group='ctaTest')
    refit_anim(needed_hmms, 'RN5', rec_group='ctaTrain')
    refit_anim(needed_hmms, 'RN27')
    refit_anim(needed_hmms, 'RN17')
    refit_anim(needed_hmms, 'RN18')
    refit_anim(needed_hmms, 'RN21')
    refit_anim(needed_hmms, 'RN24')


def fit_full_hmms(needed_hmms, h5_file):
    base_params = {'n_trials': 15, 'unit_type': 'single', 'dt': 0.001,
                   'max_iter': 500, 'n_repeats': 25, 'time_start': -250,
                   'time_end': 2000, 'n_states': 2, 'area': 'GC',
                   'hmm_class': 'ConstrainedHMM', 'threshold': 1e-5}

    handler = hmma.CustomHandler(h5_file)
    for (exp_name, rec_group, rec_dir), group in needed_hmms.groupby(['exp_name', 'rec_group', 'rec_dir']):
        if rec_group not in ['preCTA', 'postCTA']:
            continue

        params = base_params.copy()
        params['taste'] = group.taste.to_list()
        params['channel'] = group.channel.to_list()
        params['rec_dir'] = rec_dir
        params['notes'] = '%s_%s' % (exp_name, rec_group)
        handler.add_params(params)

    handler.run()


def get_hmm_firing_rate_PCs(best_hmms):
    best = best_hmms.dropna(subset=['hmm_id'])
    out = []
    for (en, rg, rd), group in best.groupby(['exp_name', 'rec_group', 'rec_dir']):
        for i, row in group.iterrows():
            hid = row['hmm_id']
            taste = row['taste']
            h5_file = hmma.get_hmm_h5(rd)
            hmm, time, params = phmm.load_hmm_from_hdf5(h5_file, hid)
            emission = hmm.emissions
            for state, emr in emission.T:
                rates, trials = hmma.get_state_firing_rates(rd, hid, state)
                row_id = [(en, rg, hid, taste, state, 'model')]
                rates = np.vstack(emr, rates)
                tid = [(en, rg, hid, taste, state, t) for t in trials]
                row_id.extend(tid)
                row_id = np.vstack(row_id)

        # TODO: Finish this


def get_hmm_trial_info(HA, sorting='params #5'):
    out = []
    for i, hmm, params, row in HA.iterhmms(sorting=sorting):
        #seqs = hmm.stat_arrays['best_sequences']
        seqs = hmm.stat_arrays['gamma_probabilities']
        seqs = seqs.argmax(axis=1)

        time = hmm.stat_arrays['time']

        # Only looking at the presence of states past t=0
        tidx = np.where(time > 0)[0]
        seqs = seqs[:, tidx]

        es = row['early_state']
        ls = row['late_state']
        if np.isnan(es) or np.isnan(ls):
            continue

        for j, trial in enumerate(seqs):
            tmp = row.copy()
            tmp['trial'] = j
            if es in trial and ls in trial:
                tmp['state_presence'] = 'both'
            elif es in trial:
                tmp['state_presence'] = 'early_only'
            elif ls in trial:
                tmp['state_presence'] = 'late_only'
            else:
                tmp['state_presence'] = 'neither'

            out.append(tmp)

    df = pd.DataFrame(out)
    df = agg.apply_grouping_cols(df, HA.project)
    return df


def apply_unit_firing_rates(df):
    def get_firing_rates(row):
        rec = row['rec_dir']
        unit = row['unit_num']
        t, sa = h5io.get_spike_data(rec, unit)
        dim = h5io.get_digital_mapping(rec, 'in')
        if isinstance(sa, dict):
            spikes = np.vstack(list(sa.values()))
        else:
            spikes = sa

        bidx = np.where(t < 0)[0]
        ridx = np.where(t > 0)[0]
        baseline = np.mean(np.sum(spikes[:, bidx], axis=1))
        response = np.mean(np.sum(spikes[:, ridx], axis=1))
        return pd.Series({'baseline_firing': baseline, 'response_firing': response})

    df[['baseline_firing', 'response_firing']] = df.apply(get_firing_rates, axis=1)
    df['norm_response_firing'] = df.eval('response_firing - baseline_firing')
    return df


def consolidate_results():
    save_dir = os.path.join(os.path.expanduser('~'), 'Dropbox', 'Harmonia',
                            'Share', 'Stk11_Results')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)

    os.mkdir(save_dir)
    sds = {'data': os.path.join(save_dir, 'data'),
           'plots': os.path.join(save_dir, 'plots'),
           'stats': os.path.join(save_dir, 'stats')}

    for k, v in sds.items():
        os.mkdir(v)

    proj_dir = DATA_DIR
    proj = load_project(proj_dir)
    PA = ProjectAnalysis(proj)
    HA = HmmAnalysis(proj)

    d1 = os.path.join(PA.save_dir, 'single_unit_responses')
    d2 = os.path.join(PA.save_dir, 'mds_analysis')
    d3 = os.path.join(HA.save_dir, 'coding_analysis')
    d4 = os.path.join(HA.save_dir, 'confusion_analysis')
    d5 = os.path.join(HA.save_dir, 'timing_analysis')
    d6 = os.path.join(PA.save_dir, 'held_unit_response_changes')
    d7 = os.path.join(d6, 'Held_Unit_Plots')
    d8 = os.path.join(PA.save_dir, 'pca_analysis')
    files = [
        os.path.join(d1, 'unit_taste_responsivity.feather'),
        os.path.join(d1, 'unit_pal_discrim.feather'),
        os.path.join(d1, 'palatability_data.npz'),
        os.path.join(d8, 'pc_data.feather'),
        os.path.join(d8, 'pc_dQ_v_dN_data.feather'),
        # Taste responsive
        os.path.join(d1, 'unit_firing_rates.svg'),
        os.path.join(d1, 'unit_firing_rates.txt'),
        os.path.join(d1, 'taste_responsive.svg'),
        os.path.join(d1, 'taste_responsive-stats.svg'),
        os.path.join(d1, 'taste_responsive-stats.txt'),
        # Pal responsive
        os.path.join(d1, 'Mean_Spearman.svg'),
        os.path.join(d1, 'Mean_Spearman-comparison.svg'),
        os.path.join(d1, 'Mean_Spearman-comparison.txt'),
        os.path.join(d1, 'Mean_Spearman-simple_comparison.svg'),
        os.path.join(d1, 'Mean_Spearman-simple_comparison.txt'),
        os.path.join(d1, 'Mean_Spearman-differences.svg'),
        os.path.join(d1, 'palatability_spearman_corr.svg'),
        os.path.join(d1, 'palatability_spearman_corr.txt'),
        # Taste discriminative
        os.path.join(d1, 'taste_discriminative.svg'),
        os.path.join(d1, 'taste_discriminative_comparison.txt'),
        os.path.join(d1, 'taste_discriminative_comparison.svg'),
        os.path.join(d1, 'Taste_responsive_over_time-Saccharin.svg'),
        # MDS analysis
        os.path.join(d2, 'Saccharin_MDS_distances.svg'),
        os.path.join(d2, 'Saccharin_MDS_distances.txt'),
        os.path.join(d2, 'Saccharin_MDS_distances-alternate.svg'),
        os.path.join(d2, 'Saccharin_MDS_distances-alternate.txt'),
        os.path.join(d2, 'FullDim_MDS_distances.svg'),
        os.path.join(d2, 'FullDim_MDS_distances.txt'),
        # HMM identity and palatability coding
        os.path.join(d3, 'coding_correlations-exclude.svg'),
        os.path.join(d3, 'coding_comparison-exclude.svg'),
        os.path.join(d3, 'coding_comparison-exclude.txt'),
        # HMM identity and palatability confusion
        os.path.join(d4, 'confusion_correlations-exclude.svg'),
        os.path.join(d4, 'confusion_comparison-exclude.svg'),
        os.path.join(d4, 'confusion_comparison-exclude.txt'),
        os.path.join(d4, 'confusion_differences.svg'),
        # HMM transition timing
        os.path.join(d5, 'timing_correlations-exclude.svg'),
        os.path.join(d5, 'early_end_distributions.svg'),
        os.path.join(d5, 'early_end_distributions.txt'),
        os.path.join(d5, 'late_start_distributions.svg'),
        os.path.join(d5, 'late_start_distributions.txt'),
        os.path.join(d5, 'Saccharin_early_end_distributions.svg'),
        os.path.join(d5, 'Saccharin_early_end_distributions.txt'),
        os.path.join(d5, 'Citric Acid_early_end_distributions.svg'),
        os.path.join(d5, 'Citric Acid_early_end_distributions.txt'),
        os.path.join(d5, 'Quinine_early_end_distributions.svg'),
        os.path.join(d5, 'Quinine_early_end_distributions.txt'),
        os.path.join(d5, 'NaCl_early_end_distributions.svg'),
        os.path.join(d5, 'NaCl_early_end_distributions.txt'),
        os.path.join(d5, 'Saccharin_late_start_distributions.svg'),
        os.path.join(d5, 'Saccharin_late_start_distributions.txt'),
        os.path.join(d5, 'Saccharin_timing_comparison-exclude.svg'),
        os.path.join(d5, 'Saccharin_timing_comparison-exclude.txt'),
        os.path.join(HA.save_dir, 'HMM_Median_Gamma_Probs.png'),
        os.path.join(HA.save_dir, 'HMM_Median_Gamma_Probs-exclude.svg'),
        os.path.join(HA.save_dir, 'HMM_parameters.txt'),
        os.path.join(HA.save_dir, 'hmm_trial_breakdown.svg'),
        os.path.join(HA.save_dir, 'hmm_trial_breakdown.txt'),
        os.path.join(proj_dir, 'Stk11_Project_project.p'),
        *HA.files.values(),
        *PA.files.values(),
        HA.files['hmm_timings'].replace('.feather', '.txt'),
        os.path.join(HA.save_dir, 'bic_comparison.svg'),
        os.path.join(HA.save_dir, 'Saccharin_Sequences.svg'),
        # Held Unit Response changes
        os.path.join(d6, 'Saccharin_responses_changed-exclude.txt'),
        os.path.join(d6, 'Saccharin_responses_changed-Cre_v_BadGFP.txt'),
        os.path.join(d6, 'Saccharin_responses_changed-GFP_v_GFP.txt'),
        os.path.join(d7, 'Saccharin_responses_changed-exclude.svg'),
        os.path.join(d7, 'Saccharin_responses_changed-Cre_v_BadGFP.svg'),
        os.path.join(d7, 'Saccharin_responses_changed-GFP_v_GFP.svg'),
        os.path.join(HA.save_dir, 'hmm_trial_breakdown.feather'),
        os.path.join(PA.save_dir, 'Saccharin_consumption.svg'),
        os.path.join(PA.save_dir, 'Saccharin_consumption.txt')
    ]

    ext_map = {'feather': 'data', 'npy': 'data', 'npz': 'data', 'p': 'data',
               'svg': 'plots', 'png': 'plots', 'txt': 'stats', 'json': 'data'}
    missing = []
    for f in files:
        fn, ext = os.path.splitext(f)
        dest = sds[ext_map[ext[1:]]]
        if os.path.isfile(f):
            fn2 = shutil.copy(f, dest)
        else:
            missing.append(f)

        print(f'copied {f} --> {fn2}\n')

    for f in missing:
        print(f'Missing file: {f}\n')


if __name__ == "__main__":
    print('Hello World')
