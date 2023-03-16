import os
import glob
from scipy.ndimage.filters import gaussian_filter1d
from scipy import stats as scistats
from blechpy.analysis import poissonHMM as ph
from blechpy import load_dataset
from blechpy.dio import hmmIO
from blechpy.utils.particles import AnonHMMInfoParticle
from scipy.stats import mode, spearmanr
import numpy as np
import pandas as pd
from itertools import permutations, product
import statsmodels.api as sm
import analysis_stats as stats
from tqdm import tqdm
import pingouin as pg
from joblib import Parallel, delayed, cpu_count
from collections import Counter
import aggregation as agg
import itertools
from statistics import median
from more_itertools import flatten

def deduce_state_order(best_paths):
    '''Looks at best paths and determines the most common ordering of states by
    getting the mode at each position in the sequence order. Return dict that
    has states as keys and order as value
    '''
    n_states = len(np.unique(best_paths))
    trial_orders = [get_simple_order(trial) for trial in best_paths]
    i = 0
    out = {}
    tmp = trial_orders.copy()
    while len(tmp) > 0:
        # get first state in each sequence, unless state has already been assigned an order
        a = [x[0] for x in tmp if x[0] not in out.values()]
        if len(a) == 0:
            tmp = a
            continue

        common = mode(a).mode[0]  # get most common state in this position
        out[i] = common
        # Remove first position
        tmp = [x[1:] for x in tmp if len(x)>1]
        i += 1

    # check that every state is in out
    states = np.unique(best_paths)
    for x in states:
        if x not in out.values():
            out[i] = x
            i += 1

    # Actually I want it flipped so it maps state -> order
    out = {v:k for k,v in out.items()}
    return out


def get_simple_order(seq):
    '''returns order of items in a sequence without repeats, so [1,1,2,2,1,3]
    gives [1,2,3]
    '''
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def get_absolute_order(seq):
    '''returns orders of items in a sequence, so [1,1,2,2,1,3] gives [1,2,1,3]
    '''
    out = [seq[0]]
    for x in seq:
        if x != out[-1]:
            out.append(x)

    return out


def get_state_breakdown(rec_dir, hmm_id, h5_file=None):
    if rec_dir[-1] == os.sep:
        rec_dir = rec_dir[:-1]

    if h5_file is None:
        handler = ph.HmmHandler(rec_dir)
        hmm, hmm_time, params = handler.get_hmm(hmm_id)
    else:
        hmm, hmm_time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)

    hmm_id = params['hmm_id']
    n_states = params['n_states']
    dt = params['dt']
    time_start = params['time_start']
    time_end = params['time_end']
    max_iter = params['max_iter']
    threshold = params['threshold']
    unit_type = params['unit_type']
    channel = params['channel']
    n_trials = params['n_trials']
    spikes, dt, time = ph.get_hmm_spike_data(rec_dir, unit_type, channel,
                                             time_start=time_start,
                                             time_end=time_end, dt=dt,
                                             trials=n_trials)
    best_paths = hmm.stat_arrays['best_sequences'].astype('int')
    state_order = deduce_state_order(best_paths)
    taste = params['taste']
    n_cells = params['n_cells']
    n_iterations = hmm.iteration

    # Parse out rec_group and time_group
    tmp = os.path.basename(rec_dir).split('_')
    rec_group = tmp[-3]
    exp = tmp[0]
    if 'Train' in rec_group or 'pre' in rec_group:
        time_group = 'preCTA'
    elif 'Test' in rec_group or 'post' in rec_group:
        time_group = 'postCTA'

    dat = load_dataset(rec_dir)
    row = {'exp_name': exp, 'rec_dir': rec_dir, 'rec_name': dat.data_name,
           'hmm_id': hmm_id, 'taste': taste, 'channel': channel, 'trial': None,
           'n_states': n_states, 'hmm_state': None, 'ordered_state': None,
           'trial_ordered_state': None, 't_start': None, 't_end': None,
           'duration': None, 'cost': None, 'time_group': time_group,
           'rec_group': rec_group, 'whole_trial': False, 'n_cells': n_cells,
           'recurrence_in_trial': None}

    out = []
    for trial, (trial_path, trial_spikes) in enumerate(zip(best_paths, spikes)):
        trial_order = get_absolute_order(trial_path)
        tmp_path = trial_path.copy()
        tmp_time = hmm_time.copy()
        multiplicity = {x:0 for x in np.unique(trial_path)}
        for i, state in enumerate(trial_order):
            tmp_row = row.copy()
            tmp_row['trial'] = trial
            tmp_row['hmm_state'] = state
            tmp_row['trial_ordered_state'] = i
            tmp_row['ordered_state'] = state_order[state]
            tmp_row['t_start'] = tmp_time[0]
            multiplicity[state] += 1
            tmp_row['recurrence_in_trial'] = multiplicity[state]
            if i == len(trial_order) - 1:
                tmp_row['t_end'] = tmp_time[-1]
                if i == 0:
                    tmp_row['whole_trial'] = True
            else:
                end_idx = np.min(np.where(tmp_path != state))
                tmp_row['t_end'] = tmp_time[end_idx-1]
                # Trim tmp_path and tmp_time
                tmp_path = tmp_path[end_idx:]
                tmp_time = tmp_time[end_idx:]

            tmp_row['duration'] = tmp_row['t_end'] - tmp_row['t_start']

            # Compute cost
            idx = np.where((time >= tmp_row['t_start']) & (time <= tmp_row['t_end']))[0]
            fr = np.sum(trial_spikes[:, idx], axis=1) / (tmp_row['duration']/1000)
            est = hmm.emission[:, state]
            tmp_row['cost'] = np.sum((fr-est)**2)**0.5
            out.append(tmp_row)

    return pd.DataFrame(out)


def check_hmms(hmm_df):
    hmm_df['asymptotic_ll'] = hmm_df.apply(check_ll_asymptote)


def check_ll_asymptote(row):
    thresh = 1e-3
    rec_dir = row['rec_dir']
    hmm_id = row['hmm_id']
    n_iter = row['n_iterations']-1
    h5_file = get_hmm_h5(rec_dir)
    hmm, time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
    ll_hist = hmm.stat_arrays['fit_LL']
    filt_ll = gaussian_filter1d(ll_hist, 4)
    # TODO: Finish this
    diff_ll = np.diff(filt_ll)
    if len(ll_hist) == 0:
        return 'no_hist'

    # Linear fit, if overall trend is decreasing, it fails
    z = np.polyfit(range(len(ll_hist)), filt_ll, 1)
    if z[0] <= 0:
        return 'decreasing'

    # Check if it has plateaued
    if all(np.abs(diff_ll[n_iter-5:n_iter]) <= thresh):
        return 'plateau'

    # if its a maxima and hasn't plateaued it needs to continue fitting
    if np.max(filt_ll) == filt_ll[n_iter]:
        return 'increasing'

    return 'flux'

def get_early_and_late_firing_rates(rec_dir, hmm_id, early_state, late_state, units=None):
    '''Early state gives the firing rate during the first occurence of that
    state in each trial. Late state gives the instance of that state that
    occurs after the early state in each trial. Trials that are all 1 state are
    dropped and trials where the late state does not occur at all after the
    early state
    '''
    h5_file = get_hmm_h5(rec_dir)
    hmm , time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
    channel = params['channel']
    n_trials = params['n_trials']
    t_start = params['t_start']
    t_end = params['t_end']
    dt = params['dt']
    if units is None:
        units = params['unit_type']

    spike_array, dt, s_time = ph.get_hmm_spike_data(rec_dir, units,
                                                      channel,
                                                      time_start=t_start,
                                                      time_end=t_end, dt=dt,
                                                      trials=n_trials)
    # spike_array is trial x neuron x time
    n_trials, n_cells, n_steps = spike_array.shape
    early_rates = []
    late_rates = []
    dropped_trials = []
    labels = [] # trial, early_start, early_end, late_start, late_end
    for trial, (spikes, path) in enumerate(zip(spike_array, hmm.stat_arrays['best_sequences'])):
        if not early_state in path or not late_state in path:
            dropped_trials.append(trial)
            continue

        if len(np.unique(path)) == 1:
            dropped_trials.append(trial)
            continue

        # only grab first instance of early state
        eidx = np.where(path == early_state)[0]
        lidx = np.where(path == late_state)[0]
        ei1 = eidx[0]  # First instance of early state
        if not any(lidx > ei1):
            # if not late state after early state, drop trial
            dropped_trials.append(trial)
            continue

        idx2 = np.where(path != early_state)[0]
        if len(idx2) == 0 or not any(idx2 > ei1):
            ei2 = len(path)-1
        else:
            ei2 = np.min(idx2[idx2>ei1])

        li1 = np.min(lidx[lidx > ei2])
        idx2 = np.where(path != late_state)[0]
        if len(idx2) == 0 or not any(idx2 > li1):
            li2 = len(path) - 1
        else:
            li2 = np.min(idx2[idx2 > li1])

        et1 = time[ei1]
        et2 = time[ei2]
        lt1 = time[li1]
        lt2 = time[li2]
        labels.append((trial, et1, et2, lt1, lt2))
        e_si = np.where((s_time >= et1) & (s_time < et2))[0]
        l_si = np.where((s_time >= lt1) & (s_time < lt2))[0]
        e_tmp = np.sum(spikes[:, e_si], axis=-1) / (dt*len(e_si))
        l_tmp = np.sum(spikes[:, l_si], axis=-1) / (dt*len(l_si))
        early_rates.append(e_tmp)
        late_rates.append(l_tmp)

    out_labels = np.array(labels)
    early_out = np.array(early_rates)
    late_out = np.array(late_rates)
    return out_labels, early_out, late_out

def write_id_pal_to_text(save_file, group, early_id=None,
                         late_id=None, early_pal=None, late_pal=None,
                         label=None):
    rec_dir = group.rec_dir.unique()[0]
    rec_name = os.path.basename(rec_dir).split('_')
    rec_name = '_'.join(rec_name[:-2])
    info_table = []
    for i, row in group.iterrows():
        rec_dir = row['rec_dir']
        hmm_id = row['hmm_id']
        h5_file = get_hmm_h5(rec_dir)
        hmm, time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
        state_map = deduce_state_order(hmm.stat_arrays['best_sequences'])
        # hmm_id, early_state, late_state, early_hmm_state, late_hmm_state
        early_state = row['early_state']
        late_state = row['late_state']
        info_table.append((hmm_id, early_state, late_state))

    info_df = pd.DataFrame(info_table, columns=['hmm_id', 'early_state',
                                                'late_state'])
    out = []
    out.append(rec_name)
    out.append(rec_dir)
    out.append(label)
    out.append('='*80)
    out.append(info_df.to_string(index=False))
    out.append('-'*80)
    out.append('')
    out.append('Naive Bayes Taste Identity Classifier Accuracy')
    out.append('Computed with Leave 1 Out training and testing')
    out.append('Tastes classified: %s' % str(group.taste.to_list()))
    out.append('Early State ID Classification Accuracy: %0.2f' % early_id.accuracy)
    out.append('Late State ID Classification Accuracy: %0.2f' % late_id.accuracy)
    out.append('-'*80)
    out.append('')
    out.append('LDA Palatability Classification')
    out.append('Trained and Tested with all data points')
    out.append('*'*80)
    out.append('')
    out.append('Early State Pal Classification Accuracy: %0.2f' % early_pal.accuracy)
    out.append('Late State Pal Classification Accuracy: %0.2f' % late_pal.accuracy)
    # out.append('Early State Regression')
    # out.append(early_pal.summary('palatability'))
    # out.append('')
    # out.append('*'*80)
    # out.append('')
    # out.append('Late State Regression')
    # out.append(late_pal.summary('palatability'))
    with open(save_file, 'w') as f:
        f.write('\n'.join(out))


## Helper Functions ##

def analyze_state_correlations(timings,groupings,xfeat,features):
    timings = timings.loc[timings.single_state ==False]
    df = pd.DataFrame()
    feats = []
    corrs = []
    p_values = []
    nms = []

    grouped_timings = timings.groupby(groupings)

    for nm, grp in grouped_timings:
        print(grp)

        grp = grp.loc[:,grp.columns != '']

        for feat in grp.columns:
                if feat in features:
                    feats.append(feat)
                    corr, p_value = spearmanr(grp[feat], grp[xfeat])
                    corrs.append(corr)
                    p_values.append(p_value)
                    nms.append(nm)

    df['Feature'] = feats
    df['Correlation'] = corrs
    df['p_value'] = p_values
    df['bonf_pvalue'] = df.p_value*len(features)
    df[groupings] = pd.DataFrame(nms)
    
    return df

def get_state_firing_rates(rec_dir, hmm_id, state, units=None, min_dur=50, max_dur = 3000,
                           remove_baseline=False, other_state=None):
    '''returns an Trials x Neurons array of firing rates giving the mean firing
    rate of each neuron in the first instance of state in each trial

    Parameters
    ----------
    rec_dir : str, recording directory
    hmm_id : int, id number of the hmm to get states from
    state: int, identity of the state
    units: list of str, optional
        which unit names to use. if not provided units are queried based on hmm
        params
    min_dur: int, optional
        minimum duration in ms of a state for it to be used. Default is 50ms.

    Returns
    -------
    np.ndarray : Trials x Neuron matrix of firing rates (rows x cols)

    Raises
    ------

    '''
    h5_file = get_hmm_h5(rec_dir)
    hmm , time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
    channel = params['channel']
    n_trials = params['n_trials']
    t_start = params['time_start']
    t_end = params['time_end']
    dt = params['dt']
    unit_type = params['unit_type']
    area = params['area']
    seqs = hmm.stat_arrays['best_sequences']
    if units is not None:
        unit_type = units
    spike_array, s_dt, s_time = ph.get_hmm_spike_data(rec_dir, unit_type,
                                                      channel,
                                                      time_start=t_start,
                                                      time_end=t_end, dt=dt,
                                                      trials=n_trials, area=area)
    if s_time[0] < 0:
        idx = np.where(s_time < 0)[0]
        prestim = spike_array[:,:, idx]
        prestim = np.sum(prestim, axis=-1) / (s_dt*len(idx))
        baseline = np.mean(prestim, axis=0)
    else:
        baseline = 0

    #state must be present for at least 50ms each post-stimulus
    check_states = [state, other_state] if other_state is not None else [state]
    #check_states = [state]
    valid_trials = agg.get_valid_trials(seqs, check_states, min_pts= 50/(dt*1000), time=time)

    # spike_array is trial x neuron x time
    n_trials, n_cells, n_steps = spike_array.shape
    rates = []
    trial_nums = []
    for trial, (spikes, path) in enumerate(zip(spike_array, seqs)):
        
        if trial not in valid_trials: # Skip if state is not in trial
            continue

        instate = (path == state).astype(int)
        lag = np.insert(instate,0,0)
        lead = np.append(instate,0)
        
        edges = lead-lag
        edgeons = np.where(edges==1)[0]
        edgeoffs = np.where(edges==-1)[0]
        
        statelens = edgeoffs-edgeons
        longest = np.where(statelens==max(statelens))[0][0] #get longest instance of state
        onidx = edgeons[longest]
        offidx = edgeoffs[longest] -1

        t1 = time[onidx]
        t2 = time[offidx]

        si = np.where((s_time >= t1) & (s_time < t2))[0]
        tmp = np.sum(spikes[:, si], axis=-1) / (dt*len(si))
        if remove_baseline and s_time[0] < 0:
            tmp = tmp - baseline
            
        # Skip trial if this particular state is shorter than min_dur
        # This is because a state this short a) can't provide a good firing
        # rate and b) is probably forced by the constraints and not real
        duration = abs(t2-t1)
        min_dur_check = (duration > min_dur)
        max_dur_check = True#(duration < max_dur)
        if state > 0:
            med_check = median([t1,t2]) > 50
            end_check = (t2 > 200)
        else:
            med_check = True
            end_check = True
        
        if not all([min_dur_check, max_dur_check, end_check, med_check]):
            continue
        else:
            trial_nums.append(trial)
            rates.append(tmp)
    
        if any(np.isnan(rates[0])):
            raise Exception("empty spike array")

    return np.array(rates), np.array(trial_nums)

def get_baseline_rates(rec_dir, hmm_id, units=None, min_dur=50, max_dur = 3000,):
    h5_file = get_hmm_h5(rec_dir)
    hmm , time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
    channel = params['channel']
    n_trials = params['n_trials']
    t_start = -250#params['time_start']
    t_end = 0#params['time_end']
    dt = params['dt']
    unit_type = params['unit_type']
    area = params['area']
    if units is not None:
        unit_type = units
    spike_array, s_dt, s_time = ph.get_hmm_spike_data(rec_dir, unit_type,
                                                      channel,
                                                      time_start=t_start,
                                                      time_end=t_end, dt=dt,
                                                      trials=n_trials, area=area)
    


#group is the groupby df
#label_col is the heading of the taste col
#all_units is all_units_table
#TODO: filter out trials that don't have more than n_states-1 states
def get_classifier_data(group, states, label_col, all_units,
                        remove_baseline=False):
    units = get_common_units(group, all_units)
    if units == {}:
        return None, None, None
    
    # if other_state_col is not None:
    #     state2 = 0#row[other_state_col]
    # else:
    #     state2 = None
        
    labels = []
    rates = []
    identifiers = []
    for i, row in group.iterrows():
        rec_dir = row['rec_dir']
        hmm_id = int(row['hmm_id'])
        label = row[label_col]
        b_state = states[label+'_bsln']
        e_state = states[label+'_early']
        l_state = states[label+'_late']
        
        un = units[rec_dir]
        h5_file = get_hmm_h5(rec_dir)
        hmm, time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
        #if error is "selection lists cannot have repeated values" then you have repeated units in all_units, go fix PA.detect_held_units()
        
        #get baseline data:
        tmp_ps_r, tmp_ps_trials = get_state_firing_rates(rec_dir,hmm_id,b_state, 
                                                    units = un,
                                                    remove_baseline = remove_baseline,
                                                    other_state=None)
        tmp_ps_l = np.repeat('prestim', tmp_ps_r.shape[0])
        
        tmp_ps_id = [(rec_dir,hmm_id,row[label_col], x, b_state, True) for x in tmp_ps_trials]
        
        if len(tmp_ps_r) != 0:
            labels.append(tmp_ps_l)
            rates.append(tmp_ps_r)
            identifiers.extend(tmp_ps_id)
        else:
            continue
            
        #get early state data:
        tmp_e_r, tmp_e_trials = get_state_firing_rates(rec_dir, hmm_id, e_state,
                                                   units=un,
                                                   remove_baseline=remove_baseline,
                                                   other_state=None)
        
        tmp_e_l = np.repeat(row[label_col]+'_early', tmp_e_r.shape[0])
        tmp_e_id = [(rec_dir, hmm_id, row[label_col], x, e_state, False) for x in tmp_e_trials]
        
        if len(tmp_e_r) != 0:
            labels.append(tmp_e_l)
            rates.append(tmp_e_r)
            identifiers.extend(tmp_e_id)
        
        #get late state data:
        tmp_l_r, tmp_l_trials = get_state_firing_rates(rec_dir, hmm_id, l_state,
                                                   units=un,
                                                   remove_baseline=remove_baseline,
                                                   other_state=None) 
        tmp_l_l = np.repeat(row[label_col]+'_late', tmp_l_r.shape[0])
        tmp_l_id = [(rec_dir, hmm_id, row[label_col], x, l_state, False) for x in tmp_l_trials]
        
        if len(tmp_l_r) != 0:
            labels.append(tmp_l_l)
            rates.append(tmp_l_r)
            identifiers.extend(tmp_l_id)
            
    # if no valid trials were found for any taste
    if len(rates) == 0:
        return None, None, None

    labels = np.concatenate(labels)
    rates = np.vstack(rates)
    identifiers = np.array(identifiers)  # rec_dir, hmm_id, taste, trial_#
    return labels, rates, identifiers

#the error that happens is that all_units has duplicate entries for DS39, must filter them out at some point
def get_common_units(group, all_units):
    held = np.array(all_units.held_unit_name.unique())
    rec_dirs = group.rec_dir.unique()
    if len(rec_dirs) == 1:
        rd = rec_dirs[0]
        out = {rd: all_units.query('rec_dir == @rd')['unit_name'].to_list()}
        return out

    for rd in group.rec_dir.unique():
        tmp = all_units.query('rec_dir == @rd').dropna(subset=['held_unit_name'])
        units = np.array(tmp['held_unit_name'])
        held = np.intersect1d(held, units)

    out = {}
    if len(held) == 0:
        return out

    for rd in group.rec_dir.unique():
        tmp = all_units[all_units['held_unit_name'].isin(held) &
                        (all_units['rec_dir'] == rd)]
        out[rd] = tmp['unit_name'].to_list()

    return out



def check_single_state_trials(row, min_dur=1):
    '''takes a row from hmm_overview and determines the number of single state decoded paths
    min_dur signifies the minimum time in ms that a state must be present to
    count
    '''
    rec_dir = row['rec_dir']
    hmm_id = row['hmm_id']
    h5_file = get_hmm_h5(rec_dir)
    hmm, time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
    dt = params['dt'] * 1000  # convert from sec to ms
    min_idx = int(min_dur/dt)
    paths = hmm.stat_arrays['best_sequences']
    single_state_trials = 0
    for path in paths:
        info = summarize_sequence(path)
        idx = np.where((info[:,-1] >= min_dur))[0]# & (info[:,-1]))[0]# < 3000))[0]
        if len(idx) < 2:
            single_state_trials += 1

    return single_state_trials


def summarize_sequence(path):
    '''takes a 1-D sequences of categorical info and returns a matrix with
    columns: state, start_idx, end_idx, duration in samples
    '''
    tmp_path = path.copy()
    out = []
    a = np.where(np.diff(path) != 0)[0]
    starts = np.insert(a+1,0,0)
    ends = np.insert(a, len(a), len(path)-1)
    for st, en in zip(starts, ends):
        out.append((path[st], st, en, en-st+1))

    return np.array(out)


def is_sequence_valid(seq, time, early_state, late_state, min_dur=50):
    early_state = int(early_state)
    late_state = int(late_state)
    seq = seq.astype('int')
    dt = np.unique(np.diff(time))[0]
    min_idx = int(min_dur / dt)
    info = summarize_sequence(seq)
    good_seg = np.where(info[:,-1] >= min_idx)[0]
    good_info = info[good_seg, :]
    n_early = np.sum(info[:, 0] == early_state)
    n_late = np.sum(info[:, 0] == late_state)


    # if entrire sequence is one state, reject it
    if len(np.unique(seq)) == 1:
        return False

    # if late state is not in trial, reject it
    if late_state not in seq or late_state not in good_info[:,0]:
        return False

    # if the only instance of the early state is the first state and its
    # duration is less than min_dur, reject it
    if (seq[0] == 'early_state' and
        info[0,-1] < min_idx and
        n_early == 1):
        return False

    # if after the early state there is no late state >min_dur, reject it
    if n_early > 0:
        e1 = np.where(info[:,0] == early_state)[0][0]
        t1 = np.where(good_info[:,1] >= info[e1,2])[0]
        if len(t1) == 0:
            return False

        if late_state not in good_info[t1, 0]:
            return False

    # if after time 0, only the early state is present or neither state is present
    # then reject trial
    idx = np.where(time > 0)[0]
    if (all(seq[idx] == early_state) or
        (early_state not in seq[idx] and late_state not in seq[idx])):
        return False


def is_state_in_seq(seq, state, min_pts=1, time=None):
    '''returns True if given state is present and
    has more than min_pts consecutive points in that state. If time is given,
    this will only consider t>0
    '''
    if time is not None:
        tidx = np.where(time > 0)[0]
        seq = seq.copy()[tidx]

    if state not in seq:
        return False

    summary = summarize_sequence(seq)
    idx = np.where(summary[:,0] == state)[0]
    summary = summary[idx, :]
    if not any(summary[:,-1] >= min_pts):
        return False

    return True

## HMM Organization/Sorting ##

def make_necessary_hmm_list(all_units, min_cells=3, area='GC'):
    df = all_units.query('single_unit == True and area == @area')
    id_cols = ['rec_dir', 'exp_name', 'exp_group', 'rec_group', 'time_group']
    out = []
    for name, group in df.groupby(id_cols):
        if len(group) < min_cells:
            continue

        dat = load_dataset(name[0])
        dim = dat.dig_in_mapping
        n_cells = len(group)
        for i, row in dim.iterrows():
            if (row['exclude']):# or row['name'].lower() == 'Spont'):
                continue
            if row['name'] == 'Spont':
                continue
            tmp = {k: v for k,v in zip(id_cols, name)}
            tmp['taste'] = row['name']
            tmp['channel'] = row['channel']
            tmp['palatability'] = row['palatability_rank']
            tmp['n_cells'] = n_cells
            if tmp['palatability'] < 1:
                # Fix palatability so that Water and Saccharin are -1
                tmp['palatability'] = -1

            out.append(tmp)

    return pd.DataFrame(out)


def make_best_hmm_list(all_units, sorted_hmms, min_cells=3, area='GC', sorting='best'):
    df = make_necessary_hmm_list(all_units, min_cells=min_cells, area=area)
    sorted_hmms = sorted_hmms.query('sorting == @sorting')
    sorted_hmms = sorted_hmms.query('taste != "Spont"')
    hmm_df = sorted_hmms.set_index(['exp_name', 'rec_group', 'taste'])
    def apply_info(row):
        #print(row)
        exp, rec, tst = row[['exp_name', 'rec_group', 'taste']]
        #print(exp,rec,tst)
        hid, srt, ns, nt, es, ls, notes = None, None, None, None, None, None, None
        if (exp, rec, tst) in hmm_df.index: #this condition failing
            print("good")
            hr = hmm_df.loc[exp, rec, tst]
            #hr = hr.iloc[0]
            hid, srt, ns, nt, es, ls, notes = hr[['hmm_id', 'sorting', 'n_states','n_trials','early_state', 'late_state', 'notes']]
        else:
            raise KeyError('missing data at', exp, ' ', rec, ' ', tst)

        return pd.Series({'hmm_id':hid, 'sorting':srt, 'n_states':ns, 'n_trials':nt, 'early_state':es, 'late_state':ls, 'notes': notes})

    df[['hmm_id', 'sorting', 'n_states', 'n_trials', 'early_state', 'late_state', 'notes']] = df.apply(apply_info, axis=1)
    return df


def sort_hmms(df, required_params=None):
    '''Adds four columns to hmm_overview dataframe, [sorting, sort_type, early_state, late_state].
    sorting can be best, reject, or refit. sort_type is "params" if params
    failed to meet requirements or "auto" if rest of this algo sorts them, if
    HMMs are sorted by user this will be "manual"
    '''
    out_df = df.copy()
    if required_params is not None:
        qry = ' and '.join(['{} == "{}"'.format(k,v) for k,v in required_params.items()])
        df = df.query(qry)

    # One HMM for each animal rec and taste
    df = df.query('n_states == 3 or n_states == 2')
    met_params = np.array((df.index))
    out_df['sorting'] = 'rejected'
    out_df['sort_method'] = 'params'
    out_df.loc[met_params, 'sort_method'] = 'auto'
    print('sorting hmms...')
    dfgrp = df.groupby(['exp_name', 'rec_group', 'taste'])
    for name, group in tqdm(dfgrp, total=len(dfgrp)):
        grp = group.query('single_state_trials < 7')
        if len(grp) > 0:
            best_idx = grp.log_likelihood.idxmax()
            out_df.loc[best_idx, 'sorting'] = 'best'
            continue

        grp2 = group.query('single_state_trials >= 7')
        if len(grp2) > 0:
            best_idx = grp2.log_likelihood.idxmax()
            out_df.loc[best_idx, 'sorting'] = 'refit'
        # grp = group.query('ll_check == "plateau" and single_state_trials < 7')
        # # If some HMMs plateaued then pick the one with the best log likelihood
        # if len(grp) > 0:
        #     best_idx = grp.max_log_prob.idxmax()
        #     out_df.loc[best_idx, 'sorting'] = 'best'
        #     continue

        # # otherwise pick an increasing to refit
        # grp = group[group['ll_check'] == 'increasing']
        # if len(grp) > 0:
        #     refit_idx = grp.max_log_prob.idxmax()
        #     out_df.loc[refit_idx, 'sorting'] = 'refit'
        #     continue

        # # otherwise put all in refit
        # out_df.loc[group.index, 'sorting'] = 'refit'

    out_df['early_state'] = np.nan
    out_df['late_state'] = np.nan
    #TODO: Way to choose early and late state
    return out_df


def get_hmm_h5(rec_dir):
    tmp = glob.glob(rec_dir + os.sep + '**' + os.sep + '*HMM_Analysis.hdf5', recursive=True)
    if len(tmp)>1:
        raise ValueError(str(tmp))

    if len(tmp) == 0:
        return None

    return tmp[0]


def sort_hmms_by_rec(df, required_params=None):
    '''Finds best HMMs but uses same parameter set for each recording. Attempts
    tom minimize single state trials and maximize log likelihood
    '''
    out_df = df.copy()
    if required_params is not None:
        qry = ' and '.join(['{} == "{}"'.format(k,v) for k,v in required_params.items()])
        df = df.query(qry)

    df = df[df.notes.str.contains('fix')]
    # One HMM for each animal rec and taste
    met_params = np.array((df.index))
    out_df['sorting'] = 'rejected'
    out_df['sort_method'] = 'params'
    out_df.loc[met_params, 'sort_method'] = 'auto'
    print('sorting hmms...')
    dfgrp = df.groupby(['exp_name', 'rec_group'])
    key_params = ['dt', 'n_states', 'time_start', 'time_end', 'notes', 'unit_type']
    for name, group in tqdm(dfgrp, total=len(dfgrp)):
        tastes = group.taste.unique()
        good_params = []
        ok_params = []
        for pset, subgroup in group.groupby(key_params):
            if not all([x in tastes for x in subgroup.taste]):
                continue

            tmp = {k:v for k,v in zip(key_params, pset)}
            tmp['LL'] = subgroup.log_likelihood.sum()
            if all([x < 7 for x in subgroup.single_state_trials]):
                good_params.append(tmp)
            else:
                ok_params.append(tmp)

        print('Found %i good params for %s' % (len(good_params), '_'.join(name)))
        if len(good_params) > 0:
            idx = np.argmax([x['LL'] for x in good_params])
            best_params = good_params[idx]
        elif len(ok_params) > 0:
            idx = np.argmax([x['LL'] for x in ok_params])
            best_params = ok_params[idx]
        else:
            continue

        _ = best_params.pop('LL')
        qstr = ' and '.join(['{} == "{}"'.format(k,v) for k,v in best_params.items()])
        tmp_df = group.query(qstr)
        best_idx = tmp_df.index
        out_df.loc[best_idx, 'sorting'] = 'best'

    out_df['early_state'] = np.nan
    out_df['late_state'] = np.nan
    #TODO: Way to choose early and late state
    return out_df


## HMM Constraints ##

def PI_A_constrained(PI, A, B):
    '''Constrains HMM to always start in state 0 then move into any other
    state. States are only allowed to transition into higher number states
    '''
    n_states = len(PI)
    PI[0] = 1.0
    PI[1:] = 0.0
    A[-1, :-1] = 0.0
    A[-1, -1] = 1.0
    # This will make states consecutive
    if n_states > 2:
        for i in np.arange(1,n_states-1):
            A[i, :i] = 0.0
            A[i,:] = A[i,:]/np.sum(A[i,:])

    return PI, A, B


def A_contrained(PI, A, B):
    '''Constrains HMM to always start in state 0 then move into any other
    state. States are only allowed to transition into higher number states
    '''
    n_states = len(PI)
    PI[0] = 1.0
    PI[1:] = 0.0
    A[-1, :-1] = 0.0
    A[-1, -1] = 1.0
    # This will make states consecutive
    if n_states > 2:
        for i in np.arange(1,n_states-1):
            A[i, :i] = 0.0
            A[i,:] = A[i,:]/np.sum(A[i,:])

    return PI, A, B


def sequential_constrained(PI, A, B):
    '''Forces all state to occur sequentially
    '''
    n_states = len(PI)
    PI[0] = 1.0
    PI[1:] = 0.0
    for i in np.arange(n_states):
        if i > 0:
            A[i, :i] = 0.0

        if i < n_states-2:
            A[i, i+2:] = 0.0

        A[i, :] = A[i,:]/np.sum(A[i,:])

    A[-1, :] = 0.0
    A[-1, -1] = 1.0

    return PI, A, B

#Rabbit hole: NB_classifier_accuracy>get_classifier_data>get_state_firing_rates
def analyze_NB_state_classification(best_hmms,all_units):
    '''generates list of combinations (state_df) for every possible combination of HMM states 
    and then uses them as the prior for decoding
    Parameters
    ----------
    best_hmms : Produced by HA.get_best_hmms() 
    all_units : Produced by PA.get_unit_info()
    Returns
    -------
    NB_res : list with every single decode
    NB_meta : table with metadata to index NB_res and pull out best decodes from NB_res
    '''
    
    label_col = 'taste'
    id_cols = ['exp_name','exp_group','time_group']
    NB_res = []; metadict = []
    all_units = all_units.query('area == "GC" and single_unit == True')
    best_hmms = best_hmms.dropna(subset=['hmm_id'])
    best_hmms['single_state_trials'] = best_hmms.apply(lambda x: check_single_state_trials(x,min_dur = 50), axis = 1)
    
    
    for name, group in best_hmms.groupby(id_cols):
        state_df = generate_state_combos(group)
        el_tastes = state_df.columns
        el_tastes = list(filter(lambda x: 'bsln' not in x, el_tastes))
        n_trials = dict(zip(group.taste,group.n_trials))
        n_single_state = dict(zip(group.taste,group.single_state_trials))
        en = name[0]; eg = name[1]; tg = name[2]
        
        for i, states in state_df.iterrows():
            #res = NB_classifier_accuracy(group,row,label_col,all_units, other_state)
            
            labels, rates, identifiers = get_classifier_data(group, states, label_col, all_units,
                                                             remove_baseline=False)
            
            if (labels is not None) & (rates is not None):
                model = stats.NBClassifier(labels, rates, row_id=identifiers)
                res = model.leave1out_fit()
            
            if (res is not None) & (labels is not None):
                NB_res.append(res)  
                n_trials_decoded = []
                
                res_frame = pd.DataFrame()
                res_frame['X'] = list(res.X)
                res_frame['Y'] = res.Y
                res_frame['Y_pred'] = res.Y_predicted
                res_frame['row_ID'] = list(res.row_id)
                
                tasteidx = res.Y != 'prestim'
                Y_taste = res.Y[tasteidx]
                Y_pred_taste = res.Y_predicted[tasteidx]
                
                tasteacc = sum(Y_taste==Y_pred_taste)/len(Y_taste)
                
                
                for i in el_tastes:
                    n_dec = np.count_nonzero(res.Y==i)
                    n_trials_decoded.append(n_dec)
                    
                n_trials_missed = sum(group.n_trials*2)-sum(n_trials_decoded)
                n_trials_decoded = dict(zip(el_tastes,n_trials_decoded))
                
                meta_row = {'exp_name':en,
                            'time_group':tg,
                            'exp_group':eg,
                            'accuracies':res.accuracy, 
                            'taste_acc':tasteacc,
                            'hmm_state':states, 
                            'n_trials': n_trials,
                            'n_single_state': n_single_state,
                            'n_trials_dec': n_trials_decoded,
                            'n_trials_missed': n_trials_missed
                            }
                
                meta_row.update(states)
                print('decode:', name,' ',i)
                print(meta_row)
                metadict.append(meta_row)
                
    NB_meta = pd.DataFrame(metadict)
    
    NB_meta['earlyness'] = [x.sum() for x in NB_meta.hmm_state]
    NB_meta['tot_trials'] = pd.DataFrame(list(NB_meta.n_trials)).sum(axis=1)
    NB_meta['performance'] = (1-(NB_meta['n_trials_missed']/NB_meta['tot_trials']))*NB_meta['taste_acc'] #(NB_meta['accuracies']*1E-2)
    
    return NB_res, NB_meta #should do process_NB_classification next


def generate_state_combos(group):
    id_tastes = group.taste
    
    b_list = []
    e_list = []
    l_list = []
    
    for i, row in group.iterrows():
        h5 = get_hmm_h5(row['rec_dir'])
        hmm, _, _ = ph.load_hmm_from_hdf5(h5, row['hmm_id'])
        tmp = find_poss_epoch_states(hmm)
        
        b = np.unique(tmp[:,0])
        e = np.unique(tmp[:,1]) 
        l = np.unique(tmp[:,2])
        
        b_list.append(b)
        e_list.append(e)
        l_list.append(l)
        
    state_list = b_list + e_list + l_list 
    state_combos = itertools.product(*state_list) #we are excluding the first state since the HMM constraint is that first state must be baseline
    
    b_labs = id_tastes+'_bsln'
    e_labs = id_tastes+'_early'
    l_labs = id_tastes+'_late'
    el_tastes = pd.concat([e_labs,l_labs])
    
    taste_labs = pd.concat([b_labs,e_labs,l_labs])
    state_df = pd.DataFrame(state_combos,columns=taste_labs)
    
    for j, taste in enumerate(id_tastes):
        b_tst = b_labs.iloc[j]
        e_tst = e_labs.iloc[j]
        l_tst = l_labs.iloc[j]
        state_df = state_df.loc[(state_df[l_tst] > state_df[e_tst]) & (state_df[e_tst] > state_df[b_tst])]
        
    print(state_df)
    return state_df, el_tastes
    
def nb_group(group, all_units):
    label_col = 'taste'
    [state_df, el_tastes] = generate_state_combos(group)
    n_trials = dict(zip(group.taste,group.n_trials))
    n_single_state = dict(zip(group.taste,group.single_state_trials))
    name = group[['exp_name','exp_group','time_group']].drop_duplicates().values.tolist()[0]
    en = name[0]; eg = name[1]; tg = name[2]
    NB_res, metadict= [], []
    for i, states in state_df.iterrows():

        labels, rates, identifiers = get_classifier_data(group, states, label_col, all_units,
                                                         remove_baseline=False)
        
        if (labels is not None) & (rates is not None):
            model = stats.NBClassifier(labels, rates, row_id=identifiers)
            res = model.leave1out_fit()
            
            
        if (res is not None) & (labels is not None):
            NB_res.append(res)  
            n_trials_decoded = []
            
            res_frame = pd.DataFrame()
            res_frame['X'] = list(res.X)
            res_frame['Y'] = res.Y
            res_frame['Y_pred'] = res.Y_predicted
            res_frame['row_ID'] = list(res.row_id)
            
            tasteidx = res.Y != 'prestim'
            Y_taste = res.Y[tasteidx]
            Y_pred_taste = res.Y_predicted[tasteidx]
            
            tasteacc = sum(Y_taste==Y_pred_taste)/len(Y_taste)
            
            
            for i in el_tastes:
                n_dec = np.count_nonzero(res.Y==i)
                n_trials_decoded.append(n_dec)
                
            n_trials_missed = sum(group.n_trials*2)-sum(n_trials_decoded)
            n_trials_decoded = dict(zip(el_tastes,n_trials_decoded))
            
            meta_row = {'exp_name':en,
                        'time_group':tg,
                        'exp_group':eg,
                        'accuracies':res.accuracy, 
                        'taste_acc':tasteacc,
                        'hmm_state':states, 
                        'n_trials': n_trials,
                        'n_single_state': n_single_state,
                        'n_trials_dec': n_trials_decoded,
                        'n_trials_missed': n_trials_missed
                        }
            
            meta_row.update(states)
            print('decode:', name,' ',i)
            print(meta_row)
            metadict.append(meta_row)
            
    meta = pd.DataFrame(metadict)
    return [res, meta]

#Rabbit hole: NB_classifier_accuracy>get_classifier_data>get_state_firing_rates
def analyze_NB_state_classification_parallel(best_hmms,all_units):
    '''generates list of combinations (state_df) for every possible combination of HMM states 
    and then uses them as the prior for decoding
    Parameters
    ----------
    best_hmms : Produced by HA.get_best_hmms() 
    all_units : Produced by PA.get_unit_info()
    Returns
    -------
    NB_res : list with every single decode
    NB_meta : table with metadata to index NB_res and pull out best decodes from NB_res
    '''
    id_cols = ['exp_name','exp_group','time_group']
    all_units = all_units.query('area == "GC" and single_unit == True')
    best_hmms = best_hmms.dropna(subset=['hmm_id'])
    best_hmms['single_state_trials'] = best_hmms.apply(lambda x: check_single_state_trials(x,min_dur = 50), axis = 1)
    
    groupedBH = best_hmms.groupby(id_cols)
    
    delayed_func = delayed(nb_group)
    res = Parallel(n_jobs = -2)(delayed_func(subgroup, all_units)
                                for name, subgroup in groupedBH)
    
    res_list = [list(x) for x in zip(*res)]
    NB_res = res_list[0]
    NB_meta = pd.concat(res_list[1])
    
    NB_meta['earlyness'] = [x.sum() for x in NB_meta.hmm_state]
    NB_meta['tot_trials'] = pd.DataFrame(list(NB_meta.n_trials)).sum(axis=1)
    NB_meta['performance'] = (1-(NB_meta['n_trials_missed']/NB_meta['tot_trials']))*NB_meta['taste_acc'] #(NB_meta['accuracies']*1E-2)
    
    return NB_res, NB_meta #should do process_NB_classification next


def process_NB_classification(NB_meta,NB_res):
    
    best_NB = NB_meta
    best_NB = best_NB[best_NB.groupby(['exp_name','time_group']).performance.transform('max') == best_NB.performance]
    best_NB = best_NB[best_NB.groupby(['exp_name','time_group']).earlyness.transform('min')==best_NB.earlyness]
    best_NB = best_NB.loc[best_NB.groupby(['exp_name','time_group']).accuracies.idxmin()]
    
    best_res = []
    for index, row in best_NB.iterrows():

        res_row = NB_res[index]
        tastes = np.unique(res_row.Y)
        
        dec_probs = pd.DataFrame(res_row.model.predict_proba(res_row.X),columns = tastes)
        dec_probs[['exp_name','time_group','exp_group']] = row[['exp_name', 'time_group','exp_group']]
        info = pd.DataFrame(res_row.row_id, columns = ['rec_dir','hmm_id','trial_ID','trial_num','hmm_state','prestim_state'])
        Y = pd.DataFrame(res_row.Y, columns = ['Y'])
        dec_probs = pd.concat([dec_probs, info,Y],axis = 1)
        best_res.append(dec_probs)
        
        p_correct = []
        for ind2, row2 in dec_probs.iterrows():
            p_correct.append(row2[row2.Y])
            # if row2['prestim_state'] == 'False':
            #     p_correct.append(row2[row2.Y])
            # else:
            #     p_correct.append(row2.prestim)
            
        dec_probs['p_correct']=p_correct
        
    decode_data = pd.concat(best_res)
    decode_data.prestim_state = decode_data.prestim_state.map({'True':True,'False':False})
    decode_data.time_group = decode_data.time_group.astype(int)
    decode_data.trial_num = decode_data.trial_num.astype(int)
    
    infocols = ['exp_name', 'time_group', 'exp_group']
    full_trials = []
    n_trials = pd.DataFrame(list(best_NB.n_trials))    
    b_trls = n_trials.add_suffix('_bsln')
    e_trls = n_trials.add_suffix('_early')
    l_trls = n_trials.add_suffix('_late')
    all_trials = pd.concat([b_trls,e_trls,l_trls],axis = 1)
    for i, row in all_trials.iterrows():

        row_trials = [np.arange(x) for x in row.values]
        row_trials = pd.DataFrame(row_trials)
        #prestim_trials = row_trials.copy()
        row_trials['Y'] = row.index
        #row_trials['prestim_state'] = False
        row_trials[infocols] = best_NB[infocols].iloc[i]
        # prestim_trials['trial_ID'] = row.index
        # prestim_trials['prestim_state'] = True
        # prestim_trials[infocols] = best_NB[infocols].iloc[i]
        row_trials = row_trials.melt(id_vars = ['Y']+infocols, value_name = 'trial_num')
        #prestim_trials = prestim_trials.melt(id_vars = ['trial_ID','prestim_state']+infocols, value_name = 'trial_num')
        #row_trials = pd.concat([row_trials, prestim_trials])
        full_trials.append(row_trials)

    full_trials = pd.concat(full_trials)
    
    full_trials.pop('variable')
    full_trials = full_trials[~full_trials.trial_num.isna()]
    full_trials.trial_num = full_trials.trial_num.astype(int)
    full_trials.time_group = full_trials.time_group.astype(int)
    
    cols = ['exp_name', 
            'time_group', 
            'exp_group',
            'trial_ID', 
            'prestim_state',
            'rec_dir',
            'hmm_state',
            'hmm_id',
            'Y']
    
    grcols = ['exp_name', 
            'time_group', 
            'exp_group',
            'Y']
    
    cats = decode_data[cols].drop_duplicates()
    cats = cats.set_index(grcols)
    full_trials = full_trials.set_index(grcols)
    
    full_trials = cats.join(full_trials,on=grcols, how = "outer").dropna()
    
    full_trials = full_trials.reset_index()
    decode_data = decode_data.reset_index()
    decode_data.pop('index')
    decode_data = full_trials.merge(decode_data, how = "outer").fillna(0)
    
    ss_trials = []
    rd_id = []
    hid = []
    trls = []
    min_dur = 50
    for nm, grp in decode_data.groupby(['rec_dir', 'hmm_id']):
        rec_dir = nm[0]
        hmm_id = nm[1]
        h5_file = get_hmm_h5(rec_dir)
        hmm, time, params = ph.load_hmm_from_hdf5(h5_file, hmm_id)
        dt = params['dt'] * 1000
        paths = hmm.stat_arrays['best_sequences']
        for trial, path in enumerate(paths):
            info = summarize_sequence(path)
            idx = np.where((info[:,-1] >= min_dur))[0] # & (info[:,-1] < 3000))[0]
            rd_id.append(rec_dir)
            hid.append(hmm_id)
            trls.append(trial)
            if len(idx) < 2:
                ss_trials.append(True)
            else:
                ss_trials.append(False)
                
    ss_dict = {'rec_dir':rd_id, 'hmm_id':hid, 'trial_num': trls, 'single_state': ss_trials}
    ss_frame = pd.DataFrame(ss_dict)        
    decode_data = decode_data.merge(ss_frame, on = ['rec_dir','hmm_id','trial_num'])
    
    return decode_data

def process_LinReg_pal_classification(results):
    columns = ['rec_dir', 'hmm_id', 'Y', 'Y_pred', 'err', 'trial_num','hmm_state', 'iteration']
    df = pd.DataFrame(columns = columns)
    iteration = 0
    idcols = ['rec_dir', 'hmm_id', 'Y', 'trial_num','hmm_state','trash']
    
    mse = []
    accuracies = []
    n_trials = []
    for i in results:
        df1 = pd.DataFrame(i.row_id, columns = idcols)
        df1 = df1.drop(columns=['trash'])
        df1['iteration'] = iteration
        df1['Y_pred'] = i.Y_predicted
        df1['err'] = i.Y_predicted - i.Y
        
        iteration = iteration+1
        df = df.append(df1)
        
        mse.append(i.loss)
        accuracies.append(i.accuracy)
        n_trials.append(len(i.Y))
    
    groups = df.groupby(['iteration','rec_dir','Y','hmm_state']).groups.keys()
    groups = pd.DataFrame(groups, columns = ['iteration','rec_dir','Y','hmm_state'])
    groups.Y = 'pal_' + groups.Y
    groups = groups.pivot(index = ['iteration','rec_dir'], columns = 'Y', values = 'hmm_state').reset_index()
    
    meta = pd.DataFrame(list(zip(mse,accuracies,n_trials)),columns =['mse','accuracies','n_trials'])
    meta = pd.concat([meta,groups],axis=1)
    max_trials = meta.n_trials.max()
    meta['pct_trials'] = meta.n_trials/max_trials
    meta['performance'] = meta.accuracies*(meta.pct_trials*1e-2)
    meta['best_mse'] = meta.groupby(['rec_dir'])['mse'].transform(min)
    meta['score'] = (meta.best_mse/meta.mse) * (meta.pct_trials*1e-2)
    
    best = meta
    # best = best.loc[best.groupby(['rec_dir']).performance.transform('max') == best.performance]
    # best = best.loc[best.groupby(['rec_dir']).mse.transform('min') == best.mse]
    
    best = best.loc[best.groupby(['rec_dir']).score.transform('max') == best.score]
    
    all_trls = df
    all_trls.Y = pd.to_numeric(all_trls.Y)
    all_trls.trial_num = pd.to_numeric(all_trls.trial_num)
    all_trls.Y_pred = pd.to_numeric(all_trls.Y_pred)
    
    all_trls = all_trls.loc[all_trls.iteration.isin(best.iteration)]
    
    return meta, all_trls
#label_col is the heading of taste col
#group is the rows of sorted_hmms for the grouping
#state_df is a df with different states and which state corresponds

#TODO: use spearaman state corr to identify most pal correlated state
#test LDA classifier with 1 axis
#test linear regression classifier with one axis

def LinReg_pal_classification(best_hmms, all_units):
    label_col = 'palatability'
    id_cols = ['exp_name','exp_group','time_group']
    linreg_res = []
    
    all_units = all_units.query('area == "GC" and single_unit == True')
    best_hmms = best_hmms.dropna(subset=['hmm_id'])
    
    for name, group in best_hmms.groupby(id_cols):
        id_pal = group.palatability
        
        state_list = []
        for i in id_pal:
            n_states = group.n_states.loc[group.palatability == i]
            n_states = n_states.iloc[0]
            if n_states == 3:
                sts = [2]
            elif n_states == 4:
                sts = range(2,n_states)
            elif n_states == 5:
                sts = range(3,n_states)

            state_list.append(sts)
        
        state_combos = itertools.product(*state_list) #we are excluding the first state since the HMM constraint is that first state must be baseline
        state_df = pd.DataFrame(state_combos,columns=id_pal)
        n_trials = dict(zip(group.taste,group.n_trials))
        
        en = name[0]; eg = name[1]; tg = name[2]
        
        for i, states in state_df.iterrows():

            labels,rates,identifiers = get_classifier_data(group,states,label_col,all_units,
                                                          remove_baseline = False, other_state_col = None)
            if labels is None: continue
            n_cells = rates.shape[1]
            if n_cells <2: continue
            
            model = stats.LinRegressor(labels,rates,row_id = identifiers)
            results = model.leave1out_fit()
            if results is not None:
                print('success')
                linreg_res.append(results)
                
    return linreg_res

def Spearman_state_corr(best_hmms, all_units):
    label_col = 'palatability'
    id_cols = ['exp_name','exp_group','time_group']
    
    all_units = all_units.query('area == "GC" and single_unit == True')
    best_hmms = best_hmms.dropna(subset=['hmm_id'])
    
    for name, group in best_hmms.groupby(id_cols):
        id_pal = group.palatability
        
        state_list = []
        for i in id_pal:
            n_states = group.n_states.loc[group.palatability == i]
            n_states = n_states.iloc[0]
            if n_states == 3:
                sts = [2]
            elif n_states == 4:
                sts = range(2,n_states)
            elif n_states == 5:
                sts = range(3,n_states)

            state_list.append(sts)
        
        state_combos = itertools.product(*state_list) #we are excluding the first state since the HMM constraint is that first state must be baseline
        state_df = pd.DataFrame(state_combos,columns=id_pal)
        n_trials = dict(zip(group.taste,group.n_trials))
        
        en = name[0]; eg = name[1]; tg = name[2]
        
        rhos = []; ps = []; sts = []
        for i, states in state_df.iterrows():

            labels,rates,identifiers = get_classifier_data(group,states,label_col,all_units,
                                                          remove_baseline = True, other_state_col = None)
            
            for i in rates.transpose():
                rho, p = spearmanr(labels,i)
                rhos.append(rho)
                ps.append(p)
                sts.append(states.transpose())
                
                
    return rhos, ps        
    

def LDA_classifier_accuracy(group, states, label_col, all_units, other_state_col=None):
    '''uses rec_dir and hmm_id to creating firing rate array (trials x cells)
    label_col is the column used to label trials, state_col is used to identify
    which hmm state to use for classification

    Parameters
    ----------
    group : pd.DataFrame
        must have columns: rec_dir, hmm_id and columns that provide the labels
        for classification and the hmm state to be used
    label_col: str, column of dataframe that provides the classification labels
    state_col: str, column of dataframe that provides the hmm state to use
    all_units: pd.DataFrame
        dataframe of all units with columns rec_dir, area, single_unit, held_unit_name

    Returns
    -------
    float : accuracy [0,1]
    '''
    
    def analyze_pal_corr(best_hmms,all_units):
        label_col = 'palatability'
        id_cols = ['exp_name','exp_group','time_group']
        NB_res = []
        metadict = []
        
        all_units = all_units.query('area == "GC" and single_unit == True')
        best_hmms = best_hmms.dropna(subset=['hmm_id'])
        
        for name, group in best_hmms.groupby(id_cols):
            id_pal = group.palatability
            
            state_list = []
            for i in id_pal:
                n_states = group.n_states.loc[group.palatability == i]
                n_states = n_states.iloc[0]
                sts = range(1,n_states)
                state_list.append(sts)
            
            state_combos = itertools.product(*state_list) #we are excluding the first state since the HMM constraint is that first state must be baseline
            state_df = pd.DataFrame(state_combos,columns=id_pal)
            n_trials = dict(zip(group.taste,group.n_trials))
            
            en = name[0]
            eg = name[1]
            tg = name[2]
            
            for i, states in state_df.iterrows():
                #res = Spearman_state_corr(group,states,all_units)
                LDA_classifier_accuracy(group,label_col, states,)
            eg = group.exp_group.iloc[0]

    labels, rates, identifiers = get_classifier_data(group, states, label_col,all_units,
                                                     remove_baseline=True,
                                                     other_state_col=other_state_col)
    if labels is None:
        return None
    
    n_cells = rates.shape[1]
    if n_cells < 2:
        return None
    
    model = stats.LDAClassifier(labels, rates,n_components=1, row_id=identifiers)
    results = model.leave1out_fit()
    return results


def NB_classifier_confusion(group, label_col, state_col, all_units,
                            train_labels=None, test_labels=None, other_state_col=None):
    if len(train_labels) != 2:
        raise ValueError('2 training labels are required for confusion calculations')

    if len(test_labels) != 1:
        raise ValueError('Too many test labels')

    labels, rates, identifiers = get_classifier_data(group,  label_col,
                                                     state_col, all_units,
                                                     remove_baseline=True,
                                                     other_state_col=other_state_col)
    if labels is None:
        return None

    n_cells = rates.shape[1]
    if n_cells < 2:
        return None

    train_idx = np.where([x in train_labels for x in labels])[0]
    test_idx = np.where([x in test_labels for x in labels])[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        return None

    model = stats.NBClassifier(labels[train_idx], rates[train_idx, :],
                               row_id=identifiers[train_idx, :])
    model.fit()
    predictions = model.predict(rates[test_idx, :])
    counts = [len(np.where(predictions == x)[0]) for x in train_labels]
    #return counts
    return 100 * counts[0] / np.sum(counts)  ## returns % nacl
    #return counts[1]/np.sum(counts)
    #q_count = len(np.where(predictions == 'Quinine')[0])
    #return 100 * q_count / len(predictions)


def LDA_classifier_confusion(group, label_col, state_col, all_units,
                            train_labels=None, test_labels=None, other_state_col=None):
    if len(train_labels) != 2:
        raise ValueError('2 training labels are required for confusion calculations')

    if len(test_labels) != 1:
        raise ValueError('Too many test labels')

    labels, rates, identifiers = get_classifier_data(group,  label_col,
                                                     state_col, all_units,
                                                     remove_baseline=True,
                                                     other_state_col=other_state_col)
    if labels is None:
        return None, None

    n_cells = rates.shape[1]
    if n_cells < 2:
        return None, None

    train_idx = np.where([x in train_labels for x in labels])[0]
    test_idx = np.where([x in test_labels for x in labels])[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        return None, None

    model = stats.LDAClassifier(labels[train_idx], rates[train_idx, :],
                                row_id=identifiers[train_idx, :])
    model.fit()
    predictions = model.predict(rates[test_idx, :])
    #counts = [np.sum(predictions == x) for x in train_labels]
    #q_count = len(np.where(predictions == 'Quinine')[0])
    #return 100 * q_count / len(predictions)
    #return counts[1]/np.sum(counts)
    counts = [len(np.where(predictions == x)[0]) for x in train_labels]
    #return counts
    return 100 * counts[0] / np.sum(counts), np.mean(predictions)  ## returns % nacl

#did I run HA.mark_early_and_late_states
def choose_bsln_early_late_states(hmm, bsln_window = [-250,0], early_window=[201,1600], late_window=[1601, 3000]):
    '''picks state that most often appears in the late_window as the late
    state, and the state that most commonly appears in the early window
    (excluding the late state) as the early state.

    Parameters
    ----------
    hmm: phmm.PoissonHMM
    early_window: list of int, time window in ms [start, end]
    late_window: list of int, time window in ms [start, end]

    Returns
    -------
    int, int : early_state, late_state
        early_state = None if it cannout choose one
    '''
    seqs = hmm.stat_arrays['best_sequences']
    time = hmm.stat_arrays['time']
    trial_win = np.where(time > 0)[0]
    
    bidx = np.where((time >= bsln_window[0]) & (time < bsln_window[1]))[0]
    eidx = np.where((time >= early_window[0]) & (time < early_window[1]))[0]
    lidx = np.where((time >= late_window[0]) & (time < late_window[1]))[0]
    #drop single trial states
    good_trials = []
    for i, s in enumerate(seqs):
        if len(np.unique(s[trial_win])) != 1:
            good_trials.append(i)

    good_trials = np.array(good_trials)
    if len(good_trials) == 0:
        return None, None, None

    seqs = seqs[good_trials, :]
    n_trials = seqs.shape[0]

    bbins = list(np.arange(0,hmm.n_states-1))
    ebins = list(np.arange(1, hmm.n_states))
    lbins = list(np.arange(1, hmm.n_states))
    
    bcount = []
    ecount = []
    lcount = []
    
    for i,j,k in zip(bbins, ebins, lbins):
        bcount.append(np.sum(seqs[:, bidx] == i))
        ecount.append(np.sum(seqs[:, eidx] == j))
        lcount.append(np.sum(seqs[:, lidx] == k))

    #lcount, lbins = np.histogram(seqs[:, lidx], np.arange(hmm.n_states+1))
    #ecount, ebins = np.histogram(seqs[:, eidx], np.arange(hmm.n_states+1))
    pairs = [[x,y,z] for x,y,z in product(bbins, ebins,lbins) if (x!=y & y!=z & z!=x) & (x < y < z)]
    pairs = np.array(pairs)
    probs = []
    for x,y,z in pairs:
        i0 = bbins.index(x)
        i1 = ebins.index(y)
        i2 = lbins.index(z)
        p0 = bcount[i0] / n_trials
        p1 = ecount[i1] / n_trials
        p2 = lcount[i2] / n_trials
        probs.append(p0 * p1 * p2)

    best_idx = np.argmax(probs)
    bsln_state, early_state, late_state = pairs[best_idx]

    # early_state = ebins[np.argmax(ecount)]
    # if early_state in lbins:
    #     idx = list(lbins).index(early_state)
    #     lbins = list(lbins)
    #     lbins.pop(idx)
    #     lcount = list(lcount)
    #     lcount.pop(idx)

    # late_state = lbins[np.argmax(lcount)]

    # tmp = 0
    # early_state = None
    # for count, idx in zip(ecount, ebins):
    #     if count > tmp and idx != late_state:
    #         early_state = idx
    #         tmp = count

    return bsln_state, early_state, late_state


def find_poss_epoch_states(hmm, bsln_window = [-250,0], early_window=[201,1600], late_window=[1601, 3000]):
    '''picks state that most often appears in the late_window as the late
    state, and the state that most commonly appears in the early window
    (excluding the late state) as the early state.

    Parameters
    ----------
    hmm: phmm.PoissonHMM
    early_window: list of int, time window in ms [start, end]
    late_window: list of int, time window in ms [start, end]

    Returns
    -------
    int, int : early_state, late_state
        early_state = None if it cannout choose one
    '''
    seqs = hmm.stat_arrays['best_sequences']
    time = hmm.stat_arrays['time']
    trial_win = np.where(time > 0)[0]
    
    bidx = np.where((time >= bsln_window[0]) & (time < bsln_window[1]))[0]
    eidx = np.where((time >= early_window[0]) & (time < early_window[1]))[0]
    lidx = np.where((time >= late_window[0]) & (time < late_window[1]))[0]
    #drop single trial states
    good_trials = []
    for i, s in enumerate(seqs):
        if len(np.unique(s[trial_win])) != 1:
            good_trials.append(i)

    good_trials = np.array(good_trials)
    if len(good_trials) == 0:
        return None, None, None

    seqs = seqs[good_trials, :]

    bbins = list(np.arange(0,hmm.n_states-1))
    ebins = list(np.arange(1, hmm.n_states))
    lbins = list(np.arange(1, hmm.n_states))
    
    bcount = []
    ecount = []
    lcount = []
    
    for i,j,k in zip(bbins, ebins, lbins):
        bcount.append(np.sum(seqs[:, bidx] == i))
        ecount.append(np.sum(seqs[:, eidx] == j))
        lcount.append(np.sum(seqs[:, lidx] == k))
        
    best_bsln_idx = np.argmax(bcount)
    bbins = [bbins[best_bsln_idx]]
    pairs = [[x,y,z] for x,y,z in product(bbins, ebins,lbins) if (x!=y & y!=z & z!=x) & (x < y < z)]
    pairs = np.array(pairs)

    return pairs


## State timing analysis ##
#031320 check this function for bug on early and late state
#031420 check here if "trial_group" makes it in
def analyze_hmm_state_timing(best_hmms, min_dur=1):
    '''create output array with columns: exp_name, exp_group, rec_dir,
    rec_group, time_group, exp_group, taste, hmm_id, trial, state_group,
    state_num, t_start, t_end, duration
    ignore trials with only 1 state > min_dur ms
    '''
    out_keys = ['exp_name', 'exp_group', 'time_group', 'palatability',
                'rec_group', 'rec_dir', 'n_cells', 'taste', 'hmm_id', 'trial',
                'state_group', 'state_num', 't_start', 't_end', 't_med','duration', 'pos_in_trial',
                'unit_type', 'area', 'dt', 'n_states', 'notes', 'valid']
    
    best_hmms = best_hmms.dropna(subset=['hmm_id', 'early_state']).copy()

    best_hmms.loc[:,'hmm_id'] = best_hmms['hmm_id'].astype('int')
    #best_hmms['ID_state'] = best_hmms['ID_state'].astype('int')
    id_cols = ['exp_name', 'exp_group', 'time_group', 'rec_group',
               'rec_dir', 'taste', 'hmm_id', 'palatability']
    param_cols = ['n_cells', 'n_states', 'dt', 'area', 'unit_type', 'notes']
    # State group is early or late
    out = []
    for i, row in best_hmms.iterrows():
        template = dict.fromkeys(out_keys)  
        for k in id_cols:
            template[k] = row[k]

        h5_file = get_hmm_h5(row['rec_dir'])
        hmm, time, params = ph.load_hmm_from_hdf5(h5_file, row['hmm_id'])
        for k in param_cols:
            template[k] = params[k]

        dt = params['dt'] * 1000  # dt in ms
        min_pts = int(min_dur/dt)
        row_id = hmm.stat_arrays['row_id'] # hmm_id, channel, taste, trial, all string
        best_seqs = hmm.stat_arrays['best_sequences']
        for ids, path in zip(row_id, best_seqs):
            tmp = template.copy()
            tmp['trial'] = int(ids[-1])
            tmp['trial_group'] = (lambda x: int(x/5)+1)(tmp['trial'])
            
            id_valid = is_state_in_seq(path, row['early_state'], min_pts=min_pts, time=time)
            #tmp['valid'] = (e_valid and l_valid)
            tmp['valid'] = (id_valid)

            summary = summarize_sequence(path).astype('int')
            # Skip single state trials
            if summary.shape[0] < 2:
                # Instead mark single state trials
                tmp['single_state'] = True
            else:
                tmp['single_state'] = False

            early_flag = False
            late_flag = False
            for j, s_row in enumerate(summary):
                s_tmp = tmp.copy()
                s_tmp['state_num'] = s_row[0]
                s_tmp['t_start'] = time[s_row[1]]
                s_tmp['t_end'] = time[s_row[2]]
                s_tmp['t_med'] = median([s_tmp['t_end'], s_tmp['t_start']])
                s_tmp['duration'] = s_row[3]/dt
                # only first appearance of state is marked as early or late
                if s_row[0] == row['early_state'] and not early_flag:
                    s_tmp['state_group'] = 'ID'
                    ID_flag = True

                s_tmp['pos_in_trial'] = j
                out.append(s_tmp)
            
    return pd.DataFrame(out)

def getModeHmm(hmm):
    best_seqs = hmm.stat_arrays['best_sequences']
    gamma = hmm.stat_arrays['gamma_probabilities']
    mode_seqs, _ = scistats.mode(best_seqs)
    mode_seqs = mode_seqs.flatten().astype(int)
    mode_seqs = np.ravel(mode_seqs)
    mode_gamma = gamma[:,mode_seqs,np.arange(gamma.shape[2])]
    return mode_seqs, mode_gamma

def binstate(best_hmms,trial_group = 5):
    def getbinstateprob(row, trial_group):
        h5_file = get_hmm_h5(row['rec_dir'])
        hmm, time, params = ph.load_hmm_from_hdf5(h5_file, row["hmm_id"])
        mode_seqs, mode_gamma = getModeHmm(hmm)

        rowids = hmm.stat_arrays['row_id']
        time = hmm.stat_arrays['time']
        colnames  =['hmm_id','dig_in','taste','trial']
        outdf = pd.DataFrame(rowids,columns = colnames)
        outdf['trial_group'] = (outdf.trial.astype(int)/trial_group).astype(int)
        nmcols = ['exp_name','exp_group','rec_group','time_group']
        outdf[nmcols] = row[nmcols]
        colnames = outdf.columns
        gammadf = pd.DataFrame(mode_gamma,columns = time) 
        outdf = pd.concat([outdf,gammadf], axis = 1)
        outdf = pd.melt(outdf,id_vars = colnames, value_vars = list(time), var_name = 'time', value_name = 'gamma_mode')
        return(outdf)
    out = best_hmms.apply(lambda x: getbinstateprob(x,5), axis = 1).tolist()
    out = pd.concat(out)
    return out
    
def binwrong(best_hmms, trial_group = 5):
    def getwrongbin(row, trial_group):
        h5_file = get_hmm_h5(row['rec_dir'])
        hmm, time, params = ph.load_hmm_from_hdf5(h5_file, row["hmm_id"])
        mode_seqs, mode_gamma = getModeHmm(hmm) #get mode state # and gamma prob

        rowids = hmm.stat_arrays['row_id']
        time = hmm.stat_arrays['time']
        colnames  =['hmm_id','dig_in','taste','trial']
        outdf = pd.DataFrame(rowids,columns = colnames)
        outdf['trial_group'] = (outdf.trial.astype(int)/trial_group).astype(int)
        nmcols = ['exp_name','exp_group','rec_group','time_group']
        outdf[nmcols] = row[nmcols]
        colnames = outdf.columns
        gammadf = pd.DataFrame(mode_gamma,columns = time) 
        outdf = pd.concat([outdf,gammadf], axis = 1)
        outdf = pd.melt(outdf,id_vars = colnames, value_vars = list(time), var_name = 'time', value_name = 'gamma_mode')

def analyze_classified_hmm_state_timing(best_hmms, decodes, min_dur=1):
    '''create output array with columns: exp_name, exp_group, rec_dir,
    rec_group, time_group, cta_group, taste, hmm_id, trial, state_group,
    state_num, t_start, t_end, duration
    ignore trials with only 1 state > min_dur ms
    '''
    
    #need to add: session trial & deviation from "standard" solution
    
    out_keys = ['exp_name', 'exp_group', 'time_group', 'palatability',
                'rec_group', 'rec_dir', 'n_cells', 'taste', 'hmm_id','trial_num', 'state_num', 't_start', 't_end', 't_med','duration', 'pos_in_trial',
                'unit_type', 'area', 'dt', 'n_states','notes','single_state']
    
    best_hmms = best_hmms.dropna(subset=['hmm_id']).copy()

    best_hmms.loc[:,'hmm_id'] = best_hmms['hmm_id'].astype('int')
    #best_hmms[col_name] = best_hmms[col_name].astype('int')
    id_cols = ['exp_name', 'exp_group', 'time_group', 'rec_group',
               'rec_dir', 'taste', 'hmm_id', 'palatability']
    param_cols = ['n_cells', 'n_states', 'dt', 'area', 'unit_type', 'notes']
    decodes['trial_num'] = decodes.trial_num.astype(int)
    # State group is early or late
    out = []
    for i, row in best_hmms.iterrows():
        template = dict.fromkeys(out_keys)  
        for k in id_cols:
            template[k] = row[k]

        h5_file = get_hmm_h5(row['rec_dir'])
        hmm, time, params = ph.load_hmm_from_hdf5(h5_file, row['hmm_id'])
        for k in param_cols:
            template[k] = params[k]

        dt = params['dt'] * 1000  # dt in ms
        min_pts = int(min_dur/dt)
        row_id = hmm.stat_arrays['row_id'] # hmm_id, channel, taste, trial, all string
        best_seqs = hmm.stat_arrays['best_sequences']
        
        for ids, path in zip(row_id, best_seqs):
            decsub = decodes
            trlno = int(ids[-1])

            # decsub = decsub.loc[(decsub.exp_name == row.exp_name) & (decsub.time_group == int(row.time_group)) &\
            #                     (decsub.trial_num == trlno) & (decsub.hmm_id == row.hmm_id)]
                
            tmp = template.copy()
            tmp['trial_num'] = int(ids[-1])
            tmp['trial_group'] = (lambda x: int(x/5)+1)(tmp['trial_num'])
            #tmp[col_name] = row[col_name]
            
            #id_valid = is_state_in_seq(path, row[col_name], min_pts=min_pts, time=time)
            #tmp['valid'] = (id_valid)
            
            summary = summarize_sequence(path).astype('int')
            # Skip single state trials
            if summary.shape[0] < 2:
                # Instead mark single state trials
                tmp['single_state'] = True
            else:
                tmp['single_state'] = False

            for j, s_row in enumerate(summary):
                s_tmp = tmp.copy()
                s_tmp['state_num'] = s_row[0]
                s_tmp['t_start'] = time[s_row[1]]
                s_tmp['t_end'] = time[s_row[2]]
                s_tmp['t_med'] = median([s_tmp['t_end'], s_tmp['t_start']])
                s_tmp['duration'] = s_row[3]/dt
                        
                s_tmp['pos_in_trial'] = j
                out.append(s_tmp)
    
    
    out = pd.DataFrame(out)
    
    Y_index = decodes[['rec_dir','hmm_id','hmm_state','trial_ID','Y']].drop_duplicates()
    Y_index = Y_index.rename(columns = {'hmm_state':'state_num','trial_ID':'taste','Y':'state_group'})
    Y_index[['hmm_id','state_num']] = Y_index[['hmm_id','state_num']].astype(int)
    out = out.merge(Y_index, on=['rec_dir','hmm_id','state_num','taste'])
    #mask = out.Y=='prestim'
    #out.loc[mask, 'state_group'] = 'prestim'
    #out = out.dropna(subset = ['state_group'])
    out = out.loc[out.duration >= min_dur]
    idx = out.groupby(['rec_dir','palatability','trial_num','state_num'])['duration'].transform(max) == out['duration']
    out = out[idx]
    
    return out

def describe_hmm_state_timings(timing):
    def header(txt):
        tmp = '-'*80 + '\n' + txt + '\n' + '-'*80
        return tmp

    timing = timing.query('valid == True').copy()
    # First look at Saccharin
    sdf = timing.query('taste == "Suc"')
    esdf = sdf.query('state_group == "early"')
    lsdf = sdf.query('state_group == "late"')
    out = []

    out.append(header('Sucrose Early State Analysis'))
    out.append('Animals in data & trial counts')
    out.append(esdf.groupby(['exp_name','exp_group','time_group'])['trial'].count().to_string())
    out.append('')
    out.append('Single State Trials: %i' % esdf.single_state.sum())
    out.append('Single state trials removed for analysis')
    out.append('')
    out.append('Early State End Times')
    out.append('='*80)
    esdf = esdf.query('single_state == False') # drop single state trials
    out.append(esdf.groupby(['exp_group','time_group'])['t_end'].describe().to_string())

    out.append('')
    out.append('Mixed Anova')
    aov = esdf.mixed_anova(dv='t_end', between='exp_group',within ='time_group', subject ='exp_name')
    out.append(aov.to_string(index=False))
    out.append('')
    out.append('*'*80)
    
    out.append('')
    out.append('Early State Durations')
    out.append('='*80)
    out.append(esdf.groupby(['exp_group','time_group'])['duration'].describe().to_string())
    # out.append('')
    # out.append(esdf.groupby(['cta_group',
    #                          'time_group'])['duration'].describe().to_string())
    out.append('')
    out.append('Mixed Anova')
    aov = esdf.mixed_anova(dv='duration', between='exp_group',within='time_group',subject='exp_name')
    out.append(aov.to_string(index=False))

    out.append('')
    out.append('*'*80)


    out.append(header('Suc Late State Analysis'))
    out.append('Animals in data & trial counts')
    out.append(lsdf.groupby(['exp_name', 'exp_group',
                             'time_group'])['trial'].count().to_string())
    out.append('')
    out.append('Single State Trials: %i' % lsdf.single_state.sum())
    out.append('Single state trials removed for analysis')
    out.append('')
    out.append('Late State Start Times')
    out.append('='*80)
    lsdf = lsdf.query('single_state == False') # drop single state trials
    out.append(lsdf.groupby(['exp_group','time_group'])['t_start'].describe().to_string())

    out.append('')
    out.append('Mixed Anova')
    aov = lsdf.mixed_anova(dv='t_start', between='exp_group', within='time_group', subject='exp_name')
    out.append(aov.to_string(index=False))

    out.append('')
    out.append('*'*80)

    out.append('Late State Durations')
    out.append('='*80)
    out.append(lsdf.groupby(['exp_group','time_group'])['duration'].describe().to_string())

    out.append('')
    out.append('Mixed Anova')

    aov = lsdf.mixed_anova(dv='duration', between='exp_group',within='time_group',subject='exp_name')
    out.append(aov.to_string(index=False))

    out.append('')
    out.append('*'*80)

    return '\n'.join(out)


## For fitting Anon's HMMS ##
class CustomHandler(ph.HmmHandler):
    def __init__(self, h5_file):
        self.root_dir = os.path.dirname(h5_file)
        self.save_dir = self.root_dir
        self.h5_file = h5_file

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        self.plot_dir = os.path.join(self.save_dir, 'HMM_Plots')
        if not os.path.isdir(self.plot_dir):
            os.mkdir(self.plot_dir)

        if not os.path.isfile(h5_file):
            hmmIO.setup_hmm_hdf5(h5_file, infoParticle=AnonHMMInfoParticle)

        self.load_params()

    def load_params(self):
        self._data_params = []
        self._fit_params = []
        h5_file = self.h5_file
        if not os.path.isfile(h5_file):
            return

        overview = self.get_data_overview()
        if overview.empty:
            return

        keep_keys = list(ph.HMM_PARAMS.keys())
        keep_keys.append('rec_dir')
        for i in overview.hmm_id:
            _, _, _, _, p = hmmIO.read_hmm_from_hdf5(h5_file, i)
            for k in list(p.keys()):
                if k not in keep_keys:
                    _ = p.pop(k)

            self.add_params(p)

    def add_params(self, params):
        if isinstance(params, list):
            for p in params:
                self.add_params(p)

            return
        elif not isinstance(params, dict):
            raise ValueError('Input must be a dict or list of dicts')

        # Fill in blanks with defaults
        for k, v in ph.HMM_PARAMS.items():
            if k not in params.keys():
                params[k] = v
                print('Parameter %s not provided. Using default value: %s'
                      % (k, repr(v)))

        # Grab existing parameters
        data_params = self._data_params
        fit_params = self._fit_params

        # require additional rec_dir parameter
        if 'rec_dir' not in params.keys():
            raise ValueError('recording directory must be provided')

        # Get taste and trial info from dataset
        dat = load_dataset(params['rec_dir'])
        dat._change_root(params['rec_dir'])
        dim = dat.dig_in_mapping.query('exclude == False and spike_array == True')

        if params['taste'] is None:
            tastes = dim['name'].tolist()
            single_taste = True
        elif isinstance(params['taste'], list):
            tastes = [t for t in params['taste'] if any(dim['name'] == t)]
            single_taste = False
        elif params['taste'] == 'all':
            tastes = dim['name'].tolist()
            single_taste = False
        else:
            tastes = [params['taste']]
            single_taste = True

        dim = dim.set_index('name')
        if not hasattr(dat, 'dig_in_trials'):
            dat.create_trial_list()

        trials = dat.dig_in_trials
        hmm_ids = [x['hmm_id'] for x in data_params]
        if single_taste:
            for t in tastes:
                p = params.copy()

                p['taste'] = t
                # Skip if parameter is already in parameter set
                if any([hmmIO.compare_hmm_params(p, dp) for dp in data_params]):
                    print('Parameter set already in data_params, '
                          'to re-fit run with overwrite=True')
                    continue

                if t not in dim.index:
                    print('Taste %s not found in dig_in_mapping or marked to exclude. Skipping...' % t)
                    continue

                if p['hmm_id'] is None:
                    hid = ph.get_new_id(hmm_ids)
                    p['hmm_id'] = hid
                    hmm_ids.append(hid)

                p['channel'] = dim.loc[t, 'channel']
                unit_names = ph.query_units(dat, p['unit_type'], area=p['area'])
                p['n_cells'] = len(unit_names)
                if p['n_trials'] is None:
                    p['n_trials'] = len(trials.query('name == @t'))

                data_params.append(p)
                for i in range(p['n_repeats']):
                    fit_params.append(p.copy())

        else:
            if any([hmmIO.compare_hmm_params(params, dp) for dp in data_params]):
                print('Parameter set already in data_params, '
                      'to re-fit run with overwrite=True')
                return

            channels = [dim.loc[x,'channel'] for x in tastes]
            params['taste'] = tastes
            params['channel'] = channels

            # this is basically meaningless right now, since this if clause
            # should only be used with ConstrainedHMM which will fit 5
            # baseline states and 2 states per taste
            params['n_states'] = params['n_states']*len(tastes)

            if params['hmm_id'] is None:
                hid = ph.get_new_id(hmm_ids)
                params['hmm_id'] = hid
                hmm_ids.append(hid)

            unit_names = ph.query_units(dat, params['unit_type'],
                                        area=params['area'])
            params['n_cells'] = len(unit_names)
            if params['n_trials'] is None:
                params['n_trials'] = len(trials.query('name == @t'))

            data_params.append(params)
            for i in range(params['n_repeats']):
                fit_params.append(params.copy())

        self._data_params = data_params
        self._fit_params = fit_params

    def run(self, parallel=True, overwrite=False, constraint_func=None):
        h5_file = self.h5_file
        if overwrite:
            fit_params = self._fit_params
        else:
            fit_params = [x for x in self._fit_params if not x['fitted']]

        if len(fit_params) == 0:
            return

        print('Running fittings')
        if parallel:
            n_cpu = np.min((cpu_count()-1, len(fit_params)))
        else:
            n_cpu = 1

        results = Parallel(n_jobs=n_cpu, verbose=100)(delayed(ph.fit_hmm_mp)
                                                     (p['rec_dir'], p, h5_file,
                                                      constraint_func)
                                                     for p in fit_params)


        ph.memory.clear(warn=False)
        print('='*80)
        print('Fitting Complete')
        print('='*80)
        print('HMMs written to hdf5:')
        for hmm_id, written in results:
            print('%s : %s' % (hmm_id, written))

        #self.plot_saved_models()
        self.load_params()

## New confusion Analysis ##
def stratified_shuffle_split(labels, data, repeats, test_label):
    '''generator to split data by unique labels and sample with replacement
    from each to generate new groups of same size. 
    rows of data are observations.

    Returns
    -------
    train_data, train_labels, test_data, test_labels
    '''
    groups = np.unique(labels)
    counts = {}
    datasets = {}
    for grp in groups:
        idx = np.where(labels == grp)[0]
        counts[grp] = idx.shape[0]
        datasets[grp] = data[idx, :]

    rng = np.random.default_rng()
    for i in range(repeats):
        tmp_lbls = []
        tmp_data = []
        for grp in groups:
            N = counts[grp]
            idx = rng.choice(N, N, replace=True)
            tmp_data.append(datasets[grp][idx, :])
            tmp_lbls.extend(np.repeat(grp, N))

        tmp_lbls = np.array(tmp_lbls)
        tmp_data = np.vstack(tmp_data)
        train_idx = np.where(tmp_lbls != test_label)[0]
        test_idx = np.where(tmp_lbls == test_label)[0]
        train = tmp_data[train_idx, :]
        train_lbls = tmp_lbls[train_idx]
        test = tmp_data[test_idx, :]
        test_lbls = tmp_lbls[test_idx]
        yield train, train_lbls, test, test_lbls


def run_classifier(train, train_labels, test, test_labels, classifier=stats.NBClassifier):
    model = classifier(train_labels, train)
    model.fit()
    predictions = model.predict(test)
    accuracy = 100 * sum((x == y for x,y in zip(predictions, test_labels))) / len(test_labels)
    return accuracy, predictions


def saccharin_confusion_analysis(best_hmms, all_units, area='GC',
                                 single_unit=True, repeats=20):
    '''create output dataframe with columns: exp_name, time_group, exp_group,
    cta_group, state_group, ID_confusion, pal_confusion, n_cells, 
    '''
    all_units = all_units.query('(area == @area) and (single_unit == @single_unit)')
    best_hmms = best_hmms.dropna(subset=['hmm_id', 'early_state', 'late_state'])
    out_keys = ['exp_name', 'exp_group', 'time_group', 'cta_group',
                'state_group', 'ID_confusion', 'pal_confusion',
                'pal_counts_nacl', 'pal_counts_ca', 'pal_counts_quinine',
                'n_cells', 'nacl_trials', 'ca_trials', 'quinine_trials', 'sacc_trials']
    template = dict.fromkeys(out_keys)
    id_cols = ['exp_name', 'exp_group', 'time_group']
    state_columns = ['early_state', 'late_state']
    other_state = {'early_state': 'late_state', 'late_state': 'early_state'}
    id_tastes = ['NaCl', 'Quinine', 'Saccharin']
    pal_tastes = ['NaCl', 'Citric Acid', 'Quinine', 'Saccharin']
    pal_map = {'NaCl': 3, 'Citric Acid': 2, 'Quinine': 1, 'Saccharin': -1}
    out = []
    for name, group in best_hmms.groupby(id_cols):
        for state_col in state_columns:
            tmp = template.copy()
            for k,v in zip(id_cols, name):
                tmp[k] = v

            tmp['state_group'] = state_col.replace('_state', '')

            if group.taste.isin(id_tastes).sum() != len(id_tastes):
                continue

            if group.taste.isin(pal_tastes).sum() == len(pal_tastes):
                run_pal = True
            else:
                run_pal = False

            group = group[group.taste.isin(pal_tastes)]

            labels, rates, identifiers = get_classifier_data(group, 'taste',
                                                             state_col,
                                                             all_units,
                                                             remove_baseline=True,
                                                             other_state_col=other_state[state_col])
            trials = Counter(labels)
            if rates is None or any([trials[x] < 2 for x in id_tastes]):
                # if no valid trials were found for NaCl, Quinine or Saccharin
                continue

            tmp['n_cells'] = rates.shape[1]
            tmp['nacl_trials'] = trials['NaCl']
            tmp['ca_trials'] = trials['Citric Acid']
            tmp['quinine_trials'] = trials['Quinine']
            tmp['sacc_trials'] = trials['Saccharin']
            for train, train_lbls, test, test_lbls \
                    in stratified_shuffle_split(labels, rates, repeats, 'Saccharin'):
                row = tmp.copy()
                tst_l = test_lbls.copy()
                tst_l[:] = 'NaCl'
                id_acc, _ = run_classifier(train, train_lbls, test, tst_l,
                                           classifier=stats.NBClassifier)
                row['ID_confusion'] = id_acc
                if run_pal:
                    train_l = np.fromiter(map(pal_map.get, train_lbls), int)
                    tst_l = np.fromiter(map(pal_map.get, tst_l), int)
                    pal_acc, pred = run_classifier(train, train_l, test, tst_l,
                                                   classifier=stats.LDAClassifier)
                    row['pal_confusion'] = pal_acc
                    row['pal_confusion_score'] = np.mean(pred)
                    counts = Counter(pred)
                    row['pal_counts_nacl'] = counts[pal_map['NaCl']]
                    row['pal_counts_ca'] = counts[pal_map['Citric Acid']]
                    row['pal_counts_quinine'] = counts[pal_map['Quinine']]

                out.append(row)

    return pd.DataFrame(out)

    