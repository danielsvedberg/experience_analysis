import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import blechpy
from blechpy import dio
from multiprocessing import Pool

def get_held_resp(PA):
    PA.detect_held_units(overwrite=False)  # this part also gets the all units file
    [all_units, held_df] = PA.get_unit_info(overwrite=False)  # run and check for correct area then run get best hmm

    # check df for nan in HMMID or early or late state
    held_df = held_df[held_df.held != False]
    held_df.reset_index(drop=True, inplace=True)
    id_vars = ['held_unit_name', 'exp_group', 'exp_name', 'held', 'area']
    filter_cols = id_vars + ["rec1", "rec2", "unit1", "unit2"]
    held_df = held_df[filter_cols]
    held_df_long = pd.melt(held_df, id_vars=id_vars, value_vars=['rec1', 'rec2'], value_name='rec_dir',
                           var_name='rec_order')
    held_df_long2 = pd.melt(held_df, id_vars=id_vars, value_vars=['unit1', 'unit2'], value_name='unit_num',
                            var_name='unit_order')
    # drop the unit_order columns
    held_df_long = held_df_long.drop(['rec_order'], axis=1)
    held_df_long2 = held_df_long2.drop(['unit_order'], axis=1)

    held_df_long = pd.concat([held_df_long, held_df_long2], axis=1)

    final_cols = ['held_unit_name', 'exp_group', 'exp_name', 'held', 'rec_dir', 'unit_num']
    held_df_long = held_df_long[final_cols]
    # remove the  duplicated columns
    held_df_long = held_df_long.T.drop_duplicates().T
    held_df_long = held_df_long.drop_duplicates()
    held_df_long['unit_name'] = held_df_long['unit_num']

    resp_units, pal_units = PA.process_single_units(
        overwrite=False)  # run the single-unit analysis, check function to see if all parts are working
    respidxs = resp_units[['rec_dir', 'unit_name', 'taste', 'taste_responsive']].drop_duplicates()
    # get rows from held_df_long where [rec_dir, unit_name] is in respidxs
    held_resp = held_df_long.merge(respidxs, on=['rec_dir', 'unit_name']).drop_duplicates()
    # group by held_unit_name and remove all groups where no row of the group has held == True
    for name, group in held_resp.groupby(['held_unit_name']):
        if not any(group[['taste_responsive', 'taste']]):
            held_resp = held_resp[held_resp.held_unit_name != name]

    return held_resp

def get_arrays(name, group, query_name='spike_array', query_func=dio.h5io.get_spike_data):
    # get spike arrays or rate arrays (or whatever) for each unit in each recording
    # use the query_func to get the arrays
    # query_name
    # Initialize lists for this group
    sessiontrials = []
    tastetrials = []
    queried_arr = []
    timedata = []
    rec_dir = []
    held_unit_name = []
    interj3 = []
    digins = []
    unit_nums = []

    dat = blechpy.load_dataset(name)
    dinmap = dat.dig_in_mapping.query('spike_array ==True')
    tastemap = dinmap[['channel', 'name']]
    # rename column 'name' to 'taste'
    tastemap = tastemap.rename(columns={'name': 'taste'})
    group = group.merge(tastemap, on=['taste'])
    unittable = dat.get_unit_table()
    digintrials = dat.dig_in_trials
    digintrials['tasteExposure'] = digintrials.groupby(['name', 'channel']).cumcount() + 1
    digintrials = digintrials.loc[digintrials.name != 'Experiment'].reset_index(drop=True)

    for i, row in group.iterrows():
        print(i)
        unum = unittable.loc[unittable.unit_name == row.unit_num]
        unum = unum.unit_num.item()

        trials = digintrials.loc[digintrials.channel == row['channel']]
        time, arrays = query_func(row.rec_dir, unum, row['channel'])
        for k, array in enumerate(arrays):
            sessiontrials.append(trials.trial_num.iloc[k])
            tastetrials.append(trials.tasteExposure.iloc[k])
            queried_arr.append(array)
            timedata.append(time)
            rec_dir.append(name)
            held_unit_name.append(row.held_unit_name)
            unit_nums.append(group['unit_num'][i])
            digins.append(row.channel)
    # Construct a list of dictionaries for this group
    data_dicts = [{
        query_name: qa,
        'time_array': td,
        'session_trial': ss,
        'taste_trial': tt,
        'rec_dir': rd,
        'held_unit_name': hun,
        'din': di,
        'unit_num': un
    } for qa, td, ss, tt, rd, hun, di, un in
        zip(queried_arr, timedata, sessiontrials, tastetrials, rec_dir, held_unit_name, digins, unit_nums)]
    return data_dicts


def get_rate_arrays(name, group):
    return get_arrays(name, group, query_name='rate_array', query_func=dio.h5io.get_rate_data)

# Split the data into chunks for parallel processing
def get_rate_array_df(PA):
    held_resp = get_held_resp(PA)
    groups = list(held_resp.groupby(['rec_dir']))
    # Use multiprocessing to process each group in parallel
    with Pool(processes=4) as pool:  # Adjust the number of processes based on your CPU cores
        results = pool.starmap(get_rate_arrays, groups)

    # Flatten the results and construct the DataFrame
    all_data = [item for sublist in results for item in sublist]
    df = pd.DataFrame(all_data)

    proj = PA.project
    rec_info = proj.get_rec_info()
    ri_formerge = rec_info[['exp_name', 'exp_group', 'rec_num', 'rec_dir']]
    # rename rec_num to session
    ri_formerge = ri_formerge.rename(columns={'rec_num': 'session'})

    # apply columns from ri_formerge to df along rec_dir column
    df = df.merge(ri_formerge, on=['rec_dir'])
    df = df.loc[df.din < 4].reset_index(drop=True)

    # make column with miniumum session trial
    df['min_session_trial'] = df.groupby(['rec_dir'])['session_trial'].transform(min)
    df['session_trial'] = df['session_trial'] - df['min_session_trial']

    return df
