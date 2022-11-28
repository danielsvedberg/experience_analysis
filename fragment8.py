#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:00:35 2022

@author: dsvedberg
"""
best_hmms = HA.get_best_hmms(sorting = 'best_BIC')#, overwrite = True)# sorting = 'params #4') #HA has no attribute project analysis #this is not getting the early and late states 
#heads up, params #5 doesn't cover all animals, you want params #4
#play around with state timings in mark_early_and_late_states()

#HA.plot_grouped_BIC()
#HA.plot_best_BIC()
#check df for nan in HMMID or early or late state
sns.set(font_scale = 3)

LR_timings,LR_trials,LR_res, LR_meta = HA.analyze_pal_linear_regression()
LR_timings['err'] = LR_timings.Y_pred - LR_timings.palatability
LR_timings.err= LR_timings.err.astype(float)
LR_timings['SquaredError'] = LR_timings.err**2
features = ['t_start', 't_end', 't_med', 'err', 'SquaredError','Y_pred']
groupings = ['time_group','exp_group','taste']
LR_corrs = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
groupings = ['time_group', 'exp_group']
LR_corrs_alltst = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)

##Plot accuracy & timing of linear regression###########################################

nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 10)
sns.set(font_scale = 2)
nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 5)
sns.set(font_scale = 1.5)
nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 5)
LR_timings,LR_trials,LR_res, LR_meta = HA.analyze_pal_linear_regression()
nplt.plot_pal_data(df,save_dir = HA.save_dir)
nplt.plot_pal_data(LR_timings,save_dir = HA.save_dir)
nplt.plot_pal_data(LR_trials,save_dir = HA.save_dir)
nplt.plot_pal_data(LR_timings,save_dir = HA.save_dir)
df = LR_timings
df = df.rename(columns = {'palatability': 'pal-rating', 'Y_pred': 'predicted-pal','time_group':'session','trial_num':'trial'})
g = sns.lmplot(data = df, x = 'pal-rating', y = 'predicted_pal', row = 'session', hue = 'trial')
g = sns.lmplot(data = df, x = 'pal-rating', y = 'predicted-pal', row = 'session', hue = 'trial')
if save_dir:
    nm1 = '_pal_xyplot.svg'
    sf = os.path.join(save_dir, nm1)
    g.savefig(sf)
    print(sf)
save_dir = HA.save_dir
if save_dir:
    nm1 = '_pal_xyplot.svg'
    sf = os.path.join(save_dir, nm1)
    g.savefig(sf)
    print(sf)
nplt.plot_pal_data(LR_timings,save_dir = HA.save_dir)
plt.close('all')
nplt.plot_pal_data(LR_timings,save_dir = HA.save_dir)
plt.close('all')
nplt.plot_pal_data(LR_timings,save_dir = HA.save_dir)
plt.close('all')
nplt.plot_pal_data(LR_timings,save_dir = HA.save_dir)
sns.set(font_scale = 1.5)
nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 5)
nplt.plot_pal_data(LR_timings,save_dir = HA.save_dir)
%debug
LR_timings['err'] = LR_timings.Y_pred - LR_timings.palatability
LR_timings.err= LR_timings.err.astype(float)
LR_timings['SquaredError'] = LR_timings.err**2
features = ['t_start', 't_end', 't_med', 'err', 'SquaredError','Y_pred']
groupings = ['time_group','exp_group','taste']
LR_corrs = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
groupings = ['time_group', 'exp_group']
LR_corrs_alltst = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)

##Plot accuracy & timing of linear regression###########################################

nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 5)
nplt.plot_pal_data(LR_timings,save_dir = HA.save_dir)
sns.set(font_scale = 1.5)
nplt.plot_pal_data(LR_timings,save_dir = HA.save_dir)
sns.set(font_scale = 1.75)
nplt.plot_pal_data(LR_timings,save_dir = HA.save_dir)
nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 5)
groupings = ['time_group','exp_group','state_group']
features = ['t_start','t_end','t_med','duration','p_correct']
df = NB_timings
df = df.loc[df.p_correct > 0.5]
NB_corrs = hmma.analyze_state_correlations(df, groupings,'trial_num', features)
groupings = ['time_group','exp_group']
LR_corrs = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 5)
[NB_res,NB_meta,NB_decode,NB_best,NB_timings] = HA.analyze_NB_ID(overwrite = True)
grcols = ['rec_dir','trial_num','taste','state_num']
NB_decode['state_num'] = NB_decode['hmm_state'].astype(int)
NB_decsub = NB_decode[grcols+['p_correct']]
NB_timings = NB_timings.merge(NB_decsub, on = grcols)

groupings = ['time_group','exp_group','state_group']
features = ['t_start','t_end','t_med','duration','p_correct']
df = NB_timings
df = df.loc[df.p_correct > 0.5]
NB_corrs = hmma.analyze_state_correlations(df, groupings,'trial_num', features)
sns.set(font_scale = 1.75)
NB_decode['taste'] = NB_decode.trial_ID
grcols = ['rec_dir','trial_num','taste','state_num']
NB_decode['state_num'] = NB_decode['hmm_state'].astype(int)
NB_decsub = NB_decode[grcols+['p_correct']]
NB_timings = NB_timings.merge(NB_decsub, on = grcols)

groupings = ['time_group','exp_group','state_group']
features = ['t_start','t_end','t_med','duration','p_correct']
df = NB_timings
df = df.loc[df.p_correct > 0.5]
NB_corrs = hmma.analyze_state_correlations(df, groupings,'trial_num', features)
nplt.plot_NB_timing(NB_timings, plotdir = HA.save_dir, trial_group_size = 5)
nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 5)
LR_corrs_alltst = hmma.analyze_state_correlations(LR_timings, groupings, 'trial_num', features)
sns.set(font_scale = 1.5)
nplt.plot_LR(LR_timings, plotdir = HA.save_dir, trial_group_size = 5)
df = LR_timings
df['trial-group'] = df.trial_num/trial_group_size
df['trial-group'] = df['trial-group'].astype(int)
df = df.loc[df.single_state == False]
df = df.loc[df.state_group != 'Prestim']

df = df.rename(columns = {'trial_num':'trial','time_group':'session', 'exp_group':'condition', 't_start': 't(start)', 't_end':'t(end)', 't_med':'t(median)','err':'error', 'Y_pred':'pred-pal'})
y_facs = ['duration','t(start)', 't(end)', 't(median)', 'error', 'SquaredError','pred-pal']
trial_group_size = 5
df['trial-group'] = df.trial_num/trial_group_size
df['trial-group'] = df['trial-group'].astype(int)
df = df.loc[df.single_state == False]
df = df.loc[df.state_group != 'Prestim']

df = df.rename(columns = {'trial_num':'trial','time_group':'session', 'exp_group':'condition', 't_start': 't(start)', 't_end':'t(end)', 't_med':'t(median)','err':'error', 'Y_pred':'pred-pal'})
y_facs = ['duration','t(start)', 't(end)', 't(median)', 'error', 'SquaredError','pred-pal']
plot_trialwise_lm(df, x_col = 'trial', y_facs = y_facs, save_dir = plotdir, save_prefix = 'LR')
def plot_trialwise_lm(df, x_col, y_facs, save_dir = None, save_prefix = None):
    
    for ycol in y_facs:
        g = sns.lmplot(data = df, x = x_col, y = ycol, 
                       hue = 'condition', col = 'session', row = 'taste', aspect = 1, height = 20,
                       facet_kws={"margin_titles": True})
        g.tight_layout()
        g.set_titles(row_template = '{row_name}')
        #sns.move_legend(g, "lower center", bbox_to_anchor=(.4, 1), ncol=2, title=None, frameon=False)
        
        
        h = sns.lmplot(data = df, x = x_col, y = ycol,
                       hue = 'condition', col = 'session', aspect = 2, height = 8)
        h.tight_layout()
        h.set_titles(row_template = '{row_name}')
        #sns.move_legend(h, "lower center", bbox_to_anchor=(.4, 1), ncol=2, title=None, frameon=False)
        
        if x_col == 'trial-group':
            g.set(xlim = (-0.25,5.25), xticks = [0,1,2,3,4,5])
            g.set_xticklabels(['1-5','6-10','11-15','16-20','21-25','26-30'], rotation = 60)
            h.set(xlim = (-0.25,5.25), xticks = [0,1,2,3,4,5])
            h.set_xticklabels(['1-5','6-10','11-15','16-20','21-25','26-30'], rotation = 60)
        
        if ycol == 'p(correct)':
            g.set(ylim = (-0.1,1.1), yticks = [0,0.25, 0.5, 0.75, 1])
            h.set(ylim = (-0.1,1.1), yticks = [0,0.25, 0.5, 0.75, 1])
        
        if save_dir is not None:  
            nm = save_prefix+x_col+'_VS_'+ycol+'_LMP.svg'
            sf = os.path.join(save_dir,nm)
            g.savefig(sf)
            print(sf)
            nm2 = save_prefix+x_col+'_VS_'+ycol+'_alltsts_LMP.svg'
            sf2 = os.path.join(save_dir,nm2)
            h.savefig(sf2)
            print(sf2)
        plt.close('all')
plot_trialwise_lm(df, x_col = 'trial', y_facs = y_facs, save_dir = plotdir, save_prefix = 'LR')
plotdir = HA.save_dir
plot_trialwise_lm(df, x_col = 'trial', y_facs = y_facs, save_dir = plotdir, save_prefix = 'LR')

