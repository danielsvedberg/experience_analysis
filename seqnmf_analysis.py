import blechpy
import numpy as np
import blechpy.dio.h5io as h5io
import pandas as pd
from joblib import Parallel, delayed
import trialwise_analysis as ta
import analysis as ana
import matplotlib.pyplot as plt
import feather
import seqnmf

proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy' # directory where the project is
proj = blechpy.load_project(proj_dir) #load the project
rec_info = proj.rec_info.copy() #get the rec_info table
rec_dirs = rec_info['rec_dir']

PA = ana.ProjectAnalysis(proj)

def get_trial_info(dat):
    dintrials = dat.dig_in_trials
    dintrials['taste_trial'] = 1
    #groupby name and cumsum taste trial
    dintrials['taste_trial'] = dintrials.groupby('name')['taste_trial'].cumsum()
    #rename column trial_num to 'session_trial'
    dintrials = dintrials.rename(columns={'trial_num':'session_trial','name':'taste'})
    #select just the columns 'taste_trial', 'taste', 'session_trial', 'channel', and 'on_time'
    dintrials = dintrials[['taste_trial', 'taste', 'session_trial', 'channel', 'on_time']]
    return dintrials

def reconstruct(W, H):
    N, K, L = W.shape
    _, T = H.shape

    # Zero-pad H by L on both sides
    H_padded = np.zeros((K, T + 2 * L))
    H_padded[:, L:L + T] = H
    T_padded = T + 2 * L
    X_hat = np.zeros((N, T_padded))

    for tau in range(1, L + 1):  # Loop over each lag
        H_shifted = np.roll(H_padded, shift=(tau - 1), axis=1)
        for n in range(N):  # Loop over each neuron
            for k in range(K):  # Loop over each factor
                X_hat[n, :] += W[n, k, tau - 1] * H_shifted[k, :]

    # Remove zero-padding
    X_hat = X_hat[:, L:-L]
    return X_hat

def DISSX(H1, W1, H2, W2):
    K = H1.shape[0]
    C = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            Xhat1 = reconstruct(W1[:, i, :], H1[i, :])
            Xhat2 = reconstruct(W2[:, j, :], H2[j, :])
            num = np.dot(Xhat1.ravel(), Xhat2.ravel())
            denom = np.sqrt(np.dot(Xhat1.ravel(), Xhat1.ravel()) * np.dot(Xhat2.ravel(), Xhat2.ravel())) + np.finfo(
                float).eps
            C[i, j] = num / denom

    maxrow = np.max(C, axis=1)
    maxcol = np.max(C, axis=0)
    maxrow[np.isnan(maxrow)] = 0
    maxcol[np.isnan(maxcol)] = 0
    diss = 1 / 2 / K * (2 * K - np.sum(maxrow) - np.sum(maxcol))
    return diss


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
def SimpleWHPlot(W, H, Data=None, plotAll=True):
    N, K, L = W.shape
    _, T = H.shape
    color_palette = np.array(
        [[0, .6, .3], [.7, 0, .7], [1, .6, 0], [.1, .3, .9], [1, .1, .1], [0, .9, .3], [.4, .2, .7],
         [.7, .2, .1], [.1, .8, 1], [1, .3, .7], [.2, .8, .2], [.7, .4, 1], [.9, .6, .4], [0, .6, 1],
         [1, .1, .3]])
    color_palette = np.tile(color_palette, (int(np.ceil(K / color_palette.shape[0])), 1))
    kColors = color_palette[:K, :]

    plt.figure(figsize=(15, 6))
    gs = plt.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1], hspace=0.05, wspace=0.05)

    # Plot W
    axW = plt.subplot(gs[0, 0])
    for ki in range(K):
        for li in range(L):
            axW.plot(W[:, ki, li] + ki * L * 2, np.arange(N) + 1, color=kColors[ki], linewidth=2)
    axW.set_ylim(1, N)
    axW.invert_yaxis()
    axW.axis('off')

    # Plot Data or Reconstructed Data
    axData = plt.subplot(gs[0, 1])
    if Data is not None and Data.shape[1] > 0:
        plotData = True
        toPlot = Data
    else:
        plotData = False
        toPlot = reconstruct(W, H)  # Assuming reconstruct function is defined as before
    if not plotAll:
        indPlot = range(2 * L, min(2 * L + int(np.ceil((K * (L + L / 10)) / 1 * toPlot.shape[1])), toPlot.shape[1]))
    else:
        indPlot = range(toPlot.shape[1])
    im = axData.imshow(toPlot[:, indPlot], aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(im, ax=axData, orientation='vertical')
    axData.axis('off')

    # Plot H
    axH = plt.subplot(gs[1, 1])
    for ki, color in zip(range(K), kColors):
        axH.fill_between(np.arange(len(indPlot)), H[ki, indPlot] + ki * 0.5, ki * 0.5, color=color)
    axH.set_xlim(0, len(indPlot))
    axH.axis('off')

    plt.show()


rec_dir = rec_dirs[0]
dat = blechpy.load_dataset(rec_dir)
dintrials = get_trial_info(dat)
time_array, spike_array = h5io.get_spike_data(rec_dir)
spikes = spike_array['dig_in_0']
reshaped = spikes.reshape(spikes.shape[0], -1)  # -1 tells numpy to calculate the size of this dimension
X = reshaped

# Fit with seqNMF
K = 5
L = 50
lambda_ = 0.005
W, H = seqnmf.seqnmf(X, K, L, lambda_)

import time
from scipy.special import comb
# Procedure for choosing K
start_time = time.time()
Ws = {}
Hs = {}
numfits = 3  # number of fits to compare
Diss = np.zeros((int(comb(numfits, 2, exact=True)), 10))

for k in range(1, 11):
    print(f'running seqNMF with K = {k}')
    for ii in range(numfits):
        Ws[ii, k], Hs[ii, k] , cost, loadings, power = seqnmf.seqnmf(X, k, L, 0)
    inds = np.array([index for index in comb(np.arange(numfits), 2, exact=True)])
    for i, (ind1, ind2) in enumerate(inds):
        Diss[i, k-1] = DISSX(Hs[ind1, k], Ws[ind1, k], Hs[ind2, k], Ws[ind2, k])


def process_rec_dir(rec_dir):
    df_list = []
    dat = blechpy.load_dataset(rec_dir)
    dintrials = get_trial_info(dat)
    time_array, spike_array = h5io.get_spike_data(rec_dir)
    for din, spikes in spike_array.items():
        print(din)

        reshaped = spikes.reshape(spikes.shape[0], -1)  # -1 tells numpy to calculate the size of this dimension

        W, H, cost, loadings, power = seqnmf.seqnmf(reshaped, K=5, L=5, Lambda=0.01)
        seqnmf.plot(W,H).show()

    df = pd.concat(df_list, ignore_index=True)
    #add index info to df from dintrials using merge on taste_trial and channel
    df = pd.merge(df, dintrials, on=['taste_trial', 'channel'])
    #remove all rows where taste == 'Spont'
    df = df.loc[df['taste'] != 'Spont']
    #subtract the min of 'session_trial' from 'session_trial' to get the session_trial relative to the start of the recording
    df['session_trial'] = df['session_trial'] - df['session_trial'].min()
    return df

for rec_dir in rec_dirs:
    process_rec_dir(rec_dir)