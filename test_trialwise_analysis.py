#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:46:14 2023

@author: danielsvedberg
"""
from trialwise_analysis import *

# Generate the synthetic data in long-form
synthetic_data = generate_synthetic_data()
synthetic_data.head()  # Displaying the first few rows of the generated data

# Perform nonlinear regression on the synthetic data
fitted_parameters, r2 = nonlinear_regression(synthetic_data)

# generate datapoints from model for plotting
fitted_data = generate_fitted_data(fitted_parameters)

#plot scatterplot with fit for each group
fig, ax = plt.subplots(figsize=(5, 5))
sns.scatterplot(data=synthetic_data, x='Time', y='Value', hue='Subject', legend=False, ax=ax)
sns.lineplot(data=fitted_data, x='Time', y='Value', hue='Subject', legend=False, ax=ax)
plt.show()

#now generate the metafit, which is the average of each individual's fit
#metaparams, metar2, meanr2 = nonlinear_metaregression(synthetic_data)
#metafit = generate_metafit(metaparams,time_steps=30)


#plot the metafit with the 95% CI around it
"""
fig, ax = plt.subplots(figsize=(5, 5))
sns.scatterplot(data=synthetic_data, x='Time', y='Value', hue='Subject', legend=False)
ax.plot(metafit.Time, metafit.model_mean, color = 'black')
ax.fill_between(metafit.Time, metafit.model_high, metafit.model_low, color = 'gray', alpha = 0.5)
plt.show()
"""

#generate the null distribution of the metafit r2
shuffdf = iter_shuffle(synthetic_data, niter=2000)

metar2ci = np.percentile(shuffdf.metaR2, [2.5, 97.5])
meanr2ci = np.percentile(shuffdf.r2_est, [2.5, 97.5])

#plot the null distribution, and the r2 value of the 
counts, bins = np.histogram(shuffdf.r2_est, bins = 50, range = (-1, 1))
mask = (shuffdf.r2_est >= meanr2ci[0]) & (shuffdf.r2_est <= meanr2ci[1])
fig, ax = plt.subplots(figsize=(5, 5), sharey=True)
ax.hist(shuffdf.r2_est[mask], bins)
ax.hist(shuffdf.r2_est, bins, histtype = 'step', color='black')
ax.axvline(x=meanr2['bootmean'])
plt.show()


shuffdf['metaR2'] = np.round(shuffdf['metaR2'], 3)
counts, bins = np.histogram(shuffdf.metaR2, bins = 50, range = (-1, 1))
mask = (shuffdf.metaR2 >= metar2ci[0]) & (shuffdf.metaR2 <= metar2ci[1])
fig, ax = plt.subplots(figsize=(5,5))
ax.hist(shuffdf.metaR2[mask], bins)
ax.hist(shuffdf.metaR2, bins, histtype='step', color='black')
ax.axvline(x=metar2)
plt.show()