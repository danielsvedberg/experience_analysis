import blechpy
import blechpy.dio.h5io as h5io
import analysis as ana
import os


proj_dir = '/media/dsvedberg/Ubuntu Disk/taste_experience_resorts_copy'  # directory where the project is
proj = blechpy.load_project(proj_dir)  # load the project
PA = ana.ProjectAnalysis(proj)
save_dir = PA.save_dir  # get the save directory for the project
euc_dist_demo_folder = os.path.join(save_dir, 'euc_dist_demo')  # create a folder to save the results
if not os.path.exists(euc_dist_demo_folder):
    os.mkdir(euc_dist_demo_folder)


rec_info = proj.get_rec_info()  # get the recording info

DS39_D1 = rec_info[rec_info['exp_name'] == 'DS39']
DS39_D1 = DS39_D1[DS39_D1['rec_num'] == 1]

# get the firing rate data for just dig_in_2
rec_dir = DS39_D1['rec_dir'].values[0]
dat = blechpy.load_dataset(rec_dir)
time_array, rate_array = h5io.get_rate_data(rec_dir)
data = rate_array['dig_in_2']


#get the PCA of the first 3 components across dim 0
import numpy as np
from sklearn.decomposition import PCA

# Reshape data to bring neurons into the feature column
reshaped_data = data.reshape(data.shape[0], -1).T  # Reshaped to (30*7000, 74)

# Initialize and fit PCA for the first 3 components
pca = PCA(n_components=3)
principal_components_scores = pca.fit_transform(reshaped_data)  # Shape will be (210000, 3)

# Reshape back to (3, 30, 7000)
# Each row in principal_components_scores is a trial-time, and we have 3 columns for each PC
final_output = principal_components_scores.T.reshape(3, 30, 7000)
final_output = final_output[:,:, 2000:4000]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.cm import get_cmap
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
from mpl_toolkits.mplot3d import Axes3D

# Assuming final_output is already defined and shaped (3, 30, 7000 or similar)
# This assumes final_output has been processed as in your previous code

# Downsample final_output to reduce the number of points for visualization
nbins = final_output.shape[2]
newnbins = nbins // 20
final_output = np.mean(final_output.reshape(3, 30, newnbins, 20), axis=3)

num_trials = final_output.shape[1]
time_points = final_output.shape[2]

# Use a comprehensive color map that fits the number of trials
cmap = get_cmap('tab20b', num_trials)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
max_vals = np.max(final_output, axis=(1, 2))
min_vals = np.min(final_output, axis=(1, 2))

ax.set_xlim([min_vals[0], max_vals[0]])
ax.set_ylim([min_vals[1], max_vals[1]])
ax.set_zlim([min_vals[2], max_vals[2]])

scatters = [ax.plot([], [], [], color=cmap(i % 20), markersize=5, linewidth=1)[0] for i in range(num_trials)]

def init():
    for scatter in scatters:
        scatter.set_data([], [])
        scatter.set_3d_properties([])
    return scatters

def animate(frame):
    trial = frame // time_points
    t = frame % time_points

    x = final_output[0, trial, :t + 1]
    y = final_output[1, trial, :t + 1]
    z = final_output[2, trial, :t + 1]
    scatters[trial].set_data(x, y)
    scatters[trial].set_3d_properties(z)

    # Spin the view: rotate the azimuth angle (azim) by 0.5 degree per frame
    ax.view_init(azim=frame/2)
    return scatters

total_frames = num_trials * time_points
ani = FuncAnimation(fig, animate, frames=total_frames, interval=1, init_func=init, blit=False)

plt.show()
# Save the animation
save_path= os.path.join(euc_dist_demo_folder, 'animation.mp4')
ani.save(save_path, writer='ffmpeg', fps=40)  # Adjust fps for speed control

#%% plot with 2 example trials
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Assuming final_output has been defined and has shape (3, 30, downscaled time bins)
# Selecting trials 1 and 20 (0-based indexing as trials 0 and 19)

trial_indices = [0, 19]  # Adjust according to your indices
time_points = final_output.shape[2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set axis limits based on the data of the selected trials
max_vals = np.max(final_output[:, trial_indices, :], axis=(1, 2))
min_vals = np.min(final_output[:, trial_indices, :], axis=(1, 2))

ax.set_xlim([min_vals[0], max_vals[0]])
ax.set_ylim([min_vals[1], max_vals[1]])
ax.set_zlim([min_vals[2], max_vals[2]])

# Define a list of markers and colors for each trial
markers = ['o', '^']  # Example: 'o' for trial 1, '^' for trial 20
colors = ['blue', 'green']  # Example colors for trials

# Initialize scatter plots for each trial
scatters = []
for idx, trial_index in enumerate(trial_indices):
    x = final_output[0, trial_index, :]
    y = final_output[1, trial_index, :]
    z = final_output[2, trial_index, :]
    scatter = ax.scatter(x, y, z, color=colors[idx], marker=markers[idx], s=50)
    scatters.append(scatter)

def init():
    # Optional: Initialize anything you need at the start of the animation
    return scatters

def animate(frame):
    # Spin the plot by adjusting the azimuth angle, increased step size for faster rotation
    ax.view_init(elev=30, azim=frame)  # Increase the multiplier here for faster rotation
    return scatters

# Creating the animation
ani = FuncAnimation(fig, animate, frames=np.arange(0, 360, 1), init_func=init, blit=False, repeat=True)
save_path= os.path.join(euc_dist_demo_folder, 'twotrials.mp4')
ani.save(save_path, writer='ffmpeg', fps=30)  # Save the animation as a GIF

plt.show()

#%% plot with 2 example trials and a line connecting them for euc distance

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Assuming final_output has been defined and has shape (3, 30, downscaled time bins)
# Selecting trials 1 and 20 (0-based indexing as trials 0 and 19)
trial_indices = [0, 19]  # Adjust according to your indices
time_points = final_output.shape[2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set axis limits based on the data of the selected trials
max_vals = np.max(final_output[:, trial_indices, :], axis=(1, 2))
min_vals = np.min(final_output[:, trial_indices, :], axis=(1, 2))
ax.set_xlim([min_vals[0], max_vals[0]])
ax.set_ylim([min_vals[1], max_vals[1]])
ax.set_zlim([min_vals[2], max_vals[2]])

# Define a list of markers and colors for each trial
markers = ['o', '^']  # Example: 'o' for trial 1, '^' for trial 20
colors = ['blue', 'green']  # Example colors for trials

# Store scatter plots for animation updates
scatters = []
for idx, trial_index in enumerate(trial_indices):
    scatter, = ax.plot([], [], [], marker=markers[idx], color=colors[idx], linestyle='None', markersize=5)
    scatters.append(scatter)


# Function to initialize the animation
def init():
    for scatter in scatters:
        scatter.set_data([], [])
        scatter.set_3d_properties([])
    return scatters


# Animation function: this is called sequentially
def animate(t):
    for idx, trial_index in enumerate(trial_indices):
        x = final_output[0, trial_index, :t + 1]
        y = final_output[1, trial_index, :t + 1]
        z = final_output[2, trial_index, :t + 1]
        scatters[idx].set_data(x, y)
        scatters[idx].set_3d_properties(z)

    # Draw line connecting the current points of the trials
    if t > 0:  # To ensure we have at least one point to connect
        ax.plot([final_output[0, trial_indices[0], t - 1], final_output[0, trial_indices[1], t - 1]],
                [final_output[1, trial_indices[0], t - 1], final_output[1, trial_indices[1], t - 1]],
                [final_output[2, trial_indices[0], t - 1], final_output[2, trial_indices[1], t - 1]],
                color='gray', linestyle='-', linewidth=1)
        ax.view_init(elev=45, azim=t)  # Adjust elevation and rotation speed as needed
    return scatters


# Creating the animation
ani = FuncAnimation(fig, animate, frames=time_points, init_func=init, blit=False, repeat=False)
save_path= os.path.join(euc_dist_demo_folder, 'twotrials_euc_dists.mp4')
ani.save(save_path, writer='ffmpeg', fps=15)  # Save the animation as a GIF

#%% plot avg trial w trial 0
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Assuming final_output has been defined and has shape (3, 30, downscaled time bins)
average_trajectory = np.mean(final_output, axis=1)  # Average across all trials
trial_0_trajectory = final_output[:, 0, :]  # Trajectory for Trial 0

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Calculate limits for the axes based on both trajectories
combined_data = np.concatenate((average_trajectory, trial_0_trajectory), axis=1)
max_vals = np.max(combined_data, axis=1)
min_vals = np.min(combined_data, axis=1)
ax.set_xlim([min_vals[0], max_vals[0]])
ax.set_ylim([min_vals[1], max_vals[1]])
ax.set_zlim([min_vals[2], max_vals[2]])

# Initialize scatter plots for average and Trial 0 trajectories
scatter_avg, = ax.plot([], [], [], 'ro-', markersize=5, linewidth=2, label='Average Trajectory')
scatter_trial_0, = ax.plot([], [], [], 'b^-', markersize=5, linewidth=2, label='Trial 0')

# Initialize lines list for connecting lines
connecting_lines = []

def init():
    scatter_avg.set_data([], [])
    scatter_avg.set_3d_properties([])
    scatter_trial_0.set_data([], [])
    scatter_trial_0.set_3d_properties([])
    return scatter_avg, scatter_trial_0

def animate(t):
    # Plot points for the current time step
    scatter_avg.set_data(average_trajectory[0, :t+1], average_trajectory[1, :t+1])
    scatter_avg.set_3d_properties(average_trajectory[2, :t+1])
    scatter_trial_0.set_data(trial_0_trajectory[0, :t+1], trial_0_trajectory[1, :t+1])
    scatter_trial_0.set_3d_properties(trial_0_trajectory[2, :t+1])

    # Add a new line connecting the current points
    if t > 0:
        new_line, = ax.plot([average_trajectory[0, t], trial_0_trajectory[0, t]],
                            [average_trajectory[1, t], trial_0_trajectory[1, t]],
                            [average_trajectory[2, t], trial_0_trajectory[2, t]],
                            color='gray', linestyle='-', linewidth=1)
        connecting_lines.append(new_line)

    # Rotate the view
    ax.view_init(elev=45, azim=t)  # Adjust elevation and rotation speed as needed

    return [scatter_avg, scatter_trial_0] + connecting_lines

# Create the animation
ani = FuncAnimation(fig, animate, frames=len(trial_0_trajectory[0]), init_func=init,
                    blit=False, repeat=False, interval=1000)  # Slower animation with 100ms interval

ax.legend()
save_path= os.path.join(euc_dist_demo_folder, 'avg_v_zero_euc_dists.mp4')
ani.save(save_path, writer='ffmpeg', fps=15)  # Save the animation as a GIF

#%% plot avg trial w trial 25
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Assuming final_output has been defined and has shape (3, 30, downscaled time bins)
average_trajectory = np.mean(final_output, axis=1)  # Average across all trials
trial_0_trajectory = final_output[:, 25, :]  # Trajectory for Trial 0

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Calculate limits for the axes based on both trajectories
combined_data = np.concatenate((average_trajectory, trial_0_trajectory), axis=1)
max_vals = np.max(combined_data, axis=1)
min_vals = np.min(combined_data, axis=1)
ax.set_xlim([min_vals[0], max_vals[0]])
ax.set_ylim([min_vals[1], max_vals[1]])
ax.set_zlim([min_vals[2], max_vals[2]])

# Initialize scatter plots for average and Trial 0 trajectories
scatter_avg, = ax.plot([], [], [], 'ro-', markersize=5, linewidth=2, label='Average Trajectory')
scatter_trial_0, = ax.plot([], [], [], 'g^-', markersize=5, linewidth=2, label='Trial 25')

# Initialize lines list for connecting lines
connecting_lines = []

def init():
    scatter_avg.set_data([], [])
    scatter_avg.set_3d_properties([])
    scatter_trial_0.set_data([], [])
    scatter_trial_0.set_3d_properties([])
    return scatter_avg, scatter_trial_0

def animate(t):
    # Plot points for the current time step
    scatter_avg.set_data(average_trajectory[0, :t+1], average_trajectory[1, :t+1])
    scatter_avg.set_3d_properties(average_trajectory[2, :t+1])
    scatter_trial_0.set_data(trial_0_trajectory[0, :t+1], trial_0_trajectory[1, :t+1])
    scatter_trial_0.set_3d_properties(trial_0_trajectory[2, :t+1])

    # Add a new line connecting the current points
    if t > 0:
        new_line, = ax.plot([average_trajectory[0, t], trial_0_trajectory[0, t]],
                            [average_trajectory[1, t], trial_0_trajectory[1, t]],
                            [average_trajectory[2, t], trial_0_trajectory[2, t]],
                            color='gray', linestyle='-', linewidth=1)
        connecting_lines.append(new_line)

    # Rotate the view
    ax.view_init(elev=45, azim=t)  # Adjust elevation and rotation speed as needed

    return [scatter_avg, scatter_trial_0] + connecting_lines

# Create the animation
ani = FuncAnimation(fig, animate, frames=len(trial_0_trajectory[1]), init_func=init,
                    blit=False, repeat=False)  # Slower animation with 100ms interval

ax.legend()
save_path= os.path.join(euc_dist_demo_folder, 'avg_v_25_euc_dists.mp4')
ani.save(save_path, writer='ffmpeg', fps=15)  # Save the animation as a GIF

#%%

#%% plot avg trial w trial 25
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Assuming final_output has been defined and has shape (3, 30, downscaled time bins)
average_trajectory = np.mean(final_output[:,13:,:], axis=1)  # Average across all trials
trial_0_trajectory = final_output[:, 25, :]  # Trajectory for Trial 0

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Calculate limits for the axes based on both trajectories
combined_data = np.concatenate((average_trajectory, trial_0_trajectory), axis=1)
max_vals = np.max(combined_data, axis=1)
min_vals = np.min(combined_data, axis=1)
ax.set_xlim([min_vals[0], max_vals[0]])
ax.set_ylim([min_vals[1], max_vals[1]])
ax.set_zlim([min_vals[2], max_vals[2]])

# Initialize scatter plots for average and Trial 0 trajectories
scatter_avg, = ax.plot([], [], [], 'ro-', markersize=5, linewidth=2, label='Trial 13-29 Avg')
scatter_trial_0, = ax.plot([], [], [], 'g^-', markersize=5, linewidth=2, label='Trial 25')

# Initialize lines list for connecting lines
connecting_lines = []

def init():
    scatter_avg.set_data([], [])
    scatter_avg.set_3d_properties([])
    scatter_trial_0.set_data([], [])
    scatter_trial_0.set_3d_properties([])
    return scatter_avg, scatter_trial_0

def animate(t):
    # Plot points for the current time step
    scatter_avg.set_data(average_trajectory[0, :t+1], average_trajectory[1, :t+1])
    scatter_avg.set_3d_properties(average_trajectory[2, :t+1])
    scatter_trial_0.set_data(trial_0_trajectory[0, :t+1], trial_0_trajectory[1, :t+1])
    scatter_trial_0.set_3d_properties(trial_0_trajectory[2, :t+1])

    # Add a new line connecting the current points
    if t > 0:
        new_line, = ax.plot([average_trajectory[0, t], trial_0_trajectory[0, t]],
                            [average_trajectory[1, t], trial_0_trajectory[1, t]],
                            [average_trajectory[2, t], trial_0_trajectory[2, t]],
                            color='gray', linestyle='-', linewidth=1)
        connecting_lines.append(new_line)

    # Rotate the view
    ax.view_init(elev=30, azim=t)  # Adjust elevation and rotation speed as needed

    return [scatter_avg, scatter_trial_0] + connecting_lines

# Create the animation
ani = FuncAnimation(fig, animate, frames=len(trial_0_trajectory[1]), init_func=init,
                    blit=False, repeat=False)  # Slower animation with 100ms interval

ax.legend()
save_path= os.path.join(euc_dist_demo_folder, 'lateavg_v_25_euc_dists.mp4')
ani.save(save_path, writer='ffmpeg', fps=15)  # Save the animation as a GIF

#%% plot early trials avg vs early trial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Assuming final_output has been defined and has shape (3, 30, downscaled time bins)
average_trajectory = np.mean(final_output[:,:13,:], axis=1)  # Average across all trials
trial_0_trajectory = final_output[:, 0, :]  # Trajectory for Trial 0

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Calculate limits for the axes based on both trajectories
combined_data = np.concatenate((average_trajectory, trial_0_trajectory), axis=1)
max_vals = np.max(combined_data, axis=1)
min_vals = np.min(combined_data, axis=1)
ax.set_xlim([min_vals[0], max_vals[0]])
ax.set_ylim([min_vals[1], max_vals[1]])
ax.set_zlim([min_vals[2], max_vals[2]])

# Initialize scatter plots for average and Trial 0 trajectories
scatter_avg, = ax.plot([], [], [], 'bo-', markersize=5, linewidth=2, label='Trial 0-12 Avg')
scatter_trial_0, = ax.plot([], [], [], 'g^-', markersize=5, linewidth=2, label='Trial 0')

# Initialize lines list for connecting lines
connecting_lines = []

def init():
    scatter_avg.set_data([], [])
    scatter_avg.set_3d_properties([])
    scatter_trial_0.set_data([], [])
    scatter_trial_0.set_3d_properties([])
    return scatter_avg, scatter_trial_0

def animate(t):
    # Plot points for the current time step
    scatter_avg.set_data(average_trajectory[0, :t+1], average_trajectory[1, :t+1])
    scatter_avg.set_3d_properties(average_trajectory[2, :t+1])
    scatter_trial_0.set_data(trial_0_trajectory[0, :t+1], trial_0_trajectory[1, :t+1])
    scatter_trial_0.set_3d_properties(trial_0_trajectory[2, :t+1])

    # Add a new line connecting the current points
    if t > 0:
        new_line, = ax.plot([average_trajectory[0, t], trial_0_trajectory[0, t]],
                            [average_trajectory[1, t], trial_0_trajectory[1, t]],
                            [average_trajectory[2, t], trial_0_trajectory[2, t]],
                            color='gray', linestyle='-', linewidth=1)
        connecting_lines.append(new_line)

    # Rotate the view
    ax.view_init(elev=30, azim=t)  # Adjust elevation and rotation speed as needed

    return [scatter_avg, scatter_trial_0] + connecting_lines

# Create the animation
ani = FuncAnimation(fig, animate, frames=len(trial_0_trajectory[1]), init_func=init,
                    blit=False, repeat=False)  # Slower animation with 100ms interval

ax.legend()
save_path= os.path.join(euc_dist_demo_folder, 'earlyavg_v_0_euc_dists.mp4')
ani.save(save_path, writer='ffmpeg', fps=15)  # Save the animation as a GIF

#%% average of trials 0-12 and 13-29 plotted together
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Assuming final_output has been defined and has shape (3, 30, downscaled time bins)
# Calculating the average trajectory for trials 0 through 12
average_trajectory_0_12 = np.mean(final_output[:, 0:13, :], axis=1)  # Average across trials 0 to 12

# Calculating the average trajectory for trials 13 through 29
average_trajectory_13_29 = np.mean(final_output[:, 13:30, :], axis=1)  # Average across trials 13 to 29

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Calculate limits for the axes based on both sets of average trajectories
combined_data = np.concatenate((average_trajectory_0_12, average_trajectory_13_29), axis=1)
max_vals = np.max(combined_data, axis=1)
min_vals = np.min(combined_data, axis=1)
ax.set_xlim([min_vals[0], max_vals[0]])
ax.set_ylim([min_vals[1], max_vals[1]])
ax.set_zlim([min_vals[2], max_vals[2]])

# Initialize the plots for the average trajectories
line_0_12, = ax.plot(average_trajectory_0_12[0], average_trajectory_0_12[1], average_trajectory_0_12[2],
                     'o-', color='blue', markersize=5, linewidth=2, label='Average Trajectory (Trials 0-12)')

line_13_29, = ax.plot(average_trajectory_13_29[0], average_trajectory_13_29[1], average_trajectory_13_29[2],
                      'o-', color='red', markersize=5, linewidth=2, label='Average Trajectory (Trials 13-29)')

def init():
    # Ensure the lines are already plotted, and only the view is adjusted in the animation.
    return line_0_12, line_13_29,

def animate(t):
    # Rotate the view for spinning effect
    ax.view_init(elev=30, azim=t)  # Elevation is constant, azimuth increases to rotate
    return line_0_12, line_13_29,

# Create the animation
ani = FuncAnimation(fig, animate, frames=360, init_func=init, blit=False, repeat=True, interval=100)

plt.show()
save_path= os.path.join(euc_dist_demo_folder, 'avgs.mp4')
ani.save(save_path, writer='ffmpeg', fps=20)  # Save the animation as a GIF


#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Assuming final_output has been defined and has shape (3, 30, downscaled time bins)
# Calculating the average trajectory for trials 0 through 13
average_trajectory = np.mean(final_output[:, 0:13, :], axis=1)  # Average across trials 0 to 13

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Calculate limits for the axes based on the average trajectory
max_vals = np.max(average_trajectory, axis=1)
min_vals = np.min(average_trajectory, axis=1)
ax.set_xlim([min_vals[0], max_vals[0]])
ax.set_ylim([min_vals[1], max_vals[1]])
ax.set_zlim([min_vals[2], max_vals[2]])

# Initialize the plot for the average trajectory
line, = ax.plot(average_trajectory[0], average_trajectory[1], average_trajectory[2],
                'o-', color='green', markersize=5, linewidth=2, label='Average Trajectory (Trials 0-12)')

def init():
    # Ensure the line is already plotted, and only the view is adjusted in the animation.
    return line,

def animate(t):
    # Rotate the view for spinning effect
    ax.view_init(elev=10, azim=t)  # Elevation is constant, azimuth increases to rotate
    return line,

# Create the animation
ani = FuncAnimation(fig, animate, frames=360, init_func=init, blit=False, repeat=True, interval=100)

ax.legend()

save_path= os.path.join(euc_dist_demo_folder, 'earlyavg.mp4')
ani.save(save_path, writer='ffmpeg', fps=15)  # Save the animation as a GIF

#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Assuming final_output has been defined and has shape (3, 30, downscaled time bins)
# Calculating the average trajectory for trials 0 through 13
average_trajectory = np.mean(final_output[:, 13:, :], axis=1)  # Average across trials 0 to 13

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Calculate limits for the axes based on the average trajectory
max_vals = np.max(average_trajectory, axis=1)
min_vals = np.min(average_trajectory, axis=1)
ax.set_xlim([min_vals[0], max_vals[0]])
ax.set_ylim([min_vals[1], max_vals[1]])
ax.set_zlim([min_vals[2], max_vals[2]])

# Initialize the plot for the average trajectory
line, = ax.plot(average_trajectory[0], average_trajectory[1], average_trajectory[2],
                'o-', color='red', markersize=5, linewidth=2, label='Average Trajectory (Trials 13-29)')

def init():
    # Ensure the line is already plotted, and only the view is adjusted in the animation.
    return line,

def animate(t):
    # Rotate the view for spinning effect
    ax.view_init(elev=10, azim=t)  # Elevation is constant, azimuth increases to rotate
    return line,

# Create the animation
ani = FuncAnimation(fig, animate, frames=360, init_func=init, blit=False, repeat=True, interval=100)

ax.legend()

save_path= os.path.join(euc_dist_demo_folder, 'lateavg.mp4')
ani.save(save_path, writer='ffmpeg', fps=15)  # Save the animation as a GIF

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Assuming final_output has been defined and has shape (3, 30, downscaled time bins)
# Calculate the average trajectory for all trials
average_trajectory = np.mean(final_output, axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set limits for the axes based on the entire dataset
max_vals = np.max(final_output, axis=(0, 2))
min_vals = np.min(final_output, axis=(0, 2))
ax.set_xlim([min_vals[0], max_vals[0]])
ax.set_ylim([min_vals[1], max_vals[1]])
ax.set_zlim([min_vals[2], max_vals[2]])

# Initialize the plot for the average trajectory
line_avg, = ax.plot(average_trajectory[0], average_trajectory[1], average_trajectory[2],
                     'o-', color='red', markersize=4, linewidth=2, label='Average Trajectory')

# List to store trial lines and connecting bars
trial_lines = []
connecting_bars = []
frames_per_trial = 30  # Number of frames to spin smoothly for each trial

def init():
    return [line_avg]

def animate(frame):
    trial_index = frame // frames_per_trial  # Determine the trial based on the frame
    phase = frame % frames_per_trial  # Phase of spinning for the current trial

    if phase == 0:
        # Clear previous trial lines and connecting bars at the start of each trial phase
        global trial_lines, connecting_bars
        while trial_lines:
            trial_lines.pop().remove()
        while connecting_bars:
            connecting_bars.pop().remove()

        # Plot the current trial trajectory
        trial_trajectory = final_output[:, trial_index, :]
        line_trial, = ax.plot(trial_trajectory[0], trial_trajectory[1], trial_trajectory[2],
                              'o-', color='blue', markersize=4, linewidth=2, label=f'Trial {trial_index}')
        trial_lines.append(line_trial)

        # Add connecting bars between the average trajectory and the current trial
        for i in range(trial_trajectory.shape[1]):
            bar, = ax.plot([average_trajectory[0, i], trial_trajectory[0, i]],
                           [average_trajectory[1, i], trial_trajectory[1, i]],
                           [average_trajectory[2, i], trial_trajectory[2, i]],
                           'gray', linestyle='-', linewidth=0.5)
            connecting_bars.append(bar)

        # Update legend to reflect the current trial
        ax.legend([line_avg, line_trial], ['Average Trajectory', f'Trial {trial_index}'])

    # Rotate the view smoothly for each phase
    ax.view_init(elev=30, azim=frame)  # Incremental rotation for dynamic effect

    return [line_avg] + trial_lines + connecting_bars

# Create the animation
total_frames = final_output.shape[1] * frames_per_trial
ani = FuncAnimation(fig, animate, frames=total_frames, init_func=init,
                    blit=False, repeat=True, interval=100)  # Quick interval for smooth rotation

#
# def init():
#     # This will plot the average trajectory and prepare other elements
#     line_avg.set_data([], [])
#     line_avg.set_3d_properties([])
#     return [line_avg] + trial_lines + connecting_bars
#
# def animate(trial_index):
#     # Clear previous trial lines and connecting bars
#     for line in trial_lines:
#         line.remove()
#     trial_lines.clear()
#
#     for bar in connecting_bars:
#         bar.remove()
#     connecting_bars.clear()
#
#     # Plot the current trial trajectory
#     trial_trajectory = final_output[:, trial_index, :]
#
#     line_avg, = ax.plot(average_trajectory[0], average_trajectory[1], average_trajectory[2],
#                         'o-', color='red', markersize=4, linewidth=2, label='Average Trajectory')
#
#     line_trial, = ax.plot(trial_trajectory[0], trial_trajectory[1], trial_trajectory[2],
#                           'o-', color='blue', markersize=4, linewidth=2, label=f'Trial {trial_index}')
#     trial_lines.append(line_trial)
#
#     # Add connecting bars between the average trajectory and the current trial
#     for i in range(trial_trajectory.shape[1]):
#         bar, = ax.plot([average_trajectory[0, i], trial_trajectory[0, i]],
#                        [average_trajectory[1, i], trial_trajectory[1, i]],
#                        [average_trajectory[2, i], trial_trajectory[2, i]],
#                        'gray', linestyle='-', linewidth=0.5)
#         connecting_bars.append(bar)
#
#     # Rotate the view for spinning effect
#     ax.view_init(elev=45, azim=trial_index*3)  # Rotate the view slightly with each frame
#
#     # Manage legends and labels
#     ax.legend([line_avg, line_trial], ['Average Trajectory', f'Trial {trial_index}'])
#
#
#     return [line_avg, line_trial] + connecting_bars
#
# # Create the animation
# ani = FuncAnimation(fig, animate, frames=final_output.shape[1], init_func=init,
#                     blit=False, repeat=True)  # Slowing down the animation for clarity

save_path= os.path.join(euc_dist_demo_folder, 'every_trial_v_avg.mp4')
ani.save(save_path, writer='ffmpeg', fps=30)  # Save the animation as a GIF
#%% static plot of the rates
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import get_cmap

# Assuming final_output has been defined and has shape (3, 30, downscaled time bins)
# For demonstration, use your processed data
# Select trials 1 and 20 (indexing from 0, these are trials 0 and 19)

trial_indices = [3, 25]  # Adjust according to your 1-based indices if needed
time_points = final_output.shape[2]

# Use a continuous colormap to represent time progression
cmap = get_cmap('viridis', time_points)

fig = plt.figure(figsize=(5, 6))
ax = fig.add_subplot(111, projection='3d')

# Set axis limits based on the data of the selected trials
max_vals = np.max(final_output[:, trial_indices, :], axis=(1, 2))
min_vals = np.min(final_output[:, trial_indices, :], axis=(1, 2))

ax.set_xlim([min_vals[0], max_vals[0]])
ax.set_ylim([min_vals[1], max_vals[1]])
ax.set_zlim([min_vals[2], max_vals[2]])

markers = ['o', '^']  # Example: 'o' for trial 1, '^' for trial 20
# Plot each trial
for i, trial_index in enumerate(trial_indices):
    x = final_output[0, trial_index, :]
    y = final_output[1, trial_index, :]
    z = final_output[2, trial_index, :]

    # Color each point by its index in time
    for t in range(time_points):
        ax.scatter(x[t], y[t], z[t], color=cmap(t), s=15, marker=markers[i])  # s is the size of the point

import matplotlib.lines as mlines
tr3 = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                          markersize=12, label='Trial 3')
tr25 = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                          markersize=12, label='Trial 25')

plt.legend(handles=[tr3, tr25], loc='upper right')

# Optional: Add colorbar to indicate the progression of time
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=2000))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='horizontal')
cbar.set_label('Time post-stim. (ms)')
#plt.title('3D Plot of Trials 1 and 20')
plt.show()

#save as svg
save_path= os.path.join(euc_dist_demo_folder, 'static_example_trials.svg')
plt.savefig(save_path, format='svg')  # Save the plot as SVG file