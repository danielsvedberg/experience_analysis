from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

def optimize_t(mat, dm):
    '''
    This function takes a distance matrix and a trial matrix and returns the optimal distance threshold for clustering
    as calculated by the silhouette score.
    :param mat: your original data matrix, where each row is a trial and each column is a variable (i.e. neuron)
    :param dm: your distance matrix
    :return: cluster_labels: the cluster labels for each trial, best_t: the optimized distance threshold
    '''
    #optimize distance threshold t using silhouette score###
    # first we calculate the min and max possible t values using the distances
    linkages = linkage(dm, method='ward')
    distances = linkages[:, 2]
    min_t = distances.min()
    max_t = distances.max()

    # then we calculate the silhouette score for a range of t values
    best_score = -1
    best_t = None
    cluster_labels = None
    for t in np.linspace(min_t, max_t, 1000):
        clabs = fcluster(linkages, t=t, criterion='distance')  # get the cluster labels for this t value
        # impose the requirement that there be at least 2 clusters, and fewer clusters than trials
        if (len(np.unique(clabs)) > 1) and (len(np.unique(clabs)) < len(mat)):
            score = silhouette_score(dm, clabs)
            if score > best_score:
                best_score = score
                cluster_labels = clabs
                best_t = t
    return cluster_labels, best_t
def make_consensus_matrix(mat_list):
    """
    This function takes a list of trial matrices and a list of identifying information for each matrix and returns a
    consensus matrix
    :param
        mat_list: a list of trial matrices, wherein each, each row is a trial and each column is a variable (i.e. neuron)
    :return:
        consensus_matrix: a consensus matrix
        cluster_label_arrays: a list of cluster labels for each data matrix
        thresholds: a list of optimized distance thresholds for each matrix
    """

    #find the matrix with greatest number of trials
    n_trials = 0
    for mat in mat_list:
        if mat.shape[1] > n_trials:
            n_trials = mat.shape[1]
    #find the number of matrices to take the consensus over
    n_mats = len(mat_list)

    #initialize lists to store the cluster labels and thresholds for each matrix
    thresholds = []
    cluster_label_arrays = []

    #make an empty consensus matrix populated by nans. This will be filled with 1s and 0s to indicate whether two trials are in the same cluster
    consensus_matrix = np.empty((n_trials, n_trials, n_mats))
    consensus_matrix[:] = np.nan

    for m, mat in enumerate(mat_list): #iterate over the matrices, index number will be in m, matrix will be in mat
        dm = pdist(mat, metric='euclidean') #get the linearized distance matrix

        cluster_labels, best_t = optimize_t(mat, dm) #get optimized threshold and the cluster labels
        cluster_label_arrays.append(cluster_labels)
        thresholds.append(best_t)

        #loop through each pairing of trials and create the clustering-distance matrix for each data matrix
        for i in range(n_trials):
            for j in range(n_trials):
                if j >= i:
                    if cluster_labels[i] != cluster_labels[j]: #if the pair of trials are in different clusters, set the value 1
                        consensus_matrix[i, j, m] = 1
                    else: #if the pair of trials are in the same cluster, set the value 0
                        consensus_matrix[i, j, m] = 0

    # replace all nan with 0
    consensus_matrix = np.nan_to_num(consensus_matrix)
    # fold the consensus matrix to make it symmetrical
    consensus_matrix = (consensus_matrix + consensus_matrix.T) / 2
    #calculate the final consensus matrix by taking the mean of the consensus matrices for each data matrix
    consensus_matrix = np.nanmean(consensus_matrix, axis=2)

    return consensus_matrix, cluster_label_arrays, thresholds
