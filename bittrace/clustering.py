import numpy as np
from bittrace.gpu_kernels import hamming_distance_matrix_gpu

def gpu_kmedoids_update(pop, assignments, K):
    """
    Args:
        pop: (N, M) np.uint8, full population
        assignments: (N,) int, cluster assignment for each member (0..K-1)
        K: number of clusters
    Returns:
        new_medoids: (K, M) np.uint8, updated medoid for each cluster
    """
    new_medoids = np.zeros((K, pop.shape[1]), dtype=np.uint8)
    for k in range(K):
        # Indices of members in cluster k
        cluster_idx = np.where(assignments == k)[0]
        if len(cluster_idx) == 0:
            # No members; keep old medoid or random
            continue
        cluster = pop[cluster_idx]
        # Step 1: GPU all-pair Hamming distances
        dist_matrix = hamming_distance_matrix_gpu(cluster, cluster)
        # Step 2: Find member with lowest total distance
        row_sums = dist_matrix.sum(axis=1)
        min_idx = np.argmin(row_sums)
        new_medoids[k] = cluster[min_idx]
    return new_medoids
