import numpy as np
from bittrace.gpu_kernels import hamming_distance_matrix_gpu

def gpu_kmedoids_update(pop, assignments, K, prev_medoids_idx=None):
    """
    Args:
        pop: (N, M) np.uint8, full population (bit-packed)
        assignments: (N,) int, cluster assignment for each member (0..K-1)
        K: number of clusters
        prev_medoids_idx: (K,) int or None. If given, used for clusters with no members.

    Returns:
        new_medoids_idx: (K,) int, indices into pop for the new medoid of each cluster.
    """
    new_medoids_idx = np.full(K, -1, dtype=int)
    for k in range(K):
        cluster_idx = np.where(assignments == k)[0]
        if len(cluster_idx) == 0:
            # No members; keep old medoid index if available, else random
            if prev_medoids_idx is not None:
                new_medoids_idx[k] = prev_medoids_idx[k]
            else:
                new_medoids_idx[k] = np.random.randint(0, pop.shape[0])
            continue
        cluster = pop[cluster_idx]
        # GPU Hamming distance matrix
        dist_matrix = hamming_distance_matrix_gpu(cluster, cluster)
        row_sums = dist_matrix.sum(axis=1)
        min_idx = np.argmin(row_sums)
        # Map back to index in original pop
        new_medoids_idx[k] = cluster_idx[min_idx]
    return new_medoids_idx
