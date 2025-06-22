# bittrace/gpu_kernels.py

import numpy as np
from numba import cuda, uint8, int32

# Helper to compute launch dims for 2D kernels optimized for large bitstrings
def compute_launch_dims(shape0, shape1):
    # Threads per block: keep within 1024 max CUDA threads/block
    # Tune for bitstring dimension (shape1) and population (shape0)
    threads_x = 256  # along bitstring length (columns)
    threads_y = 4    # along population (rows)
    threadsperblock = (threads_y, threads_x)
    blockspergrid_x = (shape0 + threads_y - 1) // threads_y
    blockspergrid_y = (shape1 + threads_x - 1) // threads_x
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    return blockspergrid, threadsperblock

# ------------------------
# Bitwise Population Kernels
# ------------------------

@cuda.jit
def population_xor_kernel(population, reference, out):
    i, j = cuda.grid(2)
    if i < population.shape[0] and j < population.shape[1]:
        out[i, j] = population[i, j] ^ reference[j]

def population_xor_gpu(population, reference):
    N, M = population.shape
    out = np.empty_like(population)
    blockspergrid, threadsperblock = compute_launch_dims(N, M)
    population_xor_kernel[blockspergrid, threadsperblock](population, reference, out)
    return out

@cuda.jit
def hamming_distance_matrix_kernel(A, B, out):
    i, j = cuda.grid(2)
    if i < A.shape[0] and j < B.shape[0]:
        dist = 0
        for k in range(A.shape[1]):
            val = A[i, k] ^ B[j, k]
            count = 0
            for b in range(8):
                count += val & 1
                val >>= 1
            dist += count
        out[i, j] = dist

def hamming_distance_matrix_gpu(A, B):
    N, M = A.shape
    K = B.shape[0]
    out = np.empty((N, K), dtype=np.int32)
    blockspergrid, threadsperblock = compute_launch_dims(N, K)
    hamming_distance_matrix_kernel[blockspergrid, threadsperblock](A, B, out)
    return out

@cuda.jit
def population_bitwise_op_kernel(pop, ref, out, op_code):
    i, j = cuda.grid(2)
    if i < pop.shape[0] and j < pop.shape[1]:
        a = pop[i, j]
        b = ref[j]
        if op_code == 0:  # XOR
            out[i, j] = a ^ b
        elif op_code == 1:  # AND
            out[i, j] = a & b
        elif op_code == 2:  # OR
            out[i, j] = a | b
        elif op_code == 3:  # NAND
            out[i, j] = ~(a & b)
        elif op_code == 4:  # NOT (ignore ref)
            out[i, j] = ~a
        else:
            out[i, j] = 0

def population_bitwise_gpu(pop, ref, op='xor'):
    op_dict = {'xor': 0, 'and': 1, 'or': 2, 'nand': 3, 'not': 4}
    op_code = op_dict[op]
    N, M = pop.shape
    out = np.empty_like(pop)
    blockspergrid, threadsperblock = compute_launch_dims(N, M)
    ref_arg = ref if op != 'not' else np.zeros(M, dtype=np.uint8)
    population_bitwise_op_kernel[blockspergrid, threadsperblock](pop, ref_arg, out, op_code)
    return out

@cuda.jit
def population_mutation_with_mask_kernel(pop, mutation_mask):
    i, j = cuda.grid(2)
    if i < pop.shape[0] and j < pop.shape[1]:
        pop[i, j] ^= mutation_mask[i, j]

def population_mutation_gpu(pop, mutation_mask):
    N, M = pop.shape
    blockspergrid, threadsperblock = compute_launch_dims(N, M)
    population_mutation_with_mask_kernel[blockspergrid, threadsperblock](pop, mutation_mask)

# ------------------------
# GPU k-Medoids Kernels
# ------------------------

@cuda.jit(device=True)
def popcount8(x):
    count = 0
    for i in range(8):
        count += (x >> i) & 1
    return count

@cuda.jit
def cluster_distance_matrix_kernel(population, cluster_assignments, cluster_id,
                                   member_indices, dist_matrix):
    i, j = cuda.grid(2)
    cluster_size = member_indices.shape[0]
    M = population.shape[1]

    if i < cluster_size and j < cluster_size:
        idx_i = member_indices[i]
        idx_j = member_indices[j]
        dist = 0
        for k in range(M):
            val = population[idx_i, k] ^ population[idx_j, k]
            dist += popcount8(val)
        dist_matrix[i, j] = dist

@cuda.jit
def medoid_selection_kernel(dist_matrix, medoid_idx_out):
    cluster_size = dist_matrix.shape[0]

    min_sum = 1 << 30  # large number
    min_idx = 0
    for i in range(cluster_size):
        row_sum = 0
        for j in range(cluster_size):
            row_sum += dist_matrix[i, j]
        if row_sum < min_sum:
            min_sum = row_sum
            min_idx = i

    medoid_idx_out[0] = min_idx

def gpu_kmedoids_update(population, cluster_assignments, num_clusters):
    N, M = population.shape
    medoid_indices = np.empty(num_clusters, dtype=np.int32)

    for cluster_id in range(num_clusters):
        members = np.where(cluster_assignments == cluster_id)[0]
        cluster_size = len(members)
        if cluster_size == 0:
            medoid_indices[cluster_id] = -1
            continue

        d_members = cuda.to_device(members)
        dist_matrix = cuda.device_array((cluster_size, cluster_size), dtype=np.int32)

        threadsperblock = (16, 16)
        blockspergrid_x = (cluster_size + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (cluster_size + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        cluster_distance_matrix_kernel[blockspergrid, threadsperblock](
            population, cluster_assignments, cluster_id, d_members, dist_matrix
        )

        medoid_idx_out = cuda.device_array(1, dtype=np.int32)
        medoid_selection_kernel[1, 1](dist_matrix, medoid_idx_out)

        medoid_local_idx = medoid_idx_out.copy_to_host()[0]
        medoid_indices[cluster_id] = members[medoid_local_idx]

    return medoid_indices
