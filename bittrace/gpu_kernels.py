# bittrace/gpu_kernels.py

import numpy as np
from numba import cuda, int32, uint8
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

# ------------------------
# Helpers
# ------------------------

def compute_launch_dims(shape0, shape1):
    threads_x = 256
    threads_y = 4
    threadsperblock = (threads_y, threads_x)
    blockspergrid = ((shape0 + threads_y - 1) // threads_y,
                     (shape1 + threads_x - 1) // threads_x)
    return blockspergrid, threadsperblock

# ------------------------
# Bitwise Operations
# ------------------------

@cuda.jit
def population_bitwise_op_kernel(pop, ref, out, op_code):
    i, j = cuda.grid(2)
    if i < pop.shape[0] and j < pop.shape[1]:
        a = pop[i, j]
        b = ref[j]
        if op_code == 0: out[i, j] = a ^ b
        elif op_code == 1: out[i, j] = a & b
        elif op_code == 2: out[i, j] = a | b
        elif op_code == 3: out[i, j] = ~(a & b)
        elif op_code == 4: out[i, j] = ~a
        elif op_code == 5: out[i, j] = a
        else: out[i, j] = 0

def population_bitwise_gpu(pop, ref, op='xor'):
    op_dict = {'xor': 0, 'and': 1, 'or': 2, 'nand': 3, 'not': 4, 'identity': 5}
    op_code = op_dict[op]
    N, M = pop.shape
    out = np.empty_like(pop)
    ref_arg = np.zeros(M, dtype=np.uint8) if ref is None or op in ('not', 'identity') else ref
    blocks, threads = compute_launch_dims(N, M)
    population_bitwise_op_kernel[blocks, threads](pop, ref_arg, out, op_code)
    return out

@cuda.jit
def population_mutation_with_mask_kernel(pop, mutation_mask):
    i, j = cuda.grid(2)
    if i < pop.shape[0] and j < pop.shape[1]:
        pop[i, j] ^= mutation_mask[i, j]

def population_mutation_gpu(pop, mutation_mask):
    N, M = pop.shape
    blocks, threads = compute_launch_dims(N, M)
    population_mutation_with_mask_kernel[blocks, threads](pop, mutation_mask)

# ------------------------
# XOR Kernel
# ------------------------

@cuda.jit
def population_xor_kernel(population, reference, out):
    i, j = cuda.grid(2)
    if i < population.shape[0] and j < population.shape[1]:
        out[i, j] = population[i, j] ^ reference[j]

def population_xor_gpu(population, reference):
    N, M = population.shape
    out = np.empty_like(population)
    blocks, threads = compute_launch_dims(N, M)
    population_xor_kernel[blocks, threads](population, reference, out)
    return out

# ------------------------
# Mutation Kernel
# ------------------------

@cuda.jit
def mutate_population_bitwise(pop, mutation_rate, rng_states):
    i, j = cuda.grid(2)
    if i < pop.shape[0] and j < pop.shape[1]:
        idx = i * pop.shape[1] + j
        r = xoroshiro128p_uniform_float32(rng_states, idx)
        if r < mutation_rate:
            raw = xoroshiro128p_uniform_float32(rng_states, idx + 1)
            mask = int(raw * 256) & 0xFF
            pop[i, j] ^= mask

def launch_mutation(population, mutation_rate):
    N, M = population.shape
    blocks, threads = compute_launch_dims(N, M)
    rng_states = create_xoroshiro128p_states(N * M, seed=42)
    d_pop = cuda.to_device(population)
    mutate_population_bitwise[blocks, threads](d_pop, mutation_rate, rng_states)
    d_pop.copy_to_host(population)

# ------------------------
# Hamming Distance Matrix
# ------------------------

@cuda.jit
def hamming_distance_matrix_kernel(A, B, out):
    i, j = cuda.grid(2)
    if i < A.shape[0] and j < B.shape[0]:
        dist = 0
        for k in range(A.shape[1]):
            val = A[i, k] ^ B[j, k]
            for b in range(8):
                dist += val & 1
                val >>= 1
        out[i, j] = dist

def hamming_distance_matrix_gpu(A, B):
    N, M = A.shape
    K = B.shape[0]
    out = np.empty((N, K), dtype=np.int32)
    blocks, threads = compute_launch_dims(N, K)
    hamming_distance_matrix_kernel[blocks, threads](A, B, out)
    return out

# ------------------------
# k-Medoids GPU
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
    size = member_indices.shape[0]
    if i < size and j < size:
        dist = 0
        for k in range(population.shape[1]):
            val = population[member_indices[i], k] ^ population[member_indices[j], k]
            dist += popcount8(val)
        dist_matrix[i, j] = dist

@cuda.jit
def medoid_selection_kernel(dist_matrix, medoid_idx_out):
    size = dist_matrix.shape[0]
    min_sum = 1 << 30
    min_idx = 0
    for i in range(size):
        s = 0
        for j in range(size):
            s += dist_matrix[i, j]
        if s < min_sum:
            min_sum = s
            min_idx = i
    medoid_idx_out[0] = min_idx

def gpu_kmedoids_update(population, cluster_assignments, num_clusters):
    N, M = population.shape
    medoids = np.empty(num_clusters, dtype=np.int32)

    for cluster_id in range(num_clusters):
        members = np.where(cluster_assignments == cluster_id)[0]
        if len(members) == 0:
            medoids[cluster_id] = -1
            continue

        d_members = cuda.to_device(members)
        dist_matrix = cuda.device_array((len(members), len(members)), dtype=np.int32)
        blocks = ((len(members) + 15) // 16, (len(members) + 15) // 16)
        threads = (16, 16)

        cluster_distance_matrix_kernel[blocks, threads](population, cluster_assignments,
                                                        cluster_id, d_members, dist_matrix)

        medoid_idx_out = cuda.device_array(1, dtype=np.int32)
        medoid_selection_kernel[1, 1](dist_matrix, medoid_idx_out)

        local_idx = medoid_idx_out.copy_to_host()[0]
        medoids[cluster_id] = members[local_idx]

    return medoids
