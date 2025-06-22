import numpy as np
import pytest
from numba import cuda
from bittrace.gpu_kernels import *
from bittrace.clustering import *


def cpu_population_xor(pop, ref):
    return np.bitwise_xor(pop, ref)

def cpu_hamming_distance_matrix(A, B):
    N, M = A.shape
    K = B.shape[0]
    out = np.zeros((N, K), dtype=np.int32)
    for i in range(N):
        for j in range(K):
            out[i, j] = np.unpackbits(np.bitwise_xor(A[i], B[j])).sum()
    return out

def test_population_xor():
    N, M = 32, 16
    pop = np.random.randint(0, 256, (N, M), dtype=np.uint8)
    ref = np.random.randint(0, 256, M, dtype=np.uint8)
    gpu_out = population_xor_gpu(pop, ref)
    cpu_out = cpu_population_xor(pop, ref)
    assert np.all(gpu_out == cpu_out)

def test_hamming_distance_matrix():
    N, M, K = 16, 8, 20
    A = np.random.randint(0, 256, (N, M), dtype=np.uint8)
    B = np.random.randint(0, 256, (K, M), dtype=np.uint8)
    gpu_out = hamming_distance_matrix_gpu(A, B)
    cpu_out = cpu_hamming_distance_matrix(A, B)
    assert np.all(gpu_out == cpu_out)

@pytest.mark.parametrize("op", ['xor', 'and', 'or', 'nand', 'not'])
def test_population_bitwise_gpu(op):
    N, M = 20, 10
    pop = np.random.randint(0, 256, (N, M), dtype=np.uint8)
    ref = np.random.randint(0, 256, M, dtype=np.uint8)
    # CPU reference
    if op == 'xor':
        cpu = np.bitwise_xor(pop, ref)
    elif op == 'and':
        cpu = np.bitwise_and(pop, ref)
    elif op == 'or':
        cpu = np.bitwise_or(pop, ref)
    elif op == 'nand':
        cpu = np.bitwise_not(np.bitwise_and(pop, ref))
    elif op == 'not':
        cpu = np.bitwise_not(pop)
    gpu = population_bitwise_gpu(pop, ref, op=op)
    assert np.all(gpu == cpu)

def test_hamming_assignment():
    N, M, K = 30, 16, 3
    pop = np.random.randint(0, 256, (N, M), dtype=np.uint8)
    meds = np.random.randint(0, 256, (K, M), dtype=np.uint8)
    gpu_dists = hamming_distance_matrix_gpu(pop, meds)
    cpu_dists = np.zeros((N, K), dtype=np.int32)
    for i in range(N):
        for j in range(K):
            cpu_dists[i, j] = np.unpackbits(np.bitwise_xor(pop[i], meds[j])).sum()
    assert np.all(gpu_dists == cpu_dists)
    # Test assignments match
    assert np.all(np.argmin(gpu_dists, axis=1) == np.argmin(cpu_dists, axis=1))

def test_gpu_kmedoids_update():
    N, M, K = 40, 16, 3
    pop = np.random.randint(0, 256, (N, M), dtype=np.uint8)
    assignments = np.random.randint(0, K, N)
    meds = gpu_kmedoids_update(pop, assignments, K)
    assert meds.shape == (K, M)
    # Optionally check that each medoid is a real cluster member
    for k in range(K):
        cluster_idx = np.where(assignments == k)[0]
        if len(cluster_idx) > 0:
            found = any(np.all(meds[k] == pop[i]) for i in cluster_idx)
            assert found

def test_population_mutation_with_mask():
    N, M = 10, 5
    # Original population: random uint8
    pop = np.random.randint(0, 256, (N, M), dtype=np.uint8)
    pop_gpu = cuda.to_device(pop.copy())

    # Create random mutation mask with ~10% bits set
    mutation_rate = 0.1
    mask = np.zeros((N, M), dtype=np.uint8)
    num_bits = N * M * 8
    num_mutations = int(num_bits * mutation_rate)

    # Randomly pick bit positions to flip
    mutation_indices = np.random.choice(num_bits, num_mutations, replace=False)
    for idx in mutation_indices:
        byte_idx = idx // 8
        bit_idx = idx % 8
        row = byte_idx // M
        col = byte_idx % M
        mask[row, col] |= (1 << bit_idx)

    mask_gpu = cuda.to_device(mask)

    # Launch kernel
    threadsperblock = (16, 16)
    blockspergrid = ((N + 15) // 16, (M + 15) // 16)
    population_mutation_with_mask_kernel[blockspergrid, threadsperblock](pop_gpu, mask_gpu)

    # Copy back mutated population
    mutated_pop = pop_gpu.copy_to_host()

    # Validate mutation correctness: bits flipped exactly where mask has 1 bits
    for i in range(N):
        for j in range(M):
            expected = pop[i, j] ^ mask[i, j]
            assert mutated_pop[i, j] == expected