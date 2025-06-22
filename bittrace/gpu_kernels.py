# bittrace/gpu_kernels.py

import numpy as np
from numba import cuda, uint8, uint16

@cuda.jit
def population_xor_kernel(population, reference, out):
    """
    Bitwise XOR: Each sample in `population` (N, M) with `reference` (M,)
    """
    i, j = cuda.grid(2)
    if i < population.shape[0] and j < population.shape[1]:
        out[i, j] = population[i, j] ^ reference[j]

def population_xor_gpu(population, reference):
    """
    Launch kernel on population (N, M) and reference (M,)
    Returns out (N, M)
    """
    N, M = population.shape
    out = np.empty_like(population)
    threadsperblock = (16, 16)
    blockspergrid_x = (N + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (M + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    population_xor_kernel[blockspergrid, threadsperblock](population, reference, out)
    return out

@cuda.jit
def hamming_distance_matrix_kernel(A, B, out):
    """
    Computes Hamming distance between all pairs in A (N, M) and B (K, M).
    """
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
    """
    Computes all-pair Hamming distance (N, K) between A (N, M) and B (K, M)
    Returns out (N, K)
    """
    N, M = A.shape
    K = B.shape[0]
    out = np.empty((N, K), dtype=np.int32)
    threadsperblock = (8, 8)
    blockspergrid_x = (N + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (K + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    hamming_distance_matrix_kernel[blockspergrid, threadsperblock](A, B, out)
    return out

@cuda.jit
def population_bitwise_op_kernel(pop, ref, out, op_code):
    """
    Apply a bitwise operation between each row of `pop` and `ref` (broadcasted).
    op_code: 0=XOR, 1=AND, 2=OR, 3=NAND, 4=NOT(pop only)
    """
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
        elif op_code == 4:  # NOT (ignores ref)
            out[i, j] = ~a
        else:
            out[i, j] = 0  # invalid op

def population_bitwise_gpu(pop, ref, op='xor'):
    """
    Flexible GPU bitwise op for (N, M) pop and (M,) ref.
    op: 'xor', 'and', 'or', 'nand', 'not'
    """
    op_dict = {'xor': 0, 'and': 1, 'or': 2, 'nand': 3, 'not': 4}
    op_code = op_dict[op]
    N, M = pop.shape
    out = np.empty_like(pop)
    threadsperblock = (16, 16)
    blockspergrid = ((N + 15) // 16, (M + 15) // 16)
    # For NOT, just ignore ref; pass dummy zeros
    ref_arg = ref if op != 'not' else np.zeros(M, dtype=np.uint8)
    population_bitwise_op_kernel[blockspergrid, threadsperblock](pop, ref_arg, out, op_code)
    return out

@cuda.jit
def population_mutation_with_mask_kernel(pop, mutation_mask):
    """
    Mutate population bits by XORing with mutation_mask.
    pop: (N, M) uint8
    mutation_mask: (N, M) uint8 (random bits set where mutation happens)
    """
    i, j = cuda.grid(2)
    if i < pop.shape[0] and j < pop.shape[1]:
        pop[i, j] ^= mutation_mask[i, j]