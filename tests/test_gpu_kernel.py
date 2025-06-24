# tests/test_gpu_kernel.py

import pytest
import numpy as np
from numba import cuda
from bittrace import gpu_kernels as gpu
from numba.cuda.random import create_xoroshiro128p_states

def test_hamming_distance_matrix_small():
    A = np.array([[0b00001111, 0b11110000]], dtype=np.uint8)
    B = np.array([[0b11110000, 0b00001111]], dtype=np.uint8)
    dist = gpu.hamming_distance_matrix_gpu(A, B)
    assert dist.shape == (1, 1)
    assert dist[0, 0] == 16

def test_hamming_distance_matrix_multiple():
    A = np.array([[0b00000000], [0b11111111]], dtype=np.uint8)
    B = np.array([[0b00000000], [0b11111111]], dtype=np.uint8)
    dist = gpu.hamming_distance_matrix_gpu(A, B)
    assert dist.shape == (2, 2)
    assert dist[0, 0] == 0
    assert dist[1, 1] == 0
    assert dist[0, 1] == 8
    assert dist[1, 0] == 8

def test_gpu_kmedoids_update_basic():
    population = np.array([
        [0b00001111, 0b11110000],
        [0b00001111, 0b11110001],
        [0b11110000, 0b00001111]
    ], dtype=np.uint8)
    cluster_assignments = np.array([0, 0, 1])
    medoids = gpu.gpu_kmedoids_update(population, cluster_assignments, 2)
    assert medoids.shape == (2,)
    assert all(0 <= idx < len(population) for idx in medoids)

@pytest.mark.parametrize("op, expected", [
    ('xor',     np.array([[90, 63]], dtype=np.uint8)),      # 0b01011010, 0b00111111
    ('and',     np.array([[160, 192]], dtype=np.uint8)),    # 0b10100000, 0b11000000
    ('or',      np.array([[250, 255]], dtype=np.uint8)),    # 0b11111010, 0b11111111
    ('nand',    np.array([[95, 63]], dtype=np.uint8)),      # ~(AND) results
    ('not',     np.array([[85, 51]], dtype=np.uint8)),      # ~0b10101010 = 0b01010101, etc.
])
def test_population_bitwise_op_all_ops(op, expected):
    pop = np.array([[0b10101010, 0b11001100]], dtype=np.uint8)
    ref = np.array([0b11110000, 0b11110011], dtype=np.uint8)
    out = gpu.population_bitwise_gpu(pop, ref, op=op)
    assert np.array_equal(out, expected)

def test_population_mutation_with_mask_kernel_shape():
    N, M = 2, 8
    pop = np.zeros((N, M), dtype=np.uint8)
    mutation_mask = np.full((N, M), 0xFF, dtype=np.uint8)
    threadsperblock = (16, 16)
    blockspergrid = ((N + 15) // 16, (M + 15) // 16)
    gpu.population_mutation_with_mask_kernel[blockspergrid, threadsperblock](pop, mutation_mask)
    expected = np.full((N, M), 0xFF, dtype=np.uint8)
    assert np.array_equal(pop, expected)

def test_popcount8_device_function_indirect():
    @cuda.jit
    def test_kernel(out):
        i = cuda.grid(1)
        if i == 0:
            total = 0
            for val in range(256):
                total += gpu.popcount8(val)
            out[0] = total

    out = cuda.device_array(1, dtype=np.int32)
    test_kernel[1, 1](out)
    result = out.copy_to_host()[0]
    assert result == 1024

def test_gpu_bitwise_mutation_randomized():
    pop = np.zeros((4, 4), dtype=np.uint8)
    orig = pop.copy()
    mutation_rate = 1.0

    block = (16, 16)
    grid = ((pop.shape[0] + 15) // 16, (pop.shape[1] + 15) // 16)
    rng_states = create_xoroshiro128p_states(pop.size, seed=123)

    d_pop = cuda.to_device(pop)
    gpu.mutate_population_bitwise[grid, block](d_pop, mutation_rate, rng_states)
    d_pop.copy_to_host(pop)

    assert not np.array_equal(pop, orig), "Mutation did not alter population"

def test_gpu_bitwise_mutation_low_rate():
    pop = np.zeros((100, 4), dtype=np.uint8)
    orig = pop.copy()
    mutation_rate = 0.01

    block = (16, 16)
    grid = ((pop.shape[0] + 15) // 16, (pop.shape[1] + 15) // 16)
    rng_states = create_xoroshiro128p_states(pop.size, seed=321)

    d_pop = cuda.to_device(pop)
    gpu.mutate_population_bitwise[grid, block](d_pop, mutation_rate, rng_states)
    d_pop.copy_to_host(pop)

    diff = np.sum(pop != orig)
    assert 0 < diff < pop.size, f"Unexpected number of mutations: {diff}"
