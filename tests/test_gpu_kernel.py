import numpy as np
import pytest
from bittrace import gpu_kernels as gpu

def test_population_bitwise_op_xor():
    pop = np.array([[0b10101010, 0b11110000]], dtype=np.uint8)
    ref = np.array([0b11001100, 0b10101010], dtype=np.uint8)
    expected = np.array([[0b01100110, 0b01011010]], dtype=np.uint8)
    out = gpu.population_bitwise_gpu(pop, ref, op='xor')
    assert np.array_equal(out, expected)

def test_population_bitwise_op_not():
    pop = np.array([[0b10101010, 0b11110000]], dtype=np.uint8)
    expected = np.array([[~0b10101010 & 0xFF, ~0b11110000 & 0xFF]], dtype=np.uint8)
    out = gpu.population_bitwise_gpu(pop, np.zeros(pop.shape[1], dtype=np.uint8), op='not')
    assert np.array_equal(out, expected)

def test_hamming_distance_matrix_small():
    A = np.array([[0b00001111, 0b11110000]], dtype=np.uint8)
    B = np.array([[0b11110000, 0b00001111]], dtype=np.uint8)
    dist = gpu.hamming_distance_matrix_gpu(A, B)
    assert dist.shape == (1,1)
    assert dist[0,0] == 16

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

@pytest.mark.parametrize("N,M", [(1, 16), (5, 16)])
def test_population_mutation_with_mask_kernel_shape(N, M):
    pop = np.zeros((N, M), dtype=np.uint8)
    mutation_mask = np.full((N, M), 0xFF, dtype=np.uint8)  # flip all bits
    if N == 0 or M == 0:
        pytest.skip("Empty input, skipping kernel launch")
    threadsperblock = (16, 16)
    blockspergrid = ((N + 15) // 16, (M + 15) // 16)
    gpu.population_mutation_with_mask_kernel[blockspergrid, threadsperblock](pop, mutation_mask)
    expected = np.full((N, M), 0xFF, dtype=np.uint8)
    assert np.array_equal(pop, expected)

def test_popcount8_device_function_indirect():
    from numba import cuda

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

@pytest.mark.parametrize("op", ['xor', 'and', 'or', 'nand', 'not'])
def test_population_bitwise_op_all_ops(op):
    pop = np.array([[0b10101010, 0b11001100]], dtype=np.uint8)
    ref = np.array([0b11110000, 0b00001111], dtype=np.uint8)
    out = gpu.population_bitwise_gpu(pop, ref, op=op)
    assert out.shape == pop.shape
    assert out.dtype == np.uint8

def test_population_xor_kernel_basic():
    pop = np.array([[0b11110000]], dtype=np.uint8)
    ref = np.array([0b10101010], dtype=np.uint8)
    out = gpu.population_xor_gpu(pop, ref)
    expected = np.array([[0b01011010]], dtype=np.uint8)
    assert np.array_equal(out, expected)

@pytest.mark.parametrize("N,M", [(1, 16), (5, 16)])
def test_empty_population_mutation(N, M):
    pop = np.zeros((N, M), dtype=np.uint8)
    mutation_mask = np.zeros_like(pop)
    if N == 0 or M == 0:
        pytest.skip("Empty input, skipping kernel launch")
    threadsperblock = (16, 16)
    blockspergrid = ((N + 15) // 16, (M + 15) // 16)
    gpu.population_mutation_with_mask_kernel[blockspergrid, threadsperblock](pop, mutation_mask)
    assert pop.shape == (N, M)

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
