import numpy as np
import pytest
import bittrace.gpu_kernels as gpu

def test_population_bitwise_gpu():
    pop = np.array([[0b10101010, 0b11110000],
                    [0b01010101, 0b00001111]], dtype=np.uint8)
    ref = np.array([0b11110000, 0b10101010], dtype=np.uint8)
    out_xor = gpu.population_bitwise_gpu(pop, ref, op='xor')
    expected = np.bitwise_xor(pop, ref)
    assert np.array_equal(out_xor, expected)
    out_and = gpu.population_bitwise_gpu(pop, ref, op='and')
    expected = np.bitwise_and(pop, ref)
    assert np.array_equal(out_and, expected)
    out_or = gpu.population_bitwise_gpu(pop, ref, op='or')
    expected = np.bitwise_or(pop, ref)
    assert np.array_equal(out_or, expected)
    out_not = gpu.population_bitwise_gpu(pop, None, op='not')
    assert np.array_equal(out_not, np.bitwise_xor(pop, 0xFF))
    out_id = gpu.population_bitwise_gpu(pop, None, op='identity')
    assert np.array_equal(out_id, pop)

def test_launch_mutation_in_place():
    pop = np.zeros((10, 8), dtype=np.uint8)
    pop_before = pop.copy()
    gpu.launch_mutation(pop, mutation_rate=1.0)
    assert not np.array_equal(pop, pop_before)

def test_hamming_distance_matrix_gpu():
    A = np.array([[0b11110000, 0b00001111],
                  [0b10101010, 0b01010101]], dtype=np.uint8)
    B = np.array([[0b00000000, 0b00000000],
                  [0b11111111, 0b11111111]], dtype=np.uint8)
    dmat = gpu.hamming_distance_matrix_gpu(A, B)
    assert dmat.shape == (2, 2)
    assert np.all(dmat >= 0)
    assert dmat.dtype == np.int32

def test_gpu_popcount_rows():
    arr = np.array([[0b11111111]*8, [0b00000000]*8], dtype=np.uint8)
    counts = gpu.gpu_popcount_rows(arr)
    assert counts[0] == 64
    assert counts[1] == 0

def test_gpu_hamming_layerwise():
    X_layers = np.array([
        [[0b11110000, 0b00001111], [0b00000000, 0b00000000]],
        [[0b10101010, 0b01010101], [0b11111111, 0b11111111]],
        [[0b00000000, 0b11110000], [0b00001111, 0b10101010]]
    ], dtype=np.uint8)
    proto_layers = np.array([[0b00000000, 0b00000000], [0b11111111, 0b11111111]], dtype=np.uint8)
    scores = gpu.gpu_hamming_layerwise(X_layers, proto_layers)
    assert scores.shape == (3,)
    assert np.all(scores >= 0)

if __name__ == "__main__":
    pytest.main([__file__])
