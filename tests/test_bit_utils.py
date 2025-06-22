import numpy as np
import pytest

from bittrace.bit_utils import (bitcount8, hamming_distance_matrix, pack_bits, unpack_bits,
                                bitwise_and_packed, bitwise_or_packed, bitwise_xor_packed, bitwise_not_packed,
                                popcount_packed, hamming_distance_packed, hamming_distance_matrix_numba)

def test_bitcount8_simple():
    assert bitcount8(0b00000000) == 0
    assert bitcount8(0b11111111) == 8
    assert bitcount8(0b10101010) == 4

def test_pack_unpack_bits():
    arr = np.array([[0, 1, 1, 0, 0, 1, 0, 1]], dtype=np.uint8)
    packed = pack_bits(arr)
    unpacked = unpack_bits(packed, 8)
    np.testing.assert_array_equal(arr, unpacked)

def test_hamming_distance_matrix_basic():
    # Two arrays: one [00000001], other [00000011]
    X = np.array([[1]], dtype=np.uint8)
    Y = np.array([[3]], dtype=np.uint8)
    D = hamming_distance_matrix(X, Y)
    assert D.shape == (1, 1)
    assert D[0, 0] == 1  # Only 1 bit differs (1: 00000001, 3: 00000011)

def test_hamming_distance_matrix_multi():
    X = np.array([[0b00001111], [0b11110000]], dtype=np.uint8)
    Y = np.array([[0b00000000], [0b11111111]], dtype=np.uint8)
    D = hamming_distance_matrix(X, Y)
    # D[0,0]: 00001111 vs 00000000 → 4 bits differ
    # D[0,1]: 00001111 vs 11111111 → 4 bits differ
    # D[1,0]: 11110000 vs 00000000 → 4 bits differ
    # D[1,1]: 11110000 vs 11111111 → 4 bits differ
    assert np.all(D == 4)

if __name__ == "__main__":
    pytest.main([__file__])
