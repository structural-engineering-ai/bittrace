import numpy as np
from numba import njit
import pytest

from bittrace.bit_utils import (
    bitcount8,
    pack_bits,
    unpack_bits,
    bitwise_and_packed,
    bitwise_or_packed,
    bitwise_xor_packed,
    bitwise_not_packed,
    popcount_packed,
    hamming_distance_packed,
    hamming_distance_matrix_numba,
)

def test_bitcount8_simple():
    assert bitcount8(0b00000000) == 0
    assert bitcount8(0b11111111) == 8
    assert bitcount8(0b10101010) == 4

def test_pack_unpack_bits():
    arr = np.array([[0, 1, 1, 0, 0, 1, 0, 1]], dtype=np.uint8)
    packed = pack_bits(arr)
    unpacked = unpack_bits(packed, 8)
    np.testing.assert_array_equal(arr, unpacked)

def test_bitwise_primitives():
    a = np.array([[0b10101010]], dtype=np.uint8)
    b = np.array([[0b11001100]], dtype=np.uint8)
    assert np.all(bitwise_and_packed(a, b) == [[0b10001000]])
    assert np.all(bitwise_or_packed(a, b) == [[0b11101110]])
    assert np.all(bitwise_xor_packed(a, b) == [[0b01100110]])
    assert np.all(bitwise_not_packed(a) == [[0b01010101]])

def test_popcount_packed():
    arr = np.array([[0b11110000], [0b00001111]], dtype=np.uint8)
    counts = popcount_packed(arr)
    assert np.all(counts == 4)

def test_hamming_distance_packed():
    a = np.array([[0b10101010]], dtype=np.uint8)
    b = np.array([[0b11110000]], dtype=np.uint8)
    d = hamming_distance_packed(a, b)
    assert np.all(d == 4)

@njit(parallel=True)
def hamming_distance_matrix_numba(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # Ensure correct type (no float!); this is redundant if always correct upstream, but extra safe.
    X = X.astype(np.uint8)
    Y = Y.astype(np.uint8)
    n, m, n_bytes = X.shape[0], Y.shape[0], X.shape[1]
    out = np.empty((n, m), dtype=np.uint16)
    for i in prange(n):
        for j in range(m):
            d = 0
            for k in range(n_bytes):
                v = int(X[i, k]) ^ int(Y[j, k])
                while v:
                    v &= v - 1
                    d += 1
            out[i, j] = d
    return out

if __name__ == "__main__":
    pytest.main([__file__])
