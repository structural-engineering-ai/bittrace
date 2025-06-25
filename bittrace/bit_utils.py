import numpy as np
from numba import njit, prange

@njit
def bitcount8(x):
    count = 0
    for i in range(8):
        count += (x >> i) & 1
    return count

# --------------------------
# Packing/Unpacking
# --------------------------

def pack_bits(arr):
    """Pack bits along axis=1 (as in MNIST, shape [N, 784] to [N, 98])."""
    return np.packbits(arr.astype(np.uint8), axis=1)

def unpack_bits(arr, bit_length):
    """Unpack and truncate to bit_length columns."""
    unpacked = np.unpackbits(arr.astype(np.uint8), axis=1)
    return unpacked[:, :bit_length]

# --------------------------
# Pure NumPy Bitwise Ops
# --------------------------

def bitwise_and_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_and(a.astype(np.uint8), b.astype(np.uint8))

def bitwise_or_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_or(a.astype(np.uint8), b.astype(np.uint8))

def bitwise_xor_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(a.astype(np.uint8), b.astype(np.uint8))

def bitwise_not_packed(a: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(a.astype(np.uint8), 0xFF)

# --------------------------
# Hamming & Popcount (NumPy)
# --------------------------

def popcount_packed(a: np.ndarray) -> np.ndarray:
    """Counts 1 bits per row in a packed uint8 array [N, B]."""
    lookup = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)
    a = a.astype(np.uint8)
    return np.array([lookup[row].sum() for row in a], dtype=np.uint16)

def hamming_distance_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rowwise Hamming distance between two packed uint8 arrays of same shape."""
    return popcount_packed(np.bitwise_xor(a.astype(np.uint8), b.astype(np.uint8)))

# --------------------------
# Fast Numba Hamming/Popcount (CPU Parallel)
# --------------------------

@njit(parallel=True)
def hamming_distance_matrix_numba(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
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

@njit(parallel=True)
def popcount_matrix_numba(X: np.ndarray) -> np.ndarray:
    n, n_bytes = X.shape
    out = np.empty(n, dtype=np.uint16)
    for i in prange(n):
        d = 0
        for k in range(n_bytes):
            v = int(X[i, k])
            while v:
                v &= v - 1
                d += 1
        out[i] = d
    return out

# --------------------------
# Row-wise Byte Shift Ops
# --------------------------

def roll_bytes_right(a: np.ndarray, shift: int = 1) -> np.ndarray:
    """Rowwise roll (circular shift) of bytes right by 'shift'."""
    return np.roll(a, shift=shift, axis=1)

def roll_bytes_left(a: np.ndarray, shift: int = 1) -> np.ndarray:
    """Rowwise roll (circular shift) of bytes left by 'shift'."""
    return np.roll(a, shift=-shift, axis=1)

# --------------------------
# Expose only what you need
# --------------------------
__all__ = [
    "bitcount8",
    "pack_bits", "unpack_bits",
    "bitwise_and_packed", "bitwise_or_packed", "bitwise_xor_packed", "bitwise_not_packed",
    "popcount_packed", "hamming_distance_packed",
    "hamming_distance_matrix_numba", "popcount_matrix_numba",
    "roll_bytes_right", "roll_bytes_left"
]
