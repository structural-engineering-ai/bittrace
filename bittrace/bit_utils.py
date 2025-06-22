import numpy as np
from numba import njit, prange

@njit
def bitcount8(x):
    count = 0
    for i in range(8):
        count += (x >> i) & 1
    return count

@njit(parallel=True)
def hamming_distance_matrix(X, Y):
    n, m = X.shape[0], Y.shape[0]
    d = np.empty((n, m), dtype=np.uint16)
    for i in prange(n):
        for j in range(m):
            count = 0
            for k in range(X.shape[1]):
                count += bitcount8(X[i, k] ^ Y[j, k])
            d[i, j] = count
    return d

def pack_bits(arr):
    return np.packbits(arr, axis=1)

def unpack_bits(arr, bit_length):
    unpacked = np.unpackbits(arr, axis=1)
    return unpacked[:, :bit_length]

# -------- PURE NUMPY PRIMITIVES --------

def bitwise_and_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise AND between two packed uint8 arrays."""
    return np.bitwise_and(a, b)

def bitwise_or_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise OR between two packed uint8 arrays."""
    return np.bitwise_or(a, b)

def bitwise_xor_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise XOR between two packed uint8 arrays."""
    return np.bitwise_xor(a, b)

def bitwise_not_packed(a: np.ndarray) -> np.ndarray:
    """Bitwise NOT for packed uint8 array."""
    return np.bitwise_xor(a, 0xFF)

def popcount_packed(a: np.ndarray) -> np.ndarray:
    """Counts 1 bits for each row in a packed uint8 array."""
    lookup = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)
    return np.array([lookup[row].sum() for row in a], dtype=np.uint16)

def hamming_distance_packed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rowwise Hamming distance between two packed uint8 arrays of same shape."""
    return popcount_packed(np.bitwise_xor(a, b))

# --------- NUMBA FAST VERSIONS (CPU) ---------

@njit(parallel=True)
def hamming_distance_matrix_numba(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    n, m, n_bytes = X.shape[0], Y.shape[0], X.shape[1]
    out = np.empty((n, m), dtype=np.uint16)
    for i in prange(n):
        for j in range(m):
            d = 0
            for k in range(n_bytes):
                v = X[i, k] ^ Y[j, k]
                # Brian Kernighan's method for bit counting
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
            v = X[i, k]
            while v:
                v &= v - 1
                d += 1
        out[i] = d
    return out

# ---- ROW/BYTEWISE ROLL OPERATIONS ----

def roll_bytes_right(a: np.ndarray, shift: int = 1) -> np.ndarray:
    """Rowwise roll (circular shift) of bytes right by 'shift'."""
    return np.roll(a, shift=shift, axis=1)

def roll_bytes_left(a: np.ndarray, shift: int = 1) -> np.ndarray:
    """Rowwise roll (circular shift) of bytes left by 'shift'."""
    return np.roll(a, shift=-shift, axis=1)