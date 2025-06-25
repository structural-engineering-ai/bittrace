import numpy as np
from numba import cuda
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
# Bitwise Operations (Main Op)
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

# ------------------------
# GPU: POPCOUNT (Rowwise, for Hamming)
# ------------------------

@cuda.jit
def popcount8_packed(arr, out):
    """
    arr: [N, n_bytes] uint8
    out: [N] int32 (sum of bit-ones per row)
    """
    n = arr.shape[0]
    b = arr.shape[1]
    i = cuda.grid(1)
    if i < n:
        total = 0
        for j in range(b):
            v = arr[i, j]
            v = v - ((v >> 1) & 0x55)
            v = (v & 0x33) + ((v >> 2) & 0x33)
            total += (((v + (v >> 4)) & 0x0F) * 0x01)
        out[i] = total

def gpu_popcount_rows(arr):
    N = arr.shape[0]
    out = np.zeros(N, dtype=np.int32)
    threadsperblock = 128
    blockspergrid = (N + threadsperblock - 1) // threadsperblock
    popcount8_packed[blockspergrid, threadsperblock](arr, out)
    return out

# ------------------------
# GPU: Hamming Distance Matrix
# ------------------------

@cuda.jit
def hamming_distance_matrix_kernel(A, B, out):
    i, j = cuda.grid(2)
    if i < A.shape[0] and j < B.shape[0]:
        dist = 0
        for k in range(A.shape[1]):
            val = A[i, k] ^ B[j, k]
            # popcount8, branchless:
            val = val - ((val >> 1) & 0x55)
            val = (val & 0x33) + ((val >> 2) & 0x33)
            dist += ((val + (val >> 4)) & 0x0F)
        out[i, j] = dist

def hamming_distance_matrix_gpu(A, B):
    """
    A: [N, M], B: [K, M]  (packed bit arrays)
    Returns: [N, K] Hamming distances
    """
    N, M = A.shape
    K = B.shape[0]
    out = np.empty((N, K), dtype=np.int32)
    blocks, threads = compute_launch_dims(N, K)
    hamming_distance_matrix_kernel[blocks, threads](A, B, out)
    return out

# ------------------------
# GPU: Layerwise Hamming (for BitTrace)
# ------------------------

def gpu_hamming_layerwise(X_layers, proto_layers):
    """
    X_layers: [N, L, B] packed uint8 (samples, layers, bytes)
    proto_layers: [L, B] packed uint8 (prototype for each layer)
    Returns: [N] int32 (total Hamming per sample, summed over all layers)
    """
    N, L, B = X_layers.shape
    total_hamming = np.zeros(N, dtype=np.int32)
    arr_dev = cuda.device_array((N, B), dtype=np.uint8)
    out_dev = cuda.device_array(N, dtype=np.int32)
    for l in range(L):
        arr_host = np.bitwise_xor(X_layers[:, l, :], proto_layers[l][None, :])
        cuda.to_device(arr_host, to=arr_dev)
        threadsperblock = 128
        blockspergrid = (N + threadsperblock - 1) // threadsperblock
        popcount8_packed[blockspergrid, threadsperblock](arr_dev, out_dev)
        total_hamming += out_dev.copy_to_host()
    return total_hamming

# ------------------------
# GPU: Population Mutation (in place)
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
# GPU K-MEDOIDS CLUSTERING (for BitBlocks/BitTrace)
# ------------------------

def gpu_kmedoids_clustering(residues, K, max_iter=20, seed=42):
    """
    residues: [N, bytes_per] packed uint8 (on host, operated on GPU)
    K: number of clusters (bitblocks/classes)
    Returns: medoid_indices [K], assignments [N]
    """
    N = residues.shape[0]
    rng = np.random.default_rng(seed)
    medoid_indices = rng.choice(N, K, replace=False)
    medoids = residues[medoid_indices]
    assignments = np.zeros(N, dtype=np.int32)

    for it in range(max_iter):
        # Step 1: Assign each residue to closest medoid (GPU)
        dist_mat = hamming_distance_matrix_gpu(residues, medoids)
        assignments = np.argmin(dist_mat, axis=1)
        changed = False
        new_medoids = np.zeros_like(medoids)
        for k in range(K):
            members = np.where(assignments == k)[0]
            if len(members) == 0:
                # fallback: re-pick randomly
                new_medoids[k] = residues[rng.integers(N)]
                changed = True
                continue
            members_arr = residues[members]
            dists_intra = hamming_distance_matrix_gpu(members_arr, members_arr)
            distsums = dists_intra.sum(axis=1)
            new_idx = members[np.argmin(distsums)]
            new_medoids[k] = residues[new_idx]
            if not np.array_equal(medoids[k], residues[new_idx]):
                changed = True
        if not changed:
            break
        medoids = new_medoids
    # Final assignment
    dist_mat = hamming_distance_matrix_gpu(residues, medoids)
    assignments = np.argmin(dist_mat, axis=1)
    final_indices = np.array(
        [np.where((residues == m).all(axis=1))[0][0] for m in medoids]
    )
    return final_indices, assignments

# ------------------------
# GPU PARALLEL BLOCKWISE PREDICTION
# ------------------------

def predict_blocks_gpu(X, block_medoids):
    """
    X: [N, bytes_per] packed bits, block_medoids: [C, bytes_per]
    Returns: [N, C] Hamming distances (lower = closer to that block)
    """
    return hamming_distance_matrix_gpu(X, block_medoids)
