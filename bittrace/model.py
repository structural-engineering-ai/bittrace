# bittrace/model.py

import numpy as np
from bittrace import gpu_kernels as gpu
import time

class BitTraceModel:
    def __init__(self, population_init, num_clusters, device=True):
        self.population = population_init  # np.uint8 packed bits shape (N, M)
        self.num_clusters = num_clusters
        self.device = device
        self.medoids_idx = np.random.choice(len(population_init), num_clusters, replace=False)
        self.medoids = self.population[self.medoids_idx]

    def bitwise_layer_op(self, op_name, reference):
        """GPU-accelerated bitwise op across population with reference vector."""
        out = gpu.population_bitwise_gpu(self.population, reference, op=op_name)
        self.population = out

    def hamming_distances(self):
        """Compute GPU Hamming distances between population and medoids."""
        dist_mat = gpu.hamming_distance_matrix_gpu(self.population, self.medoids)
        return dist_mat

    def kmedoids_assign(self, dist_mat):
        """Assign clusters based on min distance in dist_mat."""
        cluster_assignments = np.argmin(dist_mat, axis=1)
        assignment_distances = dist_mat[np.arange(dist_mat.shape[0]), cluster_assignments]
        return cluster_assignments, assignment_distances

    def update_medoids(self, cluster_assignments):
        """Update medoids using GPU k-medoids update kernel."""
        new_medoids_idx = gpu.gpu_kmedoids_update(self.population, cluster_assignments, self.num_clusters)
        # Filter invalid (-1) medoids fallback to previous medoids
        for i, idx in enumerate(new_medoids_idx):
            if idx == -1:
                new_medoids_idx[i] = self.medoids_idx[i]
        self.medoids_idx = new_medoids_idx
        self.medoids = self.population[self.medoids_idx]

    def mutate_population(self, mutation_rate):
        """Mutate population bits in-place using GPU kernel and random mask."""
        N, M = self.population.shape
        mutation_mask_bits = np.random.rand(N, M * 8) < mutation_rate
        mutation_mask_packed = np.packbits(mutation_mask_bits.astype(np.uint8), axis=1)
        threadsperblock = (16, 16)
        blockspergrid_x = (N + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (M + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        gpu.population_mutation_with_mask_kernel[blockspergrid, threadsperblock](self.population, mutation_mask_packed)

    def save_checkpoint(self, checkpoint_dir, generation):
        """Save checkpoint with population and medoids indices."""
        path = f"{checkpoint_dir}/checkpoint_gen_{generation}.npz"
        np.savez_compressed(path, population=self.population, medoids_idx=self.medoids_idx)
        print(f"Checkpoint saved to {path}")
