import numpy as np
from bittrace import gpu_kernels as gpu

BITWISE_OPS = ["xor", "and", "or", "nand", "not", "identity"]

class BitTraceModel:
    def __init__(self, config, population_init=None):
        self.bit_length = config["bit_length"]
        self.num_layers = config["num_layers"]
        self.allowed_ops = config.get("allowed_ops", BITWISE_OPS)
        self.byte_length = (self.bit_length + 7) // 8
        self.num_clusters = config.get("num_clusters", 10)
        self.config = config

        # Initialize population
        if population_init is not None:
            self.population = population_init
        else:
            self.population = None

        # Initialize medoids and medoids_idx
        if self.population is not None and self.population.size > 0:
            n = len(self.population)
            k = self.num_clusters
            # Ensure k does not exceed population size
            self.medoids_idx = np.arange(min(k, n))
            self.medoids = self.population[self.medoids_idx]
        else:
            self.medoids_idx = None
            self.medoids = None

        # Initialize layer plan to a random plan to avoid errors
        self.layer_plan = self.random_layer_plan()

    def forward_layers(self):
        if self.population is None:
            raise RuntimeError("Population not set on model.")
        if not hasattr(self, 'layer_plan') or self.layer_plan is None:
            raise RuntimeError("Layer plan not set on model.")
        for op, ref in self.layer_plan:
            self.population = gpu.population_bitwise_gpu(self.population, ref, op=op)

    def _apply_op(self, X, op, ref=None):
        if op == "not":
            return np.bitwise_xor(X, 0xFF)
        elif op == "identity":
            return X
        elif op in ["and", "or", "xor", "nand"]:
            if ref is None:
                raise ValueError(f"Bitwise op {op} needs a reference.")
            fn = getattr(np, f"bitwise_{op}" if op != "nand" else "bitwise_and")
            Y = fn(X, ref)
            if op == "nand":
                Y = np.bitwise_not(Y)
            return Y
        else:
            raise ValueError(f"Unsupported op: {op}")

    def run_batch(self, X, layer_plan):
        X = X.copy()
        for op, ref in layer_plan:
            X = self._apply_op(X, op, ref)
        return X

    def evaluate_supervised_accuracy(self, residues, y_true, n_clusters=None, bitwise_kmedoids=None, hungarian_remap=None, return_pred=False):
        n_clusters = n_clusters or self.num_clusters
        labels, centroids = bitwise_kmedoids(residues, n_clusters=n_clusters, n_iter=10)
        y_pred = hungarian_remap(labels, y_true)
        acc = (y_pred == y_true).mean()
        if return_pred:
            return acc, y_pred
        return acc

    def bit_entropy(self, residues):
        unpacked = np.unpackbits(residues, axis=1)[:, :self.bit_length]
        p = unpacked.mean(axis=0)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        entropy = -np.mean(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        return entropy

    def random_layer_plan(self, seed=None):
        rng = np.random.default_rng(seed)
        plan = []
        for _ in range(self.num_layers):
            op = rng.choice(self.allowed_ops)
            ref = rng.integers(0, 256, self.byte_length, dtype=np.uint8) if op not in ["identity", "not"] else None
            plan.append((op, ref))
        return plan

    def hamming_distances(self):
        if self.medoids is None:
            raise RuntimeError("Medoids not set on model.")
        return gpu.hamming_distance_matrix_gpu(self.population, self.medoids)

    def kmedoids_assign(self, dist_matrix):
        """
        Assign clusters based on distance matrix by choosing the cluster with minimum distance.
        Then update medoids using GPU accelerated k-medoids update.
        """
        cluster_assignments = np.argmin(dist_matrix, axis=1)  # Assign cluster indices by nearest medoid

        medoids = gpu.gpu_kmedoids_update(self.population, cluster_assignments, self.num_clusters)

        # Ensure medoids is a flat numpy array of ints to avoid inhomogeneous shape errors
        medoids = np.array(medoids, dtype=np.int32).flatten()

        # Debug: print medoids info (can comment out after debugging)
        print(f"[DEBUG] Medoids shape: {medoids.shape}, dtype: {medoids.dtype}")
        print(f"[DEBUG] Medoids: {medoids}")

        return cluster_assignments, medoids

    def update_medoids(self, cluster_assignments):
        self.medoids_idx = gpu.gpu_kmedoids_update(self.population, cluster_assignments, self.num_clusters)
        valid_indices = self.medoids_idx >= 0
        self.medoids = np.zeros((self.num_clusters, self.byte_length), dtype=np.uint8)
        self.medoids[valid_indices] = self.population[self.medoids_idx[valid_indices]]

    def mutate_population(self, mutation_rate):
        mutation_mask = np.random.rand(*self.population.shape) < mutation_rate
        mutation_mask = mutation_mask.astype(np.uint8) * 255
        gpu.population_mutation_gpu(self.population, mutation_mask)

    def predict(self, X, label_map=None):
        dist_matrix = gpu.hamming_distance_matrix_gpu(X, self.medoids)
        cluster_assignments = np.argmin(dist_matrix, axis=1)
        if label_map is not None:
            return np.array([label_map[c] for c in cluster_assignments])
        return cluster_assignments
