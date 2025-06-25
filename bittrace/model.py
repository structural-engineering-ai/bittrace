import numpy as np
from bittrace import gpu_kernels as gpu

class BitTraceModel:
    def __init__(self, bit_length, num_layers, pop_size=32, mutation_rate=0.02, use_gpu=True):
        self.bit_length = bit_length
        self.num_layers = num_layers
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.use_gpu = use_gpu

        self.bytes_per = (bit_length + 7) // 8
        self.population = np.random.randint(0, 256, size=(pop_size, self.bytes_per), dtype=np.uint8)
        self.layer_plan = self._generate_random_layer_plan(num_layers, self.bytes_per)
        self.accuracy_log = []
        self.residues = None

    def _generate_random_layer_plan(self, num_layers, width):
        ops = ['xor', 'and', 'or', 'nand', 'not', 'identity']
        return [(np.random.choice(ops), np.random.randint(0, 256, size=width, dtype=np.uint8))
                for _ in range(num_layers)]

    def set_layer_plan(self, plan):
        self.layer_plan = plan

    def evolve(self):
        gpu.launch_mutation(self.population, self.mutation_rate)
        # Mutate the layer plan with a small chance
        if np.random.rand() < 0.2:
            self.layer_plan = self._generate_random_layer_plan(self.num_layers, self.bytes_per)

    def forward(self, X):
        out = X.copy()
        for op, ref in self.layer_plan:
            out = gpu.population_bitwise_gpu(out, ref, op=op)
        return out

    def predict_labels(self, transformed, pop_subset=None):
        """Binary prediction by minimal Hamming distance to each individual in the population (or a subset)."""
        if pop_subset is None:
            pop_subset = self.population
        dists = gpu.hamming_distance_matrix_gpu(transformed, pop_subset)
        min_idx = np.argmin(dists, axis=1)
        # Each row: which pop member is closest
        preds = (min_idx == 0).astype(np.uint8)  # Assume pop[0] is the "positive class"
        return preds

    def fit(self, X_train, y_train, X_val=None, y_val=None, generations=100, elite_frac=0.1):
        """
        Evolutionary population training loop:
        - Keeps elite fraction each generation
        - Mutates rest based on elite templates
        """
        pop_size = self.pop_size
        elite_count = max(1, int(pop_size * elite_frac))
        best_acc = 0
        best_population = None
        best_layer_plan = None
        self.accuracy_log = []
        rng = np.random.default_rng(42)

        for gen in range(generations):
            # 1. Evaluate accuracy of each individual (over validation set)
            scores = []
            for i in range(pop_size):
                # Temporary single-member "model"
                proto = self.population[i:i+1]
                # Forward with this "proto" as population
                self_clone = BitTraceModel(
                    bit_length=self.bit_length,
                    num_layers=self.num_layers,
                    pop_size=1,
                    mutation_rate=self.mutation_rate,
                    use_gpu=self.use_gpu,
                )
                self_clone.population = proto.copy()
                self_clone.layer_plan = list(self.layer_plan)
                preds = self_clone.predict_labels(self_clone.forward(X_val if X_val is not None else X_train))
                acc = np.mean(preds == (y_val if y_val is not None else y_train))
                scores.append(acc)

            scores = np.array(scores)
            self.accuracy_log.append(scores.max())

            # 2. Elitism: keep best N individuals
            elite_idx = np.argsort(scores)[-elite_count:]
            elites = self.population[elite_idx].copy()

            # 3. Mutate rest based on elites
            new_population = []
            for i in range(pop_size):
                if i < elite_count:
                    new_population.append(elites[i % elite_count].copy())
                else:
                    # Pick a random elite and mutate it
                    base = elites[rng.integers(elite_count)].copy()
                    noise = rng.integers(0, 256, size=base.shape, dtype=np.uint8)
                    mask = rng.random(base.shape) < self.mutation_rate
                    base[mask] ^= noise[mask]
                    new_population.append(base)
            self.population = np.stack(new_population, axis=0)

            # 4. Occasionally mutate layer plan (structure)
            if rng.random() < 0.25:
                self.layer_plan = self._generate_random_layer_plan(self.num_layers, self.bytes_per)

            # 5. Track and restore best so far
            max_acc = scores.max()
            if max_acc > best_acc:
                best_acc = max_acc
                best_idx = scores.argmax()
                best_population = self.population[best_idx:best_idx+1].copy()
                best_layer_plan = list(self.layer_plan)

            if gen % 10 == 0 or gen == generations - 1:
                print(f"[Gen {gen:04d}] Best Acc: {max_acc:.4f} | Overall Best: {best_acc:.4f}")

        # Restore best found at end
        if best_population is not None:
            self.population = best_population
            self.layer_plan = best_layer_plan


    def evaluate_accuracy(self, X, y_true):
        transformed = self.forward(X)
        # Use best member (assumed index 0) for class assignment
        proto = self.population[0:1]
        dists = gpu.hamming_distance_matrix_gpu(transformed, proto)
        scores = 1.0 - (dists.flatten() / self.bit_length)
        preds = (scores >= 0.5).astype(np.uint8)
        acc = np.mean(preds == y_true)
        print(f"[evaluate_accuracy] Accuracy: {acc:.4f}")
        return acc

    def save_checkpoint(self, path):
        # Save using pickle to handle the layer_plan (list of tuples)
        np.savez(path,
                 population=self.population,
                 layer_plan=np.array(self.layer_plan, dtype=object),
                 bit_length=self.bit_length,
                 accuracy_log=np.array(self.accuracy_log),
                 allow_pickle=True)

    def load_checkpoint(self, path):
        data = np.load(path, allow_pickle=True)
        self.population = data['population']
        self.layer_plan = data['layer_plan'].tolist()
        self.bit_length = int(data['bit_length'])
        self.accuracy_log = data['accuracy_log'].tolist()
