# bittrace/model.py

import numpy as np
from bittrace import gpu_kernels as gpu

class BitTraceModel:
    def __init__(self, bit_length, num_layers, pop_size):
        self.bit_length = bit_length
        self.num_layers = num_layers
        self.pop_size = pop_size

        self.bytes_per_individual = (bit_length + 7) // 8
        self.population = np.random.randint(0, 256, size=(pop_size, self.bytes_per_individual), dtype=np.uint8)
        self.layer_plan = self._generate_random_layer_plan(num_layers, self.bytes_per_individual)
        self.accuracy_log = []
        self.residues = None

    def _generate_random_layer_plan(self, num_layers, width):
        ops = ['xor', 'and', 'or', 'nand', 'not', 'identity']
        return [(np.random.choice(ops), np.random.randint(0, 256, size=width, dtype=np.uint8))
                for _ in range(num_layers)]

    def set_layer_plan(self, plan):
        self.layer_plan = plan

    def evolve(self, mutation_rate=0.01):
        gpu.launch_mutation(self.population, mutation_rate)

    def forward(self, X):
        """
        Apply the model's bitwise transformation pipeline to the input batch X.
        X: [N, B] np.uint8 bit-packed inputs
        Returns: transformed [N, B] residues
        """
        out = X.copy()
        for op, ref in self.layer_plan:
            out = gpu.population_bitwise_gpu(out, ref, op=op)
        return out

    def fit(self, X_train, y_train, generations, mutation_rate=0.01, eval_fn=None):
        """
        Run evolution for a fixed number of generations.
        """
        for gen in range(generations):
            self.evolve(mutation_rate)

            # Evaluate fitness: apply model to training data
            transformed = self.forward(X_train)

            if eval_fn:
                acc = eval_fn(transformed, y_train)
                self.accuracy_log.append(acc)

    def predict(self, X):
        return self.forward(X)

    def save_checkpoint(self, path):
        np.savez(path,
                 population=self.population,
                 layer_plan=self.layer_plan,
                 bit_length=self.bit_length,
                 residues=self.residues,
                 accuracy_log=self.accuracy_log)

    def load_checkpoint(self, path):
        data = np.load(path, allow_pickle=True)
        self.population = data['population']
        self.layer_plan = data['layer_plan'].tolist()
        self.bit_length = int(data['bit_length'])
        self.residues = data['residues']
        self.accuracy_log = data['accuracy_log'].tolist()
