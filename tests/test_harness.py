# tests/test_harness.py

import numpy as np
import time
import os
from bittrace.model import BitTraceModel
from bittrace.trainer import train_bittrace_full

def test_small_training_run():
    population_size = 100    # Small population for quick test
    bit_length_bytes = 32    # 32 bytes = 256 bits packed
    num_clusters = 5
    generations = 10
    mutation_rate = 0.01
    checkpoint_every = 5
    checkpoint_dir = "./checkpoints_test"

    os.makedirs(checkpoint_dir, exist_ok=True)

    population_init = np.random.randint(0, 256, size=(population_size, bit_length_bytes), dtype=np.uint8)
    model = BitTraceModel(population_init, num_clusters=num_clusters, device=True)

    start = time.time()
    trained_model = train_bittrace_full(
        model,
        num_generations=generations,
        mutation_rate=mutation_rate,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir,
    )
    end = time.time()

    print(f"Training completed in {end - start:.3f} seconds.")
    assert trained_model.medoids.shape == (num_clusters, bit_length_bytes)

if __name__ == "__main__":
    test_small_training_run()
    print("Test harness run complete.")
