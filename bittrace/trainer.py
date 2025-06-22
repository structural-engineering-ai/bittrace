import time
import os
import numpy as np
from bittrace import gpu_kernels as gpu
import csv

def compute_accuracy(model, X, y):
    dist_mat = gpu.hamming_distance_matrix_gpu(X, model.medoids)
    cluster_assignments = np.argmin(dist_mat, axis=1)
    accuracy = np.mean(cluster_assignments == y)
    return accuracy

def save_checkpoint(model, checkpoint_dir, generation):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_gen_{generation}.npz")
    np.savez_compressed(path,
                        population=model.population,
                        medoids_idx=model.medoids_idx,
                        num_clusters=model.num_clusters)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(path):
    data = np.load(path)
    return data

def train_bittrace_full(
    model,
    num_generations,
    mutation_rate,
    checkpoint_every,
    checkpoint_dir,
    val_data=None,
    early_stopping_patience=10,
    resume_from_checkpoint=None,
    log_csv_path=None
):
    best_val_acc = -np.inf
    best_model_state = None
    patience_counter = 0

    # Resume if specified
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Resuming from checkpoint {resume_from_checkpoint}")
        data = load_checkpoint(resume_from_checkpoint)
        model.population = data['population']
        model.medoids_idx = data['medoids_idx']
        model.medoids = model.population[model.medoids_idx]

    # Setup CSV logging
    if log_csv_path:
        csv_file = open(log_csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Generation", "Time", "ValAccuracy"])
    else:
        csv_writer = None

    for gen in range(num_generations):
        start_time = time.time()

        # Example operation: XOR with random reference (you can extend)
        reference = np.random.randint(0, 256, model.population.shape[1], dtype=np.uint8)
        model.bitwise_layer_op('xor', reference)

        dist_mat = model.hamming_distances()
        cluster_assignments, _ = model.kmedoids_assign(dist_mat)

        model.update_medoids(cluster_assignments)
        model.mutate_population(mutation_rate)

        elapsed = time.time() - start_time
        val_acc = None

        if val_data is not None:
            val_X, val_y = val_data
            val_acc = compute_accuracy(model, val_X, val_y)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {
                    'population': model.population.copy(),
                    'medoids_idx': model.medoids_idx.copy()
                }
                patience_counter = 0
            else:
                patience_counter += 1

        print(f"Gen {gen+1}/{num_generations} Time: {elapsed:.3f}s"
              + (f" Val Accuracy: {val_acc:.4f}" if val_acc is not None else ""))

        if csv_writer:
            csv_writer.writerow([gen+1, f"{elapsed:.3f}", val_acc if val_acc else ""])

        if (gen + 1) % checkpoint_every == 0:
            save_checkpoint(model, checkpoint_dir, gen + 1)

        if val_data is not None and patience_counter >= early_stopping_patience:
            print(f"Early stopping at gen {gen+1} with best val accuracy {best_val_acc:.4f}")
            break

    if best_model_state is not None:
        model.population = best_model_state['population']
        model.medoids_idx = best_model_state['medoids_idx']
        model.medoids = model.population[model.medoids_idx]

    if csv_writer:
        csv_file.close()

    return model
