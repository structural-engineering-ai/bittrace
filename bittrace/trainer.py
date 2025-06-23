import time
import csv
import numpy as np
import os
from bittrace import gpu_kernels as gpu
from bittrace.checkpoint import ModelCheckpoint

def compute_accuracy(model, X, y, label_map=None):
    """Compute accuracy using the model's predict method."""
    y_pred = model.predict(X, label_map=label_map)
    return np.mean(y_pred == y)

def train_bittrace_full(
    model,
    num_generations,
    mutation_rate,
    checkpoint_every,
    checkpoint_dir,
    val_data=None,
    early_stopping_patience=10,
    resume_from_checkpoint=None,
    log_csv_path=None,
    label_map=None,
    run_name="bittrace_model",
    verbose=True
):
    from bittrace.checkpoint import ModelCheckpoint

    ckpt_util = ModelCheckpoint(checkpoint_dir, model_name=run_name)

    best_val_acc = -np.inf
    best_model_state = None
    patience_counter = 0

    # Resume if specified
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        if verbose:
            print(f"Resuming from checkpoint {resume_from_checkpoint}")
        state = ModelCheckpoint.load(resume_from_checkpoint)
        model.population = state['population']
        model.medoids_idx = state['medoids_idx']
        model.medoids = model.population[model.medoids_idx]
    else:
        # Initialize medoids before training if not resumed
        if model.population is None:
            raise RuntimeError("Model population not set before training.")
        n = len(model.population)
        k = model.num_clusters
        # Initialize medoids indices to first k members or fewer if population smaller
        model.medoids_idx = np.arange(min(k, n))
        model.medoids = model.population[model.medoids_idx]

    # Setup CSV logging
    csv_file = None
    csv_writer = None
    if log_csv_path:
        csv_file = open(log_csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Generation", "Time", "ValAccuracy"])

    print_interval = 10  # Print every N generations

    if verbose:
        print(f"\n{'Gen':<6} {'Time':<10} {'ValAccuracy':<10}")

    for gen in range(num_generations):
        start_time = time.time()
        model.forward_layers()  # <<< run the stack of layers on population
        dist_mat = model.hamming_distances()
        cluster_assignments, _ = model.kmedoids_assign(dist_mat)
        model.update_medoids(cluster_assignments)
        model.mutate_population(mutation_rate)

        elapsed = time.time() - start_time
        val_acc = None

        if val_data is not None:
            val_X, val_y = val_data
            val_acc = compute_accuracy(model, val_X, val_y, label_map=label_map)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {
                    'population': model.population.copy(),
                    'medoids_idx': model.medoids_idx.copy()
                }
                patience_counter = 0
            else:
                patience_counter += 1

        # Print header once, then row at interval/first/last gen
        if verbose and (gen == 0 or (gen + 1) % print_interval == 0 or gen == num_generations - 1):
            print(f"{gen+1:<6} {elapsed:<10.3f} {val_acc if val_acc is not None else '':<10.4f}")

        if csv_writer:
            csv_writer.writerow([gen+1, f"{elapsed:.3f}", val_acc if val_acc is not None else ""])

        if val_data is not None and patience_counter >= early_stopping_patience:
            if verbose:
                print(f"Early stopping at gen {gen+1} with best val accuracy {best_val_acc:.4f}")
            break

    if best_model_state is not None:
        model.population = best_model_state['population']
        model.medoids_idx = best_model_state['medoids_idx']
        model.medoids = model.population[model.medoids_idx]

    # Save final best model (to fixed file—overwritten by meta-runner logic)
    final_ckpt_path = ckpt_util.save(model, generation="final", extra_info={"val_acc": best_val_acc})

    if csv_writer:
        csv_file.close()

    # Attach for meta-runner's access, optional
    model.best_val_acc = best_val_acc

    return model, final_ckpt_path
