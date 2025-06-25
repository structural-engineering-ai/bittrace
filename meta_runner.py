# meta_runner.py

import os
import numpy as np
import csv
from config import META_CONFIG, BITTRACE_CONFIG
from bittrace.data_loader import load_bittrace_digit
from bittrace.model import BitTraceModel

def meta_runner():
    digit = META_CONFIG.get("digit", 0)  # Default: single digit (update as needed)
    print(f"\n=== MetaRunner: Starting meta-arch search for digit={digit} ===")

    # --- Load Data ---
    X_train, y_train = load_bittrace_digit(digit, split="train", base_dir=BITTRACE_CONFIG["bitblock_sets_dir"])
    X_val, y_val     = load_bittrace_digit(digit, split="val", base_dir=BITTRACE_CONFIG["bitblock_sets_dir"])
    X_test, y_test   = load_bittrace_digit(digit, split="test", base_dir=BITTRACE_CONFIG["bitblock_sets_dir"])
    print(f"Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    n_trials      = META_CONFIG.get("tier1_trials", 32)
    pop_size      = META_CONFIG.get("tier1_pop_size", 32)
    generations   = META_CONFIG.get("tier1_generations", 25)
    val_threshold = META_CONFIG.get("tier1_threshold", 0.60)
    out_csv       = f"meta_trials_digit{digit}.csv"
    best_csv      = f"meta_besties_digit{digit}.csv"

    rng = np.random.default_rng(META_CONFIG["random_seed"])
    all_results = []
    best_models = []

    with open(out_csv, "w", newline='') as fout, open(best_csv, "w", newline='') as fbest:
        writer_all = csv.writer(fout)
        writer_best = csv.writer(fbest)
        writer_all.writerow(["trial", "bit_length", "num_layers", "mutation_rate", "val_acc", "test_acc"])
        writer_best.writerow(["trial", "bit_length", "num_layers", "mutation_rate", "val_acc", "test_acc"])

        overall_best_acc = 0
        overall_best_model = None

        for trial in range(n_trials):
            # --- Random config ---
            bit_length = int(rng.integers(*META_CONFIG["bit_length_range"]))
            num_layers = int(rng.integers(*META_CONFIG["num_layers_range"]))
            mutation_rate = float(rng.uniform(*META_CONFIG["mutation_rate_range"]))

            print(f"\n[T1:Trial {trial+1}/{n_trials}] bit_length={bit_length} | num_layers={num_layers} | mutation={mutation_rate:.4f}")

            # --- Initialize Model ---
            model = BitTraceModel(
                bit_length=bit_length,
                num_layers=num_layers,
                pop_size=pop_size,
                mutation_rate=mutation_rate,
                use_gpu=True,
            )

            # --- Train ---
            model.fit(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                generations=generations
            )
            val_acc = model.evaluate_accuracy(X_val, y_val)
            test_acc = model.evaluate_accuracy(X_test, y_test)

            writer_all.writerow([trial+1, bit_length, num_layers, mutation_rate, val_acc, test_acc])
            fout.flush()

            # Track "passing" configs
            if val_acc >= val_threshold:
                writer_best.writerow([trial+1, bit_length, num_layers, mutation_rate, val_acc, test_acc])
                fbest.flush()
                best_models.append({
                    "trial": trial+1,
                    "bit_length": bit_length,
                    "num_layers": num_layers,
                    "mutation_rate": mutation_rate,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "model": model,
                })
                print(f"  [✓] Candidate PASSED (val_acc={val_acc:.3f})")

            # Track overall best
            if val_acc > overall_best_acc:
                overall_best_acc = val_acc
                overall_best_model = model

        print("\n=== Tier 1 Search Finished ===")
        if not best_models:
            print("[!] No candidate passed threshold, rerun or adjust parameters.")
        else:
            # Pick best of the best
            winner = max(best_models, key=lambda d: d["val_acc"])
            print(f"[MetaRunner] BEST Tier1 Model: bit_length={winner['bit_length']}, num_layers={winner['num_layers']}, "
                  f"mutation={winner['mutation_rate']:.4f}, val_acc={winner['val_acc']:.4f}, test_acc={winner['test_acc']:.4f}")
            # Save champion checkpoint
            winner['model'].save_checkpoint(f"meta_champion_digit{digit}.npz")

if __name__ == "__main__":
    meta_runner()
