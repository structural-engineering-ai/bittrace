# meta_runner.py

import os
import random
import numpy as np
import config
from bittrace import data_pipeline as dp
from bittrace.plan_evolver import BitTracePlanEvolver

def main():
    meta_cfg = config.config
    candidate_widths = meta_cfg["candidate_widths"]
    layer_min, layer_max = meta_cfg["layer_range"]
    meta_batch_size = meta_cfg["meta_batch_size"]
    val_threshold = meta_cfg["val_threshold"]
    checkpoint_dir = meta_cfg["meta_checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, "CURRENT_BEST.npz")

    attempted = set()
    best_val_acc = -np.inf
    search_round = 0

    losers_log = []
    winners_log = []

    # Load train/val once with no fixed bit length (will repack per candidate)
    (X_orig, y), _ = dp.prepare_stratified_train_val(
        base_train_folder=meta_cfg["train_data_folder"],
        total_samples=meta_cfg["total_samples"],
        split_ratios=meta_cfg["split_ratios"],
        cache_folder=meta_cfg["cache_folder"],
        random_seed=meta_cfg["random_seed"],
        use_cache=meta_cfg["use_cache"],
        bit_length=None,
        use_random_offset=True
    )

    while True:
        batch_candidates = []
        for _ in range(meta_batch_size):
            tries = 0
            while True:
                bit_length = random.choice(candidate_widths)
                num_layers = random.randint(layer_min, layer_max)
                key = (bit_length, num_layers)
                if key not in attempted:
                    attempted.add(key)
                    break
                tries += 1
                if tries > 20:
                    print("No more unique candidates; meta-search exhausted.")
                    # Print summary before exit
                    print("\nAll winners:")
                    print(f"{'Bits':<6} {'Layers':<8} {'ValAcc':<8} {'Gen':<4} {'Plan':<30}")
                    for row in winners_log:
                        plan_str = str(row[4])[:30]
                        print(f"{row[0]:<6} {row[1]:<8} {row[2]:<8.4f} {row[3]:<4} {plan_str:<30}")
                    print("\nAll losers:")
                    print(f"{'Bits':<6} {'Layers':<8} {'ValAcc':<8} {'Gen':<4} {'Plan':<30}")
                    for row in losers_log:
                        plan_str = str(row[4])[:30]
                        print(f"{row[0]:<6} {row[1]:<8} {row[2]:<8.4f} {row[3]:<4} {plan_str:<30}")
                    return
            batch_candidates.append({
                "bit_length": bit_length,
                "num_layers": num_layers,
            })

        print(f"\n[Meta] Batch {search_round} | Candidates: " +
              ", ".join([f"{c['bit_length']}b/{c['num_layers']}l" for c in batch_candidates]))

        batch_results = []
        for cand in batch_candidates:
            # Repack original X for this candidate's bit length
            packed_X = dp.pack_images(
                X_orig,
                bit_length=cand["bit_length"],
                use_random_offset=True
            )

            plan_config = dict(meta_cfg)
            plan_config.update({
                "bit_length": cand["bit_length"],
                "num_layers": cand["num_layers"]
            })

            evolver = BitTracePlanEvolver(
                X=packed_X,
                y=y,
                config=plan_config
            )
            best, logs = evolver.run(verbose=False)

            batch_results.append({
                "bit_length": cand["bit_length"],
                "num_layers": cand["num_layers"],
                "best_acc": best["acc"],
                "generation": best["gen"],
                "best_plan": best["plan"],
                "logs": logs,
                "ckpt_path": ""
            })

        batch_results = sorted(batch_results, key=lambda r: r["best_acc"], reverse=True)
        best_in_batch = batch_results[0]

        print(f"\n{'Candidate':<16} {'Layers':<8} {'BestAcc':<10}")
        print('-' * 40)
        for result in batch_results:
            print(f"{result['bit_length']}b/{result['num_layers']}l".ljust(16),
                  f"{result['num_layers']:<8}", f"{result['best_acc']:<10.4f}")
        print('-' * 40)
        print(f"*** Winner: {best_in_batch['bit_length']}b/{best_in_batch['num_layers']}l | val_acc={best_in_batch['best_acc']:.4f} ***")

        for loser in batch_results[1:]:
            losers_log.append([
                loser["bit_length"], loser["num_layers"], loser["best_acc"],
                loser["generation"], loser["best_plan"], ""
            ])
        winners_log.append([
            best_in_batch["bit_length"], best_in_batch["num_layers"], best_in_batch["best_acc"],
            best_in_batch["generation"], best_in_batch["best_plan"], best_ckpt_path
        ])

        if best_in_batch["best_acc"] > best_val_acc:
            best_val_acc = best_in_batch["best_acc"]
            print(f"NEW META-BEST: {best_val_acc:.4f} ({best_in_batch['bit_length']}b/{best_in_batch['num_layers']}l)")
            np.savez_compressed(
                best_ckpt_path,
                best_plan=best_in_batch["best_plan"],
                best_acc=best_in_batch["best_acc"],
                generation=best_in_batch["generation"],
                logs=best_in_batch["logs"],
                bit_length=best_in_batch["bit_length"],
                num_layers=best_in_batch["num_layers"]
            )

        if best_in_batch["best_acc"] >= val_threshold:
            print(f"\n=== Survivor found! {best_in_batch['bit_length']}b/{best_in_batch['num_layers']}l reached threshold {val_threshold:.2f} ===")
            break

        search_round += 1

if __name__ == "__main__":
    print("="*60)
    print("  BitTrace Meta-Runner")
    print("="*60)
    main()
