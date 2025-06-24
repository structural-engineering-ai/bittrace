# meta_runner.py (Tiered Evolution Logic Embedded)

import os
import random
import numpy as np
import config
from bittrace import data_pipeline as dp
from bittrace.plan_evolver import BitTracePlanEvolver


def run_meta_batch(meta_cfg, X_orig, y):
    candidate_widths = meta_cfg["candidate_widths"]
    layer_min, layer_max = meta_cfg["layer_range"]
    meta_batch_size = meta_cfg["meta_batch_size"]

    batch_candidates = []
    attempted = set()

    for _ in range(meta_batch_size):
        while True:
            bit_length = random.choice(candidate_widths)
            num_layers = random.randint(layer_min, layer_max)
            key = (bit_length, num_layers)
            if key not in attempted:
                attempted.add(key)
                batch_candidates.append(key)
                break

    batch_results = []
    for bit_length, num_layers in batch_candidates:
        packed_X = dp.pack_images(
            X_orig,
            bit_length=bit_length,
            use_random_offset=True
        )
        plan_config = dict(meta_cfg)
        plan_config.update({
            "bit_length": bit_length,
            "num_layers": num_layers,
            "pop_size": meta_cfg["inner_pop_size"],
            "generations": meta_cfg["inner_generations"]
        })
        evolver = BitTracePlanEvolver(X=packed_X, y=y, config=plan_config)
        best, logs = evolver.run(verbose=False)

        batch_results.append({
            "bit_length": bit_length,
            "num_layers": num_layers,
            "acc": best["acc"],
            "generation": best["gen"],
            "plan": best["plan"],
            "logs": logs
        })

    return batch_results


def tiered_evolution():
    cfg = config.config
    val_threshold = cfg["val_threshold"]
    meta_ckpt_path = os.path.join(cfg["meta_checkpoint_dir"], "CURRENT_BEST.npz")
    os.makedirs(cfg["meta_checkpoint_dir"], exist_ok=True)

    (X_orig, y), _ = dp.prepare_stratified_train_val(
        base_train_folder=cfg["train_data_folder"],
        total_samples=cfg["total_samples"],
        split_ratios=cfg["split_ratios"],
        cache_folder=cfg["cache_folder"],
        random_seed=cfg["random_seed"],
        use_cache=cfg["use_cache"],
        bit_length=None,
        use_random_offset=True
    )

    search_round = 0
    while True:
        print(f"\n[Meta] Batch {search_round}")
        batch_results = run_meta_batch(cfg, X_orig, y)
        survivors = [res for res in batch_results if res["acc"] >= 0.15]

        if not survivors:
            print("No survivors over threshold. Big bang - restarting search...")
            search_round += 1
            continue

        survivors.sort(key=lambda r: r["acc"], reverse=True)
        best = survivors[0]

        print(f"\nTier 1 survivor: {best['bit_length']}b/{best['num_layers']}l with acc={best['acc']:.4f}")

        # Tier 2 Deep Evolution
        packed_X = dp.pack_images(X_orig, bit_length=best["bit_length"], use_random_offset=True)
        plan_config = dict(cfg)
        plan_config.update({
            "bit_length": best["bit_length"],
            "num_layers": best["num_layers"],
            "pop_size": cfg["population_size"],
            "generations": cfg["num_generations"]
        })

        deep_evolver = BitTracePlanEvolver(X=packed_X, y=y, config=plan_config)
        final, logs = deep_evolver.run(verbose=True)

        if final["acc"] >= val_threshold:
            print(f"\n=== SUCCESS: Survivor reached {final['acc']:.4f} ===")
            np.savez_compressed(
                meta_ckpt_path,
                best_plan=final["plan"],
                best_acc=final["acc"],
                generation=final["gen"],
                logs=logs,
                bit_length=plan_config["bit_length"],
                num_layers=plan_config["num_layers"]
            )
            break
        else:
            print(f"Survivor failed deep evolution. Restarting meta search...")
            search_round += 1


if __name__ == "__main__":
    print("=" * 60)
    print("  BitTrace Meta-Runner (Tiered Evolution Mode)")
    print("=" * 60)
    tiered_evolution()
