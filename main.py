import argparse
import os
import numpy as np
import config

from bittrace.model import BitTraceModel
from bittrace.trainer import train_bittrace_full
from bittrace.checkpoint import ModelCheckpoint
from bittrace import data_pipeline as dp
from bittrace import dashboard

def main(show_plots=True):
    print("="*60)
    print("  BitTrace — Bitwise Evolutionary Training")
    print("="*60)

    bit_length = config.config.get("bit_length", 1024)
    num_layers = config.config.get("num_layers", 32)

    use_random_offset = config.config.get("use_random_offset", False)

    # --- 1. Load training data ---
    print("Loading train and val splits from cache...")
    (train_X, train_y), (val_X, val_y) = dp.prepare_stratified_train_val(
        base_train_folder=config.config["train_data_folder"],
        total_samples=config.config["total_samples"],
        split_ratios=config.config["split_ratios"],
        cache_folder=config.config["cache_folder"],
        random_seed=config.config["random_seed"],
        use_cache=config.config["use_cache"],
        bit_length=bit_length,
        use_random_offset=use_random_offset
    )

    # --- 2. Build Model ---
    model = BitTraceModel(
        config=dict(config.config, bit_length=bit_length, num_layers=num_layers)
    )

    # Initialize population and medoids for the model
    model.population = train_X.copy()
    model.medoids_idx = np.arange(min(model.num_clusters, len(model.population)))
    model.medoids = model.population[model.medoids_idx]

    checkpointer = ModelCheckpoint(
        config.config["checkpoint_dir"],
        model_name="bittrace_model"
    )

    # --- 3. Train ---
    trained_model, final_ckpt_path = train_bittrace_full(
        model,
        num_generations=config.config["num_generations"],
        mutation_rate=config.config["mutation_rate"],
        checkpoint_every=config.config["checkpoint_every"],
        checkpoint_dir=config.config["checkpoint_dir"],
        val_data=(val_X, val_y),
        early_stopping_patience=config.config.get("early_stopping_patience", 10),
        resume_from_checkpoint=config.config.get("resume_from_checkpoint", None),
        log_csv_path=config.config.get("log_csv_path", None),
        run_name="bittrace_model"
    )

    checkpointer.save(trained_model, generation="final_model")

    dashboard.generate_dashboard(
        final_ckpt_path,
        train=(train_X, train_y),
        val=(val_X, val_y),
        test=None,
        show_plots=show_plots,
        output_dir=config.config["checkpoint_dir"]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true", help="Disable dashboard plots")
    args = parser.parse_args()
    main(show_plots=not args.no_plots)
