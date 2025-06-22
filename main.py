import os
from bittrace.model import BitTraceModel
from bittrace.trainer import train_bittrace_full
import bittrace.data_pipeline as dp
from bittrace.visualizer import plot_accuracy, plot_umap, extract_embeddings
from bittrace.config import config

def main():
    print("=" * 60)
    print("  BitTrace — Bitwise Evolutionary Training")
    print("=" * 60)

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    log_csv = os.path.join(config["checkpoint_dir"], "training_log.csv")

    # Prepare data (train/val)
    (train_X, train_y), (val_X, val_y) = dp.prepare_stratified_train_val(
        base_train_folder=config["train_data_folder"],
        total_samples=config["total_samples"],
        split_ratios=config["split_ratios"],
        cache_folder=config["cache_folder"],
        random_seed=config["random_seed"],
        use_cache=config["use_cache"],
    )

    # Initialize population from training data (bit-packed)
    population_init = train_X.copy()
    model = BitTraceModel(
        population_init,
        num_clusters=config["num_clusters"]
    )

    # Train the population-based model
    trained_model = train_bittrace_full(
        model,
        num_generations=config["num_generations"],
        mutation_rate=config["mutation_rate"],
        checkpoint_every=config["checkpoint_every"],
        checkpoint_dir=config["checkpoint_dir"],
        val_data=(val_X, val_y),
        early_stopping_patience=config.get("early_stopping_patience", 10),
        log_csv_path=log_csv,
    )

    # Post-training analysis
    plot_accuracy(log_csv)
    embeddings = extract_embeddings(trained_model)
    plot_umap(embeddings, val_y)

    print("BitTrace training and analysis complete.")

if __name__ == "__main__":
    main()
