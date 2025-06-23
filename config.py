# config.py

config = {
    # Data folders
    "train_data_folder": "./data/training",
    "test_data_folder": "./data/testing",
    "checkpoint_dir": "./checkpoints",

    # Dataset splitting
    "total_samples": "ALL",      # 'ALL' or integer ≤ available train samples
    "split_ratios": {            # Sum to 1.0, test usually 0 here (test loaded separately)
        "train": 0.8,
        "val": 0.2,
        "test": 0.0,
    },
    "random_seed": 42,
    "use_cache": True,
    "cache_folder": "./data/prepacked_cache",

    # Model and training parameters
    "population_size": 500,
    "num_layers": 32,
    "bit_length": 1024,
    "num_clusters": 10,
    "num_generations": 500,
    "mutation_rate": 0.02,
    "checkpoint_every": 0,

    # Early stopping
    "early_stopping_patience": 50,

    # Architecture meta-search params
    "candidate_widths": [1024, 2048, 4096],        # Bit widths to try
    "layer_range": (10, 256),                       # Min/max layers
    "meta_batch_size": 10,                         # Candidates per batch
    "val_threshold": 0.90,                         # Survival threshold
    "meta_early_stopping": 5,                      # Generations with no improvement
    "fail_log_csv": "failures.csv",                # Extinction log
    "besties_log_csv": "besties.csv",              # Winner log
    "meta_checkpoint_dir": "meta_checkpoints",      # Where to save best models
}
