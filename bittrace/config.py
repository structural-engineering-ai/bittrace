# config.py

config = {
    # Data folders
    "train_data_folder": "./data/training",
    "test_data_folder": "./data/testing",

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
    "population_size": 1000,
    "bit_length_bytes": 64,
    "num_clusters": 10,
    "num_generations": 100,
    "mutation_rate": 0.01,
    "checkpoint_every": 10,
    "checkpoint_dir": "checkpoints",

    # Early stopping
    "early_stopping_patience": 10,
}
