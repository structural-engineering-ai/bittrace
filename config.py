# config.py

config = {
    # === Data Paths ===
    "train_data_folder": "./data/training",            # Folder containing training set
    "test_data_folder": "./data/testing",              # Folder containing test set
    "cache_folder": "./data/prepacked_cache",          # Location for cached packed bit arrays
    "checkpoint_dir": "./checkpoints",                 # Checkpoints for main model
    "meta_checkpoint_dir": "meta_checkpoints",         # Checkpoints for meta-runner survivors

    # === Data Loading & Splitting ===
    "total_samples": "ALL",                            # Use "ALL" or integer to subsample training data
    "split_ratios": {                                  # Data split ratios (must sum to 1.0)
        "train": 0.8,
        "val": 0.2,
        "test": 0.0,                                    # Typically 0 if test loaded separately
    },
    "random_seed": 42,                                 # For reproducibility
    "use_cache": True,                                 # Cache packed bit arrays to speed up loading

    # === BitTrace Model Parameters ===
    "bit_length": 1024,                                # Bit width of the model input
    "num_layers": 32,                                  # Number of layers in the bitblock pipeline
    "population_size": 500,                            # Individuals per generation in evolution
    "num_generations": 500,                            # Total training generations
    "mutation_rate": 0.02,                             # Bitwise mutation probability
    "num_clusters": 10,                                # Clusters used for classification
    "checkpoint_every": 0,                             # Set to >0 to save periodic checkpoints

    # === Training Behavior ===
    "early_stopping_patience": 50,                     # Generations to wait before halting on no improvement

    # === Meta-Architecture Search (Meta Runner) ===
    "inner_generations": 10,                           # Inner training cycles for each candidate
    "inner_pop_size": 8,                               # Population size for inner evolution
    "candidate_widths": [1024, 2048, 4096],            # Bit lengths to explore
    "layer_range": (10, 256),                          # Min/max layer depths to try
    "meta_batch_size": 10,                             # Candidates evaluated per batch
    "val_threshold": 0.90,                             # Minimum accuracy to declare survivor
    "meta_early_stopping": 5,                          # Early stop meta search after no improvement

    # === Logging (CSV) ===
    "fail_log_csv": "failures.csv",                    # Log of candidates that failed to reach threshold
    "besties_log_csv": "besties.csv",                  # Log of all successful survivors
}
