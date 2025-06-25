import numpy as np

BITTRACE_CONFIG = {
    "image_shape": (28, 28),
    "bit_length": 1024,
    "num_layers": 8,
    "mutation_rate": 0.02,
    "noise_rate": 0.1,
    "use_random_offset": True,
    "train_val_test_split": (0.7, 0.2, 0.1),
    "included_labels": [0],      # Single digit for binary classifier; [0,1,...] for ensemble
    "random_seed": 42,
    "pop_size": 128,              # Evolutionary population size (for main loop)
    "generations": 1000,         # Number of generations for Tier 2 search

    # --- Path definitions ---
    "bitblock_sets_dir": "./bitblock_sets",      # Per-digit binary data
    "ensemble_train_X": "./bitblock_sets/ensemble_train_X.npz",
    "ensemble_train_y": "./bitblock_sets/ensemble_train_y.npz",
    "ensemble_val_X": "./bitblock_sets/ensemble_val_X.npz",
    "ensemble_val_y": "./bitblock_sets/ensemble_val_y.npz",
    "ensemble_test_X": "./bitblock_sets/ensemble_test_X.npz",
    "ensemble_test_y": "./bitblock_sets/ensemble_test_y.npz",
}

META_CONFIG = {
    "image_shape": BITTRACE_CONFIG["image_shape"],
    "bit_length_range": (1028, 2048),
    "num_layers_range": (32, 128),        # will enforce divisibility by 8
    "mutation_rate_range": (0.01, 0.2),
    "search_trials": 32,                # How many meta-architectures to try (Tier 1)
    "meta_generations": 10,             # Generations per meta architecture (Tier 2)
    "meta_pop_size": 32,                # Population for meta evolution (if different from base)
    "included_labels": [0], # All digits for meta search / bitblocks
    "bitblock_sets_dir": BITTRACE_CONFIG["bitblock_sets_dir"],
    "random_seed": 42,
    # Optionally add:
    "layer_divisor": 8,                 # divisor for num_layers in sampling
}

def sample_meta_config():
    rng = np.random.default_rng(META_CONFIG["random_seed"])
    bit_length = int(rng.integers(*META_CONFIG["bit_length_range"]).item())  # <-- FIXED
    min_bits = np.prod(META_CONFIG["image_shape"])
    if bit_length < min_bits:
        bit_length = min_bits

    layer_min, layer_max = META_CONFIG["num_layers_range"]
    divisor = META_CONFIG.get("layer_divisor", 8)
    possible_layers = [n for n in range(layer_min, layer_max + 1) if n % divisor == 0]
    if not possible_layers:
        raise ValueError(f"No num_layers divisible by {divisor} in range!")
    num_layers = int(rng.choice(possible_layers))

    mutation_rate = float(rng.uniform(*META_CONFIG["mutation_rate_range"]))
    return {
        "image_shape": META_CONFIG["image_shape"],
        "bit_length": bit_length,
        "num_layers": num_layers,
        "mutation_rate": mutation_rate,
        "noise_rate": BITTRACE_CONFIG["noise_rate"],
        "use_random_offset": BITTRACE_CONFIG["use_random_offset"],
        "included_labels": META_CONFIG["included_labels"],
        "bitblock_sets_dir": META_CONFIG["bitblock_sets_dir"],
        "random_seed": META_CONFIG["random_seed"],
        "pop_size": META_CONFIG.get("meta_pop_size", 16),
        "generations": META_CONFIG.get("meta_generations", 50),
    }
