import os
from bittrace.model import BitTraceModel
from bittrace.trainer import train_bittrace_full
from bittrace import data_pipeline as dp
from bittrace import config


def test_small_training_run():
    os.makedirs(config.config["checkpoint_dir"], exist_ok=True)
    (train_X, train_y), (val_X, val_y) = dp.prepare_stratified_train_val(
        base_train_folder=config.config["train_data_folder"],
        total_samples=100,  # small subset for quick test
        split_ratios={"train": 0.8, "val": 0.2, "test": 0.0},
        cache_folder=config.config["cache_folder"],
        random_seed=config.config["random_seed"],
        use_cache=True,
    )

    population_init = train_X.copy()
    model = BitTraceModel(population_init, num_clusters=config.config["num_clusters"])

    trained_model = train_bittrace_full(
        model,
        num_generations=5,  # short test run
        mutation_rate=config.config["mutation_rate"],
        checkpoint_every=2,
        checkpoint_dir=config.config["checkpoint_dir"],
        val_data=(val_X, val_y),
        early_stopping_patience=3,
    )

    # Basic assert: final population shape unchanged
    assert trained_model.population.shape == population_init.shape

if __name__ == "__main__":
    test_small_training_run()
    print("Training integration test passed.")
