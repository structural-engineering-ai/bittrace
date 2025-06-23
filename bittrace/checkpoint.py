import os
import time
import numpy as np

class ModelCheckpoint:
    def __init__(self, directory, model_name="bittrace_model"):
        self.directory = directory
        self.model_name = model_name
        os.makedirs(directory, exist_ok=True)

    def save(self, model, generation=None, extra_info=None):
        """
        Save full model state to disk. Always includes a timestamp and model_name for tracking.
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # gen_part = f"_gen{generation}" if generation is not None else ""
        # fname = f"{self.model_name}{gen_part}_{timestamp}.npz"
        # path = os.path.join(self.directory, fname)
        path = os.path.join(self.directory, "CURRENT_BEST.npz")

        state = {
            "population": model.population,
            "medoids_idx": model.medoids_idx,
            "num_clusters": model.num_clusters,
            "model_name": getattr(model, "name", self.model_name),
            "model_config": getattr(model, "config", {}),
            "timestamp": timestamp
        }
        # Optionally save structural params
        for attr in ["bit_length", "num_layers"]:
            if hasattr(model, attr):
                state[attr] = getattr(model, attr)
        # Add any extras
        if extra_info is not None:
            if isinstance(extra_info, dict):
                state.update(extra_info)
            else:
                raise ValueError("extra_info must be a dict if provided")
        np.savez_compressed(path, **state)
        # print(f"ModelCheckpoint: Saved model to {path}")
        return path

    @staticmethod
    def load(path):
        """
        Loads model state and metadata from disk.
        Returns: dict containing everything stored.
        """
        data = np.load(path, allow_pickle=True)
        # For backwards compatibility, coerce to dict if necessary
        if hasattr(data, "items"):
            return dict(data.items())
        else:
            return data
