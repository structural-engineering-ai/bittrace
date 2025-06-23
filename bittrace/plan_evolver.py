# bittrace/plan_evolver.py

import numpy as np
from copy import deepcopy
from bittrace.model import BitTraceModel  # Assumes your notebook model is now here
from bittrace import data_pipeline as dp

class BitTracePlanEvolver:
    def __init__(self, X, y, config, allowed_ops=None):
        self.X = X
        self.y = y
        self.config = config
        self.generations = config.get("generations", 10)
        self.pop_size = config.get("pop_size", 16)
        self.allowed_ops = allowed_ops or config.get("allowed_ops", ["AND", "OR", "XOR", "NAND", "NOT"])
        self.bit_length = config["bit_length"]
        self.num_layers = config["num_layers"]
        self.model = BitTraceModel(config)
        self.logs = {
            "accuracy_log": [],
            "entropy_log": [],
            "layer_plan_log": [],
        }
        self.best = {
            "plan": None,
            "acc": -1,
            "residues": None,
            "y_pred": None,
            "gen": -1,
        }

    def run(self, verbose=False):
        for gen in range(self.generations):
            best_gen_acc = -1
            best_gen_plan = None
            best_gen_residues = None
            best_gen_y_pred = None
            best_gen_entropy = None

            # Population: pop_size random plans
            candidates = [self.model._random_layer_plan() for _ in range(self.pop_size)]
            for plan in candidates:
                try:
                    residues = self.model.run_batch(self.X, plan)
                    acc, y_pred = self.model.evaluate_supervised_accuracy(residues, self.y, return_pred=True)
                    entropy = self.model.bit_entropy(residues)
                    if acc > best_gen_acc:
                        best_gen_acc = acc
                        best_gen_plan = deepcopy(plan)
                        best_gen_residues = residues.copy()
                        best_gen_y_pred = y_pred.copy()
                        best_gen_entropy = entropy
                except Exception:
                    pass

            self.logs["accuracy_log"].append(best_gen_acc)
            self.logs["entropy_log"].append(best_gen_entropy if best_gen_residues is not None else np.nan)
            self.logs["layer_plan_log"].append(deepcopy(best_gen_plan))

            if verbose:
                ent_str = f"{best_gen_entropy:.4f}" if best_gen_entropy is not None else "None"
                print(f"Generation {gen+1}/{self.generations}: Accuracy={best_gen_acc:.4f}, Entropy={ent_str}")

            # Track best
            if best_gen_acc > self.best["acc"]:
                self.best = {
                    "plan": deepcopy(best_gen_plan),
                    "acc": best_gen_acc,
                    "residues": best_gen_residues.copy(),
                    "y_pred": best_gen_y_pred.copy(),
                    "gen": gen + 1,
                }
                if verbose:
                    print(f"[*] New best accuracy: {best_gen_acc:.4f} at generation {gen+1}")

        return self.best, self.logs
