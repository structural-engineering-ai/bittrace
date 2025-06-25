# bittrace/plan_evolver.py

import numpy as np
from copy import deepcopy
from bittrace.model import BitTraceModel
from bittrace import data_loader as dp


class BitTracePlanEvolver:
    def __init__(self, X, y, config, allowed_ops=None):
        self.X = X
        self.y = y
        self.config = config
        self.generations = config.get("generations", 10)
        self.pop_size = config.get("pop_size", 16)
        self.bit_length = config["bit_length"]
        self.num_layers = config["num_layers"]
        self.allowed_ops = allowed_ops or config.get("allowed_ops", ["AND", "OR", "XOR", "NAND", "NOT"])

        # Pass required args to BitTraceModel
        self.model = BitTraceModel(
            bit_length=self.bit_length,
            num_layers=self.num_layers,
            pop_size=self.pop_size,
        )

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
            best_gen = {
                "acc": -1,
                "plan": None,
                "residues": None,
                "y_pred": None,
                "entropy": None
            }

            # Generate new random population
            candidates = [
                self.model._generate_random_layer_plan(self.num_layers, self.model.bytes_per_individual)
                for _ in range(self.pop_size)
            ]

            for plan in candidates:
                try:
                    residues = self.model.run_batch(self.X, plan)
                    acc, y_pred = self.model.evaluate_supervised_accuracy(residues, self.y, return_pred=True)
                    
                    try:
                        entropy = self.model.bit_entropy(residues)
                        print(f"[Entropy Debug] type={type(entropy)}, value={entropy}")
                    except Exception as e:
                        print(f"[DEBUG] Entropy error: {e}")
                        entropy = None

                    if acc > best_gen["acc"]:
                        best_gen.update({
                            "acc": acc,
                            "plan": deepcopy(plan),
                            "residues": residues.copy(),
                            "y_pred": y_pred.copy(),
                            "entropy": entropy,
                        })
                except Exception:
                    continue  # skip broken plans

            self.logs["accuracy_log"].append(best_gen["acc"])
            self.logs["entropy_log"].append(best_gen["entropy"] if best_gen["residues"] is not None else np.nan)
            self.logs["layer_plan_log"].append(deepcopy(best_gen["plan"]))

            if verbose:
                ent_str = f"{best_gen['entropy']:.4f}" if best_gen["entropy"] is not None else "None"
                print(f"Generation {gen+1}/{self.generations}: Accuracy={best_gen['acc']:.4f}, Entropy={ent_str}")

            if best_gen["acc"] > self.best["acc"]:
                self.best.update({
                    "plan": deepcopy(best_gen["plan"]),
                    "acc": best_gen["acc"],
                    "residues": best_gen["residues"].copy(),
                    "y_pred": best_gen["y_pred"].copy(),
                    "gen": gen + 1,
                })
                if verbose:
                    print(f"[*] New best accuracy: {best_gen['acc']:.4f} at generation {gen+1}")

        # If no valid plans were ever found (acc <= 0), return what we did get
        if self.best["acc"] <= 0:
            try:
                fallback_plan = self.model._generate_random_layer_plan(self.num_layers, self.model.bytes_per_individual)
                fallback_residues = self.model.run_batch(self.X, fallback_plan)
                acc, _ = self.model.evaluate_supervised_accuracy(fallback_residues, self.y, return_pred=False)
                self.best["acc"] = acc
                self.best["plan"] = fallback_plan
                self.best["residues"] = fallback_residues
                self.best["gen"] = -1
            except:
                pass  # If fallback also fails, retain -1.0

        return self.best, self.logs
