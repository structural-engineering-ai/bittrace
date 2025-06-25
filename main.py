import numpy as np
from config import BITTRACE_CONFIG
from bittrace.data_loader import load_bittrace_digit
from bittrace.model import BitTraceModel

def main():
    cfg = BITTRACE_CONFIG
    digit = cfg["included_labels"][0]  # single digit run

    # Load Data
    print(f"Loading training/validation data for digit={digit} from: {cfg['bitblock_sets_dir']}")
    X_train, y_train = load_bittrace_digit(digit, split="train", base_dir=cfg["bitblock_sets_dir"])
    X_val, y_val     = load_bittrace_digit(digit, split="val", base_dir=cfg["bitblock_sets_dir"])
    X_test, y_test   = load_bittrace_digit(digit, split="test", base_dir=cfg["bitblock_sets_dir"])
    print(f"Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # Initialize Model
    model = BitTraceModel(
        bit_length=cfg["bit_length"],
        num_layers=cfg["num_layers"],
        pop_size=cfg.get("pop_size", 32),
        mutation_rate=cfg["mutation_rate"],
        use_gpu=True,
    )

    # Train / Evolve
    print(f"\n[BitTrace] Training population of {cfg.get('pop_size', 32)} for {cfg.get('generations', 100)} generations")
    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        generations=cfg.get("generations", 100),
        elite_frac=cfg.get("elite_frac", 0.2),
    )

    # Validation
    best_val_acc = model.evaluate_accuracy(X_val, y_val)
    print(f"[BitTrace] Final Validation Accuracy: {best_val_acc:.4f}")

    # Test
    best_test_acc = model.evaluate_accuracy(X_test, y_test)
    print(f"[BitTrace] Final Test Accuracy: {best_test_acc:.4f}")

    # Save Best Model
    out_path = f"bittrace_digit{digit}_champion.npz"
    model.save_checkpoint(out_path)
    print(f"[BitTrace] Champion checkpoint saved to: {out_path}")

if __name__ == "__main__":
    main()
