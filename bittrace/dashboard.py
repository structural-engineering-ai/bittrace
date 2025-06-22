import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

def dataset_stats(name, X, y, print_table=True):
    labels, counts = np.unique(y, return_counts=True)
    df = pd.DataFrame({'Label': labels, 'Count': counts})
    if print_table:
        print(f"\n{name} Set Distribution:")
        print(df)
    return df

def plot_validation_curve(log_csv, show=True):
    df = pd.read_csv(log_csv)
    plt.figure(figsize=(6,3))
    plt.plot(df['generation'], df['val_accuracy'], label="Validation")
    if 'train_accuracy' in df.columns:
        plt.plot(df['generation'], df['train_accuracy'], label="Train", alpha=0.5)
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    return df

def cluster_summary(model, val_y, output_csv=None, print_table=True):
    try:
        assignments = model.cluster_assignments
        medoids = model.medoids
    except AttributeError:
        print("Cluster assignments/medoids missing on model.")
        return None

    num_clusters = medoids.shape[0]
    summary = []
    for c in range(num_clusters):
        idx = np.where(assignments == c)[0]
        members = len(idx)
        if members == 0:
            purity = 0
            majority_label = None
        else:
            labels = val_y[idx]
            majority_label = np.argmax(np.bincount(labels))
            purity = np.sum(labels == majority_label) / members
        summary.append({
            "Cluster": c,
            "Medoid Index": int(getattr(model, "medoid_indices", [None]*num_clusters)[c]),
            "Num Members": members,
            "Majority Label": int(majority_label) if majority_label is not None else None,
            "Purity": purity,
            "Sample Indices": idx[:5].tolist(),
        })
    df = pd.DataFrame(summary)
    if print_table:
        print("\nCluster Summary Table:")
        print(df)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Cluster table saved to {output_csv}")
    return df

def plot_confusion(y_true, y_pred, title="Confusion Matrix", show=True):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]), ha='center', va='center', color='black', fontsize=8)
    plt.tight_layout()
    if show:
        plt.show()
    return cm

def model_structure(model):
    # Display number of layers, bitstring length, layer plan/map if present
    layers = getattr(model, 'num_layers', None) or getattr(model, 'layers', None)
    bitlen = getattr(model, 'bit_length', None)
    layer_map = getattr(model, 'layer_plan', None)
    print("\nModel Structure:")
    print(f"  Layers: {layers}")
    print(f"  Bitstring Length: {bitlen}")
    if layer_map is not None:
        print("  Layer Plan:")
        print(layer_map)
    else:
        print("  Layer Plan: (not available)")
    return {"layers": layers, "bit_length": bitlen, "layer_map": layer_map}

def accuracy_stats(model, data, labels, name=""):
    # Assumes model has predict method
    try:
        preds = model.predict(data)
        acc = accuracy_score(labels, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        return acc, preds
    except Exception:
        print(f"Cannot compute {name} accuracy (missing predict or data).")
        return None, None

def generate_dashboard(checkpoint_path, train, val, test=None, show_plots=True, output_dir=None):
    """
    train/val/test: tuple (X, y)
    """
    # --- 1. Load Model ---
    print("\n===== BitTrace Experiment Dashboard =====")
    import numpy as np
    data = np.load(checkpoint_path, allow_pickle=True)
    # Recover model state; this will vary depending on your serialization logic
    # Here we assume you have a BitTraceModel.load_checkpoint() or similar
    try:
        from bittrace.model import BitTraceModel
        model = BitTraceModel.load_checkpoint(checkpoint_path)
    except Exception:
        model = None
        print("WARNING: Could not load BitTraceModel object—using checkpoint dict only.")

    # --- 2. Dataset Stats ---
    train_X, train_y = train
    val_X, val_y = val
    dataset_stats("Train", train_X, train_y)
    dataset_stats("Val", val_X, val_y)
    if test is not None:
        test_X, test_y = test
        dataset_stats("Test", test_X, test_y)

    # --- 3. Validation Curve ---
    log_csv = None
    if output_dir:
        log_csv = f"{output_dir}/training_log.csv"
    else:
        # Try to guess from checkpoint location
        log_csv = checkpoint_path.replace(".npz", ".csv")
    try:
        plot_validation_curve(log_csv, show=show_plots)
    except Exception as e:
        print(f"Validation curve failed: {e}")

    # --- 4. Cluster Table ---
    if model is not None:
        cluster_summary(model, val_y, output_csv=(f"{output_dir}/cluster_table.csv" if output_dir else None))

    # --- 5. Model Stats ---
    if model is not None:
        model_structure(model)
    else:
        # Print what we can from checkpoint
        print("Model stats unavailable—raw checkpoint keys:", list(data.keys()))

    # --- 6. Accuracy & Confusion ---
    if model is not None:
        val_acc, val_pred = accuracy_stats(model, val_X, val_y, name="Val")
        if val_pred is not None:
            plot_confusion(val_y, val_pred, title="Val Confusion", show=show_plots)
        if test is not None:
            test_acc, test_pred = accuracy_stats(model, test_X, test_y, name="Test")
            if test_pred is not None:
                plot_confusion(test_y, test_pred, title="Test Confusion", show=show_plots)
        # Optionally: train accuracy/confusion
        train_acc, train_pred = accuracy_stats(model, train_X, train_y, name="Train")
        if train_pred is not None:
            plot_confusion(train_y, train_pred, title="Train Confusion", show=show_plots)

    print("\n===== BitTrace Dashboard Complete =====")

# END of bittrace/dashboard.py
