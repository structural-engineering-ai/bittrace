# bittrace/dashboard.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linear_sum_assignment
from bittrace.checkpoint import ModelCheckpoint

def dataset_stats(name, X, y, print_table=True):
    labels, counts = np.unique(y, return_counts=True)
    df = pd.DataFrame({'Label': labels, 'Count': counts})
    if print_table:
        print(f"\n{name} Set Distribution:")
        print(df)
    return df

def plot_validation_curve(log_csv, show=True):
    df = pd.read_csv(log_csv)
    # Accept both 'generation' and 'Generation'
    gen_col = "generation" if "generation" in df.columns else "Generation"
    acc_col = "val_accuracy" if "val_accuracy" in df.columns else "ValAccuracy"
    plt.figure(figsize=(6,3))
    plt.plot(df[gen_col], df[acc_col], label="Validation")
    if 'train_accuracy' in df.columns:
        plt.plot(df[gen_col], df['train_accuracy'], label="Train", alpha=0.5)
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    return df

def compute_hungarian_mapping(y_true, y_pred, n_clusters, n_labels):
    cost_matrix = np.zeros((n_clusters, n_labels), dtype=int)
    for c in range(n_clusters):
        for l in range(n_labels):
            cost_matrix[c, l] = -np.sum((y_pred == c) & (y_true == l))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {c: l for c, l in zip(row_ind, col_ind)}
    return mapping

def cluster_summary(model, X, y, print_table=True, output_csv=None):
    assignments = model.predict(X)
    n_clusters = len(np.unique(assignments))
    label_set = np.unique(y)
    min_label = label_set.min()
    summary = []
    for c in range(n_clusters):
        idx = np.where(assignments == c)[0]
        members = len(idx)
        if members == 0:
            majority_label = None
            purity = 0.0
            label_counts = {}
        else:
            labels = y[idx]
            bincounts = np.bincount(labels - min_label, minlength=label_set.max() - min_label + 1)
            majority_label = bincounts.argmax() + min_label
            purity = bincounts[majority_label - min_label] / members
            label_counts = {int(lbl + min_label): int(cnt) for lbl, cnt in enumerate(bincounts) if cnt > 0}
        summary.append({
            "Cluster": c,
            "Num Members": members,
            "Majority Label": int(majority_label) if majority_label is not None else None,
            "Purity": purity,
            "Label Counts": label_counts,
            "Sample Indices": idx[:5].tolist()
        })
    df = pd.DataFrame(summary)
    if print_table:
        print("\nCluster Summary Table:")
        print(df)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Cluster table saved to {output_csv}")
    return df, assignments

def print_cluster_mapping(y_true, y_pred, n_clusters, n_labels):
    label_map = np.zeros(n_clusters, dtype=int)
    for c in range(n_clusters):
        mask = (y_pred == c)
        if np.sum(mask) > 0:
            label_map[c] = np.bincount(y_true[mask], minlength=n_labels).argmax()
        else:
            label_map[c] = -1
    print("\nMajority-vote cluster-to-label mapping:")
    for c in range(n_clusters):
        print(f"  Cluster {c}: {label_map[c]}")
    return label_map

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
    print(f"\nModel Name: {getattr(model, 'name', 'unnamed')}")
    print(f"Config: {getattr(model, 'config', {})}")
    print(f"  Medoids shape: {getattr(model, 'medoids', np.array([])).shape}")
    print(f"  Population shape: {getattr(model, 'population', np.array([])).shape}")
    print(f"  Number of clusters: {getattr(model, 'num_clusters', None)}")

def accuracy_stats(model, X, y, label_map=None, name=""):
    preds = model.predict(X, label_map=label_map)
    acc = accuracy_score(y, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    return acc, preds

def generate_dashboard(checkpoint_path, train, val, test=None, show_plots=True, output_dir=None):
    print("\n===== BitTrace Experiment Dashboard =====")
    state = ModelCheckpoint.load(checkpoint_path)
    from bittrace.model import BitTraceModel
    model = BitTraceModel(
        population_init=state["population"],
        num_clusters=int(state["num_clusters"]),
        device=True
    )
    model.medoids_idx = state["medoids_idx"]
    model.medoids = model.population[model.medoids_idx]
    model.name = state.get("model_name", "unnamed")
    model.config = state.get("model_config", {})

    train_X, train_y = train
    val_X, val_y = val
    test_X, test_y = test if test is not None else (None, None)

    dataset_stats("Train", train_X, train_y)
    dataset_stats("Val", val_X, val_y)
    if test is not None and test_X is not None:
        dataset_stats("Test", test_X, test_y)

    # Validation curve
    log_csv = f"{output_dir}/training_log.csv" if output_dir else checkpoint_path.replace(".npz", ".csv")
    try:
        plot_validation_curve(log_csv, show=show_plots)
    except Exception as e:
        print(f"Validation curve failed: {e}")

    print("\n== On Validation Set ==")
    cluster_df, val_assignments = cluster_summary(model, val_X, val_y, output_csv=(f"{output_dir}/cluster_table.csv" if output_dir else None))

    n_clusters = model.num_clusters
    n_labels = len(np.unique(val_y))
    label_map = print_cluster_mapping(val_y, val_assignments, n_clusters, n_labels)

    mapping = compute_hungarian_mapping(val_y, val_assignments, n_clusters, n_labels)
    hungarian_label_map = np.array([mapping.get(c, -1) for c in range(n_clusters)])
    print("\nHungarian-matched cluster-to-label mapping:")
    for c in range(n_clusters):
        print(f"  Cluster {c}: {hungarian_label_map[c]}")

    print("\nMajority-vote mapped accuracy/confusion:")
    acc_maj, pred_maj = accuracy_stats(model, val_X, val_y, label_map=label_map, name="Val (majority-mapped)")
    plot_confusion(val_y, pred_maj, title="Val Confusion (majority-mapped)", show=show_plots)

    print("\nHungarian-matched mapped accuracy/confusion:")
    acc_hung, pred_hung = accuracy_stats(model, val_X, val_y, label_map=hungarian_label_map, name="Val (Hungarian-mapped)")
    plot_confusion(val_y, pred_hung, title="Val Confusion (Hungarian-mapped)", show=show_plots)

    print("\nRaw cluster assignment accuracy/confusion (for reference):")
    raw_acc = accuracy_score(val_y, val_assignments)
    print(f"Raw cluster assignment accuracy: {raw_acc:.4f}")
    plot_confusion(val_y, val_assignments, title="Val Confusion (raw cluster)", show=show_plots)

    model_structure(model)
    print("\n===== BitTrace Dashboard Complete =====")
