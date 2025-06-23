import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import pytest
from bittrace import dashboard

class MockModel:
    def __init__(self, n_samples=20, n_clusters=3, n_labels=2):
        self.population = np.random.randint(0, 256, size=(n_samples, 8), dtype=np.uint8)
        self.num_clusters = n_clusters
        self.medoids = self.population[:n_clusters]
        self.name = "mockmodel"
        self.config = {"example": True}

    def predict(self, X, label_map=None):
        clusters = np.random.randint(0, self.num_clusters, size=X.shape[0])
        if label_map is not None:
            return label_map[clusters]
        return clusters

def test_cluster_summary_and_mapping(tmp_path):
    np.random.seed(42)
    n_samples = 50
    n_clusters = 4
    n_labels = 3
    model = MockModel(n_samples=n_samples, n_clusters=n_clusters, n_labels=n_labels)
    y = np.random.randint(0, n_labels, size=n_samples)
    df, assignments = dashboard.cluster_summary(model, model.population, y, print_table=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == n_clusters

    # Test mapping functions
    label_map = dashboard.print_cluster_mapping(y, assignments, n_clusters, n_labels)
    mapping = dashboard.compute_hungarian_mapping(y, assignments, n_clusters, n_labels)
    hungarian_label_map = np.array([mapping.get(c, -1) for c in range(n_clusters)])
    assert label_map.shape == (n_clusters,)
    assert hungarian_label_map.shape == (n_clusters,)

def test_accuracy_stats_and_confusion():
    n = 10
    n_clusters = 2
    model = MockModel(n_samples=n, n_clusters=n_clusters)
    X = np.random.randint(0, 256, size=(n, 8), dtype=np.uint8)
    y = np.random.randint(0, n_clusters, size=n)
    acc, preds = dashboard.accuracy_stats(model, X, y, name="Test")
    assert 0 <= acc <= 1
    cm = dashboard.plot_confusion(y, preds, show=False)
    assert cm.shape[0] == cm.shape[1]

def test_dataset_stats():
    y = np.array([0, 1, 0, 2, 1, 1])
    df = dashboard.dataset_stats("Test", None, y, print_table=False)
    assert set(df.Label) == {0, 1, 2}
    assert df.Count.sum() == 6

def test_validation_curve(tmp_path):
    import pandas as pd
    log_csv = tmp_path / "log.csv"
    pd.DataFrame({"generation": [1,2,3], "val_accuracy": [0.1, 0.2, 0.3]}).to_csv(log_csv, index=False)
    df = dashboard.plot_validation_curve(str(log_csv), show=False)
    assert "val_accuracy" in df

def test_model_structure():
    model = MockModel()
    out = dashboard.model_structure(model)
    assert isinstance(out, dict) or out is None
