import pytest
import numpy as np
from bittrace import data_loader

def test_bittrace_digit_loader():
    for digit in range(10):
        for split in ['train', 'val', 'test']:
            X, y = data_loader.load_bittrace_digit(digit, split)
            assert X.shape[0] == y.shape[0], f"Mismatch for digit {digit} split {split}"
            assert X.ndim == 2 and y.ndim == 1
            assert X.dtype == 'uint8'
            assert set(np.unique(y)).issubset({0, 1})

def test_ensemble_split_loader():
    for split in ['train', 'val', 'test']:
        X, y = data_loader.load_ensemble_split(split)
        assert X.shape[0] == y.shape[0]
        assert X.ndim == 2 and y.ndim == 1
        assert X.dtype == 'uint8'
        # For ensemble, should have at least 2 unique class labels (0-9)
        assert np.unique(y).shape[0] >= 2
