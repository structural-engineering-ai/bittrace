import os
import numpy as np

def _check_file_exists(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing required file: {path}")

def _check_folder_exists(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Missing required folder: {path}")

def load_bittrace_digit(digit, split='train', base_dir='./bitblock_sets'):
    """
    Load train/val/test split for a single digit’s BitTrace.
    Args:
        digit (int): Class/digit (0-9)
        split (str): 'train', 'val', 'test'
        base_dir (str): Folder containing per-digit sets
    Returns:
        (X, y): tuple of numpy arrays
    """
    set_dir = os.path.join(base_dir, f"{digit}_set")
    _check_folder_exists(set_dir)
    x_path = os.path.join(set_dir, f"{split}_X.npz")
    y_path = os.path.join(set_dir, f"{split}_y.npz")
    _check_file_exists(x_path)
    _check_file_exists(y_path)
    X = np.load(x_path)['arr_0']
    y = np.load(y_path)['arr_0']
    return X, y

def load_ensemble_split(split='val', base_dir='./bitblock_sets'):
    """
    Load ensemble (meta/combined) split for BitBlock multiclass validation/testing.
    Args:
        split (str): 'train', 'val', 'test'
        base_dir (str): Folder with ensemble files
    Returns:
        (X, y): tuple of numpy arrays
    """
    x_path = os.path.join(base_dir, f'ensemble_{split}_X.npz')
    y_path = os.path.join(base_dir, f'ensemble_{split}_y.npz')
    _check_file_exists(x_path)
    _check_file_exists(y_path)
    X = np.load(x_path)['arr_0']
    y = np.load(y_path)['arr_0']
    return X, y

def print_set_stats(X, y, name=""):
    print(f"{name}: X shape = {X.shape}, y shape = {y.shape}, unique y = {np.unique(y)}")

def sanity_check_all_digits(base_dir='./bitblock_sets'):
    """Print stats for all digit splits."""
    for digit in range(10):
        for split in ['train', 'val', 'test']:
            try:
                X, y = load_bittrace_digit(digit, split, base_dir)
                print_set_stats(X, y, f"Digit {digit} [{split}]")
            except Exception as e:
                print(f"Digit {digit} [{split}]: {e}")

def sanity_check_ensemble_sets(base_dir='./bitblock_sets'):
    """Print stats for all ensemble/meta splits."""
    for split in ['train', 'val', 'test']:
        try:
            X, y = load_ensemble_split(split, base_dir)
            print_set_stats(X, y, f"Ensemble [{split}]")
        except Exception as e:
            print(f"Ensemble [{split}]: {e}")

# CLI self-test example:
if __name__ == "__main__":
    BASE = "./bitblock_sets"
    print("Sanity check on all digit splits:")
    sanity_check_all_digits(BASE)
    print("\nSanity check on ensemble/meta splits:")
    sanity_check_ensemble_sets(BASE)
