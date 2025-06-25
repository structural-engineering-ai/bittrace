import os
import numpy as np

def load_bitblocks_mixed_split(split='train', base_dir='./bitblock_sets'):
    """
    Loads and combines all digits’ positive samples for the given split.
    Returns:
        X_full: [N, ...] packed bit arrays (all digits, only true-positives for each)
        y_full: [N,] true digit labels (0-9)
        idxs:   [(digit, idx_within_digit)] for traceability
    """
    all_X = []
    all_y = []
    idxs = []
    for digit in range(10):
        set_dir = os.path.join(base_dir, f"{digit}_set")
        X = np.load(os.path.join(set_dir, f"{split}_X.npz"))['arr_0']
        y = np.load(os.path.join(set_dir, f"{split}_y.npz"))['arr_0']
        pos_mask = (y == 1)
        X_pos = X[pos_mask]
        y_pos = np.full(X_pos.shape[0], digit, dtype=np.uint8)
        all_X.append(X_pos)
        all_y.append(y_pos)
        idxs.extend([(digit, i) for i in range(X_pos.shape[0])])
    X_full = np.concatenate(all_X, axis=0)
    y_full = np.concatenate(all_y, axis=0)
    p = np.random.permutation(len(y_full))
    return X_full[p], y_full[p], [idxs[i] for i in p]

def save_ensemble_set(split='train', base_dir='./bitblock_sets'):
    X, y, idxs = load_bitblocks_mixed_split(split=split, base_dir=base_dir)
    np.savez_compressed(os.path.join(base_dir, f'ensemble_{split}_X.npz'), X)
    np.savez_compressed(os.path.join(base_dir, f'ensemble_{split}_y.npz'), y)
    print(f"Saved ensemble {split} set: X.shape={X.shape}, y.shape={y.shape}")

if __name__ == "__main__":
    for split in ['train', 'val', 'test']:
        save_ensemble_set(split, base_dir='./bitblock_sets')
