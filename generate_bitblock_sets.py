import os
import numpy as np
import argparse
from collections import Counter
from PIL import Image

# Configurable parameters
BASE_DIR = './data'                  # Root where /training and /testing live
OUTPUT_DIR = './bitblock_sets'
SPLIT_RATIOS = {'train': 0.7, 'val': 0.2, 'test': 0.1}
SEED = 42

def load_mnist_from_folders(base_dir):
    """Load all images and labels from MNIST-style directory (training/testing/0/*.png, etc.)"""
    images = []
    labels = []
    for split in ['training', 'testing']:
        split_dir = os.path.join(base_dir, split)
        for digit in range(10):
            digit_dir = os.path.join(split_dir, str(digit))
            if not os.path.exists(digit_dir):
                continue
            for fname in os.listdir(digit_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = Image.open(os.path.join(digit_dir, fname)).convert('L')
                    arr = np.array(img)
                    images.append(arr)
                    labels.append(digit)
    X = np.stack(images)
    y = np.array(labels)
    return X, y

def pack_images(images):
    # Input: [N, 28, 28] uint8 (0..255)
    # Output: [N, 98] uint8 (packed bits)
    binarized = (images > 128).astype(np.uint8)
    N, H, W = binarized.shape
    flat = binarized.reshape(N, H*W)
    pad_len = (-flat.shape[1]) % 8
    if pad_len > 0:
        flat = np.pad(flat, ((0, 0), (0, pad_len)), constant_values=0)
    packed = np.packbits(flat, axis=1)
    return packed

def save_split(split, X, y, outdir):
    np.savez_compressed(os.path.join(outdir, f"{split}_X.npz"), X)
    np.savez_compressed(os.path.join(outdir, f"{split}_y.npz"), y)

def build_and_save_bitblock_set(X, y, digit, output_dir, split_ratios, seed=42, total_samples='all'):
    np.random.seed(seed)
    digit = int(digit)
    # Positive and negative indices
    pos_idx = np.where(y == digit)[0]
    neg_idx = np.where(y != digit)[0]
    if total_samples == 'all':
        n_pos = len(pos_idx)
        n_neg = n_pos  # balanced
    else:
        n_pos = min(total_samples // 2, len(pos_idx))
        n_neg = min(total_samples - n_pos, len(neg_idx))
    pos_idx = np.random.choice(pos_idx, n_pos, replace=False)
    neg_idx = np.random.choice(neg_idx, n_neg, replace=False)
    # Prepare arrays
    all_X = np.concatenate([X[pos_idx], X[neg_idx]], axis=0)
    all_y = np.concatenate([np.ones(n_pos, dtype=np.uint8), np.zeros(n_neg, dtype=np.uint8)], axis=0)
    # Shuffle
    perm = np.random.permutation(len(all_y))
    all_X, all_y = all_X[perm], all_y[perm]
    # Split
    N = len(all_y)
    n_train = int(N * split_ratios['train'])
    n_val = int(N * split_ratios['val'])
    n_test = N - n_train - n_val
    train_X, val_X, test_X = all_X[:n_train], all_X[n_train:n_train+n_val], all_X[n_train+n_val:]
    train_y, val_y, test_y = all_y[:n_train], all_y[n_train:n_train+n_val], all_y[n_train+n_val:]
    # Pack bits
    train_X, val_X, test_X = map(pack_images, [train_X, val_X, test_X])
    # Save
    outdir = os.path.join(output_dir, f"{digit}_set")
    os.makedirs(outdir, exist_ok=True)
    save_split('train', train_X, train_y, outdir)
    save_split('val', val_X, val_y, outdir)
    save_split('test', test_X, test_y, outdir)
    # Print stats
    print(f"[{digit}_set] train: {train_X.shape}, val: {val_X.shape}, test: {test_X.shape}")
    print(f"[{digit}_set] train label dist: {Counter(train_y)} | val: {Counter(val_y)} | test: {Counter(test_y)}")

def verify_bitblock_sets(output_dir):
    for digit in range(10):
        outdir = os.path.join(output_dir, f"{digit}_set")
        print(f"== Verifying {outdir} ==")
        for split in ['train', 'val', 'test']:
            X_path = os.path.join(outdir, f"{split}_X.npz")
            y_path = os.path.join(outdir, f"{split}_y.npz")
            if not os.path.exists(X_path) or not os.path.exists(y_path):
                print(f"  {split} missing!")
                continue
            X = np.load(X_path)['arr_0']
            y = np.load(y_path)['arr_0']
            print(f"  {split}: X={X.shape}, y={y.shape}, label dist={Counter(y)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=BASE_DIR, help="Root data dir with training/testing folders")
    parser.add_argument('--outdir', type=str, default=OUTPUT_DIR, help="Output dir for bitblock sets")
    parser.add_argument('--samples', type=int, default=-1, help="Total samples per digit set ('all'=all positives)")
    parser.add_argument('--verify', action='store_true', help="Verify bitblock sets")
    parser.add_argument('--regenerate', action='store_true', help="Rebuild all bitblock sets")
    args = parser.parse_args()

    if args.verify:
        verify_bitblock_sets(args.outdir)
        return

    X, y = load_mnist_from_folders(args.data)
    print(f"Loaded MNIST images: X={X.shape}, y={y.shape}")
    total_samples = 'all' if args.samples < 1 else args.samples
    for digit in range(10):
        print(f"=== Building {digit}_set ===")
        build_and_save_bitblock_set(
            X, y, digit, args.outdir, SPLIT_RATIOS, seed=SEED, total_samples=total_samples
        )
    print("All BitBlock sets generated.")

if __name__ == "__main__":
    main()
