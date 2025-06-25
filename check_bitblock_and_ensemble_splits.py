import os
import numpy as np
from collections import Counter

def print_split_stats(set_name, split, y):
    counts = Counter(y)
    n = len(y)
    class_counts = ', '.join(f"{int(k)}:{v}" for k, v in sorted(counts.items()))
    print(f"{set_name:^15} | {split:^7} | {n:^8} | {class_counts}")

def check_bitblock_splits(base_dir='./bitblock_sets'):
    print(f"\n{'='*30}\nBitBlock Per-Digit Splits\n{'='*30}")
    print(f"{'Digit':^15} | {'Split':^7} | {'Samples':^8} | {'Class counts'}")
    print('-' * 55)
    for digit in range(10):
        set_dir = os.path.join(base_dir, f"{digit}_set")
        if not os.path.isdir(set_dir):
            print(f"{str(digit):^15} | {'--':^7} | {'--':^8} | (missing set)")
            continue
        for split in ['train', 'val', 'test']:
            y_path = os.path.join(set_dir, f"{split}_y.npz")
            if not os.path.exists(y_path):
                print(f"{str(digit):^15} | {split:^7} | {'--':^8} | (missing file)")
                continue
            y = np.load(y_path)['arr_0']
            print_split_stats(str(digit), split, y)

def check_ensemble_splits(base_dir='./bitblock_sets'):
    print(f"\n{'='*30}\nEnsemble/Meta Splits\n{'='*30}")
    print(f"{'Set':^15} | {'Split':^7} | {'Samples':^8} | {'Class counts'}")
    print('-' * 55)
    for split in ['train', 'val', 'test']:
        y_path = os.path.join(base_dir, f'ensemble_{split}_y.npz')
        if not os.path.exists(y_path):
            print(f"{'ensemble':^15} | {split:^7} | {'--':^8} | (missing file)")
            continue
        y = np.load(y_path)['arr_0']
        print_split_stats('ensemble', split, y)

if __name__ == "__main__":
    base_dir = './bitblock_sets'
    check_bitblock_splits(base_dir)
    check_ensemble_splits(base_dir)
