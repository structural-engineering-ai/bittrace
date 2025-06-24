# bittrace/data_pipeline.py

import os
from PIL import Image
import numpy as np
import json
from collections import Counter

def unpack_bits(packed_array, original_shape):
    """
    Unpack a packed uint8 binary array back to a full binary array
    shaped (N, H, W).
    """
    N = packed_array.shape[0]
    num_bits = np.prod(original_shape)
    unpacked = np.unpackbits(packed_array, axis=1)[:, :num_bits]
    return unpacked.reshape((N,) + original_shape)

def random_offset_embed(image_bits, target_bit_length, rng=None):
    """
    Embed a flat image bit array (length ≤ target_bit_length) into a longer bit array at a random offset.
    """
    if rng is None:
        rng = np.random
    image_len = len(image_bits)
    if image_len > target_bit_length:
        raise ValueError("Image is too large for target embedding width")
    offset = rng.randint(0, target_bit_length - image_len + 1)
    out = np.zeros(target_bit_length, dtype=np.uint8)
    out[offset:offset + image_len] = image_bits
    return out


def load_image_paths_per_class(base_folder):
    """
    Returns dict {label: [image_path, ...]} of all images per class.
    """
    class_images = {}
    for label_str in sorted(os.listdir(base_folder)):
        label_path = os.path.join(base_folder, label_str)
        if not os.path.isdir(label_path):
            continue
        images = [
            os.path.join(label_path, f)
            for f in os.listdir(label_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        class_images[int(label_str)] = images
    return class_images

def count_total_samples(class_images):
    """
    Count total number of images available across all classes.
    """
    return sum(len(imgs) for imgs in class_images.values())

def sample_and_split_class_images(class_images, total_samples, split_ratios, random_seed=42):
    """
    Stratified sampling and splitting per class.
    """
    np.random.seed(random_seed)
    all_labels = sorted(class_images.keys())
    splits = {'train': ([], []), 'val': ([], []), 'test': ([], [])}

    total_available = count_total_samples(class_images)
    if isinstance(total_samples, str) and total_samples.upper() == 'ALL':
        total_samples = total_available
    elif total_samples > total_available:
        print(f"Warning: requested total_samples={total_samples} exceeds available={total_available}. Using {total_available} instead.")
        total_samples = total_available

    for label in all_labels:
        imgs = class_images[label]
        n_class = len(imgs)
        class_sample_count = int(total_samples * (n_class / total_available))
        class_sample_count = min(class_sample_count, n_class)

        imgs = np.array(imgs)
        indices = np.arange(n_class)
        np.random.shuffle(indices)
        sampled_indices = indices[:class_sample_count]
        sampled_imgs = imgs[sampled_indices]

        n_train = int(class_sample_count * split_ratios['train'])
        n_val = int(class_sample_count * split_ratios['val'])
        n_test = class_sample_count - n_train - n_val

        train_imgs = sampled_imgs[:n_train]
        val_imgs = sampled_imgs[n_train:n_train+n_val]
        test_imgs = sampled_imgs[n_train+n_val:n_train+n_val+n_test]

        splits['train'][0].extend(train_imgs.tolist())
        splits['train'][1].extend([label] * len(train_imgs))

        splits['val'][0].extend(val_imgs.tolist())
        splits['val'][1].extend([label] * len(val_imgs))

        splits['test'][0].extend(test_imgs.tolist())
        splits['test'][1].extend([label] * len(test_imgs))

    return splits

def load_and_binarize_images(image_paths):
    """
    Load images and binarize.
    """
    images = []
    for path in image_paths:
        img = Image.open(path).convert('L')
        arr = np.array(img)
        binary = (arr > 128).astype(np.uint8)
        images.append(binary)
    if len(images) == 0:
        return np.array([])  # Handle empty case gracefully
    images = np.stack(images)
    return images

def pack_images(images_bin, bit_length=None, use_random_offset=False, rng=None):
    """
    Pack binary images into uint8 arrays, with optional random offset embedding.
    """
    if images_bin.size == 0:
        packed_length = (bit_length or 784 + 7) // 8
        return np.empty((0, packed_length), dtype=np.uint8)
    N, H, W = images_bin.shape
    flat_images = images_bin.reshape(N, H * W)
    if bit_length is None:
        bit_length = H * W
    packed_length = (bit_length + 7) // 8

    if use_random_offset and bit_length > H * W:
        # Embed each image at a random offset in the bitstring
        if rng is None:
            rng = np.random
        result = np.zeros((N, bit_length), dtype=np.uint8)
        for i in range(N):
            result[i] = random_offset_embed(flat_images[i], bit_length, rng)
        # Now pack
        packed = np.packbits(result, axis=1)
    else:
        # Standard: pad to next byte if needed, then pack
        pad_len = (-flat_images.shape[1]) % 8
        if pad_len > 0:
            flat_images = np.pad(flat_images, ((0, 0), (0, pad_len)), constant_values=0)
        packed = np.packbits(flat_images, axis=1)
        if bit_length and packed.shape[1] > packed_length:
            packed = packed[:, :packed_length]  # Trim if over

    return packed


def save_npz_cache(packed_images, labels, filepath):
    """
    Save packed images and labels to compressed npz.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, images=packed_images, labels=labels)
    print(f"Saved cached dataset: {filepath}")

def load_npz_cache(filepath):
    """
    Load packed images and labels from compressed npz.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cache file not found: {filepath}")
    data = np.load(filepath)
    return data['images'], data['labels']

def tally_class_distribution(labels):
    """
    Count occurrences of each class label in the labels array.
    Returns dict with label (str) keys and counts.
    """
    counts = Counter(labels.tolist())
    # Convert keys to strings for JSON compatibility
    return {str(k): v for k, v in counts.items()}

def save_split_statistics(stats_dict, filepath):
    """
    Save the stats dict (with train/val/test keys) as a JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"Saved dataset statistics to {filepath}")

def save_npz_cache_with_stats(packed_images, labels, npz_path, stats_path):
    """
    Save dataset npz cache and class distribution stats JSON.
    """
    save_npz_cache(packed_images, labels, npz_path)
    stats = tally_class_distribution(labels)
    save_split_statistics(stats, stats_path)

def prepare_stratified_train_val(
    base_train_folder,
    total_samples,
    split_ratios,
    cache_folder,
    random_seed=42,
    use_cache=True,
    bit_length=None,
    use_random_offset=False,
    rng=None
):
    """
    Prepare train and val splits from training folder with stratified sampling and caching.
    Returns:
        (train_X, train_y), (val_X, val_y)
    """
    train_cache_path = os.path.join(cache_folder, 'train.npz')
    train_stats_path = os.path.join(cache_folder, 'train_stats.json')
    val_cache_path = os.path.join(cache_folder, 'val.npz')
    val_stats_path = os.path.join(cache_folder, 'val_stats.json')

    if use_cache and os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        print("Loading train and val splits from cache...")
        train_X, train_y = load_npz_cache(train_cache_path)
        val_X, val_y = load_npz_cache(val_cache_path)
    else:
        print("Preparing train and val splits from raw data...")
        class_images = load_image_paths_per_class(base_train_folder)
        splits = sample_and_split_class_images(class_images, total_samples, split_ratios, random_seed)

        train_imgs, train_labels = splits['train']
        val_imgs, val_labels = splits['val']

        train_bin = load_and_binarize_images(train_imgs)
        val_bin = load_and_binarize_images(val_imgs)

        train_X = pack_images(train_bin, bit_length=bit_length, use_random_offset=use_random_offset, rng=rng)
        val_X = pack_images(val_bin, bit_length=bit_length, use_random_offset=use_random_offset, rng=rng)
        train_y = np.array(train_labels)
        val_y = np.array(val_labels)

        save_npz_cache_with_stats(train_X, train_y, train_cache_path, train_stats_path)
        save_npz_cache_with_stats(val_X, val_y, val_cache_path, val_stats_path)

    return (train_X, train_y), (val_X, val_y)


def prepare_test_set(
    base_test_folder,
    cache_folder,
    use_cache=True,
    bit_length=None,
    use_random_offset=False,
    rng=None
):
    """
    Prepare test set by loading all data (no sampling), with caching.
    Returns:
        test_X, test_y
    """
    test_cache_path = os.path.join(cache_folder, 'test.npz')
    test_stats_path = os.path.join(cache_folder, 'test_stats.json')

    if use_cache and os.path.exists(test_cache_path):
        print("Loading test split from cache...")
        test_X, test_y = load_npz_cache(test_cache_path)
    else:
        print("Preparing test split from raw data...")
        class_images = load_image_paths_per_class(base_test_folder)
        all_imgs = []
        all_labels = []
        for label, imgs in class_images.items():
            all_imgs.extend(imgs)
            all_labels.extend([label] * len(imgs))

        test_bin = load_and_binarize_images(all_imgs)
        test_X = pack_images(test_bin, bit_length=bit_length, use_random_offset=use_random_offset, rng=rng)
        test_y = np.array(all_labels)

        save_npz_cache_with_stats(test_X, test_y, test_cache_path, test_stats_path)

    return test_X, test_y
