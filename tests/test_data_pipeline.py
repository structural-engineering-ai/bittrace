import os
import shutil
import numpy as np
import json
import pytest
from PIL import Image
from bittrace import data_pipeline as dp
from bittrace.data_pipeline import random_offset_embed

TRAIN_DATA_FOLDER = './tests/sample_data/training'
TEST_DATA_FOLDER = './tests/sample_data/testing'
CACHE_FOLDER = './tests/cache_test'

def cleanup_cache():
    if os.path.exists(CACHE_FOLDER):
        shutil.rmtree(CACHE_FOLDER)

def test_generator_creates_correct_folders():
    base_path = './tests/sample_data'
    for split in ['training', 'testing']:
        split_path = os.path.join(base_path, split)
        assert os.path.exists(split_path)
        for digit in range(10):
            digit_path = os.path.join(split_path, str(digit))
            assert os.path.exists(digit_path)

def test_generator_creates_expected_number_of_images():
    base_path = './tests/sample_data'
    samples_per_class = 5  # must match generator setting
    for split in ['training', 'testing']:
        for digit in range(10):
            digit_path = os.path.join(base_path, split, str(digit))
            files = [f for f in os.listdir(digit_path) if f.endswith('.png')]
            assert len(files) == samples_per_class

def test_generated_images_have_correct_size_and_mode():
    base_path = os.path.join(TRAIN_DATA_FOLDER, '0')
    sample_files = os.listdir(base_path)[:2]  # test first two images
    for file in sample_files:
        img = Image.open(os.path.join(base_path, file))
        assert img.size == (28, 28)
        assert img.mode == 'L'  # grayscale

def test_generated_images_are_binary():
    base_path = os.path.join(TRAIN_DATA_FOLDER, '0')
    sample_files = os.listdir(base_path)[:2]
    for file in sample_files:
        img = Image.open(os.path.join(base_path, file))
        pixels = list(img.getdata())
        unique_vals = set(pixels)
        assert unique_vals.issubset({0, 255})

def test_load_image_paths_per_class():
    class_images = dp.load_image_paths_per_class(TRAIN_DATA_FOLDER)
    assert isinstance(class_images, dict)
    for k, v in class_images.items():
        assert isinstance(k, int)
        assert all(isinstance(p, str) for p in v)

def test_sample_and_split_class_images():
    class_images = dp.load_image_paths_per_class(TRAIN_DATA_FOLDER)
    splits = dp.sample_and_split_class_images(class_images, total_samples=50,
                                             split_ratios={"train":0.6,"val":0.4,"test":0.0})
    assert "train" in splits and "val" in splits and "test" in splits
    for split in ["train","val"]:
        imgs, labels = splits[split]
        assert len(imgs) == len(labels)

def test_load_and_pack_images():
    class_images = dp.load_image_paths_per_class(TRAIN_DATA_FOLDER)
    splits = dp.sample_and_split_class_images(
        class_images, total_samples=50,
        split_ratios={"train":0.7,"val":0.3,"test":0.0}
    )
    train_imgs, train_labels = splits["train"]
    if len(train_imgs) == 0:
        pytest.skip("No training images sampled, skipping test")

    images_bin = dp.load_and_binarize_images(train_imgs)
    packed = dp.pack_images(images_bin)
    assert packed.dtype == np.uint8
    assert packed.shape[0] == len(train_imgs)

def test_caching_and_stats():
    cleanup_cache()
    class_images = dp.load_image_paths_per_class(TRAIN_DATA_FOLDER)
    splits = dp.sample_and_split_class_images(class_images, total_samples=20,
                                             split_ratios={"train":0.5,"val":0.5,"test":0.0})
    train_imgs, train_labels = splits["train"]
    val_imgs, val_labels = splits["val"]

    train_bin = dp.load_and_binarize_images(train_imgs)
    val_bin = dp.load_and_binarize_images(val_imgs)

    train_X = dp.pack_images(train_bin)
    val_X = dp.pack_images(val_bin)
    train_y = np.array(train_labels)
    val_y = np.array(val_labels)

    train_npz = os.path.join(CACHE_FOLDER, "train.npz")
    train_stats = os.path.join(CACHE_FOLDER, "train_stats.json")
    val_npz = os.path.join(CACHE_FOLDER, "val.npz")
    val_stats = os.path.join(CACHE_FOLDER, "val_stats.json")

    dp.save_npz_cache_with_stats(train_X, train_y, train_npz, train_stats)
    dp.save_npz_cache_with_stats(val_X, val_y, val_npz, val_stats)

    # Check files exist
    assert os.path.exists(train_npz)
    assert os.path.exists(train_stats)
    assert os.path.exists(val_npz)
    assert os.path.exists(val_stats)

    # Load and verify stats json content
    with open(train_stats) as f:
        stats_data = json.load(f)
    assert all(k.isdigit() for k in stats_data.keys())
    assert sum(stats_data.values()) == len(train_y)

def test_prepare_stratified_train_val_and_test():
    cleanup_cache()
    train_val = dp.prepare_stratified_train_val(
        base_train_folder=TRAIN_DATA_FOLDER,
        total_samples=30,
        split_ratios={"train":0.7,"val":0.3,"test":0.0},
        cache_folder=CACHE_FOLDER,
        random_seed=123,
        use_cache=False,
    )
    (train_X, train_y), (val_X, val_y) = train_val

    assert train_X.shape[0] == len(train_y)
    assert val_X.shape[0] == len(val_y)

    test_X, test_y = dp.prepare_test_set(
        base_test_folder=TEST_DATA_FOLDER,
        cache_folder=CACHE_FOLDER,
        use_cache=False,
    )
    assert test_X.shape[0] == len(test_y)

def test_random_offset_embed_basic():
    image_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)  # 8 bits
    target_bit_length = 16
    rng = np.random.RandomState(123)  # deterministic

    out = random_offset_embed(image_bits, target_bit_length, rng)
    assert out.sum() == image_bits.sum()
    assert (out == 1).sum() == (image_bits == 1).sum()
    # Check that the embedded region matches image_bits somewhere in out
    nonzero_start = np.argmax(out != 0)
    assert np.all(out[nonzero_start:nonzero_start+len(image_bits)] == image_bits)

def test_random_offset_embed_randomness():
    image_bits = np.random.randint(0, 2, 32, dtype=np.uint8)
    target_bit_length = 64
    positions = []
    for seed in range(5):
        rng = np.random.RandomState(seed)
        out = random_offset_embed(image_bits, target_bit_length, rng)
        # Find where the image was embedded
        pos = np.argmax([np.all(out[i:i+32] == image_bits) for i in range(target_bit_length-32+1)])
        positions.append(pos)
    assert len(set(positions)) > 1, "Offset is not random across seeds"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
