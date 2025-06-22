from bittrace.data_pipeline import save_prepacked_dataset

if __name__ == "__main__":
    print("Preprocessing training data...")
    save_prepacked_dataset('./data/training', './data/prepacked_training.npz')

    print("Preprocessing testing data...")
    save_prepacked_dataset('./data/testing', './data/prepacked_testing.npz')

    print("Preprocessing completed.")
