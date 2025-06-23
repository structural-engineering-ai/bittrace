import matplotlib.pyplot as plt
import numpy as np
import umap

def plot_accuracy(csv_path):
    gens, times, accs = [], [], []
    with open(csv_path, 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            gen, t, acc = parts
            gens.append(int(gen))
            times.append(float(t))
            accs.append(float(acc) if acc else np.nan)

    plt.figure(figsize=(8,5))
    plt.plot(gens, accs, marker='o')
    plt.title("Validation Accuracy over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_umap(embeddings, labels):
    if len(embeddings) != len(labels):
        raise ValueError(f"plot_umap: len(embeddings)={len(embeddings)} but len(labels)={len(labels)}")
    reducer = umap.UMAP()
    embedding_2d = reducer.fit_transform(embeddings)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='tab10', s=5)
    plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("UMAP Projection of BitTrace Residues")
    plt.tight_layout()
    plt.show()

def extract_embeddings(model, bit_length=None):
    """
    Unpack model.population from (N, M) packed bytes to (N, bit_length) bits.
    """
    bits = np.unpackbits(model.population, axis=1)
    if bit_length is not None and bits.shape[1] > bit_length:
        bits = bits[:, :bit_length]
    return bits
