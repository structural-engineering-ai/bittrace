import matplotlib.pyplot as plt
import numpy as np
import umap

def plot_accuracy(csv_path):
    gens, times, accs = [], [], []
    with open(csv_path, 'r') as f:
        next(f)  # skip header
        for line in f:
            gen, t, acc = line.strip().split(',')
            gens.append(int(gen))
            times.append(float(t))
            accs.append(float(acc) if acc else None)

    plt.figure(figsize=(8,5))
    plt.plot(gens, accs, marker='o')
    plt.title("Validation Accuracy over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Validation Accuracy")
    plt.grid(True)
    plt.show()

def plot_umap(embeddings, labels):
    reducer = umap.UMAP()
    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='tab10', s=5)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("UMAP Projection of BitTrace Residues")
    plt.show()

def extract_embeddings(model):
    # Residues = population bit arrays (N, M)
    # Unpack bits for each byte to a full bit vector
    bits = np.unpackbits(model.population, axis=1)
    return bits
