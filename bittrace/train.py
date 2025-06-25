import numpy as np
from bittrace.model import BitTraceModel
from bittrace.data_loader import load_bittrace_digit

def run_bittrace_population_evolution(config, digit=0):
    np.random.seed(config.get("random_seed", 42))

    X_train, y_train = load_bittrace_digit(digit, split='train', base_dir=config["bitblock_sets_dir"])
    X_val, y_val = load_bittrace_digit(digit, split='val', base_dir=config["bitblock_sets_dir"])

    pop_size = config.get("pop_size", 8)
    generations = config.get("generations", 100)
    mutation_rate = config.get("mutation_rate", 0.02)
    bit_length = config.get("bit_length", 1028)
    num_layers = config.get("num_layers", 8)

    # Initialize population
    population = [
        BitTraceModel(
            bit_length=bit_length,
            num_layers=num_layers,
            pop_size=1,
            mutation_rate=mutation_rate,
        ) for _ in range(pop_size)
    ]

    best_acc = 0.0
    best_model = None

    for gen in range(generations):
        # Mutate all: Each model produces a mutated offspring
        offspring = []
        for parent in population:
            child = BitTraceModel(
                bit_length=parent.bit_length,
                num_layers=parent.num_layers,
                pop_size=1,
                mutation_rate=parent.mutation_rate,
            )
            child.layer_plan = list(parent.layer_plan)
            child.population = parent.population.copy()
            child.evolve()
            # Optionally mutate the layer plan
            if np.random.rand() < 0.25:
                child.layer_plan = child._generate_random_layer_plan(child.num_layers, child.bytes_per)
            offspring.append(child)

        # Combine parents and offspring
        all_candidates = population + offspring

        # Evaluate all on validation set
        scores = []
        for model in all_candidates:
            transformed = model.forward(X_val)
            preds = model.predict_labels(transformed)
            acc = np.mean(preds == y_val)
            scores.append(acc)

        # Select N-best for next generation
        indices = np.argsort(scores)[-pop_size:]  # Best pop_size
        population = [all_candidates[i] for i in indices]
        best_gen_acc = max([scores[i] for i in indices])
        if best_gen_acc > best_acc:
            best_acc = best_gen_acc
            best_model = population[np.argmax([scores[i] for i in indices])]

        if gen % 10 == 0 or gen == generations - 1:
            print(f"[Gen {gen}] Best acc: {best_gen_acc:.4f} | Overall best: {best_acc:.4f}")

    # Save/checkpoint best model
    if best_model:
        best_model.save_checkpoint(f"bittrace_digit{digit}_best.npz")
    print(f"[FINAL] Best Validation Accuracy: {best_acc:.4f}")
    return best_model, best_acc
