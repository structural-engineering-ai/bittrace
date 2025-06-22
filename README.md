# BitTrace

**BitTrace** is an experimental evolutionary AI framework built entirely from bitwise operations.  
Instead of gradients and floating-point weights, BitTrace evolves cascades of atomic bitwise logic on packed binary arrays to classify, cluster, and analyze patterns.

---

## Features

- **Bit-native learning:** Models operate directly on high-dimensional packed bitstrings.
- **Evolutionary optimization:** No gradient descent—just population-based search over bitwise “layer plans”.
- **Minimal, fast, unconventional:** Designed for computational efficiency and digital clarity.
- **Emergent clustering:** Uses bitwise clustering (Hamming/k-medoids) and Hungarian alignment for label mapping.
- **Pluggable data:** Built for MNIST but can be adapted for any binary or image-like input.

---

## Quick Start

```bash
git clone https://github.com/yourusername/bittrace.git
cd bittrace
python -m venv .venv
source .venv/bin/activate     # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
# Run your training scripts or notebooks here!
