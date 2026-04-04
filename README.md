# Procrastination-Ambition Duality (PAD)

> A model-agnostic framework for computation-optimal neural inference. More details at [kabir.codes](https://kabir.codes).

---

## What is PAD?

Most neural networks are wasteful by design — every layer, every forward pass, full compute, every time. PAD challenges that assumption.

**Procrastination-Ambition Duality** is a PyTorch framework built around a simple but powerful idea: every layer in a network can be characterized by two parameters.

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| **Procrastination** | P | Probability of routing to a cheap identity skip-connection |
| **Ambition** | A | Blend factor when the expensive path is taken |

**Effective Compute = (1 - P) x A**

A layer with P=0.7 and A=0.8 uses, on average, just **24%** of its full compute budget. PAD learns which layers can afford to be lazy — and by how much.

---

## How It Works

PAD wraps any existing PyTorch model using forward hooks, injecting stochastic **Bernoulli gates** (LazyGates) at the layer level. No architecture rewrite required.

Each gate sits between two paths:
- **Expensive path** — the original computation (Conv, Linear, etc.)
- **Cheap path** — an identity skip-connection

Gate parameters (P, A) are not learned via gradient descent. Instead, they are evolved using **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy), with **Gumbel-Softmax relaxation** for differentiable training-time approximation.

The fitness function rewards laziness while enforcing a quality floor:

```
F(theta) = Q(theta) * exp(beta * P) * (1 + eta * A)
```

Where Q is task quality (F1, accuracy, etc.), beta rewards procrastination, and eta prevents ambition from collapsing to zero.

---

## Results

Up to **70% compute reduction** with preserved output quality across tested architectures, achieved purely through evolved gate parameters with no retraining of base model weights.

---

## Stack

- **PyTorch** — forward hook injection, differentiable gating
- **CMA-ES** — evolutionary optimization of gate genomes
- **Gumbel-Softmax** — smooth Bernoulli relaxation during training
- **Python 3.8+**

---

## Quick Start

```python
from lazyai import LazyWrapper, create_quick_tuner

# Wrap any existing model
lazy_model = LazyWrapper(model, sample_input=sample_input, initial_p=0.5)

# Define your quality evaluator
def evaluate_quality(wrapped_model):
    output = wrapped_model(val_data)
    mse = ((output - val_targets) ** 2).mean().item()
    return 1 / (1 + mse)

# Evolve optimal gate parameters
tuner = create_quick_tuner(
    lazy_wrapper=lazy_model,
    quality_evaluator=evaluate_quality,
    beta=1.0,
    q_min=0.85,
    max_generations=50
)

best_genome, history = tuner.evolve()
lazy_model.set_flat_genome(best_genome)

print(f"Compute reduction: {1 - lazy_model.effective_compute:.1%}")
```

---

## Installation

```bash
git clone https://github.com/Kcbir/lazy.git
cd lazy
pip install -e .
```

---

## Learn More

Full write-up, theory, and extended documentation at [kabir.codes](https://kabir.codes).

---

*Why compute when you can procrastinate?*