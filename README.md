# LazyAI

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██       █████  ███████ ██    ██  █████  ██                                ║
║   ██      ██   ██    ███   ██  ██  ██   ██ ██                                ║
║   ██      ███████   ███     ████   ███████ ██                                ║
║   ██      ██   ██  ███       ██    ██   ██ ██                                ║
║   ███████ ██   ██ ███████    ██    ██   ██ ██                                ║
║                                                                               ║
║   Evolutionary Gating for Computation-Optimal Neural Inference               ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

> **Make your neural networks lazy — compute only when necessary, at only the precision required.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

---

## Overview

LazyAI is a framework for **adaptive computation** that wraps any PyTorch neural network with stochastic gating mechanisms. Each layer learns when to skip expensive computation and when to activate at full precision.

### The Core Idea: Procrastination-Ambition Duality (PAD)

Every computational decision is characterized by two parameters:

| Parameter | Symbol | Range | Meaning |
|-----------|--------|-------|---------|
| **Procrastination** | P | [0, 1] | Probability of skipping expensive computation |
| **Ambition** | A | [0, 1] | Blend factor when not skipping (precision) |

**Effective Compute** = (1 - P) × A

For example, with P=0.7 and A=0.8, only **24%** of full computation is used on average.

### Key Features

- **LazyGate**: Dual-path stochastic skip mechanism with differentiable sampling
- **LazyWrapper**: Automatically inject gates into any PyTorch model
- **LazyTuner**: CMA-ES evolutionary optimization for gate parameters
- **PAD Fitness**: Reward laziness while maintaining quality

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/lazyai.git
cd lazyai

# Install (development mode)
pip install -e .

# Or just use directly
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Requirements:**
- Python ≥ 3.8
- PyTorch ≥ 2.0
- NumPy

---

## Quick Start

### 1. Basic Gate

```python
import torch.nn as nn
from lazyai import LazyGate

# Wrap an expensive layer
conv = nn.Conv1d(64, 64, kernel_size=7, padding=3)
gate = LazyGate(conv, initial_p=0.7, initial_a=0.8)

# Only 24% computation expected!
print(f"Effective compute: {gate.effective_compute:.0%}")  # 24%

# Forward pass
x = torch.randn(8, 64, 100)
y = gate(x)  # Stochasically skips 70% of the time
```

### 2. Wrap an Entire Model

```python
from lazyai import LazyWrapper

# Your existing model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(32, 64, 7, padding=3)
        self.conv2 = nn.Conv1d(64, 64, 5, padding=2)
        self.conv3 = nn.Conv1d(64, 32, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.conv3(x)

# Wrap with lazy gates
model = MyModel()
sample_input = torch.randn(1, 32, 100)
lazy_model = LazyWrapper(model, sample_input=sample_input, initial_p=0.5)

print(f"Gates injected: {lazy_model.num_gates}")
print(f"Genome size: {lazy_model.genome_size}")
print(lazy_model.summary())
```

### 3. Evolve Optimal Parameters

```python
from lazyai import create_quick_tuner

# Define your quality evaluator
def evaluate_quality(wrapped_model):
    # Return quality score ∈ [0, 1]
    output = wrapped_model(val_data)
    mse = ((output - val_targets) ** 2).mean().item()
    return 1 / (1 + mse)

# Create and run tuner
tuner = create_quick_tuner(
    lazy_wrapper=lazy_model,
    quality_evaluator=evaluate_quality,
    beta=1.0,           # Energy reward for laziness
    eta=0.5,            # Ambition bonus
    q_min=0.85,         # Minimum quality floor
    population_size=25,
    max_generations=50
)

best_genome, history = tuner.evolve()

# Apply the evolved parameters
lazy_model.set_flat_genome(best_genome)
print(f"Final compute reduction: {1 - lazy_model.effective_compute:.1%}")
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LazyAI Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐     ┌──────────────┐     ┌─────────────┐                │
│   │  LazyGate   │────▶│ LazyWrapper  │────▶│ LazyTuner   │                │
│   │  (p, a)     │     │   (model)    │     │  (CMA-ES)   │                │
│   └─────────────┘     └──────────────┘     └─────────────┘                │
│         │                    │                    │                        │
│         ▼                    ▼                    ▼                        │
│   ┌─────────────────────────────────────────────────────────┐             │
│   │                     PAD Framework                        │             │
│   │                                                          │             │
│   │   Forward:  y = z·a·f(x) + (1 - z·a)·skip(x)           │             │
│   │   Fitness:  J = Q · exp(β·P) · (1 + η·A)               │             │
│   │                                                          │             │
│   └─────────────────────────────────────────────────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Mathematical Details

#### LazyGate Forward Pass

```
z ~ Bernoulli(1 - p)          # Gate decision (compute or skip)
y_cheap = skip(x)             # Cheap path (e.g., identity, 1x1 conv)
y_expensive = f(x)            # Full computation (only if z=1)

y = z·a·y_expensive + (1 - z·a)·y_cheap
```

#### PAD Fitness Function

```
J(θ) = Q(θ) · exp(β · P̄) · (1 + η · Ā)

Where:
  Q(θ) = task quality (F1, accuracy, etc.)
  P̄ = mean procrastination across layers
  Ā = weighted mean ambition
  β = energy coefficient (reward laziness)
  η = ambition bonus coefficient
```

#### CMA-ES Evolution

LazyTuner uses Covariance Matrix Adaptation Evolution Strategy:

```
x ~ N(m, σ²C)                  # Sample population
Rank by fitness                # Select elite
m ← Σ wᵢ · x_{i:λ}            # Update mean
C ← adapt covariance          # Learn search shape
σ ← control step-size         # Adjust exploration
```

---

## API Reference

### LazyGate

```python
LazyGate(
    expensive_fn: nn.Module,           # The expensive computation
    cheap_fn: nn.Module = Identity(),  # Skip path (auto-inferred if None)
    initial_p: float = 0.5,            # Skip probability
    initial_a: float = 1.0,            # Ambition when computing
    name: str = "gate"
)

# Properties
gate.p                    # Get/set procrastination
gate.a                    # Get/set ambition
gate.effective_compute    # (1-p) × a
gate.stats                # GateStatistics object

# Methods
gate.reset_stats()
gate.set_deterministic(True/False)
gate.enable_gate_training(temperature=1.0)  # Gumbel-Softmax
```

### LazyWrapper

```python
LazyWrapper(
    model: nn.Module,
    sample_input: torch.Tensor,        # For shape inference
    initial_p: float = 0.5,
    initial_a: float = 1.0,
    min_flops_threshold: float = 1000  # Only gate expensive layers
)

# Properties
lazy.num_gates            # Number of injected gates
lazy.genome_size          # 2 × num_gates
lazy.procrastination      # Global P̄
lazy.ambition             # Global Ā (weighted)
lazy.effective_compute    # Expected compute fraction
lazy.flops_saved_ratio    # Actual runtime savings

# Methods
lazy.get_flat_genome()    # [p₁, a₁, p₂, a₂, ...]
lazy.set_flat_genome(genome)
lazy.reset_stats()
lazy.summary()
```

### LazyTuner

```python
LazyTuner(
    genome_size: int,
    fitness_fn: Callable[[List[float]], float],
    config: LazyTunerConfig = None,
    initial_genome: List[float] = None,
    verbose: bool = True
)

# Methods
best_genome, history = tuner.evolve(max_generations=100)
stats = tuner.step()                    # Single generation
tuner.inject_solution(genome, weight)   # Warm start
```

### LazyTunerConfig

```python
LazyTunerConfig(
    # PAD fitness
    beta: float = 1.0,           # Laziness reward
    eta: float = 0.5,            # Ambition bonus
    q_min: float = 0.5,          # Quality floor
    
    # CMA-ES
    sigma: float = 0.3,          # Initial step size
    population_size: int = 20,   # λ
    elite_ratio: float = 0.5,    # μ/λ
    max_generations: int = 100
)
```

---

## Project Structure

```
lazyai/
├── __init__.py              # Package entry point
├── core/
│   ├── __init__.py          # Core module exports
│   ├── gate.py              # LazyGate implementation
│   ├── wrapper.py           # LazyWrapper implementation
│   └── tuner.py             # LazyTuner (CMA-ES) implementation
│
docs/
├── LAZY_GATE.md             # Gate documentation
├── LAZY_WRAPPER_TUNER.md    # Wrapper & tuner documentation
│
tests/
└── test_lazyai.py           # Comprehensive test suite
```

---

## Experimental Results

Coming soon: Benchmarks on time-series anomaly detection datasets (SMD, PSM, MSL/SMAP).

**Target metrics:**
- F1 score vs baseline (quality preservation)
- FLOPs reduction (compute savings)
- Latency improvement (real-time speedup)
- Energy consumption (edge device metrics)

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python test_lazyai.py`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## Citation

If you use LazyAI in your research, please cite:

```bibtex
@software{lazyai2026,
  title={LazyAI: Evolutionary Gating for Computation-Optimal Neural Inference},
  author={LazyAI Research},
  year={2026},
  url={https://github.com/yourusername/lazyai}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Why compute when you can procrastinate?</i>
</p>
