# LazyWrapper & LazyTuner: Complete API Reference

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   LAZY WRAPPER: Automatic Gate Injection                                     ║
║   LAZY TUNER:   CMA-ES Evolutionary Optimization                             ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

## Table of Contents

1. [LazyWrapper](#lazywrapper)
   - [Overview](#wrapper-overview)
   - [Architecture](#wrapper-architecture)
   - [API Reference](#wrapper-api)
   - [Examples](#wrapper-examples)

2. [LazyTuner](#lazytuner)
   - [CMA-ES Background](#cma-es-background)
   - [PAD Fitness Function](#pad-fitness)
   - [API Reference](#tuner-api)
   - [Evolution Examples](#tuner-examples)

3. [Integration Guide](#integration)

---

## LazyWrapper

### Wrapper Overview

`LazyWrapper` transforms any PyTorch model into a lazy-compute-aware model by automatically inserting `LazyGate` modules around expensive layers.

**Key Features:**
- **Automatic layer discovery** via forward hook tracing
- **Dimension-aware cheap paths** with 1×1 projections
- **Unified genome interface** for evolutionary optimization
- **Runtime statistics** for compute/skip monitoring

### Wrapper Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Original Network                            │
│  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐                │
│  │  f₁  │───▶│  f₂  │───▶│  f₃  │───▶│  f₄  │───▶ out       │
│  └──────┘    └──────┘    └──────┘    └──────┘                │
└────────────────────────────────────────────────────────────────┘
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    LazyWrapper                                 │
│  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐                │
│  │Gate₁ │───▶│Gate₂ │───▶│Gate₃ │───▶│Gate₄ │───▶ out       │
│  │(p,a) │    │(p,a) │    │(p,a) │    │(p,a) │                │
│  └──────┘    └──────┘    └──────┘    └──────┘                │
│      │           │           │           │                   │
│  ┌─────────────────────────────────────────┐                 │
│  │          LazyGateConfig                  │                 │
│  │  Genome: [(p₁,a₁), (p₂,a₂), ...]        │                 │
│  └─────────────────────────────────────────┘                 │
└────────────────────────────────────────────────────────────────┘
```

### Wrapper API

#### `LazyWrapper.__init__`

```python
LazyWrapper(
    model: nn.Module,
    sample_input: Optional[torch.Tensor] = None,
    gate_layers: Optional[Set[Type[nn.Module]]] = None,
    initial_p: float = 0.5,
    initial_a: float = 1.0,
    min_flops_threshold: float = 1000,
)
```

**Arguments:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | *required* | Network to wrap |
| `sample_input` | `Tensor` | `None` | Example input for shape inference |
| `gate_layers` | `Set[Type]` | `GATABLE_LAYERS` | Layer types to gate |
| `initial_p` | `float` | `0.5` | Initial procrastination |
| `initial_a` | `float` | `1.0` | Initial ambition |
| `min_flops_threshold` | `float` | `1000` | Min FLOPs to gate |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_gates` | `int` | Number of injected gates |
| `genome_size` | `int` | Flat genome size (2 × num_gates) |
| `procrastination` | `float` | Global P̄ (mean skip probability) |
| `ambition` | `float` | Global Ā (weighted ambition) |
| `effective_compute` | `float` | Expected compute fraction |
| `flops_saved_ratio` | `float` | Actual FLOP savings |
| `gate_config` | `LazyGateConfig` | Gate configuration object |

#### Methods

```python
# Genome manipulation
set_genome(genome: List[Tuple[float, float]]) -> None
set_flat_genome(flat: List[float]) -> None
get_genome() -> List[Tuple[float, float]]
get_flat_genome() -> List[float]

# Statistics
reset_stats() -> None
summary() -> str

# Training modes
enable_gate_training(temperature: float = 1.0) -> None
disable_gate_training() -> None
```

### Wrapper Examples

#### Basic Usage

```python
import torch
import torch.nn as nn
from lazyai import LazyWrapper

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Create and wrap
model = SimpleCNN()
sample = torch.randn(1, 32, 100)
lazy_model = LazyWrapper(model, sample_input=sample, initial_p=0.6)

# Check structure
print(lazy_model.summary())
# LazyWrapper
#   Wrapped model: SimpleCNN
#   Number of gates: 3
#   Global P (procrastination): 0.600
#   Global A (ambition): 1.000
#   Effective compute: 40.0%
```

#### Genome Evolution

```python
# Get current genome
genome = lazy_model.get_flat_genome()
# [0.6, 1.0, 0.6, 1.0, 0.6, 1.0]  # 3 layers × 2 params

# Apply evolved genome
evolved_genome = [0.8, 0.9, 0.7, 0.85, 0.5, 0.95]
lazy_model.set_flat_genome(evolved_genome)

# Check new stats
print(f"New P: {lazy_model.procrastination:.3f}")
print(f"New A: {lazy_model.ambition:.3f}")
print(f"Effective: {lazy_model.effective_compute:.1%}")
```

---

## LazyTuner

### CMA-ES Background

**Covariance Matrix Adaptation Evolution Strategy (CMA-ES)** is a derivative-free optimization algorithm ideal for:
- Non-convex, rugged fitness landscapes
- Moderate dimensionality (2-100 parameters)
- Black-box objective functions

CMA-ES maintains a multivariate Gaussian search distribution:

$$\mathbf{x} \sim \mathcal{N}(\mathbf{m}, \sigma^2 \mathbf{C})$$

Where:
- $\mathbf{m}$: Mean (current best estimate)
- $\sigma$: Step-size (global scaling)
- $\mathbf{C}$: Covariance matrix (search shape)

#### Evolution Steps

1. **Sample** λ offspring: $\mathbf{x}_k = \mathbf{m} + \sigma \cdot \mathbf{B}\mathbf{D}\mathbf{z}_k$
2. **Evaluate** fitness for each offspring
3. **Rank** and select μ best (elite)
4. **Update** mean: $\mathbf{m} \leftarrow \sum_{i=1}^{\mu} w_i \mathbf{x}_{i:\lambda}$
5. **Adapt** covariance: $\mathbf{C} \leftarrow (1-c_1-c_\mu)\mathbf{C} + c_1 \mathbf{p}_c\mathbf{p}_c^T + c_\mu \sum w_i \mathbf{y}_i\mathbf{y}_i^T$
6. **Control** step-size: $\sigma \leftarrow \sigma \cdot \exp\left(\frac{c_\sigma}{d_\sigma}\left(\frac{\|\mathbf{p}_\sigma\|}{E\|\mathcal{N}(0,I)\|} - 1\right)\right)$

### PAD Fitness

The PAD fitness function balances quality with laziness:

$$J(\theta) = Q(\theta) \cdot \exp(\beta \cdot \bar{P}) \cdot (1 + \eta \cdot \bar{A})$$

**Components:**

| Symbol | Name | Description |
|--------|------|-------------|
| $Q$ | Quality | Task metric (F1, accuracy, etc.) |
| $\bar{P}$ | Mean Procrastination | Average skip probability |
| $\bar{A}$ | Weighted Ambition | $\frac{1}{L}\sum_l (1-p_l) \cdot a_l$ |
| $\beta$ | Energy Coefficient | Rewards higher skip rates |
| $\eta$ | Ambition Bonus | Rewards selective precision |

**Quality Floor:**
$$J = 0.1 \cdot Q \quad \text{if } Q < Q_{\min}$$

### Tuner API

#### `LazyTunerConfig`

```python
LazyTunerConfig(
    # PAD coefficients
    beta: float = 1.0,            # Energy bonus for laziness
    eta: float = 0.5,             # Ambition bonus coefficient
    q_min: float = 0.5,           # Minimum acceptable quality
    
    # CMA-ES parameters
    sigma: float = 0.3,           # Initial step size
    population_size: int = 20,    # λ offspring per generation
    elite_ratio: float = 0.5,     # μ/λ selection ratio
    
    # Termination
    max_generations: int = 100,
    target_fitness: float = inf,
    convergence_threshold: float = 1e-6,
    
    # Lazy optimizations
    cache_threshold: float = 0.01,
    max_cache_size: int = 1000,
)
```

#### `LazyTuner.__init__`

```python
LazyTuner(
    genome_size: int,
    fitness_fn: Callable[[List[float]], float],
    config: Optional[LazyTunerConfig] = None,
    initial_genome: Optional[List[float]] = None,
    verbose: bool = True,
)
```

#### Methods

```python
# Evolution
step() -> EvolutionStats           # One generation
evolve() -> Tuple[Genome, History] # Full evolution

# Solution injection (warm starts)
inject_solution(genome: List[float], weight: float = 0.5) -> None
```

#### `EvolutionStats`

```python
@dataclass
class EvolutionStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    best_genome: List[float]
    mean_p: float      # Average P
    mean_a: float      # Average A
    cache_hits: int
    evaluations: int
    sigma: float
```

### Tuner Examples

#### Basic Evolution

```python
from lazyai import LazyTuner, LazyTunerConfig, compute_pad_fitness

# Define fitness function
def evaluate(genome):
    # Apply genome to model
    lazy_model.set_flat_genome(genome)
    lazy_model.reset_stats()
    
    # Evaluate quality (e.g., validation F1)
    quality = evaluate_model(lazy_model)
    
    # Compute PAD fitness
    ps = genome[0::2]  # Skip probabilities
    ays = genome[1::2]  # Ambitions
    mean_p = sum(ps) / len(ps)
    mean_a = sum((1-p)*a for p,a in zip(ps,ays)) / len(ps)
    
    return compute_pad_fitness(quality, mean_p, mean_a, beta=1.0, eta=0.5, q_min=0.85)

# Create tuner
config = LazyTunerConfig(
    beta=1.0,
    eta=0.5,
    q_min=0.85,
    population_size=30,
    max_generations=50
)

tuner = LazyTuner(
    genome_size=lazy_model.genome_size,
    fitness_fn=evaluate,
    config=config,
    initial_genome=lazy_model.get_flat_genome()
)

# Run evolution
best_genome, history = tuner.evolve()

# Apply best solution
lazy_model.set_flat_genome(best_genome)
print(f"Final P: {lazy_model.procrastination:.3f}")
print(f"Final A: {lazy_model.ambition:.3f}")
print(f"Compute saved: {1 - lazy_model.effective_compute:.1%}")
```

#### Using `create_quick_tuner`

```python
from lazyai import create_quick_tuner

# Simple setup
tuner = create_quick_tuner(
    lazy_wrapper=lazy_model,
    quality_evaluator=lambda m: validate(m),  # Returns quality ∈ [0,1]
    beta=1.0,
    eta=0.5,
    q_min=0.85
)

best, history = tuner.evolve()
```

---

## Integration Guide

### Full Pipeline Example

```python
import torch
import torch.nn as nn
from lazyai import (
    LazyWrapper, 
    LazyTuner, 
    LazyTunerConfig,
    create_quick_tuner
)

# 1. Define your model
class AnomalyDetector(nn.Module):
    def __init__(self, in_dim=32, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden // 2, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden // 2, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, in_dim, 7, padding=3),
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

# 2. Create and wrap
model = AnomalyDetector()
sample = torch.randn(1, 32, 100)
lazy_model = LazyWrapper(model, sample_input=sample, initial_p=0.5)

print(f"Gates injected: {lazy_model.num_gates}")
print(f"Genome size: {lazy_model.genome_size}")

# 3. Define quality evaluator
def evaluate_quality(lazy_wrapper):
    lazy_wrapper.eval()
    with torch.no_grad():
        # Run on validation data
        output = lazy_wrapper(val_data)
        mse = ((output - val_data) ** 2).mean().item()
        
        # Convert MSE to quality score ∈ [0, 1]
        quality = 1 / (1 + mse)
    return quality

# 4. Create tuner and evolve
tuner = create_quick_tuner(
    lazy_wrapper=lazy_model,
    quality_evaluator=evaluate_quality,
    beta=1.5,           # Strong laziness incentive
    eta=0.3,            # Moderate ambition bonus
    q_min=0.80,         # Quality floor
    population_size=25,
    max_generations=40
)

best_genome, history = tuner.evolve()

# 5. Apply and report
lazy_model.set_flat_genome(best_genome)
print("\n" + "="*60)
print("OPTIMIZATION COMPLETE")
print("="*60)
print(f"Quality:          {evaluate_quality(lazy_model):.4f}")
print(f"Procrastination:  {lazy_model.procrastination:.3f}")
print(f"Ambition:         {lazy_model.ambition:.3f}")
print(f"Compute Saved:    {1 - lazy_model.effective_compute:.1%}")
print(lazy_model.summary())
```

### Monitoring Evolution

```python
import matplotlib.pyplot as plt

# Plot fitness evolution
generations = [s.generation for s in history]
best = [s.best_fitness for s in history]
mean = [s.mean_fitness for s in history]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(generations, best, 'b-', label='Best')
plt.plot(generations, mean, 'r--', label='Mean')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.title('Evolution Progress')

# Plot P-A trajectory
ps = [s.mean_p for s in history]
ays = [s.mean_a for s in history]

plt.subplot(1, 2, 2)
plt.scatter(ps, ays, c=generations, cmap='viridis')
plt.colorbar(label='Generation')
plt.xlabel('Mean P (Procrastination)')
plt.ylabel('Mean A (Ambition)')
plt.title('PAD Trajectory')
plt.tight_layout()
plt.savefig('evolution.png')
```

---

## Mathematical Appendix

### Effective Compute Derivation

For a single gate with parameters $(p, a)$:

$$\mathbb{E}[\text{compute}] = (1 - p) \cdot a$$

For $L$ gates with parameters $\{(p_l, a_l)\}_{l=1}^L$:

$$\text{Effective Compute} = \frac{\sum_l (1-p_l) \cdot a_l \cdot C_l}{\sum_l C_l}$$

Where $C_l$ is the FLOP cost of layer $l$.

### Fitness Gradient (For Reference)

The gradient of PAD fitness with respect to quality:

$$\frac{\partial J}{\partial Q} = \exp(\beta \bar{P}) \cdot (1 + \eta \bar{A})$$

This shows how fitness scales with quality — at high $\bar{P}$ and $\bar{A}$, quality improvements yield larger fitness gains.

### CMA-ES Complexity

- **Time**: $O(\lambda n^2)$ per generation (dominated by covariance eigen-decomposition)
- **Space**: $O(n^2)$ for covariance matrix
- **Typical convergence**: $O(n)$ to $O(n^2)$ generations

For PAD with $L$ layers: $n = 2L$, so complexity is $O(L^2)$ per generation.

---

*LazyAI v0.2.0 — Evolutionary Gating for Computation-Optimal Inference*
