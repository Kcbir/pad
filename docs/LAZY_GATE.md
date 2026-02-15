# LazyGate: The Dual-Path Stochastic Skip Mechanism

## Mathematical Foundation

LazyGate implements the core computational unit of the PAD (Procrastination-Ambition Duality) framework. Every gate is parameterized by two evolved values:

| Parameter | Symbol | Domain | Meaning |
|-----------|--------|--------|---------|
| **Procrastination** | $p$ | $[0, 1]$ | Probability of skipping the expensive computation |
| **Ambition** | $a$ | $[0, 1]$ | Contribution strength when computing |

### The Forward Equation

Given input $x$, expensive path $f$, and cheap path $\text{skip}$:

$$z \sim \text{Bernoulli}(1 - p)$$

$$y = z \cdot a \cdot f(x) + (1 - z \cdot a) \cdot \text{skip}(x)$$

This formulation has elegant properties:

| Condition | Behavior |
|-----------|----------|
| $p = 1$ (max lazy) | Always skip: $y = \text{skip}(x)$ |
| $p = 0, a = 1$ (max effort) | Always compute fully: $y = f(x)$ |
| $p = 0, a = 0$ (compute but don't care) | Always skip: $y = \text{skip}(x)$ |
| $p = 0.5, a = 0.5$ | 50% chance to compute, 50% contribution |

### Effective Computation

The **expected fraction** of expensive computation actually used:

$$\mathbb{E}[\text{effective compute}] = (1 - p) \cdot a$$

This directly maps to FLOP savings:

$$\text{FLOP savings ratio} = 1 - (1 - p) \cdot a$$

**Example**: With $p = 0.8$ (80% skip) and $a = 0.5$ (50% ambition):
- Effective compute: $(1 - 0.8) \times 0.5 = 10\%$
- FLOP savings: $90\%$

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                           Input x                              │
│                              │                                 │
│                ┌─────────────┴─────────────┐                   │
│                ▼                           ▼                   │
│       ┌─────────────────┐         ┌─────────────────┐         │
│       │  Expensive Path │         │   Cheap Path    │         │
│       │  f(x)           │         │   skip(x)       │         │
│       │                 │         │                 │         │
│       │  • Full Conv    │         │  • Identity     │         │
│       │  • Full Linear  │         │  • 1x1 Conv     │         │
│       │  • Full Attn    │         │  • Projection   │         │
│       └────────┬────────┘         └────────┬────────┘         │
│                │                           │                   │
│                ▼                           ▼                   │
│       ┌─────────────────────────────────────────────┐         │
│       │  Gate Decision: z ~ Bernoulli(1 - p)        │         │
│       │                                             │         │
│       │  y = z·a·f(x) + (1 - z·a)·skip(x)          │         │
│       └─────────────────────────────────────────────┘         │
│                              │                                 │
│                           Output y                             │
└────────────────────────────────────────────────────────────────┘
```

### The Cheap Path

The cheap path serves as the "lazy fallback" when the gate decides to skip:

| Scenario | Cheap Path | Cost |
|----------|------------|------|
| Same dimensions (in = out) | `nn.Identity()` | 0 FLOPs |
| Different dimensions | `nn.Conv1d(k=1)` or `nn.Linear` | Minimal |

The cheap path is always executed (it's cheap!). The expensive path is only computed when the gate samples $z = 1$.

---

## Gumbel-Softmax Reparameterization

For differentiable training (optional fine-tuning after evolution), LazyGate supports Gumbel-Softmax sampling:

### The Concrete Distribution

Convert skip probability $p$ to logit:

$$\ell = \log\frac{1-p}{p}$$

Sample with Gumbel noise:

$$z_{\text{soft}} = \sigma\left(\frac{\ell + g_1 - g_0}{\tau}\right)$$

where $g_i \sim \text{Gumbel}(0,1) = -\log(-\log(U))$ and $U \sim \text{Uniform}(0,1)$

### Temperature $\tau$

| $\tau$ | Behavior |
|--------|----------|
| $\tau \to 0$ | Hard discrete (but gradient vanishes) |
| $\tau = 1$ | Balanced soft |
| $\tau \to \infty$ | Uniform random |

### Straight-Through Estimator

When `hard=True`:
- **Forward**: Round to hard decision
- **Backward**: Gradient flows through soft sample

```python
y_hard = round(y_soft)  # Forward
gradient = ∂L/∂y_soft   # Backward (as if no rounding)
```

---

## API Reference

### `LazyGate`

```python
class LazyGate(nn.Module):
    def __init__(
        self,
        expensive_fn: nn.Module,          # The full computation
        cheap_fn: Optional[nn.Module],    # The skip path (auto-created if None)
        in_features: Optional[int],       # For auto cheap_fn
        out_features: Optional[int],      # For auto cheap_fn
        initial_p: float = 0.5,           # Skip probability
        initial_a: float = 1.0,           # Ambition
        flops_expensive: float = 0.0,     # For statistics
        flops_cheap: float = 0.0,         # For statistics
        name: str = "gate"                # Identifier
    )
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `p` | `float` | Skip probability (read/write) |
| `a` | `float` | Ambition (read/write) |
| `effective_compute` | `float` | Expected $(1-p) \cdot a$ |
| `stats` | `GateStatistics` | Runtime statistics |

### Methods

| Method | Description |
|--------|-------------|
| `forward(x)` | Run dual-path computation |
| `enable_gate_training(τ)` | Enable Gumbel-Softmax |
| `disable_gate_training()` | Use hard Bernoulli |
| `set_deterministic(compute)` | Force always-compute or always-skip |
| `reset_stats()` | Clear statistics |

### `GateStatistics`

```python
@dataclass
class GateStatistics:
    compute_count: int       # Times expensive path taken
    skip_count: int          # Times cheap path taken
    total_flops_full: float  # FLOPs if always computed
    total_flops_actual: float # FLOPs actually used
    
    @property
    def skip_ratio(self) -> float: ...
    
    @property
    def flops_saved_ratio(self) -> float: ...
```

### `LazyGateConfig`

Manages multiple gates as a single genome for evolutionary optimization:

```python
config = LazyGateConfig([gate1, gate2, gate3])

# Get/set as structured genome
genome = config.get_genome()  # [(p1,a1), (p2,a2), (p3,a3)]
config.set_genome(new_genome)

# Get/set as flat vector (for CMA-ES)
flat = config.get_flat_genome()  # [p1, a1, p2, a2, p3, a3]
config.set_flat_genome(new_flat)

# Aggregate statistics
print(config.total_skip_ratio)
print(config.total_flops_saved_ratio)
print(config.summary())
```

---

## Usage Examples

### Basic Usage

```python
import torch
import torch.nn as nn
from lazyai import LazyGate

# Create a lazy convolution
conv = nn.Conv1d(64, 64, kernel_size=7, padding=3)
gate = LazyGate(
    expensive_fn=conv,
    initial_p=0.7,    # Skip 70% of the time
    initial_a=0.9,    # When computing, use 90% contribution
    name="conv_block"
)

# Forward pass
x = torch.randn(32, 64, 100)  # (batch, channels, seq_len)
y = gate(x)

# Check statistics
print(f"Skip ratio: {gate.stats.skip_ratio:.1%}")
print(f"FLOPs saved: {gate.stats.flops_saved_ratio:.1%}")
```

### Factory Function

```python
from lazyai import create_lazy_conv1d

# Automatically creates cheap path and estimates FLOPs
gate = create_lazy_conv1d(
    in_channels=32,
    out_channels=64,
    kernel_size=7,
    padding=3,
    initial_p=0.5,
    initial_a=1.0,
    seq_len=100,
    name="encoder_block_1"
)
```

### Managing Multiple Gates

```python
from lazyai import LazyGate, LazyGateConfig

gates = [
    LazyGate(layer1, initial_p=0.3, name="block1"),
    LazyGate(layer2, initial_p=0.5, name="block2"),
    LazyGate(layer3, initial_p=0.7, name="block3"),
]

config = LazyGateConfig(gates)

# Set all gates from an evolved genome
evolved_genome = [(0.8, 0.6), (0.5, 0.9), (0.3, 1.0)]
config.set_genome(evolved_genome)

# Print summary
print(config.summary())
```

Output:
```
┌─────────────────┬───────┬───────┬────────────┬───────────┐
│ Gate            │   p   │   a   │ Eff.Comp.  │ Skip Rate │
├─────────────────┼───────┼───────┼────────────┼───────────┤
│ block1          │ 0.800 │ 0.600 │      12.0% │     80.0% │
│ block2          │ 0.500 │ 0.900 │      45.0% │     50.0% │
│ block3          │ 0.300 │ 1.000 │      70.0% │     30.0% │
├─────────────────┼───────┼───────┼────────────┼───────────┤
│ TOTAL           │       │       │      42.3% │     53.3% │
└─────────────────┴───────┴───────┴────────────┴───────────┘
```

---

## Design Rationale

### Why Two Parameters?

The PAD framework argues that every computational strategy can be characterized by:

1. **How lazy?** — What fraction of computation to skip (P)
2. **How good?** — What quality level to aim for when working (A)

These are orthogonal axes:
- High P, High A: "PhD student" — procrastinates but delivers brilliance
- Low P, Low A: "Reflex arc" — always on but minimal processing
- High P, Low A: "Lazy student" — skips class and barely passes
- Low P, High A: "GPT-4" — always on, maximum effort

### Why Dual-Path?

The dual-path architecture enables:

1. **Dimension handling**: Cheap path can project between different dimensions
2. **Gradient flow**: Both paths contribute, enabling fine-tuning
3. **Residual blending**: Ambition $a$ creates a learned residual connection
4. **Zero-cost skip**: When dimensions match, cheap = identity = free

### Why Bernoulli Sampling?

Stochastic gating (vs. learned threshold) provides:

1. **Exploration**: Random samples let evolution discover good configurations
2. **Noise tolerance**: Model learns to work with probabilistic computation
3. **Batch diversity**: Different samples in a batch take different paths

### Why Evolution Over Gradients?

The gate parameters $(p, a)$ are evolved by CMA-ES rather than trained:

1. **Non-differentiable objective**: FLOPs savings don't have gradients
2. **Quality floor constraint**: "Pass the exam" is a hard constraint
3. **Discrete decisions**: Gradient-based methods struggle with discrete choices
4. **Laziness reward**: Exponential reward for $p$ is non-standard

---

## Next Steps

LazyGate is the atomic unit. The full LazyAI stack:

1. ✅ **LazyGate**: Dual-path stochastic skip (this module)
2. ⏳ **LazyWrapper**: Automatically wraps any `nn.Module` with gates
3. ⏳ **LazyTuner**: CMA-ES evolutionary optimization of gate genomes
