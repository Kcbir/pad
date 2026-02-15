# LazyAI — Complete Technical Reference

> **Evolutionary Gating for Computation-Optimal Neural Inference**
>
> Version 0.2.0 · MIT License

---

## Table of Contents

1. [Overview](#1-overview)
2. [The PAD Framework](#2-the-pad-framework)
   - 2.1 [Procrastination & Ambition](#21-procrastination--ambition)
   - 2.2 [Forward Pass Equation](#22-forward-pass-equation)
   - 2.3 [Effective Compute](#23-effective-compute)
3. [Architecture](#3-architecture)
4. [LazyGate](#4-lazygate)
   - 4.1 [Concept](#41-concept)
   - 4.2 [Gumbel-Softmax Sampling](#42-gumbel-softmax-sampling)
   - 4.3 [FLOP Estimation](#43-flop-estimation)
   - 4.4 [API Reference](#44-api-reference)
5. [LazyWrapper](#5-lazywrapper)
   - 5.1 [Automatic Gate Injection](#51-automatic-gate-injection)
   - 5.2 [Shape Inference & Cheap Paths](#52-shape-inference--cheap-paths)
   - 5.3 [Recursion Guard](#53-recursion-guard)
   - 5.4 [Genome Interface](#54-genome-interface)
   - 5.5 [API Reference](#55-api-reference)
6. [LazyTuner](#6-lazytuner)
   - 6.1 [CMA-ES Algorithm](#61-cma-es-algorithm)
   - 6.2 [PAD Fitness Function](#62-pad-fitness-function)
   - 6.3 [Evolution Parameters](#63-evolution-parameters)
   - 6.4 [API Reference](#64-api-reference)
7. [Mathematical Appendix](#7-mathematical-appendix)
   - 7.1 [Complete CMA-ES Derivation](#71-complete-cma-es-derivation)
   - 7.2 [Covariance Matrix Adaptation](#72-covariance-matrix-adaptation)
   - 7.3 [Step-Size Control (CSA)](#73-step-size-control-csa)
8. [Quick Start](#8-quick-start)
9. [Examples](#9-examples)
   - 9.1 [Single Gate](#91-single-gate)
   - 9.2 [Full Pipeline](#92-full-pipeline)
   - 9.3 [Anomaly Detection Demo](#93-anomaly-detection-demo)
10. [File Structure](#10-file-structure)
11. [Testing](#11-testing)

---

## 1. Overview

LazyAI makes neural networks lazy. Instead of running every layer at full precision on every input, it wraps expensive layers in **stochastic gates** that probabilistically skip computation or reduce its intensity. An evolutionary optimizer (CMA-ES) then discovers the optimal laziness configuration — maximizing compute savings while maintaining task quality.

**Core thesis:** Most neural network layers do redundant work most of the time. By introducing two parameters per layer — how often to skip (procrastination) and how much effort to invest when not skipping (ambition) — we can dramatically reduce inference cost.

**Three components:**

| Component | Purpose | Key Idea |
|-----------|---------|----------|
| `LazyGate` | Stochastic skip mechanism | Dual-path: expensive vs cheap, blended by Bernoulli gate |
| `LazyWrapper` | Automatic gate injection | Hook-based wrapping of any `nn.Module` |
| `LazyTuner` | Evolutionary optimization | CMA-ES finds optimal P,A parameters across all gates |

**Dependencies:** PyTorch ≥ 2.0, NumPy

---

## 2. The PAD Framework

### 2.1 Procrastination & Ambition

Every gated layer is characterized by two scalar parameters:

$$P \in [0, 1] \quad \text{(Procrastination — probability of skipping expensive compute)}$$

$$A \in [0, 1] \quad \text{(Ambition — blend factor when not skipping)}$$

**Interpretation:**
- **P = 0:** Always compute (no laziness)
- **P = 1:** Always skip (maximum laziness)
- **A = 1:** Use full expensive path output when active
- **A = 0:** Even when "active," rely entirely on cheap path

### 2.2 Forward Pass Equation

The core forward pass of a LazyGate:

$$y = z \cdot a \cdot f(x) + (1 - z \cdot a) \cdot \text{skip}(x)$$

where:
- $f(x)$ is the **expensive path** (original layer computation)
- $\text{skip}(x)$ is the **cheap path** (identity, 1×1 conv, or simple projection)
- $z \sim \text{Bernoulli}(1 - p)$ is the stochastic gate activation
- $a = A$ is the ambition scalar

When $z = 0$ (skip): output = $\text{skip}(x)$
When $z = 1$ (compute): output = $a \cdot f(x) + (1 - a) \cdot \text{skip}(x)$

### 2.3 Effective Compute

The expected fraction of expensive computation utilized:

$$\text{Effective Compute} = (1 - P) \times A$$

This metric directly maps to FLOP reduction. If $P = 0.7$ and $A = 0.8$:

$$\text{Effective Compute} = 0.3 \times 0.8 = 0.24 \quad \text{(76\% reduction)}$$

For a full model with $N$ gates, the **global** metrics are:

$$\bar{P} = \frac{1}{N} \sum_{i=1}^{N} P_i \qquad \bar{A} = \frac{1}{N} \sum_{i=1}^{N} (1 - P_i) \cdot A_i$$

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Input Tensor x                         │
│                         │                                 │
│              ┌──────────┴──────────┐                     │
│              ▼                     ▼                     │
│    ┌──────────────────┐  ┌──────────────────┐           │
│    │  Expensive Path  │  │   Cheap Path     │           │
│    │    f(x)          │  │   skip(x)        │           │
│    │  (original op)   │  │  (identity/1x1)  │           │
│    └────────┬─────────┘  └────────┬─────────┘           │
│             │                     │                      │
│             ▼                     ▼                      │
│    ┌────────────────────────────────────────┐            │
│    │ z ~ Bernoulli(1 - P)                   │            │
│    │                                         │            │
│    │ y = z·A·f(x) + (1 - z·A)·skip(x)      │            │
│    └────────────────┬───────────────────────┘            │
│                     │                                     │
│                     ▼                                     │
│              Output Tensor y                              │
└──────────────────────────────────────────────────────────┘
```

**Full Pipeline:**

```
Model → LazyWrapper.wrap() → [LazyGate₁, ..., LazyGateₙ] → genome vector
                                                                    │
         LazyTuner (CMA-ES) ← fitness evaluation ← quality metric ←┘
                │
                ▼
         Optimized genome → set_flat_genome() → Lazy inference
```

---

## 4. LazyGate

### 4.1 Concept

`LazyGate` is the atomic building block. It wraps a single expensive operation with a stochastic dual-path mechanism.

**Key behaviors:**
- During **training**: uses Gumbel-Softmax for differentiable sampling (straight-through estimator)
- During **evaluation**: uses hard Bernoulli sampling (true stochastic skip)
- In **deterministic mode**: uses soft blending without sampling for reproducible outputs

### 4.2 Gumbel-Softmax Sampling

To make the discrete Bernoulli gate differentiable during training, we use the Gumbel-Softmax trick:

$$g_0, g_1 \sim -\log(-\log(U)), \quad U \sim \text{Uniform}(0, 1)$$

$$z_{\text{soft}} = \sigma\!\left(\frac{\ell + g_1 - g_0}{\tau}\right)$$

where:
- $\ell = \log\!\frac{1-p}{p}$ is the log-odds (logit) of firing
- $\tau$ is the temperature (lower ↓ = more discrete)
- $\sigma$ is the sigmoid function

**Straight-through estimator:** In training, the forward pass uses the hard sample $z_{\text{hard}} = \mathbb{1}[z_{\text{soft}} > 0.5]$, but gradients flow through $z_{\text{soft}}$.

### 4.3 FLOP Estimation

LazyAI estimates FLOPs for gated layers to quantify savings:

**Conv1d:**
$$\text{FLOPs} = 2 \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot K \cdot L$$

where $C_{\text{in}} =$ input channels, $C_{\text{out}} =$ output channels, $K =$ kernel size, $L =$ sequence length.

**Linear:**
$$\text{FLOPs} = 2 \cdot F_{\text{in}} \cdot F_{\text{out}} \cdot B$$

where $F_{\text{in}} =$ input features, $F_{\text{out}} =$ output features, $B =$ batch size.

### 4.4 API Reference

#### `LazyGate(expensive_fn, cheap_fn=None, initial_p=0.5, initial_a=0.9, min_p=0.0, max_p=0.95, temperature=1.0, name="")`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expensive_fn` | `nn.Module` | required | The original expensive operation |
| `cheap_fn` | `nn.Module` | `None` | Cheap alternative (defaults to `nn.Identity()`) |
| `initial_p` | `float` | `0.5` | Initial procrastination |
| `initial_a` | `float` | `0.9` | Initial ambition |
| `min_p` / `max_p` | `float` | `0.0` / `0.95` | Clamp range for P |
| `temperature` | `float` | `1.0` | Gumbel-Softmax temperature |
| `name` | `str` | `""` | Human-readable gate name |

**Properties:**
- `.p → float` — Current procrastination value
- `.a → float` — Current ambition value
- `.effective_compute → float` — `(1-p) × a`
- `.stats → GateStatistics` — Call/skip/fire counts
- `.deterministic → bool` — Toggle deterministic mode

**Methods:**
- `.forward(x) → Tensor` — Gated forward pass
- `.get_genome() → Tensor` — Returns `[raw_p, raw_a]` (logit-space)
- `.set_genome(genome)` — Sets from `[raw_p, raw_a]`
- `.reset_stats()` — Zero all counters

#### `GateStatistics`

Tracks per-gate runtime statistics:
- `.total_calls`, `.skipped`, `.fired`
- `.skip_rate → float`
- `.fire_rate → float`

#### `LazyGateConfig(n_gates, names=None)`

Manages genome vectors across multiple gates:
- `.encode(gates) → Tensor` — Flatten all gates into single genome
- `.decode(genome, gates)` — Write genome back to gates

#### `gumbel_softmax_sample(logit, temperature=1.0) → Tensor`

Standalone Gumbel-Softmax binary sample.

#### `create_lazy_conv1d(in_ch, out_ch, kernel_size, **kwargs) → LazyGate`

Factory: creates Conv1d with matching cheap path (1×1 conv if dimensions differ, identity otherwise).

---

## 5. LazyWrapper

### 5.1 Automatic Gate Injection

`LazyWrapper` takes any `nn.Module` and automatically wraps its gatable layers with `LazyGate` instances using **forward hooks** (no model surgery required).

**Gatable layers** (wrapped by default):
- `nn.Conv1d`, `nn.Conv2d`
- `nn.Linear`
- `nn.MultiheadAttention`
- `nn.TransformerEncoderLayer`

**Non-gatable layers** (always skipped):
- `nn.BatchNorm1d/2d`, `nn.LayerNorm`, `nn.GroupNorm`
- `nn.Dropout`, `nn.Embedding`
- `nn.ReLU`, `nn.GELU`, `nn.SiLU`
- Pooling, padding, flatten layers

### 5.2 Shape Inference & Cheap Paths

`LazyWrapper` runs a **probe forward pass** with `sample_input` to detect the input/output shapes of each gatable layer. Based on dimension analysis:

| Scenario | Cheap Path Created |
|----------|-------------------|
| Same dimensions | `nn.Identity()` |
| Different channels (Conv1d) | `nn.Conv1d(C_in, C_out, 1)` (1×1 conv) |
| Different channels (Conv2d) | `nn.Conv2d(C_in, C_out, 1)` (1×1 conv) |
| Different features (Linear) | `nn.Linear(F_in, F_out)` |

### 5.3 Recursion Guard

When a forward hook intercepts a layer call and routes through a `LazyGate`, the gate internally calls `expensive_fn(x)` — which is the original layer, which would trigger the hook again, causing infinite recursion.

**Solution:** A `_in_hook` flag prevents re-entry:

```python
def hook(module, input, output):
    if self._in_hook:
        return output          # Already inside gate → pass through
    self._in_hook = True
    result = gate(input[0])    # Gate calls expensive_fn → triggers hook → blocked
    self._in_hook = False
    return result
```

### 5.4 Genome Interface

The genome is a flat vector of size `2 × num_gates` (two parameters per gate: raw P and raw A in logit space):

$$\text{genome} = [\underbrace{p_1^{\text{raw}}, a_1^{\text{raw}}}_{\text{Gate 1}}, \underbrace{p_2^{\text{raw}}, a_2^{\text{raw}}}_{\text{Gate 2}}, \ldots]$$

Conversion to/from `[0, 1]` range uses sigmoid:

$$P_i = \sigma(p_i^{\text{raw}}) \qquad A_i = \sigma(a_i^{\text{raw}})$$

### 5.5 API Reference

#### `LazyWrapper(model, sample_input, initial_p=0.5, initial_a=0.9, temperature=1.0)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | required | Model to wrap |
| `sample_input` | `Tensor` | required | Sample input for shape inference |
| `initial_p` | `float` | `0.5` | Default P for all gates |
| `initial_a` | `float` | `0.9` | Default A for all gates |
| `temperature` | `float` | `1.0` | Gumbel-Softmax temperature |

**Properties:**
- `.num_gates → int` — Number of injected gates
- `.genome_size → int` — Length of flat genome vector
- `.gates → List[LazyGate]` — All gate instances
- `.procrastination → float` — Mean P across gates ($\bar{P}$)
- `.ambition → float` — Weighted mean A ($\bar{A}$)
- `.effective_compute → float` — Mean $(1 - P_i) \times A_i$
- `.flops_saved_ratio → float` — Estimated FLOP reduction

**Methods:**
- `.forward(x) → Tensor` — Forward pass through gated model
- `.get_flat_genome() → Tensor` — Get all gate parameters as flat vector
- `.set_flat_genome(genome)` — Set all gate parameters from flat vector
- `.reset_stats()` — Reset all gate statistics
- `.summary() → str` — Human-readable gate status table

#### `DualPathBlock(expensive, cheap, initial_p, initial_a)`

Pre-built block with explicit expensive and cheap paths. Drop-in replacement for any `nn.Module`.

#### `create_lazy_conv1d_block(in_ch, out_ch, kernel_size, **kwargs) → DualPathBlock`

Factory for Conv1d dual-path blocks.

---

## 6. LazyTuner

### 6.1 CMA-ES Algorithm

The **Covariance Matrix Adaptation Evolution Strategy** (CMA-ES) is a derivative-free optimizer ideal for the low-dimensional PAD genome (typically 4–20 parameters). It maintains a multivariate normal search distribution and adapts both its mean and covariance to drive the search toward high-fitness regions.

**Why CMA-ES for PAD:**
- PAD genomes are small (2 params per gate × few gates)
- Quality metrics (F1, accuracy) are non-differentiable
- CMA-ES handles rugged, noisy fitness landscapes naturally
- No gradient computation required → zero training overhead

**Algorithm outline per generation:**

1. **Sample** λ candidates from $\mathcal{N}(\mathbf{m}, \sigma^2 \mathbf{C})$
2. **Evaluate** fitness for each candidate
3. **Rank** candidates by fitness (descending)
4. **Update** mean $\mathbf{m}$ toward best candidates
5. **Update** evolution paths $\mathbf{p}_\sigma$, $\mathbf{p}_c$
6. **Update** covariance $\mathbf{C}$ via rank-1 and rank-μ updates
7. **Adapt** step-size $\sigma$ via Cumulative Step-size Adaptation (CSA)

### 6.2 PAD Fitness Function

$$J = Q \cdot \exp(\beta \cdot \bar{P}) \cdot (1 + \eta \cdot \bar{A})$$

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| $Q$ | Task quality metric (F1, accuracy, etc.) | `[0, 1]` |
| $\bar{P}$ | Mean procrastination across gates | `[0, 1]` |
| $\bar{A}$ | Weighted mean ambition | `[0, 1]` |
| $\beta$ | Laziness reward strength | `1.0 – 3.0` |
| $\eta$ | Ambition bonus coefficient | `0.1 – 0.5` |
| $Q_{\min}$ | Quality floor (fitness = 0 if $Q < Q_{\min}$) | task-dependent |

**Design rationale:**
- $\exp(\beta \cdot \bar{P})$: Exponential reward for skipping more — creates strong pressure toward laziness
- $(1 + \eta \cdot \bar{A})$: Linear bonus for high ambition — rewards quality when active
- $Q_{\min}$ floor: Hard constraint preventing catastrophic quality collapse

### 6.3 Evolution Parameters

All parameters follow Hansen's canonical CMA-ES formulation (2016):

**Weights:**
$$w_i = \ln(\mu + 0.5) - \ln(i + 1), \quad i = 0, \ldots, \mu - 1$$

Normalized so $\sum w_i = 1$. Effective selection mass:

$$\mu_{\text{eff}} = \frac{1}{\sum w_i^2}$$

**Adaptation rates:**

$$c_\sigma = \frac{\mu_{\text{eff}} + 2}{n + \mu_{\text{eff}} + 5}$$

$$d_\sigma = 1 + 2 \max\!\left(0, \sqrt{\frac{\mu_{\text{eff}} - 1}{n + 1}} - 1\right) + c_\sigma$$

$$c_c = \frac{4 + \mu_{\text{eff}}/n}{n + 4 + 2\mu_{\text{eff}}/n}$$

$$c_1 = \frac{2}{(n + 1.3)^2 + \mu_{\text{eff}}}$$

$$c_\mu = \min\!\left(1 - c_1, \; \frac{2(\mu_{\text{eff}} - 2 + 1/\mu_{\text{eff}})}{(n+2)^2 + \mu_{\text{eff}}}\right)$$

$$\chi_n = \sqrt{n}\left(1 - \frac{1}{4n} + \frac{1}{21n^2}\right)$$

### 6.4 API Reference

#### `LazyTuner(genome_size, fitness_fn, config=None, initial_genome=None, verbose=False)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `genome_size` | `int` | required | Dimension of genome vector |
| `fitness_fn` | `Callable` | required | `genome → float` fitness function |
| `config` | `LazyTunerConfig` | `None` | Hyperparameters |
| `initial_genome` | `Tensor` | `None` | Starting point |
| `verbose` | `bool` | `False` | Print per-generation stats |

**Methods:**
- `.evolve() → (Tensor, List[EvolutionStats])` — Run full CMA-ES. Returns best genome and per-generation statistics.
- `.step() → EvolutionStats` — Single generation step.
- `.best_genome → Tensor` — Current best solution.
- `.best_fitness → float` — Current best fitness value.
- `.generation → int` — Current generation counter.

#### `LazyTunerConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `beta` | `1.0` | Laziness reward coefficient |
| `eta` | `0.2` | Ambition bonus coefficient |
| `q_min` | `0.5` | Quality floor |
| `sigma` | `0.3` | Initial step-size |
| `population_size` | `None` | λ (auto: `4 + ⌊3 ln n⌋`) |
| `max_generations` | `50` | Evolution budget |

#### `EvolutionStats`

Per-generation statistics:
- `.generation`, `.best_fitness`, `.mean_fitness`, `.sigma`
- `.best_genome`, `.mean_p`, `.mean_a`

#### Utilities

```python
compute_pad_fitness(quality, mean_p, mean_a, beta, eta, q_min) → float
genome_to_pa(genome) → List[Tuple[float, float]]    # [(p₁, a₁), (p₂, a₂), ...]
genome_summary(genome) → Tuple[float, float]         # (mean_p, weighted_mean_a)
create_quick_tuner(wrapper, quality_fn, beta, eta) → LazyTuner
```

#### `PADFitnessFactory(wrapper, quality_fn, beta, eta, q_min)`

Creates fitness functions that automatically:
1. Set genome on wrapper
2. Evaluate quality
3. Compute PAD fitness

```python
factory = PADFitnessFactory(lazy_model, quality_fn, beta=2.0, eta=0.3, q_min=0.6)
tuner = LazyTuner(lazy_model.genome_size, factory, config)
```

---

## 7. Mathematical Appendix

### 7.1 Complete CMA-ES Derivation

Given genome dimension $n$, population size $\lambda$, parent count $\mu = \lfloor \lambda / 2 \rfloor$:

**Search distribution at generation $g$:**

$$\mathbf{x}_k^{(g+1)} \sim \mathbf{m}^{(g)} + \sigma^{(g)} \cdot \mathcal{N}(\mathbf{0}, \mathbf{C}^{(g)}), \quad k = 1, \ldots, \lambda$$

**Mean update** (weighted recombination of $\mu$ best):

$$\mathbf{m}^{(g+1)} = \sum_{i=1}^{\mu} w_i \cdot \mathbf{x}_{i:\lambda}^{(g+1)}$$

where $\mathbf{x}_{i:\lambda}$ is the $i$-th best candidate by fitness.

### 7.2 Covariance Matrix Adaptation

**Evolution path** (cumulation):

$$\mathbf{p}_c^{(g+1)} = (1 - c_c) \cdot \mathbf{p}_c^{(g)} + h_\sigma \sqrt{c_c(2 - c_c) \cdot \mu_{\text{eff}}} \cdot \frac{\mathbf{m}^{(g+1)} - \mathbf{m}^{(g)}}{\sigma^{(g)}}$$

where $h_\sigma$ is a stalling indicator (prevents $\|\mathbf{p}_c\|$ from growing when $\sigma$ is too small).

**Covariance update** (rank-1 + rank-μ):

$$\mathbf{C}^{(g+1)} = (1 - c_1 - c_\mu) \cdot \mathbf{C}^{(g)} + c_1 \cdot \mathbf{p}_c \cdot \mathbf{p}_c^\top + c_\mu \sum_{i=1}^{\mu} w_i \cdot \mathbf{y}_{i:\lambda} \cdot \mathbf{y}_{i:\lambda}^\top$$

where $\mathbf{y}_{i:\lambda} = (\mathbf{x}_{i:\lambda} - \mathbf{m}^{(g)}) / \sigma^{(g)}$.

### 7.3 Step-Size Control (CSA)

**Conjugate evolution path:**

$$\mathbf{p}_\sigma^{(g+1)} = (1 - c_\sigma) \cdot \mathbf{p}_\sigma^{(g)} + \sqrt{c_\sigma(2 - c_\sigma) \cdot \mu_{\text{eff}}} \cdot \mathbf{C}^{-1/2} \cdot \frac{\mathbf{m}^{(g+1)} - \mathbf{m}^{(g)}}{\sigma^{(g)}}$$

**Step-size adaptation:**

$$\sigma^{(g+1)} = \sigma^{(g)} \cdot \exp\!\left(\frac{c_\sigma}{d_\sigma} \left(\frac{\|\mathbf{p}_\sigma^{(g+1)}\|}{\chi_n} - 1\right)\right)$$

**Intuition:** If $\|\mathbf{p}_\sigma\| > \chi_n$, steps are correlated (going somewhere) → increase $\sigma$. If $\|\mathbf{p}_\sigma\| < \chi_n$, steps are anti-correlated (oscillating) → decrease $\sigma$.

---

## 8. Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd "Lazy AI"

# No pip install needed — import directly
python -c "from lazyai import LazyGate; print('OK')"
```

### Minimal Example

```python
import torch
import torch.nn as nn
from lazyai import LazyGate, LazyWrapper, LazyTuner, LazyTunerConfig

# 1. Define your model
model = nn.Sequential(
    nn.Conv1d(32, 64, 7, padding=3),
    nn.ReLU(),
    nn.Conv1d(64, 64, 5, padding=2),
    nn.ReLU(),
    nn.Conv1d(64, 32, 3, padding=1),
)

# 2. Wrap with lazy gates
sample = torch.randn(1, 32, 100)
lazy = LazyWrapper(model, sample_input=sample, initial_p=0.5, initial_a=0.9)
print(f"Gates: {lazy.num_gates}, Genome: {lazy.genome_size}")

# 3. Define quality metric
def quality_fn(wrapper):
    x = torch.randn(8, 32, 100)
    with torch.no_grad():
        y = wrapper(x)
    return 1.0 - y.abs().mean().item()  # Placeholder metric

# 4. Evolve optimal laziness
config = LazyTunerConfig(beta=2.0, eta=0.3, q_min=0.3, max_generations=20)

def fitness(genome):
    lazy.set_flat_genome(genome)
    q = quality_fn(lazy)
    from lazyai import compute_pad_fitness, genome_summary
    mp, ma = genome_summary(genome)
    return compute_pad_fitness(q, mp, ma, config.beta, config.eta, config.q_min)

tuner = LazyTuner(lazy.genome_size, fitness, config, verbose=True)
best, history = tuner.evolve()

# 5. Apply and measure
lazy.set_flat_genome(best)
print(lazy.summary())
print(f"Effective compute: {lazy.effective_compute:.1%}")
```

---

## 9. Examples

### 9.1 Single Gate

```python
from lazyai import LazyGate, create_lazy_conv1d

# Manual gate construction
conv = nn.Conv1d(64, 64, 7, padding=3)
gate = LazyGate(conv, initial_p=0.7, initial_a=0.8, name="conv_block")

x = torch.randn(4, 64, 100)
y = gate(x)  # Same shape: (4, 64, 100)

print(f"P={gate.p:.3f}, A={gate.a:.3f}, Eff={(1-gate.p)*gate.a:.1%}")
print(f"Skipped: {gate.stats.skipped}/{gate.stats.total_calls}")

# Factory function (auto cheap path)
gate2 = create_lazy_conv1d(32, 64, kernel_size=7, padding=3, initial_p=0.5)
```

### 9.2 Full Pipeline

```python
from lazyai import LazyWrapper, create_quick_tuner

model = YourModel()
sample = torch.randn(1, *input_shape)
lazy = LazyWrapper(model, sample)

def quality(wrapper):
    """Evaluate on your validation set."""
    return compute_your_metric(wrapper, val_loader)

tuner = create_quick_tuner(lazy, quality, beta=2.0, eta=0.3)
best, history = tuner.evolve()
lazy.set_flat_genome(best)

# Now use lazy(x) for inference — compute savings are automatic
```

### 9.3 Anomaly Detection Demo

A complete working example is provided in `demo_anomaly.py`:

```
python demo_anomaly.py
```

This demo:
1. Generates 300 synthetic multivariate time-series samples with injected anomalies (spikes, level shifts, noise bursts)
2. Trains a 4-layer 1D-CNN autoencoder (TinyAutoencoder)
3. Wraps it with `LazyWrapper` (auto-detects Conv1d layers)
4. Runs CMA-ES evolution (15 generations, λ=15) to find optimal P,A values
5. Reports F1 score retention vs compute savings

**Expected output:** ~80–95% quality retained with 30–60% compute saved, depending on the random seed and anomaly distribution.

---

## 10. File Structure

```
Lazy AI/
├── lazyai/
│   ├── __init__.py              # v0.2.0 package entry, full public API
│   └── core/
│       ├── __init__.py          # Module exports
│       ├── gate.py              # LazyGate, GateStatistics, Gumbel-Softmax (~496 lines)
│       ├── wrapper.py           # LazyWrapper, DualPathBlock, hook system (~572 lines)
│       └── tuner.py             # LazyTuner, CMA-ES, PAD fitness (~752 lines)
├── docs/
│   ├── LAZY_GATE.md             # Gate API documentation
│   └── LAZY_WRAPPER_TUNER.md    # Wrapper & tuner API + math appendix
├── test_lazyai.py               # Comprehensive test suite (4 suites, 28 tests)
├── test_lazy_gate.py            # Original gate-only tests (7 tests)
├── demo_anomaly.py              # Self-contained anomaly detection demo
├── lazy.md                      # This document
├── README.md                    # Project README
├── CRITICAL_REVIEW_AND_ROADMAP.md
└── THE_PROCRASTINATION_AMBITION_DUALITY.md
```

**Total library size:** ~1,820 lines of implementation code across 3 modules.

---

## 11. Testing

```bash
# Run all tests
python -m pytest test_lazyai.py -v

# Or with unittest
python -m unittest test_lazyai -v
```

**Test suites:**

| Suite | Tests | What It Validates |
|-------|-------|-------------------|
| `TestLazyGate` | 6 | Gate forward, Gumbel-Softmax, statistics, genome I/O, deterministic mode |
| `TestLazyWrapper` | 7 | Auto-wrapping, shape inference, genome interface, hook recursion, summary |
| `TestLazyTuner` | 7 | CMA-ES initialization, single step, full evolution, fitness computation, factory |
| `TestIntegration` | 8 | End-to-end pipeline, quality retention, FLOP savings, genome round-trip |

**Most recent test results (all passing):**

```
LazyGate:    ✓ PASSED (6 tests)
LazyWrapper: ✓ PASSED (7 tests, 72.6% FLOPs saved)
LazyTuner:   ✓ PASSED (7 tests, fitness 1.3562 → 2.4298)
Integration: ✓ PASSED (8 tests, 99.1% theoretical / 89.6% actual savings)

Total: 4/4 suites passed, 28/28 tests passed
```

---

*LazyAI v0.2.0 — Because the best computation is the one you don't do.*
