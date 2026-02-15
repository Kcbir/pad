"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   █   ████ ████ █   █   ████ █                                               ║
║   █   █  █   █  █   █   █  █ █                                               ║
║   █   ████  █   █ █ █   ████ █                                               ║
║   █   █  █ █     █ █    █  █ █                                               ║
║   ███ █  █ ████   █     █  █ █                                               ║
║                                                                               ║
║   Evolutionary Gating for Computation-Optimal Neural Inference               ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

A framework for making neural networks lazy — computing only when necessary,
at only the precision required, to achieve maximum efficiency with minimal
quality loss.

Core Components:
    LazyGate:    Dual-path stochastic skip mechanism
    LazyWrapper: Automatically wrap arbitrary models with gates
    LazyTuner:   CMA-ES evolutionary optimization of gate parameters

═══════════════════════════════════════════════════════════════════════════════
                    MATHEMATICAL FOUNDATION: PAD FRAMEWORK
═══════════════════════════════════════════════════════════════════════════════

The Procrastination-Ambition Duality (PAD) characterizes computation:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  P (Procrastination): ∈ [0, 1] — probability of skipping compute   │
    │  A (Ambition):        ∈ [0, 1] — blend factor when not skipping    │
    │                                                                     │
    │  Effective Compute = (1 - P) × A                                   │
    │                                                                     │
    │  Forward:  y = z·a·f(x) + (1 - z·a)·skip(x)                        │
    │            where z ~ Bernoulli(1 - p)                              │
    │                                                                     │
    │  Fitness:  J = Q · exp(β·P) · (1 + η·A)                            │
    └─────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                              QUICK START
═══════════════════════════════════════════════════════════════════════════════

    >>> import torch.nn as nn
    >>> from lazyai import LazyGate, LazyWrapper, LazyTuner
    
    # 1. Basic Gate
    >>> gate = LazyGate(nn.Conv1d(64, 64, 7), initial_p=0.7, initial_a=0.8)
    >>> print(gate.effective_compute)  # 0.24 (76% reduction!)
    
    # 2. Wrap Entire Model
    >>> model = YourNetwork()
    >>> lazy_model = LazyWrapper(model, sample_input=torch.randn(1, 32, 100))
    
    # 3. Evolve Optimal Parameters
    >>> tuner = LazyTuner(lazy_model.genome_size, fitness_fn, config)
    >>> best_genome, history = tuner.evolve()
    >>> lazy_model.set_flat_genome(best_genome)

Author: LazyAI Research
License: MIT
Version: 0.2.0
"""

__version__ = "0.2.0"
__author__ = "LazyAI Research"

# ═══════════════════════════════════════════════════════════════════════════════
#                              PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

from .core import (
    # Gate - stochastic skip mechanism
    LazyGate,
    LazyGateConfig,
    GateStatistics,
    gumbel_softmax_sample,
    create_lazy_conv1d,
    estimate_conv1d_flops,
    estimate_linear_flops,
    # Wrapper - automatic gate injection
    LazyWrapper,
    DualPathBlock,
    create_lazy_conv1d_block,
    GATABLE_LAYERS,
    NON_GATABLE_LAYERS,
    # Tuner - CMA-ES evolution
    LazyTuner,
    LazyTunerConfig,
    EvolutionStats,
    PADFitnessFactory,
    compute_pad_fitness,
    genome_to_pa,
    genome_summary,
    create_quick_tuner,
)

__all__ = [
    # Gate
    "LazyGate",
    "LazyGateConfig",
    "GateStatistics",
    "gumbel_softmax_sample",
    "create_lazy_conv1d",
    "estimate_conv1d_flops",
    "estimate_linear_flops",
    # Wrapper
    "LazyWrapper",
    "DualPathBlock",
    "create_lazy_conv1d_block",
    "GATABLE_LAYERS",
    "NON_GATABLE_LAYERS",
    # Tuner
    "LazyTuner",
    "LazyTunerConfig",
    "EvolutionStats",
    "PADFitnessFactory",
    "compute_pad_fitness",
    "genome_to_pa",
    "genome_summary",
    "create_quick_tuner",
]

