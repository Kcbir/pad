"""
LazyAI Core Module

The fundamental building blocks of laziness-optimal computation.

Components:
    Gate:    LazyGate, LazyGateConfig, GateStatistics
    Wrapper: LazyWrapper, DualPathBlock
    Tuner:   LazyTuner, LazyTunerConfig, PADFitnessFactory
"""

# Gate module - stochastic skip mechanism
from .gate import (
    LazyGate,
    LazyGateConfig,
    GateStatistics,
    GateGenome,
    gumbel_softmax_sample,
    create_lazy_conv1d,
    estimate_conv1d_flops,
    estimate_linear_flops,
)

# Wrapper module - automatic gate injection
from .wrapper import (
    LazyWrapper,
    DualPathBlock,
    create_lazy_conv1d_block,
    GATABLE_LAYERS,
    NON_GATABLE_LAYERS,
)

# Tuner module - CMA-ES evolutionary optimization
from .tuner import (
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
    "GateGenome",
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

