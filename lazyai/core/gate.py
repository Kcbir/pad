"""
LazyGate: The Dual-Path Stochastic Skip Mechanism

Mathematical Foundation:
━━━━━━━━━━━━━━━━━━━━━━━━
Given input x, layer function f, skip probability p ∈ [0,1], and ambition a ∈ [0,1]:

    z ~ Bernoulli(1 - p)        # Gate decision: 1 = compute, 0 = skip
    
    y = z · a · f(x) + (1 - z·a) · skip(x)

Where skip(x) is the cheap path (identity or learned projection).

The effective computation fraction is: E[z·a] = (1-p)·a
The expected FLOP savings: 1 - (1-p)·a

This formulation elegantly unifies:
- p controls WHEN to compute (temporal sparsity / procrastination)
- a controls HOW MUCH to commit when computing (quality target / ambition)

Author: LazyAI
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GateStatistics:
    """Running statistics for a LazyGate instance."""
    
    compute_count: int = 0      # Times the expensive path was taken
    skip_count: int = 0         # Times the cheap path was taken
    total_flops_full: float = 0 # FLOPs if always computed
    total_flops_actual: float = 0  # FLOPs actually used
    
    def record(self, computed: bool, flops_expensive: float, flops_cheap: float, 
               ambition: float) -> None:
        """Record a single forward pass decision."""
        if computed:
            self.compute_count += 1
            # Actual FLOPs = cheap always runs + expensive scaled by ambition
            self.total_flops_actual += flops_cheap + ambition * flops_expensive
        else:
            self.skip_count += 1
            self.total_flops_actual += flops_cheap
        
        self.total_flops_full += flops_cheap + flops_expensive
    
    @property
    def skip_ratio(self) -> float:
        """Fraction of times the gate chose to skip."""
        total = self.compute_count + self.skip_count
        return self.skip_count / total if total > 0 else 0.0
    
    @property
    def flops_saved_ratio(self) -> float:
        """Fraction of FLOPs saved compared to always-compute."""
        if self.total_flops_full == 0:
            return 0.0
        return 1 - (self.total_flops_actual / self.total_flops_full)
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.compute_count = 0
        self.skip_count = 0
        self.total_flops_full = 0
        self.total_flops_actual = 0


def gumbel_softmax_sample(logits: torch.Tensor, temperature: float = 1.0, 
                          hard: bool = False) -> torch.Tensor:
    """
    Gumbel-Softmax reparameterization for differentiable discrete sampling.
    
    For binary case with logit l = log(p / (1-p)):
        z_soft = σ((l + g₁ - g₀) / τ)
    
    Where gᵢ ~ Gumbel(0,1) = -log(-log(Uniform(0,1)))
    
    Args:
        logits: Log-odds of computing (positive = more likely to compute)
        temperature: τ controls smoothness. τ→0 becomes hard, τ→∞ becomes uniform.
        hard: If True, use straight-through estimator (hard in forward, soft in backward)
    
    Returns:
        Soft or hard sample in [0, 1] (probability of computing)
    """
    # Generate Gumbel noise
    u = torch.rand_like(logits).clamp(1e-10, 1 - 1e-10)
    gumbel = -torch.log(-torch.log(u))
    
    # Soft sample
    y_soft = torch.sigmoid((logits + gumbel) / temperature)
    
    if hard:
        # Straight-through: round in forward, identity in backward
        y_hard = (y_soft > 0.5).float()
        return y_hard - y_soft.detach() + y_soft
    
    return y_soft


class LazyGate(nn.Module):
    """
    A stochastic dual-path gate that decides whether to compute or skip.
    
    Architecture:
    ┌────────────────────────────────────────────────────────────┐
    │                         Input x                            │
    │                            │                               │
    │              ┌─────────────┴─────────────┐                 │
    │              ▼                           ▼                 │
    │     ┌─────────────────┐         ┌─────────────────┐       │
    │     │  Expensive Path │         │   Cheap Path    │       │
    │     │     f(x)        │         │   skip(x)       │       │
    │     └────────┬────────┘         └────────┬────────┘       │
    │              │                           │                 │
    │              ▼                           ▼                 │
    │     ┌─────────────────────────────────────────────┐       │
    │     │  y = z·a·f(x) + (1 - z·a)·skip(x)          │       │
    │     │  where z ~ Bernoulli(1 - p)                 │       │
    │     └─────────────────────────────────────────────┘       │
    │                            │                               │
    │                         Output y                           │
    └────────────────────────────────────────────────────────────┘
    
    Parameters are evolved by LazyTuner, not trained by gradient descent.
    However, Gumbel-Softmax enables optional gradient-based fine-tuning.
    """
    
    def __init__(
        self,
        expensive_fn: nn.Module,
        cheap_fn: Optional[nn.Module] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        initial_p: float = 0.5,
        initial_a: float = 1.0,
        flops_expensive: float = 0.0,
        flops_cheap: float = 0.0,
        name: str = "gate"
    ):
        """
        Args:
            expensive_fn: The full computation module (e.g., Conv1d layer)
            cheap_fn: The skip path module. If None, creates identity or projection.
            in_features: Input dimension (for auto-creating cheap_fn)
            out_features: Output dimension (for auto-creating cheap_fn)
            initial_p: Initial skip probability ∈ [0,1]
            initial_a: Initial ambition ∈ [0,1]
            flops_expensive: FLOPs for expensive path (for statistics)
            flops_cheap: FLOPs for cheap path (for statistics)
            name: Identifier for debugging/visualization
        """
        super().__init__()
        
        self.expensive_fn = expensive_fn
        self.name = name
        self.flops_expensive = flops_expensive
        self.flops_cheap = flops_cheap
        
        # Create cheap path if not provided
        if cheap_fn is not None:
            self.cheap_fn = cheap_fn
        elif in_features is not None and out_features is not None:
            if in_features == out_features:
                self.cheap_fn = nn.Identity()
            else:
                # 1x1 linear projection (learned during base model training)
                self.cheap_fn = nn.Linear(in_features, out_features, bias=False)
        else:
            self.cheap_fn = nn.Identity()
        
        # Gate parameters (not nn.Parameter — evolved, not gradient-trained)
        # Stored as raw floats, converted to tensors during forward
        self._p = initial_p  # Skip probability
        self._a = initial_a  # Ambition
        
        # For Gumbel-Softmax fine-tuning (optional)
        self.temperature = 1.0
        self.hard_sampling = True  # Use hard samples by default
        
        # Statistics tracker
        self.stats = GateStatistics()
        
        # Training mode flag for gate (different from nn.Module.training)
        self._gate_training = False
    
    @property
    def p(self) -> float:
        """Skip probability (procrastination index)."""
        return self._p
    
    @p.setter
    def p(self, value: float) -> None:
        """Set skip probability, clamped to [0, 1]."""
        self._p = max(0.0, min(1.0, value))
    
    @property
    def a(self) -> float:
        """Ambition index."""
        return self._a
    
    @a.setter
    def a(self, value: float) -> None:
        """Set ambition, clamped to [0, 1]."""
        self._a = max(0.0, min(1.0, value))
    
    @property
    def effective_compute(self) -> float:
        """
        Expected fraction of expensive computation executed.
        
        E[z·a] = (1 - p) · a
        
        - p=0, a=1: 100% compute (no laziness)
        - p=1, a=any: 0% compute (maximum laziness)
        - p=0.5, a=0.5: 25% effective compute
        """
        return (1 - self._p) * self._a
    
    def get_logit(self, device: torch.device) -> torch.Tensor:
        """
        Convert skip probability p to logit for Gumbel-Softmax.
        
        logit = log((1-p) / p) = log(compute_prob / skip_prob)
        
        Positive logit → more likely to compute
        Negative logit → more likely to skip
        """
        # Clamp to avoid log(0)
        p_clamped = max(1e-7, min(1 - 1e-7, self._p))
        logit = math.log((1 - p_clamped) / p_clamped)
        return torch.tensor([logit], device=device)
    
    def sample_gate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample the gate decision z.
        
        Returns:
            z: Tensor of shape (batch_size, 1, ...) with same dims as needed for broadcast
        """
        batch_size = x.shape[0]
        device = x.device
        
        if self._gate_training:
            # Differentiable sampling with Gumbel-Softmax
            logit = self.get_logit(device).expand(batch_size)
            z = gumbel_softmax_sample(logit, self.temperature, self.hard_sampling)
        else:
            # Hard Bernoulli sampling (non-differentiable but accurate)
            compute_prob = 1 - self._p
            z = (torch.rand(batch_size, device=device) < compute_prob).float()
        
        # Reshape for broadcasting: (batch, 1, 1, ...) to match x dimensions
        for _ in range(x.dim() - 1):
            z = z.unsqueeze(-1)
        
        return z
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dual-path gate.
        
        y = z · a · f(x) + (1 - z·a) · skip(x)
        
        Where z ~ Bernoulli(1 - p)
        """
        # Sample gate decision
        z = self.sample_gate(x)
        
        # Cheap path always runs (it's cheap!)
        y_cheap = self.cheap_fn(x)
        
        # Determine if ANY sample in the batch needs expensive computation
        # This is an optimization: if z=0 for all, skip expensive entirely
        needs_expensive = (z.sum() > 0).item()
        
        if needs_expensive:
            y_expensive = self.expensive_fn(x)
            
            # Blend: y = z·a·expensive + (1 - z·a)·cheap
            za = z * self._a
            y = za * y_expensive + (1 - za) * y_cheap
            
            # Record statistics (per-sample tracking)
            for i in range(z.shape[0]):
                computed = z[i].item() > 0.5
                self.stats.record(computed, self.flops_expensive, 
                                  self.flops_cheap, self._a)
        else:
            # Pure cheap path — no expensive computation at all
            y = y_cheap
            
            # Record all as skipped
            for _ in range(z.shape[0]):
                self.stats.record(False, self.flops_expensive, 
                                  self.flops_cheap, self._a)
        
        return y
    
    def set_deterministic(self, compute: bool) -> None:
        """
        Force deterministic behavior (for deployment or debugging).
        
        Args:
            compute: If True, always compute. If False, always skip.
        """
        self._p = 0.0 if compute else 1.0
    
    def enable_gate_training(self, temperature: float = 1.0) -> None:
        """Enable differentiable Gumbel-Softmax sampling for fine-tuning."""
        self._gate_training = True
        self.temperature = temperature
    
    def disable_gate_training(self) -> None:
        """Disable Gumbel-Softmax, use hard Bernoulli sampling."""
        self._gate_training = False
    
    def reset_stats(self) -> None:
        """Reset gate statistics."""
        self.stats.reset()
    
    def __repr__(self) -> str:
        return (
            f"LazyGate(name={self.name}, p={self._p:.3f}, a={self._a:.3f}, "
            f"effective_compute={self.effective_compute:.1%})"
        )


def estimate_conv1d_flops(
    in_channels: int, 
    out_channels: int, 
    kernel_size: int, 
    seq_len: int
) -> float:
    """
    Estimate FLOPs for a 1D convolution.
    
    FLOPs ≈ 2 × in_channels × out_channels × kernel_size × seq_len
    
    The factor of 2 accounts for multiply-accumulate operations.
    """
    return 2 * in_channels * out_channels * kernel_size * seq_len


def estimate_linear_flops(in_features: int, out_features: int) -> float:
    """
    Estimate FLOPs for a linear layer.
    
    FLOPs ≈ 2 × in_features × out_features
    """
    return 2 * in_features * out_features


def create_lazy_conv1d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    initial_p: float = 0.5,
    initial_a: float = 1.0,
    seq_len: int = 100,
    name: str = "conv_gate"
) -> LazyGate:
    """
    Factory function to create a LazyGate wrapping a Conv1d layer.
    
    The cheap path is a 1x1 Conv (if channels differ) or identity.
    """
    expensive = nn.Conv1d(in_channels, out_channels, kernel_size, 
                          stride=stride, padding=padding)
    
    if in_channels == out_channels and stride == 1:
        cheap = nn.Identity()
        flops_cheap = 0.0
    else:
        cheap = nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False)
        flops_cheap = estimate_conv1d_flops(in_channels, out_channels, 1, 
                                            seq_len // stride)
    
    flops_expensive = estimate_conv1d_flops(in_channels, out_channels, 
                                            kernel_size, seq_len // stride)
    
    return LazyGate(
        expensive_fn=expensive,
        cheap_fn=cheap,
        initial_p=initial_p,
        initial_a=initial_a,
        flops_expensive=flops_expensive,
        flops_cheap=flops_cheap,
        name=name
    )


# Type alias for genome representation
GateGenome = Tuple[float, float]  # (p, a)


class LazyGateConfig:
    """
    Configuration container for multiple gates.
    
    Enables bulk operations and serialization of gate parameters.
    """
    
    def __init__(self, gates: list[LazyGate]):
        self.gates = gates
    
    def get_genome(self) -> list[GateGenome]:
        """Extract (p, a) pairs from all gates as a flat genome."""
        return [(g.p, g.a) for g in self.gates]
    
    def set_genome(self, genome: list[GateGenome]) -> None:
        """Apply (p, a) pairs from genome to all gates."""
        assert len(genome) == len(self.gates), \
            f"Genome length {len(genome)} != gate count {len(self.gates)}"
        for gate, (p, a) in zip(self.gates, genome):
            gate.p = p
            gate.a = a
    
    def get_flat_genome(self) -> list[float]:
        """Get genome as flat list [p₁, a₁, p₂, a₂, ...]."""
        flat = []
        for g in self.gates:
            flat.extend([g.p, g.a])
        return flat
    
    def set_flat_genome(self, flat: list[float]) -> None:
        """Set genome from flat list."""
        assert len(flat) == 2 * len(self.gates), \
            f"Flat genome length {len(flat)} != 2 × gate count"
        for i, gate in enumerate(self.gates):
            gate.p = flat[2 * i]
            gate.a = flat[2 * i + 1]
    
    @property
    def total_effective_compute(self) -> float:
        """Average effective compute across all gates."""
        if not self.gates:
            return 0.0
        return sum(g.effective_compute for g in self.gates) / len(self.gates)
    
    @property
    def total_skip_ratio(self) -> float:
        """Aggregate skip ratio across all gates."""
        total_skips = sum(g.stats.skip_count for g in self.gates)
        total = sum(g.stats.compute_count + g.stats.skip_count for g in self.gates)
        return total_skips / total if total > 0 else 0.0
    
    @property
    def total_flops_saved_ratio(self) -> float:
        """Aggregate FLOP savings across all gates."""
        total_full = sum(g.stats.total_flops_full for g in self.gates)
        total_actual = sum(g.stats.total_flops_actual for g in self.gates)
        if total_full == 0:
            return 0.0
        return 1 - (total_actual / total_full)
    
    def reset_all_stats(self) -> None:
        """Reset statistics for all gates."""
        for g in self.gates:
            g.reset_stats()
    
    def summary(self) -> str:
        """Generate a summary table of all gates."""
        lines = [
            "┌─────────────────┬───────┬───────┬────────────┬───────────┐",
            "│ Gate            │   p   │   a   │ Eff.Comp.  │ Skip Rate │",
            "├─────────────────┼───────┼───────┼────────────┼───────────┤"
        ]
        for g in self.gates:
            skip_rate = g.stats.skip_ratio
            lines.append(
                f"│ {g.name:15s} │ {g.p:5.3f} │ {g.a:5.3f} │ "
                f"{g.effective_compute:10.1%} │ {skip_rate:9.1%} │"
            )
        lines.append("├─────────────────┼───────┼───────┼────────────┼───────────┤")
        lines.append(
            f"│ {'TOTAL':15s} │       │       │ "
            f"{self.total_effective_compute:10.1%} │ {self.total_skip_ratio:9.1%} │"
        )
        lines.append("└─────────────────┴───────┴───────┴────────────┴───────────┘")
        return "\n".join(lines)
