"""
LazyWrapper: Automatic Gate Injection for Neural Networks

Mathematical Foundation:
━━━━━━━━━━━━━━━━━━━━━━━━
Given a network with L layers {f₁, f₂, ..., f_L}, LazyWrapper transforms it into
a gated network where each layer fₗ becomes a LazyGate with parameters (pₗ, aₗ).

The wrapped forward pass:
    x₀ = input
    xₗ = LazyGate(fₗ, pₗ, aₗ)(xₗ₋₁)    for l = 1, ..., L
    output = x_L

Total network statistics:
    P_global = (1/L) Σₗ pₗ                    # Average procrastination
    A_global = (1/L) Σₗ (1 - pₗ) · aₗ         # Weighted average ambition
    FLOPs_saved = 1 - Σₗ(1-pₗ)·aₗ·Cₗ / Σₗ Cₗ  # Actual FLOP savings

Author: LazyAI
License: MIT
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from collections import OrderedDict

import torch
import torch.nn as nn

from .gate import (
    LazyGate, 
    LazyGateConfig, 
    estimate_conv1d_flops, 
    estimate_linear_flops
)


# Layer types that are worth gating (expensive computations)
GATABLE_LAYERS: Set[Type[nn.Module]] = {
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.Linear,
    nn.MultiheadAttention,
    nn.TransformerEncoderLayer,
    nn.TransformerDecoderLayer,
    nn.LSTM,
    nn.GRU,
}

# Layers to never gate (cheap or essential)
NON_GATABLE_LAYERS: Set[Type[nn.Module]] = {
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.LayerNorm,
    nn.Dropout,
    nn.ReLU,
    nn.GELU,
    nn.SiLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.Softmax,
    nn.Identity,
    nn.Flatten,
    nn.Unflatten,
}


def estimate_layer_flops(layer: nn.Module, input_shape: Tuple[int, ...]) -> float:
    """
    Estimate FLOPs for a layer given input shape.
    
    This is a heuristic; precise counting requires tracing.
    
    Args:
        layer: The nn.Module to analyze
        input_shape: Expected input shape (batch, ...)
    
    Returns:
        Estimated FLOPs for one forward pass
    """
    if isinstance(layer, nn.Conv1d):
        # Shape: (batch, channels, seq_len)
        seq_len = input_shape[-1] if len(input_shape) >= 3 else 100
        return estimate_conv1d_flops(
            layer.in_channels, 
            layer.out_channels, 
            layer.kernel_size[0],
            seq_len // layer.stride[0]
        )
    
    elif isinstance(layer, nn.ConvTranspose1d):
        seq_len = input_shape[-1] if len(input_shape) >= 3 else 100
        return estimate_conv1d_flops(
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size[0],
            seq_len * layer.stride[0]
        )
    
    elif isinstance(layer, nn.Linear):
        return estimate_linear_flops(layer.in_features, layer.out_features)
    
    elif isinstance(layer, (nn.Conv2d, nn.Conv3d)):
        # Rough estimate: treat as multiple 1D convolutions
        if isinstance(layer, nn.Conv2d):
            spatial = input_shape[-2:] if len(input_shape) >= 4 else (32, 32)
            return estimate_conv1d_flops(
                layer.in_channels, layer.out_channels,
                layer.kernel_size[0] * layer.kernel_size[1],
                spatial[0] * spatial[1]
            )
        else:
            # Conv3d
            return estimate_conv1d_flops(
                layer.in_channels, layer.out_channels,
                8 * 8 * 8,  # Rough kernel estimate
                32 * 32 * 32
            )
    
    elif isinstance(layer, nn.MultiheadAttention):
        # Attention: O(n² · d) where n=seq_len, d=embed_dim
        seq_len = input_shape[0] if len(input_shape) >= 2 else 100
        return 4 * seq_len * seq_len * layer.embed_dim
    
    elif isinstance(layer, (nn.LSTM, nn.GRU)):
        seq_len = input_shape[0] if len(input_shape) >= 2 else 100
        hidden = layer.hidden_size
        inp = layer.input_size
        num_gates = 4 if isinstance(layer, nn.LSTM) else 3
        return 2 * seq_len * (inp * hidden + hidden * hidden) * num_gates
    
    else:
        # Default estimate based on parameter count
        params = sum(p.numel() for p in layer.parameters())
        return 2 * params  # Rough: 2 FLOPs per parameter


def create_cheap_path(
    layer: nn.Module, 
    in_shape: Tuple[int, ...],
    out_shape: Tuple[int, ...]
) -> nn.Module:
    """
    Create the cheap skip path for a layer.
    
    Handles dimension mismatches with minimal-cost projections.
    """
    # Same shape: identity
    if in_shape == out_shape:
        return nn.Identity()
    
    # 1D: (batch, channels, seq_len)
    if len(in_shape) == 3 and len(out_shape) == 3:
        in_c, in_len = in_shape[1], in_shape[2]
        out_c, out_len = out_shape[1], out_shape[2]
        
        modules = []
        
        # Channel projection
        if in_c != out_c:
            modules.append(nn.Conv1d(in_c, out_c, kernel_size=1, bias=False))
        
        # Length adjustment
        if in_len != out_len:
            if out_len < in_len:
                # Downsample with adaptive pooling
                modules.append(nn.AdaptiveAvgPool1d(out_len))
            else:
                # Upsample with interpolation
                modules.append(nn.Upsample(size=out_len, mode='linear', align_corners=False))
        
        return nn.Sequential(*modules) if modules else nn.Identity()
    
    # 2D: (batch, channels, height, width)
    if len(in_shape) == 4 and len(out_shape) == 4:
        in_c = in_shape[1]
        out_c = out_shape[1]
        
        modules = []
        if in_c != out_c:
            modules.append(nn.Conv2d(in_c, out_c, kernel_size=1, bias=False))
        
        if in_shape[2:] != out_shape[2:]:
            modules.append(nn.AdaptiveAvgPool2d(out_shape[2:]))
        
        return nn.Sequential(*modules) if modules else nn.Identity()
    
    # Linear: (batch, features)
    if len(in_shape) == 2 and len(out_shape) == 2:
        in_f, out_f = in_shape[1], out_shape[1]
        if in_f != out_f:
            return nn.Linear(in_f, out_f, bias=False)
        return nn.Identity()
    
    # Fallback: can't create cheap path, must compute
    return None


class LazyWrapper(nn.Module):
    """
    Wraps an existing neural network with LazyGates.
    
    Architecture:
    ┌────────────────────────────────────────────────────────────┐
    │                    Original Network                        │
    │  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐            │
    │  │  f₁  │───▶│  f₂  │───▶│  f₃  │───▶│  f₄  │───▶ out   │
    │  └──────┘    └──────┘    └──────┘    └──────┘            │
    └────────────────────────────────────────────────────────────┘
                              ▼
    ┌────────────────────────────────────────────────────────────┐
    │                    LazyWrapper                             │
    │  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐            │
    │  │Gate₁ │───▶│Gate₂ │───▶│Gate₃ │───▶│Gate₄ │───▶ out   │
    │  │(p,a) │    │(p,a) │    │(p,a) │    │(p,a) │            │
    │  └──────┘    └──────┘    └──────┘    └──────┘            │
    │      ▲           ▲           ▲           ▲               │
    │      │           │           │           │               │
    │  ┌─────────────────────────────────────────┐             │
    │  │          LazyGateConfig                  │             │
    │  │  Genome: [(p₁,a₁), (p₂,a₂), ...]        │             │
    │  │  Evolved by LazyTuner                    │             │
    │  └─────────────────────────────────────────┘             │
    └────────────────────────────────────────────────────────────┘
    
    Usage:
        model = YourNetwork()
        lazy_model = LazyWrapper(model, sample_input=torch.randn(1, 32, 100))
        
        # Access gate configuration
        config = lazy_model.gate_config
        config.set_flat_genome(evolved_params)
        
        # Forward pass uses gates
        output = lazy_model(input)
        print(lazy_model.flops_saved_ratio)
    """
    
    def __init__(
        self,
        model: nn.Module,
        sample_input: Optional[torch.Tensor] = None,
        gate_layers: Optional[Set[Type[nn.Module]]] = None,
        initial_p: float = 0.5,
        initial_a: float = 1.0,
        min_flops_threshold: float = 1000,
    ):
        """
        Args:
            model: The network to wrap
            sample_input: Example input tensor for shape inference
            gate_layers: Layer types to gate (default: GATABLE_LAYERS)
            initial_p: Default skip probability for all gates
            initial_a: Default ambition for all gates
            min_flops_threshold: Only gate layers with FLOPs > this
        """
        super().__init__()
        
        self.model = model
        self.gate_layers = gate_layers or GATABLE_LAYERS
        self.initial_p = initial_p
        self.initial_a = initial_a
        self.min_flops_threshold = min_flops_threshold
        
        # Will be populated by _analyze_and_wrap
        self.gates: nn.ModuleList = nn.ModuleList()
        self._gate_map: Dict[str, int] = {}  # layer_name -> gate_index
        self._layer_order: List[str] = []
        self._hooks: List[Any] = []
        
        # Analyze model and create gates
        if sample_input is not None:
            self._analyze_and_wrap(sample_input)
        else:
            # Deferred wrapping - user must provide sample_input
            self._wrapped = False
    
    def _analyze_and_wrap(self, sample_input: torch.Tensor) -> None:
        """
        Analyze the model structure and wrap gatable layers.
        
        Uses forward hooks to capture intermediate shapes.
        """
        shapes: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
        hooks = []
        
        def make_hook(name: str):
            def hook(module, inp, out):
                in_shape = inp[0].shape if isinstance(inp, tuple) else inp.shape
                out_shape = out.shape if isinstance(out, torch.Tensor) else out[0].shape
                shapes[name] = (tuple(in_shape), tuple(out_shape))
            return hook
        
        # Register hooks on all modules
        for name, module in self.model.named_modules():
            if name:  # Skip root
                hooks.append(module.register_forward_hook(make_hook(name)))
        
        # Forward pass to capture shapes
        with torch.no_grad():
            self.model.eval()
            _ = self.model(sample_input)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        # Create gates for gatable layers
        gate_idx = 0
        for name, module in self.model.named_modules():
            if name and type(module) in self.gate_layers:
                if name not in shapes:
                    continue
                
                in_shape, out_shape = shapes[name]
                flops = estimate_layer_flops(module, in_shape)
                
                if flops < self.min_flops_threshold:
                    continue
                
                # Create cheap path
                cheap_fn = create_cheap_path(module, in_shape, out_shape)
                if cheap_fn is None:
                    # Can't create cheap path, skip gating this layer
                    continue
                
                # Estimate cheap path FLOPs
                if isinstance(cheap_fn, nn.Identity):
                    flops_cheap = 0
                else:
                    flops_cheap = flops * 0.1  # Rough estimate: cheap = 10% of expensive
                
                # Create the gate
                gate = LazyGate(
                    expensive_fn=module,
                    cheap_fn=cheap_fn,
                    initial_p=self.initial_p,
                    initial_a=self.initial_a,
                    flops_expensive=flops,
                    flops_cheap=flops_cheap,
                    name=name
                )
                
                self.gates.append(gate)
                self._gate_map[name] = gate_idx
                self._layer_order.append(name)
                gate_idx += 1
        
        self._wrapped = True
        self._install_forward_hooks()
    
    def _install_forward_hooks(self) -> None:
        """
        Install forward hooks that route computation through gates.
        
        Uses a recursion guard to prevent infinite loops.
        """
        # Remove any existing hooks
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        
        # Recursion guard
        self._in_hook = False
        
        def make_gate_hook(gate_idx: int):
            def hook(module, inp, out):
                # Guard against recursion
                if self._in_hook:
                    return out  # Use normal output during gate's internal call
                
                self._in_hook = True
                try:
                    # Route through the gate
                    gate = self.gates[gate_idx]
                    x = inp[0] if isinstance(inp, tuple) else inp
                    result = gate(x)
                    return result
                finally:
                    self._in_hook = False
            return hook
        
        for name, module in self.model.named_modules():
            if name in self._gate_map:
                gate_idx = self._gate_map[name]
                h = module.register_forward_hook(make_gate_hook(gate_idx))
                self._hooks.append(h)
    
    @property
    def gate_config(self) -> LazyGateConfig:
        """Get the configuration object for all gates."""
        return LazyGateConfig(list(self.gates))
    
    @property
    def num_gates(self) -> int:
        """Number of gates in the wrapper."""
        return len(self.gates)
    
    @property
    def genome_size(self) -> int:
        """Size of the flat genome (2 * num_gates for p,a pairs)."""
        return 2 * len(self.gates)
    
    @property
    def procrastination(self) -> float:
        """Global P: average skip probability across all gates."""
        if not self.gates:
            return 0.0
        return sum(g.p for g in self.gates) / len(self.gates)
    
    @property
    def ambition(self) -> float:
        """Global A: weighted average ambition."""
        if not self.gates:
            return 0.0
        # Weight by (1-p): ambition matters more when we're computing
        total_weight = sum(1 - g.p for g in self.gates)
        if total_weight == 0:
            return 0.0
        return sum((1 - g.p) * g.a for g in self.gates) / total_weight
    
    @property
    def effective_compute(self) -> float:
        """Expected fraction of full computation used."""
        return self.gate_config.total_effective_compute
    
    @property
    def flops_saved_ratio(self) -> float:
        """Actual FLOP savings based on runtime statistics."""
        return self.gate_config.total_flops_saved_ratio
    
    def set_genome(self, genome: List[Tuple[float, float]]) -> None:
        """Set gate parameters from structured genome."""
        self.gate_config.set_genome(genome)
    
    def set_flat_genome(self, flat: List[float]) -> None:
        """Set gate parameters from flat genome."""
        self.gate_config.set_flat_genome(flat)
    
    def get_genome(self) -> List[Tuple[float, float]]:
        """Get structured genome [(p₁,a₁), ...]."""
        return self.gate_config.get_genome()
    
    def get_flat_genome(self) -> List[float]:
        """Get flat genome [p₁, a₁, p₂, a₂, ...]."""
        return self.gate_config.get_flat_genome()
    
    def reset_stats(self) -> None:
        """Reset statistics for all gates."""
        self.gate_config.reset_all_stats()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the wrapped model.
        
        Gates intercept layer computations via hooks.
        """
        return self.model(x)
    
    def summary(self) -> str:
        """Generate a summary of the wrapper."""
        lines = [
            f"LazyWrapper",
            f"  Wrapped model: {self.model.__class__.__name__}",
            f"  Number of gates: {self.num_gates}",
            f"  Genome size: {self.genome_size}",
            f"  Global P (procrastination): {self.procrastination:.3f}",
            f"  Global A (ambition): {self.ambition:.3f}",
            f"  Effective compute: {self.effective_compute:.1%}",
            "",
            "Gate Details:",
            self.gate_config.summary()
        ]
        return "\n".join(lines)
    
    def enable_gate_training(self, temperature: float = 1.0) -> None:
        """Enable Gumbel-Softmax training for all gates."""
        for gate in self.gates:
            gate.enable_gate_training(temperature)
    
    def disable_gate_training(self) -> None:
        """Disable Gumbel-Softmax, use hard sampling."""
        for gate in self.gates:
            gate.disable_gate_training()


class DualPathBlock(nn.Module):
    """
    A pre-built block with built-in dual-path structure for easier wrapping.
    
    Use this when building networks from scratch for LazyAI.
    
    Architecture:
        y = gate(expensive(x)) + (1 - gate) * cheap(x)
    
    Where:
        expensive = full convolution or linear
        cheap = 1x1 conv or identity
    """
    
    def __init__(
        self,
        expensive_fn: nn.Module,
        cheap_fn: Optional[nn.Module] = None,
        initial_p: float = 0.5,
        initial_a: float = 1.0,
        name: str = "block"
    ):
        super().__init__()
        
        self.gate = LazyGate(
            expensive_fn=expensive_fn,
            cheap_fn=cheap_fn or nn.Identity(),
            initial_p=initial_p,
            initial_a=initial_a,
            name=name
        )
    
    @property
    def p(self) -> float:
        return self.gate.p
    
    @p.setter
    def p(self, value: float):
        self.gate.p = value
    
    @property
    def a(self) -> float:
        return self.gate.a
    
    @a.setter
    def a(self, value: float):
        self.gate.a = value
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)


def create_lazy_conv1d_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    activation: nn.Module = nn.ReLU(),
    initial_p: float = 0.5,
    initial_a: float = 1.0,
    name: str = "conv_block"
) -> DualPathBlock:
    """
    Factory for creating Conv1d blocks with built-in lazy gating.
    
    The expensive path: Conv1d -> Activation
    The cheap path: 1x1 Conv (if channels differ) or Identity
    """
    expensive = nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
        activation
    )
    
    if in_channels == out_channels and stride == 1:
        cheap = nn.Identity()
    else:
        cheap = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
    
    return DualPathBlock(expensive, cheap, initial_p, initial_a, name)
