"""
LazyGate Validation Test

Demonstrates that LazyGate works correctly and produces meaningful statistics.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/Users/kabir/Desktop/Lazy AI')

from lazyai import LazyGate, LazyGateConfig, create_lazy_conv1d

def test_basic_gate():
    """Test basic LazyGate functionality."""
    print("=" * 60)
    print("TEST 1: Basic LazyGate")
    print("=" * 60)
    
    # Create a simple linear layer with a gate
    expensive = nn.Linear(64, 64)
    gate = LazyGate(
        expensive_fn=expensive,
        in_features=64,
        out_features=64,
        initial_p=0.7,  # 70% skip probability
        initial_a=0.8,  # 80% ambition
        flops_expensive=2 * 64 * 64,
        name="test_gate"
    )
    
    print(f"\nGate: {gate}")
    print(f"Expected effective compute: {(1 - 0.7) * 0.8:.1%}")
    
    # Run many forward passes to get statistics
    x = torch.randn(100, 64)  # 100 samples
    
    gate.reset_stats()
    for _ in range(10):  # 10 batches
        _ = gate(x)
    
    print(f"\nAfter 1000 samples:")
    print(f"  Skip ratio: {gate.stats.skip_ratio:.1%}")
    print(f"  FLOPs saved: {gate.stats.flops_saved_ratio:.1%}")
    
    # The skip ratio should be close to p=0.7
    assert 0.6 < gate.stats.skip_ratio < 0.8, f"Skip ratio {gate.stats.skip_ratio} not close to 0.7"
    print("\n✓ Basic gate test passed!")


def test_dimension_mismatch():
    """Test that LazyGate handles dimension changes correctly."""
    print("\n" + "=" * 60)
    print("TEST 2: Dimension Mismatch Handling")
    print("=" * 60)
    
    # Expensive path changes dimensions
    expensive = nn.Linear(32, 64)
    gate = LazyGate(
        expensive_fn=expensive,
        in_features=32,
        out_features=64,
        initial_p=0.5,
        initial_a=1.0,
        name="dim_change_gate"
    )
    
    print(f"\nGate: {gate}")
    print(f"Input dim: 32, Output dim: 64")
    
    x = torch.randn(16, 32)
    y = gate(x)
    
    assert y.shape == (16, 64), f"Expected shape (16, 64), got {y.shape}"
    print(f"Output shape: {y.shape} ✓")
    print("\n✓ Dimension mismatch test passed!")


def test_conv1d_gate():
    """Test the Conv1d factory function."""
    print("\n" + "=" * 60)
    print("TEST 3: Conv1d LazyGate")
    print("=" * 60)
    
    gate = create_lazy_conv1d(
        in_channels=32,
        out_channels=64,
        kernel_size=7,
        padding=3,
        initial_p=0.6,
        initial_a=0.9,
        seq_len=100,
        name="conv_gate"
    )
    
    print(f"\nGate: {gate}")
    print(f"FLOPs (expensive): {gate.flops_expensive:,.0f}")
    print(f"FLOPs (cheap): {gate.flops_cheap:,.0f}")
    print(f"Expected savings ratio: {gate.flops_expensive / (gate.flops_expensive + gate.flops_cheap):.1%}")
    
    x = torch.randn(8, 32, 100)  # (batch, channels, seq_len)
    
    gate.reset_stats()
    for _ in range(20):
        _ = gate(x)
    
    print(f"\nAfter {8 * 20} samples:")
    print(f"  Skip ratio: {gate.stats.skip_ratio:.1%}")
    print(f"  FLOPs saved: {gate.stats.flops_saved_ratio:.1%}")
    
    print("\n✓ Conv1d gate test passed!")


def test_gate_config():
    """Test LazyGateConfig for managing multiple gates."""
    print("\n" + "=" * 60)
    print("TEST 4: LazyGateConfig")
    print("=" * 60)
    
    # Create multiple gates
    gates = [
        LazyGate(nn.Linear(64, 64), initial_p=0.3, initial_a=1.0, name="block_1"),
        LazyGate(nn.Linear(64, 64), initial_p=0.5, initial_a=0.8, name="block_2"),
        LazyGate(nn.Linear(64, 64), initial_p=0.7, initial_a=0.6, name="block_3"),
        LazyGate(nn.Linear(64, 64), initial_p=0.9, initial_a=0.5, name="block_4"),
    ]
    
    config = LazyGateConfig(gates)
    
    print("\nInitial configuration:")
    print(f"Genome: {config.get_genome()}")
    print(f"Flat genome: {config.get_flat_genome()}")
    print(f"Total effective compute: {config.total_effective_compute:.1%}")
    
    # Run some data through
    x = torch.randn(32, 64)
    config.reset_all_stats()
    
    for g in gates:
        x = g(x)
    
    print("\n" + config.summary())
    
    # Test genome modification
    new_genome = [(0.2, 0.9), (0.4, 0.8), (0.6, 0.7), (0.8, 0.6)]
    config.set_genome(new_genome)
    
    print("\nAfter genome update:")
    print(f"Genome: {config.get_genome()}")
    print(f"Total effective compute: {config.total_effective_compute:.1%}")
    
    print("\n✓ Gate config test passed!")


def test_deterministic_mode():
    """Test forced deterministic behavior."""
    print("\n" + "=" * 60)
    print("TEST 5: Deterministic Mode")
    print("=" * 60)
    
    gate = LazyGate(nn.Linear(32, 32), initial_p=0.5, name="det_gate")
    x = torch.randn(16, 32)
    
    # Force always compute
    gate.set_deterministic(compute=True)
    gate.reset_stats()
    for _ in range(10):
        _ = gate(x)
    
    print(f"\nForced compute mode:")
    print(f"  Skip ratio: {gate.stats.skip_ratio:.1%} (expected 0%)")
    assert gate.stats.skip_ratio == 0.0
    
    # Force always skip
    gate.set_deterministic(compute=False)
    gate.reset_stats()
    for _ in range(10):
        _ = gate(x)
    
    print(f"\nForced skip mode:")
    print(f"  Skip ratio: {gate.stats.skip_ratio:.1%} (expected 100%)")
    assert gate.stats.skip_ratio == 1.0
    
    print("\n✓ Deterministic mode test passed!")


def test_gumbel_softmax():
    """Test differentiable Gumbel-Softmax mode."""
    print("\n" + "=" * 60)
    print("TEST 6: Gumbel-Softmax Differentiability")
    print("=" * 60)
    
    gate = LazyGate(nn.Linear(32, 32), initial_p=0.5, name="gumbel_gate")
    gate.enable_gate_training(temperature=1.0)
    
    x = torch.randn(16, 32, requires_grad=True)
    y = gate(x)
    loss = y.sum()
    loss.backward()
    
    print(f"\nGumbel-Softmax enabled (τ=1.0)")
    print(f"Input grad exists: {x.grad is not None}")
    print(f"Input grad norm: {x.grad.norm():.4f}")
    
    assert x.grad is not None, "Gradients should flow through Gumbel-Softmax"
    
    gate.disable_gate_training()
    print("\n✓ Gumbel-Softmax test passed!")


def test_extreme_parameters():
    """Test edge cases with extreme p and a values."""
    print("\n" + "=" * 60)
    print("TEST 7: Extreme Parameters")
    print("=" * 60)
    
    x = torch.randn(32, 64)
    
    # Test p=0 (always compute)
    gate = LazyGate(nn.Linear(64, 64), initial_p=0.0, initial_a=1.0, name="always_on")
    gate.reset_stats()
    for _ in range(5):
        _ = gate(x)
    print(f"\np=0, a=1 (always compute):")
    print(f"  Skip ratio: {gate.stats.skip_ratio:.1%}")
    assert gate.stats.skip_ratio == 0.0
    
    # Test p=1 (always skip)
    gate = LazyGate(nn.Linear(64, 64), initial_p=1.0, initial_a=1.0, name="always_off")
    gate.reset_stats()
    for _ in range(5):
        _ = gate(x)
    print(f"\np=1, a=1 (always skip):")
    print(f"  Skip ratio: {gate.stats.skip_ratio:.1%}")
    assert gate.stats.skip_ratio == 1.0
    
    # Test a=0 (zero ambition when computing)
    gate = LazyGate(nn.Linear(64, 64), initial_p=0.0, initial_a=0.0, name="zero_ambition")
    print(f"\np=0, a=0 (compute but zero ambition):")
    print(f"  Effective compute: {gate.effective_compute:.1%}")
    assert gate.effective_compute == 0.0
    
    print("\n✓ Extreme parameters test passed!")


if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("█" + " " * 20 + "LazyGate Tests" + " " * 24 + "█")
    print("█" * 60 + "\n")
    
    test_basic_gate()
    test_dimension_mismatch()
    test_conv1d_gate()
    test_gate_config()
    test_deterministic_mode()
    test_gumbel_softmax()
    test_extreme_parameters()
    
    print("\n" + "█" * 60)
    print("█" + " " * 15 + "All Tests Passed! ✓" + " " * 22 + "█")
    print("█" * 60 + "\n")
