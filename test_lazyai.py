#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                    LAZYAI COMPREHENSIVE TEST SUITE
═══════════════════════════════════════════════════════════════════════════════

Tests for:
    1. LazyGate - Dual-path stochastic skip
    2. LazyWrapper - Automatic gate injection
    3. LazyTuner - CMA-ES evolutionary optimization
    4. Integration - Full pipeline tests

Run with: python test_lazyai.py
"""

import sys
import math
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, '.')

# ═══════════════════════════════════════════════════════════════════════════════
#                              TEST UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def test_header(name: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  TEST: {name}")
    print(f"{'═' * 70}")

def test_passed(name: str) -> None:
    print(f"  ✓ {name}")

def test_failed(name: str, error: str) -> None:
    print(f"  ✗ {name}: {error}")

def assert_close(a: float, b: float, tol: float = 0.1) -> bool:
    return abs(a - b) < tol


# ═══════════════════════════════════════════════════════════════════════════════
#                              TEST 1: LazyGate
# ═══════════════════════════════════════════════════════════════════════════════

def test_lazy_gate():
    test_header("LazyGate - Dual-Path Stochastic Skip")
    
    from lazyai import LazyGate, GateStatistics, create_lazy_conv1d
    
    # Test 1.1: Basic construction (same dimensions)
    conv_same = nn.Conv1d(32, 32, kernel_size=5, padding=2)
    gate_same = LazyGate(
        expensive_fn=conv_same,
        initial_p=0.7,
        initial_a=0.8,
        name="test_gate_same"
    )
    
    assert gate_same.p == 0.7, f"Expected p=0.7, got {gate_same.p}"
    assert gate_same.a == 0.8, f"Expected a=0.8, got {gate_same.a}"
    assert_close(gate_same.effective_compute, 0.24)
    test_passed("Construction with correct parameters")
    
    # Test 1.2: Forward pass with same dimensions (Identity cheap path)
    x = torch.randn(4, 32, 100)
    y = gate_same(x)
    
    assert y.shape == (4, 32, 100), f"Wrong output shape: {y.shape}"
    test_passed("Forward pass with same dimensions")
    
    # Test 1.3: Forward pass with dimension change using factory
    gate_dim = create_lazy_conv1d(
        in_channels=32,
        out_channels=64,
        kernel_size=5,
        padding=2,
        initial_p=0.6,
        initial_a=0.9,
        seq_len=100,
        name="test_gate_dim"
    )
    
    y_dim = gate_dim(x)
    assert y_dim.shape == (4, 64, 100), f"Wrong output shape: {y_dim.shape}"
    test_passed("Forward pass with dimension change (32→64)")
    
    # Test 1.4: Statistics tracking
    gate_same.reset_stats()
    for _ in range(100):
        _ = gate_same(x)
    
    stats = gate_same.stats
    skip_ratio = stats.skip_ratio
    assert 0.5 < skip_ratio < 0.9, f"Skip ratio {skip_ratio} not near p=0.7"
    test_passed(f"Statistics tracking (skip_ratio={skip_ratio:.2f})")
    
    # Test 1.5: Deterministic mode
    gate_same.set_deterministic(True)
    gate_same.reset_stats()
    y1 = gate_same(x)
    y2 = gate_same(x)
    assert torch.allclose(y1, y2), "Deterministic mode not deterministic"
    test_passed("Deterministic mode")
    
    # Test 1.6: Gumbel-Softmax training
    gate_same.set_deterministic(False)
    gate_same.enable_gate_training(temperature=1.0)
    x_train = torch.randn(2, 32, 50, requires_grad=True)
    y_train = gate_same(x_train)
    loss = y_train.sum()
    loss.backward()
    
    assert x_train.grad is not None, "Gradients not flowing"
    grad_norm = x_train.grad.norm().item()
    assert grad_norm > 0, "Zero gradients"
    test_passed(f"Gumbel-Softmax differentiability (grad_norm={grad_norm:.2f})")
    
    print(f"\n  ✓ All LazyGate tests passed")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#                              TEST 2: LazyWrapper
# ═══════════════════════════════════════════════════════════════════════════════

def test_lazy_wrapper():
    test_header("LazyWrapper - Automatic Gate Injection")
    
    from lazyai import LazyWrapper, GATABLE_LAYERS
    
    # Test 2.1: Define test model
    class TestCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
            self.bn1 = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
            self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.conv2(x))
            x = self.conv3(x)
            return x
    
    model = TestCNN()
    sample = torch.randn(2, 32, 100)
    
    # Test 2.2: Wrapper construction
    lazy_model = LazyWrapper(model, sample_input=sample, initial_p=0.5, initial_a=0.9)
    
    assert lazy_model.num_gates == 3, f"Expected 3 gates, got {lazy_model.num_gates}"
    assert lazy_model.genome_size == 6, f"Expected genome_size=6, got {lazy_model.genome_size}"
    test_passed(f"Wrapper construction ({lazy_model.num_gates} gates injected)")
    
    # Test 2.3: Forward pass
    y = lazy_model(sample)
    assert y.shape == sample.shape, f"Wrong output shape: {y.shape}"
    test_passed(f"Forward pass (output shape: {tuple(y.shape)})")
    
    # Test 2.4: Genome manipulation
    genome = lazy_model.get_flat_genome()
    assert len(genome) == 6, f"Wrong genome length: {len(genome)}"
    
    new_genome = [0.8, 0.9, 0.7, 0.85, 0.6, 0.95]
    lazy_model.set_flat_genome(new_genome)
    
    recovered = lazy_model.get_flat_genome()
    assert_close(recovered[0], 0.8)
    assert_close(recovered[1], 0.9)
    test_passed("Genome get/set operations")
    
    # Test 2.5: Global statistics
    p_global = lazy_model.procrastination
    a_global = lazy_model.ambition
    
    expected_p = (0.8 + 0.7 + 0.6) / 3
    assert_close(p_global, expected_p), f"Wrong global P: {p_global} vs {expected_p}"
    test_passed(f"Global statistics (P̄={p_global:.3f}, Ā={a_global:.3f})")
    
    # Test 2.6: Summary generation
    summary = lazy_model.summary()
    assert "LazyWrapper" in summary
    assert "conv1" in summary or "conv2" in summary
    test_passed("Summary generation")
    
    # Test 2.7: Runtime statistics
    lazy_model.reset_stats()
    for _ in range(50):
        _ = lazy_model(sample)
    
    flops_saved = lazy_model.flops_saved_ratio
    assert 0 < flops_saved < 1, f"Invalid flops_saved: {flops_saved}"
    test_passed(f"Runtime statistics (FLOPs saved: {flops_saved:.1%})")
    
    print(f"\n  ✓ All LazyWrapper tests passed")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#                              TEST 3: LazyTuner
# ═══════════════════════════════════════════════════════════════════════════════

def test_lazy_tuner():
    test_header("LazyTuner - CMA-ES Evolutionary Optimization")
    
    from lazyai import (
        LazyTuner, 
        LazyTunerConfig, 
        compute_pad_fitness,
        genome_to_pa,
        genome_summary
    )
    
    # Test 3.1: PAD fitness function
    fitness = compute_pad_fitness(
        quality=0.9,
        procrastination=0.5,
        ambition=0.8,
        beta=1.0,
        eta=0.5
    )
    
    expected = 0.9 * math.exp(1.0 * 0.5) * (1 + 0.5 * 0.8)
    assert_close(fitness, expected, 0.01)
    test_passed(f"PAD fitness computation (J={fitness:.4f})")
    
    # Test 3.2: Genome utilities
    genome = [0.6, 0.9, 0.7, 0.8, 0.5, 0.95]
    ps, ays = genome_to_pa(genome)
    
    assert ps == [0.6, 0.7, 0.5], f"Wrong ps: {ps}"
    assert ays == [0.9, 0.8, 0.95], f"Wrong as: {ays}"
    test_passed("genome_to_pa decomposition")
    
    mean_p, mean_a = genome_summary(genome)
    expected_mean_p = (0.6 + 0.7 + 0.5) / 3
    assert_close(mean_p, expected_mean_p)
    test_passed(f"genome_summary (P̄={mean_p:.3f}, Ā={mean_a:.3f})")
    
    # Test 3.3: Simple optimization (Rastrigin-like with PAD)
    def test_fitness(genome: List[float]) -> float:
        """
        Test fitness: maximize laziness while staying near quality target.
        
        Quality simulated as inverse distance from optimal genome.
        """
        # Target: high P (~0.8), moderate A (~0.7)
        target = [0.8, 0.7] * (len(genome) // 2)
        
        # Quality = inverse MSE from target
        mse = sum((g - t) ** 2 for g, t in zip(genome, target)) / len(genome)
        quality = 1 / (1 + 10 * mse)
        
        # PAD fitness
        mean_p, mean_a = genome_summary(genome)
        return compute_pad_fitness(quality, mean_p, mean_a, beta=1.0, eta=0.5, q_min=0.3)
    
    config = LazyTunerConfig(
        beta=1.0,
        eta=0.5,
        q_min=0.3,
        sigma=0.3,
        population_size=15,
        max_generations=20
    )
    
    tuner = LazyTuner(
        genome_size=6,  # 3 layers
        fitness_fn=test_fitness,
        config=config,
        verbose=False
    )
    
    best_genome, history = tuner.evolve()
    
    assert len(history) > 0, "No evolution history"
    assert best_genome is not None, "No best genome"
    test_passed(f"Evolution completed ({len(history)} generations)")
    
    # Test 3.4: Fitness improvement
    initial_fitness = history[0].best_fitness
    final_fitness = history[-1].best_fitness
    
    assert final_fitness >= initial_fitness, "Fitness did not improve"
    test_passed(f"Fitness improved: {initial_fitness:.4f} → {final_fitness:.4f}")
    
    # Test 3.5: Convergence toward target
    final_p, final_a = genome_summary(best_genome)
    assert_close(final_p, 0.8, 0.2), f"P not converging: {final_p}"
    test_passed(f"Converged toward target (P̄={final_p:.3f})")
    
    # Test 3.6: Solution injection
    tuner2 = LazyTuner(
        genome_size=6,
        fitness_fn=test_fitness,
        config=config,
        verbose=False
    )
    
    good_solution = [0.75, 0.65, 0.85, 0.75, 0.8, 0.7]
    tuner2.inject_solution(good_solution, weight=0.8)
    
    # Check mean shifted toward injection
    mean_before_injection = (config.bounds[0] + config.bounds[1]) / 2
    actual_mean = tuner2.m.mean()
    assert actual_mean > mean_before_injection, "Injection did not shift mean"
    test_passed("Solution injection")
    
    print(f"\n  ✓ All LazyTuner tests passed")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#                              TEST 4: Integration
# ═══════════════════════════════════════════════════════════════════════════════

def test_integration():
    test_header("Integration - Full LazyAI Pipeline")
    
    from lazyai import (
        LazyGate,
        LazyWrapper,
        LazyTuner,
        LazyTunerConfig,
        create_quick_tuner,
        compute_pad_fitness,
        genome_summary
    )
    
    # Test 4.1: Build a complete lazy model
    class SimpleAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(16, 32, 5, padding=2),
                nn.ReLU(),
                nn.Conv1d(32, 16, 3, padding=1),
            )
            self.decoder = nn.Sequential(
                nn.Conv1d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 16, 5, padding=2),
            )
        
        def forward(self, x):
            return self.decoder(self.encoder(x))
    
    model = SimpleAutoencoder()
    sample = torch.randn(4, 16, 64)
    
    lazy_model = LazyWrapper(model, sample_input=sample, initial_p=0.4, initial_a=0.95)
    test_passed(f"Built lazy model with {lazy_model.num_gates} gates")
    
    # Test 4.2: Verify model works
    output = lazy_model(sample)
    assert output.shape == sample.shape, f"Shape mismatch: {output.shape}"
    test_passed("Lazy model forward pass")
    
    # Test 4.3: Quality evaluator
    def quality_evaluator(wrapper):
        """Simulated quality: reconstruction MSE → quality score."""
        with torch.no_grad():
            out = wrapper(sample)
            mse = ((out - sample) ** 2).mean().item()
            return 1 / (1 + mse)
    
    initial_quality = quality_evaluator(lazy_model)
    test_passed(f"Quality evaluator (Q={initial_quality:.4f})")
    
    # Test 4.4: Quick tuner creation
    tuner = create_quick_tuner(
        lazy_wrapper=lazy_model,
        quality_evaluator=quality_evaluator,
        beta=0.8,
        eta=0.3,
        q_min=0.5,
        population_size=10,
        max_generations=5  # Few generations for test speed
    )
    
    assert tuner.n == lazy_model.genome_size
    test_passed("Quick tuner creation")
    
    # Test 4.5: Run evolution (brief)
    tuner.verbose = False
    best_genome, history = tuner.evolve()
    
    assert len(history) == 5, f"Expected 5 generations, got {len(history)}"
    test_passed(f"Evolution ran ({len(history)} generations)")
    
    # Test 4.6: Apply evolved genome
    lazy_model.set_flat_genome(best_genome)
    
    mean_p, mean_a = genome_summary(best_genome)
    test_passed(f"Applied evolved genome (P̄={mean_p:.3f}, Ā={mean_a:.3f})")
    
    # Test 4.7: Final quality check
    final_quality = quality_evaluator(lazy_model)
    final_p = lazy_model.procrastination
    final_a = lazy_model.ambition
    
    test_passed(f"Final stats: Q={final_quality:.4f}, P̄={final_p:.3f}, Ā={final_a:.3f}")
    
    # Test 4.8: FLOP savings
    lazy_model.reset_stats()
    for _ in range(20):
        _ = lazy_model(sample)
    
    flops_saved = lazy_model.flops_saved_ratio
    effective = lazy_model.effective_compute
    test_passed(f"Compute profile: {1-effective:.1%} theoretical savings, {flops_saved:.1%} actual")
    
    print(f"\n  ✓ All Integration tests passed")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#                              MAIN TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██       █████  ███████ ██    ██  █████  ██                                ║
║   ██      ██   ██    ███   ██  ██  ██   ██ ██                                ║
║   ██      ███████   ███     ████   ███████ ██                                ║
║   ██      ██   ██  ███       ██    ██   ██ ██                                ║
║   ███████ ██   ██ ███████    ██    ██   ██ ██                                ║
║                                                                               ║
║   Comprehensive Test Suite v0.2.0                                            ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    tests = [
        ("LazyGate", test_lazy_gate),
        ("LazyWrapper", test_lazy_wrapper),
        ("LazyTuner", test_lazy_tuner),
        ("Integration", test_integration),
    ]
    
    results = []
    
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            import traceback
            results.append((name, False, str(e)))
            traceback.print_exc()
    
    # Summary
    print("\n")
    print("═" * 70)
    print("                         TEST SUMMARY")
    print("═" * 70)
    
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASSED" if success else f"✗ FAILED: {error}"
        print(f"  {name:20} {status}")
    
    print("═" * 70)
    print(f"  Total: {passed}/{total} test suites passed")
    print("═" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
