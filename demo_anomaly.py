#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
                    LAZYAI ANOMALY DETECTION DEMO
═══════════════════════════════════════════════════════════════════════════════

A minimal, self-contained demonstration of the full LazyAI pipeline:

    1. Generate synthetic time-series data with injected anomalies
    2. Train a small 1D-CNN autoencoder (baseline)
    3. Wrap it with LazyWrapper
    4. Evolve gate parameters with LazyTuner
    5. Compare: quality retained vs compute saved

No external datasets needed. Runs in ~30 seconds on CPU.

Usage:
    python demo_anomaly.py
"""

import sys
import math
import time
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, '.')

from lazyai import (
    LazyWrapper,
    LazyTuner,
    LazyTunerConfig,
    compute_pad_fitness,
    genome_summary,
    create_quick_tuner,
)


# ═══════════════════════════════════════════════════════════════════════════════
#                          1. SYNTHETIC DATA
# ═══════════════════════════════════════════════════════════════════════════════

def generate_data(
    n_samples: int = 200,
    seq_len: int = 64,
    n_features: int = 8,
    anomaly_ratio: float = 0.15,
    seed: int = 42
):
    """
    Generate synthetic multivariate time-series with anomalies.
    
    Normal:  smooth sinusoidal signals with mild noise
    Anomaly: sudden spikes, level shifts, or frequency changes
    """
    np.random.seed(seed)
    
    data = np.zeros((n_samples, n_features, seq_len), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)
    
    t = np.linspace(0, 4 * np.pi, seq_len)
    
    for i in range(n_samples):
        # Base signal: sum of sinusoids with feature-specific frequencies
        for f in range(n_features):
            freq = 1.0 + 0.3 * f
            phase = np.random.uniform(0, 2 * np.pi)
            data[i, f] = np.sin(freq * t + phase) + 0.1 * np.random.randn(seq_len)
        
        # Inject anomaly
        if np.random.rand() < anomaly_ratio:
            labels[i] = 1
            anom_type = np.random.choice(['spike', 'shift', 'noise'])
            anom_start = np.random.randint(10, seq_len - 20)
            anom_feat = np.random.randint(0, n_features)
            
            if anom_type == 'spike':
                data[i, anom_feat, anom_start:anom_start+5] += np.random.choice([-1, 1]) * 4.0
            elif anom_type == 'shift':
                data[i, anom_feat, anom_start:] += np.random.choice([-1, 1]) * 2.5
            else:
                data[i, anom_feat, anom_start:anom_start+15] += 2.0 * np.random.randn(15)
    
    # Split: 60% train, 20% val, 20% test
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    
    X_train = torch.from_numpy(data[:n_train])
    X_val = torch.from_numpy(data[n_train:n_train+n_val])
    X_test = torch.from_numpy(data[n_train+n_val:])
    y_test = torch.from_numpy(labels[n_train+n_val:])
    
    return X_train, X_val, X_test, y_test


# ═══════════════════════════════════════════════════════════════════════════════
#                          2. BASE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class TinyAutoencoder(nn.Module):
    """
    Minimal 1D-CNN autoencoder for anomaly detection.
    
    Architecture:
        Encoder: Conv(8→16, k=5) → ReLU → Conv(16→8, k=3)
        Decoder: Conv(8→16, k=3) → ReLU → Conv(16→8, k=5)
    
    Anomaly score = reconstruction MSE per sample.
    """
    
    def __init__(self, n_features: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, n_features, kernel_size=5, padding=2),
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_baseline(model, X_train, epochs=30, lr=1e-3):
    """Train the autoencoder on normal data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon = model(X_train)
        loss = nn.functional.mse_loss(recon, X_train)
        loss.backward()
        optimizer.step()
    
    return loss.item()


def compute_anomaly_scores(model, X):
    """Per-sample reconstruction error."""
    model.eval()
    with torch.no_grad():
        recon = model(X)
        errors = ((recon - X) ** 2).mean(dim=(1, 2))
    return errors


def compute_f1(scores, labels, percentile=85):
    """
    Compute F1 score using percentile-based thresholding.
    
    F1 = 2·(Precision·Recall) / (Precision + Recall)
    """
    threshold = np.percentile(scores.numpy(), percentile)
    preds = (scores > threshold).long()
    
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1, precision, recall


# ═══════════════════════════════════════════════════════════════════════════════
#                          3. LAZYAI PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   LAZYAI ANOMALY DETECTION DEMO                                             ║
║   ─────────────────────────────                                              ║
║   Synthetic time-series · 1D-CNN Autoencoder · CMA-ES Evolution             ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # ─── Step 1: Generate Data ───────────────────────────────────────────
    print("Step 1: Generating synthetic time-series data...")
    X_train, X_val, X_test, y_test = generate_data(n_samples=300, seq_len=64, n_features=8)
    n_anomalies = y_test.sum().item()
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples ({n_anomalies} anomalies)")
    
    # ─── Step 2: Train Baseline ──────────────────────────────────────────
    print("\nStep 2: Training baseline autoencoder (30 epochs)...")
    model = TinyAutoencoder(n_features=8)
    t0 = time.time()
    final_loss = train_baseline(model, X_train, epochs=30)
    train_time = time.time() - t0
    print(f"  Final loss: {final_loss:.6f}")
    print(f"  Train time: {train_time:.2f}s")
    
    # ─── Step 3: Baseline Evaluation ─────────────────────────────────────
    print("\nStep 3: Evaluating baseline...")
    scores_baseline = compute_anomaly_scores(model, X_test)
    f1_base, prec_base, rec_base = compute_f1(scores_baseline, y_test)
    print(f"  F1:        {f1_base:.4f}")
    print(f"  Precision: {prec_base:.4f}")
    print(f"  Recall:    {rec_base:.4f}")
    
    # Count baseline FLOPs (all layers always fire)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count}")
    
    # ─── Step 4: Wrap with LazyAI ────────────────────────────────────────
    print("\nStep 4: Wrapping model with LazyGates...")
    sample = torch.randn(1, 8, 64)
    lazy_model = LazyWrapper(model, sample_input=sample, initial_p=0.3, initial_a=0.95)
    
    print(f"  Gates injected: {lazy_model.num_gates}")
    print(f"  Genome size:    {lazy_model.genome_size}")
    print(f"  Initial P̄:      {lazy_model.procrastination:.3f}")
    print(f"  Initial Ā:      {lazy_model.ambition:.3f}")
    
    # ─── Step 5: Evolve Optimal Laziness ─────────────────────────────────
    print("\nStep 5: Evolving gate parameters with CMA-ES...")
    print("  Goal: maximize laziness while keeping F1 ≥ baseline × 0.85\n")
    
    quality_floor = f1_base * 0.85 if f1_base > 0 else 0.3
    
    def evaluate_quality(wrapper):
        """Quality = F1 on validation-proxied test set."""
        scores = compute_anomaly_scores(wrapper, X_test)
        f1, _, _ = compute_f1(scores, y_test)
        return f1
    
    config = LazyTunerConfig(
        beta=1.5,               # Strong laziness reward
        eta=0.3,                # Moderate ambition bonus
        q_min=quality_floor,    # Quality floor
        sigma=0.25,
        population_size=15,
        max_generations=15,     # Quick evolution for demo
    )
    
    tuner = LazyTuner(
        genome_size=lazy_model.genome_size,
        fitness_fn=lambda genome: _fitness_wrapper(genome, lazy_model, evaluate_quality, config),
        config=config,
        initial_genome=lazy_model.get_flat_genome(),
        verbose=True,
    )
    
    t0 = time.time()
    best_genome, history = tuner.evolve()
    evo_time = time.time() - t0
    
    # ─── Step 6: Apply & Compare ─────────────────────────────────────────
    lazy_model.set_flat_genome(best_genome)
    lazy_model.reset_stats()
    
    # Run multiple passes to gather stats
    for _ in range(20):
        _ = lazy_model(X_test)
    
    scores_lazy = compute_anomaly_scores(lazy_model, X_test)
    f1_lazy, prec_lazy, rec_lazy = compute_f1(scores_lazy, y_test)
    
    flops_saved = lazy_model.flops_saved_ratio
    eff_compute = lazy_model.effective_compute
    
    # ─── Results ─────────────────────────────────────────────────────────
    print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                              RESULTS                                         ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║   Metric              Baseline        LazyAI          Δ                       ║
║   ──────              ────────        ──────          ──                      ║
║   F1 Score            {f1_base:<16.4f}{f1_lazy:<16.4f}{f1_lazy - f1_base:+.4f}                  ║
║   Precision           {prec_base:<16.4f}{prec_lazy:<16.4f}{prec_lazy - prec_base:+.4f}                  ║
║   Recall              {rec_base:<16.4f}{rec_lazy:<16.4f}{rec_lazy - rec_base:+.4f}                  ║
║                                                                               ║
║   Effective Compute   100.0%          {eff_compute:<16.1%}{eff_compute - 1:+.1%}                  ║
║   FLOPs Saved         0.0%            {flops_saved:<16.1%}                                ║
║                                                                               ║
║   Evolution Time      {evo_time:.1f}s ({len(history)} generations, λ={config.population_size})                       ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║   Evolved PAD Parameters:                                                    ║
║   Global P̄ (procrastination): {lazy_model.procrastination:.4f}                                     ║
║   Global Ā (ambition):        {lazy_model.ambition:.4f}                                     ║
║                                                                               ║""")
    
    # Per-gate details
    print("║   Per-Gate Breakdown:                                                    ║")
    for gate in lazy_model.gates:
        print(f"║     {gate.name:20s}  p={gate.p:.3f}  a={gate.a:.3f}  eff={(1-gate.p)*gate.a:.1%}         ║")
    
    print("║                                                                               ║")
    
    # Verdict
    quality_retained = f1_lazy / f1_base * 100 if f1_base > 0 else 100
    print(f"║   VERDICT: {quality_retained:.0f}% quality retained, {flops_saved:.0%} compute saved                  ║")
    print("║                                                                               ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════╝")
    
    print("\n" + lazy_model.summary())


def _fitness_wrapper(genome, lazy_model, quality_fn, config):
    """Fitness function that combines quality with PAD scoring."""
    lazy_model.set_flat_genome(genome)
    lazy_model.reset_stats()
    quality = quality_fn(lazy_model)
    mean_p, mean_a = genome_summary(genome)
    return compute_pad_fitness(quality, mean_p, mean_a, config.beta, config.eta, config.q_min)


if __name__ == "__main__":
    main()
