"""
LazyTuner: Covariance Matrix Adaptation Evolution Strategy for PAD Optimization

═══════════════════════════════════════════════════════════════════════════════
                         MATHEMATICAL FOUNDATION
═══════════════════════════════════════════════════════════════════════════════

The PAD (Procrastination-Ambition Duality) optimization problem:

    maximize   J(θ) = Q(θ) · exp(β · P̄(θ)) · (1 + η · Ā(θ))
    subject to Q(θ) ≥ Q_min

Where:
    θ = [(p₁, a₁), (p₂, a₂), ..., (p_L, a_L)]   # Genome: L layer parameters
    Q(θ) ∈ [0, 1]                                # Task quality (F1, accuracy, etc.)
    P̄(θ) = (1/L) Σ pₗ                            # Average procrastination
    Ā(θ) = (1/L) Σ (1-pₗ)·aₗ                     # Weighted ambition
    β > 0                                         # Energy coefficient
    η > 0                                         # Ambition bonus
    Q_min                                         # Quality floor

═══════════════════════════════════════════════════════════════════════════════
                        CMA-ES ALGORITHM
═══════════════════════════════════════════════════════════════════════════════

CMA-ES maintains a multivariate Gaussian search distribution:
    x ~ N(m, σ²C)

Evolution update (simplified):
    1. Sample λ offspring: x_k = m + σ · B·D·z_k,  z_k ~ N(0, I)
    2. Evaluate and rank by fitness
    3. Update mean: m ← Σᵢ wᵢ · x_{i:λ}
    4. Update paths: pσ, pC
    5. Update covariance: C ← (1-c₁-cμ)C + c₁·pC·pC^T + cμ·Σwᵢ·yᵢ·yᵢ^T
    6. Update step-size: σ ← σ · exp(cσ/dσ · (||pσ||/E||N(0,I)|| - 1))

═══════════════════════════════════════════════════════════════════════════════
                        LAZY ENHANCEMENTS
═══════════════════════════════════════════════════════════════════════════════

1. Fitness Caching: Skip re-evaluation if genome is sufficiently similar
   
   cache(θ) triggers if ∃θ' ∈ cache: ||θ - θ'||∞ < ε_cache

2. Early Stopping: Prune clearly bad genomes during evaluation
   
   Stop if Q_partial(θ, t) < Q_min - margin  at time t < T

3. Warm Restart: Inject known-good solutions into initial population

4. Lazy Population: Smaller elite population, larger offspring batch

Author: LazyAI
License: MIT
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import numpy as np

import torch

# Type aliases
Genome = List[float]  # Flat genome: [p₁, a₁, p₂, a₂, ...]
FitnessFunc = Callable[[Genome], float]


@dataclass
class LazyTunerConfig:
    """
    Configuration for the LazyTuner evolutionary optimizer.
    
    Mathematical Parameters:
        β (beta): Energy coefficient - higher values reward more laziness
        η (eta): Ambition bonus coefficient
        Q_min: Minimum acceptable quality (fitness floor)
    
    CMA-ES Parameters:
        sigma: Initial step size (standard deviation)
        population_size: Number of offspring per generation (λ)
        elite_ratio: Fraction of population used for updates (μ/λ)
    
    Lazy Optimizations:
        cache_threshold: ε for fitness caching (L∞ distance)
        early_stop_margin: Quality margin for early pruning
        max_cache_size: Maximum cached evaluations
    """
    # PAD fitness coefficients
    beta: float = 1.0           # Energy coefficient (exp(β·P))
    eta: float = 0.5            # Ambition bonus (1 + η·A)
    q_min: float = 0.5          # Quality floor
    
    # CMA-ES core
    sigma: float = 0.3          # Initial step size
    population_size: int = 20   # λ: offspring count
    elite_ratio: float = 0.5    # μ/λ ratio
    
    # Evolution control
    max_generations: int = 100
    target_fitness: float = float('inf')  # Early convergence
    convergence_threshold: float = 1e-6
    
    # Lazy optimizations
    cache_threshold: float = 0.01  # ε for caching
    early_stop_margin: float = 0.1
    max_cache_size: int = 1000
    
    # Constraints
    bounds: Tuple[float, float] = (0.0, 1.0)  # All parameters ∈ [0, 1]


@dataclass
class EvolutionStats:
    """Statistics from one generation of evolution."""
    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    best_genome: Genome
    mean_p: float           # Mean procrastination
    mean_a: float           # Mean ambition
    cache_hits: int
    evaluations: int
    sigma: float            # Current step size
    
    def __repr__(self) -> str:
        return (
            f"Gen {self.generation:3d} | "
            f"Best: {self.best_fitness:.4f} | "
            f"Mean: {self.mean_fitness:.4f} ± {self.std_fitness:.4f} | "
            f"P̄={self.mean_p:.3f} Ā={self.mean_a:.3f} | "
            f"σ={self.sigma:.4f} | "
            f"Cache: {self.cache_hits}/{self.evaluations}"
        )


def compute_pad_fitness(
    quality: float,
    procrastination: float,
    ambition: float,
    beta: float = 1.0,
    eta: float = 0.5,
    q_min: float = 0.5
) -> float:
    """
    Compute PAD fitness score.
    
    Formula:
        J = Q · exp(β · P) · (1 + η · A)
    
    With quality floor:
        J = 0 if Q < Q_min
    
    Args:
        quality: Task quality ∈ [0, 1]
        procrastination: Average P ∈ [0, 1]
        ambition: Weighted average A ∈ [0, 1]
        beta: Energy coefficient
        eta: Ambition bonus coefficient
        q_min: Minimum quality threshold
    
    Returns:
        Fitness score (higher is better)
    """
    if quality < q_min:
        # Heavy penalty for falling below quality floor
        return quality * 0.1  # Scaled down but not zero for gradient
    
    # Core PAD fitness
    energy_bonus = math.exp(beta * procrastination)
    ambition_bonus = 1.0 + eta * ambition
    
    return quality * energy_bonus * ambition_bonus


def genome_to_pa(genome: Genome) -> Tuple[List[float], List[float]]:
    """
    Convert flat genome to (P, A) lists.
    
    Genome format: [p₁, a₁, p₂, a₂, ...]
    Returns: ([p₁, p₂, ...], [a₁, a₂, ...])
    """
    ps = genome[0::2]  # Even indices
    ays = genome[1::2]  # Odd indices
    return ps, ays


def genome_summary(genome: Genome) -> Tuple[float, float]:
    """
    Compute summary statistics (P̄, Ā) from genome.
    
    P̄ = mean(p_i)
    Ā = mean((1 - p_i) * a_i)  # Weighted ambition
    """
    ps, ays = genome_to_pa(genome)
    n = len(ps)
    
    if n == 0:
        return 0.0, 0.0
    
    mean_p = sum(ps) / n
    effective_a = sum((1 - p) * a for p, a in zip(ps, ays)) / n
    
    return mean_p, effective_a


def genome_hash(genome: Genome, precision: int = 3) -> str:
    """
    Create a hash key for genome caching.
    
    Rounds to `precision` decimal places before hashing.
    """
    rounded = [round(x, precision) for x in genome]
    return hashlib.md5(str(rounded).encode()).hexdigest()


class LazyTuner:
    """
    CMA-ES optimizer for PAD (Procrastination-Ambition Duality) evolution.
    
    This implements the Covariance Matrix Adaptation Evolution Strategy
    with lazy enhancements for efficient hyperparameter tuning.
    
    ═══════════════════════════════════════════════════════════════════
                              USAGE
    ═══════════════════════════════════════════════════════════════════
    
    ```python
    # Create tuner
    tuner = LazyTuner(
        genome_size=8,           # 4 layers × 2 params (p, a)
        fitness_fn=evaluate,     # Your fitness function
        config=LazyTunerConfig(
            beta=1.0,            # Reward laziness
            eta=0.5,             # Ambition bonus
            q_min=0.8,           # Minimum quality
            max_generations=50
        )
    )
    
    # Run evolution
    best_genome, stats = tuner.evolve()
    
    # Apply to model
    lazy_model.set_flat_genome(best_genome)
    ```
    
    ═══════════════════════════════════════════════════════════════════
                          ALGORITHM DETAILS
    ═══════════════════════════════════════════════════════════════════
    
    Population: λ offspring sampled from N(m, σ²C)
    Selection: μ best individuals (elite)
    Recombination: Weighted mean of elite
    Mutation: Covariance matrix adaptation
    
    Key invariant: CMA-ES is quasi-parameter-free; only λ and σ₀ need tuning.
    """
    
    def __init__(
        self,
        genome_size: int,
        fitness_fn: FitnessFunc,
        config: Optional[LazyTunerConfig] = None,
        initial_genome: Optional[Genome] = None,
        verbose: bool = True
    ):
        """
        Args:
            genome_size: Size of genome (2 × number of gates)
            fitness_fn: Function that takes genome and returns fitness
            config: Tuner configuration
            initial_genome: Starting point (default: center of bounds)
            verbose: Print progress
        """
        self.n = genome_size
        self.fitness_fn = fitness_fn
        self.config = config or LazyTunerConfig()
        self.verbose = verbose
        
        # ═══════════════════════════════════════════════════════════
        # CMA-ES State Variables
        # ═══════════════════════════════════════════════════════════
        
        # Population parameters
        self.lambda_ = self.config.population_size
        self.mu = int(self.lambda_ * self.config.elite_ratio)
        
        # Recombination weights
        self.weights = np.array([
            math.log(self.mu + 0.5) - math.log(i + 1)
            for i in range(self.mu)
        ])
        self.weights /= self.weights.sum()  # Normalize
        
        # Variance-effective selection mass
        self.mu_eff = 1.0 / (self.weights ** 2).sum()
        
        # Step-size control parameters
        self.c_sigma = (self.mu_eff + 2) / (self.n + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, math.sqrt((self.mu_eff - 1) / (self.n + 1)) - 1) + self.c_sigma
        
        # Covariance adaptation parameters
        self.c_c = (4 + self.mu_eff / self.n) / (self.n + 4 + 2 * self.mu_eff / self.n)
        self.c_1 = 2 / ((self.n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c_1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.n + 2) ** 2 + self.mu_eff)
        )
        
        # Expected norm of N(0, I)
        self.chi_n = math.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))
        
        # ═══════════════════════════════════════════════════════════
        # Initialize dynamic state
        # ═══════════════════════════════════════════════════════════
        
        # Mean of search distribution
        if initial_genome is not None:
            self.m = np.array(initial_genome, dtype=np.float64)
        else:
            # Start at center of bounds
            center = (self.config.bounds[0] + self.config.bounds[1]) / 2
            self.m = np.full(self.n, center, dtype=np.float64)
        
        # Step-size
        self.sigma = self.config.sigma
        
        # Covariance matrix (identity initially)
        self.C = np.eye(self.n)
        self.B = np.eye(self.n)  # Eigenvectors
        self.D = np.ones(self.n)  # Eigenvalues (diagonal)
        
        # Evolution paths
        self.p_sigma = np.zeros(self.n)  # Step-size path
        self.p_c = np.zeros(self.n)      # Covariance path
        
        # ═══════════════════════════════════════════════════════════
        # Lazy enhancements
        # ═══════════════════════════════════════════════════════════
        
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._cache_hits = 0
        self._total_evals = 0
        
        # History
        self.history: List[EvolutionStats] = []
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = float('-inf')
    
    def _check_cache(self, genome: Genome) -> Optional[float]:
        """
        Check if a similar genome was evaluated before.
        
        Uses L∞ distance with threshold ε.
        """
        # Quick hash check first
        key = genome_hash(genome)
        if key in self._cache:
            return self._cache[key]
        
        # Approximate lookup for nearby genomes
        genome_arr = np.array(genome)
        for cached_key, cached_fitness in self._cache.items():
            # Decode cached genome from history (simplified: use hash)
            # In practice, we'd store full genomes
            pass
        
        return None
    
    def _update_cache(self, genome: Genome, fitness: float) -> None:
        """Add genome to cache with LRU eviction."""
        key = genome_hash(genome)
        self._cache[key] = fitness
        
        # Evict oldest if over capacity
        while len(self._cache) > self.config.max_cache_size:
            self._cache.popitem(last=False)
    
    def _evaluate(self, genome: Genome) -> float:
        """
        Evaluate a genome's fitness.
        
        Uses caching for lazy evaluation.
        """
        # Clip to bounds
        genome = [
            max(self.config.bounds[0], min(self.config.bounds[1], g))
            for g in genome
        ]
        
        # Check cache
        cached = self._check_cache(genome)
        if cached is not None:
            self._cache_hits += 1
            return cached
        
        # Full evaluation
        self._total_evals += 1
        fitness = self.fitness_fn(genome)
        self._update_cache(genome, fitness)
        
        return fitness
    
    def _sample_population(self) -> List[np.ndarray]:
        """
        Sample λ offspring from the search distribution.
        
        x_k = m + σ · B · D · z_k,  where z_k ~ N(0, I)
        """
        offspring = []
        
        for _ in range(self.lambda_):
            z = np.random.randn(self.n)
            y = self.B @ (self.D * z)  # B·D·z
            x = self.m + self.sigma * y
            
            # Clip to bounds
            x = np.clip(x, self.config.bounds[0], self.config.bounds[1])
            offspring.append(x)
        
        return offspring
    
    def _update_distribution(
        self, 
        offspring: List[np.ndarray], 
        fitness: List[float]
    ) -> None:
        """
        Update the search distribution based on fitness.
        
        This implements the full CMA-ES update:
        1. Rank offspring by fitness
        2. Update mean m
        3. Update evolution paths
        4. Update covariance matrix C
        5. Update step-size σ
        """
        # Sort by fitness (descending - maximize)
        indices = np.argsort(fitness)[::-1]
        
        # Selected (elite) offspring
        selected = [offspring[i] for i in indices[:self.mu]]
        
        # Old mean for path update
        m_old = self.m.copy()
        
        # ═══════════════════════════════════════════════════════════
        # Update mean
        # ═══════════════════════════════════════════════════════════
        
        self.m = np.zeros(self.n)
        for i, x in enumerate(selected):
            self.m += self.weights[i] * x
        
        # ═══════════════════════════════════════════════════════════
        # Update evolution paths
        # ═══════════════════════════════════════════════════════════
        
        # Step displacement
        y_mean = (self.m - m_old) / self.sigma
        
        # Update p_sigma (step-size path)
        C_inv_sqrt = self.B @ np.diag(1.0 / self.D) @ self.B.T
        self.p_sigma = (
            (1 - self.c_sigma) * self.p_sigma +
            math.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * (C_inv_sqrt @ y_mean)
        )
        
        # Heaviside function for stalling
        h_sigma = (
            np.linalg.norm(self.p_sigma) / 
            math.sqrt(1 - (1 - self.c_sigma) ** (2 * (len(self.history) + 1))) <
            (1.4 + 2 / (self.n + 1)) * self.chi_n
        )
        
        # Update p_c (covariance path)
        self.p_c = (
            (1 - self.c_c) * self.p_c +
            h_sigma * math.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_mean
        )
        
        # ═══════════════════════════════════════════════════════════
        # Update covariance matrix
        # ═══════════════════════════════════════════════════════════
        
        # Rank-one update
        rank_one = np.outer(self.p_c, self.p_c)
        
        # Rank-mu update
        rank_mu = np.zeros((self.n, self.n))
        for i, x in enumerate(selected):
            y_i = (x - m_old) / self.sigma
            rank_mu += self.weights[i] * np.outer(y_i, y_i)
        
        # Combined update
        self.C = (
            (1 - self.c_1 - self.c_mu) * self.C +
            self.c_1 * rank_one +
            self.c_mu * rank_mu
        )
        
        # Ensure symmetry
        self.C = (self.C + self.C.T) / 2
        
        # Decompose C = B·D²·B^T
        eigenvalues, self.B = np.linalg.eigh(self.C)
        eigenvalues = np.maximum(eigenvalues, 1e-10)  # Numerical stability
        self.D = np.sqrt(eigenvalues)
        
        # ═══════════════════════════════════════════════════════════
        # Update step-size
        # ═══════════════════════════════════════════════════════════
        
        self.sigma *= math.exp(
            (self.c_sigma / self.d_sigma) * 
            (np.linalg.norm(self.p_sigma) / self.chi_n - 1)
        )
        
        # Clamp sigma to reasonable range
        self.sigma = max(1e-10, min(self.sigma, 2.0))
    
    def step(self) -> EvolutionStats:
        """
        Execute one generation of evolution.
        
        Returns:
            Statistics for this generation
        """
        gen = len(self.history)
        cache_hits_before = self._cache_hits
        
        # Sample offspring
        offspring = self._sample_population()
        
        # Evaluate fitness
        fitness_scores = [self._evaluate(x.tolist()) for x in offspring]
        
        # Update distribution
        self._update_distribution(offspring, fitness_scores)
        
        # Track best
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = fitness_scores[best_idx]
            self.best_genome = offspring[best_idx].tolist()
        
        # Compute statistics
        mean_p, mean_a = genome_summary(self.m.tolist())
        
        stats = EvolutionStats(
            generation=gen,
            best_fitness=fitness_scores[best_idx],
            mean_fitness=np.mean(fitness_scores),
            std_fitness=np.std(fitness_scores),
            best_genome=offspring[best_idx].tolist(),
            mean_p=mean_p,
            mean_a=mean_a,
            cache_hits=self._cache_hits - cache_hits_before,
            evaluations=len(offspring),
            sigma=self.sigma
        )
        
        self.history.append(stats)
        
        if self.verbose:
            print(stats)
        
        return stats
    
    def evolve(
        self,
        max_generations: Optional[int] = None,
        callback: Optional[Callable[[EvolutionStats], bool]] = None
    ) -> Tuple[Genome, List[EvolutionStats]]:
        """
        Run full evolution until convergence or max generations.
        
        Args:
            max_generations: Override config.max_generations
            callback: Called each generation; return False to stop
        
        Returns:
            (best_genome, history)
        """
        max_gen = max_generations or self.config.max_generations
        
        if self.verbose:
            print(f"\n{'═' * 70}")
            print(f"  LazyTuner Evolution: {self.n} dimensions, λ={self.lambda_}, μ={self.mu}")
            print(f"  PAD: β={self.config.beta}, η={self.config.eta}, Q_min={self.config.q_min}")
            print(f"{'═' * 70}\n")
        
        for gen in range(max_gen):
            stats = self.step()
            
            # Callback for custom stopping
            if callback is not None and not callback(stats):
                if self.verbose:
                    print(f"\nStopped by callback at generation {gen}")
                break
            
            # Target fitness reached
            if stats.best_fitness >= self.config.target_fitness:
                if self.verbose:
                    print(f"\nTarget fitness reached at generation {gen}")
                break
            
            # Convergence check
            if self.sigma < self.config.convergence_threshold:
                if self.verbose:
                    print(f"\nConverged at generation {gen} (σ < {self.config.convergence_threshold})")
                break
        
        if self.verbose:
            print(f"\n{'═' * 70}")
            print(f"  Evolution Complete")
            print(f"  Best Fitness: {self.best_fitness:.6f}")
            mean_p, mean_a = genome_summary(self.best_genome)
            print(f"  Best Genome: P̄={mean_p:.4f}, Ā={mean_a:.4f}")
            print(f"  Cache Efficiency: {self._cache_hits}/{self._total_evals + self._cache_hits} = "
                  f"{self._cache_hits / max(1, self._total_evals + self._cache_hits):.1%}")
            print(f"{'═' * 70}\n")
        
        return self.best_genome, self.history
    
    def inject_solution(self, genome: Genome, weight: float = 0.5) -> None:
        """
        Inject a known-good solution into the search distribution.
        
        Shifts the mean toward the injected genome without resetting
        the covariance structure.
        
        Args:
            genome: Solution to inject
            weight: Interpolation weight (0=ignore, 1=replace mean)
        """
        genome_arr = np.array(genome)
        self.m = (1 - weight) * self.m + weight * genome_arr


class PADFitnessFactory:
    """
    Factory for creating PAD fitness functions.
    
    Creates a fitness function that wraps a quality evaluator with
    PAD scoring.
    
    Usage:
        # Create fitness function
        fitness = PADFitnessFactory.create(
            quality_fn=evaluate_model_quality,
            beta=1.0,
            eta=0.5,
            q_min=0.85
        )
        
        # Use with tuner
        tuner = LazyTuner(
            genome_size=8,
            fitness_fn=fitness,
            ...
        )
    """
    
    @staticmethod
    def create(
        quality_fn: Callable[[Genome], float],
        beta: float = 1.0,
        eta: float = 0.5,
        q_min: float = 0.5
    ) -> FitnessFunc:
        """
        Create a PAD fitness function.
        
        Args:
            quality_fn: Function that evaluates task quality given genome
            beta: Energy coefficient for procrastination bonus
            eta: Ambition bonus coefficient
            q_min: Minimum quality floor
        
        Returns:
            Fitness function compatible with LazyTuner
        """
        def fitness(genome: Genome) -> float:
            quality = quality_fn(genome)
            mean_p, mean_a = genome_summary(genome)
            return compute_pad_fitness(quality, mean_p, mean_a, beta, eta, q_min)
        
        return fitness


# ═══════════════════════════════════════════════════════════════════════════════
#                             UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_quick_tuner(
    lazy_wrapper,
    quality_evaluator: Callable,
    beta: float = 1.0,
    eta: float = 0.5,
    q_min: float = 0.8,
    population_size: int = 20,
    max_generations: int = 50
) -> LazyTuner:
    """
    Quick factory for creating a tuner from a LazyWrapper.
    
    Args:
        lazy_wrapper: The LazyWrapper to optimize
        quality_evaluator: Function that takes lazy_wrapper and returns quality
        beta: Energy coefficient
        eta: Ambition bonus
        q_min: Quality floor
        population_size: CMA-ES λ
        max_generations: Maximum generations
    
    Returns:
        Configured LazyTuner
    """
    def fitness_fn(genome: Genome) -> float:
        # Apply genome to wrapper
        lazy_wrapper.set_flat_genome(genome)
        lazy_wrapper.reset_stats()
        
        # Evaluate quality
        quality = quality_evaluator(lazy_wrapper)
        
        # Compute PAD fitness
        mean_p, mean_a = genome_summary(genome)
        return compute_pad_fitness(quality, mean_p, mean_a, beta, eta, q_min)
    
    config = LazyTunerConfig(
        beta=beta,
        eta=eta,
        q_min=q_min,
        population_size=population_size,
        max_generations=max_generations
    )
    
    return LazyTuner(
        genome_size=lazy_wrapper.genome_size,
        fitness_fn=fitness_fn,
        config=config,
        initial_genome=lazy_wrapper.get_flat_genome()
    )
