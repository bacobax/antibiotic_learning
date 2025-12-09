"""
Metrics collection and summary computation for agent comparison.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from rl.reward import KernelPopulationMaintenanceReward


@dataclass
class RunMetrics:
    """Collected metrics from a single agent run."""
    agent_name: str
    steps: int = 0
    populations: List[int] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    dose_steps: List[int] = field(default_factory=list)
    count_steps: List[int] = field(default_factory=list)
    sequence_steps: List[int] = field(default_factory=list)
    noop_steps: List[int] = field(default_factory=list)
    dose_amounts: List[float] = field(default_factory=list)
    budget_history: List[float] = field(default_factory=list)
    early_termination_reason: Optional[str] = None
    
    # Summary stats
    initial_population: int = 0
    final_population: int = 0
    min_population: int = 0
    max_population: int = 0
    mean_population: float = 0.0
    std_population: float = 0.0
    
    # Budget
    initial_budget: float = 0.0
    final_budget: float = 0.0
    budget_spent: float = 0.0
    
    # Action counts
    action_counts: Dict[str, int] = field(default_factory=dict)
    
    # Target tracking
    target_population: int = 0
    steps_in_target_band: int = 0
    target_band_ratio: float = 0.0
    
    # Population maintenance quality
    mean_absolute_error: float = 0.0
    mean_squared_error: float = 0.0
    
    # Kernel-based population maintenance scores
    gaussian_kernel_score: float = 0.0
    laplace_kernel_score: float = 0.0
    
    def compute_summary(self, target_population: int, tolerance: float = 0.15, zero_distance: float = 50.0):
        """Compute summary statistics from collected data."""
        self.target_population = target_population
        
        if self.populations:
            self.initial_population = self.populations[0]
            self.final_population = self.populations[-1]
            self.min_population = min(self.populations)
            self.max_population = max(self.populations)
            self.mean_population = float(np.mean(self.populations))
            self.std_population = float(np.std(self.populations))
            
            # Compute how well population was maintained around target
            errors = [abs(p - target_population) for p in self.populations]
            self.mean_absolute_error = float(np.mean(errors))
            self.mean_squared_error = float(np.mean([e**2 for e in errors]))
            
            # Count steps within target band
            lower = target_population * (1 - tolerance)
            upper = target_population * (1 + tolerance)
            self.steps_in_target_band = sum(1 for p in self.populations if lower <= p <= upper)
            self.target_band_ratio = float(self.steps_in_target_band / len(self.populations))
            
            # Compute kernel-based population maintenance scores
            gaussian_kernel = KernelPopulationMaintenanceReward(
                target_population=float(target_population),
                kernel_type="gaussian",
                peak_reward=1.0,
                max_penalty=1.0,
                zero_distance=zero_distance,
            )
            laplace_kernel = KernelPopulationMaintenanceReward(
                target_population=float(target_population),
                kernel_type="laplace",
                peak_reward=1.0,
                max_penalty=1.0,
                zero_distance=zero_distance,
            )
            
            gaussian_sum = sum(gaussian_kernel(p) for p in self.populations)
            laplace_sum = sum(laplace_kernel(p) for p in self.populations)
            
            self.gaussian_kernel_score = float(gaussian_sum / len(self.populations))
            self.laplace_kernel_score = float(laplace_sum / len(self.populations))
        
        if self.budget_history:
            self.initial_budget = float(self.budget_history[0])
            self.final_budget = float(self.budget_history[-1])
            self.budget_spent = float(self.initial_budget - self.final_budget)