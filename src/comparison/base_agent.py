"""
Action types and abstract base class for comparison agents.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Optional, Tuple


class ActionType(Enum):
    """Discrete set of actions any agent can perform."""
    NOOP = auto()
    COUNT = auto()
    SEQUENCE = auto()
    DOSE = auto()


class BaseComparisonAgent(ABC):
    """Abstract base class for all comparison agents."""
    
    def __init__(
        self,
        name: str,
        target_population: int,
        initial_budget: float,
        noop_cost: float = 0.0,
        count_cost: float = 0.5,
        sequence_cost: float = 2.5,
        dose_cost: float = 2.0,
        dose_cost_per_unit: float = 2.0,
        dose_scale: float = 0.5,
        **kwargs,  # Accept but ignore extra args like total_steps for backward compatibility
    ):
        self.name = name
        self.target_population = target_population
        self.initial_budget = initial_budget
        self.budget_remaining = initial_budget
        
        # Cost parameters
        self.noop_cost = noop_cost
        self.count_cost = count_cost
        self.sequence_cost = sequence_cost
        self.dose_cost = dose_cost
        self.dose_cost_per_unit = dose_cost_per_unit
        self.dose_scale = dose_scale
        
        # State tracking
        self.step_count = 0
        self.last_count_population: Optional[int] = None
        self.action_counts = {action.name: 0 for action in ActionType}
    
    def is_budget_exhausted(self) -> bool:
        """Check if budget is exhausted (can't afford even the cheapest non-noop action)."""
        min_action_cost = min(self.count_cost, self.dose_cost)
        return self.budget_remaining < min_action_cost
    
    def compute_action_cost(self, action_type: ActionType, dose_strength: float = 0.0) -> float:
        """Compute the cost of an action."""
        if action_type == ActionType.NOOP:
            return self.noop_cost
        elif action_type == ActionType.COUNT:
            return self.count_cost
        elif action_type == ActionType.SEQUENCE:
            return self.sequence_cost
        elif action_type == ActionType.DOSE:
            dose_amount = max(0.0, min(1.0, dose_strength)) * self.dose_scale
            return self.dose_cost + dose_amount * self.dose_cost_per_unit
        return 0.0
    
    def can_afford(self, action_type: ActionType, dose_strength: float = 0.0) -> bool:
        """Check if the agent can afford an action."""
        cost = self.compute_action_cost(action_type, dose_strength)
        return cost <= self.budget_remaining + 1e-6
    
    def spend_budget(self, action_type: ActionType, dose_strength: float = 0.0) -> float:
        """Deduct the cost of an action from budget. Returns cost spent."""
        cost = self.compute_action_cost(action_type, dose_strength)
        self.budget_remaining = max(0.0, self.budget_remaining - cost)
        return cost
    
    @abstractmethod
    def select_action(self, population: int) -> Tuple[ActionType, float]:
        """
        Select an action based on current population.
        
        Args:
            population: Current population count
            
        Returns:
            Tuple of (action_type, dose_strength)
        """
        pass
    
    def step(self, population: int) -> Tuple[ActionType, float]:
        """Execute one step: select action, update state, return action."""
        action_type, dose_strength = self.select_action(population)
        
        # Enforce budget constraints
        if not self.can_afford(action_type, dose_strength):
            # Fall back to cheaper actions
            for fallback in [ActionType.COUNT, ActionType.NOOP]:
                if self.can_afford(fallback):
                    action_type = fallback
                    dose_strength = 0.0
                    break
            else:
                action_type = ActionType.NOOP
                dose_strength = 0.0
        
        # Update state
        self.spend_budget(action_type, dose_strength)
        self.action_counts[action_type.name] += 1
        self.step_count += 1
        
        if action_type == ActionType.COUNT:
            self.last_count_population = population
        
        return action_type, dose_strength