"""
Random agent that performs casual, tiny distributed doses and counts.
"""

import random
from typing import Tuple
from .base_agent import BaseComparisonAgent, ActionType


class RandomAgent(BaseComparisonAgent):
    """
    Random agent that performs casual, tiny distributed doses and counts.
    
    Actions are distributed randomly over time with small dose strengths
    to maintain some semblance of control without intelligent planning.
    """
    
    def __init__(
        self,
        target_population: int,
        total_steps: int,
        initial_budget: float,
        dose_probability: float = 0.05,
        count_probability: float = 0.1,
        min_dose_strength: float = 0.05,
        max_dose_strength: float = 0.2,
        **kwargs
    ):
        super().__init__(
            name="Random Agent",
            target_population=target_population,
            total_steps=total_steps,
            initial_budget=initial_budget,
            **kwargs
        )
        self.dose_probability = dose_probability
        self.count_probability = count_probability
        self.min_dose_strength = min_dose_strength
        self.max_dose_strength = max_dose_strength
        
        # Adjust probabilities based on budget
        self._calibrate_probabilities()
    
    def _calibrate_probabilities(self):
        """Adjust probabilities to ensure budget lasts the whole simulation."""
        # Estimate expected costs
        avg_dose_strength = (self.min_dose_strength + self.max_dose_strength) / 2
        avg_dose_cost = self.dose_cost + avg_dose_strength * self.dose_scale * self.dose_cost_per_unit
        
        expected_dose_cost = self.dose_probability * self.total_steps * avg_dose_cost
        expected_count_cost = self.count_probability * self.total_steps * self.count_cost
        expected_total_cost = expected_dose_cost + expected_count_cost
        
        # Scale down probabilities if expected cost exceeds budget
        if expected_total_cost > self.initial_budget * 0.9:
            scale = (self.initial_budget * 0.9) / expected_total_cost
            self.dose_probability *= scale
            self.count_probability *= scale
    
    def select_action(self, population: int) -> Tuple[ActionType, float]:
        """Select action randomly with calibrated probabilities."""
        rand = random.random()
        
        # Try to dose
        if rand < self.dose_probability:
            strength = random.uniform(self.min_dose_strength, self.max_dose_strength)
            if self.can_afford(ActionType.DOSE, strength):
                return ActionType.DOSE, strength
        
        # Try to count
        if rand < self.dose_probability + self.count_probability:
            if self.can_afford(ActionType.COUNT):
                return ActionType.COUNT, 0.0
        
        # Default to NOOP
        return ActionType.NOOP, 0.0