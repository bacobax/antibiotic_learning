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
    
    The simulation terminates when budget is exhausted, so probabilities
    are fixed rather than calibrated based on total steps.
    """
    
    def __init__(
        self,
        target_population: int,
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
            initial_budget=initial_budget,
            **kwargs
        )
        self.dose_probability = dose_probability
        self.count_probability = count_probability
        self.min_dose_strength = min_dose_strength
        self.max_dose_strength = max_dose_strength
    
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