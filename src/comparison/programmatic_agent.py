"""
Budget-aware programmatic agent that distributes actions over time.
"""

from typing import Tuple
from .base_agent import BaseComparisonAgent, ActionType


class ProgrammaticAgent(BaseComparisonAgent):
    """
    Budget-aware programmatic agent that distributes actions over time.
    
    Uses proportional control for dosing and schedules actions to avoid
    burning through the budget too quickly.
    """
    
    def __init__(
        self,
        target_population: int,
        initial_budget: float,
        kp: float = 0.005,
        tolerance: float = 0.15,
        dose_cooldown: int = 20,
        count_interval: int = 15,
        **kwargs
    ):
        super().__init__(
            name="Programmatic Agent",
            target_population=target_population,
            initial_budget=initial_budget,
            **kwargs
        )
        self.kp = kp
        self.tolerance = tolerance
        self.dose_cooldown = dose_cooldown
        self.cooldown_timer = 0
        
        # Fixed count interval (not dependent on total steps)
        self.count_interval = count_interval
        self.min_count_interval = max(3, count_interval // 2)
        self.last_count_step = -self.min_count_interval
    
    def select_action(self, population: int) -> Tuple[ActionType, float]:
        """Select action using proportional control with budget awareness."""
        # Update cooldown
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
        
        # Calculate error from target
        error = population - self.target_population
        threshold = self.target_population * self.tolerance
        
        # Check if we need to dose (population too high)
        if error > threshold and self.cooldown_timer == 0:
            # Calculate dose strength proportionally to error
            raw_strength = self.kp * error
            strength = max(0.2, min(1.0, raw_strength))
            
            # Check if we can afford this dose
            if self.can_afford(ActionType.DOSE, strength):
                self.cooldown_timer = self.dose_cooldown
                return ActionType.DOSE, strength
        
        # Count periodically (budget-aware interval)
        steps_since_count = self.step_count - self.last_count_step
        if steps_since_count >= self.count_interval and self.can_afford(ActionType.COUNT):
            self.last_count_step = self.step_count
            return ActionType.COUNT, 0.0
        
        # Default to NOOP
        return ActionType.NOOP, 0.0