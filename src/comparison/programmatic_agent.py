"""
Budget-aware programmatic agent that distributes actions over time.
"""

from typing import Dict, Optional, Tuple
from .base_agent import BaseComparisonAgent, ActionType
from simulation.simulation_config import ANTIBIOTIC_TYPES, TRAIT_KEYS

# Lazy import of RL config loader to optionally sync defaults with RL training config
try:
    from rl.config_loader import load_config, CompleteConfig  # type: ignore
except Exception:
    load_config = None  # type: ignore
    CompleteConfig = object  # type: ignore


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
        sequence_interval: int = 50,
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

        # Fixed cadence
        self.count_interval = count_interval
        self.sequence_interval = sequence_interval
        self.last_count_step = -self.count_interval
        self.last_sequence_step = -self.sequence_interval

        # Latest sequencing output (when available)
        # Expected shape: dict with keys in TRAIT_KEYS (e.g. "enzyme_weight")
        self.last_sequence_traits: Optional[Dict[str, float]] = None

        # Runner will read this when ActionType.DOSE is returned
        self.selected_antibiotic: Optional[str] = None

        self.last_action: Optional[ActionType] = None

    def update_sequence_data(self, traits: Dict[str, float]) -> None:
        """Update the agent with a new sequencing readout."""
        self.last_sequence_traits = {k: float(v) for k, v in traits.items() if k in TRAIT_KEYS}

    def _choose_antibiotic_from_traits(self) -> Optional[str]:
        """Pick the antibiotic that is least affected by the current trait profile.

        We interpret ANTIBIOTIC_TYPES[*][trait_key] as "how effective this resistance
        mechanism is against that antibiotic". Given measured trait strengths, we
        choose the antibiotic that minimizes the weighted sum.
        """
        if not self.last_sequence_traits:
            return None

        # Ensure all expected keys exist (missing traits treated as 0)
        trait_vec = {k: float(self.last_sequence_traits.get(k, 0.0)) for k in TRAIT_KEYS}

        best_ab: Optional[str] = None
        best_score = float("inf")
        for ab_name, ab_def in ANTIBIOTIC_TYPES.items():
            score = 0.0
            for k in TRAIT_KEYS:
                score += trait_vec[k] * float(ab_def.get(k, 0.0))
            if score < best_score:
                best_score = score
                best_ab = ab_name
        return best_ab
    
    def select_action(self, population: int) -> Tuple[ActionType, float]:
        """Select action using proportional control with budget awareness."""
        # Default choice (runner fallback) unless dosing sets it
        self.selected_antibiotic = None

        # Update cooldown
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
        
        # Priority order requested:
        # 1) COUNT (every 5 steps)
        # 2) NOOP
        # 3) DOSE (only if last counted pop exceeds threshold)
        # 4) SEQUENCE (every 50 steps)

        # 1) COUNT periodically (budget-aware)
        steps_since_count = self.step_count - self.last_count_step
        if steps_since_count >= self.count_interval and self.can_afford(ActionType.COUNT):
            self.last_count_step = self.step_count
            self.last_action = ActionType.COUNT
            return ActionType.COUNT, 0.0

        # 2) Default NOOP unless we have a reason to do something else
        action_type: ActionType = ActionType.NOOP
        strength: float = 0.0

        # 3) DOSE decision uses the most recent COUNT measurement
        # (gated to avoid acting on instantaneous population without counting)
        if self.last_count_population is not None:
            error = self.last_count_population - self.target_population
            threshold = self.target_population * self.tolerance
            if error > threshold and self.cooldown_timer == 0:
                raw_strength = self.kp * error
                strength = max(0.2, min(1.0, raw_strength))
                if self.can_afford(ActionType.DOSE, strength):
                    self.selected_antibiotic = self._choose_antibiotic_from_traits()
                    self.cooldown_timer = self.dose_cooldown
                    self.last_action = ActionType.DOSE
                    return ActionType.DOSE, strength

        # 4) SEQUENCE periodically (budget-aware)
        steps_since_sequence = self.step_count - self.last_sequence_step
        if steps_since_sequence >= self.sequence_interval and self.can_afford(ActionType.SEQUENCE):
            self.last_sequence_step = self.step_count
            self.last_action = ActionType.SEQUENCE
            return ActionType.SEQUENCE, 0.0

        self.last_action = action_type
        return action_type, strength