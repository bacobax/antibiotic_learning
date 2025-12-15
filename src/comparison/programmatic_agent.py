"""
Budget-aware programmatic agent that distributes actions over time.
"""

from typing import Tuple, Optional, Union
from .base_agent import BaseComparisonAgent, ActionType

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
        # Alignment with RL env long_run.yaml defaults
        t_count_freshness: int = 10,
        max_count_window: int = 25,
        count_min_elapsed: int = 6,
        count_max_elapsed: int = 20,
        critical_ratio: float = 2.0,
        dosing_margin: float = 20.0,
        # Optional RL config path or loaded config to sync parameters
        rl_config: Optional[Union[str, CompleteConfig]] = None,
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

        # If an RL config is provided, override the defaults with values from the YAML
        if rl_config is not None and load_config is not None:
            # Accept either a path (str) or a CompleteConfig instance
            try:
                if isinstance(rl_config, str):
                    cfg = load_config(rl_config)
                else:
                    cfg = rl_config
                # Extract timing & dosing-related defaults
                timing = cfg.environment.timing
                rewards = cfg.environment.rewards
                self.t_count_freshness = timing.t_count_freshness
                self.max_count_window = timing.max_count_window
                self.count_min_elapsed = timing.count_window.min_elapsed
                self.count_max_elapsed = timing.count_window.max_elapsed
                self.critical_ratio = timing.critical_ratio
                if rewards and rewards.informed_dosing is not None:
                    self.dosing_margin = rewards.informed_dosing.dosing_margin
            except Exception:
                # If config loading fails, silently fall back to explicit parameters
                pass
        
        # Fixed count interval (not dependent on total steps)
        # counting timings
        self.count_interval = count_interval
        self.min_count_interval = max(3, count_min_elapsed)
        self.count_min_elapsed = count_min_elapsed
        self.count_max_elapsed = count_max_elapsed
        self.max_count_window = max_count_window
        self.t_count_freshness = t_count_freshness
        # Critical state detection & dosing margin
        self.critical_ratio = critical_ratio
        self.dosing_margin = dosing_margin
        self.last_count_step = -self.min_count_interval
    
    def select_action(self, population: int) -> Tuple[ActionType, float]:
        """Select action using proportional control with budget awareness."""
        # Update cooldown
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
        # Determine what population the agent actually observes.
        # The environment is partially observable for the programmatic agent:
        # the agent only sees the last counted population (`self.last_count_population`)
        # If no count has been performed yet, it should perform a COUNT when reasonable.
        observed_population = self.last_count_population

        # Calculate error from target using observed population when available
        error = None if observed_population is None else (observed_population - self.target_population)
        threshold = self.target_population * self.tolerance
        
        # If we don't have an observed population yet, try to perform a COUNT when it's allowed
        steps_since_count = self.step_count - self.last_count_step
        if observed_population is None:
            if steps_since_count >= self.min_count_interval and self.can_afford(ActionType.COUNT):
                self.last_count_step = self.step_count
                return ActionType.COUNT, 0.0

        # If count is stale (past max_count_window) try to re-count
        if steps_since_count >= self.max_count_window and self.can_afford(ActionType.COUNT):
            self.last_count_step = self.step_count
            return ActionType.COUNT, 0.0

        # Critical detection: if observed pop exceeds target x critical_ratio, dose immediately
        if error is not None and observed_population >= self.target_population * self.critical_ratio and self.cooldown_timer == 0:
            raw_strength = self.kp * (observed_population - self.target_population)
            strength = max(0.05, min(1.0, raw_strength))
            if self.can_afford(ActionType.DOSE, strength):
                self.cooldown_timer = self.dose_cooldown
                return ActionType.DOSE, strength

        # Check if we need to dose (population too high) based on observed population
        # Use dosing margin from RL config as an absolute threshold above target
        dosing_threshold = self.target_population + self.dosing_margin
        if error is not None and observed_population > dosing_threshold and self.cooldown_timer == 0:
            # Calculate dose strength proportionally to error
            raw_strength = self.kp * error
            strength = max(0.05, min(1.0, raw_strength))

            # Check if we can afford this dose
            if self.can_afford(ActionType.DOSE, strength):
                self.cooldown_timer = self.dose_cooldown
                return ActionType.DOSE, strength
        
        # Count periodically (budget-aware interval)
        if steps_since_count >= self.count_interval and self.can_afford(ActionType.COUNT):
            self.last_count_step = self.step_count
            return ActionType.COUNT, 0.0
        
        # Default to NOOP
        return ActionType.NOOP, 0.0