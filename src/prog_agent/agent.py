import json
import os
from collections import deque
from enum import Enum, auto
from typing import Any, Deque, Dict, Optional, Union

class ActionType(Enum):
    """Discrete set of actions the agent can perform."""
    NOOP = auto()      # Take no action
    COUNT = auto()     # Record population size (active monitoring)
    SEQUENCE = auto()  # Initiate delayed measurement of traits
    DOSE = auto()      # Apply antibiotic

class BacterialControlAgent:
    """
    A programmatic agent that controls bacterial population using a state-machine
    and proportional controller logic.
    
    It observes only the total population count and cannot see hidden states like persistors.
    """
    
    def __init__(self, target_population, log_path="src/prog_agent/prog_agent_logs/agent_log.jsonl", 
                 kp=0.005, dose_cooldown=20, sequence_delay=10, tolerance=0.15,
                 initial_budget=1000.0, noop_cost=0.0, count_cost=0.5, sequence_cost=2.5,
                 dose_cost=2.0, dose_cost_per_unit=2.0):
        """
        Initialize the agent.

        Args:
            target_population (int): The desired stable population count.
            log_path (str): Path to save the interaction logs.
            kp (float): Proportional gain for antibiotic dosing (Strength = Kp * Error).
            dose_cooldown (int): Minimum simulation steps between consecutive doses.
            sequence_delay (int): Number of steps it takes for a sequence result to arrive.
            tolerance (float): Fraction of target population defining the stability zone.
            initial_budget (float): Starting credits for the run (mirrors RL env default of 100).
            noop_cost/count_cost/sequence_cost (float): Fixed action costs.
            dose_cost (float): Base cost for attempting a dose.
            dose_cost_per_unit (float): Variable cost per applied antibiotic unit (pre-scale).
        """
        self.target_population = target_population
        self.log_path = log_path
        self.kp = kp
        self.dose_cooldown = dose_cooldown
        self.sequence_delay = sequence_delay
        self.tolerance = tolerance
        self.initial_budget = float(initial_budget)
        self.current_budget = float(initial_budget)
        self.budget_spent = 0.0
        self.noop_cost = float(noop_cost)
        self.count_cost = float(count_cost)
        self.sequence_cost = float(sequence_cost)
        self.dose_cost = float(dose_cost)
        self.dose_cost_per_unit = float(dose_cost_per_unit)
        self.dose_scale_hint = 1.0
        self.max_affordable_dose_strength = 1.0
        self.affordable_actions: Dict[str, bool] = {
            action.name: True for action in ActionType
        }

        # Internal State / Memory
        self.history = deque(maxlen=100)  # Track last 100 population counts
        self.step_count = 0

        # Sequencing State
        self.is_sequencing = False
        self.sequence_timer = 0
        self.last_traits = None  # Stores the result of the last sequence
        self.sequence_pending_result = None

        # Control State
        self.cooldown_timer = 0
        self.current_pressure = 0.0
        self.last_action = None

        # Initialize logging
        self._init_log()

    def _init_log(self):
        """Ensure log directory exists and initialize file."""
        if os.path.dirname(self.log_path):
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        # Create/Overwrite log file
        with open(self.log_path, 'w') as f:
            pass

    def _parse_observation(self, observation):
        """Extract population count and optional sequence results from observation payload."""
        sequence_payload = None

        if isinstance(observation, dict):
            population = int(observation.get("population", 0))
            sequence_payload = observation.get("sequence_result")
            budget = observation.get("budget")
            if budget is not None:
                self.current_budget = float(budget)
            budget_spent = observation.get("budget_spent")
            if budget_spent is not None:
                self.budget_spent = float(budget_spent)
            initial_budget = observation.get("initial_budget")
            if initial_budget is not None:
                self.initial_budget = float(initial_budget)
            max_dose = observation.get("max_affordable_dose_strength")
            if max_dose is not None:
                self.max_affordable_dose_strength = float(max_dose)
            dose_scale = observation.get("dose_scale")
            if dose_scale is not None:
                self.dose_scale_hint = max(1e-8, float(dose_scale))
            affordable = observation.get("affordable_actions")
            if isinstance(affordable, dict):
                self.affordable_actions = {k: bool(v) for k, v in affordable.items()}
        else:
            population = int(observation)

        if sequence_payload is not None:
            self.last_traits = sequence_payload
            self.sequence_pending_result = sequence_payload
            self.is_sequencing = False
            self.sequence_timer = 0

        return population

    def _estimate_trend(self, window=5):
        """Estimate population change per step using a simple moving difference."""
        if len(self.history) < 2:
            return 0.0

        window = max(2, min(window, len(self.history)))
        recent = list(self.history)[-window:]
        diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        if not diffs:
            return 0.0
        return sum(diffs) / len(diffs)

    def receive_sequence_results(self, traits_data):
        """
        Callback to receive data from a completed sequence action.
        
        Args:
            traits_data (dict): Average traits of the population (e.g., {'enzyme': 0.4, ...})
        """
        self.last_traits = traits_data
        self.sequence_pending_result = traits_data
        self.is_sequencing = False
        self.sequence_timer = 0

    def step(self, observation):
        """
        Main decision loop called every simulation step.

        Args:
            observation: Either an integer population count or a mapping with
                "population" and optional "sequence_result" payload.

        Returns:
            dict: The chosen action, e.g., {"type": ActionType.DOSE, "strength": 0.5}
        """
        self.step_count += 1
        population_count = self._parse_observation(observation)
        self.history.append(population_count)
        
        # 1. Update Internal Timers
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            
        if self.is_sequencing:
            self.sequence_timer -= 1
            if self.sequence_timer <= 0:
                self.is_sequencing = False
                # The simulation is expected to have called receive_sequence_results 
                # or will provide data shortly.

        # 2. Analyze State
        error = population_count - self.target_population
        threshold = self.target_population * self.tolerance
        trend = self._estimate_trend()
        
        action = {"type": ActionType.COUNT, "strength": 0.0}  # Default fallback
        
        # 3. Select Action based on Strategy
        
        # CASE A: Overshoot -> Apply Control (Dose)
        if error > threshold:
            if self.cooldown_timer == 0 and trend >= 0:
                # Proportional Control: Higher error = Stronger dose
                # We clamp the dose between 0.0 and 1.0
                raw_strength = self.kp * error
                strength = max(0.0, min(1.0, raw_strength))
                
                # Only dose if strength is significant
                if strength > 0.01:
                    action = {
                        "type": ActionType.DOSE, 
                        "strength": strength
                    }
                    self.cooldown_timer = self.dose_cooldown
                    self.current_pressure = strength
            else:
                # Cooling down or natural decline already happening, just monitor
                action = {"type": ActionType.COUNT, "strength": 0.0}

        # CASE B: Undershoot -> Recover (NoOp)
        elif error < -threshold:
            # Population is too low, do nothing to allow growth unless it keeps dropping
            action = {"type": ActionType.NOOP, "strength": 0.0}
            if trend < 0:
                # Still falling -> keep a closer eye
                action = {"type": ActionType.COUNT, "strength": 0.0}
            
        # CASE C: Stable -> Observe Evolution (Sequence)
        else:
            # Population is within target range. This is the time to measure evolution.
            if not self.is_sequencing:
                action = {"type": ActionType.SEQUENCE, "strength": 0.0}
                self.is_sequencing = True
                self.sequence_timer = self.sequence_delay
            else:
                # Already sequencing, just wait and count
                action = {"type": ActionType.COUNT, "strength": 0.0}

        # 4. Enforce budget/affordability rules
        action = self._enforce_budget(action)

        # 5. Log Outcome
        self.last_action = action["type"]
        self._log_step(population_count, action, trend)
        
        return action

    def _enforce_budget(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_type = action.get("type", ActionType.NOOP)
        dose_strength = float(action.get("strength", 0.0))

        if action_type == ActionType.DOSE:
            clamped_strength = self._clamp_dose_strength(dose_strength)
            if clamped_strength <= 0.0 and not self._is_affordable(ActionType.DOSE, clamped_strength):
                return self._fallback_action()
            action["strength"] = clamped_strength
            if not self._is_affordable(ActionType.DOSE, clamped_strength):
                return self._fallback_action()
            return action

        if not self._is_affordable(action_type, 0.0):
            return self._fallback_action()
        return action

    def _fallback_action(self) -> Dict[str, Any]:
        for candidate in (ActionType.COUNT, ActionType.NOOP):
            if self._is_affordable(candidate, 0.0):
                return {"type": candidate, "strength": 0.0}
        return {"type": ActionType.NOOP, "strength": 0.0}

    def _clamp_dose_strength(self, requested_strength: float) -> float:
        strength = max(0.0, min(1.0, requested_strength))
        if self.current_budget is None:
            return strength
        max_strength = self._max_affordable_dose_strength()
        return min(strength, max_strength)

    def _max_affordable_dose_strength(self) -> float:
        if self.current_budget is None:
            return 1.0
        remaining = self.current_budget - self.dose_cost
        if remaining <= 0.0:
            return 0.0
        denom = self.dose_cost_per_unit * max(self.dose_scale_hint, 1e-8)
        if denom <= 0.0:
            return 1.0 if remaining >= 0 else 0.0
        return max(0.0, min(1.0, remaining / denom))

    def _compute_action_cost(self, action_type: ActionType, strength: float = 0.0) -> float:
        if action_type == ActionType.NOOP:
            return self.noop_cost
        if action_type == ActionType.COUNT:
            return self.count_cost
        if action_type == ActionType.SEQUENCE:
            return self.sequence_cost
        if action_type == ActionType.DOSE:
            dose_amount = max(0.0, min(1.0, strength)) * max(self.dose_scale_hint, 1e-8)
            return self.dose_cost + dose_amount * self.dose_cost_per_unit
        return 0.0

    def _is_affordable(self, action_type: ActionType, strength: float = 0.0) -> bool:
        if self.current_budget is None:
            return True
        affordability_hint = self.affordable_actions.get(action_type.name)
        if affordability_hint is False:
            return False
        cost = self._compute_action_cost(action_type, strength)
        return cost <= (self.current_budget + 1e-6)

    def _log_step(self, population, action, trend):
        """Log the current step's state and action to JSONL."""
        action_cost = self._compute_action_cost(action["type"], action.get("strength", 0.0))
        entry = {
            "step": self.step_count,
            "population": population,
            "target": self.target_population,
            "error": population - self.target_population,
            "action": action["type"].name,
            "dose_strength": action.get("strength", 0.0),
            "is_sequencing": self.is_sequencing,
            "trend": trend,
            "cooldown_remaining": self.cooldown_timer,
            "sequence_timer": self.sequence_timer if self.is_sequencing else 0,
            "last_action": self.last_action.name if self.last_action else None,
            # Log last known traits if available
            "last_traits": self.last_traits,
            "budget_initial": self.initial_budget,
            "budget_remaining": self.current_budget,
            "budget_spent": self.budget_spent,
            "affordable_actions": self.affordable_actions,
            "max_affordable_dose_strength": self.max_affordable_dose_strength,
            "action_cost_estimate": action_cost,
            "action_affordable": self._is_affordable(action["type"], action.get("strength", 0.0))
        }
        
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")
