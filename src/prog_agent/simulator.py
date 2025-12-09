"""Programmatic control loop that connects BacterialControlAgent with the Mesa model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from simulation.model import BacteriaModel
from simulation.simulation_config import ANTIBIOTIC_TYPES
from .agent import ActionType, BacterialControlAgent


@dataclass
class SequencingJob:
    """Track an in-flight sequencing measurement."""

    remaining: int
    samples: List[Dict[str, float]] = field(default_factory=list)


class ProgrammaticAgentSimulator:
    """Bind the programmatic agent to the simulation model with delayed sequencing."""

    def __init__(
        self,
        model: BacteriaModel,
        agent: BacterialControlAgent,
        *,
        antibiotic_type: Optional[str] = None,
        dose_scale: float = 1.0,
        initial_budget: float = 100.0,
        noop_cost: float = 0.0,
        count_cost: float = 0.5,
        sequencing_cost: float = 2.5,
        dose_cost: float = 2.0,
        dose_cost_per_unit: float = 2.0,
        sequence_window: Optional[int] = None,
        log_interval: int = 50,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.agent = agent
        self.antibiotic_type = (
            antibiotic_type
            if antibiotic_type in ANTIBIOTIC_TYPES
            else model.current_antibiotic
        )
        self.dose_scale = max(0.0, float(dose_scale))
        self.initial_budget = max(0.0, float(initial_budget))
        self.budget_remaining = float(self.initial_budget)
        self.budget_spent = 0.0
        self.noop_cost = float(noop_cost)
        self.count_cost = float(count_cost)
        self.sequencing_cost = float(sequencing_cost)
        self.dose_cost = float(dose_cost)
        self.dose_cost_per_unit = float(dose_cost_per_unit)
        self.sequence_window = (
            int(sequence_window)
            if sequence_window is not None
            else int(agent.sequence_delay)
        )
        self.log_interval = max(1, int(log_interval))
        self.verbose = verbose
        self.sequence_job: Optional[SequencingJob] = None
        self.pending_sequence_result: Optional[Dict[str, float]] = None
        self.step_index = 0
        self.action_counts: Dict[str, int] = {
            action.name: 0 for action in ActionType
        }
        self.last_action: Optional[ActionType] = None
        self.last_action_strength: float = 0.0
        self.sequence_jobs_completed = 0
        self._sync_agent_budget()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, max_steps: int) -> None:
        """Execute the closed loop for the requested number of steps."""
        for _ in range(max_steps):
            self.step()
        if self.verbose:
            self._print_summary()

    def step(self) -> None:
        """Execute a single agent + environment step."""
        observation = self._build_observation()
        action = self.agent.step(observation)
        self._apply_action(action)
        self.model.step()
        self._advance_sequence_job()
        self.step_index += 1

        if self.verbose and self.step_index % self.log_interval == 0:
            self._print_status()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_observation(self) -> Dict[str, Any]:
        obs: Dict[str, Any] = {"population": len(self.model.agent_set)}
        if self.pending_sequence_result is not None:
            obs["sequence_result"] = self.pending_sequence_result
            self.pending_sequence_result = None
        obs["initial_budget"] = self.initial_budget
        obs["budget"] = self.budget_remaining
        obs["budget_spent"] = self.budget_spent
        obs["dose_scale"] = self.dose_scale
        obs["max_affordable_dose_strength"] = self._max_affordable_dose_strength()
        obs["affordable_actions"] = self._get_affordable_actions()
        obs["action_costs"] = self._preview_action_costs()
        return obs

    def _apply_action(self, action: Dict[str, Any]) -> None:
        resolved_type, resolved_strength, action_cost = self._resolve_action_with_budget(
            action.get("type", ActionType.NOOP),
            action.get("strength", 0.0),
        )
        self.last_action = resolved_type
        self.last_action_strength = resolved_strength
        self.action_counts[self.last_action.name] += 1
        self._spend_budget(action_cost)

        if self.last_action == ActionType.DOSE:
            self._apply_dose(resolved_strength)
        elif self.last_action == ActionType.SEQUENCE:
            self._schedule_sequence_job()
        # COUNT and NOOP do not require explicit handling beyond logging

    def _apply_dose(self, normalized_strength: float) -> None:
        if normalized_strength <= 0:
            return
        antibiotic = self.antibiotic_type or self.model.current_antibiotic
        if antibiotic is None:
            return
        amount = normalized_strength * self.dose_scale
        self.model.apply_antibiotic(antibiotic, amount)

    def _schedule_sequence_job(self) -> None:
        if self.sequence_job is None:
            self.sequence_job = SequencingJob(remaining=max(1, self.sequence_window))

    def _advance_sequence_job(self) -> None:
        if self.sequence_job is None:
            return
        stats = self.model.get_population_stats()
        avg_traits = stats.get("avg_traits", {})
        sample = {trait: float(value) for trait, value in avg_traits.items()}
        sample["population"] = float(stats.get("total", 0))
        self.sequence_job.samples.append(sample)
        self.sequence_job.remaining -= 1

        if self.sequence_job.remaining <= 0:
            aggregated = self._aggregate_samples(self.sequence_job.samples)
            self.pending_sequence_result = aggregated
            self.sequence_jobs_completed += 1
            self.sequence_job = None

    @staticmethod
    def _aggregate_samples(samples: List[Dict[str, float]]) -> Dict[str, float]:
        if not samples:
            return {}
        totals: Dict[str, float] = {}
        for sample in samples:
            for key, value in sample.items():
                totals[key] = totals.get(key, 0.0) + float(value)
        count = float(len(samples))
        return {key: value / count for key, value in totals.items()}

    def _print_status(self) -> None:
        population = len(self.model.agent_set)
        seq_eta = self.sequence_job.remaining if self.sequence_job else 0
        msg = (
            f"[ProgrammaticAgent] step={self.step_index} pop={population} "
            f"action={self.last_action.name if self.last_action else 'N/A'}"
        )
        if self.last_action == ActionType.DOSE:
            msg += f" strength={self.last_action_strength:.3f}"
        msg += f" budget={self.budget_remaining:.1f}/{self.initial_budget:.1f}"
        if self.sequence_job is not None:
            msg += f" seq_eta={seq_eta}"
        print(msg)

    def _print_summary(self) -> None:
        summary = self.get_summary()
        action_counts = ", ".join(
            f"{name}:{count}" for name, count in summary["action_counts"].items()
        )
        print(
            "[ProgrammaticAgent] finished after "
            f"{summary['steps']} steps | pop={summary['final_population']} | "
            f"sequences={summary['sequence_jobs_completed']} | "
            f"budget={summary['budget_remaining']:.1f}/{summary['initial_budget']:.1f} | "
            f"{action_counts}"
        )

    def get_summary(self) -> Dict[str, Any]:
        return {
            "steps": self.step_index,
            "final_population": len(self.model.agent_set),
            "antibiotic_type": self.antibiotic_type,
            "dose_scale": self.dose_scale,
            "action_counts": dict(self.action_counts),
            "sequence_jobs_completed": self.sequence_jobs_completed,
            "initial_budget": self.initial_budget,
            "budget_remaining": self.budget_remaining,
            "budget_spent": self.budget_spent,
        }

    def _sync_agent_budget(self) -> None:
        if hasattr(self.agent, "initial_budget"):
            self.agent.initial_budget = float(self.initial_budget)
        if hasattr(self.agent, "current_budget"):
            self.agent.current_budget = float(self.budget_remaining)
        if hasattr(self.agent, "budget_spent"):
            self.agent.budget_spent = float(self.budget_spent)

    def _compute_action_cost(self, action_type: ActionType, strength: float = 0.0) -> float:
        if action_type == ActionType.NOOP:
            return self.noop_cost
        if action_type == ActionType.COUNT:
            return self.count_cost
        if action_type == ActionType.SEQUENCE:
            return self.sequencing_cost
        if action_type == ActionType.DOSE:
            dose_amount = max(0.0, min(1.0, float(strength))) * self.dose_scale
            return self.dose_cost + dose_amount * self.dose_cost_per_unit
        return 0.0

    def _max_affordable_dose_strength(self) -> float:
        remaining = self.budget_remaining - self.dose_cost
        if remaining <= 0.0:
            return 0.0
        denom = self.dose_cost_per_unit * max(self.dose_scale, 1e-8)
        if denom <= 0.0:
            return 1.0 if remaining >= 0.0 else 0.0
        return max(0.0, min(1.0, remaining / denom))

    def _get_affordable_actions(self) -> Dict[str, bool]:
        affordability: Dict[str, bool] = {}
        for action in ActionType:
            strength = 1.0 if action == ActionType.DOSE else 0.0
            cost = self._compute_action_cost(action, strength)
            affordability[action.name] = cost <= (self.budget_remaining + 1e-6)
        return affordability

    def _preview_action_costs(self) -> Dict[str, float]:
        costs: Dict[str, float] = {}
        for action in ActionType:
            strength = self.last_action_strength if action == ActionType.DOSE else 0.0
            costs[action.name] = self._compute_action_cost(action, strength)
        return costs

    def _coerce_action_type(self, candidate: Any) -> ActionType:
        if isinstance(candidate, ActionType):
            return candidate
        if isinstance(candidate, str):
            try:
                return ActionType[candidate]
            except KeyError:
                return ActionType.NOOP
        return ActionType.NOOP

    def _select_affordable_fallback(self) -> ActionType:
        if self._compute_action_cost(ActionType.COUNT) <= (self.budget_remaining + 1e-6):
            return ActionType.COUNT
        return ActionType.NOOP

    def _resolve_action_with_budget(self, candidate_type: Any, strength: Any) -> tuple[ActionType, float, float]:
        resolved_type = self._coerce_action_type(candidate_type)
        resolved_strength = float(strength)
        if resolved_type != ActionType.DOSE:
            resolved_strength = 0.0
        else:
            resolved_strength = max(0.0, min(1.0, resolved_strength))

        cost = self._compute_action_cost(resolved_type, resolved_strength)
        if cost <= (self.budget_remaining + 1e-6):
            return resolved_type, resolved_strength, cost

        if resolved_type == ActionType.DOSE:
            affordable_strength = self._max_affordable_dose_strength()
            if affordable_strength > 0.0:
                resolved_strength = min(resolved_strength, affordable_strength)
                cost = self._compute_action_cost(resolved_type, resolved_strength)
                if cost <= (self.budget_remaining + 1e-6):
                    return resolved_type, resolved_strength, cost

        fallback = self._select_affordable_fallback()
        fallback_cost = self._compute_action_cost(fallback, 0.0)
        if fallback_cost > self.budget_remaining:
            fallback_cost = 0.0
        return fallback, 0.0, fallback_cost

    def _spend_budget(self, amount: float) -> None:
        spend = max(0.0, min(amount, self.budget_remaining))
        self.budget_remaining = max(0.0, self.budget_remaining - spend)
        self.budget_spent += spend
        self._sync_agent_budget()


def run_headless_simulation(
    steps: int = 500,
    target_population: int = 180,
    antibiotic_type: Optional[str] = None,
    dose_scale: float = 0.5,
    initial_budget: float = 100.0,
    noop_cost: float = 0.0,
    count_cost: float = 0.5,
    sequencing_cost: float = 2.5,
    dose_cost: float = 2.0,
    dose_cost_per_unit: float = 2.0,
    sequence_delay: int = 10,
    kp: float = 0.005,
    tolerance: float = 0.15,
) -> ProgrammaticAgentSimulator:
    """Convenience function to spin up a headless simulation for scripting/tests."""
    model = BacteriaModel(enable_individual_tracking=False, max_individual_history=1)
    agent = BacterialControlAgent(
        target_population=target_population,
        kp=kp,
        tolerance=tolerance,
        sequence_delay=sequence_delay,
        initial_budget=initial_budget,
        noop_cost=noop_cost,
        count_cost=count_cost,
        sequence_cost=sequencing_cost,
        dose_cost=dose_cost,
        dose_cost_per_unit=dose_cost_per_unit,
    )
    simulator = ProgrammaticAgentSimulator(
        model,
        agent,
        antibiotic_type=antibiotic_type,
        dose_scale=dose_scale,
        initial_budget=initial_budget,
        noop_cost=noop_cost,
        count_cost=count_cost,
        sequencing_cost=sequencing_cost,
        dose_cost=dose_cost,
        dose_cost_per_unit=dose_cost_per_unit,
        sequence_window=sequence_delay,
    )
    simulator.run(steps)
    return simulator