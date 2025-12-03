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
        return obs

    def _apply_action(self, action: Dict[str, Any]) -> None:
        action_type = action.get("type", ActionType.NOOP)
        strength = float(action.get("strength", 0.0))
        if isinstance(action_type, ActionType):
            self.last_action = action_type
        else:
            # Allow enums encoded as strings
            try:
                self.last_action = ActionType[action_type]
            except Exception:
                self.last_action = ActionType.NOOP
        self.last_action_strength = strength
        self.action_counts[self.last_action.name] += 1

        if self.last_action == ActionType.DOSE:
            self._apply_dose(strength)
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
            f"sequences={summary['sequence_jobs_completed']} | {action_counts}"
        )

    def get_summary(self) -> Dict[str, Any]:
        return {
            "steps": self.step_index,
            "final_population": len(self.model.agent_set),
            "antibiotic_type": self.antibiotic_type,
            "dose_scale": self.dose_scale,
            "action_counts": dict(self.action_counts),
            "sequence_jobs_completed": self.sequence_jobs_completed,
        }


def run_headless_simulation(
    steps: int = 500,
    target_population: int = 180,
    antibiotic_type: Optional[str] = None,
    dose_scale: float = 0.5,
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
    )
    simulator = ProgrammaticAgentSimulator(
        model,
        agent,
        antibiotic_type=antibiotic_type,
        dose_scale=dose_scale,
        sequence_window=sequence_delay,
    )
    simulator.run(steps)
    return simulator