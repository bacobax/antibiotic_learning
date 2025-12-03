"""Regression tests for the programmatic agent simulator."""

from __future__ import annotations

from prog_agent.agent import ActionType
from prog_agent.simulator import run_headless_simulation


def test_run_headless_simulation_emits_summary() -> None:
    simulator = run_headless_simulation(steps=5, target_population=150, sequence_delay=3)
    summary = simulator.get_summary()

    assert summary["steps"] == 5
    assert isinstance(summary["final_population"], int)
    assert summary["dose_scale"] == 0.5
    assert summary["sequence_jobs_completed"] >= 0
    assert set(summary["action_counts"].keys()) == {action.name for action in ActionType}