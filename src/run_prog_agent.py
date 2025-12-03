"""Headless runner that lets the programmatic control agent drive the simulation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

PROJECT_SRC = Path(__file__).resolve().parent
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from prog_agent.agent import BacterialControlAgent
from prog_agent.simulator import ProgrammaticAgentSimulator
from simulation.model import BacteriaModel
from simulation.simulation_config import ANTIBIOTIC_TYPES

DEFAULT_LOG_PATH = Path("prog_agent/prog_agent_logs/agent_log.jsonl")
ANTIBIOTIC_CHOICES = ["auto"] + sorted(ANTIBIOTIC_TYPES.keys())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the rule-based BacterialControlAgent inside the Mesa simulation",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of simulation steps to execute",
    )
    parser.add_argument(
        "--target-population",
        type=int,
        default=180,
        help="Desired steady-state population size",
    )
    parser.add_argument(
        "--initial-population",
        type=int,
        default=None,
        help="Override the initial bacteria count (defaults to config value)",
    )
    parser.add_argument(
        "--antibiotic",
        type=str,
        choices=ANTIBIOTIC_CHOICES,
        default="auto",
        help="Which antibiotic field to apply when dosing (auto = use model default)",
    )
    parser.add_argument(
        "--dose-scale",
        type=float,
        default=0.5,
        help="Physical antibiotic quantity per unit of agent-selected strength",
    )
    parser.add_argument(
        "--initial-budget",
        type=float,
        default=100.0,
        help="Starting action budget (matches RL environment default of 100)",
    )
    parser.add_argument(
        "--noop-cost",
        type=float,
        default=0.0,
        help="Cost for NOOP actions (usually zero)",
    )
    parser.add_argument(
        "--count-cost",
        type=float,
        default=0.5,
        help="Fixed cost for COUNT actions",
    )
    parser.add_argument(
        "--sequence-cost",
        type=float,
        default=2.5,
        help="Fixed cost for SEQUENCE actions",
    )
    parser.add_argument(
        "--dose-cost",
        type=float,
        default=2.0,
        help="Base cost charged whenever a DOSE is attempted",
    )
    parser.add_argument(
        "--dose-cost-per-unit",
        type=float,
        default=2.0,
        help="Variable cost per antibiotic unit (strength × dose-scale)",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=0.005,
        help="Proportional gain translating population error into dose strength",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.15,
        help="Fractional tolerance band around the target population",
    )
    parser.add_argument(
        "--dose-cooldown",
        type=int,
        default=20,
        help="Minimum number of steps between two antibiotic doses",
    )
    parser.add_argument(
        "--sequence-delay",
        type=int,
        default=10,
        help="Number of steps before sequencing data returns to the agent",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Optional destination for the agent's JSONL interaction log",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Optional path to save a JSON summary of the run",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Print a status line every N steps (ignored when --quiet is set)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress periodic status logs (final summary still prints)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed Python and NumPy RNGs for reproducible runs",
    )
    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        help="Keep individual-level tracking (slower, but useful for debugging)",
    )
    parser.add_argument(
        "--max-individual-history",
        type=int,
        default=100,
        help="History length to keep when tracking individuals is enabled",
    )
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _resolve_path(candidate: Optional[Union[str, Path]]) -> Optional[Path]:
    if candidate is None:
        return None
    path = Path(candidate).expanduser().resolve()
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_summary(summary: Dict[str, Any], summary_path: Optional[str]) -> None:
    if not summary_path:
        return
    path = _resolve_path(summary_path)
    if path is None:
        return
    path.write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = _parse_args()

    if args.seed is not None:
        _seed_everything(args.seed)

    antibiotic_override = None if args.antibiotic == "auto" else args.antibiotic
    log_path = (
        _resolve_path(args.log_path)
        if args.log_path
        else _resolve_path(DEFAULT_LOG_PATH)
    )

    model = BacteriaModel(
        N=args.initial_population,
        enable_individual_tracking=args.enable_tracking,
        max_individual_history=max(1, args.max_individual_history),
    )

    agent_kwargs: Dict[str, Any] = {
        "target_population": args.target_population,
        "kp": args.kp,
        "tolerance": args.tolerance,
        "dose_cooldown": args.dose_cooldown,
        "sequence_delay": args.sequence_delay,
        "initial_budget": args.initial_budget,
        "noop_cost": args.noop_cost,
        "count_cost": args.count_cost,
        "sequence_cost": args.sequence_cost,
        "dose_cost": args.dose_cost,
        "dose_cost_per_unit": args.dose_cost_per_unit,
    }
    if log_path is not None:
        agent_kwargs["log_path"] = str(log_path)
    agent = BacterialControlAgent(**agent_kwargs)

    simulator = ProgrammaticAgentSimulator(
        model,
        agent,
        antibiotic_type=antibiotic_override,
        dose_scale=args.dose_scale,
        initial_budget=args.initial_budget,
        noop_cost=args.noop_cost,
        count_cost=args.count_cost,
        sequencing_cost=args.sequence_cost,
        dose_cost=args.dose_cost,
        dose_cost_per_unit=args.dose_cost_per_unit,
        sequence_window=args.sequence_delay,
        log_interval=args.log_interval,
        verbose=not args.quiet,
    )

    simulator.run(args.steps)
    summary = simulator.get_summary()
    summary["log_path"] = agent.log_path

    header = "\nSummary:" if not args.quiet else "Summary:"
    print(header)
    print(json.dumps(summary, indent=2))

    _save_summary(summary, args.summary_path)


if __name__ == "__main__":
    main()
