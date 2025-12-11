#!/usr/bin/env python3
"""
Compare multiple agents (RL, Programmatic, Random) without UI visualization.

This script runs agents in headless mode and produces:
- Summary statistics for all agents
- Population over time charts with action markers
- Comparison metrics for scheduling task performance
- Radar chart showing normalized performance metrics

Usage:
    python compare_agents.py --steps 500 --rl-checkpoints checkpoint1.pt:config1.yaml checkpoint2.pt:config2.yaml
    python compare_agents.py --steps 500 --rl-checkpoints checkpoint.pt  # Uses default config
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

import numpy as np

# Add src to path for imports
PROJECT_SRC = Path(__file__).resolve().parent
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from simulation.model import BacteriaModel
from comparison import (
    ProgrammaticAgent,
    RandomAgent,
    RunMetrics,
    run_agent,
    run_rl_agent,
    plot_comparison,
    print_comparison_table,
    plot_radar_chart,
)


def run_rl_agent_multiple_times(
    config_path: str,
    checkpoint_path: str,
    target_population: int,
    initial_budget: float,
    tolerance: float,
    zero_distance: float,
    population_cap: int,
    verbose: bool,
    seed: int,
    num_runs: int = 1,
) -> RunMetrics:
    """Run an RL agent multiple times and return the run with the best Gaussian kernel score."""
    if num_runs <= 1:
        return run_rl_agent(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            target_population=target_population,
            initial_budget=initial_budget,
            tolerance=tolerance,
            zero_distance=zero_distance,
            population_cap=population_cap,
            verbose=verbose,
            seed=seed,
        )
    
    best_metrics = None
    best_score = float('-inf')
    
    for run_idx in range(num_runs):
        # Use different seed for each run
        run_seed = seed + run_idx
        if verbose:
            print(f"  Run {run_idx + 1}/{num_runs} (seed: {run_seed})")
        
        metrics = run_rl_agent(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            target_population=target_population,
            initial_budget=initial_budget,
            tolerance=tolerance,
            zero_distance=zero_distance,
            population_cap=population_cap,
            verbose=False,  # Suppress verbose output for individual runs
            seed=run_seed,
        )
        
        if metrics.gaussian_kernel_score > best_score:
            best_score = metrics.gaussian_kernel_score
            best_metrics = metrics
        
        if verbose:
            print(f"    Gaussian score: {metrics.gaussian_kernel_score:.4f}")
    
    if verbose and num_runs > 1:
        print(f"  Best run: Gaussian score = {best_score:.4f}")
    
    return best_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare multiple agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target-population", type=int, default=100,
        help="Target population for all agents"
    )
    parser.add_argument(
        "--initial-budget", type=float, default=300.0,
        help="Initial budget for all agents"
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.15,
        help="Tolerance band around target (fraction)"
    )
    parser.add_argument(
        "--zero-distance", type=float, default=50.0,
        help="Distance from target where kernel reward equals 0"
    )
    parser.add_argument(
        "--population-cap", type=int, default=1000,
        help="Stop agent if population exceeds this value"
    )
    parser.add_argument(
        "--rl-checkpoints", type=str, nargs='+', default=None,
        help="RL checkpoint:config pairs. Format: 'checkpoint.pt:config.yaml' or just 'checkpoint.pt' (uses default config)"
    )
    parser.add_argument(
        "--rl-config", type=str, default="src/rl/configs/training_config_simple_rewards.yaml",
        help="Default RL config file (used when no config specified in --rl-checkpoints)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-plot", type=str, default=None,
        help="Path to save comparison plot (PNG)"
    )
    parser.add_argument(
        "--output-radar", type=str, default=None,
        help="Path to save radar chart (PNG)"
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save metrics as JSON"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Don't display plots"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print progress during simulation"
    )
    parser.add_argument(
        "--skip-programmatic", action="store_true",
        help="Skip the programmatic agent"
    )
    parser.add_argument(
        "--skip-random", action="store_true",
        help="Skip the random agent"
    )
    parser.add_argument(
        "--rl-runs", type=int, default=1,
        help="Number of runs per RL agent (best Gaussian score is selected)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    seed_for_models = args.seed if args.seed is not None else random.randint(0, 2**31)
    
    # Parse RL checkpoint:config pairs
    rl_checkpoint_configs = []  # List of (checkpoint_path, config_path) tuples
    raw_checkpoints = args.rl_checkpoints or []
    
    if not raw_checkpoints:
        # Try to find default checkpoints
        default_checkpoints = [
            "checkpoints/gaussian/checkpoint_1500.pt",
            "checkpoints/checkpoint_200.pt",
            "src/checkpoints/new_margin_penalty/checkpoint_8000.pt",
        ]
        for ckpt in default_checkpoints:
            if Path(ckpt).exists():
                raw_checkpoints.append(ckpt)
                break
    
    # Parse and validate checkpoint:config pairs
    for entry in raw_checkpoints:
        if ':' in entry:
            # Format: checkpoint_path:config_path
            parts = entry.split(':', 1)
            checkpoint_path = parts[0]
            config_path = parts[1]
        else:
            # Format: checkpoint_path (use default config)
            checkpoint_path = entry
            config_path = args.rl_config
        
        if not Path(checkpoint_path).exists():
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            continue
        if not Path(config_path).exists():
            print(f"Warning: Config not found: {config_path}")
            continue
        
        rl_checkpoint_configs.append((checkpoint_path, config_path))
    
    print(f"Target: {args.target_population}, Budget: {args.initial_budget}")
    print(f"Population cap: {args.population_cap}")
    if rl_checkpoint_configs:
        print(f"RL Agents:")
        for ckpt, cfg in rl_checkpoint_configs:
            print(f"  - Checkpoint: {ckpt}")
            print(f"    Config: {cfg}")
    print()
    
    all_metrics: List[RunMetrics] = []
    
    # Run Programmatic Agent
    if not args.skip_programmatic:
        print("=" * 60)
        print("Running Programmatic Agent...")
        print("=" * 60)
        random.seed(seed_for_models)
        np.random.seed(seed_for_models)
        
        prog_model = BacteriaModel()
        prog_agent = ProgrammaticAgent(
            target_population=args.target_population,
            initial_budget=args.initial_budget,
            count_interval=5
        )
        prog_metrics = run_agent(
            agent=prog_agent,
            model=prog_model,
            target_population=args.target_population,
            tolerance=args.tolerance,
            zero_distance=args.zero_distance,
            population_cap=args.population_cap,
            verbose=args.verbose,
        )
        all_metrics.append(prog_metrics)
        print(f"Programmatic Agent finished: final_pop={prog_metrics.final_population}, budget_spent={prog_metrics.budget_spent:.1f}")
        if prog_metrics.early_termination_reason:
            print(f"  Early termination: {prog_metrics.early_termination_reason}")
    
    # Run Random Agent
    if not args.skip_random:
        print("\n" + "=" * 60)
        print("Running Random Agent...")
        print("=" * 60)
        random.seed(seed_for_models)
        np.random.seed(seed_for_models)
        
        random_model = BacteriaModel()
        random_agent = RandomAgent(
            target_population=args.target_population,
            initial_budget=args.initial_budget,
        )
        random_metrics = run_agent(
            agent=random_agent,
            model=random_model,
            target_population=args.target_population,
            tolerance=args.tolerance,
            zero_distance=args.zero_distance,
            population_cap=args.population_cap,
            verbose=args.verbose,
        )
        all_metrics.append(random_metrics)
        print(f"Random Agent finished: final_pop={random_metrics.final_population}, budget_spent={random_metrics.budget_spent:.1f}")
        if random_metrics.early_termination_reason:
            print(f"  Early termination: {random_metrics.early_termination_reason}")
    
    # Run RL Agents
    for checkpoint_path, config_path in rl_checkpoint_configs:
        checkpoint_name = Path(checkpoint_path).stem
        config_name = Path(config_path).stem
        print("\n" + "=" * 60)
        if args.rl_runs > 1:
            print(f"Running RL Agent ({checkpoint_name} + {config_name}) - {args.rl_runs} runs...")
        else:
            print(f"Running RL Agent ({checkpoint_name} + {config_name})...")
        print("=" * 60)
        random.seed(seed_for_models)
        np.random.seed(seed_for_models)
        
        rl_metrics = run_rl_agent_multiple_times(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            target_population=args.target_population,
            initial_budget=args.initial_budget,
            tolerance=args.tolerance,
            zero_distance=args.zero_distance,
            population_cap=args.population_cap,
            verbose=args.verbose,
            seed=seed_for_models,
            num_runs=args.rl_runs,
        )
        all_metrics.append(rl_metrics)
        print(f"RL Agent finished: final_pop={rl_metrics.final_population}, budget_spent={rl_metrics.budget_spent:.1f}")
        if rl_metrics.early_termination_reason:
            print(f"  Early termination: {rl_metrics.early_termination_reason}")
    
    if not all_metrics:
        print("No agents were run. Check your configuration.")
        sys.exit(1)
    
    # Print comparison
    print_comparison_table(all_metrics)
    
    # Save metrics if requested
    if args.output_json:
        metrics_dict = {
            "agents": [],
            "settings": {
                "target_population": args.target_population,
                "initial_budget": args.initial_budget,
                "tolerance": args.tolerance,
                "zero_distance": args.zero_distance,
                "population_cap": args.population_cap,
                "seed": seed_for_models,
                "rl_config": args.rl_config,
            }
        }
        for m in all_metrics:
            metrics_dict["agents"].append({
                "name": m.agent_name,
                "steps": m.steps,
                "initial_population": m.initial_population,
                "final_population": m.final_population,
                "mean_population": m.mean_population,
                "std_population": m.std_population,
                "min_population": m.min_population,
                "max_population": m.max_population,
                "target_population": m.target_population,
                "steps_in_target_band": m.steps_in_target_band,
                "target_band_ratio": m.target_band_ratio,
                "mean_absolute_error": m.mean_absolute_error,
                "mean_squared_error": m.mean_squared_error,
                "gaussian_kernel_score": m.gaussian_kernel_score,
                "laplace_kernel_score": m.laplace_kernel_score,
                "initial_budget": m.initial_budget,
                "final_budget": m.final_budget,
                "budget_spent": m.budget_spent,
                "action_counts": m.action_counts,
                "early_termination_reason": m.early_termination_reason,
            })
        with open(args.output_json, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"\nMetrics saved to: {args.output_json}")
    
    # Generate plots
    output_path = args.output_plot or "agent_comparison.png"
    plot_comparison(
        all_metrics,
        args.target_population,
        args.tolerance,
        output_path=output_path,
        show=not args.no_show,
    )
    
    # Generate radar chart
    radar_path = args.output_radar or "agent_radar_comparison.png"
    plot_radar_chart(
        all_metrics,
        output_path=radar_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
