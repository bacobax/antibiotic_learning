# Agent Comparison Tool

Compare multiple bacterial control agents (Programmatic, Random, and RL) in a headless simulation environment.

## Usage Example

```bash
python src/compare_agents.py \
    --steps 400 \
    --target-population 100 \
    --rl-checkpoints \
        src/checkpoints/new_margin_penalty/checkpoint_10000.pt \
        src/checkpoints/new_margin_penalty/checkpoint_8000.pt \
        src/checkpoints/new_margin_penalty/checkpoint_6000.pt \
    --rl-runs 5 \
    --seed 42 \
    --verbose \
    --output-plot comparison_results/cross_architecture.png \
    --output-radar comparison_results/cross_architecture_radar.png \
    --output-json comparison_results/cross_architecture_metrics.json
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--steps` | 500 | Number of simulation steps |
| `--target-population` | 100 | Target population for all agents |
| `--initial-budget` | 300.0 | Initial budget for all agents |
| `--tolerance` | 0.15 | Tolerance band around target (fraction) |
| `--zero-distance` | 50.0 | Distance from target where kernel reward = 0 |
| `--population-cap` | 1000 | Early termination if population exceeds this |
| `--rl-checkpoints` | Auto-detected | Paths to RL checkpoint files |
| `--rl-config` | `src/rl/configs/training_config_simple_rewards.yaml` | RL configuration file |
| `--rl-runs` | 1 | Number of runs per RL agent (best Gaussian score selected) |
| `--seed` | Random | Random seed for reproducibility |
| `--output-plot` | `agent_comparison.png` | Path for comparison plot |
| `--output-radar` | `agent_radar_comparison.png` | Path for radar chart |
| `--output-json` | None | Path for metrics JSON |
| `--no-show` | False | Don't display plots (for batch runs) |
| `--verbose` | False | Print progress during simulation |
| `--skip-programmatic` | False | Skip the programmatic agent |
| `--skip-random` | False | Skip the random agent |
