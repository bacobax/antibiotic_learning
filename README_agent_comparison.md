# Agent Comparison Tool

This tool compares multiple bacterial control agents (Programmatic, Random, and RL) in a headless simulation environment. It generates comprehensive statistics, population charts, and radar visualizations.

## Features

- **Multiple Agent Types**: Programmatic, Random/Casual, and Reinforcement Learning agents
- **Budget-Aware Scheduling**: Agents distribute actions over time to avoid burning through budget
- **Population Capping**: Early termination if population exceeds threshold
- **Multi-Checkpoint Support**: Compare multiple RL models simultaneously
- **Comprehensive Metrics**: Gaussian/Laplace kernel scores, target tracking, budget efficiency
- **Dual Visualization**: Time-series plots + radar chart for performance comparison

## Quick Start

```bash
# Basic comparison with all three agent types
python src/compare_agents.py --steps 500 --target-population 100 --seed 42

# Save results to comparison_results folder
python src/compare_agents.py \
    --steps 100 \
    --target-population 100 \
    --seed 42 \
    --no-show \
    --output-plot comparison_results/basic_comparison.png \
    --output-radar comparison_results/performance_radar.png \
    --output-json comparison_results/metrics.json
```

## Command Examples

### 1. Multiple RL Checkpoint Configurations

#### Training Progression Analysis
```bash
# Compare early vs late training checkpoints
python src/compare_agents.py \
    --steps 200 \
    --target-population 100 \
    --rl-checkpoints src/checkpoints/gaussian/checkpoint_325.pt src/checkpoints/gaussian/checkpoint_1500.pt \
    --seed 42 \
    --output-plot comparison_results/training_progression.png \
    --output-radar comparison_results/training_progression_radar.png
```

#### Cross-Architecture Comparison with Multiple Runs
```bash
# Compare different training approaches with 5 runs per RL agent (best run selected)
python src/compare_agents.py \
    --steps 500 \
    --target-population 100 \
    --rl-checkpoints \
        src/checkpoints/gaussian/checkpoint_final_1000.pt \
        src/checkpoints/new_margin_penalty/checkpoint_8000.pt \
        src/checkpoints/improved_cont_head/checkpoint_3500.pt \
    --rl-runs 5 \
    --seed 42 \
    --verbose \
    --output-plot comparison_results/cross_architecture.png \
    --output-radar comparison_results/cross_architecture_radar.png \
    --output-json comparison_results/cross_architecture_metrics.json
```

### 2. Step Count Analysis

```bash
# Quick validation (150 steps)
python src/compare_agents.py \
    --steps 150 \
    --target-population 100 \
    --rl-checkpoints src/checkpoints/gaussian/checkpoint_1500.pt \
    --seed 42 \
    --output-plot comparison_results/quick_150.png \
    --output-radar comparison_results/quick_150_radar.png

# Standard test (500 steps)
python src/compare_agents.py \
    --steps 500 \
    --target-population 100 \
    --rl-checkpoints src/checkpoints/new_margin_penalty/checkpoint_8000.pt \
    --seed 42 \
    --output-plot comparison_results/standard_500.png \
    --output-radar comparison_results/standard_500_radar.png

# Extended test (1000 steps)
python src/compare_agents.py \
    --steps 1000 \
    --target-population 100 \
    --rl-checkpoints src/checkpoints/new_margin_penalty/checkpoint_10000.pt \
    --seed 42 \
    --verbose \
    --output-plot comparison_results/extended_1000.png \
    --output-radar comparison_results/extended_1000_radar.png \
    --output-json comparison_results/extended_1000_metrics.json
```

### 3. Budget Analysis

```bash
# Low budget scenario
python src/compare_agents.py \
    --steps 300 \
    --target-population 100 \
    --initial-budget 150.0 \
    --rl-checkpoints src/checkpoints/gaussian/checkpoint_1500.pt \
    --seed 42 \
    --output-plot comparison_results/low_budget.png \
    --output-radar comparison_results/low_budget_radar.png

# High budget scenario
python src/compare_agents.py \
    --steps 500 \
    --target-population 100 \
    --initial-budget 600.0 \
    --rl-checkpoints src/checkpoints/new_margin_penalty/checkpoint_8000.pt \
    --seed 42 \
    --output-plot comparison_results/high_budget.png \
    --output-radar comparison_results/high_budget_radar.png
```

### 4. Population Cap Testing

```bash
# Population cap testing
python src/compare_agents.py \
    --steps 400 \
    --target-population 100 \
    --population-cap 200 \
    --rl-checkpoints src/checkpoints/gaussian/checkpoint_final_1000.pt \
    --seed 42 \
    --verbose \
    --output-plot comparison_results/population_cap.png \
    --output-radar comparison_results/population_cap_radar.png
```

### 5. Batch Analysis

#### Step Count Sweep
```bash
# Same model, different durations
for steps in 200 500 1000; do
    python src/compare_agents.py \
        --steps $steps \
        --target-population 100 \
        --rl-checkpoints src/checkpoints/new_margin_penalty/checkpoint_8000.pt \
        --seed 42 \
        --no-show \
        --output-plot comparison_results/steps_${steps}.png \
        --output-radar comparison_results/steps_${steps}_radar.png
done
```

#### Checkpoint Comparison
```bash
# Different models, same duration
checkpoints=(
    "src/checkpoints/gaussian/checkpoint_final_1000.pt"
    "src/checkpoints/new_margin_penalty/checkpoint_10000.pt"
    "src/checkpoints/improved_cont_head/checkpoint_3500.pt"
)

for checkpoint in "${checkpoints[@]}"; do
    model_name=$(basename "$checkpoint" .pt)
    python src/compare_agents.py \
        --steps 400 \
        --target-population 100 \
        --rl-checkpoints "$checkpoint" \
        --seed 42 \
        --no-show \
        --output-plot comparison_results/${model_name}.png \
        --output-radar comparison_results/${model_name}_radar.png
done
```

### 6. Agent-Specific Testing

```bash
# Programmatic vs RL only
python src/compare_agents.py \
    --steps 400 \
    --target-population 100 \
    --skip-random \
    --rl-checkpoints src/checkpoints/new_margin_penalty/checkpoint_10000.pt \
    --seed 42 \
    --output-plot comparison_results/prog_vs_rl.png \
    --output-radar comparison_results/prog_vs_rl_radar.png

# Random baseline
python src/compare_agents.py \
    --steps 300 \
    --target-population 100 \
    --skip-programmatic \
    --rl-checkpoints src/checkpoints/gaussian/checkpoint_final_1000.pt \
    --seed 42 \
    --output-plot comparison_results/random_baseline.png \
    --output-radar comparison_results/random_baseline_radar.png
```

### 7. Kernel Score Comparison
```bash
# Different zero-distance values for kernel metrics
for zero_dist in 25.0 50.0 75.0 100.0; do
    python src/compare_agents.py \
        --steps 200 \
        --target-population 150 \
        --zero-distance $zero_dist \
        --seed 42 \
        --no-show \
        --output-plot comparison_results/kernel_${zero_dist}_comparison.png \
        --output-radar comparison_results/kernel_${zero_dist}_radar.png
done
```

## Output Files

### 1. Population Time-Series Plot (`*_comparison.png`)
- Individual population trajectories for each agent
- Action markers (DOSE in red, COUNT in purple)
- Target population line with tolerance band
- Budget usage over time
- Action distribution bar chart

### 2. Radar Chart (`*_radar.png`)
- **Gaussian Score**: Population maintenance quality (higher = better)
- **Target Band %**: Percentage of time within target tolerance (higher = better) 
- **Budget Efficiency**: Proportion of budget remaining (higher = better)
- **Population Stability**: Inverse of coefficient of variation (higher = better)
- **Low Error**: Inverse of normalized mean absolute error (higher = better)

### 3. Metrics JSON (`*_metrics.json`)
- Detailed numerical results for all agents
- Simulation settings and parameters
- Raw time-series data for further analysis

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

## Interpreting Results

### Radar Chart Metrics (All Higher = Better)

1. **Gaussian Score**: Population maintenance quality using Gaussian kernel
   - Near 1.0 = excellent population control
   - Near 0.0 = poor population control

2. **Target Band %**: Time spent within tolerance of target population
   - 100% = always within target band
   - 0% = never within target band

3. **Budget Efficiency**: Remaining budget as fraction of initial
   - 1.0 = used no budget (too conservative)
   - 0.0 = spent all budget (potentially wasteful)
   - Optimal range: 0.7-0.9

4. **Population Stability**: Inverse of coefficient of variation
   - 1.0 = perfectly stable population
   - 0.0 = highly variable population

5. **Low Error**: Inverse of normalized mean absolute error
   - 1.0 = zero error from target
   - 0.0 = very high error from target

### Expected Performance Hierarchy
1. **Programmatic Agent**: Usually best overall due to intelligent control
2. **RL Agent**: Performance depends on training quality and checkpoint
3. **Random Agent**: Baseline performance, should be worst overall

## Troubleshooting

### Common Issues

1. **ImportError**: Ensure you're running from the project root directory
2. **FileNotFoundError**: Check that checkpoint paths exist
3. **Memory Issues**: Reduce `--steps` for long simulations

### Performance Tips

- Use `--no-show` for batch experiments
- Set `--seed` for reproducible results  
- Use `--verbose` to monitor progress on long runs
- Save JSON output for detailed post-analysis

## Example Batch Script

```bash
#!/bin/bash
# Run comprehensive comparison suite

mkdir -p comparison_results/batch_$(date +%Y%m%d_%H%M%S)
BATCH_DIR="comparison_results/batch_$(date +%Y%m%d_%H%M%S)"

echo "Running agent comparison batch..."

# Standard comparison
python src/compare_agents.py \
    --steps 300 --target-population 150 --seed 42 --no-show \
    --output-plot $BATCH_DIR/standard.png \
    --output-radar $BATCH_DIR/standard_radar.png \
    --output-json $BATCH_DIR/standard.json

# Low budget scenario  
python src/compare_agents.py \
    --steps 300 --target-population 150 --initial-budget 200 --seed 42 --no-show \
    --output-plot $BATCH_DIR/low_budget.png \
    --output-radar $BATCH_DIR/low_budget_radar.png \
    --output-json $BATCH_DIR/low_budget.json

# High stress scenario
python src/compare_agents.py \
    --steps 300 --target-population 200 --population-cap 350 --seed 42 --no-show \
    --output-plot $BATCH_DIR/high_stress.png \
    --output-radar $BATCH_DIR/high_stress_radar.png \
    --output-json $BATCH_DIR/high_stress.json

echo "Batch complete. Results in $BATCH_DIR"
```

This tool provides a comprehensive framework for evaluating and comparing bacterial control strategies across multiple performance dimensions.