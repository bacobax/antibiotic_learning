# Training Configuration

This directory contains all YAML configuration files for the Recurrent PPO training.

## Configuration Files

### `training_config.yaml` (Default)
Standard production configuration with:
- 100 PPO updates
- 2048 steps per rollout
- 256 hidden dimension
- 1 RNN layer
- CPU device

Use with:
```bash
python -m rl.train
# or explicitly:
python -m rl.train --config rl/config/training_config.yaml
```

### `training_config_fast.yaml` (Testing)
Fast configuration for quick testing:
- 10 PPO updates
- 512 steps per rollout
- 128 hidden dimension
- 200 max steps per episode
- CPU device

Use with:
```bash
python -m rl.train --config rl/config/training_config_fast.yaml
```

### `training_config_production.yaml` (Full Training)
Production-grade configuration:
- 200 PPO updates
- 4096 steps per rollout
- 512 hidden dimension
- 2 RNN layers
- Smaller learning rate (1e-4)
- 128 sequence length
- CPU device

Use with:
```bash
python -m rl.train --config rl/config/training_config_production.yaml
```

## Configuration Structure

All YAML files have this structure:

```yaml
environment:          # Environment and simulation parameters
  max_steps: 1000
  k_doses: 3
  target_population: 500
  # ... more params

actions:             # Action costs and durations
  noop:
    cost: 0.0
    duration: 0
  sequencing:
    cost: 1.0
    duration: 5
  dose:
    cost_per_unit: 0.2
  # ... more actions

model:               # Neural network architecture
  hidden_dim: 256
  rnn_layers: 1
  # ... more params

ppo:                 # PPO algorithm hyperparameters
  gamma: 0.99
  gae_lambda: 0.95
  lr: 3e-4
  # ... more params

training:            # Training execution setup
  total_updates: 100
  seed: 42
  save_dir: "./checkpoints"
  # ... more params
```

## Creating Custom Configurations

1. Copy one of the existing configs:
```bash
cp training_config.yaml my_experiment.yaml
```

2. Edit the parameters:
```yaml
environment:
  max_steps: 5000
  device: "mps"  # or "cuda" if you have a GPU
  # ... modify as needed
```

3. Run training with your config:
```bash
python -m rl.train --config rl/config/my_experiment.yaml
```

## Key Parameters to Modify

### For GPU Training
```yaml
environment:
  device: "cuda"  # or "mps" for Apple Silicon
```

### For Faster Training (less stable)
```yaml
ppo:
  lr: 1e-3          # Higher learning rate
  rollout_steps: 1024  # Fewer rollout steps
  epochs: 2         # Fewer PPO epochs

training:
  total_updates: 50  # Fewer updates
```

### For More Stable Training (slower)
```yaml
ppo:
  lr: 1e-5          # Lower learning rate
  rollout_steps: 4096  # More rollout steps
  epochs: 8         # More PPO epochs
  clip_eps: 0.1     # Tighter clipping

training:
  total_updates: 500  # More updates
```

### For Different Reward Structure
```yaml
environment:
  w_pop: 2.0        # More weight on population control
  w_genome: 0.2     # Less weight on resistance reduction
  w_cost: 0.02      # Lower cost penalty
```

### For Different Action Costs
```yaml
actions:
  sequencing:
    cost: 0.5       # Make sequencing cheaper
  dose:
    cost_per_unit: 0.1  # Make dosing cheaper
```

## Loading Configurations Programmatically

```python
from rl.config_loader import load_config

# Load from rl/config/ directory
config = load_config("training_config.yaml")

# Load from absolute path
config = load_config("/path/to/my_config.yaml")

# Load defaults (if file not found or None)
config = load_config()

# Access parameters
print(config.environment.target_population)
print(config.ppo.lr)
print(config.actions.sequencing_cost)
```

## Validation

All configurations are automatically validated when loaded. Invalid values will raise a `ValueError` with a descriptive message.

Common validation errors:
- `k_doses must be > 0`: Number of antibiotics must be positive
- `total_updates must be > 0`: Number of updates must be positive
- `gamma must be in (0, 1)`: Discount factor out of range
- `lr must be > 0`: Learning rate must be positive
- `device must be 'cpu', 'cuda', or 'mps'`: Invalid device specified

## Tips

1. **Start Simple**: Begin with `training_config_fast.yaml` to test your setup quickly.

2. **Iterate**: Use the fast config to verify training works, then scale up to production.

3. **Monitor**: Check TensorBoard while training:
   ```bash
   tensorboard --logdir=./checkpoints --port=6006
   ```

4. **Save Results**: All configs are saved to checkpoints directory after each run.

5. **Reproducibility**: Use the same seed for reproducible results:
   ```yaml
   training:
     seed: 42
   ```
