# Recurrent PPO for Bacteria Antibiotic Learning

Complete implementation of Recurrent PPO with hybrid action space for the bacteria simulation.

## Overview

This package provides a fully integrable RL training pipeline that wraps the existing Mesa bacteria simulation **without modifying it**. The implementation supports:

- **Hybrid Action Space**: Discrete action selection (NOOP, COUNT, SEQUENCING, DOSE) + continuous dose vector [0,1]^K
- **Recurrent Policy**: GRU-based actor-critic for handling partial observability
- **Truncated BPTT**: Sequential training with no timestep shuffling
- **GAE(λ)**: Generalized Advantage Estimation for variance reduction
- **Clean Integration**: Thin wrapper around Mesa model with no breaking changes

## Quick Start

### 1. Smoke Test (Mock Environment)

Run a quick test with a mock environment:

```bash
python -m rl.train \
    --mock \
    --k-doses 3 \
    --total-updates 1 \
    --steps-per-rollout 256 \
    --seq-len 64 \
    --device cpu
```

### 2. Train on Real Simulation

Integrate with the actual Mesa bacteria model:

```bash
python -m rl.train \
    --k-doses 3 \
    --total-updates 100 \
    --steps-per-rollout 2048 \
    --seq-len 64 \
    --device cuda
```

## Integration Guide

To connect the RL agent with your Mesa simulation, you need two simple functions:

### 1. Model Factory

Creates a fresh Mesa model instance:

```python
def build_mesa_model():
    from model import BacteriaModel
    return BacteriaModel()
```

### 2. Observation Builder

Extracts state as a flat numpy array:

```python
def build_observation(model) -> np.ndarray:
    obs = []
    
    # Population stats
    population = len(model.agent_set)
    obs.append(population / 100.0)  # Normalize
    
    # Average resistance traits
    if population > 0:
        obs.append(np.mean([b.enzyme for b in model.agent_set]))
        obs.append(np.mean([b.efflux for b in model.agent_set]))
        obs.append(np.mean([b.membrane for b in model.agent_set]))
        obs.append(np.mean([b.repair for b in model.agent_set]))
    else:
        obs.extend([0.0, 0.0, 0.0, 0.0])
    
    # Food level
    total_food = np.sum(model.food_field)
    obs.append(total_food / 1000.0)
    
    # Antibiotic concentrations
    for ab_field in model.antibiotic_fields.values():
        obs.append(np.mean(ab_field))
    
    # Time
    obs.append(model.step_count / 1000.0)
    
    return np.array(obs, dtype=np.float32)
```

### 3. Create Environment Wrapper

```python
from rl.env_wrapper import PetriEnvWrapper

env = PetriEnvWrapper(
    mesa_model_factory=build_mesa_model,
    k_doses=3,  # Number of antibiotic types
    obs_builder=build_observation,
    scale_dose=lambda x: x * 2.0,  # Scale [0,1] to simulation units
    max_steps=1000,
)
```

That's it! The rest is handled automatically.

## Architecture

### Action Space

**Discrete Actions** (4 options):
- `0`: NOOP - Do nothing, let simulation run
- `1`: COUNT_BACTERIA - Count bacterial population (has cooldown)
- `2`: SEQUENCING - Perform genome sequencing (expensive, long cooldown)
- `3`: DOSE - Apply antibiotic doses (uses continuous vector)

**Continuous Actions** (when DOSE is selected):
- Dose vector in [0,1]^K where K = number of antibiotic types
- Example: `[0.1, 0.3, 0.5]` for 3 antibiotics
- Always produced by policy but only used when discrete action is DOSE

### Model Architecture

```
Input: obs_t [B, obs_dim]
       h_{t-1} [layers, B, hidden_dim]
       
↓ GRU Core
h_t [layers, B, hidden_dim]

↓ Branches into 3 heads:

1. Discrete Head: h_t → logits [B, 4]
2. Continuous Head: h_t → (μ, σ) [B, K] (Gaussian with tanh squashing)
3. Value Head: h_t → V(s_t) [B, 1]
```

### Training Process

1. **Rollout**: Collect `rollout_steps` of experience
2. **GAE**: Compute advantages with λ-returns
3. **Truncated BPTT**: Split into sequences of length `seq_len`
4. **PPO Update**: For each sequence:
   - Recompute action log-probs with current policy
   - Compute joint ratio: `ratio_disc * ratio_cont^(is_dose)`
   - Apply PPO clipped objective
   - Update value function with MSE
   - Add entropy bonus
5. **Repeat**: Multiple epochs over same data

## Module Structure

```
rl/
├── __init__.py          # Package exports
├── config.py            # PPOConfig dataclass and seed setting
├── env_wrapper.py       # Thin Mesa wrapper with Gym-like API
├── models.py            # RecurrentActorCritic with GRU
├── buffer.py            # RolloutBuffer for trajectory storage
├── ppo.py               # PPOTrainer with truncated BPTT
├── train.py             # Training entrypoint and CLI
└── utils.py             # GAE, normalization, grad clipping
```

## Configuration

Key hyperparameters in `PPOConfig`:

```python
@dataclass
class PPOConfig:
    obs_dim: int              # Observation dimension (auto-inferred)
    n_discrete: int = 4       # Number of discrete actions
    k_doses: int = 3          # Number of antibiotic types
    
    hidden_dim: int = 256     # GRU hidden size
    rnn_layers: int = 1       # Number of GRU layers
    
    gamma: float = 0.99       # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_eps: float = 0.2     # PPO clipping epsilon
    
    seq_len: int = 64         # Truncated BPTT length
    rollout_steps: int = 2048 # Steps before update
    epochs: int = 4           # Epochs per update
    lr: float = 3e-4          # Learning rate
```

## Command-Line Arguments

```bash
python -m rl.train [OPTIONS]

Environment:
  --k-doses INT          Number of antibiotic types (default: 3)
  --mock                 Use mock environment for testing

Training:
  --total-updates INT    Total PPO updates (default: 100)
  --steps-per-rollout    Steps per rollout (default: 2048)
  --seq-len INT          Truncated BPTT length (default: 64)
  --epochs INT           PPO epochs per update (default: 4)

Model:
  --hidden-dim INT       Hidden dimension (default: 256)
  --rnn-layers INT       Number of RNN layers (default: 1)

Optimization:
  --lr FLOAT             Learning rate (default: 3e-4)
  --gamma FLOAT          Discount factor (default: 0.99)
  --gae-lambda FLOAT     GAE lambda (default: 0.95)
  --clip-eps FLOAT       PPO clip epsilon (default: 0.2)

System:
  --device {cpu,cuda}    Device (default: cpu)
  --seed INT             Random seed (default: 42)
  --save-dir PATH        Save directory (default: ./checkpoints)
```

## Output

Training produces:

- **Checkpoints**: `checkpoints/checkpoint_{N}.pt`
  - Model state dict
  - Optimizer state
  - Config
- **Training Log**: `checkpoints/training_log.json`
  - Episode rewards
  - Loss curves
  - Entropy, clip fraction, etc.
- **Config**: `checkpoints/config.json`

## Testing

Run unit-like smoke test:

```bash
# Quick 1-update test with mock environment
python -m rl.train --mock --total-updates 1 --steps-per-rollout 128

# Should complete without errors and save checkpoint
```

## Design Principles

1. **No Breaking Changes**: Wrapper isolates RL from simulation
2. **Clean Boundaries**: Clear separation between RL and Mesa code
3. **Type Safety**: Full type hints throughout
4. **Deterministic**: Seeding for reproducibility
5. **Sequential**: Preserves temporal order for recurrent learning
6. **Tested**: Mock environment for smoke testing

## Performance Tips

- Use `--device cuda` for GPU acceleration
- Increase `--steps-per-rollout` for better sample efficiency
- Tune `--seq-len` based on episode length (typical: 32-128)
- Monitor clip fraction (should be ~0.1-0.3)
- If training is unstable, reduce `--lr` or increase `--seq-len`

## Dependencies

- **Required**: `torch`, `numpy`
- **Optional**: `tqdm` (progress bars)
- **Simulation**: `mesa`, `scipy`, `matplotlib` (only for full simulation, not for RL code)

## Citation

If you use this implementation, please cite:

```bibtex
@software{recurrent_ppo_bacteria,
  title = {Recurrent PPO for Bacteria Antibiotic Learning},
  year = {2025},
  url = {https://github.com/bacobax/antibiotic_learning}
}
```

## License

Same as parent project.
