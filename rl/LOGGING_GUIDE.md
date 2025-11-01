# Training Logging Guide

## Overview

The improved logging system provides **meaningful insights into your training progress** while removing noisy console output. Logs are saved to two places:

1. **`training.log`** - Detailed file logs with DEBUG level messages
2. **`training_log.json`** - Machine-readable metrics JSON for analysis

## Console Output (What You See)

The console shows **only important information** (INFO level and above):

```
[INFO] PPO Training Started
[INFO] Using REAL Mesa environment
[INFO] Observation dimension: 7
[INFO] Starting PPO training for 100 updates
[INFO] Config: obs_dim=7, hidden_dim=256, rnn_layers=1, k_doses=3
[INFO] Hyperparams: lr=0.0003, gamma=0.99, gae_lambda=0.95
[INFO] Rollout steps per update: 2048, PPO epochs: 4
[INFO] UPDATE    0/100 | Reward:    50.34 (±12.45) | Episodes:   5 | Actor Loss: 0.5234 | Critic Loss: 0.3821
[INFO] UPDATE   10/100 | Reward:    65.12 (±18.92) | Episodes:   6 | Actor Loss: 0.3421 | Critic Loss: 0.2156
[INFO] UPDATE   20/100 | Reward:    75.89 (±22.10) | Episodes:   7 | Actor Loss: 0.2890 | Critic Loss: 0.1654
...
[INFO] ======================================================================
[INFO] TRAINING SUMMARY
[INFO] ======================================================================
[INFO] Total time: 45.2 minutes (0.75 hours)
[INFO] Best reward achieved: 125.45
[INFO] Final reward: 120.34
[INFO] Final actor loss: 0.0821
[INFO] Reward improvement (first 10 vs last 10 updates): +45.32
[INFO] ======================================================================
```

## Key Metrics to Monitor

### Per-Update Logs (every 10 updates)

```
UPDATE 20/100 | Reward: 75.89 (±22.10) | Episodes: 7 | Actor Loss: 0.3421 | Critic Loss: 0.1654
```

**What each metric means:**

| Metric | What It Shows | Good Range |
|--------|---------------|-----------|
| **Reward** | Average episode reward | Increasing over time |
| **±** (std) | Variability in episode rewards | Lower is more consistent |
| **Episodes** | How many episodes completed | Higher = more training data |
| **Actor Loss** | Policy gradient loss | Should decrease over time |
| **Critic Loss** | Value function loss | Should decrease over time |

### Detailed Debug Logs (in `training.log`)

```
Entropy: 0.4321 | Clip Frac: 0.125 | Grad Norm: 0.8234 | Value Mean: 0.4532
```

| Metric | What It Shows | Interpretation |
|--------|---------------|-----------------|
| **Entropy** | Policy exploration level | Too low (<0.1): exploitation only; Too high (>1.0): random |
| **Clip Frac** | PPO clipping ratio | >0.5: consider reducing LR; <0.1: increase clip epsilon |
| **Grad Norm** | Gradient magnitude | NaN/Inf: numerical instability |
| **Value Mean** | Average value estimates | Should be ~0 after normalization |

## Training Health Checks

### ✅ Signs of Good Training

1. **Reward increasing** over 10-update intervals
2. **Actor/Critic losses decreasing**
3. **Clip fraction** between 0.05 and 0.5
4. **Entropy** slowly decreasing (exploration → exploitation)
5. **Gradient norm** stable (0.1 - 2.0 range)

### ⚠️ Warning Signs

| Warning | Cause | Solution |
|---------|-------|----------|
| `High clipping fraction at update X: 0.85` | Learning rate too high | Reduce `--lr` (try 1e-4) |
| `NaN detected in actor loss` | Numerical instability | Reduce `--lr` or check environment |
| `No episodes completed at update X` | Steps per rollout too small | Increase `--steps-per-rollout` |
| Reward plateauing | Insufficient exploration | Increase entropy coefficient |
| Reward oscillating wildly | High variance | Increase GAE lambda `--gae-lambda` |

## File Outputs

### `training.log`

Comprehensive log file with all messages (DEBUG and above):

```
2025-11-01 10:30:45 [INFO] PPO Training Started
2025-11-01 10:30:45 [DEBUG] Random seed set to: 42
2025-11-01 10:30:46 [DEBUG] Successfully loaded BacteriaModel
2025-11-01 10:30:47 [INFO] Observation dimension: 7
2025-11-01 10:30:48 [INFO] Starting PPO training for 100 updates
2025-11-01 10:30:48 [DEBUG] Using device: cpu
2025-11-01 10:30:50 [INFO] UPDATE    0/100 | Reward:    50.34 (±12.45) ...
2025-11-01 10:30:51 [DEBUG]   Entropy: 0.4321 | Clip Frac: 0.125 ...
2025-11-01 10:30:51 [DEBUG]   ETA: 45.2 min (2.23 updates/sec)
```

### `training_log.json`

Machine-readable metrics for post-analysis:

```json
[
  {
    "update": 0,
    "mean_episode_reward": 50.34,
    "std_episode_reward": 12.45,
    "max_episode_reward": 78.20,
    "min_episode_reward": 22.15,
    "mean_episode_length": 125.4,
    "num_episodes": 5,
    "loss_total": 0.9055,
    "loss_actor": 0.5234,
    "loss_critic": 0.3821,
    "entropy": 0.4321,
    "clip_fraction": 0.1250,
    "grad_norm": 0.8234,
    "value_mean": 0.4532,
    "advantage_mean": 0.0234
  },
  ...
]
```

## Post-Training Analysis

### Reading the JSON Log

```python
import json
import numpy as np

with open("checkpoints/training_log.json") as f:
    logs = json.load(f)

# Plot reward over time
import matplotlib.pyplot as plt

rewards = [log["mean_episode_reward"] for log in logs]
updates = [log["update"] for log in logs]

plt.plot(updates, rewards)
plt.xlabel("Update")
plt.ylabel("Mean Episode Reward")
plt.title("Training Progress")
plt.show()

# Compute statistics
print(f"Best reward: {max(rewards):.2f}")
print(f"Final reward: {rewards[-1]:.2f}")
print(f"Improvement: {rewards[-1] - rewards[0]:.2f}")
```

## Usage Examples

### Monitor training in real-time

```bash
# Terminal 1: Run training
python -m rl.train --total-updates 100 --device cuda

# Terminal 2: Monitor log file updates
tail -f checkpoints/training.log
```

### Check for training errors

```bash
# Look for warnings and errors
grep -E "WARNING|ERROR" checkpoints/training.log

# Look for NaN issues
grep -E "NaN|Inf" checkpoints/training.log
```

### Analyze training progress

```bash
# See start vs end config
head -20 checkpoints/training.log
tail -30 checkpoints/training.log
```

## Logging Levels

- **DEBUG**: Detailed diagnostic info (gradient norms, entropy, hidden states)
- **INFO**: Significant events (update progress, summaries, configs)
- **WARNING**: Potential issues (high clip fraction, no episodes)
- **ERROR**: Critical problems (NaN losses, import failures)

Only **INFO and above** appear in console to keep it clean. Check `training.log` for DEBUG details.

## Disabling Logs (if needed)

To see only errors in console:

```python
# In train.py, modify setup_logging()
ch.setLevel(logging.ERROR)  # Instead of logging.INFO
```

Or redirect console output:

```bash
python -m rl.train 2>/dev/null  # Hide all console output
tail -f checkpoints/training.log  # Watch file logs instead
```
