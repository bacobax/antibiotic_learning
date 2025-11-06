# Next-Population Prediction Head Implementation

## Summary

Implemented a neural network prediction head that learns to predict next-step population during COUNT actions, with masked MSE loss and COUNT-only reward feedback.

## Changes Made

### 1. MODEL (src/rl/models.py)

**Added Prediction Head:**
- New linear layer: `self.pred_head = nn.Linear(hidden_dim, 1)`
- Activation: `F.softplus()` for non-negative predictions
- Output: `pred_next_pop` [B, 1] - normalized population prediction

**Modified Methods:**
- `forward_step()`: Returns additional `pred_next_pop` tensor
- `act()`: Returns `pred_next_pop` in output dictionary
- `evaluate_actions()`: Computes `pred_next_pop` for all sequence steps

**No changes to:**
- Policy head (discrete actions)
- Value head
- Continuous action head (doses)

---

### 2. ENVIRONMENT (src/rl/env_wrapper.py)

**Added Supervision Signals:**
- `population_before_step`: Captured before `model.step()`
- `population_after_step`: Captured after `model.step()`
- `population_next_norm`: True next population / population_norm
- `count_was_performed`: Boolean flag (True if action == COUNT)

**Modified info dict:**
```python
info["population_next_norm"] = population_next_norm
info["count_was_performed"] = count_was_performed
```

**Critical Rule:** These values are ONLY used for:
1. Masked prediction loss computation
2. COUNT-only reward calculation
3. NOT exposed to agent observation

---

### 3. BUFFER (src/rl/buffer.py)

**Added Storage:**
- `pred_next_pop`: [T, B] - Predicted next population
- `population_next_norm`: [T, B] - True next population (supervision)
- `count_mask`: [T, B] - Binary mask (1.0 on COUNT, 0.0 otherwise)

**Modified Methods:**
- `add()`: Now accepts 3 additional tensors
- `stacked()`: Returns 3 additional tensors
- `clear()`: Clears 3 additional lists

---

### 4. AGENT (src/rl/agent.py)

**Modified `select_action()`:**
- Returns 7 values instead of 6
- Added: `pred_next_pop` tensor

---

### 5. PPO TRAINER (src/rl/ppo.py)

**Added Masked Prediction Loss:**
```python
pred_error = (new_pred_next_pop - population_next_norm_chunk) ** 2
pred_loss = (pred_error * count_mask_chunk).mean()
```

**Loss Composition:**
```python
total_loss = (
    actor_loss 
    + vf_coef * critic_loss
    + pred_loss              # NEW
    - ent_coef * entropy
)
```

**Key Property:** Loss is ZERO when `count_mask == 0.0`

**Added Metrics:**
- `loss_pred`: Masked prediction loss value

---

### 6. TRAINING LOOP (src/rl/training_utils.py)

**Added Prediction Tracking:**
- `episode_pred_error`: Per-episode prediction error accumulator
- `episode_pred_reward`: Per-episode prediction reward accumulator

**Modified Rollout:**
```python
# Extract supervision from info
population_next_norm = info.get('population_next_norm', 0.0)
count_was_performed = info.get('count_was_performed', False)

# COUNT-only reward
if count_was_performed:
    pred_error = abs(pred_next_pop_value - population_next_norm)
    pred_reward = -pred_error
    current_pred_error += pred_error
    current_pred_reward += pred_reward
else:
    pred_reward = 0.0
```

**Buffer Storage:**
```python
buffer.add(
    ...
    pred_next_pop=pred_next_pop.cpu(),
    population_next_norm=torch.tensor([population_next_norm], dtype=torch.float32),
    count_mask=torch.tensor([count_mask_value], dtype=torch.float32),
)
```

**Added Metrics:**
- `prediction/error`: Mean prediction error per episode
- `prediction/reward`: Mean prediction reward per episode

---

### 7. VISUALIZATION (src/train_with_visualization.py)

**Added Tracking:**
- Same prediction metrics as training_utils.py
- Integrated into control panel data structures
- Added to rollout_reward_components

**Consistent Logging:**
- `prediction/error`
- `prediction/reward`

---

## Reward Structure

### COUNT Action:
```python
if action == COUNT:
    pred_error = |pred_next_pop - true_next_pop|
    pred_reward = -pred_error
    total_reward += pred_reward
```

### All Other Actions:
```python
pred_reward = 0.0
```

**Critical:** Prediction reward ONLY applied when COUNT happens.

---

## Loss Computation

### Masked MSE Loss:
```python
# Compute squared error
pred_error = (predicted - target) ** 2  # [T, B]

# Apply mask (1.0 on COUNT, 0.0 otherwise)
masked_loss = (pred_error * count_mask).mean()
```

**Result:**
- COUNT steps: Loss = MSE(pred, target)
- Other steps: Loss = 0.0

---

## TensorBoard Metrics

All training systems log identical metric names:

### Prediction Metrics:
- `prediction/error` - Mean absolute prediction error per episode
- `prediction/reward` - Mean prediction reward per episode

### Loss Metrics:
- `loss_pred` - Masked prediction loss
- `loss_actor` - Policy loss
- `loss_critic` - Value loss
- `loss_total` - Combined loss

---

## Implementation Properties

✅ **Minimal Diff:** Only modified necessary files/functions
✅ **No Policy Changes:** Discrete/continuous action heads unchanged
✅ **No Value Changes:** Value head unchanged
✅ **No Observation Changes:** Agent doesn't see `population_next_norm`
✅ **COUNT-Only Reward:** Prediction reward ONLY on COUNT actions
✅ **Masked Loss:** Loss computed ONLY on COUNT steps
✅ **Consistent Logging:** Identical metric names across all systems

---

## Files Modified

1. `src/rl/models.py` - Added prediction head
2. `src/rl/env_wrapper.py` - Added supervision signals
3. `src/rl/agent.py` - Return prediction from select_action
4. `src/rl/buffer.py` - Store prediction data
5. `src/rl/ppo.py` - Masked prediction loss
6. `src/rl/training_utils.py` - Prediction tracking and logging
7. `src/train_with_visualization.py` - Prediction tracking and logging

---

## Verification

All files compile successfully:
```bash
python -m py_compile src/rl/models.py \
  src/rl/env_wrapper.py \
  src/rl/agent.py \
  src/rl/buffer.py \
  src/rl/ppo.py \
  src/rl/training_utils.py \
  src/train_with_visualization.py
```

No syntax errors detected.
