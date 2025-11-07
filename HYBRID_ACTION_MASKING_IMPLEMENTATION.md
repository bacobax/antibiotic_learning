# Hybrid Action Masking Implementation (Option C)

## Overview
Successfully implemented **continuous-dependent dose masking** for the PPO agent with hybrid action space. This is a principled approach where the agent **cannot select invalid actions** because they receive probability 0 via masked logits.

## Key Changes

### 1. Environment Side (`env_wrapper.py`)

#### Added: `get_action_mask(a_cont)`
- Computes action mask based on budget and **specific continuous dose**
- Returns binary mask `[NOOP, COUNT, SEQ, DOSE]` where:
  - `NOOP` is always valid (1.0)
  - `COUNT` is valid if `budget >= count_cost`
  - `SEQUENCING` is valid if `budget >= sequencing_cost`
  - `DOSE` is valid if `budget >= dose_cost + sum(scaled_doses) * dose_cost_per_unit`
- **Key feature**: DOSE validity depends on the exact continuous action sampled

#### Removed:
- `_check_action_affordability()` method - no longer needed
- All logic for converting unaffordable actions to NOOP
- `unaffordable_action_penalty` from reward computation
- Budget checking in `_execute_action()` (actions are guaranteed affordable)

#### Modified: `step()`
- Removed affordability pre-check
- Agent cannot select invalid actions due to masking
- Simplified flow: action → execute → reward

### 2. Policy Side (`models.py`)

#### Modified: `act()`
**NEW FLOW (Option C):**
1. Sample continuous action `a_cont` FIRST (always sampled)
2. Apply action mask (if provided): `masked_logits = logits + log(mask)`
3. Sample discrete action from **masked distribution**
4. Return both action and mask

**Key parameters:**
- Added `action_mask` parameter (shape `[B, n_discrete]`)
- Returns `action_mask` in output dict

#### Modified: `evaluate_actions()`
- Added `action_masks` parameter for sequence evaluation
- Applies same masking during training: `masked_logits = logits + log(masks)`
- Ensures consistency between action selection and policy evaluation

### 3. Agent Side (`agent.py`)

#### Modified: `__init__()`
- Added `env` parameter (needed for calling `env.get_action_mask()`)

#### Modified: `select_action()`
**NEW FLOW:**
1. Forward pass to get continuous action
2. **Call `env.get_action_mask(a_cont_np)` to compute mask**
3. Call `model.act()` with the mask
4. Return action + mask (8 values total)

#### Modified: `load_agent_from_checkpoint()`
- Added optional `env` parameter
- Allows loading without env for inference scenarios

### 4. Buffer Side (`buffer.py`)

#### Added field: `action_mask`
- Stores action masks `[T, B, n_discrete]`
- Required for consistent policy evaluation during training

#### Modified methods:
- `add()`: accepts `action_mask` parameter
- `stacked()`: returns `action_mask` in dict
- `clear()`: clears `action_mask` list

### 5. PPO Trainer (`ppo.py`)

#### Modified: `update()`
- Loads `action_masks` from data dict
- Passes `action_masks_chunk` to `model.evaluate_actions()`
- Ensures masked logits during policy evaluation

### 6. Training Utils (`training_utils.py`)

#### Modified: `rollout()`
- Unpacks 8 values from `agent.select_action()` (added `action_mask`)
- Stores `action_mask` in buffer via `buffer.add()`

#### Modified: `_initialize_agent()`
- Accepts `env` parameter
- Passes `env` to `RLAgent()` constructor

#### Modified: `train()`
- Passes `env` to `_initialize_agent()`

## Removed Systems

### ❌ Deleted:
1. **Affordability checking** - replaced by masking
2. **Action conversion to NOOP** - invalid actions have prob 0
3. **Unaffordable action penalties** - no longer needed
4. **"Not-performed" action bits** - masking is cleaner

### ✅ Kept:
1. All reward components (population, genome, cost, etc.)
2. Population prediction head
3. Observation structure
4. Dosing dynamics
5. Budget tracking and penalties

## How It Works

### Action Selection Flow
```python
# 1. Agent samples continuous action
a_cont_raw = policy.sample_continuous()  # from Gaussian
a_cont = clip(tanh(a_cont_raw), 0, 1)    # squash to [0,1]

# 2. Environment computes mask using THIS specific dose
scaled = scale_dose(a_cont)
dose_cost = fixed_cost + sum(scaled) * variable_cost
mask[DOSE] = 1.0 if budget >= dose_cost else 0.0
mask[COUNT] = 1.0 if budget >= count_cost else 0.0
mask[SEQUENCING] = 1.0 if budget >= seq_cost else 0.0
mask[NOOP] = 1.0  # always valid

# 3. Policy applies mask to logits
masked_logits = logits + log(mask)
# Invalid actions get -inf logit → probability 0

# 4. Sample discrete action from masked distribution
a_discrete = Categorical(masked_logits).sample()
# Agent CANNOT select invalid actions!
```

### Training Flow
```python
# During PPO update:
# 1. Re-evaluate actions with SAME masks from rollout
eval_dict = model.evaluate_actions(
    obs_seq, h_init, a_disc, a_cont, 
    action_masks=stored_masks  # from buffer
)

# 2. Masked logits ensure consistency
masked_logits = logits + log(stored_masks)
dist = Categorical(masked_logits)
new_logp = dist.log_prob(old_actions)

# 3. Invalid actions still have prob 0
# No need for penalties or corrections
```

## Benefits

### 1. **Principled Approach**
- Invalid actions are **impossible to select** (prob = 0)
- No need for penalties or corrections
- Agent learns only from valid action space

### 2. **No Negative Rewards**
- Removed all "punishment for invalid actions"
- Cleaner reward signal
- Easier to debug

### 3. **Budget-Aware Dosing**
- DOSE validity depends on **actual dose amount**
- Agent learns to modulate dose to stay affordable
- More realistic economic behavior

### 4. **Mathematically Sound**
- Log-probability masking is standard practice
- Preserves probability distribution properties
- Compatible with PPO's trust region

### 5. **Clean Separation**
- **Environment**: defines validity (mask)
- **Policy**: respects validity (masked logits)
- **Training**: maintains consistency (stored masks)

## Validation

### What to Check:
1. **No unaffordable actions**: Monitor that executed actions never exceed budget
2. **Masking statistics**: Log `sum(mask)` to see how many actions are valid
3. **DOSE selection**: Check if agent learns to modulate doses when broke
4. **Training stability**: Ensure loss curves are smooth (no spikes from invalid actions)

### Expected Behavior:
- Agent should learn to COUNT more when budget is low (cheap action)
- Agent should learn to reduce dose amounts when budget is tight
- No episodes should end due to "attempted unaffordable action"
- Policy should become more budget-aware over time

## Migration Notes

### For Existing Checkpoints:
- Old checkpoints will load fine (backward compatible)
- `env` parameter in `load_agent_from_checkpoint()` is optional
- Set `env=None` for inference without masking (falls back to full action space)

### For New Training:
- **Must** pass `env` to `RLAgent()` constructor
- Action masking is automatic during `select_action()`
- Buffer automatically stores masks for training

### For Visualization/Testing:
- Can load agent without env: `agent = RLAgent.load_agent_from_checkpoint(path, env=None)`
- Then set env later: `agent.env = env`
- Or pass env during load: `agent = RLAgent.load_agent_from_checkpoint(path, env=env)`

## Implementation Complete ✅

All components have been updated:
- ✅ Environment masking
- ✅ Policy masking
- ✅ Agent integration
- ✅ Buffer storage
- ✅ PPO training
- ✅ Rollout collection
- ✅ Old system removal
- ✅ Backward compatibility

The system is ready for training with hybrid action masking (Option C).
