## Table of Contents
1. Overview
2. Partial Observability
3. Action Space
4. Reward System
5. Training Dynamics

---

## Overview

This RL agent learns to manage a bacterial population by controlling antibiotic dosing, monitoring bacterial counts, and performing genomic sequencing. The environment simulates realistic delays, costs, and partial observability to model real-world antibiotic stewardship challenges.

**Key Design Principle**: The agent only "knows" what it explicitly measures, creating a partially observable Markov decision process (POMDP).

---

## Partial Observability

### Core Concept
The agent **cannot see the true state** of the bacteria simulation. Instead, it maintains cached observations that become stale over time. This mirrors real-world scenarios where:
- Lab results take time to process
- Measurements are expensive
- Acting without information is risky

### Observation Vector Structure

The agent receives a **19-dimensional observation vector** (for K=3 antibiotic types):

```python
obs = [
    budget_norm,                        # [0] Normalized remaining budget (0-1)
    target_norm,                        # [1] Normalized target population (static reference)
    last_count_norm,                    # [2] Last population count (-1 if never measured)
    
    # Genome averages (3 bacteria types × 4 traits = 12 values)
    avg_genome[0][0],                   # [3] Type 0: enzyme
    avg_genome[0][1],                   # [4] Type 0: efflux
    avg_genome[0][2],                   # [5] Type 0: repair
    avg_genome[0][3],                   # [6] Type 0: membrane
    avg_genome[1][0],                   # [7] Type 1: enzyme
    avg_genome[1][1],                   # [8] Type 1: efflux
    avg_genome[1][2],                   # [9] Type 1: repair
    avg_genome[1][3],                   # [10] Type 1: membrane
    avg_genome[2][0],                   # [11] Type 2: enzyme
    avg_genome[2][1],                   # [12] Type 2: efflux
    avg_genome[2][2],                   # [13] Type 2: repair
    avg_genome[2][3],                   # [14] Type 2: membrane
    
    # Type proportions (currently not used)
    prop[0],                            # [15] Proportion of antibiotic 0
    prop[1],                            # [16] Proportion of antibiotic 1
    prop[2],                            # [17] Proportion of antibiotic 2
    
    time_since_last_measure_norm,       # [18] Time since last COUNT/SEQ (0-1, capped at 100 steps)
    seq_pending_flag,                   # [19] 1.0 if sequencing in progress, 0.0 otherwise
    seq_eta_norm,                       # [20] Steps until sequencing result (0-1)
]
```

### Observation Gating Rules

| Observation Component | Availability | Default Value | Update Trigger |
|----------------------|--------------|---------------|----------------|
| `budget_norm` | Always available | Current budget / `budget_norm` | Every step |
| `target_norm` | Always available | Static (`target_population` / `population_norm`) | Never changes |
| `last_count_norm` | Only after COUNT | `-1.0` (sentinel) | COUNT action completes |
| `avg_genome` | Only after SEQUENCING | All zeros | SEQUENCING action completes (after delay) |
| `proportions` | Only after SEQUENCING | All zeros | SEQUENCING action completes (after delay) |
| `time_since_last_measure_norm` | Always available | `1.0` (maximum staleness) | Updated every step |
| `seq_pending_flag` | Always available | `0.0` or `1.0` | SEQUENCING action starts/completes |
| `seq_eta_norm` | Always available | Steps remaining / duration | Counts down during sequencing |

### Key Staleness Mechanics

1. **Population Count Staleness**
   - COUNT action provides instant population measurement
   - Value becomes stale immediately (environment evolves each step)
   - Agent must infer current state from stale measurements
   - Tracked via `ts_last_count` (timestep of last count)

2. **Genome Data Staleness**
   - SEQUENCING action has configurable delay (default: 5 steps)
   - Genome averages from `sequencing_duration` steps ago
   - By the time results arrive, bacteria may have evolved
   - Tracked via `ts_last_seq` (timestep when sequencing completed)

3. **Age-Based Reward Normalization**
   - Older measurements receive diminished reward contributions
   - Three normalization types: `linear`, `log`, `sqrt`
   - Example (sqrt): `normalized_reward = reward / (1 + sqrt(age))`
   - Applied to both population and genome-based rewards

### Information Asymmetry

The agent's challenge is deciding **when to pay for information** vs **when to act blind**:

```python
# Example decision tree (implicit in learned policy):
if last_count_obs is None:
    # Never measured - high uncertainty
    # Risky to dose without knowing population
    return ACTION_COUNT_BACTERIA

if (t - ts_last_count) > informed_dosing_window:
    # Count is too stale
    # Should we re-count or dose anyway?
    if budget < sequencing_cost:
        # Can't afford sequencing, dose blind?
        return ACTION_DOSE  # Penalty likely
    else:
        return ACTION_COUNT_BACTERIA

if last_seq_obs is None or (t - ts_last_seq) > informed_sequencing_window:
    # No genome data or too stale
    # Dosing without sequencing = blind to resistance
    return ACTION_SEQUENCING  # Invest in information

# Both count and sequencing fresh → informed dosing possible
return ACTION_DOSE  # Bonus reward likely
```

---

## Action Space

The action space is **hybrid discrete-continuous**:

### Discrete Action Selection (4 options)

| Action ID | Name | Description | Duration | Cost |
|-----------|------|-------------|----------|------|
| 0 | `ACTION_NOOP` | Do nothing, observe | Instant | Free |
| 1 | `ACTION_COUNT_BACTERIA` | Measure population | Instant | `count_cost` (default: 0.0) |
| 2 | `ACTION_SEQUENCING` | Genome sequencing | `sequencing_duration` steps | `sequencing_cost` (default: 1.0) |
| 3 | `ACTION_DOSE` | Apply antibiotics | Instant | `dose_cost` + per-unit costs |

### Continuous Action Parameters

Only used when `ACTION_DOSE` is selected:
```python
a_cont = [dose_0, dose_1, dose_2]  # Shape: (k_doses,)
# Each value in [0, 1], scaled by environment
# Total cost = dose_cost + sum(scaled_doses) * dose_cost_per_unit
```

### Action Execution Logic

#### 1. NOOP (Action 0)
```python
# Minimal reward shaping based on population distance from target
if |last_count_obs - target_population| <= noop_band:
    reward = +noop_reward_magnitude  # Small bonus for staying in safe zone
elif last_count_obs < target_population:
    reward = +0.5 * noop_reward_magnitude  # Slight bonus (undershooting target)
else:
    reward = -0.5 * noop_reward_magnitude  # Slight penalty (overshooting target)
```

**Use case**: Wait for sequencing results, conserve budget, or observe natural population dynamics.

#### 2. COUNT_BACTERIA (Action 1)
```python
# Instant measurement of population
cost = count_cost
budget -= cost
last_count_obs = true_population  # Cache result
ts_last_count = current_timestep

# Immediate rewards:
# 1. Regular monitoring bonus (encourage periodic counting)
if ts_last_count is not None:
    interval = current_timestep - ts_last_count
    if regular_count_min_interval <= interval <= regular_count_interval * 1.5:
        reward += regular_count_reward  # Reward for counting at good intervals

# 2. Count population reward (feedback on distance from target)
distance = |true_population - target_population|
normalized_distance = distance / population_norm
reward -= count_population_reward * normalized_distance  # Penalty for being far from target
```

**Use case**: Get fresh population data before dosing, track treatment efficacy, avoid blind dosing.

#### 3. SEQUENCING (Action 2)
```python
# Start sequencing pipeline
if not seq_pending:
    cost = sequencing_cost
    budget -= cost
    seq_pending = True
    seq_eta = sequencing_duration  # Countdown starts
    reward = 0.0  # No immediate reward, only cost

# Each step: seq_eta -= 1
# When seq_eta reaches 0:
last_seq_obs = {
    "avg_genome": avg_genome_matrix,  # [K, M] average traits
    "proportions": proportions,        # [K] type distribution
}
ts_last_seq = current_timestep
seq_pending = False
```

**Use case**: Learn bacterial resistance profiles to enable informed dosing, avoid ineffective antibiotics.

#### 4. DOSE (Action 3)
```python
# Apply antibiotics to environment
scaled_doses = scale_dose(np.clip(a_cont, 0, 1))  # [K] doses
variable_cost = sum(scaled_doses) * dose_cost_per_unit
total_cost = dose_cost + variable_cost
budget -= total_cost

# Apply to simulation
for i, antibiotic_name in enumerate(antibiotic_fields):
    model.apply_antibiotic(antibiotic_name, scaled_doses[i])

# Immediate rewards:
# 1. Cost penalty
reward = -total_cost * w_cost

# 2. Informed dosing bonus/penalty (see Reward System section)
if has_recent_count and has_recent_sequencing and population >= target:
    reward += informed_dosing_reward + informed_dosing_above_target_reward
elif not has_recent_count or not has_recent_sequencing:
    reward -= blind_dosing_penalty
if population < target:
    reward -= dosing_low_population_penalty  # BIG penalty

# 3. Safe behavior bonus (for NOT dosing when pop < target)
# Applied to other actions, not DOSE
```

**Use case**: Reduce bacterial population when it exceeds target, especially with fresh count+sequencing data.

### Affordability Gating

Before executing any action (except NOOP), the environment checks budget:

```python
def _check_action_affordability(a_discrete, a_cont):
    if a_discrete == ACTION_NOOP:
        return ACTION_NOOP, 0.0
    
    if a_discrete == ACTION_COUNT_BACTERIA:
        if budget < count_cost:
            return ACTION_NOOP, 0.0  # Silently convert to NOOP
        return ACTION_COUNT_BACTERIA, count_cost
    
    if a_discrete == ACTION_SEQUENCING:
        if budget < sequencing_cost:
            return ACTION_NOOP, 0.0  # Silently convert to NOOP
        return ACTION_SEQUENCING, sequencing_cost
    
    if a_discrete == ACTION_DOSE:
        total_cost = dose_cost + sum(scaled_doses) * dose_cost_per_unit
        if budget < total_cost:
            return ACTION_NOOP, 0.0  # Silently convert to NOOP
        return ACTION_DOSE, total_cost
```

**Effect**: Agent cannot "cheat" by acting without budget. Unaffordable actions become NOOP, with optional penalty (`unaffordable_action_penalty`).

---

## Reward System

The total reward per step is the sum of **12 independent components**:

```python
total_reward = (
    immediate_reward +
    maintenance_reward +
    budget_penalty +
    unaffordable_action_penalty +
    delayed_reward +
    survival_bonus +
    budget_conservation +
    regular_count_bonus +
    safe_behavior_bonus +
    informed_dosing_bonus +
    count_population_reward +
    critical_inaction_penalty
)
```

### 1. Immediate Reward
**When**: Every action execution  
**Purpose**: Instant feedback for action costs and basic shaping  

**Components**:
- **Cost penalty**: `-action_cost * w_cost` for all paid actions
- **NOOP shaping**: Small bonus/penalty for staying near target with NOOP
- **Regular count bonus**: Rewards counting at regular intervals (see component #8)
- **Informed dosing bonus**: Rewards/penalizes dosing decisions (see component #9)
- **Count population reward**: Immediate feedback on distance from target after COUNT (see component #10)

**Implementation** (from `env_wrapper.py:_execute_action`):
```python
def _execute_action(self, a_discrete, a_cont, action_cost):
    if a_discrete == ACTION_NOOP:
        # NOOP shaping: reward staying near target
        diff = self.last_count_obs - self.target_population
        if abs(diff) <= self.noop_band:
            return self.noop_mag  # Small bonus
        elif diff < 0:
            return self.noop_mag * 0.5  # Slight bonus (below target)
        else:
            return -self.noop_mag * 0.5  # Slight penalty (above target)
    
    if a_discrete == ACTION_COUNT_BACTERIA:
        self.budget -= action_cost
        # Regular monitoring bonus computed here
        # Count population reward computed here
        return -action_cost + regular_monitoring_bonus + count_pop_reward
    
    if a_discrete == ACTION_SEQUENCING:
        self.budget -= action_cost
        return 0.0  # No immediate reward, result delayed
    
    if a_discrete == ACTION_DOSE:
        self.budget -= action_cost
        # Informed dosing bonus/penalty computed here
        return -action_cost * self.w_cost + dosing_bonus
```

### 2. Maintenance Reward
**When**: Every step  
**Purpose**: Continuous pressure to keep population near target  
**Module**: `PopulationMaintenanceReward`

**Formula**:
```python
asymmetric_penalty = -(
    asymmetry_factor * max(0, pop - target) +  # Overshooting worse
    0.3 * max(0, target - pop)                 # Undershooting less bad
) / population_norm * weight
```

**Default Config**:
- `target_population`: 500
- `asymmetry_factor`: 3.0 (overshooting 3× worse than undershooting)
- `weight`: 0.01 (configured via `w_population_maintenance`)

**Example**:
```python
# Population = 1500 (3x target)
above = 1500 - 500 = 1000
penalty = -(3.0 * 1000 + 0) / 1000 * 0.01 = -0.03

# Population = 250 (0.5x target)
below = 500 - 250 = 250
penalty = -(0 + 0.3 * 250) / 1000 * 0.01 = -0.00075
```

**Effect**: Agent feels constant "pain" when population drifts from target, encouraging dosing when high and caution when low.

### 3. Budget Penalty
**When**: Budget reaches 0  
**Purpose**: Prevent resource exhaustion  

**Formula**:
```python
if budget <= 0.0:
    penalty = -budget_penalty  # Large negative reward
else:
    penalty = 0.0
```

**Default**: `budget_penalty = 10.0` (configurable)

**Effect**: Strongly discourages running out of budget, forcing resource management.

### 4. Unaffordable Action Penalty
**When**: Agent attempts action it cannot afford  
**Purpose**: Teach agent to respect budget constraints  

**Formula**:
```python
if action_was_unaffordable and unaffordable_action_penalty > 0:
    penalty = -unaffordable_action_penalty
else:
    penalty = 0.0
```

**Default**: `unaffordable_action_penalty = 0.0` (disabled by default)

**Effect**: When enabled, penalizes attempting expensive actions with insufficient budget.

### 5. Delayed Reward
**When**: New measurement (COUNT or SEQUENCING) arrives  
**Purpose**: Evaluate past DOSE actions with hindsight  
**Module**: `DoseRewardCompound` (CURRENTLY DISABLED)

**Original Design** (not currently used):
```python
# When measurement lands, evaluate all pending doses
for dose_event in pending_dose_events:
    # Population term (did dosing improve distance to target?)
    pre_gap = |dose_event.pre_count - target|
    post_gap = |current_count - target|
    improvement = pre_gap - post_gap
    pop_term = improvement / population_norm
    pop_term = age_normalize(pop_term, age_of_measurement)
    
    # Genome term (were antibiotics effective against resistance?)
    genome_term = -mean((dose × toxicity × susceptibility) / TOX_TIMES_DOSE_MAX)
    genome_term = age_normalize(genome_term, age_of_sequencing)
    
    reward += w_pop * pop_term + w_genome * genome_term
```

**Current Status**: 
- **Disabled in production** (v2 reward design)
- Population changes now captured by **maintenance reward** (component #2)
- Rationale: Let TD learning connect `dose → future pop drops → better rewards`
- Simpler reward signal, less reward hacking

**Historical Note**: This was the primary reward in v1, but made learning unstable due to:
1. High variance from delayed credit assignment
2. Difficulty attributing population changes to specific doses
3. Maintenance reward provides cleaner signal

### 6. Survival Bonus
**When**: Every step (if enabled)  
**Purpose**: Encourage longer episodes  
**Module**: `SurvivalBonusReward`

**Formula** (3 scaling modes):
```python
if scaling_type == "constant":
    bonus = base_bonus

elif scaling_type == "linear":
    bonus = base_bonus * (1 + scaling_factor * t / 1000)

elif scaling_type == "exponential":
    bonus = base_bonus * exp(scaling_factor * t / 1000)

bonus = min(bonus, max_bonus)  # Cap to prevent explosion
```

**Default Config**:
- `enabled`: False (disabled by default)
- `base_bonus`: 0.01
- `scaling_type`: "constant"
- `max_bonus`: 0.1

**Example** (if enabled with linear scaling):
```python
# Step 0
bonus = 0.01 * (1 + 0.1 * 0 / 1000) = 0.01

# Step 500
bonus = 0.01 * (1 + 0.1 * 500 / 1000) = 0.015

# Step 1000
bonus = 0.01 * (1 + 0.1 * 1000 / 1000) = 0.02
```

**Effect**: Provides small positive reward for each step survived, combating early episode termination.

### 7. Budget Conservation
**When**: Every step (if enabled)  
**Purpose**: Reward efficient budget usage  
**Module**: `BudgetConservationReward`

**Formula**:
```python
# 1. Spending rate penalty
spending_rate = budget_spent_this_step / initial_budget
reward -= spending_penalty_factor * spending_rate

# 2. Reserve bonus (if budget > threshold)
budget_fraction = current_budget / initial_budget
if budget_fraction >= reserve_bonus_threshold:
    reward += reserve_bonus_magnitude

# 3. Efficiency bonus (if avg spending < 0.1% per step)
avg_spending = (initial_budget - current_budget) / timestep
if avg_spending < (initial_budget / 1000):
    reward += reserve_bonus_magnitude * 0.5

total = reward * weight
```

**Default Config**:
- `enabled`: False (disabled by default)
- `weight`: 0.01
- `reserve_bonus_threshold`: 0.5 (50% budget remaining)
- `reserve_bonus_magnitude`: 0.005

**Effect**: When enabled, encourages saving budget for later, potentially at the cost of population control.

### 8. Regular Count Bonus
**When**: COUNT action is executed  
**Purpose**: Encourage periodic monitoring, discourage spam/neglect  

**Formula**:
```python
if ts_last_count is not None:
    interval = current_timestep - ts_last_count
    target_low = regular_count_interval - 3
    target_high = regular_count_interval + 3
    
    if interval < regular_count_min_interval:
        # Too frequent (spam counting)
        bonus = -regular_count_reward * 0.5
    elif target_low <= interval <= target_high:
        # Perfect timing
        bonus = regular_count_reward
    elif interval > target_high:
        # Acceptable but not optimal
        bonus = regular_count_reward * 0.5
else:
    # First count
    bonus = regular_count_reward * 0.5
```

**Default Config**:
- `regular_count_reward`: 0.0 (disabled by default)
- `regular_count_interval`: 15 steps
- `regular_count_min_interval`: 3 steps

**Example** (if enabled with reward=0.02):
```python
# Count at step 0, then step 14
interval = 14
bonus = 0.02  # Perfect timing (14 ∈ [12, 18])

# Count at step 14, then step 16
interval = 2
bonus = -0.01  # Too frequent (spam)

# Count at step 14, then step 35
interval = 21
bonus = 0.01  # Acceptable but late
```

**Effect**: Shapes counting behavior toward regular, rhythmic monitoring.

### 9. Safe Behavior Bonus
**When**: NOT dosing when population < target  
**Purpose**: Prevent over-treatment of low populations  

**Formula**:
```python
if action != ACTION_DOSE:
    steps_since_count = t - ts_last_count if ts_last_count else inf
    has_recent_count = steps_since_count <= informed_dosing_window
    
    if has_recent_count and last_count_obs < target_population:
        bonus = safe_nondosing_reward  # Reward for NOT dosing when low
else:
    bonus = 0.0
```

**Default Config**:
- `safe_nondosing_reward`: 0.0 (disabled by default)
- `informed_dosing_window`: 10 steps

**Example** (if enabled with reward=0.02):
```python
# Step 100: COUNT shows population = 300 (target = 500)
# Step 105: Agent chooses NOOP
has_recent_count = True (5 steps ago)
population_low = True (300 < 500)
bonus = 0.02  # Reward for caution

# Step 105: Agent chooses DOSE
bonus = 0.0  # No bonus (dosing when low)
```

**Effect**: Encourages conservative behavior when population is already low.

### 10. Informed Dosing Bonus/Penalty
**When**: DOSE action is executed  
**Purpose**: Reward informed decisions, penalize blind dosing  

**Formula**:
```python
# Check 1: Do we have recent COUNT?
has_recent_count = (t - ts_last_count) <= informed_dosing_window

# Check 2: Do we have recent SEQUENCING?
has_recent_sequencing = (t - ts_last_seq) <= informed_sequencing_window

# Check 3: Is population below target?
population_below_target = (last_count_obs < target_population) if has_recent_count else False

# Apply rewards/penalties
if population_below_target:
    # CRITICAL ERROR: Dosing when population already low
    bonus = -dosing_low_population_penalty  # BIG penalty

elif has_recent_count and has_recent_sequencing:
    # BEST CASE: Informed dosing with fresh data
    bonus = informed_dosing_reward
    if last_count_obs > target_population:
        bonus += informed_dosing_above_target_reward  # Extra bonus

elif not has_recent_count or not has_recent_sequencing:
    # BAD CASE: Blind dosing without information
    bonus = -blind_dosing_penalty
```

**Default Config**:
- `informed_dosing_reward`: 0.0 (disabled by default)
- `informed_dosing_above_target_reward`: 0.0
- `informed_dosing_window`: 10 steps
- `informed_sequencing_window`: 50 steps
- `blind_dosing_penalty`: 0.0
- `dosing_low_population_penalty`: 0.0

**Example** (if enabled with penalties/rewards):
```python
# Scenario 1: COUNT at step 95 (pop=600), SEQUENCING at step 80, DOSE at step 100
has_recent_count = True (5 steps ago)
has_recent_sequencing = True (20 steps ago)
population_above_target = True (600 > 500)
bonus = informed_dosing_reward + informed_dosing_above_target_reward
# = 0.05 + 0.02 = 0.07

# Scenario 2: Never counted, DOSE at step 100
has_recent_count = False
bonus = -blind_dosing_penalty
# = -0.1 (risky blind dosing)

# Scenario 3: COUNT at step 95 (pop=300), DOSE at step 100
population_below_target = True
bonus = -dosing_low_population_penalty
# = -0.5 (CRITICAL mistake)
```

**Effect**: Strongly shapes dosing behavior toward:
1. Measuring before dosing (COUNT + SEQUENCING)
2. Only dosing when population > target
3. Avoiding blind guesses

### 11. Count Population Reward
**When**: COUNT action is executed  
**Purpose**: Immediate feedback on distance from target after counting  

**Formula**:
```python
distance = |true_population - target_population|
normalized_distance = distance / population_norm
reward = -count_population_reward * normalized_distance
```

**Default Config**:
- `count_population_reward`: 0.0 (disabled by default)

**Example** (if enabled with reward=0.5):
```python
# COUNT reveals population = 800 (target = 500)
distance = 300
normalized_distance = 300 / 1000 = 0.3
reward = -0.5 * 0.3 = -0.15  # Penalty for being far

# COUNT reveals population = 520 (target = 500)
distance = 20
normalized_distance = 20 / 1000 = 0.02
reward = -0.5 * 0.02 = -0.01  # Small penalty
```

**Effect**: Provides immediate feedback loop: count → see distance → feel penalty → learn to close gap.

### 12. Critical Inaction Penalty
**When**: Population dangerously high but agent takes no action  
**Purpose**: Force intervention in critical situations  

**Formula**:
```python
# Check if we have fresh count showing critical population
has_fresh_count = (t - ts_last_count) <= critical_freshness_window
critical_threshold = target_population * critical_high_population_threshold
population_is_critical = last_count_obs >= critical_threshold

if population_is_critical and has_fresh_count:
    # Penalty 1: Not sequencing OR dosing when count shows crisis
    if action not in [ACTION_SEQUENCING, ACTION_DOSE]:
        penalty = -critical_no_action_penalty
    
    # Penalty 2: Not dosing when BOTH count AND sequencing are fresh
    has_fresh_seq = (t - ts_last_seq) <= critical_freshness_window
    if has_fresh_seq and action != ACTION_DOSE:
        penalty += -critical_no_dose_penalty
```

**Default Config**:
- `critical_high_population_threshold`: 3.0 (3× target)
- `critical_no_action_penalty`: 0.0 (disabled by default)
- `critical_no_dose_penalty`: 0.0 (disabled by default)
- `critical_freshness_window`: 5 steps

**Example** (if enabled with penalties):
```python
# Step 100: COUNT shows population = 1600 (3.2× target)
# Step 102: Agent chooses NOOP
critical_threshold = 500 * 3.0 = 1500
population_is_critical = True (1600 > 1500)
has_fresh_count = True (2 steps ago)
penalty = -critical_no_action_penalty
# = -0.2 (should have acted!)

# Step 103: Agent chooses SEQUENCING
penalty = 0.0  # SEQUENCING is acceptable response

# Step 108: COUNT fresh (3 steps ago), SEQ fresh (5 steps ago), Agent chooses COUNT
has_fresh_count = True
has_fresh_seq = True
penalty = -critical_no_action_penalty + -critical_no_dose_penalty
# = -0.2 + -0.3 = -0.5 (you have all info, DOSE NOW!)
```

**Effect**: Prevents passivity in crisis situations, forcing the agent to either gather information or dose immediately.

---

## Reward Component Summary Table

| Component | Frequency | Default Weight | Purpose | Enabled By Default |
|-----------|-----------|----------------|---------|-------------------|
| Immediate | Per action | Varies | Action costs, shaping | ✅ Yes |
| Maintenance | Per step | 0.01 | Keep population near target | ✅ Yes |
| Budget Penalty | When budget=0 | 10.0 | Prevent bankruptcy | ✅ Yes |
| Unaffordable Action Penalty | When broke | 0.0 | Teach budget limits | ❌ No |
| Delayed | On measurement | N/A | Evaluate past doses | ❌ No (disabled) |
| Survival Bonus | Per step | 0.01 | Encourage survival | ❌ No |
| Budget Conservation | Per step | 0.01 | Reward efficiency | ❌ No |
| Regular Count Bonus | Per COUNT | 0.0 | Shape monitoring rhythm | ❌ No |
| Safe Behavior Bonus | Per non-DOSE | 0.0 | Prevent over-treatment | ❌ No |
| Informed Dosing Bonus | Per DOSE | 0.0 | Reward informed decisions | ❌ No |
| Count Population Reward | Per COUNT | 0.0 | Immediate distance feedback | ❌ No |
| Critical Inaction Penalty | When crisis | 0.0 | Force intervention | ❌ No |

**Note**: Most reward components are **disabled by default**. The minimal viable reward set is:
1. **Immediate rewards** (action costs)
2. **Maintenance reward** (population control pressure)
3. **Budget penalty** (resource management)

All other components are **optional shaping** that can be enabled via configuration.

---

## Training Dynamics

### Recurrent Policy Architecture

The agent uses a **recurrent actor-critic** architecture to handle partial observability:

```python
class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim, n_discrete, k_doses, hidden_dim, rnn_layers):
        self.encoder = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, rnn_layers)
        
        # Actor heads
        self.actor_discrete = nn.Linear(hidden_dim, n_discrete)
        self.actor_continuous = nn.Linear(hidden_dim, k_doses)
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, obs, h_prev):
        # obs: [batch, seq_len, obs_dim]
        # h_prev: [rnn_layers, batch, hidden_dim]
        
        x = F.relu(self.encoder(obs))  # [batch, seq_len, hidden_dim]
        x, h_next = self.gru(x, h_prev)  # [batch, seq_len, hidden_dim]
        
        # Discrete action logits
        logits_discrete = self.actor_discrete(x)
        
        # Continuous action (only used when DOSE is selected)
        logits_continuous = self.actor_continuous(x)
        
        # Value estimate
        value = self.critic(x)
        
        return logits_discrete, logits_continuous, value, h_next
```

**Key Features**:
1. **GRU memory**: Maintains hidden state across timesteps to remember past observations
2. **Hybrid output**: Discrete action selection + continuous dose parameters
3. **Shared encoder**: Single feature extraction for both actor and critic

### Training Loop Structure

```python
def train(cfg, env, save_dir, total_updates, logger):
    agent = initialize_agent(cfg)
    
    for update_idx in range(total_updates):
        # 1. Collect rollout (N steps)
        buffer = RolloutBuffer()
        rollout_metrics = rollout(env, agent, buffer, cfg.rollout_steps, cfg)
        
        # 2. Compute advantages with GAE
        advantages, returns = compute_gae(buffer, cfg.gamma, cfg.gae_lambda)
        
        # 3. Update policy with PPO
        train_stats = agent.update_policy(buffer, advantages, returns)
        
        # 4. Log metrics
        logger.log_metrics(update_idx, rollout_metrics, train_stats)
        
        # 5. Save checkpoints
        if (update_idx + 1) % 50 == 0:
            agent.save_model(save_dir / f"checkpoint_{update_idx+1}.pt")
```

### Credit Assignment Challenge

The agent must learn long-term dependencies:

```
Step 0:   SEQUENCING (cost=1.0, reward=0.0)
Step 1:   NOOP (reward=0.01 maintenance)
Step 2:   NOOP (reward=0.01 maintenance)
Step 3:   NOOP (reward=0.01 maintenance)
Step 4:   NOOP (reward=0.01 maintenance)
Step 5:   [Sequencing result arrives]
          DOSE (reward=-2.0 cost + 0.05 informed_dosing + future benefits)
          → Bacteria population starts declining
Step 6:   Maintenance reward improves as population drops
...
Step 20:  Population reaches target
          → Cumulative reward increase from maintenance
```

**Challenges**:
1. SEQUENCING has upfront cost but delayed benefit
2. DOSE effects take multiple steps to manifest in population changes
3. Must balance immediate costs vs long-term population control

**PPO's Solution**:
- **GAE (Generalized Advantage Estimation)**: Smooths credit assignment across time
- **Value function**: Learns to predict long-term value of states
- **Recurrent policy**: Remembers past actions and measurements in hidden state

### Hyperparameter Configuration

All training parameters are configured via YAML (see `training_config.yaml`):

```yaml
ppo:
  gamma: 0.99                 # Discount factor (far-sighted planning)
  gae_lambda: 0.95            # GAE smoothing (balance bias-variance)
  clip_eps: 0.2               # PPO clipping (trust region)
  lr: 3.0e-4                  # Learning rate
  rollout_steps: 512          # Steps per update (longer = more stable)
  epochs: 4                   # PPO optimization epochs
  seq_len: 32                 # BPTT sequence length for GRU

environment:
  max_steps: 1000             # Episode length
  k_doses: 3                  # Number of antibiotic types
  
  rewards:
    population:
      target_population: 500
      w_population_maintenance: 0.01
    
    dose:
      w_pop: 1.0
      w_genome: 0.5
      w_cost: 0.05
    
    budget:
      budget_init: 100.0
      budget_penalty: 10.0
```

### Learning Curves

Expected learning progression:

**Phase 1: Random Exploration (Updates 0-50)**
- Agent tries all actions randomly
- Budget depletes quickly from expensive SEQUENCINGs and DOSEs
- Episodes end prematurely (budget=0 or population crash)
- Mean reward: -50 to -100

**Phase 2: Cost Awareness (Updates 50-100)**
- Agent learns to avoid immediate costs
- Increased NOOP usage (free action)
- Maintenance penalty accumulates from unchecked population growth
- Mean reward: -30 to -50

**Phase 3: Basic Population Control (Updates 100-200)**
- Agent learns to COUNT and DOSE when population high
- Still blind dosing (no SEQUENCING) due to cost
- Crude control loop: COUNT → if high → DOSE
- Mean reward: -10 to 0

**Phase 4: Informed Strategy (Updates 200+)**
- Agent learns value of SEQUENCING (long-term benefit > short-term cost)
- Develops information-gathering → action pipeline
- Balances budget across episode length
- Mean reward: +5 to +20

---

## Configuration Examples

### Minimal Viable Config (Default)
```yaml
environment:
  rewards:
    population:
      w_population_maintenance: 0.01  # Only maintenance reward
    budget:
      budget_penalty: 10.0  # Prevent bankruptcy
    survival_bonus:
      enabled: false  # Disabled
    budget_conservation:
      enabled: false  # Disabled
```

**Reward signal**: Maintenance + costs + budget penalty only.

### Information-Seeking Config
```yaml
actions:
  sequencing_cost: 0.5  # Reduce sequencing cost
  count_cost: 0.0  # Free counting

environment:
  rewards:
    informed_dosing:
      reward: 0.05  # Reward informed dosing
      blind_penalty: 0.1  # Penalize blind dosing
      low_population_penalty: 0.5  # BIG penalty for dosing low pop
```

**Effect**: Strong incentive to COUNT and SEQUENCE before DOSE.

### Conservative Config
```yaml
environment:
  rewards:
    regular_monitoring:
      count_reward: 0.02  # Reward periodic counting
      safe_nondosing_reward: 0.02  # Reward NOT dosing when low
    
    critical_inaction:
      no_action_penalty: 0.2  # Penalize inaction in crisis
      no_dose_penalty: 0.3  # Penalize not dosing with full info
```

**Effect**: Encourages cautious, rhythmic monitoring with intervention only when necessary.

---

## Summary

This RL agent operates under **realistic constraints**:

1. **Partial observability**: Must pay to see state
2. **Measurement delays**: SEQUENCING takes time
3. **Budget constraints**: Cannot afford unlimited actions
4. **Temporal credit assignment**: Actions have delayed consequences

The **12-component reward system** provides:
- **Core signals**: Maintenance (population control) + costs (economics)
- **Optional shaping**: 10 additional components for fine-tuning behavior

The **recurrent policy** learns to:
- Balance information gathering vs immediate action
- Manage budget across episode length
- Associate delayed measurements with past actions
- Develop strategic sequencing → dosing pipelines

The result is a policy that mirrors **real-world clinical decision-making**: measure, diagnose, treat, monitor, repeat.
