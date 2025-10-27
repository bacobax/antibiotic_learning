# Architecture Overview

## System Components

The antibiotic learning system consists of three main components:

### 1. Bacteria Simulation Module

The bacteria simulation implements an evolutionary algorithm to model population dynamics:

- **BacteriaCell**: Individual bacteria with genetic properties
  - Resistance level (0.0 to 1.0)
  - Growth rate
  - Mutation rate
  - Genome representation
  
- **BacteriaPopulation**: Population management
  - Growth dynamics (logistic model)
  - Antibiotic response
  - Natural death
  - Statistical tracking
  
- **EvolutionEngine**: Evolutionary processes
  - Selection
  - Mutation
  - Fitness calculation
  - Adaptation tracking

### 2. Reinforcement Learning Module

RL agents learn to optimize antibiotic dosing:

- **RLAgent**: Abstract base class
  - Common interface for all agents
  - Action selection
  - Learning updates
  - Model persistence
  
- **QLearningAgent**: Tabular Q-Learning
  - Discrete state/action spaces
  - Q-table representation
  - Epsilon-greedy exploration
  
- **DQNAgent**: Deep Q-Network
  - Neural network function approximation
  - Experience replay
  - Target network
  - Suitable for continuous state spaces

### 3. Environment Module

Gymnasium-compatible environment for RL training:

- **AntibioticEnv**: Main environment
  - Observation: [population_size, avg_resistance, avg_growth_rate, previous_dose]
  - Action: Discrete antibiotic dose levels
  - Reward: Population control - resistance penalty - dose penalty
  - Episode termination: Extinction or max steps

## Data Flow

```
1. Environment Reset
   ├─> Initialize BacteriaPopulation
   ├─> Initialize EvolutionEngine
   └─> Return initial observation

2. Agent-Environment Loop
   ├─> Agent observes state
   ├─> Agent selects action (dose)
   ├─> Environment applies dose
   │   ├─> Population grows
   │   ├─> Antibiotic kills susceptible cells
   │   └─> Evolution updates genetics
   ├─> Environment calculates reward
   ├─> Agent updates policy
   └─> Repeat until termination

3. Episode End
   ├─> Record metrics
   ├─> Save visualizations
   └─> Update agent parameters
```

## Key Algorithms

### Evolutionary Algorithm

1. **Reproduction**: Cells reproduce based on growth rate and carrying capacity
2. **Mutation**: Random genetic changes during reproduction
3. **Selection**: Antibiotic pressure selects for resistant cells
4. **Adaptation**: Population-level resistance increases over time

### Reinforcement Learning

**Q-Learning Update**:
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

**DQN Training**:
1. Store experience (s, a, r, s') in replay buffer
2. Sample random batch from buffer
3. Compute target: y = r + γ max Q_target(s', a')
4. Update network: minimize (Q(s,a) - y)²
5. Periodically update target network

## Reward Design

The reward function balances multiple objectives:

```python
reward = population_penalty     # Negative for high population
       + resistance_penalty     # Negative for high resistance
       + dose_penalty          # Negative for antibiotic use
       + control_bonus         # Positive for good control
       + extinction_penalty    # Large negative for extinction
```

This encourages the agent to:
- Keep population low
- Minimize resistance development
- Use minimal effective antibiotic doses
- Avoid complete extinction

## Extension Points

The architecture is designed for extension:

1. **New Agents**: Inherit from `RLAgent`
2. **Custom Environments**: Modify `AntibioticEnv`
3. **Different Bacteria Models**: Replace `BacteriaCell` implementation
4. **Alternative Rewards**: Override `_calculate_reward()`
5. **Additional Features**: Extend observation space

## Performance Considerations

- **Bacteria Simulation**: O(n) per step, where n is population size
- **Q-Learning**: O(1) per update (table lookup)
- **DQN**: O(batch_size) per update (neural network)
- **Memory**: DQN uses replay buffer (configurable size)

## Design Patterns

- **Strategy Pattern**: Interchangeable RL agents
- **Observer Pattern**: Metrics tracking
- **Factory Pattern**: Agent and environment creation
- **Template Method**: Base agent class defines training loop structure
