# API Reference

## Bacteria Module

### BacteriaCell

Individual bacteria with genetic properties and resistance traits.

```python
class BacteriaCell(resistance=0.0, growth_rate=1.0, mutation_rate=0.01, genome=None)
```

**Parameters:**
- `resistance` (float): Resistance level (0.0 to 1.0)
- `growth_rate` (float): Natural growth rate
- `mutation_rate` (float): Probability of mutation
- `genome` (np.ndarray): Optional genetic representation

**Methods:**
- `reproduce()` → BacteriaCell: Create offspring with mutations
- `survive_antibiotic(dose)` → bool: Check survival against antibiotic
- `update_age()`: Increment cell age

### BacteriaPopulation

Manages a population of bacteria cells.

```python
class BacteriaPopulation(initial_size=100, carrying_capacity=10000)
```

**Parameters:**
- `initial_size` (int): Starting population size
- `carrying_capacity` (int): Maximum sustainable population

**Methods:**
- `grow()`: Simulate population growth
- `apply_antibiotic(dose)` → int: Apply antibiotic, returns cells killed
- `step(antibiotic_dose=0.0)` → dict: Perform one time step
- `get_size()` → int: Get current population size
- `get_average_resistance()` → float: Get average resistance
- `is_extinct()` → bool: Check if population is extinct

### EvolutionEngine

Manages evolutionary processes.

```python
class EvolutionEngine(selection_pressure=0.5, mutation_rate=0.01)
```

**Parameters:**
- `selection_pressure` (float): Strength of selection (0.0 to 1.0)
- `mutation_rate` (float): Base mutation rate

**Methods:**
- `select(population, fitness_function)` → list: Apply selection
- `mutate_population(population, environmental_stress=0.0)`: Apply mutations
- `calculate_fitness(cell, antibiotic_dose=0.0)` → float: Calculate cell fitness
- `evolve_step(population, antibiotic_dose=0.0)` → dict: Perform evolution step

## Agent Module

### RLAgent (Abstract Base Class)

Base class for all RL agents.

```python
class RLAgent(action_space_size=10, learning_rate=0.001)
```

**Abstract Methods:**
- `select_action(state)` → int: Select action for given state
- `update(state, action, reward, next_state, done)`: Update agent
- `save(filepath)`: Save agent parameters
- `load(filepath)`: Load agent parameters

### QLearningAgent

Tabular Q-Learning agent.

```python
class QLearningAgent(
    state_space_size=100,
    action_space_size=10,
    learning_rate=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01
)
```

**Parameters:**
- `state_space_size` (int): Number of discrete states
- `action_space_size` (int): Number of discrete actions
- `learning_rate` (float): Learning rate (alpha)
- `gamma` (float): Discount factor
- `epsilon` (float): Initial exploration rate
- `epsilon_decay` (float): Epsilon decay rate
- `epsilon_min` (float): Minimum epsilon

**Methods:**
- `discretize_state(continuous_state)` → int: Convert state to index
- `decay_epsilon()`: Decay exploration rate

### DQNAgent

Deep Q-Network agent.

```python
class DQNAgent(
    state_dim=4,
    action_space_size=10,
    learning_rate=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    buffer_size=10000,
    batch_size=64
)
```

**Parameters:**
- `state_dim` (int): Dimension of state vector
- `action_space_size` (int): Number of discrete actions
- `learning_rate` (float): Learning rate for optimizer
- `gamma` (float): Discount factor
- `epsilon` (float): Initial exploration rate
- `epsilon_decay` (float): Epsilon decay rate
- `epsilon_min` (float): Minimum epsilon
- `buffer_size` (int): Replay buffer size
- `batch_size` (int): Training batch size

**Methods:**
- `state_to_tensor(state)` → torch.Tensor: Convert state to tensor
- `decay_epsilon()`: Decay exploration rate

## Environment Module

### AntibioticEnv

Gymnasium-compatible environment for antibiotic dosing.

```python
class AntibioticEnv(
    initial_population=100,
    carrying_capacity=10000,
    max_steps=200,
    action_space_size=10,
    max_dose=1.0,
    render_mode=None
)
```

**Parameters:**
- `initial_population` (int): Starting bacteria count
- `carrying_capacity` (int): Maximum population capacity
- `max_steps` (int): Maximum steps per episode
- `action_space_size` (int): Number of discrete dose levels
- `max_dose` (float): Maximum antibiotic dose
- `render_mode` (str): Rendering mode ('human' or 'rgb_array')

**Spaces:**
- **Observation Space**: Box(4) - [population, resistance, growth_rate, previous_dose]
- **Action Space**: Discrete(action_space_size) - Antibiotic dose levels

**Methods:**
- `reset(seed=None, options=None)` → (observation, info): Reset environment
- `step(action)` → (observation, reward, terminated, truncated, info): Execute action
- `render()`: Render environment state
- `close()`: Clean up resources

**Observation:**
```python
{
    'population_size': int,      # Current bacteria count
    'avg_resistance': float,     # Average resistance (0-1)
    'avg_growth_rate': float,    # Average growth rate
    'previous_dose': float       # Last antibiotic dose
}
```

**Reward Structure:**
- Population penalty: -population_size / 100.0
- Resistance penalty: -avg_resistance * 10.0
- Dose penalty: -dose * 2.0
- Control bonus: +5.0 (if population < 2 * initial)
- Extinction penalty: -100.0 (if extinct)

## Utilities Module

### Logger

Experiment logging and tracking.

```python
class Logger(log_dir='logs', experiment_name=None)
```

**Methods:**
- `info(message)`: Log info message
- `warning(message)`: Log warning
- `error(message)`: Log error
- `log_metrics(step, metrics_dict)`: Log metrics for step
- `save_metrics()`: Save metrics to JSON
- `log_config(config_dict)`: Save experiment configuration

### Visualizer

Visualization tools for results.

```python
class Visualizer(output_dir='results')
```

**Methods:**
- `plot_population_dynamics(history, save_path=None)`: Plot population over time
- `plot_resistance_evolution(history, save_path=None)`: Plot resistance evolution
- `plot_training_rewards(rewards, window=100, save_path=None)`: Plot training progress
- `plot_antibiotic_doses(doses, save_path=None)`: Plot dosing strategy
- `plot_combined_metrics(env_history, save_path=None)`: Plot all metrics

### MetricsTracker

Track and compute performance metrics.

```python
class MetricsTracker()
```

**Methods:**
- `reset()`: Reset all metrics
- `record_episode(total_reward, length, final_population, final_resistance, total_dose)`: Record episode
- `get_statistics(window=100)` → dict: Get statistical summary
- `print_statistics(window=100)`: Print formatted statistics
- `get_best_episode()` → dict: Get best episode information
- `calculate_success_rate(population_threshold=1000, window=100)` → float: Calculate success rate

## Example Usage

### Complete Training Loop

```python
from environment.antibiotic_env import AntibioticEnv
from agent.dqn_agent import DQNAgent
from utils.logger import Logger
from utils.visualizer import Visualizer
from utils.metrics import MetricsTracker

# Setup
env = AntibioticEnv()
agent = DQNAgent(state_dim=4, action_space_size=10)
logger = Logger(experiment_name='my_experiment')
visualizer = Visualizer()
metrics = MetricsTracker()

# Training
for episode in range(1000):
    state, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.update(state, action, reward, next_state, terminated or truncated)
        
        state = next_state
        episode_reward += reward
        done = terminated or truncated
    
    agent.decay_epsilon()
    metrics.record_episode(
        episode_reward,
        info['step'],
        info['population_size'],
        info['avg_resistance'],
        sum(env.history['doses'])
    )
    
    if episode % 100 == 0:
        logger.log_metrics(episode, {'reward': episode_reward})
        metrics.print_statistics()

# Save results
agent.save('models/final_agent.pth')
visualizer.plot_combined_metrics(env.history)
logger.save_metrics()
```
