# Antibiotic Learning

A bacteria environment simulation using evolutionary algorithms combined with reinforcement learning for optimal antibiotic dosing.

## Overview

This project simulates bacterial population dynamics and evolution under antibiotic pressure. A reinforcement learning agent learns to optimize antibiotic dosing strategies by observing bacterial concentration and receiving rewards based on population control and resistance management.

## Features

- **Bacterial Evolution Simulation**: Evolutionary algorithm modeling bacteria reproduction, mutation, and resistance development
- **Reinforcement Learning Agents**: Multiple RL algorithms (Q-Learning, DQN) for learning optimal dosing strategies
- **Gymnasium Environment**: Standard RL environment interface for easy integration with various RL libraries
- **Visualization Tools**: Plotting utilities for population dynamics, resistance evolution, and training progress
- **Metrics Tracking**: Comprehensive metrics for evaluating agent performance

## Project Structure

```
antibiotic_learning/
├── src/
│   ├── bacteria/              # Bacterial evolution components
│   │   ├── bacteria_cell.py   # Individual bacteria with genetic properties
│   │   ├── bacteria_population.py  # Population management
│   │   └── evolution.py       # Evolutionary algorithm engine
│   ├── agent/                 # Reinforcement learning agents
│   │   ├── rl_agent.py        # Base agent class
│   │   ├── q_learning_agent.py  # Tabular Q-Learning
│   │   └── dqn_agent.py       # Deep Q-Network
│   ├── environment/           # RL environment
│   │   └── antibiotic_env.py  # Gymnasium-compatible environment
│   └── utils/                 # Utilities
│       ├── logger.py          # Logging utilities
│       ├── visualizer.py      # Visualization tools
│       └── metrics.py         # Metrics tracking
├── tests/
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
├── examples/                  # Example scripts
│   ├── basic_simulation.py    # Simple bacteria simulation
│   ├── train_qlearning.py     # Train Q-Learning agent
│   └── train_rl_agent.py      # Train DQN agent
├── config/                    # Configuration files
│   └── default_config.yaml    # Default configuration
└── docs/                      # Documentation

```

## Installation

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- Gymnasium
- Matplotlib

### Setup

1. Clone the repository:
```bash
git clone https://github.com/bacobax/antibiotic_learning.git
cd antibiotic_learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### Basic Bacteria Simulation

Run a simple bacteria evolution simulation without RL:

```bash
python examples/basic_simulation.py
```

This demonstrates bacterial population dynamics under various antibiotic dosing schedules.

### Train a Q-Learning Agent

Train a tabular Q-Learning agent:

```bash
python examples/train_qlearning.py
```

### Train a DQN Agent

Train a Deep Q-Network agent for more complex scenarios:

```bash
python examples/train_rl_agent.py
```

## Usage Examples

### Custom Simulation

```python
from bacteria.bacteria_population import BacteriaPopulation
from bacteria.evolution import EvolutionEngine

# Create population
population = BacteriaPopulation(initial_size=100, carrying_capacity=10000)
evolution = EvolutionEngine(mutation_rate=0.01)

# Simulate steps
for step in range(100):
    dose = 0.5  # Antibiotic dose
    stats = population.step(antibiotic_dose=dose)
    evolution.evolve_step(population, antibiotic_dose=dose)
    print(f"Step {step}: Population={stats['population_size']}, "
          f"Resistance={stats['avg_resistance']:.3f}")
```

### Using the RL Environment

```python
from environment.antibiotic_env import AntibioticEnv

# Create environment
env = AntibioticEnv(initial_population=100, max_steps=200)

# Reset environment
obs, info = env.reset()

# Take actions
for _ in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break
```

### Training Custom Agent

```python
from environment.antibiotic_env import AntibioticEnv
from agent.dqn_agent import DQNAgent

env = AntibioticEnv()
agent = DQNAgent(state_dim=4, action_space_size=10)

for episode in range(1000):
    state, _ = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.update(state, action, reward, next_state, terminated or truncated)
        state = next_state
        done = terminated or truncated
    
    agent.decay_epsilon()
```

## Configuration

Edit `config/default_config.yaml` to customize:

- Bacteria parameters (initial population, carrying capacity, mutation rate)
- Environment settings (max steps, action space, max dose)
- Agent hyperparameters (learning rate, gamma, epsilon)
- Training parameters (episodes, save intervals)

## Testing

Run unit tests:
```bash
pytest tests/unit/ -v
```

Run integration tests:
```bash
pytest tests/integration/ -v
```

Run all tests:
```bash
pytest tests/ -v
```

## Results

Training results, logs, and visualizations are saved to:
- `logs/` - Training logs and metrics
- `results/` - Plots and visualizations
- `models/` - Saved agent models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{antibiotic_learning,
  title = {Antibiotic Learning: Bacterial Evolution and RL-based Dosing Optimization},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/bacobax/antibiotic_learning}
}
```

## Acknowledgments

This project combines concepts from:
- Evolutionary algorithms for population dynamics
- Reinforcement learning for sequential decision making
- Computational biology and antibiotic resistance research