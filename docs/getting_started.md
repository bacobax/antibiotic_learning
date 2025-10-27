# Getting Started Guide

## Introduction

Welcome to the Antibiotic Learning project! This guide will help you get started with the codebase.

## Quick Start (3 minutes)

### 1. Install Dependencies

```bash
# Install basic dependencies (required)
pip install numpy scipy matplotlib gymnasium

# Optional: Install PyTorch for DQN agent
pip install torch

# Optional: Install development dependencies
pip install pytest black flake8
```

### 2. Run the Quick Demo

```bash
python examples/quick_demo.py
```

This will run a quick 10-episode training session to verify everything works.

### 3. Run the Basic Simulation

```bash
python examples/basic_simulation.py
```

This demonstrates bacteria evolution without RL, showing how antibiotic doses affect population dynamics.

## Understanding the Code

### Core Components

1. **Bacteria Simulation** (`src/bacteria/`)
   - `bacteria_cell.py`: Individual bacteria with resistance traits
   - `bacteria_population.py`: Population-level dynamics
   - `evolution.py`: Evolutionary algorithm

2. **RL Agents** (`src/agent/`)
   - `rl_agent.py`: Base class for all agents
   - `q_learning_agent.py`: Simple tabular Q-Learning (no PyTorch needed)
   - `dqn_agent.py`: Deep Q-Network (requires PyTorch)

3. **Environment** (`src/environment/`)
   - `antibiotic_env.py`: Gymnasium environment for RL training

4. **Utilities** (`src/utils/`)
   - `logger.py`: Experiment logging
   - `visualizer.py`: Result visualization
   - `metrics.py`: Performance tracking

### Directory Structure

```
antibiotic_learning/
├── src/           # Source code
├── tests/         # Unit and integration tests
├── examples/      # Example scripts
├── config/        # Configuration files
└── docs/          # Documentation
```

## Common Tasks

### Train a Q-Learning Agent

```bash
python examples/train_qlearning.py
```

Results saved to:
- `logs/` - Training logs
- `results/qlearning_training/` - Plots
- `models/` - Trained models

### Train a DQN Agent

Requires PyTorch:
```bash
pip install torch
python examples/train_rl_agent.py
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Just unit tests
pytest tests/unit/ -v

# Just integration tests
pytest tests/integration/ -v
```

### Create Custom Simulation

```python
from bacteria.bacteria_population import BacteriaPopulation

# Create population
pop = BacteriaPopulation(initial_size=100, carrying_capacity=10000)

# Simulate with custom dosing
for step in range(100):
    dose = your_dosing_function(pop)
    stats = pop.step(antibiotic_dose=dose)
    print(f"Population: {stats['population_size']}")
```

### Train Custom Agent

```python
from environment.antibiotic_env import AntibioticEnv
from agent.q_learning_agent import QLearningAgent

env = AntibioticEnv()
agent = QLearningAgent(state_space_size=100, action_space_size=10)

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

agent.save('my_model.pkl')
```

## Configuration

Edit `config/default_config.yaml` to customize:

```yaml
bacteria:
  initial_population: 100
  carrying_capacity: 10000

environment:
  max_steps: 200
  action_space_size: 10

agent:
  learning_rate: 0.001
  gamma: 0.99
```

## Troubleshooting

### Import Errors

Make sure you're running from the project root or add src to path:
```python
import sys
sys.path.insert(0, 'src')
```

### PyTorch Not Found

If you don't need DQN, you can use Q-Learning without PyTorch:
```python
from agent.q_learning_agent import QLearningAgent  # No torch needed
```

To use DQN, install PyTorch:
```bash
pip install torch
```

### Tests Failing

Make sure dependencies are installed:
```bash
pip install -r requirements.txt
pip install pytest
```

## Next Steps

1. **Read the Documentation**
   - `docs/architecture.md` - System design
   - `docs/api_reference.md` - API details

2. **Explore Examples**
   - Modify `examples/basic_simulation.py` to try different dosing strategies
   - Adjust hyperparameters in training scripts

3. **Experiment**
   - Try different reward functions
   - Implement new agent types
   - Add more complex bacteria models

4. **Contribute**
   - Add new features
   - Improve documentation
   - Report bugs

## Resources

- **GitHub**: https://github.com/bacobax/antibiotic_learning
- **Issues**: Report bugs or request features
- **Discussions**: Ask questions and share ideas

## Support

If you encounter issues:
1. Check the troubleshooting section
2. Review the API reference
3. Open an issue on GitHub

Happy coding!
