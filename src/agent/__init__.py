"""
Agent module for reinforcement learning.

This module contains classes for RL agents that learn to optimize antibiotic dosing.
"""

from .rl_agent import RLAgent
from .q_learning_agent import QLearningAgent

# DQN requires PyTorch - import only if available
try:
    from .dqn_agent import DQNAgent
    __all__ = ["RLAgent", "QLearningAgent", "DQNAgent"]
except ImportError:
    __all__ = ["RLAgent", "QLearningAgent"]
