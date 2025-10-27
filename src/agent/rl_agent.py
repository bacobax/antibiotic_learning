"""
Base RL Agent class for antibiotic dosing.
"""

import numpy as np
from abc import ABC, abstractmethod


class RLAgent(ABC):
    """
    Abstract base class for reinforcement learning agents.
    
    Agents learn to optimize antibiotic dosing to control bacterial populations
    while minimizing resistance development.
    """
    
    def __init__(self, action_space_size=10, learning_rate=0.001):
        """
        Initialize RL agent.
        
        Args:
            action_space_size (int): Number of discrete dose levels
            learning_rate (float): Learning rate for updates
        """
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.episode = 0
        self.total_reward = 0.0
        
    @abstractmethod
    def select_action(self, state):
        """
        Select action based on current state.
        
        Args:
            state: Current environment state
            
        Returns:
            int: Action index
        """
        pass
    
    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """
        Update agent based on experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        pass
    
    @abstractmethod
    def save(self, filepath):
        """
        Save agent parameters.
        
        Args:
            filepath (str): Path to save file
        """
        pass
    
    @abstractmethod
    def load(self, filepath):
        """
        Load agent parameters.
        
        Args:
            filepath (str): Path to load file
        """
        pass
    
    def get_action_value(self, action_idx, max_dose=1.0):
        """
        Convert action index to actual dose value.
        
        Args:
            action_idx (int): Action index
            max_dose (float): Maximum dose value
            
        Returns:
            float: Actual dose value
        """
        return (action_idx / (self.action_space_size - 1)) * max_dose
    
    def reset_episode(self):
        """Reset episode-specific statistics."""
        self.episode += 1
        self.total_reward = 0.0
