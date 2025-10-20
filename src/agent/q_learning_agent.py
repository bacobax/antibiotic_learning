"""
Q-Learning agent for discrete action space.
"""

import numpy as np
import pickle
from .rl_agent import RLAgent


class QLearningAgent(RLAgent):
    """
    Q-Learning agent with epsilon-greedy exploration.
    
    Uses tabular Q-learning to learn optimal antibiotic dosing policy.
    """
    
    def __init__(self, state_space_size=100, action_space_size=10, 
                 learning_rate=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01):
        """
        Initialize Q-Learning agent.
        
        Args:
            state_space_size (int): Number of discrete states
            action_space_size (int): Number of discrete actions
            learning_rate (float): Learning rate (alpha)
            gamma (float): Discount factor
            epsilon (float): Initial exploration rate
            epsilon_decay (float): Epsilon decay rate per episode
            epsilon_min (float): Minimum epsilon value
        """
        super().__init__(action_space_size, learning_rate)
        self.state_space_size = state_space_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table
        self.q_table = np.zeros((state_space_size, action_space_size))
    
    def discretize_state(self, continuous_state):
        """
        Convert continuous state to discrete state index.
        
        Args:
            continuous_state (dict or array): Continuous state values
            
        Returns:
            int: Discrete state index
        """
        # Simple discretization - can be improved
        if isinstance(continuous_state, dict):
            population = continuous_state.get('population_size', 0)
            resistance = continuous_state.get('avg_resistance', 0)
            state_value = population / 100 + resistance * 50
        else:
            state_value = continuous_state[0] if len(continuous_state) > 0 else 0
        
        state_idx = int(min(state_value, self.state_space_size - 1))
        return max(0, state_idx)
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current environment state
            
        Returns:
            int: Action index
        """
        state_idx = self.discretize_state(state)
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)
        else:
            return np.argmax(self.q_table[state_idx])
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-table using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        state_idx = self.discretize_state(state)
        next_state_idx = self.discretize_state(next_state)
        
        # Q-learning update
        current_q = self.q_table[state_idx, action]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state_idx])
            target_q = reward + self.gamma * max_next_q
        
        # Update Q-value
        self.q_table[state_idx, action] += self.learning_rate * (target_q - current_q)
        
        self.total_reward += reward
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """
        Save Q-table and parameters.
        
        Args:
            filepath (str): Path to save file
        """
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'episode': self.episode,
            'state_space_size': self.state_space_size,
            'action_space_size': self.action_space_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath):
        """
        Load Q-table and parameters.
        
        Args:
            filepath (str): Path to load file
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = data['q_table']
        self.epsilon = data['epsilon']
        self.episode = data['episode']
        self.state_space_size = data['state_space_size']
        self.action_space_size = data['action_space_size']
