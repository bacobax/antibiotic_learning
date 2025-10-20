"""
Deep Q-Network (DQN) agent for continuous state space.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from .rl_agent import RLAgent


class DQN(nn.Module):
    """Deep Q-Network neural network."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize DQN network.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Hidden layer dimension
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        """Forward pass through network."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent(RLAgent):
    """
    Deep Q-Network agent for complex state spaces.
    
    Uses neural networks to approximate Q-values and experience replay
    for stable learning.
    """
    
    def __init__(self, state_dim=4, action_space_size=10, learning_rate=0.001,
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=10000, batch_size=64):
        """
        Initialize DQN agent.
        
        Args:
            state_dim (int): Dimension of state vector
            action_space_size (int): Number of discrete actions
            learning_rate (float): Learning rate for optimizer
            gamma (float): Discount factor
            epsilon (float): Initial exploration rate
            epsilon_decay (float): Epsilon decay rate
            epsilon_min (float): Minimum epsilon
            buffer_size (int): Size of replay buffer
            batch_size (int): Batch size for training
        """
        super().__init__(action_space_size, learning_rate)
        self.state_dim = state_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_dim, action_space_size).to(self.device)
        self.target_network = DQN(state_dim, action_space_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        self.update_counter = 0
        self.target_update_freq = 100  # Update target network every N steps
    
    def state_to_tensor(self, state):
        """
        Convert state to tensor.
        
        Args:
            state: State dict or array
            
        Returns:
            torch.Tensor: State tensor
        """
        if isinstance(state, dict):
            # Extract relevant features from state dict
            features = [
                state.get('population_size', 0) / 10000.0,  # Normalize
                state.get('avg_resistance', 0),
                state.get('avg_growth_rate', 1.0),
                state.get('cells_killed', 0) / 1000.0
            ]
            state_array = np.array(features[:self.state_dim])
        else:
            state_array = np.array(state)
        
        return torch.FloatTensor(state_array).to(self.device)
    
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current environment state
            
        Returns:
            int: Action index
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space_size)
        
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, done):
        """
        Store experience and train network.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Store experience in replay buffer
        self.memory.append((state, action, reward, next_state, done))
        
        # Train if enough samples
        if len(self.memory) >= self.batch_size:
            self._train_step()
        
        self.total_reward += reward
        self.update_counter += 1
        
        # Update target network periodically
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _train_step(self):
        """Perform one training step using experience replay."""
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        state_batch = torch.stack([self.state_to_tensor(s) for s in states])
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.stack([self.state_to_tensor(s) for s in next_states])
        done_batch = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """
        Save model parameters.
        
        Args:
            filepath (str): Path to save file
        """
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': self.episode
        }, filepath)
    
    def load(self, filepath):
        """
        Load model parameters.
        
        Args:
            filepath (str): Path to load file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.episode = checkpoint['episode']
