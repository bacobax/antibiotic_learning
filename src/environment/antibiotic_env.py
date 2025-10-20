"""
Gymnasium environment for bacteria antibiotic interaction.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bacteria.bacteria_population import BacteriaPopulation
from bacteria.evolution import EvolutionEngine


class AntibioticEnv(gym.Env):
    """
    Custom Gymnasium environment for antibiotic dosing optimization.
    
    The agent observes bacterial population state and chooses antibiotic doses
    to minimize bacterial population while avoiding resistance development.
    
    Observation Space:
        - population_size: Current number of bacteria
        - avg_resistance: Average resistance level
        - avg_growth_rate: Average growth rate
        - previous_dose: Last antibiotic dose applied
    
    Action Space:
        - Discrete doses from 0.0 to max_dose
    
    Reward:
        - Negative reward for high bacterial population
        - Negative reward for increasing resistance
        - Penalty for excessive antibiotic use
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, initial_population=100, carrying_capacity=10000,
                 max_steps=200, action_space_size=10, max_dose=1.0,
                 render_mode=None):
        """
        Initialize environment.
        
        Args:
            initial_population (int): Starting bacteria count
            carrying_capacity (int): Maximum population capacity
            max_steps (int): Maximum steps per episode
            action_space_size (int): Number of discrete dose levels
            max_dose (float): Maximum antibiotic dose
            render_mode (str): Rendering mode
        """
        super().__init__()
        
        self.initial_population = initial_population
        self.carrying_capacity = carrying_capacity
        self.max_steps = max_steps
        self.max_dose = max_dose
        self.render_mode = render_mode
        
        # Action space: discrete antibiotic doses
        self.action_space = spaces.Discrete(action_space_size)
        
        # Observation space: [population_size, avg_resistance, avg_growth_rate, previous_dose]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([float(carrying_capacity), 1.0, 2.0, max_dose]),
            dtype=np.float32
        )
        
        # Initialize components
        self.population = None
        self.evolution = None
        self.current_step = 0
        self.previous_dose = 0.0
        self.episode_reward = 0.0
        
        # Tracking
        self.history = {
            'population': [],
            'resistance': [],
            'doses': [],
            'rewards': []
        }
    
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        
        Args:
            seed (int): Random seed
            options (dict): Additional options
            
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset components
        self.population = BacteriaPopulation(
            initial_size=self.initial_population,
            carrying_capacity=self.carrying_capacity
        )
        self.evolution = EvolutionEngine()
        
        self.current_step = 0
        self.previous_dose = 0.0
        self.episode_reward = 0.0
        
        # Reset history
        self.history = {
            'population': [],
            'resistance': [],
            'doses': [],
            'rewards': []
        }
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Execute one time step in the environment.
        
        Args:
            action (int): Action index
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Convert action to dose
        dose = self._action_to_dose(action)
        self.previous_dose = dose
        
        # Apply dose to population and evolve
        stats = self.population.step(antibiotic_dose=dose)
        self.evolution.evolve_step(self.population, antibiotic_dose=dose)
        
        # Calculate reward
        reward = self._calculate_reward(stats, dose)
        self.episode_reward += reward
        
        # Update step counter
        self.current_step += 1
        
        # Check termination conditions
        terminated = self.population.is_extinct()
        truncated = self.current_step >= self.max_steps
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Update history
        self.history['population'].append(stats['population_size'])
        self.history['resistance'].append(stats['avg_resistance'])
        self.history['doses'].append(dose)
        self.history['rewards'].append(reward)
        
        return observation, reward, terminated, truncated, info
    
    def _action_to_dose(self, action):
        """
        Convert discrete action to continuous dose.
        
        Args:
            action (int): Action index
            
        Returns:
            float: Antibiotic dose
        """
        return (action / (self.action_space.n - 1)) * self.max_dose
    
    def _get_observation(self):
        """
        Get current observation.
        
        Returns:
            np.ndarray: Observation vector
        """
        return np.array([
            float(self.population.get_size()),
            float(self.population.get_average_resistance()),
            float(self.population.get_average_growth_rate()),
            float(self.previous_dose)
        ], dtype=np.float32)
    
    def _get_info(self):
        """
        Get additional information.
        
        Returns:
            dict: Info dictionary
        """
        return {
            'population_size': self.population.get_size(),
            'avg_resistance': self.population.get_average_resistance(),
            'avg_growth_rate': self.population.get_average_growth_rate(),
            'step': self.current_step,
            'episode_reward': self.episode_reward
        }
    
    def _calculate_reward(self, stats, dose):
        """
        Calculate reward based on population state and action.
        
        Goals:
        - Minimize bacterial population
        - Minimize resistance development
        - Minimize antibiotic usage
        
        Args:
            stats (dict): Population statistics
            dose (float): Applied dose
            
        Returns:
            float: Reward value
        """
        population_size = stats['population_size']
        avg_resistance = stats['avg_resistance']
        
        # Reward for low population (negative for high population)
        population_penalty = -population_size / 100.0
        
        # Penalty for high resistance
        resistance_penalty = -avg_resistance * 10.0
        
        # Penalty for antibiotic use (encourage minimal effective dose)
        dose_penalty = -dose * 2.0
        
        # Bonus for keeping population controlled
        control_bonus = 0.0
        if population_size < self.initial_population * 2:
            control_bonus = 5.0
        
        # Extinction penalty (we want control, not extinction)
        extinction_penalty = -100.0 if population_size == 0 else 0.0
        
        total_reward = (
            population_penalty +
            resistance_penalty +
            dose_penalty +
            control_bonus +
            extinction_penalty
        )
        
        return total_reward
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            str or np.ndarray: Rendered output
        """
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Population: {self.population.get_size()}")
            print(f"Avg Resistance: {self.population.get_average_resistance():.3f}")
            print(f"Previous Dose: {self.previous_dose:.3f}")
            print(f"Episode Reward: {self.episode_reward:.2f}")
            print("-" * 50)
        
        # For 'rgb_array' mode, would return visualization
        # This is a placeholder - actual implementation would create image
        return None
    
    def close(self):
        """Clean up resources."""
        pass
