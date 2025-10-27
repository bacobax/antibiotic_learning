"""
Metrics tracking for experiment evaluation.
"""

import numpy as np


class MetricsTracker:
    """
    Track and compute metrics for bacteria simulation and RL training.
    """
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.population_sizes = []
        self.resistance_levels = []
        self.doses_used = []
    
    def record_episode(self, total_reward, length, final_population, 
                      final_resistance, total_dose):
        """
        Record metrics for a completed episode.
        
        Args:
            total_reward (float): Total episode reward
            length (int): Episode length
            final_population (int): Final population size
            final_resistance (float): Final average resistance
            total_dose (float): Total antibiotic used
        """
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(length)
        self.population_sizes.append(final_population)
        self.resistance_levels.append(final_resistance)
        self.doses_used.append(total_dose)
    
    def get_statistics(self, window=100):
        """
        Get statistical summary of recent performance.
        
        Args:
            window (int): Window size for recent statistics
            
        Returns:
            dict: Statistics dictionary
        """
        recent_rewards = self.episode_rewards[-window:]
        recent_lengths = self.episode_lengths[-window:]
        recent_population = self.population_sizes[-window:]
        recent_resistance = self.resistance_levels[-window:]
        recent_doses = self.doses_used[-window:]
        
        stats = {
            'mean_reward': np.mean(recent_rewards) if recent_rewards else 0.0,
            'std_reward': np.std(recent_rewards) if recent_rewards else 0.0,
            'mean_length': np.mean(recent_lengths) if recent_lengths else 0.0,
            'mean_population': np.mean(recent_population) if recent_population else 0.0,
            'mean_resistance': np.mean(recent_resistance) if recent_resistance else 0.0,
            'mean_dose': np.mean(recent_doses) if recent_doses else 0.0,
            'total_episodes': len(self.episode_rewards)
        }
        
        return stats
    
    def print_statistics(self, window=100):
        """
        Print formatted statistics.
        
        Args:
            window (int): Window size for recent statistics
        """
        stats = self.get_statistics(window)
        print(f"\n{'='*60}")
        print(f"Statistics (last {window} episodes):")
        print(f"{'='*60}")
        print(f"Total Episodes:      {stats['total_episodes']}")
        print(f"Mean Reward:         {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Mean Length:         {stats['mean_length']:.2f}")
        print(f"Mean Population:     {stats['mean_population']:.2f}")
        print(f"Mean Resistance:     {stats['mean_resistance']:.4f}")
        print(f"Mean Dose Used:      {stats['mean_dose']:.4f}")
        print(f"{'='*60}\n")
    
    def get_best_episode(self):
        """
        Get information about the best performing episode.
        
        Returns:
            dict: Best episode information
        """
        if not self.episode_rewards:
            return None
        
        best_idx = np.argmax(self.episode_rewards)
        
        return {
            'episode': best_idx,
            'reward': self.episode_rewards[best_idx],
            'length': self.episode_lengths[best_idx],
            'population': self.population_sizes[best_idx],
            'resistance': self.resistance_levels[best_idx],
            'dose': self.doses_used[best_idx]
        }
    
    def calculate_success_rate(self, population_threshold=1000, window=100):
        """
        Calculate success rate (episodes with controlled population).
        
        Args:
            population_threshold (int): Threshold for successful control
            window (int): Window size
            
        Returns:
            float: Success rate (0.0 to 1.0)
        """
        recent_population = self.population_sizes[-window:]
        if not recent_population:
            return 0.0
        
        successes = sum(1 for pop in recent_population if pop < population_threshold)
        return successes / len(recent_population)
