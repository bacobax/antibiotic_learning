"""
Visualization utilities for bacteria evolution and RL training.
"""

import matplotlib.pyplot as plt
import numpy as np
import os


class Visualizer:
    """
    Visualization tools for bacteria simulation and RL training.
    """
    
    def __init__(self, output_dir='results'):
        """
        Initialize visualizer.
        
        Args:
            output_dir (str): Directory to save plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_population_dynamics(self, history, save_path=None):
        """
        Plot population size over time.
        
        Args:
            history (dict): Population history with 'size' key
            save_path (str): Path to save figure
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history['size'], linewidth=2)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Population Size', fontsize=12)
        plt.title('Bacterial Population Dynamics', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'population_dynamics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_resistance_evolution(self, history, save_path=None):
        """
        Plot average resistance evolution over time.
        
        Args:
            history (dict): Population history with 'avg_resistance' key
            save_path (str): Path to save figure
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history['avg_resistance'], linewidth=2, color='red')
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Average Resistance', fontsize=12)
        plt.title('Antibiotic Resistance Evolution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'resistance_evolution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_training_rewards(self, rewards, window=100, save_path=None):
        """
        Plot training rewards with moving average.
        
        Args:
            rewards (list): Episode rewards
            window (int): Window size for moving average
            save_path (str): Path to save figure
        """
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, alpha=0.3, label='Raw Rewards')
        
        # Calculate moving average
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards)), moving_avg, 
                    linewidth=2, label=f'{window}-Episode Average')
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Total Reward', fontsize=12)
        plt.title('Training Progress', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'training_rewards.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_antibiotic_doses(self, doses, save_path=None):
        """
        Plot antibiotic doses over time.
        
        Args:
            doses (list): List of doses
            save_path (str): Path to save figure
        """
        plt.figure(figsize=(10, 6))
        plt.plot(doses, linewidth=2, color='green')
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Antibiotic Dose', fontsize=12)
        plt.title('Antibiotic Dosing Strategy', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'antibiotic_doses.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_combined_metrics(self, env_history, save_path=None):
        """
        Plot multiple metrics in subplots.
        
        Args:
            env_history (dict): Environment history
            save_path (str): Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Population
        axes[0, 0].plot(env_history['population'], linewidth=2)
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Population Size')
        axes[0, 0].set_title('Population Dynamics')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Resistance
        axes[0, 1].plot(env_history['resistance'], linewidth=2, color='red')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Avg Resistance')
        axes[0, 1].set_title('Resistance Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Doses
        axes[1, 0].plot(env_history['doses'], linewidth=2, color='green')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Dose')
        axes[1, 0].set_title('Antibiotic Dosing')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Rewards
        axes[1, 1].plot(env_history['rewards'], linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].set_title('Rewards')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'combined_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
