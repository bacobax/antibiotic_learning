"""
Train RL Agent with Real-time Visualization.

This script trains the RL agent while displaying the bacteria simulation
in real-time. You can watch the agent learn and see how it interacts with
the environment during training.

Usage:
    python src/train_with_visualization.py --config src/rl/configs/training_config.yaml
    python src/train_with_visualization.py --config src/rl/configs/training_config_fast.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PyQt5 import QtWidgets, QtCore

from simulation.model import BacteriaModel
from rl.config_loader import load_config, CompleteConfig, save_config
from rl.training_config import PPOConfig, set_global_seed
from rl.agent import RLAgent
from rl.env_wrapper import PetriEnvWrapper, ACTION_DOSE, ACTION_COUNT_BACTERIA, ACTION_SEQUENCING, ACTION_NOOP
from rl.models import RecurrentActorCritic
from rl.buffer import RolloutBuffer
from rl.logger import TrainingLogger
from rl.training_utils import (
    _initialize_agent,
    _log_training_start,
    _handle_checkpoint,
    _finalize_training,
    _setup_logger_and_log_startup,
    _create_environment,
    _build_ppo_config,
    _save_configs,
)
from simulation.visualization import SimulationVisualizer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.animation as animation


ACTION_NAMES = {
    ACTION_NOOP: "NOOP",
    ACTION_COUNT_BACTERIA: "COUNT",
    ACTION_SEQUENCING: "SEQUENCE",
    ACTION_DOSE: "DOSE"
}


class VisualizationWindow(QtWidgets.QMainWindow):
    """PyQt5 window for real-time training visualization"""
    
    def __init__(self, visualizer):
        super().__init__()
        self.visualizer = visualizer
        self.setWindowTitle("RL Training - Live Simulation")
        self.setGeometry(100, 100, 1400, 700)
        
        # Embed matplotlib figure
        self.canvas = FigureCanvas(visualizer.fig)
        self.setCentralWidget(self.canvas)
        
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.addToolBar(self.toolbar)


class TrainingControlPanel(QtWidgets.QMainWindow):
    """Control panel for training visualization"""
    
    def __init__(self, on_pause_toggle, initial_budget: float = 100.0, max_steps: int = 512):
        super().__init__()
        self.setWindowTitle("Training Control Panel")
        self.setGeometry(0, 0, 800, 1400)  # Wider for 2-column layout
        
        self.on_pause_toggle = on_pause_toggle
        self.initial_budget = initial_budget  # Store for fixed axes
        self.max_steps = max_steps  # Store for fixed x-axis
        
        # Data for plots
        self.episode_numbers = []
        self.episode_lengths = []
        self.budget_spent_history = []
        self.budget_remaining_history = []
        
        # Current episode budget tracking (for step-by-step graph)
        self.current_episode_budget_steps = []
        self.current_episode_budget_values = []
        
        # Reward component tracking
        self.reward_immediate_history = []  # Stored but typically not plotted (composite)
        self.reward_action_cost_penalty_history = []  # Pure cost penalty from w_cost
        self.reward_maintenance_history = []
        self.reward_budget_penalty_history = []
        self.reward_delayed_history = []
        self.reward_survival_bonus_history = []
        self.reward_budget_conservation_history = []
        self.reward_regular_count_bonus_history = []
        self.reward_safe_behavior_bonus_history = []
        self.reward_informed_dosing_bonus_history = []
        self.reward_count_population_history = []
        self.reward_critical_inaction_penalty_history = []
        self.reward_critical_noop_penalty_history = []
        self.reward_prediction_history = []
        
        # Create central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Title
        title = QtWidgets.QLabel("Training Monitor")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Control buttons
        controls_group = QtWidgets.QGroupBox("Training Control")
        controls_layout = QtWidgets.QVBoxLayout()
        
        self.pause_button = QtWidgets.QPushButton("Start Training")
        self.pause_button.clicked.connect(on_pause_toggle)
        controls_layout.addWidget(self.pause_button)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # ===== Split Statistics into 2 columns =====
        stats_container = QtWidgets.QWidget()
        stats_layout = QtWidgets.QHBoxLayout(stats_container)
        
        # LEFT COLUMN: Training Statistics
        train_group = QtWidgets.QGroupBox("Training Statistics")
        train_layout = QtWidgets.QVBoxLayout()
        
        self.train_display = QtWidgets.QTextEdit()
        self.train_display.setReadOnly(True)
        self.train_display.setMaximumHeight(200)
        train_layout.addWidget(self.train_display)
        
        train_group.setLayout(train_layout)
        stats_layout.addWidget(train_group)
        
        # RIGHT COLUMN: Overall Action Distribution
        action_dist_group = QtWidgets.QGroupBox("Overall Action Distribution")
        action_dist_layout = QtWidgets.QVBoxLayout()
        
        self.action_dist_display = QtWidgets.QTextEdit()
        self.action_dist_display.setReadOnly(True)
        self.action_dist_display.setMaximumHeight(200)
        action_dist_layout.addWidget(self.action_dist_display)
        
        action_dist_group.setLayout(action_dist_layout)
        stats_layout.addWidget(action_dist_group)
        
        layout.addWidget(stats_container)
        
        # ===== Split Episode Statistics into 2 columns =====
        episode_container = QtWidgets.QWidget()
        episode_layout = QtWidgets.QHBoxLayout(episode_container)
        
        # LEFT COLUMN: Current Episode
        episode_group = QtWidgets.QGroupBox("Current Episode")
        episode_group_layout = QtWidgets.QVBoxLayout()
        
        self.episode_display = QtWidgets.QTextEdit()
        self.episode_display.setReadOnly(True)
        self.episode_display.setMaximumHeight(200)
        episode_group_layout.addWidget(self.episode_display)
        
        episode_group.setLayout(episode_group_layout)
        episode_layout.addWidget(episode_group)
        
        # RIGHT COLUMN: Episode Action Distribution
        episode_action_group = QtWidgets.QGroupBox("Episode Action Distribution")
        episode_action_layout = QtWidgets.QVBoxLayout()
        
        self.episode_action_display = QtWidgets.QTextEdit()
        self.episode_action_display.setReadOnly(True)
        self.episode_action_display.setMaximumHeight(200)
        episode_action_layout.addWidget(self.episode_action_display)
        
        episode_action_group.setLayout(episode_action_layout)
        episode_layout.addWidget(episode_action_group)
        
        layout.addWidget(episode_container)
        
        # ===== Graphs Row 1: Episode Length + Budget (side by side) =====
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        
        graphs_row1_container = QtWidgets.QWidget()
        graphs_row1_layout = QtWidgets.QHBoxLayout(graphs_row1_container)
        
        # Episode Length Graph
        episode_length_group = QtWidgets.QGroupBox("Episode Length Over Time")
        episode_length_layout = QtWidgets.QVBoxLayout()
        
        self.episode_length_fig = Figure(figsize=(4, 2), dpi=100)
        self.episode_length_ax = self.episode_length_fig.add_subplot(111)
        self.episode_length_canvas = FigureCanvas(self.episode_length_fig)
        self.episode_length_canvas.setMaximumHeight(200)
        
        self.episode_length_ax.set_xlabel('Episode')
        self.episode_length_ax.set_ylabel('Steps')
        self.episode_length_ax.set_title('Episode Length')
        self.episode_length_ax.grid(True, alpha=0.3)
        
        episode_length_layout.addWidget(self.episode_length_canvas)
        episode_length_group.setLayout(episode_length_layout)
        graphs_row1_layout.addWidget(episode_length_group)
        
        # Budget Tracking Graph (Final Budget per Episode)
        budget_group = QtWidgets.QGroupBox("Final Budget Remaining Per Episode")
        budget_layout = QtWidgets.QVBoxLayout()
        
        self.budget_fig = Figure(figsize=(4, 2), dpi=100)
        self.budget_ax = self.budget_fig.add_subplot(111)
        self.budget_canvas = FigureCanvas(self.budget_fig)
        self.budget_canvas.setMaximumHeight(200)
        
        self.budget_ax.set_xlabel('Episode')
        self.budget_ax.set_ylabel('Budget Remaining')
        self.budget_ax.set_title('Final Budget Per Episode')
        self.budget_ax.grid(True, alpha=0.3)
        
        budget_layout.addWidget(self.budget_canvas)
        budget_group.setLayout(budget_layout)
        graphs_row1_layout.addWidget(budget_group)
        
        layout.addWidget(graphs_row1_container)
        
        # ===== NEW: Budget Over Steps in Current Episode =====
        current_budget_group = QtWidgets.QGroupBox("Budget Over Steps (Current Episode)")
        current_budget_layout = QtWidgets.QVBoxLayout()
        
        self.current_budget_fig = Figure(figsize=(8, 2), dpi=100)
        self.current_budget_ax = self.current_budget_fig.add_subplot(111)
        self.current_budget_canvas = FigureCanvas(self.current_budget_fig)
        self.current_budget_canvas.setMaximumHeight(200)
        
        self.current_budget_ax.set_xlabel('Step')
        self.current_budget_ax.set_ylabel('Budget')
        self.current_budget_ax.set_title('Budget Evolution in Current Episode')
        self.current_budget_ax.grid(True, alpha=0.3)
        
        current_budget_layout.addWidget(self.current_budget_canvas)
        current_budget_group.setLayout(current_budget_layout)
        layout.addWidget(current_budget_group)
        
        # ===== NEW: Reward Components Graph =====
        reward_components_group = QtWidgets.QGroupBox("Reward Components Per Episode")
        reward_components_layout = QtWidgets.QVBoxLayout()
        
        self.reward_components_fig = Figure(figsize=(8, 3), dpi=100)
        self.reward_components_ax = self.reward_components_fig.add_subplot(111)
        self.reward_components_canvas = FigureCanvas(self.reward_components_fig)
        self.reward_components_canvas.setMaximumHeight(250)
        
        self.reward_components_ax.set_xlabel('Episode')
        self.reward_components_ax.set_ylabel('Reward Value')
        self.reward_components_ax.set_title('Reward Component Breakdown')
        self.reward_components_ax.grid(True, alpha=0.3)
        
        reward_components_layout.addWidget(self.reward_components_canvas)
        reward_components_group.setLayout(reward_components_layout)
        layout.addWidget(reward_components_group)
        
        # Environment statistics
        env_group = QtWidgets.QGroupBox("Environment State")
        env_layout = QtWidgets.QVBoxLayout()
        
        self.env_display = QtWidgets.QTextEdit()
        self.env_display.setReadOnly(True)
        self.env_display.setMaximumHeight(150)
        env_layout.addWidget(self.env_display)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group)
        
        layout.addStretch()
    
    def add_episode_data(self, episode_num: int, episode_length: int, budget_spent: float, budget_remaining: float, 
                         reward_components: dict = None):
        """Add data point for completed episode"""
        self.episode_numbers.append(episode_num)
        self.episode_lengths.append(episode_length)
        self.budget_spent_history.append(budget_spent)
        self.budget_remaining_history.append(budget_remaining)
        
        # Track reward components if provided
        if reward_components:
            self.reward_immediate_history.append(reward_components.get('immediate', 0.0))
            self.reward_action_cost_penalty_history.append(reward_components.get('action_cost_penalty', 0.0))
            self.reward_maintenance_history.append(reward_components.get('maintenance', 0.0))
            self.reward_budget_penalty_history.append(reward_components.get('budget_penalty', 0.0))
            self.reward_delayed_history.append(reward_components.get('delayed', 0.0))
            self.reward_survival_bonus_history.append(reward_components.get('survival_bonus', 0.0))
            self.reward_budget_conservation_history.append(reward_components.get('budget_conservation', 0.0))
            self.reward_regular_count_bonus_history.append(reward_components.get('regular_count_bonus', 0.0))
            self.reward_safe_behavior_bonus_history.append(reward_components.get('safe_behavior_bonus', 0.0))
            self.reward_informed_dosing_bonus_history.append(reward_components.get('informed_dosing_bonus', 0.0))
            self.reward_count_population_history.append(reward_components.get('count_population', 0.0))
            self.reward_critical_inaction_penalty_history.append(reward_components.get('critical_inaction_penalty', 0.0))
            self.reward_critical_noop_penalty_history.append(reward_components.get('critical_noop_penalty', 0.0))
            self.reward_prediction_history.append(reward_components.get('prediction', 0.0))
        
        # Update episode length plot
        self.episode_length_ax.clear()
        self.episode_length_ax.plot(self.episode_numbers, self.episode_lengths, 'b-', linewidth=1)
        self.episode_length_ax.set_xlabel('Episode')
        self.episode_length_ax.set_ylabel('Steps')
        self.episode_length_ax.set_title('Episode Length')
        self.episode_length_ax.grid(True, alpha=0.3)
        self.episode_length_fig.tight_layout()
        self.episode_length_canvas.draw()
        
        # Update budget plot (show only final budget remaining)
        self.budget_ax.clear()
        self.budget_ax.plot(self.episode_numbers, self.budget_remaining_history, 'g-', 
                           linewidth=1.5)
        self.budget_ax.set_xlabel('Episode')
        self.budget_ax.set_ylabel('Budget Remaining')
        self.budget_ax.set_title('Final Budget Per Episode')
        self.budget_ax.grid(True, alpha=0.3)
        self.budget_fig.tight_layout()
        self.budget_canvas.draw()
        
        # Update reward components plot if we have data
        if reward_components and len(self.episode_numbers) > 0:
            self.reward_components_ax.clear()
            
            # Plot each reward component separately (no double-counting)
            # Only plot components that are non-zero in at least one episode
            
            if any(x != 0 for x in self.reward_action_cost_penalty_history):
                self.reward_components_ax.plot(self.episode_numbers, self.reward_action_cost_penalty_history, 
                                               label='Action Cost Penalty', linewidth=1.5, alpha=0.9)
            if any(x != 0 for x in self.reward_maintenance_history):
                self.reward_components_ax.plot(self.episode_numbers, self.reward_maintenance_history, 
                                               label='Maintenance', linewidth=1.5, alpha=0.9)
            if any(x != 0 for x in self.reward_budget_penalty_history):
                self.reward_components_ax.plot(self.episode_numbers, self.reward_budget_penalty_history, 
                                               label='Budget Penalty', linewidth=1.5, alpha=0.9)
            if any(x != 0 for x in self.reward_delayed_history):
                self.reward_components_ax.plot(self.episode_numbers, self.reward_delayed_history, 
                                               label='Delayed', linewidth=1.5, alpha=0.9)
            if any(x != 0 for x in self.reward_survival_bonus_history):
                self.reward_components_ax.plot(self.episode_numbers, self.reward_survival_bonus_history, 
                                               label='Survival Bonus', linewidth=1.5, alpha=0.9)
            if any(x != 0 for x in self.reward_budget_conservation_history):
                self.reward_components_ax.plot(self.episode_numbers, self.reward_budget_conservation_history, 
                                               label='Budget Conservation', linewidth=1.5, alpha=0.9)
            if any(x != 0 for x in self.reward_regular_count_bonus_history):
                self.reward_components_ax.plot(self.episode_numbers, self.reward_regular_count_bonus_history, 
                                               label='Regular Count', linewidth=1.5, alpha=0.9)
            if any(x != 0 for x in self.reward_safe_behavior_bonus_history):
                self.reward_components_ax.plot(self.episode_numbers, self.reward_safe_behavior_bonus_history, 
                                               label='Safe Behavior', linewidth=1.5, alpha=0.9)
            if any(x != 0 for x in self.reward_informed_dosing_bonus_history):
                self.reward_components_ax.plot(self.episode_numbers, self.reward_informed_dosing_bonus_history, 
                                               label='Informed Dosing', linewidth=1.5, alpha=0.9)
            if any(x != 0 for x in self.reward_count_population_history):
                self.reward_components_ax.plot(self.episode_numbers, self.reward_count_population_history, 
                                               label='Count Population', linewidth=1.5, alpha=0.9)
            if any(x != 0 for x in self.reward_critical_inaction_penalty_history):
                self.reward_components_ax.plot(self.episode_numbers, self.reward_critical_inaction_penalty_history, 
                                               label='Critical Inaction Penalty', linewidth=1.5, alpha=0.9)
            if any(x != 0 for x in self.reward_critical_noop_penalty_history):
                self.reward_components_ax.plot(self.episode_numbers, self.reward_critical_noop_penalty_history,
                                               label='Critical NOOP Penalty', linewidth=1.5, alpha=0.9)
            if any(x != 0 for x in self.reward_prediction_history):
                self.reward_components_ax.plot(self.episode_numbers, self.reward_prediction_history,
                                               label='Prediction', linewidth=1.5, alpha=0.9)
            
            self.reward_components_ax.set_xlabel('Episode')
            self.reward_components_ax.set_ylabel('Reward Value')
            self.reward_components_ax.set_title('Reward Component Breakdown')
            self.reward_components_ax.legend(loc='best', fontsize=8, ncol=2)
            self.reward_components_ax.grid(True, alpha=0.3)
            self.reward_components_ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
            self.reward_components_fig.tight_layout()
            self.reward_components_canvas.draw()
    
    def update_training_stats(self, stats):
        # Update training statistics display (LEFT COLUMN)
        text = f"""Update: {stats.get('update', 0)}/{stats.get('total_updates', 0)}
Elapsed Time: {stats.get('elapsed', 0):.1f}s
Mean Episode Reward: {stats.get('mean_reward', 0):.2f}
Actor Loss: {stats.get('actor_loss', 0):.4f}
Critic Loss: {stats.get('critic_loss', 0):.4f}
Episodes Completed: {stats.get('episodes_completed', 0)}
"""
        self.train_display.setText(text)
        
        # Update action distribution (RIGHT COLUMN)
        total_actions = sum(stats.get('action_counts_total', {}).values())
        action_pcts = {}
        if total_actions > 0:
            for action, count in stats.get('action_counts_total', {}).items():
                action_pcts[action] = (count / total_actions) * 100
        
        action_text = f"""Total Actions: {total_actions}

NOOP:     {action_pcts.get(0, 0):5.1f}% ({stats.get('action_counts_total', {}).get(0, 0):4d})
COUNT:    {action_pcts.get(1, 0):5.1f}% ({stats.get('action_counts_total', {}).get(1, 0):4d})
SEQUENCE: {action_pcts.get(2, 0):5.1f}% ({stats.get('action_counts_total', {}).get(2, 0):4d})
DOSE:     {action_pcts.get(3, 0):5.1f}% ({stats.get('action_counts_total', {}).get(3, 0):4d})
"""
        self.action_dist_display.setText(action_text)
    
    def update_episode_stats(self, stats):
        """Update current episode statistics (LEFT COLUMN)"""
        text = f"""Step: {stats.get('step', 0)}
Episode Return: {stats.get('episode_return', 0):.2f}
Last Action: {stats.get('last_action', 'N/A')}
Last Reward: {stats.get('last_reward', 0):.4f}
Budget: {stats.get('budget', 0):.2f}
Predicted Pop: {stats.get('prediction', 0):.1f}
"""
        self.episode_display.setText(text)
        
        # Update episode action distribution (RIGHT COLUMN)
        episode_actions = sum(stats.get('action_counts_episode', {}).values())
        episode_pcts = {}
        if episode_actions > 0:
            for action, count in stats.get('action_counts_episode', {}).items():
                episode_pcts[action] = (count / episode_actions) * 100
        
        episode_action_text = f"""Total Actions: {episode_actions}

NOOP:     {episode_pcts.get(0, 0):5.1f}% ({stats.get('action_counts_episode', {}).get(0, 0):4d})
COUNT:    {episode_pcts.get(1, 0):5.1f}% ({stats.get('action_counts_episode', {}).get(1, 0):4d})
SEQUENCE: {episode_pcts.get(2, 0):5.1f}% ({stats.get('action_counts_episode', {}).get(2, 0):4d})
DOSE:     {episode_pcts.get(3, 0):5.1f}% ({stats.get('action_counts_episode', {}).get(3, 0):4d})
"""
        self.episode_action_display.setText(episode_action_text)
    
    def update_current_budget_graph(self, step: int, budget: float):
        """Update the current episode budget graph"""
        self.current_episode_budget_steps.append(step)
        self.current_episode_budget_values.append(budget)
        
        # Update plot
        self.current_budget_ax.clear()
        self.current_budget_ax.plot(self.current_episode_budget_steps, 
                                    self.current_episode_budget_values, 
                                    'b-', linewidth=1.5)
        self.current_budget_ax.set_xlabel('Step')
        self.current_budget_ax.set_ylabel('Budget')
        self.current_budget_ax.set_title('Budget Evolution in Current Episode')
        self.current_budget_ax.set_xlim(0, self.max_steps)  # Fixed x-axis
        self.current_budget_ax.set_ylim(0, self.initial_budget)  # Fixed y-axis
        self.current_budget_ax.grid(True, alpha=0.3)
        self.current_budget_fig.tight_layout()
        self.current_budget_canvas.draw()
    
    def reset_current_episode_budget(self):
        """Reset current episode budget tracking for new episode"""
        self.current_episode_budget_steps.clear()
        self.current_episode_budget_values.clear()
        
        # Clear the plot
        self.current_budget_ax.clear()
        self.current_budget_ax.set_xlabel('Step')
        self.current_budget_ax.set_ylabel('Budget')
        self.current_budget_ax.set_title('Budget Evolution in Current Episode')
        self.current_budget_ax.set_xlim(0, self.max_steps)  # Fixed x-axis
        self.current_budget_ax.set_ylim(0, self.initial_budget)  # Fixed y-axis
        self.current_budget_ax.grid(True, alpha=0.3)
        self.current_budget_fig.tight_layout()
        self.current_budget_canvas.draw()
    
    def update_env_stats(self, stats):
        """Update environment statistics"""
        text = f"""Population: {stats.get('population', 0)}
Avg Energy: {stats.get('avg_energy', 0):.2f}
Sequencing Pending: {stats.get('seq_pending', False)}
Seq ETA: {stats.get('seq_eta', 0)} steps
"""
        self.env_display.setText(text)


class TrainingVisualizer:
    """Main training visualizer that orchestrates training and visualization"""
    
    def __init__(self, config: CompleteConfig, ppo_cfg: PPOConfig, env: PetriEnvWrapper, 
                 save_dir: Path, logger: TrainingLogger, viz_interval: int = 50, 
                 steps_per_frame: int = 5, enable_tracking: bool = False):
        self.config = config
        self.ppo_cfg = ppo_cfg
        self.env = env
        self.save_dir = save_dir
        self.logger = logger
        self.paused = True  # Start paused so plots can load
        self.training_started = False  # Track if training has ever started
        self.enable_tracking = enable_tracking
        
        # Visualization control
        self.viz_interval = viz_interval
        self.steps_per_frame = steps_per_frame
        
        # Training state (matching normal training)
        self.current_update = 0
        self.total_updates = config.training.total_updates
        self.log_data = []
        self.reward_history = []
        self.loss_history = []
        self.start_time = time.time()
        
        # Episode state
        self.current_obs = None
        self.current_step = 0
        self.last_action = None
        self.last_reward = 0.0
        self.last_prediction = 0.0  # Track last population prediction (not normalized)
        self.episodes_completed = 0
        self.last_train_stats = {'loss_actor': 0.0, 'loss_critic': 0.0}
        
        # Action tracking - overall
        self.action_counts_total = {
            ACTION_NOOP: 0,
            ACTION_COUNT_BACTERIA: 0,
            ACTION_SEQUENCING: 0,
            ACTION_DOSE: 0
        }
        
        # Action tracking - current episode
        self.action_counts_episode = {
            ACTION_NOOP: 0,
            ACTION_COUNT_BACTERIA: 0,
            ACTION_SEQUENCING: 0,
            ACTION_DOSE: 0
        }
        
        # Reward component tracking for rollout
        self.rollout_reward_components = {
            'immediate': [],
            'action_cost_penalty': [],
            'maintenance': [],
            'budget_penalty': [],
            'unaffordable_action_penalty': [],
            'delayed': [],
            'survival_bonus': [],
            'budget_conservation': [],
            'regular_count_bonus': [],
            'safe_behavior_bonus': [],
            'informed_dosing_bonus': [],
            'count_population': [],
            'critical_inaction_penalty': [],
            'critical_noop_penalty': [],
            'prediction': [],
            'pred_error': [],  # Diagnostic only
        }
        
        # Current episode reward component accumulators
        self.current_episode_rewards = {
            'immediate': 0.0,
            'action_cost_penalty': 0.0,
            'maintenance': 0.0,
            'budget_penalty': 0.0,
            'unaffordable_action_penalty': 0.0,
            'delayed': 0.0,
            'survival_bonus': 0.0,
            'budget_conservation': 0.0,
            'regular_count_bonus': 0.0,
            'safe_behavior_bonus': 0.0,
            'informed_dosing_bonus': 0.0,
            'count_population': 0.0,
            'critical_inaction_penalty': 0.0,
            'critical_noop_penalty': 0.0,
            'prediction': 0.0,
            'pred_error': 0.0,  # Diagnostic only
        }
        
        # Budget tracking for rollout metrics
        self.rollout_budget_spent = []
        self.rollout_budget_remaining = []
        self.rollout_budget_per_step = []
        self.rollout_episode_returns = []
        self.rollout_episode_lengths = []
        
        # Initialize agent (reusing normal training function)
        self.agent = _initialize_agent(ppo_cfg, self.env)
        _log_training_start(ppo_cfg, self.total_updates, logger)
        
        # IMPORTANT: Reset environment to initialize the model BEFORE setting up visualization
        self.logger.log_info("Initializing environment...")
        self.current_obs = self.env.reset()
        self.agent.start_episode()
        
        # Now env.model exists and we can set up tracking and visualization
        # Individual tracking setup
        if self.enable_tracking:
            from simulation.tracking import IndividualPlotter
            self.individual_plotter = IndividualPlotter(
                self.env.model.individual_tracker,
                on_close_callback=self.on_individual_window_close
            )
            on_click = self.on_bacterium_click
            self.logger.log_info("Individual bacteria tracking enabled")
        else:
            self.individual_plotter = None
            on_click = None
            self.logger.log_info("Individual bacteria tracking disabled")
        
        # Setup visualization - get model from environment
        self.visualizer = SimulationVisualizer(model=self.env.model, on_click_callback=on_click)
        self.viz_window = VisualizationWindow(self.visualizer)
        
        # Setup control panel with initial budget and max steps from environment
        self.control_panel = TrainingControlPanel(
            on_pause_toggle=self.toggle_pause,
            initial_budget=self.env.budget_init,
            max_steps=self.env.max_steps
        )
        
        # Position windows
        self._arrange_windows()
        
        # Animation
        self.animation = None
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        self.steps_in_current_rollout = 0
    
    def _arrange_windows(self):
        """Arrange windows side by side"""
        self.control_panel.move(0, 0)
        self.control_panel.resize(800, 1400)
        self.viz_window.move(820, 0)
    
    def toggle_pause(self):
        """Toggle training pause"""
        self.paused = not self.paused
        if not self.training_started:
            self.training_started = True
        status = "Paused" if self.paused else "Running"
        self.control_panel.pause_button.setText(f"{'Resume' if self.paused else 'Pause'} Training")
        self.logger.log_info(f"Training {status}")
    
    def run(self):
        """Start training with visualization"""
        # Reset environment
        self.current_obs = self.env.reset()
        self.agent.start_episode()
        self.current_step = 0
        
        # Create animation
        self.animation = animation.FuncAnimation(
            self.visualizer.fig,
            self.update_frame,
            interval=self.viz_interval,
            blit=False,
            cache_frame_data=False
        )
        
        # Show windows
        self.control_panel.show()
        self.viz_window.show()
        
        self.logger.log_info("Training visualization started")
    
    def update_frame(self, frame):
        """Animation update - performs training steps and updates visualization"""
        try:
            if self.paused:
                return
            
            # Check if training is complete
            if self.current_update >= self.total_updates:
                self._finalize_and_exit()
                return
            
            # Perform training steps
            for _ in range(self.steps_per_frame):
                if self.paused or self.current_update >= self.total_updates:
                    break
                
                self._training_step()
            
            # Update visualization
            self.visualizer.update_main_plot()
            self.visualizer.update_graphs()
            self.visualizer.draw()
            
            # Update individual tracking plot if a bacterium is selected and tracking is enabled
            if self.enable_tracking and self.individual_plotter and self.individual_plotter.current_id is not None:
                # Check if the window is still open
                if self.individual_plotter.is_window_open():
                    self.individual_plotter.update_plots(self.individual_plotter.current_id)
            
            # Update control panel
            self._update_control_panel()
            
        except Exception as e:
            self.logger.log_error(f"Error in update: {e}")
            import traceback
            traceback.print_exc()
    
    def _training_step(self):
        """Execute one training step"""
        # Check if we need to perform policy update
        if self.steps_in_current_rollout >= self.ppo_cfg.rollout_steps:
            self._update_policy()
            self.steps_in_current_rollout = 0
            self.buffer = RolloutBuffer()
        
        # Collect one step of experience
        obs_tensor = torch.from_numpy(self.current_obs).unsqueeze(0).to(self.agent.device)
        
        # # Log NN inputs for first 1 steps with labeled values
        # if self.current_step < 1:
        #     # Observation structure from env_wrapper._build_observation():
        #     # [budget_norm, target_norm, last_count_norm, *avg_genome(12 values), *props(3 values), 
        #     #  time_since_last_measure_norm, seq_pending_flag, seq_eta_norm]
        #     print(f"\n[STEP {self.current_step}] NN Input Observation (shape={self.current_obs.shape}):")
        #     idx = 0
        #     print(f"  [{idx}] budget_norm:                {self.current_obs[idx]:.4f}")
        #     idx += 1
        #     print(f"  [{idx}] target_population_norm:     {self.current_obs[idx]:.4f}")
        #     idx += 1
        #     print(f"  [{idx}] last_count_norm:            {self.current_obs[idx]:.4f}")
        #     idx += 1
            
        #     # avg_genome: 3 bacteria types × 4 traits = 12 values
        #     print(f"  Genome averages (3 types × 4 traits):")
        #     for bac_type in range(3):
        #         print(f"    Type {bac_type}: ", end="")
        #         for trait_idx, trait_name in enumerate(['enzyme', 'efflux', 'repair', 'membrane']):
        #             print(f"{trait_name}={self.current_obs[idx]:.4f} ", end="")
        #             idx += 1
        #         print()
            
        #     # proportions: 3 antibiotic types
        #     print(f"  Proportions (3 antibiotics):")
        #     for ab_idx in range(3):
        #         print(f"    [{idx}] antibiotic_{ab_idx}_prop:      {self.current_obs[idx]:.4f}")
        #         idx += 1
            
        #     # Meta information
        #     print(f"  [{idx}] time_since_last_measure:   {self.current_obs[idx]:.4f}")
        #     idx += 1
        #     print(f"  [{idx}] seq_pending_flag:          {self.current_obs[idx]:.4f}")
        #     idx += 1
        #     print(f"  [{idx}] seq_eta_norm:              {self.current_obs[idx]:.4f}")
        #     print(f"  Summary: Min={self.current_obs.min():.4f}, Max={self.current_obs.max():.4f}, Mean={self.current_obs.mean():.4f}")
        
        with torch.no_grad():
            (
                a_disc,
                a_cont,
                logp_disc,
                logp_cont,
                value,
                pred_next_pop,
                h_prev,
                action_mask
            ) = self.agent.select_action(self.current_obs)
        
        pure_a_disc = a_disc.cpu().numpy()[0]
        pure_a_cont = a_cont.cpu().numpy()[0]
        
        # Store action for display
        self.last_action = pure_a_disc
        
        # Track action counts
        self.action_counts_total[pure_a_disc] += 1
        self.action_counts_episode[pure_a_disc] += 1
        
        # Get prediction value for passing to environment
        pred_next_pop_value = pred_next_pop.cpu().item()
        
        # Store denormalized prediction for display
        self.last_prediction = pred_next_pop_value * self.env.population_norm
        
        # Step environment (now includes prediction reward computation)
        next_obs, reward, done, info = self.env.step(pure_a_disc, pure_a_cont, pred_population=pred_next_pop_value)
        self.last_reward = reward
        
        # Extract prediction supervision and diagnostics
        population_counted_norm = info.get('population_next_norm', 0.0)
        count_was_performed = info.get('count_was_performed', False)
        count_mask_value = 1.0 if count_was_performed else 0.0
        
        # Track prediction error for diagnostics (separate from reward which is in info)
        if count_was_performed:
            pred_error = abs(pred_next_pop_value - population_counted_norm)
            self.current_episode_rewards['pred_error'] += pred_error
            # Note: prediction reward is now computed by environment and included in total reward
        
        # Accumulate reward components for current episode
        self.current_episode_rewards['immediate'] += info.get('reward_immediate', 0.0)
        self.current_episode_rewards['action_cost_penalty'] += info.get('reward_action_cost_penalty', 0.0)
        self.current_episode_rewards['maintenance'] += info.get('reward_maintenance', 0.0)
        self.current_episode_rewards['budget_penalty'] += info.get('reward_budget_penalty', 0.0)
        self.current_episode_rewards['unaffordable_action_penalty'] += info.get('reward_unaffordable_action_penalty', 0.0)
        self.current_episode_rewards['delayed'] += info.get('reward_delayed', 0.0)
        self.current_episode_rewards['survival_bonus'] += info.get('reward_survival_bonus', 0.0)
        self.current_episode_rewards['budget_conservation'] += info.get('reward_budget_conservation', 0.0)
        self.current_episode_rewards['regular_count_bonus'] += info.get('reward_regular_count_bonus', 0.0)
        self.current_episode_rewards['safe_behavior_bonus'] += info.get('reward_safe_behavior_bonus', 0.0)
        self.current_episode_rewards['informed_dosing_bonus'] += info.get('reward_informed_dosing_bonus', 0.0)
        self.current_episode_rewards['count_population'] += info.get('reward_count_population', 0.0)
        self.current_episode_rewards['critical_inaction_penalty'] += info.get('reward_critical_inaction_penalty', 0.0)
        self.current_episode_rewards['critical_noop_penalty'] += info.get('reward_critical_noop_penalty', 0.0)
        self.current_episode_rewards['prediction'] += info.get('reward_prediction', 0.0)
        
        # Store in buffer
        self.buffer.add(
            obs=obs_tensor.cpu(),
            a_disc=a_disc.cpu(),
            a_cont=a_cont.cpu(),
            logp_disc=logp_disc.cpu(),
            logp_cont=logp_cont.cpu(),
            value=value.cpu(),
            reward=torch.tensor([reward], dtype=torch.float32),
            done=torch.tensor([float(done)], dtype=torch.float32),
            h_in=h_prev.cpu(),
            pred_next_pop=pred_next_pop.cpu(),
            population_counted_norm=torch.tensor([population_counted_norm], dtype=torch.float32),
            count_mask=torch.tensor([count_mask_value], dtype=torch.float32),
            action_mask=action_mask.cpu(),
        )
        
        self.current_obs = next_obs
        self.current_step += 1
        self.steps_in_current_rollout += 1
        
        # Handle episode termination
        if done:
            # ⚠️ IMPORTANT: Get budget metrics BEFORE resetting environment!
            budget_metrics = self.env.get_episode_budget_metrics()
            
            # Store budget metrics for rollout
            self.rollout_budget_spent.append(budget_metrics['budget_spent'])
            self.rollout_budget_remaining.append(budget_metrics['current_budget'])
            self.rollout_budget_per_step.append(budget_metrics['budget_per_step'])
            
            # Store reward components for completed episode (for rollout metrics)
            for key in self.rollout_reward_components.keys():
                self.rollout_reward_components[key].append(self.current_episode_rewards[key])
            self.rollout_episode_returns.append(info['episode_return'])
            self.rollout_episode_lengths.append(info.get('t', self.current_step))
            
            # Extract reward components from accumulated values
            reward_components = {
                'immediate': self.current_episode_rewards['immediate'],
                'action_cost_penalty': self.current_episode_rewards['action_cost_penalty'],
                'maintenance': self.current_episode_rewards['maintenance'],
                'budget_penalty': self.current_episode_rewards['budget_penalty'],
                'unaffordable_action_penalty': self.current_episode_rewards['unaffordable_action_penalty'],
                'delayed': self.current_episode_rewards['delayed'],
                'survival_bonus': self.current_episode_rewards['survival_bonus'],
                'budget_conservation': self.current_episode_rewards['budget_conservation'],
                'regular_count_bonus': self.current_episode_rewards['regular_count_bonus'],
                'safe_behavior_bonus': self.current_episode_rewards['safe_behavior_bonus'],
                'informed_dosing_bonus': self.current_episode_rewards['informed_dosing_bonus'],
                'count_population': self.current_episode_rewards['count_population'],
                'critical_inaction_penalty': self.current_episode_rewards['critical_inaction_penalty'],
                'critical_noop_penalty': self.current_episode_rewards['critical_noop_penalty'],
                'prediction': self.current_episode_rewards['prediction'],
                'pred_error': self.current_episode_rewards['pred_error'],
            }
            
            # Reset episode reward accumulators
            for key in self.current_episode_rewards.keys():
                self.current_episode_rewards[key] = 0.0
            
            self.episodes_completed += 1
            self.current_obs = self.env.reset()
            self.agent.start_episode()
            self.current_step = 0
            # Reset episode action counts
            self.action_counts_episode = {
                ACTION_NOOP: 0,
                ACTION_COUNT_BACTERIA: 0,
                ACTION_SEQUENCING: 0,
                ACTION_DOSE: 0
            }
            # Update visualizer model reference
            self.visualizer.model = self.env.model
            # Update individual plotter's tracker reference if tracking is enabled
            if self.enable_tracking and self.individual_plotter:
                self.individual_plotter.tracker = self.env.model.individual_tracker
                self.logger.log_debug("Updated individual tracker reference for new episode")
            self.logger.log_debug(f"Episode {self.episodes_completed} complete, return: {info['episode_return']:.2f}")
            
            # Update control panel with episode data and reward components
            self.control_panel.add_episode_data(
                episode_num=self.episodes_completed,
                episode_length=info.get('t', self.current_step),
                budget_spent=budget_metrics['budget_spent'],
                budget_remaining=budget_metrics['current_budget'],
                reward_components=reward_components
            )
            
            # Reset current episode budget graph for new episode
            self.control_panel.reset_current_episode_budget()
    
    def _update_policy(self):
        """Update the policy using collected rollout data (matching normal training)"""
        if len(self.buffer.obs) == 0:
            return
        
        # Compute rollout metrics for logging (add all expected keys)
        total_actions = sum(self.action_counts_total.values())
        dose_action_percentage = (self.action_counts_total[ACTION_DOSE] / total_actions * 100) if total_actions > 0 else 0.0
        count_action_percentage = (self.action_counts_total[ACTION_COUNT_BACTERIA] / total_actions * 100) if total_actions > 0 else 0.0
        sequencing_action_percentage = (self.action_counts_total[ACTION_SEQUENCING] / total_actions * 100) if total_actions > 0 else 0.0
        noop_action_percentage = (self.action_counts_total[ACTION_NOOP] / total_actions * 100) if total_actions > 0 else 0.0
        
        rollout_metrics = {
            "mean_episode_reward": float(np.mean(self.rollout_episode_returns)) if self.rollout_episode_returns else 0.0,
            "std_episode_reward": float(np.std(self.rollout_episode_returns)) if self.rollout_episode_returns else 0.0,
            "max_episode_reward": float(np.max(self.rollout_episode_returns)) if self.rollout_episode_returns else 0.0,
            "min_episode_reward": float(np.min(self.rollout_episode_returns)) if self.rollout_episode_returns else 0.0,
            "mean_episode_length": float(np.mean(self.rollout_episode_lengths)) if self.rollout_episode_lengths else 0.0,
            "std_episode_length": float(np.std(self.rollout_episode_lengths)) if self.rollout_episode_lengths else 0.0,
            "min_episode_length": float(np.min(self.rollout_episode_lengths)) if self.rollout_episode_lengths else 0.0,
            "max_episode_length": float(np.max(self.rollout_episode_lengths)) if self.rollout_episode_lengths else 0.0,
            "num_episodes": self.episodes_completed,
            "mean_population_per_episode": self.env.get_bacteria_population(),
            "final_population": self.env.get_bacteria_population(),
            "dose_action_percentage": float(dose_action_percentage),
            "count_action_percentage": float(count_action_percentage),
            "sequencing_action_percentage": float(sequencing_action_percentage),
            "noop_action_percentage": float(noop_action_percentage),
            # Budget metrics (matching rollout() function)
            "mean_budget_spent": float(np.mean(self.rollout_budget_spent)) if self.rollout_budget_spent else 0.0,
            "mean_budget_remaining": float(np.mean(self.rollout_budget_remaining)) if self.rollout_budget_remaining else 0.0,
            "mean_budget_per_step": float(np.mean(self.rollout_budget_per_step)) if self.rollout_budget_per_step else 0.0,
            # Add reward component metrics (matching rollout() function)
            "rewards/immediate": float(np.mean(self.rollout_reward_components['immediate'])) if self.rollout_reward_components['immediate'] else 0.0,
            "rewards/maintenance": float(np.mean(self.rollout_reward_components['maintenance'])) if self.rollout_reward_components['maintenance'] else 0.0,
            "rewards/budget_penalty": float(np.mean(self.rollout_reward_components['budget_penalty'])) if self.rollout_reward_components['budget_penalty'] else 0.0,
            "rewards/delayed": float(np.mean(self.rollout_reward_components['delayed'])) if self.rollout_reward_components['delayed'] else 0.0,
            "rewards/survival_bonus": float(np.mean(self.rollout_reward_components['survival_bonus'])) if self.rollout_reward_components['survival_bonus'] else 0.0,
            "rewards/budget_conservation": float(np.mean(self.rollout_reward_components['budget_conservation'])) if self.rollout_reward_components['budget_conservation'] else 0.0,
            "rewards/regular_count_bonus": float(np.mean(self.rollout_reward_components['regular_count_bonus'])) if self.rollout_reward_components['regular_count_bonus'] else 0.0,
            "rewards/safe_behavior_bonus": float(np.mean(self.rollout_reward_components['safe_behavior_bonus'])) if self.rollout_reward_components['safe_behavior_bonus'] else 0.0,
            "rewards/informed_dosing_bonus": float(np.mean(self.rollout_reward_components['informed_dosing_bonus'])) if self.rollout_reward_components['informed_dosing_bonus'] else 0.0,
            "rewards/count_population": float(np.mean(self.rollout_reward_components['count_population'])) if self.rollout_reward_components['count_population'] else 0.0,
            "rewards/critical_inaction_penalty": float(np.mean(self.rollout_reward_components['critical_inaction_penalty'])) if self.rollout_reward_components['critical_inaction_penalty'] else 0.0,
            "rewards/critical_noop_penalty": float(np.mean(self.rollout_reward_components['critical_noop_penalty'])) if self.rollout_reward_components['critical_noop_penalty'] else 0.0,
            "rewards/prediction": float(np.mean(self.rollout_reward_components['prediction'])) if self.rollout_reward_components['prediction'] else 0.0,
            "rewards/total": float(np.mean(self.rollout_episode_returns)) if self.rollout_episode_returns else 0.0,
            # Prediction metrics
            "prediction/error": float(np.mean(self.rollout_reward_components['pred_error'])) if self.rollout_reward_components['pred_error'] else 0.0,
        }
        
        # Clear reward components and budget tracking for next rollout
        for key in self.rollout_reward_components.keys():
            self.rollout_reward_components[key].clear()
        self.rollout_budget_spent.clear()
        self.rollout_budget_remaining.clear()
        self.rollout_budget_per_step.clear()
        self.rollout_episode_returns.clear()
        self.rollout_episode_lengths.clear()
        
        # Update policy
        train_stats = self.agent.update_policy(self.buffer)
        self.last_train_stats = train_stats
        
        self.current_update += 1
        
        # Log and track (matching normal training)
        log_entry = {
            "update": self.current_update,
            **rollout_metrics,
            **train_stats,
        }
        self.log_data.append(log_entry)
        self.reward_history.append(rollout_metrics['mean_episode_reward'])
        self.loss_history.append(train_stats['loss_actor'])
        self.logger.log_metrics(self.current_update - 1, rollout_metrics, train_stats)
        
        # Periodic reporting (matching normal training)
        if self.current_update % 10 == 0 and self.current_update > 0:
            elapsed = time.time() - self.start_time
            self.logger.log_update(self.current_update - 1, self.total_updates, rollout_metrics, train_stats, elapsed)
        
        # Check for divergence (matching normal training)
        if np.isnan(train_stats['loss_actor']):
            self.logger.log_error("Loss became NaN, stopping training")
            self._finalize_and_exit()
            return
        
        # Checkpointing (matching normal training - reuse function)
        _handle_checkpoint(self.agent, self.current_update - 1, self.total_updates, self.save_dir, self.logger)
    
    def _finalize_and_exit(self):
        """Finalize training and close application (matching normal training)"""
        if hasattr(self, '_finalized') and self._finalized:
            return
        self._finalized = True
        
        self.logger.log_info("="*70)
        self.logger.log_info("Finalizing training...")
        
        # Calculate total time
        total_time = time.time() - self.start_time
        
        # Finalize training (reuse function from normal training)
        _finalize_training(
            self.log_data,
            self.reward_history,
            self.loss_history,
            self.total_updates,
            total_time,
            self.save_dir,
            self.logger
        )
        
        self.logger.log_info("="*70)
        self.logger.log_info("✓ Training complete!")
        self.logger.log_info(f"Logs: {self.save_dir / 'training.log'}")
        self.logger.log_info(f"Metrics: {self.save_dir / 'training_log.json'}")
        self.logger.log_info(f"TensorBoard: tensorboard --logdir={self.save_dir / self.config.training.experiment_name} --port=6006")
        
        # Stop animation
        if self.animation:
            self.animation.event_source.stop()
        
        # Close windows after a short delay
        QtCore.QTimer.singleShot(2000, self._close_all_windows)
    
    def _close_all_windows(self):
        """Close all windows and exit"""
        self.viz_window.close()
        self.control_panel.close()
        QtWidgets.QApplication.quit()
    
    def _update_control_panel(self):
        """Update control panel displays"""
        elapsed = time.time() - self.start_time
        
        # Training stats
        train_stats = {
            'update': self.current_update,
            'total_updates': self.total_updates,
            'elapsed': elapsed,
            'mean_reward': self.env.episode_return,
            'actor_loss': self.last_train_stats.get('loss_actor', 0.0),
            'critic_loss': self.last_train_stats.get('loss_critic', 0.0),
            'episodes_completed': self.episodes_completed,
            'action_counts_total': self.action_counts_total
        }
        self.control_panel.update_training_stats(train_stats)
        
        # Episode stats
        episode_stats = {
            'step': self.current_step,
            'episode_return': self.env.episode_return,
            'last_action': ACTION_NAMES.get(self.last_action, 'N/A'),
            'last_reward': self.last_reward,
            'budget': self.env.budget,
            'prediction': self.last_prediction,
            'action_counts_episode': self.action_counts_episode
        }
        self.control_panel.update_episode_stats(episode_stats)
        
        # Update current episode budget graph
        self.control_panel.update_current_budget_graph(self.current_step, self.env.budget)
        
        # Environment stats
        pop_stats = self.env.model.get_population_stats()
        env_stats = {
            'population': pop_stats['total'],
            'avg_energy': pop_stats['avg_energy'],
            'seq_pending': self.env.seq_pending,
            'seq_eta': self.env.seq_eta,
        }
        self.control_panel.update_env_stats(env_stats)
    
    def on_bacterium_click(self, bacterium_id):
        """Handle bacterium click from visualizer"""
        if self.enable_tracking and self.individual_plotter:
            self.individual_plotter.update_plots(bacterium_id)
            self.visualizer.set_highlighted_bacterium(bacterium_id)
    
    def on_individual_window_close(self):
        """Handle individual tracking window being closed"""
        self.visualizer.clear_highlight()
        if self.enable_tracking:
            self.logger.log_debug("Individual tracking window closed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train RL Agent with Real-time Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use fast config for testing
  python train_with_visualization.py --config rl/configs/training_config_fast.yaml
  
  # Use default config
  python train_with_visualization.py --config rl/configs/training_config.yaml
  
  # Custom visualization settings
  python train_with_visualization.py --config rl/configs/training_config_fast.yaml --viz-interval 100 --steps-per-frame 10
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="training_config_fast.yaml",
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--viz-interval",
        type=int,
        default=50,
        help="Milliseconds between visualization updates (default: 50ms = ~20 FPS). Lower = smoother but slower training."
    )
    
    parser.add_argument(
        "--steps-per-frame",
        type=int,
        default=5,
        help="Number of training steps per frame update (default: 5). Higher = faster training but choppier visualization."
    )
    
    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        help="Enable individual bacteria tracking and click-to-view functionality. Note: This may slow down training."
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return 1
    
    # ========================================================================
    # SETUP (matching normal training)
    # ========================================================================
    save_dir = Path(config.training.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger (reuse from normal training)
    logger = _setup_logger_and_log_startup(save_dir, config)
    
    # Set seed (reuse from normal training)
    set_global_seed(config.training.seed)
    logger.log_debug(f"✓ Random seed set to: {config.training.seed}")
    
    # Create environment (reuse from normal training)
    env = _create_environment(config, logger)
    logger.log_info(f"Observation dimension: {env.get_obs_dim()}")
    
    # Build PPO configuration (reuse from normal training)
    ppo_config = _build_ppo_config(env, config)
    
    # Save all configurations (reuse from normal training)
    _save_configs(save_dir, config, logger)
    
    # Create QApplication
    app = QtWidgets.QApplication.instance()
    if (app is None):
        app = QtWidgets.QApplication(sys.argv)
    
    # Create and run training visualizer
    print("="*70)
    print("Training with Real-time Visualization")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Total updates: {config.training.total_updates}")
    print(f"Save directory: {config.training.save_dir}")
    print(f"Visualization interval: {args.viz_interval}ms (~{1000//args.viz_interval} FPS)")
    print(f"Steps per frame: {args.steps_per_frame}")
    print(f"Individual tracking: {'Enabled' if args.enable_tracking else 'Disabled'}")
    print("="*70)
    print("\nControls:")
    print("  - Click 'Pause Training' to pause/resume")
    if args.enable_tracking:
        print("  - Click on a bacterium to view its individual tracking data")
    print("  - Watch the agent learn in real-time!")
    print("="*70)
    
    trainer = TrainingVisualizer(
        config, ppo_config, env, save_dir, logger,
        viz_interval=args.viz_interval,
        steps_per_frame=args.steps_per_frame,
        enable_tracking=args.enable_tracking
    )
    trainer.run()
    
    # Run event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
