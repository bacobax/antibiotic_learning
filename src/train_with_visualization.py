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
    
    def __init__(self, on_pause_toggle):
        super().__init__()
        self.setWindowTitle("Training Control Panel")
        self.setGeometry(0, 0, 400, 1200)  # Increased height for graphs
        
        self.on_pause_toggle = on_pause_toggle
        
        # Data for plots
        self.episode_numbers = []
        self.episode_lengths = []
        self.budget_spent_history = []
        self.budget_remaining_history = []
        
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
        
        # Training statistics
        train_group = QtWidgets.QGroupBox("Training Statistics")
        train_layout = QtWidgets.QVBoxLayout()
        
        self.train_display = QtWidgets.QTextEdit()
        self.train_display.setReadOnly(True)
        self.train_display.setMaximumHeight(200)
        train_layout.addWidget(self.train_display)
        
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        
        # Episode statistics
        episode_group = QtWidgets.QGroupBox("Current Episode")
        episode_layout = QtWidgets.QVBoxLayout()
        
        self.episode_display = QtWidgets.QTextEdit()
        self.episode_display.setReadOnly(True)
        self.episode_display.setMaximumHeight(200)
        episode_layout.addWidget(self.episode_display)
        
        episode_group.setLayout(episode_layout)
        layout.addWidget(episode_group)
        
        # ===== NEW: Episode Length Graph =====
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        
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
        layout.addWidget(episode_length_group)
        
        # ===== NEW: Budget Tracking Graph =====
        budget_group = QtWidgets.QGroupBox("Budget Usage Over Episodes")
        budget_layout = QtWidgets.QVBoxLayout()
        
        self.budget_fig = Figure(figsize=(4, 2), dpi=100)
        self.budget_ax = self.budget_fig.add_subplot(111)
        self.budget_canvas = FigureCanvas(self.budget_fig)
        self.budget_canvas.setMaximumHeight(200)
        
        self.budget_ax.set_xlabel('Episode')
        self.budget_ax.set_ylabel('Budget')
        self.budget_ax.set_title('Budget Tracking')
        self.budget_ax.grid(True, alpha=0.3)
        
        budget_layout.addWidget(self.budget_canvas)
        budget_group.setLayout(budget_layout)
        layout.addWidget(budget_group)
        
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
    
    def add_episode_data(self, episode_num: int, episode_length: int, budget_spent: float, budget_remaining: float):
        """Add data point for completed episode"""
        self.episode_numbers.append(episode_num)
        self.episode_lengths.append(episode_length)
        self.budget_spent_history.append(budget_spent)
        self.budget_remaining_history.append(budget_remaining)
        
        # Update episode length plot
        self.episode_length_ax.clear()
        self.episode_length_ax.plot(self.episode_numbers, self.episode_lengths, 'b-', linewidth=1)
        self.episode_length_ax.set_xlabel('Episode')
        self.episode_length_ax.set_ylabel('Steps')
        self.episode_length_ax.set_title('Episode Length')
        self.episode_length_ax.grid(True, alpha=0.3)
        self.episode_length_fig.tight_layout()
        self.episode_length_canvas.draw()
        
        # Update budget plot
        self.budget_ax.clear()
        self.budget_ax.plot(self.episode_numbers, self.budget_spent_history, 'r-', 
                           linewidth=1, label='Spent')
        self.budget_ax.plot(self.episode_numbers, self.budget_remaining_history, 'g-', 
                           linewidth=1, label='Remaining')
        self.budget_ax.set_xlabel('Episode')
        self.budget_ax.set_ylabel('Budget')
        self.budget_ax.set_title('Budget Tracking')
        self.budget_ax.legend(loc='best', fontsize=8)
        self.budget_ax.grid(True, alpha=0.3)
        self.budget_fig.tight_layout()
        self.budget_canvas.draw()
    
    def update_training_stats(self, stats):
        """Update training statistics display"""
        # Calculate action percentages overall
        total_actions = sum(stats.get('action_counts_total', {}).values())
        action_pcts = {}
        if total_actions > 0:
            for action, count in stats.get('action_counts_total', {}).items():
                action_pcts[action] = (count / total_actions) * 100
        
        text = f"""Update: {stats.get('update', 0)}/{stats.get('total_updates', 0)}
Elapsed Time: {stats.get('elapsed', 0):.1f}s
Mean Episode Reward: {stats.get('mean_reward', 0):.2f}
Actor Loss: {stats.get('actor_loss', 0):.4f}
Critic Loss: {stats.get('critic_loss', 0):.4f}
Episodes Completed: {stats.get('episodes_completed', 0)}

--- Overall Action Distribution ---
Total Actions: {total_actions}
NOOP:     {action_pcts.get(0, 0):5.1f}% ({stats.get('action_counts_total', {}).get(0, 0):4d})
COUNT:    {action_pcts.get(1, 0):5.1f}% ({stats.get('action_counts_total', {}).get(1, 0):4d})
SEQUENCE: {action_pcts.get(2, 0):5.1f}% ({stats.get('action_counts_total', {}).get(2, 0):4d})
DOSE:     {action_pcts.get(3, 0):5.1f}% ({stats.get('action_counts_total', {}).get(3, 0):4d})
"""
        self.train_display.setText(text)
    
    def update_episode_stats(self, stats):
        """Update current episode statistics"""
        # Calculate action percentages for current episode
        episode_actions = sum(stats.get('action_counts_episode', {}).values())
        episode_pcts = {}
        if episode_actions > 0:
            for action, count in stats.get('action_counts_episode', {}).items():
                episode_pcts[action] = (count / episode_actions) * 100
        
        text = f"""Step: {stats.get('step', 0)}
Episode Return: {stats.get('episode_return', 0):.2f}
Last Action: {stats.get('last_action', 'N/A')}
Last Reward: {stats.get('last_reward', 0):.4f}
Budget: {stats.get('budget', 0):.2f}

--- Episode Action Distribution ---
Total Actions: {episode_actions}
NOOP:     {episode_pcts.get(0, 0):5.1f}% ({stats.get('action_counts_episode', {}).get(0, 0):4d})
COUNT:    {episode_pcts.get(1, 0):5.1f}% ({stats.get('action_counts_episode', {}).get(1, 0):4d})
SEQUENCE: {episode_pcts.get(2, 0):5.1f}% ({stats.get('action_counts_episode', {}).get(2, 0):4d})
DOSE:     {episode_pcts.get(3, 0):5.1f}% ({stats.get('action_counts_episode', {}).get(3, 0):4d})
"""
        self.episode_display.setText(text)
    
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
        
        # Initialize agent (reusing normal training function)
        self.agent = _initialize_agent(ppo_cfg)
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
        
        # Setup control panel
        self.control_panel = TrainingControlPanel(on_pause_toggle=self.toggle_pause)
        
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
        self.control_panel.resize(400, 1200)
        self.viz_window.move(420, 0)
    
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
        
        with torch.no_grad():
            (
                a_disc,
                a_cont,
                logp_disc,
                logp_cont,
                value,
                h_prev
            ) = self.agent.select_action(self.current_obs)
        
        pure_a_disc = a_disc.cpu().numpy()[0]
        pure_a_cont = a_cont.cpu().numpy()[0]
        
        # Store action for display
        self.last_action = pure_a_disc
        
        # Track action counts
        self.action_counts_total[pure_a_disc] += 1
        self.action_counts_episode[pure_a_disc] += 1
        
        # Step environment
        next_obs, reward, done, info = self.env.step(pure_a_disc, pure_a_cont)
        self.last_reward = reward
        
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
        )
        
        self.current_obs = next_obs
        self.current_step += 1
        self.steps_in_current_rollout += 1
        
        # Handle episode termination
        if done:
            # ⚠️ IMPORTANT: Get budget metrics BEFORE resetting environment!
            budget_metrics = self.env.get_episode_budget_metrics()
            
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
            
            # Update control panel with episode data
            self.control_panel.add_episode_data(
                episode_num=self.episodes_completed,
                episode_length=info.get('t', self.current_step),
                budget_spent=budget_metrics['budget_spent'],
                budget_remaining=budget_metrics['current_budget']
            )
    
    def _update_policy(self):
        """Update the policy using collected rollout data (matching normal training)"""
        if len(self.buffer.obs) == 0:
            return
        
        # Compute rollout metrics for logging
        rollout_metrics = {
            "mean_episode_reward": self.env.episode_return,
            "num_episodes": self.episodes_completed,
        }
        
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
            'action_counts_episode': self.action_counts_episode
        }
        self.control_panel.update_episode_stats(episode_stats)
        
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
