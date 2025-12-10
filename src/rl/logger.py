"""
Simple and clear logging for PPO training.
Logs to: Python logs, JSON metrics, and TensorBoard (if available).
"""

import json
import logging
from collections import deque
from pathlib import Path
from typing import Dict, Optional
import numpy as np


class TrainingLogger:
    """Simple logger for training metrics. Handles all logging destinations."""
    
    def __init__(
        self,
        log_dir: Path,
        experiment_name: str = "ppo_training",
        *,
        max_metrics_entries: Optional[int] = None,
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for all logs and TensorBoard events
            experiment_name: Name for TensorBoard folder
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Python logger (console + file)
        self.logger = self._setup_python_logger()
        
        # TensorBoard writer (optional)
        self.tb_writer = self._setup_tensorboard(experiment_name)
        
        # JSON metrics file
        self.metrics_file = self.log_dir / "metrics.json"
        self.max_metrics_entries = None
        if max_metrics_entries is not None and max_metrics_entries > 0:
            self.max_metrics_entries = int(max_metrics_entries)
        self.metrics_data = (
            deque(maxlen=self.max_metrics_entries) if self.max_metrics_entries else []
        )
        
        # Track best reward
        self.best_reward = float('-inf')
        
        self.logger.info(f"Logging to: {self.log_dir}")
        if self.tb_writer:
            tb_log_dir = Path(self.tb_writer.log_dir)
            self.logger.info(f"TensorBoard logs: {tb_log_dir}")
            self.logger.info(f"View with: tensorboard --logdir={tb_log_dir.parent}")
    
    def _setup_python_logger(self) -> logging.Logger:
        """Setup Python logger with console and file output."""
        logger = logging.getLogger("PPO_Training")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        
        # File handler
        fh = logging.FileHandler(self.log_dir / "training.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(ch)
        
        return logger
    
    def _setup_tensorboard(self, experiment_name: str):
        """Setup TensorBoard writer if available."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            from datetime import datetime
            
            # Create timestamped directory for this run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tb_dir = self.log_dir / experiment_name / timestamp
            tb_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(str(tb_dir))
            return writer
        except ImportError:
            self.logger.warning("TensorBoard not installed. Skipping TensorBoard logging.")
            return None
        except Exception as e:
            self.logger.warning(f"TensorBoard initialization failed: {e}")
            return None
    
    def _flatten_config(self, obj, prefix: str = "") -> Dict[str, any]:
        """
        Recursively flatten a nested config object (dataclass or dict) into a flat dict.
        
        Args:
            obj: Object to flatten (dataclass, dict, or primitive)
            prefix: Key prefix for nested keys
        
        Returns:
            Flattened dictionary with dot-separated keys
        """
        from dataclasses import is_dataclass, asdict
        
        flat = {}
        
        if is_dataclass(obj) and not isinstance(obj, type):
            obj = asdict(obj)
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}.{key}" if prefix else key
                flat.update(self._flatten_config(value, new_key))
        elif isinstance(obj, (list, tuple)):
            # Convert lists/tuples to string representation for hparams
            flat[prefix] = str(obj)
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            # Convert None to string for TensorBoard compatibility
            flat[prefix] = obj if obj is not None else "None"
        else:
            # For other types, convert to string
            flat[prefix] = str(obj)
        
        return flat
    
    def log_hparams(self, config, final_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Log hyperparameters to TensorBoard.
        
        This logs all configuration parameters as hyperparameters, which can be viewed
        in TensorBoard's HPARAMS tab for comparing different runs.
        
        Args:
            config: CompleteConfig object or dict containing all hyperparameters
            final_metrics: Optional dict of final metrics to associate with hparams
        """
        if not self.tb_writer:
            self.logger.debug("TensorBoard not available, skipping hparam logging.")
            return
        
        try:
            # Flatten the config to a single-level dict
            hparam_dict = self._flatten_config(config)
            
            # Filter out any values that TensorBoard can't handle
            filtered_hparams = {}
            for key, value in hparam_dict.items():
                if isinstance(value, (int, float, str, bool)):
                    filtered_hparams[key] = value
                else:
                    filtered_hparams[key] = str(value)
            
            # Default metrics if not provided
            if final_metrics is None:
                final_metrics = {"hparam/placeholder": 0.0}
            
            # Log to TensorBoard
            self.tb_writer.add_hparams(filtered_hparams, final_metrics)
            self.logger.debug(f"✓ Logged {len(filtered_hparams)} hyperparameters to TensorBoard")
            
        except Exception as e:
            self.logger.warning(f"Failed to log hyperparameters to TensorBoard: {e}")
    
    # ========================================================================
    # Public Methods
    # ========================================================================
    
    def log_metrics(
        self,
        update: int,
        rollout_metrics: Dict,
        train_stats: Dict,
        extra_metrics: Optional[Dict] = None,
    ) -> None:
        """
        Log all metrics from an update.
        
        Args:
            update: Update number
            rollout_metrics: Dict with mean_episode_reward, std_episode_reward, num_episodes, etc.
            train_stats: Dict with loss_actor, loss_critic, entropy, clip_fraction, grad_norm, etc.
        """
        # Combine metrics
        all_metrics = {
            **rollout_metrics,
            **train_stats,
        }
        if extra_metrics:
            all_metrics.update(extra_metrics)
        
        # Track best reward
        if 'mean_episode_reward' in rollout_metrics:
            reward = rollout_metrics['mean_episode_reward']
            if reward > self.best_reward:
                self.best_reward = reward
                if self.tb_writer:
                    self.tb_writer.add_scalar("best_reward", self.best_reward, update)
        
        # Log to TensorBoard (without console prints)
        if self.tb_writer:
            for name, value in all_metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    self.tb_writer.add_scalar(name, value, update)
        
        # Log to JSON
        for name, value in all_metrics.items():
            if isinstance(value, (int, float)):
                entry = {
                    "update": update,
                    "metric": name,
                    "value": float(value),
                }
                if isinstance(self.metrics_data, deque):
                    self.metrics_data.append(entry)
                else:
                    self.metrics_data.append(entry)
        
        self._flush_json()

    def log_update_metrics(
        self,
        update: int,
        rollout_metrics: Dict,
        train_stats: Dict,
        extra_metrics: Optional[Dict] = None,
    ) -> None:
        """Backward-compatible alias for legacy scripts/tests."""
        self.log_metrics(update, rollout_metrics, train_stats, extra_metrics)
    
    def log_bacteria_population(self, update: int, population: int) -> None:
        """
        Log bacteria population to TensorBoard and JSON.
        
        Args:
            update: Update number (or episode number)
            population: Current bacteria population count
        """
        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar("bacteria/population", population, update)
        
        # Log to JSON
        self.metrics_data.append({
            "update": update,
            "metric": "bacteria_population",
            "value": float(population),
        })
        
        self._flush_json()
    
    def log_update(self, update: int, total_updates: int, rollout_metrics: Dict, 
                   train_stats: Dict, elapsed_time: float) -> None:
        """
        Log periodic update (every 10 updates) to console and files.
        
        Args:
            update: Current update number
            total_updates: Total updates for training
            rollout_metrics: Rollout metrics dict
            train_stats: Training stats dict
            elapsed_time: Time elapsed so far in seconds
        """
        # Calculate speed/ETA
        updates_per_sec = (update + 1) / elapsed_time if elapsed_time > 0 else 0
        eta_sec = (total_updates - update - 1) / updates_per_sec if updates_per_sec > 0 else 0
        
        # Main update log
        self.logger.info(
            f"UPDATE {update:4d}/{total_updates} | "
            f"Reward: {rollout_metrics['mean_episode_reward']:7.2f} "
            f"(±{rollout_metrics['std_episode_reward']:5.2f}) | "
            f"Episodes: {rollout_metrics['num_episodes']:3d} | "
            f"Pop: {rollout_metrics.get('mean_population_per_episode', 0):.0f} | "
            f"Budget: {rollout_metrics.get('mean_budget_remaining', 0.0):.1f} | "
            f"Loss: A={train_stats['loss_actor']:.4f} C={train_stats['loss_critic']:.4f}"
        )
        
        # Action distribution
        self.logger.info(
            f"  Actions: "
            f"DOSE={rollout_metrics.get('dose_action_percentage', 0.0):5.1f}% | "
            f"COUNT={rollout_metrics.get('count_action_percentage', 0.0):5.1f}% | "
            f"SEQ={rollout_metrics.get('sequencing_action_percentage', 0.0):5.1f}% | "
            f"NOOP={rollout_metrics.get('noop_action_percentage', 0.0):5.1f}%"
        )
        
        # Reward component breakdown (new simplified components)
        total_reward = (
            rollout_metrics.get('rewards/pre', 0.0) +
            rollout_metrics.get('rewards/post_penalties', 0.0) +
            rollout_metrics.get('rewards/kernel_maintenance', 0.0) +
            rollout_metrics.get('rewards/survival_bonus', 0.0) +
            rollout_metrics.get('rewards/prediction', 0.0) +
            rollout_metrics.get('rewards/early_termination_penalty', 0.0) +
            rollout_metrics.get('rewards/cost_penalty', 0.0)
        )
        
        self.logger.info(
            f"  Rewards: "
            f"Pre={rollout_metrics.get('rewards/pre', 0.0):+6.2f} | "
            f"PostPen={rollout_metrics.get('rewards/post_penalties', 0.0):+6.2f} | "
            f"Kernel={rollout_metrics.get('rewards/kernel_maintenance', 0.0):+6.2f} | "
            f"Survival={rollout_metrics.get('rewards/survival_bonus', 0.0):+6.2f} | "
            f"Predict={rollout_metrics.get('rewards/prediction', 0.0):+6.2f} | "
            f"EarlyTerm={rollout_metrics.get('rewards/early_termination_penalty', 0.0):+6.2f} | "
            f"Cost={rollout_metrics.get('rewards/cost_penalty', 0.0):+6.2f}"
        )
        self.logger.info(
            f"           "
            f"TOTAL (calculated)={total_reward:+7.2f} | "
            f"TOTAL (reported)={rollout_metrics.get('rewards/total', 0.0):+7.2f}"
        )
        
        # Debug info
        self.logger.debug(
            f"  Entropy: {train_stats['entropy']:.4f} | "
            f"Clip Frac: {train_stats['clip_fraction']:.3f} | "
            f"Grad Norm: {train_stats['grad_norm']:.4f} | "
            f"ETA: {eta_sec/60:.1f} min"
        )
        
        # Check for issues
        if np.isnan(train_stats['loss_actor']):
            self.logger.error(f"NaN detected in actor loss at update {update}!")
        
        if train_stats['clip_fraction'] > 0.5:
            self.logger.warning(
                f"High clipping fraction at update {update}: {train_stats['clip_fraction']:.3f}. "
                "Consider reducing learning rate."
            )
    
    def log_summary(self, total_updates: int, total_time: float, 
                   reward_history: list, loss_history: list) -> None:
        """
        Log final training summary.
        
        Args:
            total_updates: Total updates performed
            total_time: Total training time in seconds
            reward_history: List of mean rewards
            loss_history: List of actor losses
        """
        final_reward = reward_history[-1] if reward_history else 0.0
        final_loss = loss_history[-1] if loss_history else 0.0
        
        # Log to TensorBoard
        if self.tb_writer:
            self.tb_writer.add_scalar("summary/best_reward", self.best_reward, total_updates)
            self.tb_writer.add_scalar("summary/final_reward", final_reward, total_updates)
            self.tb_writer.add_scalar("summary/final_loss", final_loss, total_updates)
            self.tb_writer.add_scalar("summary/total_time_minutes", total_time / 60, total_updates)
            
            # Reward improvement
            if len(reward_history) > 20:
                early_avg = np.mean(reward_history[:10])
                late_avg = np.mean(reward_history[-10:])
                improvement = late_avg - early_avg
                self.tb_writer.add_scalar("summary/improvement", improvement, total_updates)
        
        # Print summary
        self.logger.info("="*70)
        self.logger.info("TRAINING SUMMARY")
        self.logger.info("="*70)
        self.logger.info(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        self.logger.info(f"Best reward: {self.best_reward:.4f}")
        self.logger.info(f"Final reward: {final_reward:.4f}")
        self.logger.info(f"Final loss: {final_loss:.4f}")
        
        if len(reward_history) > 20:
            early_avg = np.mean(reward_history[:10])
            late_avg = np.mean(reward_history[-10:])
            improvement = late_avg - early_avg
            self.logger.info(f"Reward improvement: {improvement:+.4f}")
        
        self.logger.info("="*70)
    
    def log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def log_debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    # ========================================================================
    # Private Methods
    # ========================================================================
    
    def _flush_json(self) -> None:
        """Save metrics to JSON file."""
        try:
            data = list(self.metrics_data) if isinstance(self.metrics_data, deque) else self.metrics_data
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to write metrics JSON: {e}")
    
    def close(self) -> None:
        """Close all logging backends."""
        if self.tb_writer:
            try:
                self.tb_writer.flush()
                self.tb_writer.close()
            except Exception as e:
                self.logger.error(f"Error closing TensorBoard: {e}")
        
        self._flush_json()
