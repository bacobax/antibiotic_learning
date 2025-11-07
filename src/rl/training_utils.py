"""
Training utilities for Recurrent PPO on bacteria simulation.

This module provides reusable training functions and helpers used by:
  - src/train.py (headless training entry point)
  - src/train_with_visualization.py (training with live visualization)

All hyperparameters are configured via YAML files.

View TensorBoard during/after training:
   tensorboard --logdir=./checkpoints --port=6006

The wrapper in env_wrapper.py handles all Mesa interaction.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

import numpy as np
import torch

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .config_loader import CompleteConfig, load_config, save_config
from .agent import RLAgent
from .env_wrapper import PetriEnvWrapper
from .models import RecurrentActorCritic
from .buffer import RolloutBuffer
from .ppo import PPOTrainer
from .logger import TrainingLogger
from .training_config import PPOConfig, set_global_seed
from simulation.model import BacteriaModel
from .env_wrapper import ACTION_DOSE, ACTION_COUNT_BACTERIA, ACTION_NOOP, ACTION_SEQUENCING


# ============================================================================
# Directory management utilities
# ============================================================================

def create_run_directory(base_dir: Path, experiment_name: str) -> Path:
    """
    Create a timestamped run directory.
    
    Format: base_dir/experiment_name/DDMMYY_HHMMSS/
    
    Args:
        base_dir: Base checkpoint directory (e.g., ./checkpoints)
        experiment_name: Experiment name (e.g., ppo_production)
    
    Returns:
        Path to the created run directory
    """
    # Get current timestamp in DDMMYY_HHMMSS format
    timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
    
    # Create directory structure: base_dir/experiment_name/DDMMYY_HHMMSS/
    run_dir = base_dir / experiment_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir


# ============================================================================
# Training loop
# ============================================================================

def rollout(
    env: PetriEnvWrapper,
    agent: RLAgent,
    buffer: RolloutBuffer,
    num_steps: int,
    cfg: PPOConfig,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Collect rollout trajectory.
    
    Args:
        env: Environment wrapper
        model: Actor-critic model
        buffer: Rollout buffer to fill
        num_steps: Number of steps to collect
        h_state: Initial hidden state [layers, 1, hidden_dim]
        cfg: PPO config
    
    Returns:
        h_state: Final hidden state
        metrics: Rollout statistics (includes mean_population_per_episode)
    """
    
    obs = env.reset()
    episode_rewards = []
    episode_lengths = []
    episode_populations = []  # Track population at end of each episode
    
    # Budget tracking per episode
    episode_budgets_spent = []
    episode_budgets_remaining = []
    episode_budgets_per_step = []
    
    # Track individual reward components per episode
    episode_reward_immediate = []
    episode_reward_maintenance = []
    episode_reward_budget_penalty = []
    episode_reward_unaffordable_action_penalty = []
    episode_reward_delayed = []
    episode_reward_survival_bonus = []
    episode_reward_budget_conservation = []
    episode_reward_regular_count_bonus = []
    episode_reward_safe_behavior_bonus = []
    episode_reward_informed_dosing_bonus = []
    episode_reward_count_population = []
    episode_reward_critical_inaction_penalty = []
    episode_reward_critical_noop_penalty = []
    episode_reward_prediction = []
    episode_reward_early_termination_penalty = []
    
    # Early termination tracking
    early_termination_count = 0
    
    # Prediction tracking (diagnostic only - error metric)
    episode_pred_error = []
    
    current_episode_reward = 0.0
    current_episode_length = 0
    dose_action_count = 0  # Track number of DOSE actions
    sequencing_action_count = 0  # Track number of SEQUENCING actions
    count_action_count = 0  # Track number of COUNT actions
    noop_action_count = 0  # Track number of NOOP actions

    total_actions = 0  # Track total actions
    
    # Accumulators for current episode reward components
    current_reward_immediate = 0.0
    current_reward_maintenance = 0.0
    current_reward_budget_penalty = 0.0
    current_reward_unaffordable_action_penalty = 0.0
    current_reward_delayed = 0.0
    current_reward_survival_bonus = 0.0
    current_reward_budget_conservation = 0.0
    current_reward_regular_count_bonus = 0.0
    current_reward_safe_behavior_bonus = 0.0
    current_reward_informed_dosing_bonus = 0.0
    current_reward_count_population = 0.0
    current_reward_critical_inaction_penalty = 0.0
    current_reward_critical_noop_penalty = 0.0
    current_reward_prediction = 0.0
    current_reward_early_termination_penalty = 0.0
    
    current_pred_error = 0.0
    
    agent.start_episode()
    
    for step in range(num_steps):
        # Prepare observation
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(cfg.device)  # [1, obs_dim]
        
        # Get action from policy (now includes action mask)
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
            ) = agent.select_action(obs)
        

        pure_a_disc = a_disc.cpu().numpy()[0]
        pure_a_cont = a_cont.cpu().numpy()[0]
        # Extract actions
        
        # Track dose actions
        if pure_a_disc == ACTION_DOSE:
            dose_action_count += 1
        if pure_a_disc == ACTION_SEQUENCING:
            sequencing_action_count += 1
        if pure_a_disc == ACTION_COUNT_BACTERIA:
            count_action_count += 1
        if pure_a_disc == ACTION_NOOP:
            noop_action_count += 1
            
        total_actions += 1
        
        # Get prediction value for passing to environment
        pred_next_pop_value = pred_next_pop.cpu().item()
        
        # Environment step (now includes prediction reward computation)
        next_obs, reward, done, info = env.step(pure_a_disc, pure_a_cont, pred_population=pred_next_pop_value)
        
        # Extract prediction supervision and diagnostics
        population_counted_norm = info.get('population_next_norm', 0.0)
        count_was_performed = info.get('count_was_performed', False)
        count_mask_value = 1.0 if count_was_performed else 0.0
        
        # Track prediction error for diagnostics (separate from reward which is in info)
        if count_was_performed:
            pred_error = abs(pred_next_pop_value - population_counted_norm)
            current_pred_error += pred_error
            # Note: prediction reward is now computed by environment and included in total reward
        
        # Accumulate reward components for current episode
        current_reward_immediate += info.get('reward_immediate', 0.0)
        current_reward_maintenance += info.get('reward_maintenance', 0.0)
        current_reward_budget_penalty += info.get('reward_budget_penalty', 0.0)
        current_reward_unaffordable_action_penalty += info.get('reward_unaffordable_action_penalty', 0.0)
        current_reward_delayed += info.get('reward_delayed', 0.0)
        current_reward_survival_bonus += info.get('reward_survival_bonus', 0.0)
        current_reward_budget_conservation += info.get('reward_budget_conservation', 0.0)
        current_reward_regular_count_bonus += info.get('reward_regular_count_bonus', 0.0)
        current_reward_safe_behavior_bonus += info.get('reward_safe_behavior_bonus', 0.0)
        current_reward_informed_dosing_bonus += info.get('reward_informed_dosing_bonus', 0.0)
        current_reward_count_population += info.get('reward_count_population', 0.0)
        current_reward_critical_inaction_penalty += info.get('reward_critical_inaction_penalty', 0.0)
        current_reward_critical_noop_penalty += info.get('reward_critical_noop_penalty', 0.0)
        current_reward_prediction += info.get('reward_prediction', 0.0)
        current_reward_early_termination_penalty += info.get('reward_early_termination_penalty', 0.0)
        
        # Track early termination occurrences
        if info.get('early_termination_triggered', False):
            early_termination_count += 1
        
        # Store in buffer (now includes action mask)
        buffer.add(
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
        
        # Update state
        obs = next_obs
        current_episode_reward += reward
        current_episode_length += 1
        
        # Handle episode termination
        if done:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            
            # Store reward components for completed episode
            episode_reward_immediate.append(current_reward_immediate)
            episode_reward_maintenance.append(current_reward_maintenance)
            episode_reward_budget_penalty.append(current_reward_budget_penalty)
            episode_reward_unaffordable_action_penalty.append(current_reward_unaffordable_action_penalty)
            episode_reward_delayed.append(current_reward_delayed)
            episode_reward_survival_bonus.append(current_reward_survival_bonus)
            episode_reward_budget_conservation.append(current_reward_budget_conservation)
            episode_reward_regular_count_bonus.append(current_reward_regular_count_bonus)
            episode_reward_safe_behavior_bonus.append(current_reward_safe_behavior_bonus)
            episode_reward_informed_dosing_bonus.append(current_reward_informed_dosing_bonus)
            episode_reward_count_population.append(current_reward_count_population)
            episode_reward_critical_inaction_penalty.append(current_reward_critical_inaction_penalty)
            episode_reward_critical_noop_penalty.append(current_reward_critical_noop_penalty)
            episode_reward_prediction.append(current_reward_prediction)
            episode_reward_early_termination_penalty.append(current_reward_early_termination_penalty)
            
            # Store prediction metrics
            episode_pred_error.append(current_pred_error)
            
            # Log population at end of episode
            final_population = env.get_bacteria_population()
            episode_populations.append(final_population)
            
            # Get budget metrics for completed episode
            budget_metrics = env.get_episode_budget_metrics()
            episode_budgets_spent.append(budget_metrics['budget_spent'])
            episode_budgets_remaining.append(budget_metrics['current_budget'])
            episode_budgets_per_step.append(budget_metrics['budget_per_step'])
            
            # Reset episode tracking
            current_episode_reward = 0.0
            current_episode_length = 0
            current_reward_immediate = 0.0
            current_reward_maintenance = 0.0
            current_reward_budget_penalty = 0.0
            current_reward_unaffordable_action_penalty = 0.0
            current_reward_delayed = 0.0
            current_reward_survival_bonus = 0.0
            current_reward_budget_conservation = 0.0
            current_reward_regular_count_bonus = 0.0
            current_reward_safe_behavior_bonus = 0.0
            current_reward_informed_dosing_bonus = 0.0
            current_reward_count_population = 0.0
            current_reward_critical_inaction_penalty = 0.0
            current_reward_critical_noop_penalty = 0.0
            current_reward_prediction = 0.0
            current_reward_early_termination_penalty = 0.0
            current_pred_error = 0.0
            
            obs = env.reset()
            # Reset hidden state on episode boundary
            agent.start_episode()
    
    # Compute metrics
    dose_action_percentage = (dose_action_count / total_actions * 100) if total_actions > 0 else 0.0
    sequencing_action_percentage = (sequencing_action_count / total_actions * 100) if total_actions > 0 else 0.0
    count_action_percentage = (count_action_count / total_actions * 100) if total_actions > 0 else 0.0
    noop_action_percentage = (noop_action_count / total_actions * 100) if total_actions > 0 else 0.0

    print(f"APPLY ANTIBIOTIC RATE: {dose_action_percentage:.2f}%")
    print(f"SEQUENCING ACTION RATE: {sequencing_action_percentage:.2f}%")
    print(f"COUNT BACTERIA ACTION RATE: {count_action_percentage:.2f}%")
    print(f"NOOP ACTION RATE: {noop_action_percentage:.2f}%")
    
    # Print prediction metrics
    mean_pred_reward = float(np.mean(episode_reward_prediction)) if episode_reward_prediction else 0.0
    mean_pred_error = float(np.mean(episode_pred_error)) if episode_pred_error else 0.0
    print(f"PREDICTION REWARD (AVG): {mean_pred_reward:.4f} | ERROR: {mean_pred_error:.4f}")
    
    metrics = {
        "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_episode_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        "max_episode_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "min_episode_reward": float(np.min(episode_rewards)) if episode_rewards else 0.0,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "std_episode_length": float(np.std(episode_lengths)) if episode_lengths else 0.0,
        "min_episode_length": float(np.min(episode_lengths)) if episode_lengths else 0.0,
        "max_episode_length": float(np.max(episode_lengths)) if episode_lengths else 0.0,
        "num_episodes": int(len(episode_rewards)),
        "mean_population_per_episode": float(np.mean(episode_populations)) if episode_populations else 0.0,
        "final_population": float(episode_populations[-1]) if episode_populations else 0.0,
        "dose_action_percentage": float(dose_action_percentage),
        "count_action_percentage": float(count_action_percentage),
        "sequencing_action_percentage": float(sequencing_action_percentage),
        "noop_action_percentage": float(noop_action_percentage),
        # Budget metrics
        "mean_budget_spent": float(np.mean(episode_budgets_spent)) if episode_budgets_spent else 0.0,
        "mean_budget_remaining": float(np.mean(episode_budgets_remaining)) if episode_budgets_remaining else 0.0,
        "mean_budget_per_step": float(np.mean(episode_budgets_per_step)) if episode_budgets_per_step else 0.0,
        # Reward component metrics with category prefixes for TensorBoard
        "rewards/immediate": float(np.mean(episode_reward_immediate)) if episode_reward_immediate else 0.0,
        "rewards/maintenance": float(np.mean(episode_reward_maintenance)) if episode_reward_maintenance else 0.0,
        "rewards/budget_penalty": float(np.mean(episode_reward_budget_penalty)) if episode_reward_budget_penalty else 0.0,
        "rewards/unaffordable_action_penalty": float(np.mean(episode_reward_unaffordable_action_penalty)) if episode_reward_unaffordable_action_penalty else 0.0,
        "rewards/delayed": float(np.mean(episode_reward_delayed)) if episode_reward_delayed else 0.0,
        "rewards/survival_bonus": float(np.mean(episode_reward_survival_bonus)) if episode_reward_survival_bonus else 0.0,
        "rewards/budget_conservation": float(np.mean(episode_reward_budget_conservation)) if episode_reward_budget_conservation else 0.0,
        "rewards/regular_count_bonus": float(np.mean(episode_reward_regular_count_bonus)) if episode_reward_regular_count_bonus else 0.0,
        "rewards/safe_behavior_bonus": float(np.mean(episode_reward_safe_behavior_bonus)) if episode_reward_safe_behavior_bonus else 0.0,
        "rewards/informed_dosing_bonus": float(np.mean(episode_reward_informed_dosing_bonus)) if episode_reward_informed_dosing_bonus else 0.0,
        "rewards/count_population": float(np.mean(episode_reward_count_population)) if episode_reward_count_population else 0.0,
        "rewards/critical_inaction_penalty": float(np.mean(episode_reward_critical_inaction_penalty)) if episode_reward_critical_inaction_penalty else 0.0,
        "rewards/critical_noop_penalty": float(np.mean(episode_reward_critical_noop_penalty)) if episode_reward_critical_noop_penalty else 0.0,
        "rewards/prediction": float(np.mean(episode_reward_prediction)) if episode_reward_prediction else 0.0,
        "rewards/early_termination_penalty": float(np.mean(episode_reward_early_termination_penalty)) if episode_reward_early_termination_penalty else 0.0,
        "rewards/total": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        # Prediction metrics
        "prediction/error": float(np.mean(episode_pred_error)) if episode_pred_error else 0.0,
        # Early termination metrics
        "early_termination/count": early_termination_count,
        "early_termination/rate": float(early_termination_count) / len(episode_rewards) if episode_rewards else 0.0,
    }
    
    return metrics


def _initialize_agent(cfg: PPOConfig, env: PetriEnvWrapper) -> RLAgent:
    """
    Create and initialize agent with model and trainer.
    
    Args:
        cfg: PPO configuration
        env: Environment wrapper (needed for action masking)
    
    Returns:
        Initialized RLAgent with trainer
    """
    model = RecurrentActorCritic(
        obs_dim=cfg.obs_dim,
        n_discrete=cfg.n_discrete,
        k_doses=cfg.k_doses,
        hidden_dim=cfg.hidden_dim,
        rnn_layers=cfg.rnn_layers,
        dose_action_index=cfg.dose_action_index
    )

    agent = RLAgent(model, cfg.device, env=env).with_trainer(cfg)
    return agent


def _log_training_start(cfg: PPOConfig, total_updates: int, logger: TrainingLogger) -> None:
    """Log training initialization information."""
    logger.log_info(f"Starting PPO training for {total_updates} updates")
    logger.log_info(f"Config: obs_dim={cfg.obs_dim}, hidden_dim={cfg.hidden_dim}, "
                    f"rnn_layers={cfg.rnn_layers}, k_doses={cfg.k_doses}")
    logger.log_info(f"Hyperparams: lr={cfg.lr}, gamma={cfg.gamma}, gae_lambda={cfg.gae_lambda}")
    logger.log_info(f"Rollout steps per update: {cfg.rollout_steps}, PPO epochs: {cfg.epochs}")
    logger.log_debug(f"Using device: {cfg.device}")


def _handle_checkpoint(
    agent: RLAgent,
    update_idx: int,
    total_updates: int,
    save_dir: Path,
    logger: TrainingLogger,
) -> None:
    """
    Save periodic and final checkpoints.
    
    Args:
        agent: RL agent to save
        update_idx: Current update index
        total_updates: Total number of updates
        save_dir: Directory to save to
        logger: Training logger
    """
    if (update_idx + 1) % 50 == 0:
        checkpoint_path = save_dir / f"checkpoint_{update_idx+1}.pt"
        agent.save_model(checkpoint_path, extra_info={"update": update_idx + 1})
        logger.log_debug(f"Saved checkpoint: {checkpoint_path}")
    
    if (update_idx + 1) == total_updates:
        checkpoint_path = save_dir / f"checkpoint_final_{update_idx+1}.pt"
        agent.save_model(checkpoint_path, extra_info={"update": update_idx + 1})
        logger.log_info(f"Saved final checkpoint: {checkpoint_path}")


def _finalize_training(
    log_data: list,
    reward_history: list,
    loss_history: list,
    total_updates: int,
    total_time: float,
    save_dir: Path,
    logger: TrainingLogger,
) -> None:
    """
    Save logs and generate training summary.
    
    Args:
        log_data: List of log entries
        reward_history: Episode reward history
        loss_history: Loss history
        total_updates: Total number of updates
        total_time: Total training time in seconds
        save_dir: Directory to save to
        logger: Training logger
    """
    log_path = save_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    logger.log_debug(f"Saved training log: {log_path}")
    logger.log_summary(total_updates, total_time, reward_history, loss_history)
    logger.close()


def train(cfg: PPOConfig, env: PetriEnvWrapper, save_dir: Path, total_updates: int, logger: TrainingLogger):
    """
    Main training loop with comprehensive logging and TensorBoard integration.
    
    Core flow:
        1. Initialize agent
        2. Collect rollouts and update policy iteratively
        3. Save checkpoints and logs
    
    Args:
        cfg: PPO configuration
        env: Environment wrapper
        save_dir: Directory to save checkpoints
        total_updates: Number of PPO updates to perform
        logger: TrainingLogger instance for all logging
    """
    # ========================================================================
    # SETUP
    # ========================================================================
    agent = _initialize_agent(cfg, env)
    _log_training_start(cfg, total_updates, logger)
    
    # Training state
    log_data = []
    reward_history = []
    loss_history = []
    start_time = time.time()
    
    # Progress bar
    iterator = tqdm(range(total_updates), desc="Training") if HAS_TQDM else range(total_updates)
    
    # ========================================================================
    # MAIN TRAINING LOOP
    # ========================================================================
    for update_idx in iterator:
        # Collect rollout
        buffer = RolloutBuffer()
        rollout_metrics = rollout(env, agent, buffer, cfg.rollout_steps, cfg)
        
        # Update policy
        train_stats = agent.update_policy(buffer)
        
        # Log and track
        log_entry = {
            "update": update_idx,
            **rollout_metrics,
            **train_stats,
        }
        log_data.append(log_entry)
        reward_history.append(rollout_metrics['mean_episode_reward'])
        loss_history.append(train_stats['loss_actor'])
        logger.log_metrics(update_idx, rollout_metrics, train_stats)
        
        # Periodic reporting
        if update_idx % 10 == 0 and update_idx > 0:
            elapsed = time.time() - start_time
            logger.log_update(update_idx, total_updates, rollout_metrics, train_stats, elapsed)
        
        # Check for divergence
        if np.isnan(train_stats['loss_actor']):
            logger.log_error("Loss became NaN, stopping training")
            break
        
        # Checkpointing
        _handle_checkpoint(agent, update_idx, total_updates, save_dir, logger)
    
    # ========================================================================
    # FINALIZE
    # ========================================================================
    total_time = time.time() - start_time
    _finalize_training(log_data, reward_history, loss_history, total_updates, total_time, save_dir, logger)


def _setup_logger_and_log_startup(
    save_dir: Path,
    config: CompleteConfig,
) -> TrainingLogger:
    """
    Initialize logger and log startup information.
    
    Args:
        save_dir: Directory to save logs
        config: Complete configuration
    
    Returns:
        Initialized TrainingLogger
    """
    logger = TrainingLogger(save_dir, experiment_name=config.training.experiment_name)
    
    logger.log_info("="*70)
    logger.log_info("PPO Training Started")
    logger.log_info("="*70)
    logger.log_info(f"Configuration from: {config.training.save_dir}")
    
    # Extract rewards config for cleaner access
    rewards = config.environment.rewards
    
    logger.log_info(f"Environment Settings:")
    logger.log_info(f"  - Max steps: {config.environment.max_steps}")
    logger.log_info(f"  - Target population: {rewards.population.target_population}")
    logger.log_info(f"  - Budget: {rewards.budget.budget_init}")
    logger.log_info(f"  - K doses (antibiotic types): {config.environment.k_doses}")
    logger.log_info(f"  - Device: {config.environment.device}")
    
    logger.log_info(f"Model Architecture:")
    logger.log_info(f"  - Hidden dim: {config.model.hidden_dim}")
    logger.log_info(f"  - RNN layers: {config.model.rnn_layers}")
    
    logger.log_info(f"PPO Hyperparameters:")
    logger.log_info(f"  - Learning rate: {config.ppo.lr}")
    logger.log_info(f"  - Gamma: {config.ppo.gamma}")
    logger.log_info(f"  - GAE lambda: {config.ppo.gae_lambda}")
    logger.log_info(f"  - Rollout steps: {config.ppo.rollout_steps}")
    logger.log_info(f"  - Epochs: {config.ppo.epochs}")
    logger.log_info(f"  - Seq len: {config.ppo.seq_len}")
    
    logger.log_info(f"Training Config:")
    logger.log_info(f"  - Total updates: {config.training.total_updates}")
    logger.log_info(f"  - Seed: {config.training.seed}")
    
    logger.log_info(f"Reward Weights:")
    logger.log_info(f"  - Population: {rewards.dose.w_pop}")
    logger.log_info(f"  - Genome: {rewards.dose.w_genome}")
    logger.log_info(f"  - Cost: {rewards.dose.w_cost}")
    logger.log_info(f"  - Population maintenance: {rewards.population.w_population_maintenance}")
    
    logger.log_info(f"Reward Modules:")
    logger.log_info(f"  - Survival bonus: {'enabled' if rewards.survival_bonus.enabled else 'disabled'}")
    if rewards.survival_bonus.enabled:
        logger.log_info(f"    - Base bonus: {rewards.survival_bonus.base_bonus}")
        logger.log_info(f"    - Scaling type: {rewards.survival_bonus.scaling_type}")
    logger.log_info(f"  - Budget conservation: {'enabled' if rewards.budget_conservation.enabled else 'disabled'}")
    if rewards.budget_conservation.enabled:
        logger.log_info(f"    - Weight: {rewards.budget_conservation.weight}")
        logger.log_info(f"    - Reserve threshold: {rewards.budget_conservation.reserve_bonus_threshold}")
    logger.log_info(f"  - Prediction reward: {'enabled' if rewards.prediction.enabled else 'disabled'}")
    if rewards.prediction.enabled:
        logger.log_info(f"    - Weight: {rewards.prediction.weight}")
    logger.log_info(f"  - Early termination: {'enabled' if rewards.early_termination.enabled else 'disabled'}")
    if rewards.early_termination.enabled:
        logger.log_info(f"    - Penalty: {rewards.early_termination.penalty}")
        logger.log_info(f"    - Population threshold: {rewards.early_termination.population_threshold}x target")
        logger.log_info(f"    - Require budget depleted: {rewards.early_termination.require_budget_depleted}")
    
    logger.log_info(f"Action Costs:")
    logger.log_info(f"  - Sequencing: {config.actions.sequencing_cost}")
    logger.log_info(f"  - Sequencing duration: {config.actions.sequencing_duration} steps")
    logger.log_info(f"  - Dose per unit: {config.actions.dose_cost_per_unit}")
    
    return logger


def _create_environment(
    config: CompleteConfig,
    logger: TrainingLogger,
) -> PetriEnvWrapper:
    """
    Create and initialize the environment.
    
    Args:
        config: Complete configuration
        logger: Training logger
    
    Returns:
        Initialized PetriEnvWrapper
    """
    logger.log_info("Creating environment...")
    
    # Extract reward configs for cleaner access
    rewards = config.environment.rewards
    
    env = PetriEnvWrapper(
        mesa_model_factory=BacteriaModel,
        k_doses=config.environment.k_doses,
        scale_dose=lambda x: x / 2 / config.environment.k_doses,
        max_steps=config.environment.max_steps,
        # Population reward params
        target_population=rewards.population.target_population,
        population_norm=rewards.population.population_norm,
        w_population_maintenance=rewards.population.w_population_maintenance,
        noop_band_factor=rewards.population.noop_band_factor,
        noop_reward_magnitude=rewards.population.noop_reward_magnitude,
        # Dose reward params
        w_pop=rewards.dose.w_pop,
        w_genome=rewards.dose.w_genome,
        w_cost=rewards.dose.w_cost,
        # Budget params
        budget_init=rewards.budget.budget_init,
        budget_norm=rewards.budget.budget_norm,
        budget_penalty=rewards.budget.budget_penalty,
        unaffordable_action_penalty=rewards.budget.unaffordable_action_penalty,
        # Action costs
        sequencing_cost=config.actions.sequencing_cost,
        sequencing_duration=config.actions.sequencing_duration,
        redundant_sequencing_penalty=rewards.sequencing.redundant_penalty,
        dose_cost=config.actions.dose_cost,
        dose_cost_per_unit=config.actions.dose_cost_per_unit,
        count_cost=config.actions.count_cost,
        # Informed dosing params
        informed_dosing_reward=rewards.informed_dosing.reward,
        informed_dosing_above_target_reward=rewards.informed_dosing.above_target_reward,
        informed_dosing_window=rewards.informed_dosing.window,
        informed_sequencing_window=rewards.informed_dosing.sequencing_window,
        blind_dosing_penalty=rewards.informed_dosing.blind_penalty,
        dosing_low_population_penalty=rewards.informed_dosing.low_population_penalty,
        # Regular monitoring params
        regular_count_reward=rewards.regular_monitoring.count_reward,
        regular_count_interval=rewards.regular_monitoring.count_interval,
        regular_count_min_interval=rewards.regular_monitoring.count_min_interval,
        safe_nondosing_reward=rewards.regular_monitoring.safe_nondosing_reward,
        count_population_reward=rewards.population.count_population_reward,
        # Critical inaction penalties
        critical_high_population_threshold=rewards.critical_inaction.high_population_threshold,
        critical_no_action_penalty=rewards.critical_inaction.no_action_penalty,
        critical_no_dose_penalty=rewards.critical_inaction.no_dose_penalty,
        critical_freshness_window=rewards.critical_inaction.freshness_window,
        critical_noop_penalty=rewards.critical_inaction.noop_penalty,
        critical_noop_threshold=rewards.critical_inaction.noop_threshold,
        dose_missing_feedback_penalty=rewards.dose.missing_feedback_penalty,
        # Prediction reward
        prediction_reward_weight=rewards.prediction.weight if rewards.prediction.enabled else 0.0,
        # Early termination
        early_termination_enabled=rewards.early_termination.enabled,
        early_termination_penalty=rewards.early_termination.penalty,
        early_termination_population_threshold=rewards.early_termination.population_threshold,
        early_termination_require_budget_depleted=rewards.early_termination.require_budget_depleted,
        # Device config
        device=config.environment.device,
        dtype=config.torch_dtype,
    )
    
    # Enable survival bonus reward if configured
    if rewards.survival_bonus.enabled:
        env.enable_survival_bonus(
            base_bonus=rewards.survival_bonus.base_bonus,
            scaling_type=rewards.survival_bonus.scaling_type,
            scaling_factor=rewards.survival_bonus.scaling_factor,
            max_bonus=rewards.survival_bonus.max_bonus,
        )
        logger.log_info(f"✓ Survival bonus enabled: base={rewards.survival_bonus.base_bonus}, "
                       f"type={rewards.survival_bonus.scaling_type}")
    
    # Enable budget conservation reward if configured
    if rewards.budget_conservation.enabled:
        env.enable_budget_conservation(
            weight=rewards.budget_conservation.weight,
            spending_penalty_factor=rewards.budget_conservation.spending_penalty_factor,
            reserve_bonus_threshold=rewards.budget_conservation.reserve_bonus_threshold,
            reserve_bonus_magnitude=rewards.budget_conservation.reserve_bonus_magnitude,
        )
        logger.log_info(f"✓ Budget conservation enabled: weight={rewards.budget_conservation.weight}, "
                       f"threshold={rewards.budget_conservation.reserve_bonus_threshold}")
    
    # Log prediction reward configuration
    if rewards.prediction.enabled:
        logger.log_info(f"✓ Prediction reward enabled: weight={rewards.prediction.weight}")
    
    logger.log_debug("✓ Environment created successfully")
    return env


def _build_ppo_config(env: PetriEnvWrapper, config: CompleteConfig) -> PPOConfig:
    """
    Build PPO configuration from CompleteConfig.
    
    Args:
        env: Initialized environment
        config: Complete configuration
    
    Returns:
        Initialized PPOConfig (from .config module)
    """
    obs_dim = env.get_obs_dim()
    
    ppo_cfg = PPOConfig(
        obs_dim=obs_dim,
        n_discrete=config.model.n_discrete,
        k_doses=config.environment.k_doses,
        hidden_dim=config.model.hidden_dim,
        rnn_layers=config.model.rnn_layers,
        gamma=config.ppo.gamma,
        gae_lambda=config.ppo.gae_lambda,
        clip_eps=config.ppo.clip_eps,
        vf_coef=config.ppo.vf_coef,
        ent_coef=config.ppo.ent_coef,
        max_grad_norm=config.ppo.max_grad_norm,
        seq_len=config.ppo.seq_len,
        rollout_steps=config.ppo.rollout_steps,
        epochs=config.ppo.epochs,
        batch_seq_len=config.ppo.batch_seq_len,
        lr=config.ppo.lr,
        device=config.environment.device,
        seed=config.training.seed,
        dose_action_index=config.model.dose_action_index,
    )
    
    return ppo_cfg


def _save_configs(
    save_dir: Path,
    config: CompleteConfig,
    logger: TrainingLogger,
) -> None:
    """
    Save configurations to disk.
    
    Args:
        save_dir: Directory to save to
        config: Complete configuration
        logger: Training logger
    """
    # Save complete config as YAML
    config_path = save_dir / "complete_config.yaml"
    try:
        save_config(config, config_path)
        logger.log_debug(f"✓ Saved complete config to: {config_path}")
    except Exception as e:
        logger.log_debug(f"Warning: Could not save config: {e}")
