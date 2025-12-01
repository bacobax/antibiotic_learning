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
import torch.nn.functional as F

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
    
    # Track individual reward components per episode (new simplified structure)
    episode_reward_pre = []
    episode_reward_post_penalties = []
    episode_reward_kernel_maintenance = []
    episode_reward_survival_bonus = []
    episode_reward_prediction = []
    episode_reward_early_termination_penalty = []
    episode_reward_cost_penalty = []
    
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
    
    # Accumulators for current episode reward components (new simplified structure)
    current_reward_pre = 0.0
    current_reward_post_penalties = 0.0
    current_reward_kernel_maintenance = 0.0
    current_reward_survival_bonus = 0.0
    current_reward_prediction = 0.0
    current_reward_early_termination_penalty = 0.0
    current_reward_cost_penalty = 0.0
    
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
                action_mask,
                prev_action_onehot,
                prev_action_cont,
                prev_pred_next_pop,
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
        
        # Accumulate reward components for current episode (new keys)
        current_reward_pre += info.get('reward_pre', 0.0)
        current_reward_post_penalties += info.get('reward_post_penalties', 0.0)
        current_reward_kernel_maintenance += info.get('reward_kernel_maintenance', 0.0)
        current_reward_survival_bonus += info.get('reward_survival_bonus', 0.0)
        current_reward_prediction += info.get('reward_prediction', 0.0)
        current_reward_early_termination_penalty += info.get('reward_early_termination_penalty', 0.0)
        current_reward_cost_penalty += info.get('reward_cost_penalty', 0.0)
        
        # Track early termination occurrences
        if info.get('early_termination_triggered', False):
            early_termination_count += 1
        
        # Build prediction head input features and store in buffer (includes action mask)
        a_disc_onehot = F.one_hot(a_disc, num_classes=cfg.n_discrete).float()
        dose_mask_tensor = (a_disc == cfg.dose_action_index).float().unsqueeze(-1)
        a_cont_for_pred = a_cont * dose_mask_tensor
        pred_action_input = torch.cat([a_disc_onehot, a_cont_for_pred], dim=-1)

        # Store in buffer (now includes action mask and action conditioning)
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
            prev_action_onehot=prev_action_onehot.cpu(),
            prev_action_cont=prev_action_cont.cpu(),
            pred_action_input=pred_action_input.cpu(),
            prev_pred_next_pop=prev_pred_next_pop.cpu(),
        )
        
        # Update state
        obs = next_obs
        current_episode_reward += reward
        current_episode_length += 1
        
        # Handle episode termination
        if done:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            
            # Store reward components for completed episode (new structure)
            episode_reward_pre.append(current_reward_pre)
            episode_reward_post_penalties.append(current_reward_post_penalties)
            episode_reward_kernel_maintenance.append(current_reward_kernel_maintenance)
            episode_reward_survival_bonus.append(current_reward_survival_bonus)
            episode_reward_prediction.append(current_reward_prediction)
            episode_reward_early_termination_penalty.append(current_reward_early_termination_penalty)
            episode_reward_cost_penalty.append(current_reward_cost_penalty)
            
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
            current_reward_informed_dose = 0.0
            current_reward_count_population = 0.0
            current_reward_critical_inaction_penalty = 0.0
            current_reward_critical_noop_penalty = 0.0
            current_reward_prediction = 0.0
            current_reward_early_termination_penalty = 0.0
            current_reward_cost_penalty = 0.0
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
        # Reward component metrics with category prefixes for TensorBoard (new simplified)
        "rewards/pre": float(np.mean(episode_reward_pre)) if episode_reward_pre else 0.0,
        "rewards/post_penalties": float(np.mean(episode_reward_post_penalties)) if episode_reward_post_penalties else 0.0,
        "rewards/kernel_maintenance": float(np.mean(episode_reward_kernel_maintenance)) if episode_reward_kernel_maintenance else 0.0,
        "rewards/survival_bonus": float(np.mean(episode_reward_survival_bonus)) if episode_reward_survival_bonus else 0.0,
        "rewards/prediction": float(np.mean(episode_reward_prediction)) if episode_reward_prediction else 0.0,
        "rewards/early_termination_penalty": float(np.mean(episode_reward_early_termination_penalty)) if episode_reward_early_termination_penalty else 0.0,
    "rewards/cost_penalty": float(np.mean(episode_reward_cost_penalty)) if episode_reward_cost_penalty else 0.0,
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
        dose_action_index=cfg.dose_action_index,
        sigmoid_scale_factor=cfg.sigmoid_scale_factor,
    )

    agent = RLAgent(model, cfg.device, env=env).with_trainer(cfg)
    return agent


def _load_checkpoint_into_agent(agent: RLAgent, checkpoint_path: str, logger: TrainingLogger) -> int:
    """
    Load checkpoint state into an existing agent.
    
    Args:
        agent: Agent to load checkpoint into
        checkpoint_path: Path to checkpoint file
        logger: Training logger
        
    Returns:
        Update number from checkpoint
    """
    import sys
    from .training_config import PPOConfig
    # Same module remapping as in load_agent_from_checkpoint
    class ConfigModule:
        PPOConfig = PPOConfig
    
    if "rl.config" not in sys.modules:
        sys.modules["rl.config"] = ConfigModule()
    
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    except ModuleNotFoundError as e:
        if "rl.config" in str(e):
            sys.modules["rl.config"] = ConfigModule()
            checkpoint = torch.load(checkpoint_path, weights_only=False)
        else:
            raise
    
    # Load model state
    agent.model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state if available
    if "optimizer_state_dict" in checkpoint and agent.trainer:
        agent.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.log_debug("✓ Loaded optimizer state")
    
    update_number = checkpoint.get("update", 0)
    logger.log_info(f"✓ Loaded checkpoint from update {update_number}")
    
    return update_number


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
    checkpoint_interval: int,
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
    if (update_idx + 1) % max(1, checkpoint_interval) == 0:
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


def train(cfg: PPOConfig, env: PetriEnvWrapper, save_dir: Path, total_updates: int, logger: TrainingLogger, checkpoint_interval: int = 50, starting_update: int = 0, agent: RLAgent = None):
    """
    Main training loop with comprehensive logging and TensorBoard integration.
    
    Core flow:
        1. Initialize agent (or use provided one for resumption)
        2. Collect rollouts and update policy iteratively
        3. Save checkpoints and logs
    
    Args:
        cfg: PPO configuration
        env: Environment wrapper
        save_dir: Directory to save checkpoints
        total_updates: Number of PPO updates to perform
        logger: TrainingLogger instance for all logging
        checkpoint_interval: Save checkpoint every N updates
        starting_update: Update number to start from (for resuming training)
        agent: Pre-initialized agent (for resuming from checkpoint)
    """
    # ========================================================================
    # SETUP
    # ========================================================================
    if agent is None:
        agent = _initialize_agent(cfg, env)
    _log_training_start(cfg, total_updates, logger)
    
    if starting_update > 0:
        logger.log_info(f"Resuming training from update {starting_update}")
    
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
    for update_idx in range(starting_update, total_updates):
        if HAS_TQDM:
            iterator.update(1)
            iterator.set_description(f"Training (update {update_idx+1}/{total_updates})")
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
        _handle_checkpoint(agent, update_idx, total_updates, save_dir, logger, checkpoint_interval)
    
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
    # Report device with CUDA details when relevant
    try:
        resolved_device = config.device_type
    except Exception:
        resolved_device = str(config.environment.device)
    logger.log_info(f"  - Device: {resolved_device}")
    try:
        import torch
        if isinstance(resolved_device, str) and resolved_device.startswith("cuda"):
            cuda_available = torch.cuda.is_available()
            logger.log_info(f"    - torch.cuda.is_available: {cuda_available}")
            if cuda_available:
                try:
                    idx = torch.cuda.current_device()
                    name = torch.cuda.get_device_name(idx)
                    logger.log_info(f"    - Using GPU: cuda:{idx} ({name})")
                except Exception:
                    pass
        elif resolved_device == "cpu":
            logger.log_info("    - Running on CPU")
    except Exception:
        pass
    
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
    logger.log_info(
        f"    - Population thresholds: ≤{rewards.early_termination.population_low_threshold}x "
        f"or ≥{rewards.early_termination.population_threshold}x target"
    )
    logger.log_info(f"    - Extinction penalty: {rewards.early_termination.extinction_penalty}")
    if rewards.early_termination.enabled:
        late_penalty = (
            rewards.early_termination.min_penalty
            if rewards.early_termination.min_penalty is not None
            else rewards.early_termination.penalty
        )
        logger.log_info(
            f"    - Penalty (early → late): {rewards.early_termination.penalty} → {late_penalty}"
        )
        logger.log_info(
            f"    - Penalty decay power: {rewards.early_termination.penalty_decay_power}"
        )
        logger.log_info(f"    - Require budget depleted: {rewards.early_termination.require_budget_depleted}")
    
    logger.log_info(f"Action Costs:")
    logger.log_info(f"  - Weight: {config.actions.cost_weight}")
    logger.log_info(f"  - NOOP: {config.actions.noop_cost}")
    logger.log_info(f"  - COUNT: {config.actions.count_cost}")
    logger.log_info(f"  - SEQUENCING: {config.actions.sequencing_cost}")
    logger.log_info(f"    \u2514 duration: {config.actions.sequencing_duration} steps")
    logger.log_info(f"  - DOSE base: {config.actions.dose_cost}")
    logger.log_info(f"  - DOSE per unit: {config.actions.dose_cost_per_unit}")
    
    return logger


def _create_environment(
    config: CompleteConfig,
    logger: TrainingLogger,
) -> PetriEnvWrapper:
    """
    Create and initialize the environment with the new simplified reward structure.
    
    Args:
        config: Complete configuration
        logger: Training logger
    
    Returns:
        Initialized PetriEnvWrapper
    """
    logger.log_info("Creating environment with simplified reward structure...")
    
    # Extract reward configs for cleaner access
    rewards = config.environment.rewards
    
    # Map old config structure to new parameter names
    # If using new config format (with 'timing' and new reward structure), use those
    # Otherwise, provide sensible defaults
    
    # Check if we have new timing config
    timing = getattr(rewards, 'timing', None)
    if timing is not None:
        # New config format
        t_count_freshness = timing.t_count_freshness
        t_seq_freshness = timing.t_seq_freshness
        max_count_window = timing.max_count_window
        critical_ratio = timing.critical_ratio
        t_min_elapsed_time_count = timing.count_window.min_elapsed
        t_max_elapsed_time_count = timing.count_window.max_elapsed
        t_min_elapsed_time_seq = timing.seq_window.min_elapsed
        t_max_elapsed_time_seq = timing.seq_window.max_elapsed
    else:
        # Old config format - use defaults
        t_count_freshness = 5
        t_seq_freshness = 8
        max_count_window = 30
        critical_ratio = getattr(rewards.critical_inaction, 'high_population_threshold', 3.0)
        t_min_elapsed_time_count = getattr(rewards.regular_monitoring, 'count_min_interval', 5)
        t_max_elapsed_time_count = getattr(rewards.regular_monitoring, 'count_interval', 30)
        t_min_elapsed_time_seq = 8
        t_max_elapsed_time_seq = 50
    
    # Extract reward scalars (new format)
    informed_dosing = getattr(rewards, 'informed_dosing', None)
    if hasattr(informed_dosing, 'penalty_dosing_under_target'):
        # New config format
        penalty_informed_dosing_under = informed_dosing.penalty_dosing_under_target
        penalty_informed_dosing_under_dose_scale = getattr(
            informed_dosing, 'penalty_dosing_under_target_dose_scale', 0.0
        )
        penalty_informed_dosing_under_dose_exponent = getattr(
            informed_dosing, 'penalty_dosing_under_target_dose_exponent', 1.0
        )
        penalty_informed_dosing_under_deficit_scale = getattr(
            informed_dosing, 'penalty_dosing_under_target_deficit_scale', 0.0
        )
        penalty_informed_dosing_under_deficit_cap = getattr(
            informed_dosing, 'penalty_dosing_under_target_deficit_cap', 1.0
        )
        penalty_informed_dosing_under_max = getattr(
            informed_dosing, 'penalty_dosing_under_target_max', None
        )
        reward_informed_dosing_above = informed_dosing.reward_dosing_above_with_seq
        reward_informed_dosing_above_without_seq = informed_dosing.reward_dosing_above_no_seq
        penalty_blind_dose = informed_dosing.penalty_blind_dose
        penalty_blind_dose_amount_scale = getattr(
            informed_dosing, 'penalty_blind_dose_amount_scale', 0.0
        )
        penalty_blind_dose_amount_exponent = getattr(
            informed_dosing, 'penalty_blind_dose_amount_exponent', 1.0
        )
        penalty_blind_dose_max = getattr(
            informed_dosing, 'penalty_blind_dose_max', None
        )
    else:
        # Old format or defaults
        penalty_informed_dosing_under = 5.0
        penalty_informed_dosing_under_dose_scale = 0.0
        penalty_informed_dosing_under_dose_exponent = 1.0
        penalty_informed_dosing_under_deficit_scale = 0.0
        penalty_informed_dosing_under_deficit_cap = 1.0
        penalty_informed_dosing_under_max = None
        reward_informed_dosing_above = 2.0
        reward_informed_dosing_above_without_seq = 1.0
        penalty_blind_dose = 3.0
        penalty_blind_dose_amount_scale = 0.0
        penalty_blind_dose_amount_exponent = 1.0
        penalty_blind_dose_max = None
    
    sequencing_rewards = getattr(rewards, 'sequencing', None)
    if sequencing_rewards and hasattr(sequencing_rewards, 'seq_already_pending_penalty'):
        seq_already_pending_penalty = sequencing_rewards.seq_already_pending_penalty
        informative_seq_reward = sequencing_rewards.informative_seq_reward
    else:
        seq_already_pending_penalty = getattr(sequencing_rewards, 'redundant_penalty', 2.0) if sequencing_rewards else 2.0
        informative_seq_reward = 1.0
    
    counting_rewards = getattr(rewards, 'counting', None)
    if counting_rewards:
        informative_count_reward = counting_rewards.informative_count_reward
    else:
        informative_count_reward = 1.0
    
    noop_rewards = getattr(rewards, 'noop', None)
    if noop_rewards:
        strategic_noop_reward = noop_rewards.strategic_noop_reward
    else:
        strategic_noop_reward = 0.5
    
    critical_penalties = getattr(rewards, 'critical_penalties', None)
    if critical_penalties:
        penalty_critical_no_dose = critical_penalties.penalty_critical_no_dose
        penalty_critical_no_count = critical_penalties.penalty_critical_no_count
    else:
        penalty_critical_no_dose = getattr(rewards.critical_inaction, 'no_dose_penalty', 5.0) if hasattr(rewards, 'critical_inaction') else 5.0
        penalty_critical_no_count = 2.0
    # Use extinction_penalty from early_termination config for extinction handling everywhere
    big_penalty = rewards.early_termination.extinction_penalty
    
    # Population maintenance (kernel-based)
    pop_maintenance = getattr(rewards, 'population_maintenance', None)
    if pop_maintenance is None:
        kernel_maintenance_enabled = True
        target_population = rewards.population.target_population
        kernel_type = "gaussian"
        kernel_peak_reward = 1.0
        kernel_max_penalty = 0.0
        kernel_zero_distance = 100.0
    else:
        kernel_maintenance_enabled = pop_maintenance.enabled
        target_population = pop_maintenance.target_population
        kernel_type = pop_maintenance.kernel_type
        kernel_peak_reward = pop_maintenance.kernel_peak_reward
        kernel_max_penalty = pop_maintenance.kernel_max_penalty
        kernel_zero_distance = pop_maintenance.kernel_zero_distance
    
    # Survival bonus
    survival_bonus_cfg = rewards.survival_bonus
    
    # Prediction reward
    prediction_cfg = rewards.prediction
    
    # Early termination
    early_term_cfg = rewards.early_termination
    
    # Budget config
    budget_cfg = rewards.budget
    
    logger.log_info("Informed dosing configuration:")
    logger.log_info(f"  - Under-target base penalty: {penalty_informed_dosing_under}")
    logger.log_info(
        f"    · Dose scale/exponent: {penalty_informed_dosing_under_dose_scale} / "
        f"{penalty_informed_dosing_under_dose_exponent}"
    )
    logger.log_info(
        f"    · Deficit scale/cap: {penalty_informed_dosing_under_deficit_scale} / "
        f"{penalty_informed_dosing_under_deficit_cap}"
    )
    logger.log_info(
        f"    · Under-target max penalty: {penalty_informed_dosing_under_max}"
    )
    logger.log_info(f"  - Blind base penalty: {penalty_blind_dose}")
    logger.log_info(
        f"    · Blind dose scale/exponent: {penalty_blind_dose_amount_scale} / "
        f"{penalty_blind_dose_amount_exponent}"
    )
    logger.log_info(f"    · Blind max penalty: {penalty_blind_dose_max}")

    spawn_range = config.environment.initial_bacteria_per_type_range
    if spawn_range is not None:
        logger.log_info(
            f"Initial bacteria per type range set to [{spawn_range[0]}, {spawn_range[1]}]"
        )

    env = PetriEnvWrapper(
        mesa_model_factory=BacteriaModel,
        k_doses=config.environment.k_doses,
        max_steps=config.environment.max_steps,
        
        # Timing and freshness thresholds
        t_count_freshness=t_count_freshness,
        t_seq_freshness=t_seq_freshness,
        max_count_window=max_count_window,
        critical_ratio=critical_ratio,
        t_min_elapsed_time_count=t_min_elapsed_time_count,
        t_max_elapsed_time_count=t_max_elapsed_time_count,
        t_min_elapsed_time_seq=t_min_elapsed_time_seq,
        t_max_elapsed_time_seq=t_max_elapsed_time_seq,
        
        # Action costs and durations
        sequencing_cost=config.actions.sequencing_cost,
        sequencing_duration=config.actions.sequencing_duration,
    noop_cost=config.actions.noop_cost,
        dose_cost=config.actions.dose_cost,
        dose_cost_per_unit=config.actions.dose_cost_per_unit,
        count_cost=config.actions.count_cost,
        sigmoid_scale_factor=config.model.sigmoid_scale_factor,
        
        # Pre-step reward scalars (informed dosing)
        penalty_informed_dosing_under=penalty_informed_dosing_under,
    penalty_informed_dosing_under_dose_scale=penalty_informed_dosing_under_dose_scale,
    penalty_informed_dosing_under_dose_exponent=penalty_informed_dosing_under_dose_exponent,
    penalty_informed_dosing_under_deficit_scale=penalty_informed_dosing_under_deficit_scale,
    penalty_informed_dosing_under_deficit_cap=penalty_informed_dosing_under_deficit_cap,
    penalty_informed_dosing_under_max=penalty_informed_dosing_under_max,
        reward_informed_dosing_above=reward_informed_dosing_above,
        reward_informed_dosing_above_without_seq=reward_informed_dosing_above_without_seq,
        penalty_blind_dose=penalty_blind_dose,
    penalty_blind_dose_amount_scale=penalty_blind_dose_amount_scale,
    penalty_blind_dose_amount_exponent=penalty_blind_dose_amount_exponent,
    penalty_blind_dose_max=penalty_blind_dose_max,
        
        # Pre-step rewards (sequencing)
        seq_already_pending_penalty=seq_already_pending_penalty,
        informative_seq_reward=informative_seq_reward,
        
        # Pre-step rewards (counting)
        informative_count_reward=informative_count_reward,
        cost_weight=config.actions.cost_weight,
        
        # Pre-step rewards (strategic NOOP)
        strategic_noop_reward=strategic_noop_reward,
        
        # Post-step penalties
        penalty_critical_no_dose=penalty_critical_no_dose,
        penalty_critical_no_count=penalty_critical_no_count,
        big_penalty=big_penalty,
        
        # Population maintenance (kernel-based)
        kernel_maintenance_enabled=kernel_maintenance_enabled,
        kernel_type=kernel_type,
        kernel_peak_reward=kernel_peak_reward,
        kernel_max_penalty=kernel_max_penalty,
        kernel_zero_distance=kernel_zero_distance,
        
        # Survival bonus
        survival_bonus_enabled=survival_bonus_cfg.enabled,
        survival_bonus_base=survival_bonus_cfg.base_bonus,
        survival_bonus_scaling_type=survival_bonus_cfg.scaling_type,
        survival_bonus_scaling_factor=survival_bonus_cfg.scaling_factor,
        survival_bonus_max=survival_bonus_cfg.max_bonus,
        
        # Prediction reward
        prediction_reward_enabled=prediction_cfg.enabled,
        prediction_reward_weight=prediction_cfg.weight,
        
        # Early termination
        early_termination_enabled=early_term_cfg.enabled,
        early_termination_penalty=early_term_cfg.penalty,
        early_termination_min_penalty=early_term_cfg.min_penalty,
        early_termination_penalty_decay_power=early_term_cfg.penalty_decay_power,
        early_termination_population_threshold=early_term_cfg.population_threshold,
        early_termination_population_low_threshold=early_term_cfg.population_low_threshold,
        early_termination_extinction_penalty=early_term_cfg.extinction_penalty,
        early_termination_require_budget_depleted=early_term_cfg.require_budget_depleted,
        
        # Environment parameters
        target_population=target_population,
        population_norm=rewards.population.population_norm,
        budget_init=budget_cfg.budget_init,
        budget_norm=budget_cfg.budget_norm,
    initial_bacteria_per_type_range=spawn_range,
        
        # Device config
        device=config.environment.device,
        dtype=config.torch_dtype,
    )
    
    logger.log_info("✓ Environment created with simplified reward structure")
    logger.log_info(f"  - Timing: t_count_freshness={t_count_freshness}, t_seq_freshness={t_seq_freshness}")
    logger.log_info(f"  - Kernel maintenance: {kernel_type} (R={kernel_peak_reward}, M={kernel_max_penalty}, zero_distance={kernel_zero_distance})")
    logger.log_info(f"  - Survival bonus: enabled={survival_bonus_cfg.enabled}")
    logger.log_info(f"  - Early termination: enabled={early_term_cfg.enabled}")
    logger.log_info(f"  - Prediction reward: enabled={prediction_cfg.enabled}")
    
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
        sigmoid_scale_factor=config.model.sigmoid_scale_factor,
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
        device=config.device_type,
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
