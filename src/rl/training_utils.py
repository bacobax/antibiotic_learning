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
    current_episode_reward = 0.0
    current_episode_length = 0
    dose_action_count = 0  # Track number of DOSE actions
    sequencing_action_count = 0  # Track number of SEQUENCING actions
    count_action_count = 0  # Track number of COUNT actions
    noop_action_count = 0  # Track number of NOOP actions

    total_actions = 0  # Track total actions
    
    agent.start_episode()
    
    for step in range(num_steps):
        # Prepare observation
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(cfg.device)  # [1, obs_dim]
        
        # Get action from policy
        with torch.no_grad():
            (
                a_disc, 
                a_cont, 
                logp_disc, 
                logp_cont, 
                value, 
                h_prev 
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
        
        # Environment step
        next_obs, reward, done, info = env.step(pure_a_disc, pure_a_cont)
        
        # Store in buffer
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
        )
        
        # Update state
        obs = next_obs
        current_episode_reward += reward
        current_episode_length += 1
        
        # Handle episode termination
        if done:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            # Log population at end of episode
            final_population = env.get_bacteria_population()
            episode_populations.append(final_population)
            current_episode_reward = 0.0
            current_episode_length = 0
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
    metrics = {
        "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_episode_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        "max_episode_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "min_episode_reward": float(np.min(episode_rewards)) if episode_rewards else 0.0,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "num_episodes": int(len(episode_rewards)),
        "mean_population_per_episode": float(np.mean(episode_populations)) if episode_populations else 0.0,
        "final_population": float(episode_populations[-1]) if episode_populations else 0.0,
        "dose_action_percentage": float(dose_action_percentage),
    }
    
    return metrics


def _initialize_agent(cfg: PPOConfig) -> RLAgent:
    """
    Create and initialize agent with model and trainer.
    
    Args:
        cfg: PPO configuration
    
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

    agent = RLAgent(model,cfg.device).with_trainer(cfg)
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
    agent = _initialize_agent(cfg)
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
    
    logger.log_info(f"Environment Settings:")
    logger.log_info(f"  - Max steps: {config.environment.max_steps}")
    logger.log_info(f"  - Target population: {config.environment.target_population}")
    logger.log_info(f"  - Budget: {config.environment.budget_init}")
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
    logger.log_info(f"  - Population: {config.environment.w_pop}")
    logger.log_info(f"  - Genome: {config.environment.w_genome}")
    logger.log_info(f"  - Cost: {config.environment.w_cost}")
    logger.log_info(f"  - Population maintenance: {config.environment.w_population_maintenance}")
    
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
    
    env = PetriEnvWrapper(
        mesa_model_factory=BacteriaModel,
        k_doses=config.environment.k_doses,
        scale_dose=lambda x: x / 2 / config.environment.k_doses,
        max_steps=config.environment.max_steps,
        target_population=config.environment.target_population,
        sequencing_cost=config.actions.sequencing_cost,
        sequencing_duration=config.actions.sequencing_duration,
        dose_cost_per_unit=config.actions.dose_cost_per_unit,
        budget_init=config.environment.budget_init,
        budget_norm=config.environment.budget_norm,
        population_norm=config.environment.population_norm,
        w_pop=config.environment.w_pop,
        w_genome=config.environment.w_genome,
        w_cost=config.environment.w_cost,
        w_population_maintenance=config.environment.w_population_maintenance,
        device=config.environment.device,
        dtype=config.torch_dtype,
    )
    
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
