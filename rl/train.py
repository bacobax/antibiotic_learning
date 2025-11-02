"""
Training entrypoint for Recurrent PPO on bacteria simulation.

INTEGRATION GUIDE:
==================

To integrate with the existing Mesa bacteria simulation:

1. Create a model factory:
   
   def build_mesa_model():
       from model import BacteriaModel
       return BacteriaModel()

2. Create an observation builder:
   
   def build_observation(model):
       import numpy as np
       # Extract relevant state
       population = len(model.agent_set)
       avg_traits = ...  # compute from bacteria
       food_level = np.sum(model.food_field)
       antibiotic_conc = ...
       # Return flat numpy array
       return np.array([...], dtype=np.float32)

3. Run training:

   python -m rl.train --k-doses 3 --total-updates 100 --steps-per-rollout 2048 --device cpu

4. View TensorBoard during/after training:

   tensorboard --logdir=./checkpoints --port=6006

The wrapper in env_wrapper.py handles all Mesa interaction.
No changes needed to existing simulation code!
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict
from .agent import RLAgent

import numpy as np
import torch

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .config import PPOConfig, set_global_seed
from .env_wrapper import PetriEnvWrapper
from .models import RecurrentActorCritic
from .buffer import RolloutBuffer
from .ppo import PPOTrainer
from .logger import TrainingLogger
from .action_config import load_action_config, get_default_action_config
from model import BacteriaModel
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


# ============================================================================
# CLI
# ============================================================================

# ============================================================================
# CLI Setup helpers
# ============================================================================

def _load_configs(args_actions_config: str) -> tuple[dict, str]:
    """
    Load action configuration from file or defaults.
    
    Args:
        args_actions_config: Path to action config file or None
    
    Returns:
        Tuple of (action_config dict, config_source string)
    """
    try:
        action_config = load_action_config(args_actions_config)
        config_source = args_actions_config if args_actions_config else "default"
    except Exception as e:
        print(f"Warning: Failed to load action config: {e}")
        action_config = get_default_action_config()
        config_source = "default (fallback)"
    return action_config, config_source


def _setup_logger_and_log_startup(
    save_dir: Path,
    experiment_name: str,
    args,
    action_config: dict,
    config_source: str,
) -> TrainingLogger:
    """
    Initialize logger and log startup information.
    
    Args:
        save_dir: Directory to save logs
        experiment_name: Experiment name for TensorBoard
        args: Parsed command line arguments
        action_config: Action configuration dict
        config_source: Source of action config
    
    Returns:
        Initialized TrainingLogger
    """
    logger = TrainingLogger(save_dir, experiment_name=experiment_name)
    env_cfg = action_config.get("environment", {})
    actions_cfg = action_config.get("actions", {})
    
    logger.log_info("="*70)
    logger.log_info("PPO Training Started")
    logger.log_info("="*70)
    logger.log_info(f"Action config: {config_source}")
    cmd_args = [f"--{k.replace('_', '-')} {v}" for k, v in vars(args).items()]
    logger.log_info(f"Command: python -m rl.train {' '.join(cmd_args)}")
    logger.log_info(f"TensorBoard: tensorboard --logdir={save_dir / experiment_name} --port=6006")
    
    logger.log_info(f"Action Costs (from config):")
    logger.log_info(f"  - Sequencing: {actions_cfg.get('sequencing', {}).get('cost', 1.0)}")
    logger.log_info(f"  - Dose: {actions_cfg.get('dose', {}).get('cost', 0.2)} per unit")
    logger.log_info(f"Environment Settings (from config):")
    logger.log_info(f"  - Target population: {env_cfg.get('target_population', 500)}")
    logger.log_info(f"  - Initial budget: {env_cfg.get('budget_init', 100.0)}")
    logger.log_info(f"  - Reward weights: pop={env_cfg.get('w_pop', 1.0)}, genome={env_cfg.get('w_genome', 0.5)}, cost={env_cfg.get('w_cost', 0.05)}")
    
    return logger


def _create_environment(
    args,
    action_config: dict,
    logger: TrainingLogger,
) -> PetriEnvWrapper:
    """
    Create and initialize the environment.
    
    Args:
        args: Parsed command line arguments
        action_config: Action configuration dict
        logger: Training logger
    
    Returns:
        Initialized PetriEnvWrapper
    """
    env_cfg = action_config.get("environment", {})
    actions_cfg = action_config.get("actions", {})
    
    logger.log_info("Using REAL Mesa environment")
    env = PetriEnvWrapper(
        mesa_model_factory=BacteriaModel,
        k_doses=args.k_doses,
        scale_dose=lambda x: x / 2 / args.k_doses,
        max_steps=env_cfg.get("max_steps", 1000),
        target_population=args.target_population or env_cfg.get("target_population", 500),
        sequencing_cost=actions_cfg.get("sequencing", {}).get("cost", 1.0),
        sequencing_duration=actions_cfg.get("sequencing", {}).get("duration", 5),
        dose_cost_per_unit=actions_cfg.get("dose", {}).get("cost", 0.2),
        budget_init=env_cfg.get("budget_init", 100.0),
        w_pop=env_cfg.get("w_pop", 1.0),
        w_genome=env_cfg.get("w_genome", 0.5),
        w_cost=env_cfg.get("w_cost", 0.05),
    )
    logger.log_debug("Successfully loaded BacteriaModel")
    return env


def _build_ppo_config(env: PetriEnvWrapper, args) -> PPOConfig:
    """
    Build PPO configuration from arguments and environment.
    
    Args:
        env: Initialized environment
        args: Parsed command line arguments
    
    Returns:
        Initialized PPOConfig
    """
    obs_dim = env.get_obs_dim()
    cfg = PPOConfig(
        obs_dim=obs_dim,
        n_discrete=4,
        k_doses=args.k_doses,
        hidden_dim=args.hidden_dim,
        rnn_layers=args.rnn_layers,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        seq_len=args.seq_len,
        rollout_steps=args.steps_per_rollout,
        epochs=args.epochs,
        batch_seq_len=args.seq_len,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        dose_action_index=ACTION_DOSE
    )
    return cfg


def _save_configs(
    save_dir: Path,
    args,
    action_config: dict,
    logger: TrainingLogger,
) -> None:
    """
    Save training and action configs to disk.
    
    Args:
        save_dir: Directory to save to
        args: Parsed command line arguments
        action_config: Action configuration dict
        logger: Training logger
    """
    # Save training config
    config_path = save_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.log_debug(f"Saved config to: {config_path}")
    
    # Save action config
    action_config_path = save_dir / "actions_config.yaml"
    try:
        import yaml
        with open(action_config_path, "w") as f:
            yaml.dump(action_config, f, default_flow_style=False, sort_keys=False)
        logger.log_debug(f"Saved action config to: {action_config_path}")
    except ImportError:
        logger.log_debug("PyYAML not available, skipping action config save")


def main():
    """Main entrypoint."""
    # ========================================================================
    # PARSE ARGUMENTS
    # ========================================================================
    parser = argparse.ArgumentParser(description="Train Recurrent PPO on bacteria simulation")
    
    # Environment
    parser.add_argument("--k-doses", type=int, default=3, help="Number of antibiotic types")
    parser.add_argument("--mock", action="store_true", help="Use mock environment for testing")
    parser.add_argument("--target-population", type=int, default=500, help="Target bacteria population for reward shaping")
    parser.add_argument("--actions-config", type=str, default=None, help="Path to YAML file defining actions and costs")
    
    # Training
    parser.add_argument("--total-updates", type=int, default=100, help="Total PPO updates")
    parser.add_argument("--steps-per-rollout", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--seq-len", type=int, default=64, help="Truncated BPTT length")
    parser.add_argument("--epochs", type=int, default=4, help="PPO epochs per update")
    
    # Model
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--rnn-layers", type=int, default=1, help="Number of RNN layers")
    
    # Optimization
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    
    # System
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Save directory")
    parser.add_argument("--experiment-name", type=str, default="ppo_training", help="Experiment name for TensorBoard")
    
    args = parser.parse_args()
    
    # ========================================================================
    # SETUP
    # ========================================================================
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configs
    action_config, config_source = _load_configs(args.actions_config)
    
    # Setup logger
    logger = _setup_logger_and_log_startup(save_dir, args.experiment_name, args, action_config, config_source)
    
    # Set seed
    set_global_seed(args.seed)
    logger.log_debug(f"Random seed set to: {args.seed}")
    
    # Create environment
    env = _create_environment(args, action_config, logger)
    
    # Build configuration
    cfg = _build_ppo_config(env, args)
    logger.log_info(f"Observation dimension: {cfg.obs_dim}")
    
    # Save configs
    _save_configs(save_dir, args, action_config, logger)
    
    # ========================================================================
    # TRAIN
    # ========================================================================
    logger.log_info("="*70)
    train(cfg, env, save_dir, args.total_updates, logger)
    logger.log_info("="*70)
    logger.log_info("Training complete!")
    logger.log_info(f"Logs: {save_dir / 'training.log'}")
    logger.log_info(f"Metrics: {save_dir / 'metrics.json'}")
    logger.log_info(f"TensorBoard: tensorboard --logdir={save_dir / args.experiment_name} --port=6006")


if __name__ == "__main__":
    main()
