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


# ============================================================================
# Mock environment for smoke testing
# ============================================================================

class MockMesaModel:
    """Minimal mock of Mesa model for testing without full simulation."""
    
    def __init__(self):
        self.step_count = 0
        self.agent_set = set(range(50))  # Mock population
        self.food_field = np.random.rand(32, 32) * 10.0
        self.antibiotic_fields = {
            "Penicillin": np.zeros((32, 32)),
            "Tetracycline": np.zeros((32, 32)),
            "Ciprofloxacin": np.zeros((32, 32)),
        }
        self.width = 100.0
        self.height = 100.0
    
    def step(self):
        """Advance one simulation step."""
        self.step_count += 1
        # Randomly remove some agents (population dynamics)
        if len(self.agent_set) > 10 and np.random.rand() < 0.1:
            self.agent_set.pop()
        # Decay antibiotics
        for field in self.antibiotic_fields.values():
            field *= 0.95
        # Regenerate food slightly
        self.food_field += np.random.rand(32, 32) * 0.1
    
    def apply_antibiotic(self, antibiotic_type: str, amount: float):
        """Apply antibiotic to field."""
        if antibiotic_type in self.antibiotic_fields:
            self.antibiotic_fields[antibiotic_type] += amount


def build_mock_observation(model: MockMesaModel) -> np.ndarray:
    """Build observation from mock model."""
    obs = []
    obs.append(len(model.agent_set) / 100.0)  # Population
    obs.append(np.mean(model.food_field) / 10.0)  # Food
    # Antibiotic concentrations
    for field in model.antibiotic_fields.values():
        obs.append(np.mean(field))
    obs.append(model.step_count / 1000.0)  # Time
    return np.array(obs, dtype=np.float32)


def create_mock_env(k_doses: int = 3, target_population: int = 500, action_config: Dict = None) -> PetriEnvWrapper:
    """Create mock environment for testing."""
    # Extract parameters from action_config or use defaults
    if action_config is None:
        action_config = get_default_action_config()
    
    env_cfg = action_config.get("environment", {})
    actions_cfg = action_config.get("actions", {})
    
    return PetriEnvWrapper(
        mesa_model_factory=MockMesaModel,
        k_doses=k_doses,
        obs_builder=build_mock_observation,
        scale_dose=lambda x: x * 2.0,  # Scale [0,1] to [0,2]
        max_steps=env_cfg.get("max_steps", 500),
        target_population=target_population or env_cfg.get("target_population", 500),
        sequencing_cost=actions_cfg.get("sequencing", {}).get("cost", 1.0),
        sequencing_duration=actions_cfg.get("sequencing", {}).get("duration", 5),
        dose_cost_per_unit=actions_cfg.get("dose", {}).get("cost", 0.2),
        budget_init=env_cfg.get("budget_init", 100.0),
        w_pop=env_cfg.get("w_pop", 1.0),
        w_genome=env_cfg.get("w_genome", 0.5),
        w_cost=env_cfg.get("w_cost", 0.05),
    )


# ============================================================================
# Training loop
# ============================================================================

def rollout(
    env: PetriEnvWrapper,
    model: RecurrentActorCritic,
    buffer: RolloutBuffer,
    num_steps: int,
    h_state: torch.Tensor,
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
    from .env_wrapper import ACTION_DOSE
    
    obs = env.reset()
    episode_rewards = []
    episode_lengths = []
    episode_populations = []  # Track population at end of each episode
    current_episode_reward = 0.0
    current_episode_length = 0
    dose_action_count = 0  # Track number of DOSE actions
    total_actions = 0  # Track total actions
    
    model.eval()
    
    for step in range(num_steps):
        # Prepare observation
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(cfg.device)  # [1, obs_dim]
        
        # Get action from policy
        with torch.no_grad():
            action_dict = model.act(
                obs_tensor, h_state, 
                dose_action_index=cfg.dose_action_index,
                deterministic=False,
            )
        
        # Extract actions
        a_disc = action_dict["a_disc"].cpu().numpy()[0]
        a_cont = action_dict["a_cont"].cpu().numpy()[0]
        logp_disc = action_dict["logp_disc"]
        logp_cont = action_dict["logp_cont"]
        value = action_dict["value"]
        h_next = action_dict["h_next"]
        
        # Track dose actions
        if a_disc == ACTION_DOSE:
            dose_action_count += 1
        total_actions += 1
        
        # Environment step
        next_obs, reward, done, info = env.step(a_disc, a_cont)
        
        # Store in buffer
        buffer.add(
            obs=obs_tensor.cpu(),
            a_disc=action_dict["a_disc"].cpu(),
            a_cont=action_dict["a_cont"].cpu(),
            logp_disc=logp_disc.cpu(),
            logp_cont=logp_cont.cpu(),
            value=value.cpu(),
            reward=torch.tensor([reward], dtype=torch.float32),
            done=torch.tensor([float(done)], dtype=torch.float32),
            h_in=h_state.cpu(),
        )
        
        # Update state
        obs = next_obs
        h_state = h_next
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
            h_state = model.init_hidden(batch_size=1, device=cfg.device)
    
    # Compute metrics
    dose_action_percentage = (dose_action_count / total_actions * 100) if total_actions > 0 else 0.0
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
    
    return h_state, metrics


def train(cfg: PPOConfig, env: PetriEnvWrapper, save_dir: Path, total_updates: int, logger: TrainingLogger):
    """
    Main training loop with comprehensive logging and TensorBoard integration.
    
    Args:
        cfg: PPO configuration
        env: Environment wrapper
        save_dir: Directory to save checkpoints
        total_updates: Number of PPO updates to perform
        logger: TrainingLogger instance for all logging
    """
    # Create model
    model = RecurrentActorCritic(
        obs_dim=cfg.obs_dim,
        n_discrete=cfg.n_discrete,
        k_doses=cfg.k_doses,
        hidden_dim=cfg.hidden_dim,
        rnn_layers=cfg.rnn_layers,
    )
    
    # Create trainer
    trainer = PPOTrainer(model, cfg)
    
    # Training log
    log_data = []
    
    # Initialize hidden state
    h_state = model.init_hidden(batch_size=1, device=cfg.device)
    
    logger.log_info(f"Starting PPO training for {total_updates} updates")
    logger.log_info(f"Config: obs_dim={cfg.obs_dim}, hidden_dim={cfg.hidden_dim}, "
                f"rnn_layers={cfg.rnn_layers}, k_doses={cfg.k_doses}")
    logger.log_info(f"Hyperparams: lr={cfg.lr}, gamma={cfg.gamma}, gae_lambda={cfg.gae_lambda}")
    logger.log_info(f"Rollout steps per update: {cfg.rollout_steps}, PPO epochs: {cfg.epochs}")
    logger.log_debug(f"Using device: {cfg.device}")
    
    # Progress bar
    iterator = tqdm(range(total_updates), desc="Training") if HAS_TQDM else range(total_updates)
    
    # Tracking for convergence monitoring
    reward_history = []
    loss_history = []
    start_time = time.time()
    
    for update_idx in iterator:
        # Collect rollout
        buffer = RolloutBuffer()
        h_state, rollout_metrics = rollout(
            env, model, buffer, cfg.rollout_steps, h_state, cfg
        )
        
        # PPO update
        model.train()
        data = buffer.stacked()
        train_stats = trainer.update(data)
        
        # Combine metrics
        log_entry = {
            "update": update_idx,
            **rollout_metrics,
            **train_stats,
        }
        log_data.append(log_entry)
        
        # Track metrics
        reward_history.append(rollout_metrics['mean_episode_reward'])
        loss_history.append(train_stats['loss_actor'])
        
        # Log metrics to all backends (TensorBoard, JSON, etc.)
        logger.log_metrics(update_idx, rollout_metrics, train_stats)
        
        # Periodic console output and diagnostic checks
        if update_idx % 10 == 0 and update_idx > 0:
            elapsed = time.time() - start_time
            logger.log_update(update_idx, total_updates, rollout_metrics, 
                            train_stats, elapsed)
        
        if np.isnan(train_stats['loss_actor']):
            break
        
        # Save checkpoint every 50 updates
        if (update_idx + 1) % 50 == 0:
            checkpoint_path = save_dir / f"checkpoint_{update_idx+1}.pt"
            torch.save({
                "update": update_idx + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "config": cfg,
            }, checkpoint_path)
            logger.log_debug(f"Saved checkpoint: {checkpoint_path}")
        
        # Save final checkpoint
        if (update_idx + 1) == total_updates:
            checkpoint_path = save_dir / f"checkpoint_final_{update_idx+1}.pt"
            torch.save({
                "update": update_idx + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "config": cfg,
            }, checkpoint_path)
            logger.log_info(f"Saved final checkpoint: {checkpoint_path}")
    
    # Save final log
    log_path = save_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    logger.log_debug(f"Saved training log: {log_path}")
    
    # Training summary
    total_time = time.time() - start_time
    logger.log_summary(total_updates, total_time, reward_history, loss_history)
    
    logger.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main entrypoint."""
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
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load action configuration
    try:
        action_config = load_action_config(args.actions_config)
        config_source = args.actions_config if args.actions_config else "default"
    except Exception as e:
        print(f"Warning: Failed to load action config: {e}")
        action_config = get_default_action_config()
        config_source = "default (fallback)"
    
    # Setup training logger (handles all logging: Python logging, TensorBoard, JSON metrics)
    logger = TrainingLogger(save_dir, experiment_name=args.experiment_name)
    
    # Extract environment parameters from config
    env_cfg = action_config.get("environment", {})
    actions_cfg = action_config.get("actions", {})
    
    # Log startup info
    logger.log_info("="*70)
    logger.log_info("PPO Training Started")
    logger.log_info("="*70)
    logger.log_info(f"Action config: {config_source}")
    cmd_args = [f"--{k.replace('_', '-')} {v}" for k, v in vars(args).items()]
    logger.log_info(f"Command: python -m rl.train {' '.join(cmd_args)}")
    logger.log_info(f"TensorBoard: tensorboard --logdir={save_dir / args.experiment_name} --port=6006")
    
    # Log action costs from config
    logger.log_info(f"Action Costs (from config):")
    logger.log_info(f"  - Sequencing: {actions_cfg.get('sequencing', {}).get('cost', 1.0)}")
    logger.log_info(f"  - Dose: {actions_cfg.get('dose', {}).get('cost', 0.2)} per unit")
    logger.log_info(f"Environment Settings (from config):")
    logger.log_info(f"  - Target population: {env_cfg.get('target_population', 500)}")
    logger.log_info(f"  - Initial budget: {env_cfg.get('budget_init', 100.0)}")
    logger.log_info(f"  - Reward weights: pop={env_cfg.get('w_pop', 1.0)}, genome={env_cfg.get('w_genome', 0.5)}, cost={env_cfg.get('w_cost', 0.05)}")
    
    # Set seed
    set_global_seed(args.seed)
    logger.log_debug(f"Random seed set to: {args.seed}")
    
    # Create environment
    if args.mock:
        logger.log_info("Using MOCK environment for smoke testing")
        env = create_mock_env(
            k_doses=args.k_doses,
            target_population=args.target_population,
            action_config=action_config
        )
    else:
        logger.log_info("Using REAL Mesa environment")
        # Import real Mesa model (only when not mocking)
        try:
            from model import BacteriaModel
            
            def build_real_env():
                return PetriEnvWrapper(
                    mesa_model_factory=BacteriaModel,
                    k_doses=args.k_doses,
                    scale_dose=lambda x: x * 2.0,
                    # Extract from config
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
            
            env = build_real_env()
            logger.log_debug("Successfully loaded BacteriaModel")
        except ImportError as e:
            logger.log_error(f"Failed to import Mesa model: {e}")
            logger.log_warning("Falling back to mock environment")
            env = create_mock_env(
                k_doses=args.k_doses,
                target_population=args.target_population,
                action_config=action_config
            )
    
    # Infer observation dimension
    obs_dim = env.get_obs_dim()
    logger.log_info(f"Observation dimension: {obs_dim}")
    
    # Build config
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
    )
    
    # Save config
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
    
    # Train
    logger.log_info("="*70)
    train(cfg, env, save_dir, args.total_updates, logger)
    logger.log_info("="*70)
    logger.log_info("Training complete!")
    logger.log_info(f"Logs: {save_dir / 'training.log'}")
    logger.log_info(f"Metrics: {save_dir / 'metrics.json'}")
    logger.log_info(f"TensorBoard: tensorboard --logdir={save_dir / args.experiment_name} --port=6006")


if __name__ == "__main__":
    main()
