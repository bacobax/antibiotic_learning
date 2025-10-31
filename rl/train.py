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
   
   python -m rl.train \\
       --k-doses 3 \\
       --total-updates 100 \\
       --steps-per-rollout 2048 \\
       --device cpu

The wrapper in env_wrapper.py handles all Mesa interaction.
No changes needed to existing simulation code!
"""
import argparse
import json
import os
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
from .env_wrapper import PetriEnvWrapper, build_observation_simple
from .models import RecurrentActorCritic
from .buffer import RolloutBuffer
from .ppo import PPOTrainer


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


def create_mock_env(k_doses: int = 3) -> PetriEnvWrapper:
    """Create mock environment for testing."""
    return PetriEnvWrapper(
        mesa_model_factory=MockMesaModel,
        k_doses=k_doses,
        obs_builder=build_mock_observation,
        scale_dose=lambda x: x * 2.0,  # Scale [0,1] to [0,2]
        max_steps=500,
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
        metrics: Rollout statistics
    """
    obs = env.reset()
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0.0
    current_episode_length = 0
    
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
            current_episode_reward = 0.0
            current_episode_length = 0
            obs = env.reset()
            # Reset hidden state on episode boundary
            h_state = model.init_hidden(batch_size=1, device=cfg.device)
    
    # Compute metrics
    metrics = {
        "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "num_episodes": int(len(episode_rewards)),
    }
    
    return h_state, metrics


def train(cfg: PPOConfig, env: PetriEnvWrapper, save_dir: Path, total_updates: int):
    """
    Main training loop.
    
    Args:
        cfg: PPO configuration
        env: Environment wrapper
        save_dir: Directory to save checkpoints
        total_updates: Number of PPO updates to perform
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
    
    # Progress bar
    iterator = tqdm(range(total_updates), desc="Training") if HAS_TQDM else range(total_updates)
    
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
        
        # Logging
        log_entry = {
            "update": update_idx,
            **rollout_metrics,
            **train_stats,
        }
        log_data.append(log_entry)
        
        # Print stats
        if update_idx % 10 == 0:
            print(f"\n[Update {update_idx}/{total_updates}]")
            print(f"  Episode Reward: {rollout_metrics['mean_episode_reward']:.2f}")
            print(f"  Episode Length: {rollout_metrics['mean_episode_length']:.1f}")
            print(f"  Actor Loss: {train_stats['loss_actor']:.4f}")
            print(f"  Critic Loss: {train_stats['loss_critic']:.4f}")
            print(f"  Entropy: {train_stats['entropy']:.4f}")
            print(f"  Clip Frac: {train_stats['clip_fraction']:.3f}")
        
        # Save checkpoint
        if (update_idx + 1) % 50 == 0 or (update_idx + 1) == total_updates:
            checkpoint_path = save_dir / f"checkpoint_{update_idx+1}.pt"
            torch.save({
                "update": update_idx + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "config": cfg,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final log
    log_path = save_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"Saved training log: {log_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main entrypoint."""
    parser = argparse.ArgumentParser(description="Train Recurrent PPO on bacteria simulation")
    
    # Environment
    parser.add_argument("--k-doses", type=int, default=3, help="Number of antibiotic types")
    parser.add_argument("--mock", action="store_true", help="Use mock environment for testing")
    
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
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Save directory")
    
    args = parser.parse_args()
    
    # Set seed
    set_global_seed(args.seed)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    if args.mock:
        print("Using MOCK environment for smoke testing...")
        env = create_mock_env(k_doses=args.k_doses)
    else:
        print("Using REAL Mesa environment...")
        # Import real Mesa model (only when not mocking)
        try:
            from model import BacteriaModel
            from .env_wrapper import build_observation_simple
            
            def build_real_env():
                return PetriEnvWrapper(
                    mesa_model_factory=BacteriaModel,
                    k_doses=args.k_doses,
                    obs_builder=build_observation_simple,
                    scale_dose=lambda x: x * 2.0,
                    max_steps=1000,
                )
            
            env = build_real_env()
        except ImportError as e:
            print(f"Failed to import Mesa model: {e}")
            print("Falling back to mock environment...")
            env = create_mock_env(k_doses=args.k_doses)
    
    # Infer observation dimension
    obs_dim = env.get_obs_dim()
    print(f"Observation dimension: {obs_dim}")
    
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
    print(f"Saved config: {config_path}")
    
    # Train
    print("\nStarting training...")
    train(cfg, env, save_dir, args.total_updates)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
