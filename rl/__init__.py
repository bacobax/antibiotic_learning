"""
Recurrent PPO with hybrid action space for bacteria antibiotic learning.

This package provides a complete RL training pipeline that wraps the existing
Mesa bacteria simulation without modifying it.

Modules:
  config: Configuration dataclasses for PPO
  config_loader: YAML configuration loading from rl/configs/ folder
  env_wrapper: Environment wrapper around Mesa simulation
  models: Neural network architectures
  buffer: Rollout buffer for trajectory collection
  ppo: PPO trainer
  agent: RL agent combining model and trainer
  reward: Reward computation modules
  logger: Training logger
  action_config: (deprecated) Use config_loader instead
"""

from .config import PPOConfig, set_global_seed
from .env_wrapper import PetriEnvWrapper
from .models import RecurrentActorCritic
from .buffer import RolloutBuffer
from .ppo import PPOTrainer
from .utils import gae_advantages, normalize, clip_grad_norm_

__all__ = [
    "PPOConfig",
    "PetriEnvWrapper",
    "RecurrentActorCritic",
    "RolloutBuffer",
    "PPOTrainer",
    "gae_advantages",
    "normalize",
    "clip_grad_norm_",
    "set_global_seed",
]
