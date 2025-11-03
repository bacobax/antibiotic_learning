"""
Configuration dataclass for Recurrent PPO.
"""
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


@dataclass
class PPOConfig:
    """
    Configuration for Recurrent PPO with hybrid action space.
    
    Attributes:
        obs_dim: Observation space dimensionality
        n_discrete: Number of discrete actions (4: NOOP, COUNT_BACTERIA, SEQUENCING, DOSE)
        k_doses: Number of antibiotic types (continuous dose dimension)
        hidden_dim: Hidden dimension for GRU/LSTM
        rnn_layers: Number of recurrent layers
        gamma: Discount factor for returns
        gae_lambda: GAE lambda parameter for advantage estimation
        clip_eps: PPO clipping epsilon
        vf_coef: Value function loss coefficient
        ent_coef: Entropy bonus coefficient
        max_grad_norm: Maximum gradient norm for clipping
        seq_len: Truncated BPTT sequence length
        rollout_steps: Total steps per rollout before update
        epochs: Number of epochs per PPO update
        batch_seq_len: Sequence length for minibatches (typically same as seq_len)
        lr: Learning rate for Adam optimizer
        device: Device to run on ("cuda" or "cpu")
        dose_action_index: Index of DOSE action in discrete action set (default=3)
        seed: Random seed for reproducibility
    """
    
    obs_dim: int
    n_discrete: int = 4  # NOOP, COUNT_BACTERIA, SEQUENCING, DOSE
    k_doses: int = 3  # Number of antibiotic types
    
    # Architecture
    hidden_dim: int = 256
    rnn_layers: int = 1
    
    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0
    
    # Training
    seq_len: int = 64  # Truncated BPTT length
    rollout_steps: int = 2048
    epochs: int = 4
    batch_seq_len: int = 64  # Same as seq_len, kept separate for flexibility
    lr: float = 3e-4
    
    # Device
    device: str = "cpu"
    
    # Action space
    dose_action_index: int = 3  # Index of DOSE in discrete action set
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.n_discrete == 4, "Expected 4 discrete actions"
        assert self.dose_action_index < self.n_discrete, "dose_action_index out of range"
        assert self.k_doses > 0, "Must have at least one antibiotic type"
        assert self.obs_dim > 0, "obs_dim must be positive"
        assert self.device in ["cpu", "cuda", "mps"], "device must be 'cpu' or 'cuda'"


def set_global_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
