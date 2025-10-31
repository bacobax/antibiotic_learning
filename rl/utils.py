"""
Utility functions for PPO training.

Includes GAE advantage computation, normalization, and gradient clipping.
"""
import torch
import torch.nn as nn


def gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Rewards, shape [T, B]
        values: Value estimates, shape [T, B]
        dones: Done flags, shape [T, B]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        advantages: GAE advantages, shape [T, B]
        returns: Value targets (advantages + values), shape [T, B]
    """
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    
    # Bootstrap value for next state (assume 0 if done)
    next_value = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    next_advantage = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    
    # Compute advantages in reverse order
    for t in reversed(range(T)):
        # TD error
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        
        delta = rewards[t] + gamma * next_val * (1.0 - dones[t]) - values[t]
        
        # GAE
        if t == T - 1:
            next_adv = next_advantage
        else:
            next_adv = advantages[t + 1]
        
        advantages[t] = delta + gamma * gae_lambda * (1.0 - dones[t]) * next_adv
    
    # Returns = advantages + values
    returns = advantages + values
    
    return advantages, returns


def normalize(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize tensor to have mean 0 and std 1.
    
    Args:
        t: Input tensor
        eps: Small constant for numerical stability
    
    Returns:
        Normalized tensor
    """
    return (t - t.mean()) / (t.std() + eps)


def clip_grad_norm_(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    """
    Clip gradient norm of parameters.
    
    Args:
        parameters: Iterable of parameters
        max_norm: Maximum gradient norm
        norm_type: Type of norm (default: 2.0 for L2)
    
    Returns:
        Total norm of gradients before clipping
    """
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)
