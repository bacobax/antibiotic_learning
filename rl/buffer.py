"""
Rollout buffer for storing trajectory data.

Maintains temporal ordering for recurrent PPO training.
"""
from typing import Dict, List

import torch


class RolloutBuffer:
    """
    Buffer for storing rollout trajectories.
    
    Stores per-timestep tensors in temporal order for recurrent PPO.
    Does NOT shuffle data to preserve sequential structure.
    
    Stores:
        obs: Observations [T, B, obs_dim]
        a_disc: Discrete actions [T, B]
        a_cont: Continuous doses [T, B, K]
        logp_disc: Discrete log-probs [T, B]
        logp_cont: Continuous log-probs [T, B]
        value: Value estimates [T, B]
        reward: Rewards [T, B]
        done: Done flags [T, B]
        h_in: Initial hidden states [T, layers, B, hidden_dim]
    """
    
    def __init__(self):
        """Initialize empty buffer."""
        self.obs: List[torch.Tensor] = []
        self.a_disc: List[torch.Tensor] = []
        self.a_cont: List[torch.Tensor] = []
        self.logp_disc: List[torch.Tensor] = []
        self.logp_cont: List[torch.Tensor] = []
        self.value: List[torch.Tensor] = []
        self.reward: List[torch.Tensor] = []
        self.done: List[torch.Tensor] = []
        self.h_in: List[torch.Tensor] = []
    
    def add(
        self,
        obs: torch.Tensor,
        a_disc: torch.Tensor,
        a_cont: torch.Tensor,
        logp_disc: torch.Tensor,
        logp_cont: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        h_in: torch.Tensor,
    ) -> None:
        """
        Add one timestep of data.
        
        All tensors should have batch dimension [B, ...].
        
        Args:
            obs: Observations, shape [B, obs_dim]
            a_disc: Discrete actions, shape [B]
            a_cont: Continuous doses, shape [B, K]
            logp_disc: Discrete log-probs, shape [B]
            logp_cont: Continuous log-probs, shape [B]
            value: Value estimates, shape [B]
            reward: Rewards, shape [B]
            done: Done flags, shape [B]
            h_in: Hidden states at step start, shape [layers, B, hidden_dim]
        """
        self.obs.append(obs.cpu())
        self.a_disc.append(a_disc.cpu())
        self.a_cont.append(a_cont.cpu())
        self.logp_disc.append(logp_disc.cpu())
        self.logp_cont.append(logp_cont.cpu())
        self.value.append(value.cpu())
        self.reward.append(reward.cpu())
        self.done.append(done.cpu())
        self.h_in.append(h_in.cpu())
    
    def stacked(self) -> Dict[str, torch.Tensor]:
        """
        Stack all buffered data into tensors.
        
        Returns:
            Dictionary with keys:
                obs: [T, B, obs_dim]
                a_disc: [T, B]
                a_cont: [T, B, K]
                logp_disc: [T, B]
                logp_cont: [T, B]
                value: [T, B]
                reward: [T, B]
                done: [T, B]
                h_in: [T, layers, B, hidden_dim]
        """
        if len(self.obs) == 0:
            raise ValueError("Buffer is empty")
        
        return {
            "obs": torch.stack(self.obs, dim=0),  # [T, B, obs_dim]
            "a_disc": torch.stack(self.a_disc, dim=0),  # [T, B]
            "a_cont": torch.stack(self.a_cont, dim=0),  # [T, B, K]
            "logp_disc": torch.stack(self.logp_disc, dim=0),  # [T, B]
            "logp_cont": torch.stack(self.logp_cont, dim=0),  # [T, B]
            "value": torch.stack(self.value, dim=0),  # [T, B]
            "reward": torch.stack(self.reward, dim=0),  # [T, B]
            "done": torch.stack(self.done, dim=0),  # [T, B]
            "h_in": torch.stack(self.h_in, dim=0),  # [T, layers, B, hidden_dim]
        }
    
    def clear(self) -> None:
        """Clear all stored data."""
        self.obs.clear()
        self.a_disc.clear()
        self.a_cont.clear()
        self.logp_disc.clear()
        self.logp_cont.clear()
        self.value.clear()
        self.reward.clear()
        self.done.clear()
        self.h_in.clear()
    
    def __len__(self) -> int:
        """Return number of timesteps stored."""
        return len(self.obs)
