"""
PPO Trainer with recurrent policy and hybrid action space.

Implements truncated BPTT for sequential training.
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from .training_config import PPOConfig
from .models import RecurrentActorCritic
from .utils import gae_advantages, normalize, clip_grad_norm_


class PPOTrainer:
    """
    Recurrent PPO trainer with hybrid action space.
    
    Implements:
        - Truncated BPTT over sequences
        - Hybrid action space (discrete + continuous)
        - GAE for advantage estimation
        - PPO clipped objective
    """
    
    def __init__(self, model: RecurrentActorCritic, cfg: PPOConfig):
        """
        Initialize PPO trainer.
        
        Args:
            model: RecurrentActorCritic model
            cfg: PPO configuration
        """
        self.model = model
        self.cfg = cfg
        self.optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
        
        # Move model to device
        self.model.to(cfg.device)
    
    def update(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform PPO update on rollout data.
        
        Args:
            data: Dictionary containing rollout tensors with shapes:
                obs: [T, B, obs_dim]
                a_disc: [T, B]
                a_cont: [T, B, K]
                logp_disc: [T, B]
                logp_cont: [T, B]
                value: [T, B]
                reward: [T, B]
                done: [T, B]
                h_in: [T, layers, B, hidden_dim]
                pred_next_pop: [T, B]
                population_counted_norm: [T, B]
                count_mask: [T, B]
        
        Returns:
            Dictionary with training statistics
        """
        # Move data to device
        obs = data["obs"].to(self.cfg.device)
        a_disc = data["a_disc"].to(self.cfg.device)
        a_cont = data["a_cont"].to(self.cfg.device)
        old_logp_disc = data["logp_disc"].to(self.cfg.device)
        old_logp_cont = data["logp_cont"].to(self.cfg.device)
        values = data["value"].to(self.cfg.device)
        rewards = data["reward"].to(self.cfg.device)
        dones = data["done"].to(self.cfg.device)
        h_in = data["h_in"].to(self.cfg.device)
        pred_next_pop = data["pred_next_pop"].to(self.cfg.device)
        population_counted_norm = data["population_counted_norm"].to(self.cfg.device)
        count_mask = data["count_mask"].to(self.cfg.device)
        
        T, B = obs.shape[:2]
        
        # Compute advantages and returns using GAE
        with torch.no_grad():
            advantages, returns = gae_advantages(
                rewards, values, dones,
                gamma=self.cfg.gamma,
                gae_lambda=self.cfg.gae_lambda,
            )
            # Normalize advantages
            advantages = normalize(advantages)
        
        # Training statistics
        stats = {
            "loss_total": 0.0,
            "loss_actor": 0.0,
            "loss_critic": 0.0,
            "loss_pred": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            "value_mean": 0.0,
            "advantage_mean": 0.0,
            "num_updates": 0,
        }
        
        # Multiple epochs over the same data
        for epoch in range(self.cfg.epochs):
            # Process data in sequential chunks (truncated BPTT)
            num_chunks = max(1, T // self.cfg.seq_len)
            
            for chunk_idx in range(num_chunks):
                t_start = chunk_idx * self.cfg.seq_len
                t_end = min(t_start + self.cfg.seq_len, T)
                
                if t_end - t_start < 2:  # Skip very short chunks
                    continue
                
                # Extract chunk
                obs_chunk = obs[t_start:t_end]  # [seq_len, B, obs_dim]
                a_disc_chunk = a_disc[t_start:t_end]
                a_cont_chunk = a_cont[t_start:t_end]
                old_logp_disc_chunk = old_logp_disc[t_start:t_end]
                old_logp_cont_chunk = old_logp_cont[t_start:t_end]
                advantages_chunk = advantages[t_start:t_end]
                returns_chunk = returns[t_start:t_end]
                h_init = h_in[t_start]  # [layers, B, hidden_dim]
                population_counted_norm_chunk = population_counted_norm[t_start:t_end]
                count_mask_chunk = count_mask[t_start:t_end]
                
                # Evaluate actions with current policy
                eval_dict = self.model.evaluate_actions(
                    obs_chunk, h_init, a_disc_chunk, a_cont_chunk
                )
                
                new_logp_disc = eval_dict["logp_disc"]
                new_logp_cont = eval_dict["logp_cont"]
                new_values = eval_dict["value"]
                new_pred_next_pop = eval_dict["pred_next_pop"]
                entropy_disc = eval_dict["entropy_disc"]
                entropy_cont = eval_dict["entropy_cont"]
                
                # Masked prediction loss (only on COUNT steps)
                pred_error = (new_pred_next_pop - population_counted_norm_chunk) ** 2
                pred_loss = (pred_error * count_mask_chunk).mean()
                
                # Compute PPO ratios
                ratio_disc = torch.exp(new_logp_disc - old_logp_disc_chunk)
                ratio_cont = torch.exp(new_logp_cont - old_logp_cont_chunk)
                
                # Joint ratio: multiply discrete and continuous ratios
                # Only apply continuous ratio where action is DOSE
                is_dose = (a_disc_chunk == self.cfg.dose_action_index).float()
                joint_ratio = ratio_disc * torch.exp(is_dose * torch.log(ratio_cont + 1e-8))
                
                # PPO clipped objective
                surr1 = joint_ratio * advantages_chunk
                surr2 = torch.clamp(
                    joint_ratio, 
                    1.0 - self.cfg.clip_eps, 
                    1.0 + self.cfg.clip_eps
                ) * advantages_chunk
                
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE)
                critic_loss = 0.5 * ((new_values - returns_chunk) ** 2).mean()
                
                # Entropy bonus (higher entropy = more exploration)
                entropy = (entropy_disc + entropy_cont).mean()
                
                # Total loss
                total_loss = (
                    actor_loss 
                    + self.cfg.vf_coef * critic_loss
                    + pred_loss
                    - self.cfg.ent_coef * entropy
                )
                
                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = clip_grad_norm_(
                    self.model.parameters(), 
                    self.cfg.max_grad_norm
                )
                self.optimizer.step()
                
                # Clip fraction (diagnostic)
                with torch.no_grad():
                    clip_frac = ((joint_ratio - 1.0).abs() > self.cfg.clip_eps).float().mean()
                
                # Accumulate stats (ensure all are Python floats, not tensors)
                stats["loss_total"] += float(total_loss.item())
                stats["loss_actor"] += float(actor_loss.item())
                stats["loss_critic"] += float(critic_loss.item())
                stats["loss_pred"] += float(pred_loss.item())
                stats["entropy"] += float(entropy.item())
                stats["clip_fraction"] += float(clip_frac.item())
                stats["grad_norm"] += float(grad_norm)
                stats["value_mean"] += float(new_values.mean().item())
                stats["advantage_mean"] += float(advantages_chunk.mean().item())
                stats["num_updates"] += 1
        
        # Average stats
        if stats["num_updates"] > 0:
            for key in stats:
                if key != "num_updates":
                    stats[key] /= stats["num_updates"]
        
        return stats
