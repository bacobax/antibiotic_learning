from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from .models import RecurrentActorCritic
from .ppo import PPOTrainer
from .buffer import RolloutBuffer
from .training_config import PPOConfig
import pickle
import sys

class RLAgent:
    def __init__(self, model: RecurrentActorCritic, device = "cuda", env = None):
        self.model = model
        self.device = device
        self.env = env  # Need env reference for action masking
        self.prev_h_state = model.init_hidden(device=device, batch_size=1)
        self.prev_action_onehot = torch.zeros(1, model.n_discrete, device=device)
        self.prev_action_cont = torch.zeros(1, model.k_doses, device=device)
        self.prev_pred_next_pop = torch.zeros(1, 1, device=device)

    def start_episode(self):
        self.prev_h_state = self.model.init_hidden(device=self.device, batch_size=1)
        self.prev_action_onehot.zero_()
        self.prev_action_cont.zero_()
        self.prev_pred_next_pop.zero_()
        self.model.eval()

    def select_action(self, obs: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action using hybrid action masking (Option C).
        
        Flow:
        1. Sample continuous action from policy
        2. Use continuous action to compute action mask from environment
        3. Apply mask to discrete logits
        4. Sample discrete action from masked distribution
        
        Returns:
            Tuple of (a_disc, a_cont, logp_disc, logp_cont, value, pred_next_pop, h_prev, action_mask, prev_action_onehot, prev_action_cont, prev_pred_next_pop)
        """
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)  # [1, obs_dim]
        prev_action_encoding = torch.cat([self.prev_action_onehot, self.prev_action_cont, self.prev_pred_next_pop], dim=-1)
        prev_action_onehot_buffer = self.prev_action_onehot.clone()
        prev_action_cont_buffer = self.prev_action_cont.clone()
        prev_pred_next_pop_buffer = self.prev_pred_next_pop.clone()
        
        # STEP 1: Get continuous action first (need it for masking)
        # We'll do a partial forward pass to get continuous action
        with torch.no_grad():
            # Get continuous action first
            logits_disc, (mu, std), _, _, _ = self.model.forward_step(obs_tensor, prev_action_encoding, self.prev_h_state)
            
            # Sample continuous action
            from torch.distributions import Normal
            dist_cont = Normal(mu, std)
            a_cont_raw = dist_cont.rsample()
            a_cont = torch.sigmoid(a_cont_raw) * self.model.sigmoid_scale_factor  # [1, K]
            
            # STEP 2: Compute action mask using the continuous action
            action_mask = None
            if self.env is not None:
                a_cont_np = a_cont.cpu().numpy()[0]  # [K]
                mask_np = self.env.get_action_mask(a_cont_np)  # [4]
                action_mask = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)  # [1, 4]
        
        # STEP 3: Now do full act() with the action mask
        with torch.no_grad():
            action_dict = self.model.act(
                obs_tensor,
                prev_action_encoding,
                self.prev_h_state,
                action_mask=action_mask,
                deterministic=False,
            )
        
        # Extract actions
        a_disc = action_dict["a_disc"]
        a_cont = action_dict["a_cont"]
        logp_disc = action_dict["logp_disc"]
        logp_cont = action_dict["logp_cont"]
        value = action_dict["value"]
        pred_next_pop = action_dict["pred_next_pop"]
        action_mask_out = action_dict["action_mask"]
        h_next = action_dict["h_next"]
        h_prev = self.prev_h_state
        self.prev_h_state = h_next

        # Update stored previous action encoding for next step
        dose_mask = (a_disc == self.model.dose_action_index).float().unsqueeze(-1)
        next_prev_cont = a_cont * dose_mask
        next_prev_onehot = F.one_hot(a_disc, num_classes=self.model.n_discrete).float()
        self.prev_action_onehot = next_prev_onehot.detach()
        self.prev_action_cont = next_prev_cont.detach()
        self.prev_pred_next_pop = pred_next_pop.view(1, 1).detach()

        return (
            a_disc,
            a_cont,
            logp_disc,
            logp_cont,
            value,
            pred_next_pop,
            h_prev,
            action_mask_out,
            prev_action_onehot_buffer,
            prev_action_cont_buffer,
            prev_pred_next_pop_buffer,
        )

    def update_policy(self, buffer: RolloutBuffer) -> dict:
        if self.trainer is None:
            raise Exception("Trainer not set for RLAgent.")
        
        self.model.train()
        data = buffer.stacked()
        stats = self.trainer.update(data)
        return stats
    
    def set_trainer(self, trainer: PPOTrainer):
        self.trainer = trainer

    def save_model(self, filepath: str, extra_info):
        if self.trainer is None:
            raise Exception("Trainer not set for RLAgent. Save not allowed")
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict" : self.trainer.optimizer.state_dict(),
            "config": self.trainer.cfg,
            **extra_info
        }, filepath)

    @staticmethod
    def load_agent_from_checkpoint(filepath: str, env=None, load_optimizer: bool = False, device="cpu"):
        """
        Load agent from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            env: Optional environment reference for action masking
            load_optimizer: Whether to return optimizer state dict for resuming training
            
        Returns:
            If load_optimizer=False: agent
            If load_optimizer=True: (agent, optimizer_state, update_number)
        """
        # Import here to avoid circular dependency
        
        # PyTorch 2.6+ requires allowlisting custom classes for security
        # Create a module remapping to handle old checkpoint files that reference rl.config.PPOConfig
        
        # Temporarily inject a fake rl.config module that redirects to training_config
        class ConfigModule:
            PPOConfig = PPOConfig
        
        if "rl.config" not in sys.modules:
            sys.modules["rl.config"] = ConfigModule()
        
        try:
            # Load with weights_only=False to handle pickled config objects
            # Map to CPU first to avoid backend-specific initialization issues, then move to target device
            checkpoint = torch.load(filepath, weights_only=False, map_location="cpu")
        except ModuleNotFoundError as e:
            # If rl.config module not found, add it and try again
            if "rl.config" in str(e):
                sys.modules["rl.config"] = ConfigModule()
                checkpoint = torch.load(filepath, weights_only=False, map_location=device)
            else:
                raise
        
        model_state_dict = checkpoint["model_state_dict"]
        
        # Initialize model architecture (assumes config is stored in checkpoint)
        cfg = checkpoint["config"]
        model = RecurrentActorCritic(
            obs_dim=cfg.obs_dim,
            n_discrete=cfg.n_discrete,
            k_doses=cfg.k_doses,
            hidden_dim=cfg.hidden_dim,
            rnn_layers=cfg.rnn_layers,
        )
        model.load_state_dict(model_state_dict, strict=False)
        
        # Ensure model is moved to the requested device (do not override with cfg.device)
        model = model.to(device)
        
        # Create agent with optional env (for action masking)
        agent = RLAgent(model=model, device=device, env=env)
        
        if load_optimizer:
            optimizer_state = checkpoint.get("optimizer_state_dict", None)
            update_number = checkpoint.get("update", 0)
            return agent, optimizer_state, update_number
        
        return agent

    def with_trainer(self, cfg: PPOConfig):
        trainer = PPOTrainer(self.model, cfg)
        self.set_trainer(trainer)
        return self