from typing import Tuple
import numpy as np
import torch
from .models import RecurrentActorCritic
from .ppo import PPOTrainer
from .buffer import RolloutBuffer
from .training_config import PPOConfig
import pickle
import sys

class RLAgent:
    def __init__(self, model: RecurrentActorCritic, device = "cuda"):
        self.model = model
        self.device = device
        self.prev_h_state = model.init_hidden(device=device, batch_size=1)

    def start_episode(self):
        self.prev_h_state = self.model.init_hidden(device=self.device, batch_size=1)
        self.model.eval()

    def select_action(self, obs: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)  # [1, obs_dim]
        
        # Get action from policy
        with torch.no_grad():
            action_dict = self.model.act(
                obs_tensor, self.prev_h_state, 
                deterministic=False,
            )
        # Extract actions
        a_disc = action_dict["a_disc"]
        a_cont = action_dict["a_cont"]
        logp_disc = action_dict["logp_disc"]
        logp_cont = action_dict["logp_cont"]
        value = action_dict["value"]
        h_next = action_dict["h_next"]
        h_prev = self.prev_h_state
        self.prev_h_state = h_next

        return a_disc , a_cont, logp_disc, logp_cont, value, h_prev

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
    def load_agent_from_checkpoint(filepath: str):
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
            checkpoint = torch.load(filepath, weights_only=False)
        except ModuleNotFoundError as e:
            # If rl.config module not found, add it and try again
            if "rl.config" in str(e):
                sys.modules["rl.config"] = ConfigModule()
                checkpoint = torch.load(filepath, weights_only=False)
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
        model.load_state_dict(model_state_dict)
        
        # Ensure model is on the correct device
        device = cfg.device
        model = model.to(device)
        
        agent = RLAgent(model=model, device=device)
        return agent

    def with_trainer(self, cfg: PPOConfig):
        trainer = PPOTrainer(self.model, cfg)
        self.set_trainer(trainer)
        return self