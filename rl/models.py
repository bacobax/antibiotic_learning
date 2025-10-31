"""
Recurrent Actor-Critic model with hybrid action space.

Supports discrete action selection and continuous dose parameters.
Uses GRU for recurrent processing of sequential observations.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def init_weights_orthogonal(m: nn.Module, gain: float = 1.0) -> None:
    """
    Initialize module weights with orthogonal initialization.
    
    Args:
        m: Module to initialize
        gain: Gain factor for orthogonal initialization
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class RecurrentActorCritic(nn.Module):
    """
    Recurrent Actor-Critic with hybrid action space.
    
    Architecture:
        - GRU core for sequential processing
        - Discrete action head (categorical distribution)
        - Continuous dose head (Gaussian with tanh squashing to [0,1])
        - Value head (state value estimate)
    
    Shapes:
        obs: [B, obs_dim] - Batch of observations
        h_prev: [layers, B, hidden_dim] - Previous hidden state
        h_next: [layers, B, hidden_dim] - Next hidden state
        logits_disc: [B, n_discrete] - Discrete action logits
        mu: [B, k_doses] - Continuous action means (pre-squash)
        std: [B, k_doses] - Continuous action stds
        value: [B, 1] - State value estimates
    """
    
    def __init__(
        self,
        obs_dim: int,
        n_discrete: int,
        k_doses: int,
        hidden_dim: int = 256,
        rnn_layers: int = 1,
    ):
        """
        Initialize Recurrent Actor-Critic.
        
        Args:
            obs_dim: Observation space dimension
            n_discrete: Number of discrete actions
            k_doses: Continuous action dimension (number of antibiotic types)
            hidden_dim: Hidden dimension for GRU
            rnn_layers: Number of GRU layers
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.n_discrete = n_discrete
        self.k_doses = k_doses
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        
        # GRU core
        self.gru = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=False,  # Expect [T, B, obs_dim]
        )
        
        # Discrete action head
        self.discrete_head = nn.Linear(hidden_dim, n_discrete)
        
        # Continuous action head (Gaussian policy)
        self.continuous_mu = nn.Linear(hidden_dim, k_doses)
        self.continuous_log_std = nn.Parameter(torch.zeros(k_doses))
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(lambda m: init_weights_orthogonal(m, gain=1.0))
        # Smaller init for output layers
        init_weights_orthogonal(self.discrete_head, gain=0.01)
        init_weights_orthogonal(self.continuous_mu, gain=0.01)
        init_weights_orthogonal(self.value_head, gain=1.0)
    
    def forward_step(
        self, 
        obs: torch.Tensor, 
        h_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Single forward step through the network.
        
        Args:
            obs: Observations, shape [B, obs_dim]
            h_prev: Previous hidden state, shape [layers, B, hidden_dim]
        
        Returns:
            logits_disc: Discrete action logits, shape [B, n_discrete]
            (mu, std): Continuous action distribution params
                mu: shape [B, k_doses] (pre-squash means)
                std: shape [B, k_doses]
            value: State value, shape [B, 1]
            h_next: Next hidden state, shape [layers, B, hidden_dim]
        """
        # obs: [B, obs_dim] -> [1, B, obs_dim] for GRU
        obs_seq = obs.unsqueeze(0)
        
        # GRU forward
        gru_out, h_next = self.gru(obs_seq, h_prev)
        # gru_out: [1, B, hidden_dim]
        
        # Extract features
        features = gru_out.squeeze(0)  # [B, hidden_dim]
        
        # Discrete action logits
        logits_disc = self.discrete_head(features)  # [B, n_discrete]
        
        # Continuous action distribution
        mu = self.continuous_mu(features)  # [B, k_doses]
        std = torch.exp(self.continuous_log_std).expand_as(mu)  # [B, k_doses]
        
        # Value estimate
        value = self.value_head(features)  # [B, 1]
        
        return logits_disc, (mu, std), value, h_next
    
    def act(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor,
        dose_action_index: int = 3,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample actions from policy.
        
        Args:
            obs: Observations, shape [B, obs_dim]
            h_prev: Previous hidden state, shape [layers, B, hidden_dim]
            dose_action_index: Index of DOSE action in discrete set
            deterministic: If True, use mode instead of sampling
        
        Returns:
            Dictionary containing:
                a_disc: Discrete actions, shape [B] (LongTensor)
                a_cont: Continuous doses, shape [B, K] (FloatTensor, in [0,1])
                logp_disc: Discrete action log-probs, shape [B]
                logp_cont: Continuous action log-probs, shape [B]
                value: Value estimates, shape [B]
                h_next: Next hidden state, shape [layers, B, hidden_dim]
        """
        logits_disc, (mu, std), value, h_next = self.forward_step(obs, h_prev)
        
        # Discrete action
        dist_disc = Categorical(logits=logits_disc)
        if deterministic:
            a_disc = logits_disc.argmax(dim=-1)
        else:
            a_disc = dist_disc.sample()
        logp_disc = dist_disc.log_prob(a_disc)
        
        # Continuous action (always sample, but only used if a_disc == DOSE)
        dist_cont = Normal(mu, std)
        if deterministic:
            a_cont_raw = mu
        else:
            a_cont_raw = dist_cont.rsample()  # Reparameterized sample
        
        # Squash to [0, 1] via tanh
        a_cont = 0.5 * (torch.tanh(a_cont_raw) + 1.0)  # [B, K]
        
        # Log-prob (pre-tanh space, no Jacobian correction for simplicity)
        logp_cont_raw = dist_cont.log_prob(a_cont_raw).sum(dim=-1)  # [B]
        
        # Mask: only use continuous log-prob if discrete action is DOSE
        is_dose = (a_disc == dose_action_index).float()  # [B]
        logp_cont = logp_cont_raw * is_dose
        
        return {
            "a_disc": a_disc,  # [B]
            "a_cont": a_cont,  # [B, K]
            "logp_disc": logp_disc,  # [B]
            "logp_cont": logp_cont,  # [B]
            "value": value.squeeze(-1),  # [B]
            "h_next": h_next,  # [layers, B, hidden_dim]
        }
    
    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        h_init: torch.Tensor,
        a_disc: torch.Tensor,
        a_cont: torch.Tensor,
        dose_action_index: int = 3,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate log-probs and values for given actions over a sequence.
        
        Used during PPO update to recompute log-probs for old actions.
        
        Args:
            obs_seq: Observation sequence, shape [T, B, obs_dim]
            h_init: Initial hidden state, shape [layers, B, hidden_dim]
            a_disc: Discrete actions, shape [T, B]
            a_cont: Continuous doses, shape [T, B, K] (in [0,1])
            dose_action_index: Index of DOSE action
        
        Returns:
            Dictionary containing:
                logp_disc: Discrete log-probs, shape [T, B]
                logp_cont: Continuous log-probs, shape [T, B]
                value: Value estimates, shape [T, B]
                entropy_disc: Discrete entropy, shape [T, B]
                entropy_cont: Continuous entropy, shape [T, B]
        """
        T, B = obs_seq.shape[:2]
        
        # Forward through GRU
        gru_out, _ = self.gru(obs_seq, h_init)  # [T, B, hidden_dim]
        
        # Compute heads
        logits_disc = self.discrete_head(gru_out)  # [T, B, n_discrete]
        mu = self.continuous_mu(gru_out)  # [T, B, k_doses]
        std = torch.exp(self.continuous_log_std).expand_as(mu)  # [T, B, k_doses]
        value = self.value_head(gru_out).squeeze(-1)  # [T, B]
        
        # Discrete distribution
        dist_disc = Categorical(logits=logits_disc)
        logp_disc = dist_disc.log_prob(a_disc)  # [T, B]
        entropy_disc = dist_disc.entropy()  # [T, B]
        
        # Continuous distribution
        # Need to invert tanh squashing: a_cont = 0.5 * (tanh(a_raw) + 1)
        # => a_raw = atanh(2 * a_cont - 1)
        a_cont_clamped = torch.clamp(a_cont, 0.01, 0.99)  # Avoid atanh singularities
        a_cont_raw = torch.atanh(2.0 * a_cont_clamped - 1.0)  # [T, B, K]
        
        dist_cont = Normal(mu, std)
        logp_cont_raw = dist_cont.log_prob(a_cont_raw).sum(dim=-1)  # [T, B]
        entropy_cont = dist_cont.entropy().sum(dim=-1)  # [T, B]
        
        # Mask by DOSE action
        is_dose = (a_disc == dose_action_index).float()  # [T, B]
        logp_cont = logp_cont_raw * is_dose
        entropy_cont = entropy_cont * is_dose
        
        return {
            "logp_disc": logp_disc,
            "logp_cont": logp_cont,
            "value": value,
            "entropy_disc": entropy_disc,
            "entropy_cont": entropy_cont,
        }
    
    def init_hidden(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        """
        Initialize hidden state.
        
        Args:
            batch_size: Batch size
            device: Device to create tensor on
        
        Returns:
            h: Zero hidden state, shape [layers, B, hidden_dim]
        """
        return torch.zeros(
            self.rnn_layers, batch_size, self.hidden_dim, 
            dtype=torch.float32, device=device
        )
