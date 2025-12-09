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
from simulation.simulation_config import N_TRAITS, N_BACTERIA_TYPES, ANTIBIOTIC_TYPES, TRAIT_KEYS

def antibiotic_vulnerabilities(expression, dtype, device) -> torch.Tensor:
    """
    Compute antibiotic vulnerabilities from bacterial expression profiles.
    
    Args:
        expression: Tensor of shape [B, K*M] where B is batch size, K is bacteria types, M is traits
                   or shape [K*M] for single sample (will be reshaped)
        dtype: Data type for tensors
        device: Device for tensors
    
    Returns:
        vulnerability: Tensor of shape [B, A] where B is batch size, A is number of antibiotics
                      or shape [A] if input was 1D
    """
    input_shape = expression.shape
    is_batched = len(input_shape) > 1
    
    if is_batched:
        # Batch mode: [B, K*M]
        B = input_shape[0]
        # Reshape to [B, K, M]
        expression_reshaped = expression.reshape(B, N_BACTERIA_TYPES, N_TRAITS)
    else:
        # Single sample: [K*M]
        expression_reshaped = expression.reshape(N_BACTERIA_TYPES, N_TRAITS)
        expression_reshaped = expression_reshaped.unsqueeze(0)  # Add batch dimension: [1, K, M]
        B = 1
    
    # Clamp for safety
    expression_reshaped = torch.clamp(expression_reshaped, 0.0, 1.0)  # [B, K, M]
    
    # Build weight matrix W from antibiotic types
    ab_names = list(ANTIBIOTIC_TYPES.keys())  # [A]
    W_rows = []
    tox_list = []
    for name in ab_names:
        ab = ANTIBIOTIC_TYPES[name]
        w = torch.tensor([ab[k] for k in TRAIT_KEYS], dtype=dtype, device=device)
        # normalize so sum=1 to keep dot products in [0,1]
        w = w / (w.sum() + 1e-8)
        W_rows.append(w)
        tox_list.append(ab["toxicity_constant"])
    W = torch.stack(W_rows, dim=0)  # [A, M]
    
    # Compute resistances: [B, K, M] @ [M, A] -> [B, K, A]
    resistances = torch.clamp(expression_reshaped @ W.T, 0.0, 1.0)  # [B, K, A]
    
    # Average across bacteria types: [B, K, A] -> [B, A]
    avg_resistance = torch.mean(resistances, dim=1)  # [B, A]
    
    # Compute vulnerabilities
    toxicities = torch.tensor(tox_list, dtype=dtype, device=device)  # [A]
    vulnerability = (1 - avg_resistance) * toxicities  # [B, A]
    
    # Remove batch dimension if input was unbatched
    if not is_batched:
        vulnerability = vulnerability.squeeze(0)  # [A]
    
    return vulnerability
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


class ExpressionsPredictor(nn.Module):
    def __init__(self, bacteria_types, genome_dim):
        super().__init__()
        input_dim = bacteria_types * genome_dim + 1  # +1 for age
        hidden_dim = input_dim - 1
        output_dim = bacteria_types * genome_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )
        self.net.apply(self.init_weights_orthogonal)

    def init_weights_orthogonal(self, m: nn.Module, gain: float = 1.0) -> None:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expressions = self.net(x)
        res = expressions + x[:, : -1]  # Residual connection

        return res

class VulnerabilityPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.expression_predictor = ExpressionsPredictor(N_BACTERIA_TYPES, N_TRAITS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expressions = self.expression_predictor(x)
        vulnerability = antibiotic_vulnerabilities(expressions, dtype=x.dtype, device=x.device)
        return vulnerability # [A]

class RecurrentActorCritic(nn.Module):
    """
    Recurrent Actor-Critic with hybrid action space.
    
    Architecture:
        - GRU core for sequential processing
        - Discrete action head (categorical distribution)
        - Continuous dose head (Gaussian with sigmoid scaling to [0, sigmoid_scale_factor])
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
        dose_action_index: int = 3,
        sigmoid_scale_factor: float = 0.2,
    ):
        """
        Initialize Recurrent Actor-Critic.
        
        Args:
            obs_dim: Observation space dimension (as built by env, will be adjusted internally)
            n_discrete: Number of discrete actions
            k_doses: Continuous action dimension (number of antibiotic types)
            hidden_dim: Hidden dimension for GRU
            rnn_layers: Number of GRU layers
            sigmoid_scale_factor: Scaling factor for sigmoid output (doses in [0, sigmoid_scale_factor])
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.n_discrete = n_discrete
        self.k_doses = k_doses
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.dose_action_index = dose_action_index
        self.sigmoid_scale_factor = sigmoid_scale_factor
        self.prev_action_dim = n_discrete + k_doses + 1
        
        # Calculate adjusted input dimension for GRU:
        # Remove genome slots (N_BACTERIA_TYPES * N_TRAITS) and related observation slots
        # Removed: genome (N_BACTERIA_TYPES*N_TRAITS) + has_last_seq (1) + last_seq_age_norm (1) + measure_age_norm (1) = N_BACTERIA_TYPES*N_TRAITS+3
        # Add vulnerability slots (number of antibiotic types)
        n_antibiotics = len(ANTIBIOTIC_TYPES)
        self.genome_age_removal = N_BACTERIA_TYPES * N_TRAITS + 3  # Slots to remove
        self.vuln_dim = n_antibiotics  # Slots to add
        self.adjusted_obs_dim = obs_dim - self.genome_age_removal + self.vuln_dim
        
        # GRU core with adjusted input size
        self.gru = nn.GRU(
            input_size=self.adjusted_obs_dim + self.prev_action_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=False,  # Expect [T, B, obs_dim]
        )

        self.vuln_predictor = VulnerabilityPredictor()
        
        # Discrete action head
        self.discrete_head = nn.Linear(hidden_dim, n_discrete)
        
        # Continuous action head (Gaussian policy with sigmoid scaling)
        self.continuous_head = nn.Linear(hidden_dim + self.vuln_dim, k_doses)
        self.continuous_log_std = nn.Parameter(torch.zeros(k_doses))
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Prediction head (next population)
        self.prediction_fc = nn.Linear(hidden_dim + self.prev_action_dim, hidden_dim)
        self.pred_head = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(lambda m: init_weights_orthogonal(m, gain=1.0))
        # Smaller init for output layers
        init_weights_orthogonal(self.discrete_head, gain=0.01)
        init_weights_orthogonal(self.continuous_head, gain=0.01)
        init_weights_orthogonal(self.value_head, gain=1.0)
        init_weights_orthogonal(self.prediction_fc, gain=1.0)
        init_weights_orthogonal(self.pred_head, gain=0.01)
    
    def forward_step(
        self, 
        obs: torch.Tensor, 
        prev_action: torch.Tensor,
        h_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single forward step through the network.
        
        Modifies observation by:
        1. Extracting genome and age from obs
        2. Computing vulnerabilities via vuln_predictor
        3. Replacing genome and age slots with vulnerabilities
        4. Feeding modified obs to GRU
        
        Args:
            obs: Observations, shape [B, obs_dim] (as built by env)
            prev_action: Previous action encoding, shape [B, prev_action_dim]
            h_prev: Previous hidden state, shape [layers, B, hidden_dim]
        
        Returns:
            logits_disc: Discrete action logits, shape [B, n_discrete]
            (mu, std): Continuous action distribution params
                mu: shape [B, k_doses] (pre-squash means)
                std: shape [B, k_doses]
            value: State value, shape [B, 1]
            features: Latent features for heads, shape [B, hidden_dim]
            h_next: Next hidden state, shape [layers, B, hidden_dim]
        """
        # Extract genome and age from observation
        # Genome is at indices [3:3+N_BACTERIA_TYPES*N_TRAITS]
        # Age is at index [3+N_BACTERIA_TYPES*N_TRAITS+2]
        avg_genome = obs[:, 3: 3 + N_BACTERIA_TYPES * N_TRAITS]
        age = obs[:, 3 + N_BACTERIA_TYPES * N_TRAITS + 2]
        avg_genome_with_age = torch.cat([avg_genome, age.unsqueeze(-1)], dim=-1)

        # Compute vulnerabilities from genome and age
        vulnerabilities = self.vuln_predictor(avg_genome_with_age)  # [B, n_antibiotics]

        # Build modified observation tensor:
        # Keep: obs[:, 0:3] (last_count_norm, has_last_count, last_count_age_norm)
        # Remove: obs[:, 3:3+N_BACTERIA_TYPES*N_TRAITS+1] (genome + has_last_seq)
        # Remove: obs[:, 3+N_BACTERIA_TYPES*N_TRAITS+2] (last_seq_age_norm, measure_age_norm already removed)
        # Add: vulnerabilities at this position
        # Keep: the rest (dose_features, t_norm)
        
        prefix = obs[:, :3]  # [B, 3] - last_count_norm, has_last_count, last_count_age_norm
        suffix_start = 3 + N_BACTERIA_TYPES * N_TRAITS + 2 + 1  # Start of measure_age_norm
        suffix = obs[:, suffix_start:]  # [B, remaining] - dose_features, t_norm
        
        # Concatenate: prefix + vulnerabilities + suffix
        obs_modified = torch.cat([prefix, vulnerabilities, suffix], dim=-1)  # [B, adjusted_obs_dim]

        # Concatenate previous action encoding to modified observation for GRU input
        gru_input = torch.cat([obs_modified, prev_action], dim=-1)  # [B, adjusted_obs_dim + prev_action_dim]
        # gru_input: [B, ...] -> [1, B, ...] for GRU
        obs_seq = gru_input.unsqueeze(0)
        
        # GRU forward
        gru_out, h_next = self.gru(obs_seq, h_prev)
        # gru_out: [1, B, hidden_dim]
        
        # Extract features
        features = gru_out.squeeze(0)  # [B, hidden_dim]
        
        # Discrete action logits
        logits_disc = self.discrete_head(features)  # [B, n_discrete]
        
        # Continuous action distribution (sigmoid-scaled)
        mu_raw = self.continuous_head(torch.cat([features, vulnerabilities], dim=-1))  # [B, k_doses]
        mu = torch.sigmoid(mu_raw) * self.sigmoid_scale_factor  # [B, k_doses] in [0, sigmoid_scale_factor]
        # Clamp log_std to prevent explosion: exp(-5) ≈ 0.007, exp(2) ≈ 7.4
        log_std_clamped = torch.clamp(self.continuous_log_std, min=-5.0, max=2.0)
        std = torch.exp(log_std_clamped).expand_as(mu)  # [B, k_doses]

        # Value estimate
        value = self.value_head(features)  # [B, 1]

        return logits_disc, (mu, std), value, features, h_next

    def _predict_next_population(
        self,
        features: torch.Tensor,
        action_one_hot: torch.Tensor,
        action_cont: torch.Tensor,
        prev_pred_next_pop: torch.Tensor,
    ) -> torch.Tensor:
        """Compute next population prediction conditioned on action."""
        pred_input = torch.cat([features, action_one_hot, action_cont, prev_pred_next_pop], dim=-1)
        hidden = F.relu(self.prediction_fc(pred_input))
        return F.softplus(self.pred_head(hidden))
    
    def act(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor,
        h_prev: torch.Tensor,
        action_mask: torch.Tensor = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample actions from policy with optional action masking.
        
        HYBRID ACTION MASKING (Option C):
        1. Sample continuous action first
        2. Use it to compute action mask (if provided)
        3. Apply mask to discrete logits
        4. Sample discrete action from masked distribution
        
        Args:
            obs: Observations, shape [B, obs_dim]
            h_prev: Previous hidden state, shape [layers, B, hidden_dim]
            action_mask: Optional action mask, shape [B, 4], where mask[i,j]=1 means action j is valid
            deterministic: If True, use mode instead of sampling
        
        Returns:
            Dictionary containing:
                a_disc: Discrete actions, shape [B] (LongTensor)
                a_cont: Continuous doses, shape [B, K] (FloatTensor, in [0,1])
                logp_disc: Discrete action log-probs, shape [B]
                logp_cont: Continuous action log-probs, shape [B]
                value: Value estimates, shape [B]
                pred_next_pop: Predicted next population (normalized), shape [B]
                h_next: Next hidden state, shape [layers, B, hidden_dim]
                action_mask: Action mask used (if provided), shape [B, 4]
        """
        logits_disc, (mu, std), value, features, h_next = self.forward_step(obs, prev_action, h_prev)
        
        # STEP 1: Sample continuous action FIRST (always sample, used for DOSE and for masking)
        # mu is already sigmoid-scaled to [0, sigmoid_scale_factor]
        dist_cont = Normal(mu, std)
        if deterministic:
            a_cont = mu
        else:
            a_cont = dist_cont.rsample()  # Reparameterized sample
        
        # Clamp to valid range [0, sigmoid_scale_factor]
        a_cont = torch.clamp(a_cont, 0.0, self.sigmoid_scale_factor)  # [B, K]
        
        # Log-prob (in sigmoid-scaled space)
        logp_cont_raw = dist_cont.log_prob(a_cont).sum(dim=-1)  # [B]
        
        # STEP 2: Apply action mask if provided
        # Mask is applied to logits: masked_logits = logits + log(mask)
        # This ensures invalid actions get probability 0
        if action_mask is not None:
            # Add small epsilon to avoid log(0)
            mask_log = torch.log(action_mask + 1e-10)  # [B, n_discrete]
            logits_disc_masked = logits_disc + mask_log
        else:
            logits_disc_masked = logits_disc
        
        # STEP 3: Sample discrete action from masked distribution
        dist_disc = Categorical(logits=logits_disc_masked)
        if deterministic:
            a_disc = logits_disc_masked.argmax(dim=-1)
        else:
            a_disc = dist_disc.sample()
        logp_disc = dist_disc.log_prob(a_disc)
        
        # Mask: only use continuous log-prob if discrete action is DOSE
        is_dose = (a_disc == self.dose_action_index).float()  # [B]
        logp_cont = logp_cont_raw * is_dose

        # Build action-conditioned prediction input
        action_one_hot = F.one_hot(a_disc, num_classes=self.n_discrete).float()
        action_cont = a_cont * is_dose.unsqueeze(-1)
        prev_pred_next_pop = prev_action[:, self.n_discrete + self.k_doses :]
        pred_next_pop = self._predict_next_population(features, action_one_hot, action_cont, prev_pred_next_pop)
        
        return {
            "a_disc": a_disc,  # [B]
            "a_cont": a_cont,  # [B, K]
            "logp_disc": logp_disc,  # [B]
            "logp_cont": logp_cont,  # [B]
            "value": value.squeeze(-1),  # [B]
            "pred_next_pop": pred_next_pop.squeeze(-1),  # [B]
            "h_next": h_next,  # [layers, B, hidden_dim]
            "action_mask": action_mask if action_mask is not None else torch.ones(obs.shape[0], self.n_discrete, device=obs.device),  # [B, 4]
        }
    
    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        h_init: torch.Tensor,
        a_disc: torch.Tensor,
        a_cont: torch.Tensor,
        pred_action_input: torch.Tensor = None,
        action_masks: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate log-probs and values for given actions over a sequence.
        
        Modifies observations by replacing genome and age slots with vulnerabilities.
        Used during PPO update to recompute log-probs for old actions.
        Supports action masking for consistency with act() method.
        
        Args:
            obs_seq: Observation sequence, shape [T, B, obs_dim] (as built by env)
            prev_action_seq: Previous action encodings, shape [T, B, n_discrete + k_doses + 1]
            h_init: Initial hidden state, shape [layers, B, hidden_dim]
            a_disc: Discrete actions, shape [T, B]
            a_cont: Continuous doses, shape [T, B, K] (in [0,1])
            pred_action_input: Optional cached action features for prediction head, shape [T, B, n_discrete + k_doses]
            action_masks: Optional action masks, shape [T, B, n_discrete]
        
        Returns:
            Dictionary containing:
                logp_disc: Discrete log-probs, shape [T, B]
                logp_cont: Continuous log-probs, shape [T, B]
                value: Value estimates, shape [T, B]
                pred_next_pop: Predicted next population (normalized), shape [T, B]
                entropy_disc: Discrete entropy, shape [T, B]
                entropy_cont: Continuous entropy, shape [T, B]
        """
        T, B = obs_seq.shape[:2]
        
        # Transform observations: replace genome and age with vulnerabilities
        # Extract genome and age from observation sequence
        avg_genome_seq = obs_seq[:, :, 3: 3 + N_BACTERIA_TYPES * N_TRAITS]  # [T, B, genome_dim]
        age_seq = obs_seq[:, :, 3 + N_BACTERIA_TYPES * N_TRAITS + 2]  # [T, B]
        
        # Reshape for vulnerability prediction: flatten batch and time dimensions
        avg_genome_flat = avg_genome_seq.reshape(-1, N_BACTERIA_TYPES * N_TRAITS)  # [T*B, genome_dim]
        age_flat = age_seq.reshape(-1)  # [T*B]
        avg_genome_with_age_flat = torch.cat([avg_genome_flat, age_flat.unsqueeze(-1)], dim=-1)  # [T*B, genome_dim+1]
        
        # Compute vulnerabilities
        vulnerabilities_flat = self.vuln_predictor(avg_genome_with_age_flat)  # [T*B, n_antibiotics]
        vulnerabilities_seq = vulnerabilities_flat.reshape(T, B, -1)  # [T, B, n_antibiotics]
        
        # Build modified observation sequence
        prefix = obs_seq[:, :, :3]  # [T, B, 3]
        suffix_start = 3 + N_BACTERIA_TYPES * N_TRAITS + 2 + 1
        suffix = obs_seq[:, :, suffix_start:]  # [T, B, remaining]
        
        obs_seq_modified = torch.cat([prefix, vulnerabilities_seq, suffix], dim=-1)  # [T, B, adjusted_obs_dim]
        
        # Forward through GRU with modified observations
        gru_input = torch.cat([obs_seq_modified, prev_action_seq], dim=-1)
        gru_out, _ = self.gru(gru_input, h_init)  # [T, B, hidden_dim]
        
        # Compute heads
        logits_disc = self.discrete_head(gru_out)  # [T, B, n_discrete]
        mu_raw = self.continuous_head(torch.cat([gru_out, vulnerabilities_seq], dim=-1))  # [T, B, k_doses]
        mu = torch.sigmoid(mu_raw) * self.sigmoid_scale_factor  # [T, B, k_doses] in [0, sigmoid_scale_factor]
        # Clamp log_std to prevent explosion: exp(-5) ≈ 0.007, exp(2) ≈ 7.4
        log_std_clamped = torch.clamp(self.continuous_log_std, min=-5.0, max=2.0)
        std = torch.exp(log_std_clamped).expand_as(mu)  # [T, B, k_doses]
        value = self.value_head(gru_out).squeeze(-1)  # [T, B]
        
        # Apply action masks if provided
        if action_masks is not None:
            # Add small epsilon to avoid log(0)
            mask_log = torch.log(action_masks + 1e-10)  # [T, B, n_discrete]
            logits_disc_masked = logits_disc + mask_log
        else:
            logits_disc_masked = logits_disc
        
        # Discrete distribution (with masked logits)
        dist_disc = Categorical(logits=logits_disc_masked)
        logp_disc = dist_disc.log_prob(a_disc)  # [T, B]
        entropy_disc = dist_disc.entropy()  # [T, B]
        
        # Continuous distribution (sigmoid-scaled space)
        # a_cont is already in [0, sigmoid_scale_factor] range
        a_cont_clamped = torch.clamp(a_cont, 0.0, self.sigmoid_scale_factor)  # Ensure valid range
        
        dist_cont = Normal(mu, std)
        logp_cont_raw = dist_cont.log_prob(a_cont_clamped).sum(dim=-1)  # [T, B]
        entropy_cont = dist_cont.entropy().sum(dim=-1)  # [T, B]
        
        # Mask by DOSE action
        is_dose = (a_disc == self.dose_action_index).float()  # [T, B]
        logp_cont = logp_cont_raw * is_dose
        entropy_cont = entropy_cont * is_dose

        prev_action_flat = prev_action_seq.reshape(-1, self.prev_action_dim)
        prev_pred_flat = prev_action_flat[:, self.n_discrete + self.k_doses :]

        if pred_action_input is not None:
            pred_input_flat = pred_action_input.reshape(-1, pred_action_input.shape[-1])
            action_one_hot_flat = pred_input_flat[:, :self.n_discrete]
            action_cont_flat = pred_input_flat[:, self.n_discrete:self.n_discrete + self.k_doses]
        else:
            action_one_hot = F.one_hot(a_disc, num_classes=self.n_discrete).float()
            action_cont = a_cont * is_dose.unsqueeze(-1)
            action_one_hot_flat = action_one_hot.reshape(-1, self.n_discrete)
            action_cont_flat = action_cont.reshape(-1, self.k_doses)

        features_flat = gru_out.reshape(-1, self.hidden_dim)
        pred_next_pop = self._predict_next_population(
            features_flat,
            action_one_hot_flat,
            action_cont_flat,
            prev_pred_flat,
        )
        pred_next_pop = pred_next_pop.view_as(value)
        
        return {
            "logp_disc": logp_disc,
            "logp_cont": logp_cont,
            "value": value,
            "pred_next_pop": pred_next_pop,
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
