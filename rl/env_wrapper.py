"""
Thin environment wrapper around Mesa bacteria simulation.

This wrapper exposes a Gym-like API without requiring Gym as a dependency.
It maintains clean separation between RL code and simulation code.
"""
from typing import Any, Callable, Dict, Tuple

import numpy as np


# Discrete action space mapping
ACTION_NOOP = 0
ACTION_COUNT_BACTERIA = 1
ACTION_SEQUENCING = 2
ACTION_DOSE = 3


class PetriEnvWrapper:
    """
    Thin wrapper around Mesa bacteria simulation for RL training.
    
    Exposes a Gym-like interface (reset, step) without requiring Gym.
    Supports hybrid action space: discrete action selection + continuous dose vector.
    
    Attributes:
        mesa_model_factory: Callable that returns a fresh Mesa model instance
        k_doses: Number of antibiotic types (dimension of continuous action)
        obs_builder: Callable that extracts observation from Mesa model
        scale_dose: Callable to scale [0,1] doses to simulation units
    """
    
    def __init__(
        self,
        mesa_model_factory: Callable[[], Any],
        k_doses: int,
        obs_builder: Callable[[Any], np.ndarray],
        scale_dose: Callable[[np.ndarray], np.ndarray] = None,
        max_steps: int = 1000,
    ):
        """
        Initialize environment wrapper.
        
        Args:
            mesa_model_factory: Function that returns a fresh Mesa model/environment
            k_doses: Number of antibiotic types (K in [0,1]^K)
            obs_builder: Function that builds observation from model
                         Signature: model -> np.ndarray with shape [obs_dim]
            scale_dose: Optional function to scale [0,1] doses to simulation units
                        Default: identity (no scaling)
            max_steps: Maximum steps per episode before truncation
        """
        self.mesa_model_factory = mesa_model_factory
        self.k_doses = k_doses
        self.obs_builder = obs_builder
        self.scale_dose = scale_dose if scale_dose is not None else lambda x: x
        self.max_steps = max_steps
        
        # Internal state
        self.model = None
        self.t = 0
        self.episode_return = 0.0
        
        # Pending events (for COUNT_BACTERIA, SEQUENCING with delays)
        self.pending_count = False
        self.pending_sequencing = False
        self.count_cooldown = 0
        self.sequencing_cooldown = 0
        
        # Action tracking for rewards
        self.last_population = 0
        self.last_action = ACTION_NOOP
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.
        
        Returns:
            obs: Initial observation, shape [obs_dim]
        """
        # Create fresh Mesa model
        self.model = self.mesa_model_factory()
        
        # Reset internal counters
        self.t = 0
        self.episode_return = 0.0
        self.pending_count = False
        self.pending_sequencing = False
        self.count_cooldown = 0
        self.sequencing_cooldown = 0
        self.last_population = len(self.model.agent_set)
        self.last_action = ACTION_NOOP
        
        # Build initial observation
        obs = self.obs_builder(self.model)
        assert isinstance(obs, np.ndarray), "obs_builder must return np.ndarray"
        assert obs.ndim == 1, f"Expected 1D obs, got shape {obs.shape}"
        
        return obs.astype(np.float32)
    
    def step(
        self, 
        a_discrete: int, 
        a_cont: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            a_discrete: Discrete action index (0=NOOP, 1=COUNT, 2=SEQ, 3=DOSE)
            a_cont: Continuous dose vector in [0,1]^K (only used if a_discrete == DOSE)
        
        Returns:
            obs: Next observation, shape [obs_dim]
            reward: Scalar reward for this step
            done: Whether episode is terminated
            info: Dict containing {"budget": float, "t": int, "population": int}
        """
        assert isinstance(a_discrete, (int, np.integer)), "a_discrete must be int"
        assert isinstance(a_cont, np.ndarray), "a_cont must be np.ndarray"
        assert a_cont.shape == (self.k_doses,), f"Expected a_cont shape ({self.k_doses},), got {a_cont.shape}"
        assert 0 <= a_discrete < 4, f"a_discrete must be in [0, 3], got {a_discrete}"
        
        # Execute action
        reward = self._execute_action(a_discrete, a_cont)
        
        # Step Mesa simulation once
        self.model.step()
        self.t += 1
        
        # Update cooldowns
        if self.count_cooldown > 0:
            self.count_cooldown -= 1
        if self.sequencing_cooldown > 0:
            self.sequencing_cooldown -= 1
        
        # Build next observation
        obs = self.obs_builder(self.model)
        obs = obs.astype(np.float32)
        
        # Check termination
        population = len(self.model.agent_set)
        done = (population == 0) or (self.t >= self.max_steps)
        
        # Build info dict (no large objects)
        info = {
            "budget": 1.0,  # Placeholder budget value
            "t": self.t,
            "population": population,
            "episode_return": self.episode_return,
        }
        
        self.episode_return += reward
        self.last_population = population
        self.last_action = a_discrete
        
        return obs, reward, done, info
    
    def _execute_action(self, a_discrete: int, a_cont: np.ndarray) -> float:
        """
        Execute discrete action and return immediate reward.
        
        Args:
            a_discrete: Discrete action index
            a_cont: Continuous dose vector (only used for DOSE action)
        
        Returns:
            reward: Immediate reward for this action
        """
        reward = 0.0
        
        if a_discrete == ACTION_NOOP:
            # Do nothing, small negative reward for passing time
            reward = -0.01
        
        elif a_discrete == ACTION_COUNT_BACTERIA:
            # Count bacteria (assume instant or use cooldown)
            if self.count_cooldown == 0:
                # Schedule count (in real implementation, this might have a delay)
                self.pending_count = True
                self.count_cooldown = 5  # 5 step cooldown
                reward = -0.05  # Small cost for counting
            else:
                # Action on cooldown
                reward = -0.1
        
        elif a_discrete == ACTION_SEQUENCING:
            # Genome sequencing (expensive, longer cooldown)
            if self.sequencing_cooldown == 0:
                self.pending_sequencing = True
                self.sequencing_cooldown = 20  # 20 step cooldown
                reward = -0.2  # Higher cost for sequencing
            else:
                # Action on cooldown
                reward = -0.15
        
        elif a_discrete == ACTION_DOSE:
            # Apply antibiotic doses
            # a_cont is in [0, 1]^K, scale if needed
            scaled_doses = self.scale_dose(a_cont)
            
            # Apply each antibiotic type to the model
            # Assume model has apply_antibiotic(antibiotic_type, amount) method
            antibiotic_types = list(self.model.antibiotic_fields.keys())
            
            for i, ab_type in enumerate(antibiotic_types[:self.k_doses]):
                dose_amount = float(scaled_doses[i])
                if dose_amount > 0.01:  # Only apply significant doses
                    self.model.apply_antibiotic(ab_type, dose_amount)
            
            # Reward based on dose amount (negative cost)
            total_dose = np.sum(scaled_doses)
            reward = -0.1 * total_dose
        
        return reward
    
    def get_obs_dim(self) -> int:
        """
        Get observation dimension by running a dummy reset.
        
        Returns:
            obs_dim: Dimensionality of observation space
        """
        obs = self.reset()
        return obs.shape[0]


def build_observation_simple(model: Any) -> np.ndarray:
    """
    Simple observation builder for bacteria simulation.
    
    Constructs a flat observation vector from model state:
    - Current population count
    - Average resistance traits
    - Food level
    - Antibiotic concentrations
    - Time step
    
    Args:
        model: Mesa bacteria model instance
    
    Returns:
        obs: Flat observation vector as np.ndarray
    """
    obs_parts = []
    
    # Population count (normalized by initial population)
    population = len(model.agent_set)
    obs_parts.append(population / 100.0)  # Normalize
    
    # Average traits (if population exists)
    if population > 0:
        avg_enzyme = np.mean([b.enzyme for b in model.agent_set])
        avg_efflux = np.mean([b.efflux for b in model.agent_set])
        avg_membrane = np.mean([b.membrane for b in model.agent_set])
        avg_repair = np.mean([b.repair for b in model.agent_set])
    else:
        avg_enzyme = avg_efflux = avg_membrane = avg_repair = 0.0
    
    obs_parts.extend([avg_enzyme, avg_efflux, avg_membrane, avg_repair])
    
    # Total food level (normalized)
    total_food = np.sum(model.food_field)
    obs_parts.append(total_food / 1000.0)  # Normalize
    
    # Average antibiotic concentrations for each type
    for ab_field in model.antibiotic_fields.values():
        avg_ab = np.mean(ab_field)
        obs_parts.append(avg_ab)
    
    # Time step (normalized)
    obs_parts.append(model.step_count / 1000.0)
    
    return np.array(obs_parts, dtype=np.float32)
