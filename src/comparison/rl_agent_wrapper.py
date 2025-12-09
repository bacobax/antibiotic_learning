"""
Wrapper to make RL agent compatible with the comparison framework.
"""

from typing import Tuple
import numpy as np
from .base_agent import BaseComparisonAgent, ActionType
from rl.agent import RLAgent
from rl.env_wrapper import (
    PetriEnvWrapper,
    ACTION_NOOP,
    ACTION_COUNT_BACTERIA,
    ACTION_SEQUENCING,
    ACTION_DOSE,
)


class RLAgentWrapper(BaseComparisonAgent):
    """Wrapper to make RL agent compatible with the comparison framework."""
    
    def __init__(
        self,
        rl_agent: RLAgent,
        env: PetriEnvWrapper,
        checkpoint_name: str,
        target_population: int,
        total_steps: int,
        initial_budget: float,
        **kwargs
    ):
        super().__init__(
            name=f"RL Agent ({checkpoint_name})",
            target_population=target_population,
            total_steps=total_steps,
            initial_budget=initial_budget,
            **kwargs
        )
        self.rl_agent = rl_agent
        self.env = env
        self.last_obs = None
        self.action_map = {
            ACTION_NOOP: ActionType.NOOP,
            ACTION_COUNT_BACTERIA: ActionType.COUNT,
            ACTION_SEQUENCING: ActionType.SEQUENCE,
            ACTION_DOSE: ActionType.DOSE,
        }
    
    def reset(self, obs: np.ndarray):
        """Reset the RL agent with initial observation."""
        self.last_obs = obs
        self.rl_agent.start_episode()
    
    def select_action(self, population: int) -> Tuple[ActionType, float]:
        """Select action using the RL policy."""
        if self.last_obs is None:
            return ActionType.NOOP, 0.0
        
        (
            a_disc, a_cont, logp_disc, logp_cont, value,
            pred_next_pop, h_prev, action_mask,
            prev_action_onehot, prev_action_cont, prev_pred_next_pop
        ) = self.rl_agent.select_action(self.last_obs)
        
        discrete_action = a_disc.item()
        continuous_action = a_cont.cpu().numpy()[0]
        
        action_type = self.action_map.get(discrete_action, ActionType.NOOP)
        dose_strength = float(np.sum(continuous_action)) if action_type == ActionType.DOSE else 0.0
        
        return action_type, dose_strength
    
    def update_obs(self, obs: np.ndarray):
        """Update the observation for next action selection."""
        self.last_obs = obs