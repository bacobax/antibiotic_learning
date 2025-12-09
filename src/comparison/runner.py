"""
Agent runners for comparison experiments.
"""

from typing import Optional
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .base_agent import BaseComparisonAgent, ActionType
from .metrics import RunMetrics
from simulation.model import BacteriaModel
from simulation.simulation_config import ANTIBIOTIC_TYPES
from rl.config_loader import load_config
from rl.agent import RLAgent
from rl.training_utils import _create_environment
from rl.env_wrapper import (
    ACTION_NOOP,
    ACTION_COUNT_BACTERIA,
    ACTION_SEQUENCING,
    ACTION_DOSE,
)


class _ConsoleLogger:
    """Minimal logger compatible with training utilities."""
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    def log_info(self, message: str) -> None:
        if self.verbose:
            print(f"[env] {message}")
    def log_debug(self, message: str) -> None:
        if self.verbose:
            print(f"[env-debug] {message}")


def run_agent(
    agent: BaseComparisonAgent,
    model: BacteriaModel,
    steps: int,
    target_population: int,
    tolerance: float = 0.15,
    zero_distance: float = 50.0,
    population_cap: int = 1000,
    verbose: bool = False,
) -> RunMetrics:
    """
    Run any agent that extends BaseComparisonAgent.
    
    Args:
        agent: The agent to run
        model: BacteriaModel instance
        steps: Number of steps to run
        target_population: Target population for metrics
        tolerance: Tolerance band for target tracking
        zero_distance: Zero distance for kernel metrics
        population_cap: Stop if population exceeds this value
        verbose: Print progress
    
    Returns:
        RunMetrics with collected data
    """
    metrics = RunMetrics(agent_name=agent.name)
    
    # Record initial state
    metrics.populations.append(len(model.agent_set))
    metrics.budget_history.append(float(agent.budget_remaining))
    
    # Run simulation step by step
    for step in tqdm(range(steps), desc=agent.name, leave=False):
        # Get current population
        population = len(model.agent_set)
        
        # Check population cap
        if population > population_cap:
            metrics.early_termination_reason = f"Population exceeded cap ({population} > {population_cap})"
            break
        
        # Get action from agent
        action_type, dose_strength = agent.step(population)
        action_name = action_type.name
        
        # Apply action to model
        if action_type == ActionType.DOSE and dose_strength > 0:
            antibiotic = model.current_antibiotic or list(ANTIBIOTIC_TYPES.keys())[0]
            amount = dose_strength * agent.dose_scale
            model.apply_antibiotic(antibiotic, amount)
        
        # Step the model
        model.step()
        
        # Record data
        pop = len(model.agent_set)
        metrics.populations.append(pop)
        metrics.actions.append(action_name)
        metrics.budget_history.append(float(agent.budget_remaining))
        
        if action_type == ActionType.DOSE:
            metrics.dose_steps.append(step)
            metrics.dose_amounts.append(dose_strength)
        elif action_type == ActionType.COUNT:
            metrics.count_steps.append(step)
        elif action_type == ActionType.SEQUENCE:
            metrics.sequence_steps.append(step)
        elif action_type == ActionType.NOOP:
            metrics.noop_steps.append(step)
        
        if verbose and (step + 1) % 50 == 0:
            print(f"[{agent.name}] Step {step+1}: pop={pop}, budget={agent.budget_remaining:.1f}")
    
    metrics.steps = step + 1
    metrics.action_counts = dict(agent.action_counts)
    metrics.compute_summary(target_population, tolerance, zero_distance)
    
    return metrics


def run_rl_agent(
    config_path: str,
    checkpoint_path: str,
    steps: int,
    target_population: int,
    initial_budget: float,
    tolerance: float = 0.15,
    zero_distance: float = 50.0,
    population_cap: int = 1000,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> RunMetrics:
    """Run an RL agent and collect metrics."""
    checkpoint_name = Path(checkpoint_path).stem
    metrics = RunMetrics(agent_name=f"RL ({checkpoint_name})")
    
    # Load config
    config = load_config(config_path)
    
    # Override some settings for comparison
    config.environment.max_steps = steps + 10
    config.environment.budget_init = initial_budget
    if hasattr(config.environment.rewards, 'population_maintenance') and config.environment.rewards.population_maintenance is not None:
        config.environment.rewards.population_maintenance.target_population = target_population
    if hasattr(config.environment.rewards, 'population') and config.environment.rewards.population is not None:
        config.environment.rewards.population.target_population = target_population
    if config.environment.rewards.early_termination is not None:
        config.environment.rewards.early_termination.enabled = False
    
    # Create the model
    model = BacteriaModel()
    
    def model_factory(**kwargs):
        model.reset()
        return model
    
    logger = _ConsoleLogger(verbose=verbose)
    env = _create_environment(config, logger, mesa_model_factory=model_factory)
    
    # Load agent from checkpoint
    if not Path(checkpoint_path).exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    agent = RLAgent.load_agent_from_checkpoint(
        checkpoint_path,
        env=env,
        device="cpu",
    )
    
    # Reset environment
    obs = env.reset()
    agent.start_episode()
    
    # Record initial state
    metrics.populations.append(len(model.agent_set))
    metrics.budget_history.append(float(env.budget))
    
    action_counts = {action.name: 0 for action in ActionType}
    
    action_map = {
        ACTION_NOOP: "NOOP",
        ACTION_COUNT_BACTERIA: "COUNT",
        ACTION_SEQUENCING: "SEQUENCE",
        ACTION_DOSE: "DOSE",
    }
    
    # Run simulation
    for step in tqdm(range(steps), desc=f"RL ({checkpoint_name})", leave=False):
        # Check population cap
        pop = len(model.agent_set)
        if pop > population_cap:
            metrics.early_termination_reason = f"Population exceeded cap ({pop} > {population_cap})"
            break
        
        # Select action
        (
            a_disc, a_cont, logp_disc, logp_cont, value,
            pred_next_pop, h_prev, action_mask,
            prev_action_onehot, prev_action_cont, prev_pred_next_pop
        ) = agent.select_action(obs)
        
        discrete_action = a_disc.item()
        continuous_action = a_cont.cpu().numpy()[0]
        
        action_name = action_map.get(discrete_action, "UNKNOWN")
        action_counts[action_name] += 1
        
        # Step environment
        obs, reward, done, info = env.step(discrete_action, continuous_action)
        
        # Record data
        pop = len(model.agent_set)
        metrics.populations.append(pop)
        metrics.actions.append(action_name)
        metrics.budget_history.append(float(env.budget))
        
        if discrete_action == ACTION_DOSE:
            metrics.dose_steps.append(step)
            dose_amount = float(np.sum(continuous_action))
            metrics.dose_amounts.append(dose_amount)
        elif discrete_action == ACTION_COUNT_BACTERIA:
            metrics.count_steps.append(step)
        elif discrete_action == ACTION_SEQUENCING:
            metrics.sequence_steps.append(step)
        elif discrete_action == ACTION_NOOP:
            metrics.noop_steps.append(step)
        
        if verbose and (step + 1) % 50 == 0:
            print(f"[RL] Step {step+1}: pop={pop}, budget={env.budget:.1f}, action={action_name}")
        
        if done:
            metrics.early_termination_reason = "Environment done signal"
            break
    
    metrics.steps = step + 1
    metrics.action_counts = action_counts
    metrics.compute_summary(target_population, tolerance, zero_distance)
    
    return metrics