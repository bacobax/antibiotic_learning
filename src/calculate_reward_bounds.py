#!/usr/bin/env python3
"""
Calculate the theoretical minimum and maximum episode rewards based on a YAML training config.

This script analyzes the reward configuration and computes:
- Maximum possible reward per step (best-case scenario)
- Minimum possible reward per step (worst-case scenario)
- Average expected reward bounds over an episode

Takes into account:
- Mutual exclusivity of action rewards (only one action per step)
- Episode termination mechanics (extinction ends episode immediately)
- Proper stacking of compatible reward components

Usage:
    python calculate_reward_bounds.py <config_path>
    python calculate_reward_bounds.py rl/configs/training_config_margin.yaml
"""

import argparse
import yaml
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class ActionRewardBounds:
    """Bounds for a single action type."""
    name: str
    min_reward: float
    max_reward: float


@dataclass
class RewardBounds:
    """Container for reward bound calculations."""
    # Per-step bounds
    min_per_step: float
    max_per_step: float
    # Episode bounds (full duration)
    min_episode: float
    max_episode: float
    # Early termination scenarios
    min_early_termination: float  # Worst case: extinction at step 1
    max_early_termination: float  # Best case before early term
    # Breakdown info
    action_bounds: List[ActionRewardBounds] = field(default_factory=list)
    background_min: float = 0.0  # Always-applied rewards (kernel, survival)
    background_max: float = 0.0
    post_step_penalty_max: float = 0.0  # Critical penalties
    
    def __str__(self) -> str:
        action_lines = ""
        for ab in self.action_bounds:
            action_lines += f"\n║    • {ab.name:<12} Min: {ab.min_reward:>8.2f}  Max: {ab.max_reward:>8.2f}               ║"
        
        return f"""
╔══════════════════════════════════════════════════════════════════════╗
║                      REWARD BOUNDS ANALYSIS                          ║
║         (Accounting for mutual exclusivity & termination)            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Background Rewards (applied every step):                            ║
║    • Max (kernel + survival at target):    {self.background_max:>10.2f}                ║
║    • Min (kernel + survival far from target): {self.background_min:>10.2f}             ║
╠══════════════════════════════════════════════════════════════════════╣
║  Action Rewards (mutually exclusive - only ONE per step):            ║{action_lines}
╠══════════════════════════════════════════════════════════════════════╣
║  Post-Step Penalties (critical state):     {self.post_step_penalty_max:>10.2f}                ║
╠══════════════════════════════════════════════════════════════════════╣
║  Per-Step Bounds (background + best/worst action + penalties):       ║
║    • Maximum reward per step:              {self.max_per_step:>10.2f}                ║
║    • Minimum reward per step:              {self.min_per_step:>10.2f}                ║
╠══════════════════════════════════════════════════════════════════════╣
║  Full Episode Bounds (if agent survives all steps):                  ║
║    • Maximum total episode reward:       {self.max_episode:>12.2f}              ║
║    • Minimum total episode reward:       {self.min_episode:>12.2f}              ║
╠══════════════════════════════════════════════════════════════════════╣
║  Early Termination Scenarios:                                        ║
║    • Worst (extinction at step 1):       {self.min_early_termination:>12.2f}              ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def calculate_kernel_reward_bounds(reward_cfg: Dict[str, Any]) -> tuple[float, float]:
    """
    Calculate bounds for kernel-based population maintenance reward.
    NOTE: This reward is ONLY applied when COUNT action is taken (count_result_landed).
    It is NOT a background reward - it's tied to the COUNT action.
    
    Returns: (min_reward, max_reward)
    """
    pop_maintenance = reward_cfg.get('population_maintenance', {})
    if not pop_maintenance.get('enabled', False):
        return 0.0, 0.0
    
    peak_reward = pop_maintenance.get('kernel_peak_reward', 1.0)  # R
    max_penalty = pop_maintenance.get('kernel_max_penalty', 0.0)  # M
    
    # At target population: reward = R (peak)
    # Far from target: reward = -M (minimum)
    return -max_penalty, peak_reward


def calculate_survival_bonus_bounds(reward_cfg: Dict[str, Any], max_steps: int) -> tuple[float, float, float]:
    """
    Calculate bounds for survival bonus.
    This IS a BACKGROUND reward applied every step regardless of action.
    
    Returns: (min_per_step, max_per_step, avg_per_step)
    """
    survival_cfg = reward_cfg.get('survival_bonus', {})
    if not survival_cfg.get('enabled', False):
        return 0.0, 0.0, 0.0
    
    base_bonus = survival_cfg.get('base_bonus', 0.01)
    scaling_type = survival_cfg.get('scaling_type', 'constant')
    scaling_factor = survival_cfg.get('scaling_factor', 1.0)
    max_bonus = survival_cfg.get('max_bonus', 0.1)
    
    if scaling_type == 'constant':
        return base_bonus, base_bonus, base_bonus
    elif scaling_type == 'linear':
        # Min at step 0: base_bonus
        # Max at max_steps: min(base_bonus * (1 + scaling_factor * max_steps), max_bonus)
        max_at_end = base_bonus * (1.0 + scaling_factor * max_steps)
        actual_max = min(max_at_end, max_bonus)
        avg = (base_bonus + actual_max) / 2
        return base_bonus, actual_max, avg
    elif scaling_type == 'exponential':
        # Min at step 0: base_bonus
        # Max at max_steps: min(base_bonus * exp(scaling_factor * max_steps / 1000), max_bonus)
        max_at_end = base_bonus * math.exp(scaling_factor * max_steps / 1000.0)
        actual_max = min(max_at_end, max_bonus)
        avg = (base_bonus + actual_max) / 2
        return base_bonus, actual_max, avg
    
    return base_bonus, max_bonus, (base_bonus + max_bonus) / 2


def calculate_dosing_action_bounds(
    reward_cfg: Dict[str, Any], 
    k_doses: int,
    sigmoid_scale_factor: float
) -> ActionRewardBounds:
    """
    Calculate bounds for DOSE action reward.
    This is MUTUALLY EXCLUSIVE with other action rewards.
    Does NOT receive kernel reward.
    
    Returns: ActionRewardBounds for dosing
    """
    dosing_cfg = reward_cfg.get('informed_dosing', {})
    
    # Best case: dosing above target with sequencing
    max_reward = dosing_cfg.get('reward_dosing_above_with_seq', 2.0)
    
    # Worst case: blind dosing with max dose
    base_penalty = dosing_cfg.get('penalty_blind_dose', 3.0)
    amount_scale = dosing_cfg.get('penalty_blind_dose_amount_scale', 0.0)
    amount_exponent = dosing_cfg.get('penalty_blind_dose_amount_exponent', 1.0)
    max_penalty_cap = dosing_cfg.get('penalty_blind_dose_max', None)
    
    # Maximum dose amount (all k_doses at max)
    max_dose_total = k_doses * sigmoid_scale_factor
    
    dose_term = amount_scale * (max_dose_total ** amount_exponent)
    worst_blind_penalty = base_penalty + dose_term
    if max_penalty_cap is not None:
        worst_blind_penalty = min(worst_blind_penalty, max_penalty_cap)
    
    # Also consider dosing under target penalty
    base_under = dosing_cfg.get('penalty_dosing_under_target', 5.0)
    dose_scale_under = dosing_cfg.get('penalty_dosing_under_target_dose_scale', 0.0)
    dose_exp_under = dosing_cfg.get('penalty_dosing_under_target_dose_exponent', 1.0)
    deficit_scale = dosing_cfg.get('penalty_dosing_under_target_deficit_scale', 0.0)
    deficit_cap = dosing_cfg.get('penalty_dosing_under_target_deficit_cap', 1.0)
    under_max_cap = dosing_cfg.get('penalty_dosing_under_target_max', None)
    
    dose_term_under = dose_scale_under * (max_dose_total ** dose_exp_under)
    deficit_term = deficit_scale * deficit_cap  # Max deficit
    worst_under_penalty = base_under + dose_term_under + deficit_term
    if under_max_cap is not None:
        worst_under_penalty = min(worst_under_penalty, under_max_cap)
    
    min_reward = -max(worst_blind_penalty, worst_under_penalty)
    
    return ActionRewardBounds(name="DOSE", min_reward=min_reward, max_reward=max_reward)


def calculate_sequencing_action_bounds(reward_cfg: Dict[str, Any]) -> ActionRewardBounds:
    """
    Calculate bounds for SEQUENCING action reward.
    This is MUTUALLY EXCLUSIVE with other action rewards.
    Does NOT receive kernel reward.
    """
    seq_cfg = reward_cfg.get('sequencing', {})
    
    max_reward = seq_cfg.get('informative_seq_reward', 1.0)
    min_reward = -seq_cfg.get('seq_already_pending_penalty', 2.0)
    
    # If penalty is 0, worst case is neutral (0)
    if min_reward == 0:
        min_reward = 0.0
    
    return ActionRewardBounds(name="SEQUENCING", min_reward=min_reward, max_reward=max_reward)


def calculate_counting_action_bounds(
    reward_cfg: Dict[str, Any],
    kernel_min: float,
    kernel_max: float
) -> ActionRewardBounds:
    """
    Calculate bounds for COUNT action reward.
    This is MUTUALLY EXCLUSIVE with other action rewards.
    COUNT action ALSO receives the kernel population maintenance reward!
    
    Args:
        reward_cfg: Reward configuration
        kernel_min: Minimum kernel reward (far from target)
        kernel_max: Maximum kernel reward (at target)
    """
    count_cfg = reward_cfg.get('counting', {})
    
    base_max_reward = count_cfg.get('informative_count_reward', 1.0)
    cost_penalty = count_cfg.get('cost_penalty', 0.0)
    
    # COUNT gets: base count reward + kernel reward
    # Best case: informative count + kernel at target
    max_reward = base_max_reward + kernel_max
    
    # Worst case: count outside informative window (0) + cost + kernel far from target
    min_reward = -cost_penalty + kernel_min
    
    return ActionRewardBounds(name="COUNT", min_reward=min_reward, max_reward=max_reward)


def calculate_noop_action_bounds(reward_cfg: Dict[str, Any]) -> ActionRewardBounds:
    """
    Calculate bounds for NOOP action reward.
    This is MUTUALLY EXCLUSIVE with other action rewards.
    Does NOT receive kernel reward.
    """
    noop_cfg = reward_cfg.get('noop', {})
    
    max_reward = noop_cfg.get('strategic_noop_reward', 0.5)
    # NOOP has no penalty, worst case is 0
    min_reward = 0.0
    
    return ActionRewardBounds(name="NOOP", min_reward=min_reward, max_reward=max_reward)


def calculate_critical_penalty_bounds(reward_cfg: Dict[str, Any]) -> float:
    """
    Calculate maximum critical penalties (post-step).
    These can stack with action rewards but are conditional on state.
    
    Returns: Maximum total penalty (as negative value)
    """
    critical_cfg = reward_cfg.get('critical_penalties', {})
    
    no_dose_penalty = critical_cfg.get('penalty_critical_no_dose', 5.0)
    no_count_penalty = critical_cfg.get('penalty_critical_no_count', 2.0)
    
    # These are mutually exclusive in practice:
    # - penalty_critical_no_dose: applies when critical AND didn't dose
    # - penalty_critical_no_count: applies when count is stale
    # In worst case, both could apply in same step
    return -(no_dose_penalty + no_count_penalty)


def calculate_termination_penalties(reward_cfg: Dict[str, Any]) -> tuple[float, float]:
    """
    Calculate early termination penalties.
    These are ONE-TIME penalties that END the episode.
    
    Returns: (base_termination_penalty, extinction_penalty)
    """
    term_cfg = reward_cfg.get('early_termination', {})
    
    if not term_cfg.get('enabled', False):
        return 0.0, 0.0
    
    penalty = term_cfg.get('penalty', 10.0)
    extinction_penalty = term_cfg.get('extinction_penalty', 50.0)
    
    return penalty, extinction_penalty


def calculate_reward_bounds(config: Dict[str, Any]) -> RewardBounds:
    """
    Calculate the theoretical min/max rewards based on config.
    
    Properly accounts for:
    - Mutual exclusivity of action rewards (only one action per step)
    - Kernel reward ONLY applies on COUNT action (not a background reward)
    - Survival bonus is the only true background reward
    - Post-step penalties that can stack with action rewards
    - Episode termination (extinction ends episode, no further rewards)
    
    Args:
        config: Loaded YAML configuration
        
    Returns:
        RewardBounds with calculated values
    """
    env_cfg = config.get('environment', {})
    model_cfg = config.get('model', {})
    reward_cfg = env_cfg.get('rewards', {})
    
    max_steps = env_cfg.get('max_steps', 2048)
    k_doses = env_cfg.get('k_doses', 3)
    sigmoid_scale_factor = model_cfg.get('sigmoid_scale_factor', 0.1)
    
    # ========================================
    # 1. KERNEL REWARD (only on COUNT action)
    # ========================================
    kernel_min, kernel_max = calculate_kernel_reward_bounds(reward_cfg)
    
    # ========================================
    # 2. SURVIVAL BONUS (true background, every step)
    # ========================================
    survival_min, survival_max, survival_avg = calculate_survival_bonus_bounds(reward_cfg, max_steps)
    
    # Only survival is a true background reward now
    background_min = survival_min
    background_max = survival_max
    background_avg = survival_avg
    
    # ========================================
    # 3. ACTION REWARDS (mutually exclusive)
    # ========================================
    # Note: COUNT action includes kernel reward
    action_bounds = [
        calculate_dosing_action_bounds(reward_cfg, k_doses, sigmoid_scale_factor),
        calculate_sequencing_action_bounds(reward_cfg),
        calculate_counting_action_bounds(reward_cfg, kernel_min, kernel_max),  # Includes kernel!
        calculate_noop_action_bounds(reward_cfg),
    ]
    
    # Best action = highest max reward among all actions
    best_action_max = max(ab.max_reward for ab in action_bounds)
    # Worst action = lowest min reward among all actions
    worst_action_min = min(ab.min_reward for ab in action_bounds)
    
    # ========================================
    # 4. POST-STEP PENALTIES (conditional)
    # ========================================
    critical_penalty_max = calculate_critical_penalty_bounds(reward_cfg)
    
    # ========================================
    # 5. TERMINATION PENALTIES (one-time, ends episode)
    # ========================================
    base_term_penalty, extinction_penalty = calculate_termination_penalties(reward_cfg)
    
    # ========================================
    # CALCULATE PER-STEP BOUNDS
    # ========================================
    # Best step: survival + best_action (COUNT with kernel at target)
    max_per_step = background_max + best_action_max
    
    # Worst step: survival + worst_action + critical penalties
    min_per_step = background_min + worst_action_min + critical_penalty_max
    
    # ========================================
    # CALCULATE EPISODE BOUNDS
    # ========================================
    
    # MAXIMUM EPISODE: Perfect play for all max_steps
    # Every step: survival bonus + best action reward
    max_episode = max_per_step * max_steps
    
    # MINIMUM EPISODE (surviving all steps):
    # Every step: survival (still alive) + worst action + critical penalties
    min_episode_surviving = min_per_step * max_steps
    
    # MINIMUM EPISODE (early termination):
    # Worst case: extinction at step 1
    # Step 1 reward + extinction penalty (episode ends, no more rewards)
    min_early_termination = min_per_step + (-(base_term_penalty + extinction_penalty))
    
    # The true minimum is the worse of these scenarios
    min_episode = min(min_episode_surviving, min_early_termination)
    
    # ========================================
    # PRINT DETAILED BREAKDOWN
    # ========================================
    print("\n" + "="*70)
    print("REWARD COMPONENT BREAKDOWN")
    print("="*70)
    
    print(f"\n{'KERNEL REWARD (only on COUNT action, not background):'}")
    print(f"  {'Population Kernel':<25} Min: {kernel_min:>8.2f}  Max: {kernel_max:>8.2f}")
    
    print(f"\n{'BACKGROUND REWARDS (applied every step regardless of action):'}")
    print(f"  {'Survival Bonus':<25} Min: {survival_min:>8.2f}  Max: {survival_max:>8.2f}")
    print(f"  {'─'*50}")
    print(f"  {'TOTAL BACKGROUND':<25} Min: {background_min:>8.2f}  Max: {background_max:>8.2f}")
    
    print(f"\n{'ACTION REWARDS (mutually exclusive - ONE per step):'}")
    for ab in action_bounds:
        note = " (includes kernel!)" if ab.name == "COUNT" else ""
        print(f"  {ab.name:<25} Min: {ab.min_reward:>8.2f}  Max: {ab.max_reward:>8.2f}{note}")
    print(f"  {'─'*50}")
    print(f"  {'Best action':<25}              Max: {best_action_max:>8.2f}")
    print(f"  {'Worst action':<25} Min: {worst_action_min:>8.2f}")
    
    print(f"\n{'POST-STEP PENALTIES (conditional on state):'}")
    print(f"  {'Critical penalties':<25} Max: {critical_penalty_max:>8.2f}")
    
    print(f"\n{'TERMINATION PENALTIES (one-time, ends episode):'}")
    print(f"  {'Base termination':<25}       {-base_term_penalty:>8.2f}")
    print(f"  {'Extinction':<25}       {-extinction_penalty:>8.2f}")
    print(f"  {'─'*50}")
    print(f"  {'TOTAL (worst case)':<25}       {-(base_term_penalty + extinction_penalty):>8.2f}")
    
    print(f"\n{'Configuration:'}")
    print(f"  • Max steps: {max_steps}")
    print(f"  • K doses: {k_doses}")
    print(f"  • Sigmoid scale factor: {sigmoid_scale_factor}")
    print("="*70)
    
    return RewardBounds(
        min_per_step=min_per_step,
        max_per_step=max_per_step,
        min_episode=min_episode,
        max_episode=max_episode,
        min_early_termination=min_early_termination,
        max_early_termination=max_per_step,  # Best single step before termination
        action_bounds=action_bounds,
        background_min=background_min,
        background_max=background_max,
        post_step_penalty_max=critical_penalty_max,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Calculate theoretical reward bounds from RL training config',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python calculate_reward_bounds.py rl/configs/training_config_margin.yaml
    python calculate_reward_bounds.py /path/to/config.yaml
        """
    )
    parser.add_argument(
        'config_path',
        type=str,
        help='Path to YAML training configuration file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed breakdown of all reward components'
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config_path)
    if not config_path.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent
        config_path = script_dir / args.config_path
        if not config_path.exists():
            print(f"Error: Config file not found: {args.config_path}")
            return 1
    
    print(f"\nLoading config from: {config_path}")
    config = load_config(config_path)
    
    bounds = calculate_reward_bounds(config)
    print(bounds)
    
    return 0


if __name__ == '__main__':
    exit(main())
