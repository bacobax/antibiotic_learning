"""
Reward modules for antibiotic learning environment.

Provides modular, type-consistent reward computation with proper tensor/number handling.
All modules return Python floats (not tensors) for consistency with RL agents.
"""

from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
import numpy as np
from simulation.simulation_config import TOX_TIMES_DOSE_MAX


class AgeNormalizer(nn.Module):
    """
    Normalizes rewards based on measurement age (staleness).
    
    Applies age-dependent decay to ensure fresher measurements are valued higher.
    Different decay functions can be applied: linear, log, or sqrt.
    """
    def __init__(self, norm_type: str = "sqrt"):
        super(AgeNormalizer, self).__init__()
        self.norm_type = norm_type
        if norm_type not in ["linear", "log", "sqrt"]:
            raise ValueError(f"Unknown normalization type: {norm_type}")

    def forward(self, x: Union[torch.Tensor, float], age: Union[int, float]) -> torch.Tensor:
        """
        Args:
            x: Reward value (tensor or float)
            age: Age of measurement in timesteps (int or float)
            
        Returns:
            Normalized reward as torch.Tensor
        """
        # Ensure x is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.to(torch.float32)
        
        # Convert age to tensor
        age_t = torch.tensor(float(age), dtype=torch.float32)
        
        if self.norm_type == "linear":
            # G(a) = 1 + a
            normalizer = 1.0 + age_t
        elif self.norm_type == "log":
            # G(a) = 1 + log(1 + a)
            normalizer = 1.0 + torch.log1p(age_t)
        elif self.norm_type == "sqrt":
            # G(a) = 1 + sqrt(a)
            normalizer = 1.0 + torch.sqrt(torch.clamp(age_t, min=0.0))
        
        normalized = x / torch.clamp(normalizer, min=1e-6)
        return normalized


class PopulationReward(nn.Module):
    """
    Computes reward based on population closeness to target.
    
    Encourages maintaining population near target P*, not at zero.
    Uses age normalization to devalue stale population measurements.
    """
    def __init__(
        self,
        target_population: float,
        population_norm: float,
        aging_type: str = "sqrt",
    ):
        super(PopulationReward, self).__init__()
        self.target_population = float(target_population)
        self.population_norm = float(population_norm)
        self.age_normalizer = AgeNormalizer(aging_type)

    def forward(
        self,
        last_count_obs: Optional[Union[int, float]],
        age: Union[int, float],
    ) -> float:
        """
        Args:
            last_count_obs: Observed population count (int/float), or None if no measurement
            age: Age of population measurement in timesteps
            
        Returns:
            Population reward as Python float
        """
        if last_count_obs is None:
            # No population data available → harsh penalty for blind action
            return -0.5
        
        # Compute gap from target
        gap = float(last_count_obs) - self.target_population
        pop_term = gap / max(1.0, self.population_norm)
        
        # Apply age-based decay
        pop_term_tensor = torch.tensor(pop_term, dtype=torch.float32)
        pop_term_tensor = self.age_normalizer(pop_term_tensor, age)
        
        # Clip to valid range
        pop_term_tensor = torch.clamp(pop_term_tensor, min=-1.0, max=1.0)
        
        return float(pop_term_tensor.item())


class CostReward(nn.Module):
    """
    Computes negative reward (penalty) for action costs.
    
    Accounts for:
      - Dose costs (proportional to dose amounts)
      - Sequencing costs (fixed one-time cost)
    """
    def __init__(self, dose_cost_per_unit: float = 0.2):
        super(CostReward, self).__init__()
        self.dose_cost_per_unit = float(dose_cost_per_unit)

    def dose_cost(self, dose_vector: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Args:
            dose_vector: Dose amounts [A]
            
        Returns:
            Cost penalty as Python float (negative value)
        """
        if isinstance(dose_vector, torch.Tensor):
            dose_vector = dose_vector.cpu().numpy()
        
        total_dose = float(np.sum(dose_vector))
        cost = total_dose * self.dose_cost_per_unit
        return -cost  # Return as penalty (negative)

    def sequencing_cost(self, cost: float = 1.0) -> float:
        """
        Args:
            cost: Sequencing cost amount
            
        Returns:
            Cost penalty as Python float (negative value)
        """
        return -float(cost)




class SurvivalBonusReward(nn.Module):
    """
    Per-step survival bonus to encourage longer episodes.
    
    Provides a small positive reward for each step the agent keeps the simulation alive.
    Can use different scaling strategies: constant, linear, or exponential.
    """
    def __init__(
        self,
        base_bonus: float = 0.01,
        scaling_type: str = "constant",
        scaling_factor: float = 1.0,
        max_bonus: float = 0.1,
    ):
        """
        Args:
            base_bonus: Base survival bonus per step
            scaling_type: How bonus scales over time ("constant", "linear", "exponential")
            scaling_factor: Multiplier for scaling (used in linear/exponential modes)
            max_bonus: Maximum bonus cap (prevents unbounded growth)
        """
        super(SurvivalBonusReward, self).__init__()
        self.base_bonus = float(base_bonus)
        self.scaling_type = scaling_type
        self.scaling_factor = float(scaling_factor)
        self.max_bonus = float(max_bonus)
        
        if scaling_type not in ["constant", "linear", "exponential"]:
            raise ValueError(f"Unknown scaling type: {scaling_type}")

    def forward(self, timestep: int) -> float:
        """
        Compute survival bonus for current timestep.
        
        Args:
            timestep: Current episode timestep
            
        Returns:
            Survival bonus as Python float (positive reward)
        """
        if self.scaling_type == "constant":
            bonus = self.base_bonus
        elif self.scaling_type == "linear":
            # Bonus increases linearly with time: base * (1 + factor * t)
            bonus = self.base_bonus * (1.0 + self.scaling_factor * timestep)
        elif self.scaling_type == "exponential":
            # Bonus grows exponentially but slowly: base * exp(factor * t / 1000)
            import math
            bonus = self.base_bonus * math.exp(self.scaling_factor * timestep / 1000.0)
        
        # Cap the bonus to prevent explosion
        bonus = min(bonus, self.max_bonus)
        
        return float(bonus)


class BudgetConservationReward(nn.Module):
    """
    Rewards efficient budget usage to encourage longer episodes.
    
    Penalizes high spending rates and rewards maintaining budget reserves.
    Encourages strategic action timing rather than constant intervention.
    """
    def __init__(
        self,
        weight: float = 0.01,
        spending_penalty_factor: float = 1.0,
        reserve_bonus_threshold: float = 0.5,
        reserve_bonus_magnitude: float = 0.005,
    ):
        """
        Args:
            weight: Overall weight for budget conservation reward
            spending_penalty_factor: Multiplier for spending rate penalty
            reserve_bonus_threshold: Budget fraction threshold for reserve bonus (e.g., 0.5 = 50%)
            reserve_bonus_magnitude: Bonus magnitude when budget is above threshold
        """
        super(BudgetConservationReward, self).__init__()
        self.weight = float(weight)
        self.spending_penalty_factor = float(spending_penalty_factor)
        self.reserve_bonus_threshold = float(reserve_bonus_threshold)
        self.reserve_bonus_magnitude = float(reserve_bonus_magnitude)

    def forward(
        self,
        budget_spent_this_step: float,
        current_budget: float,
        initial_budget: float,
        timestep: int,
    ) -> float:
        """
        Compute budget conservation reward.
        
        Args:
            budget_spent_this_step: Amount spent in current step
            current_budget: Remaining budget
            initial_budget: Starting budget for episode
            timestep: Current episode timestep
            
        Returns:
            Budget conservation reward as Python float
        """
        reward = 0.0
        
        # 1) Spending rate penalty: discourage high spending
        if budget_spent_this_step > 0 and timestep > 0:
            # Normalize spending by initial budget
            spending_rate = budget_spent_this_step / max(1.0, initial_budget)
            reward -= self.spending_penalty_factor * spending_rate
        
        # 2) Reserve bonus: reward maintaining budget above threshold
        if initial_budget > 0:
            budget_fraction = current_budget / initial_budget
            if budget_fraction >= self.reserve_bonus_threshold:
                reward += self.reserve_bonus_magnitude
        
        # 3) Efficiency bonus: reward if we're surviving with low spending
        if timestep > 10:  # Only after initial phase
            avg_spending_per_step = (initial_budget - current_budget) / timestep
            if avg_spending_per_step < (initial_budget / 1000.0):  # Very efficient
                reward += self.reserve_bonus_magnitude * 0.5
        
        return float(reward * self.weight)


# ==========================================================
# NEW MODULAR REWARD SYSTEM
# Following the pseudo-code reward structure with pre/post rewards
# ==========================================================


class KernelPopulationMaintenanceReward(nn.Module):
    """
    Kernel-based population maintenance reward.
    
    Uses kernel functions (Gaussian or Laplace) to compute smooth rewards
    based on distance from target population.
    
    Kernel formulations:
    - Gaussian: exp(-0.5 * (distance/bandwidth)^2)
    - Laplace: exp(-|distance|/bandwidth)
    """
    def __init__(
        self,
        target_population: float,
        kernel_type: str = "gaussian",
        peak_reward: float = 1.0,
        max_penalty: float = 0.0,
        zero_distance: float = 100.0,
    ):
        """
        Args:
            target_population: Target population P*
            kernel_type: "gaussian" or "laplace"
            peak_reward: Peak reward R at distance 0
            max_penalty: Maximum penalty M (minimum value is -M)
            zero_distance: Distance from target where kernel equals 0
        """
        super(KernelPopulationMaintenanceReward, self).__init__()
        self.target_population = float(target_population)
        self.kernel_type = kernel_type.lower()
        self.peak_reward = float(peak_reward)
        self.max_penalty = float(max_penalty)
        self.zero_distance = float(zero_distance)
        
        if self.kernel_type not in ["gaussian", "laplace"]:
            raise ValueError(f"Unknown kernel type: {kernel_type}. Must be 'gaussian' or 'laplace'.")
        if self.zero_distance <= 0.0:
            raise ValueError("zero_distance must be > 0")
        if self.max_penalty < 0.0:
            raise ValueError("max_penalty (M) must be >= 0")
        if (self.peak_reward + self.max_penalty) <= self.max_penalty:
            raise ValueError("peak_reward (R) must be > 0")
    
    def forward(self, population: Union[int, float]) -> float:
        """
        Compute kernel-based population maintenance reward.
        
        Args:
            population: Current population count
            
        Returns:
            Population maintenance reward as Python float in [-M, R]
        """
        import math
        
        pop = float(population)
        distance = abs(pop - self.target_population)
        
        R = self.peak_reward
        M = self.max_penalty
        if self.kernel_type == "gaussian":
            # sigma computed so that kernel equals 0 at ±zero_distance
            # sigma = zero_distance / sqrt(2 * ln((R+M)/M))
            denom = (R + M) / max(M, 1e-12)
            denom = max(denom, 1.0 + 1e-12)  # ensure > 1 for log
            sigma = self.zero_distance / math.sqrt(2.0 * math.log(denom))
            kernel_val = math.exp(-((distance) ** 2) / (2.0 * sigma ** 2))
        else:  # laplace
            # For Laplace: R(pop) = (R+M) * exp(-|d|/b) - M
            # Solve 0 at d = zero_distance => (R+M) * exp(-zd/b) - M = 0
            # exp(-zd/b) = M/(R+M) => b = zd / ln((R+M)/M)
            denom = (R + M) / max(M, 1e-12)
            denom = max(denom, 1.0 + 1e-12)
            b = self.zero_distance / math.log(denom)
            kernel_val = math.exp(-distance / b)
        
        reward = (R + M) * kernel_val - M
        
        return float(reward)


class InformedDosingReward(nn.Module):
    """
    Pre-step reward for DOSE action.
    
    Rewards informed dosing (when COUNT is fresh) and penalizes blind dosing.
    Differentiates between dosing above vs below target population and
    supports gradient-style penalties that scale with dose magnitude and
    population deficit relative to target.
    """
    def __init__(
        self,
        penalty_dosing_under_target: float = 5.0,
        penalty_dosing_under_target_dose_scale: float = 0.0,
        penalty_dosing_under_target_dose_exponent: float = 1.0,
        penalty_dosing_under_target_deficit_scale: float = 0.0,
        penalty_dosing_under_target_deficit_cap: float = 1.0,
        penalty_dosing_under_target_max: Optional[float] = None,
        reward_dosing_above_with_seq: float = 2.0,
        reward_dosing_above_no_seq: float = 1.0,
        penalty_blind_dose: float = 3.0,
        penalty_blind_dose_amount_scale: float = 0.0,
        penalty_blind_dose_amount_exponent: float = 1.0,
        penalty_blind_dose_max: Optional[float] = None,
    ):
        """
        Args:
            penalty_dosing_under_target: Penalty for dosing when population is below target
            reward_dosing_above_with_seq: Reward for dosing above target with sequencing
            reward_dosing_above_no_seq: Reward for dosing above target without sequencing
            penalty_blind_dose: Penalty for dosing without fresh count
        """
        super(InformedDosingReward, self).__init__()
        self.penalty_dosing_under_target = float(penalty_dosing_under_target)
        self.penalty_dosing_under_target_dose_scale = float(penalty_dosing_under_target_dose_scale)
        self.penalty_dosing_under_target_dose_exponent = float(max(1e-6, penalty_dosing_under_target_dose_exponent))
        self.penalty_dosing_under_target_deficit_scale = float(penalty_dosing_under_target_deficit_scale)
        self.penalty_dosing_under_target_deficit_cap = float(max(0.0, penalty_dosing_under_target_deficit_cap))
        self.penalty_dosing_under_target_max = (
            float(penalty_dosing_under_target_max)
            if penalty_dosing_under_target_max is not None
            else None
        )
        self.reward_dosing_above_with_seq = float(reward_dosing_above_with_seq)
        self.reward_dosing_above_no_seq = float(reward_dosing_above_no_seq)
        self.penalty_blind_dose = float(penalty_blind_dose)
        self.penalty_blind_dose_amount_scale = float(penalty_blind_dose_amount_scale)
        self.penalty_blind_dose_amount_exponent = float(max(1e-6, penalty_blind_dose_amount_exponent))
        self.penalty_blind_dose_max = (
            float(penalty_blind_dose_max)
            if penalty_blind_dose_max is not None
            else None
        )
        self.last_breakdown: Dict[str, Any] = {}

    @staticmethod
    def _total_dose(dose_amounts: Optional[Union[np.ndarray, torch.Tensor]]) -> float:
        if dose_amounts is None:
            return 0.0
        if isinstance(dose_amounts, torch.Tensor):
            total = float(torch.clamp(dose_amounts, min=0.0).sum().item())
        else:
            total = float(np.clip(dose_amounts, a_min=0.0, a_max=None).sum())
        return max(0.0, total)
    
    def forward(
        self,
        count_fresh: bool,
        last_count_pop: Optional[float],
        target_pop: float,
        recent_sequencing: bool,
        dose_amounts: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> float:
        """
        Compute DOSE pre-step reward.
        
        Args:
            count_fresh: Whether count is fresh (COUNT_FRESH)
            last_count_pop: Last measured population (None if never counted)
            target_pop: Target population
            recent_sequencing: Whether sequencing is recent
            
        Returns:
            DOSE reward as Python float
        """
        dose_total = self._total_dose(dose_amounts)

        if count_fresh:
            # Informed dosing
            if last_count_pop is not None and last_count_pop < target_pop:
                # Dosing when below target → penalty
                deficit_ratio = 0.0
                if target_pop > 0:
                    deficit = max(0.0, target_pop - float(last_count_pop))
                    deficit_ratio = deficit / max(target_pop, 1e-6)
                    cap = self.penalty_dosing_under_target_deficit_cap
                    if cap > 0.0:
                        deficit_ratio = min(deficit_ratio * cap, cap)
                dose_term = 0.0
                if dose_total > 0.0:
                    dose_term = self.penalty_dosing_under_target_dose_scale * (
                        dose_total ** self.penalty_dosing_under_target_dose_exponent
                    )
                deficit_term = self.penalty_dosing_under_target_deficit_scale * deficit_ratio
                penalty = self.penalty_dosing_under_target + dose_term + deficit_term
                if self.penalty_dosing_under_target_max is not None:
                    penalty = min(penalty, self.penalty_dosing_under_target_max)
                self.last_breakdown = {
                    "mode": "penalty_under_target",
                    "base": self.penalty_dosing_under_target,
                    "dose_term": dose_term,
                    "deficit_term": deficit_term,
                    "deficit_ratio": deficit_ratio,
                    "dose_magnitude": dose_total,
                    "total": penalty,
                }
                return -penalty
            else:
                # Dosing when above target → reward
                if recent_sequencing:
                    reward = self.reward_dosing_above_with_seq
                    self.last_breakdown = {
                        "mode": "reward_above_target_with_seq",
                        "reward": reward,
                        "dose_magnitude": dose_total,
                    }
                    return reward
                else:
                    reward = self.reward_dosing_above_no_seq
                    self.last_breakdown = {
                        "mode": "reward_above_target_no_seq",
                        "reward": reward,
                        "dose_magnitude": dose_total,
                    }
                    return reward
        else:
            # Blind dosing → penalty
            dose_term = 0.0
            if dose_total > 0.0:
                dose_term = self.penalty_blind_dose_amount_scale * (
                    dose_total ** self.penalty_blind_dose_amount_exponent
                )
            penalty = self.penalty_blind_dose + dose_term
            if self.penalty_blind_dose_max is not None:
                penalty = min(penalty, self.penalty_blind_dose_max)
            self.last_breakdown = {
                "mode": "penalty_blind",
                "base": self.penalty_blind_dose,
                "dose_term": dose_term,
                "dose_magnitude": dose_total,
                "total": penalty,
            }
            return -penalty


class SequencingReward(nn.Module):
    """
    Pre-step reward for SEQUENCING action.
    
    Rewards informative sequencing (within timing window) and penalizes redundant sequencing.
    """
    def __init__(
        self,
        seq_already_pending_penalty: float = 2.0,
        informative_seq_reward: float = 1.0,
    ):
        """
        Args:
            seq_already_pending_penalty: Penalty for sequencing when already pending
            informative_seq_reward: Reward for sequencing within timing window
        """
        super(SequencingReward, self).__init__()
        self.seq_already_pending_penalty = float(seq_already_pending_penalty)
        self.informative_seq_reward = float(informative_seq_reward)
    
    def forward(
        self,
        seq_pending: bool,
        t_since_last_seq: float,
        t_min_elapsed_time_seq: float,
        t_max_elapsed_time_seq: float,
    ) -> float:
        """
        Compute SEQUENCING pre-step reward.
        
        Args:
            seq_pending: Whether sequencing is already pending
            t_since_last_seq: Time since last sequencing
            t_min_elapsed_time_seq: Minimum elapsed time for informative sequencing
            t_max_elapsed_time_seq: Maximum elapsed time for informative sequencing
            
        Returns:
            SEQUENCING reward as Python float
        """
        if seq_pending:
            # Redundant sequencing → penalty
            return -self.seq_already_pending_penalty
        else:
            # Check timing window
            if t_min_elapsed_time_seq <= t_since_last_seq <= t_max_elapsed_time_seq:
                # Informative sequencing → reward
                return self.informative_seq_reward
            else:
                # Outside timing window → neutral
                return 0.0


class CountReward(nn.Module):
    """
    Pre-step reward for COUNT action.
    
    Rewards informative counting (within timing window).
    """
    def __init__(
        self,
        informative_count_reward: float = 1.0,
    ):
        """
        Args:
            informative_count_reward: Reward for counting within timing window
        """
        super(CountReward, self).__init__()
        self.informative_count_reward = float(informative_count_reward)
    
    def forward(
        self,
        t_since_last_count: float,
        t_min_elapsed_time_count: float,
        t_max_elapsed_time_count: float,
    ) -> float:
        """
        Compute COUNT pre-step reward.
        
        Args:
            t_since_last_count: Time since last count
            t_min_elapsed_time_count: Minimum elapsed time for informative counting
            t_max_elapsed_time_count: Maximum elapsed time for informative counting
            
        Returns:
            COUNT reward as Python float
        """
        reward = 0.0
        
        # Check timing window
        if t_min_elapsed_time_count <= t_since_last_count <= t_max_elapsed_time_count:
            # Informative counting → reward
            reward += self.informative_count_reward
        
        return reward


class StrategicNoopReward(nn.Module):
    """
    Pre-step reward for NOOP action.
    
    Rewards strategic waiting when population is below target and count is fresh.
    """
    def __init__(
        self,
        strategic_noop_reward: float = 0.5,
    ):
        """
        Args:
            strategic_noop_reward: Reward for strategic NOOP (waiting below target)
        """
        super(StrategicNoopReward, self).__init__()
        self.strategic_noop_reward = float(strategic_noop_reward)
    
    def forward(
        self,
        count_fresh: bool,
        last_count_pop: Optional[float],
        target_pop: float,
    ) -> float:
        """
        Compute NOOP pre-step reward.
        
        Args:
            count_fresh: Whether count is fresh
            last_count_pop: Last measured population (None if never counted)
            target_pop: Target population
            
        Returns:
            NOOP reward as Python float
        """
        if count_fresh and last_count_pop is not None:
            if last_count_pop < target_pop:
                # Strategic waiting → reward
                return self.strategic_noop_reward
        
        return 0.0


class CriticalNoDosePenalty(nn.Module):
    """
    Post-step penalty for NOT dosing when population is critically high.
    """
    def __init__(
        self,
        penalty_critical_no_dose: float = 5.0,
    ):
        """
        Args:
            penalty_critical_no_dose: Penalty for not dosing when critical
        """
        super(CriticalNoDosePenalty, self).__init__()
        self.penalty_critical_no_dose = float(penalty_critical_no_dose)
    
    def forward(
        self,
        count_fresh: bool,
        last_count_pop: Optional[float],
        target_pop: float,
        critical_ratio: float,
        action_was_dose: bool,
    ) -> float:
        """
        Compute critical no-dose penalty.
        
        Args:
            count_fresh: Whether count is fresh
            last_count_pop: Last measured population
            target_pop: Target population
            critical_ratio: Critical population ratio (e.g., 3.0)
            action_was_dose: Whether the action was DOSE
            
        Returns:
            Penalty as Python float (negative or zero)
        """
        if count_fresh and last_count_pop is not None:
            if last_count_pop > critical_ratio * target_pop:
                if not action_was_dose:
                    # Critical population, didn't dose → penalty
                    return -self.penalty_critical_no_dose
        
        return 0.0


class CriticalNoCountPenalty(nn.Module):
    """
    Post-step penalty for letting count data become stale.
    """
    def __init__(
        self,
        penalty_critical_no_count: float = 2.0,
    ):
        """
        Args:
            penalty_critical_no_count: Penalty for letting count become stale
        """
        super(CriticalNoCountPenalty, self).__init__()
        self.penalty_critical_no_count = float(penalty_critical_no_count)
    
    def forward(
        self,
        t_since_last_count: float,
        max_count_window: float,
    ) -> float:
        """
        Compute critical no-count penalty.
        
        Args:
            t_since_last_count: Time since last count
            max_count_window: Maximum allowed time without counting
            
        Returns:
            Penalty as Python float (negative or zero)
        """
        if t_since_last_count > max_count_window:
            # Count data is stale → penalty
            return -self.penalty_critical_no_count
        
        return 0.0


class ExtinctionPenalty(nn.Module):
    """
    Post-step penalty for population extinction.
    """
    def __init__(
        self,
        big_penalty: float = 50.0,
    ):
        """
        Args:
            big_penalty: Penalty for population collapse
        """
        super(ExtinctionPenalty, self).__init__()
        self.big_penalty = float(big_penalty)
    
    def forward(self, population: Union[int, float]) -> float:
        """
        Compute extinction penalty.
        
        Args:
            population: Current population
            
        Returns:
            Penalty as Python float (negative or zero)
        """
        if population <= 0:
            return -self.big_penalty
        
        return 0.0


