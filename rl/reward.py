"""
Reward modules for antibiotic learning environment.

Provides modular, type-consistent reward computation with proper tensor/number handling.
All modules return Python floats (not tensors) for consistency with RL agents.
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import numpy as np
from config import TOX_TIMES_DOSE_MAX, antibiotic_resistances


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


class GenomeReward(nn.Module):
    """
    Computes reward based on antibiotic efficacy against current population.
    
    Evaluates effectiveness of applied doses against observed population genome.
    Combines resistance and toxicity metrics to guide dose selection.
    """
    def __init__(
        self,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        aging_type: str = "sqrt",
    ):
        super(GenomeReward, self).__init__()
        self.device = device
        self.dtype = dtype
        self.age_normalizer = AgeNormalizer(aging_type)

    def forward(
        self,
        avg_genome: Optional[torch.Tensor],
        doses: torch.Tensor,
        age: Union[int, float],
    ) -> float:
        """
        Args:
            avg_genome: Average genome matrix [K, M] or None if no sequencing data
            doses: Applied dose vector [A] (tensor or numpy array)
            age: Age of sequencing measurement in timesteps
            
        Returns:
            Genome-based efficacy reward as Python float
        """
        # No sequencing data → neutral
        if avg_genome is None:
            return 0.0
        
        # Ensure tensors are on correct device/dtype
        if isinstance(avg_genome, np.ndarray):
            avg_genome = torch.from_numpy(avg_genome).to(self.device).to(self.dtype)
        else:
            avg_genome = avg_genome.to(self.device).to(self.dtype)
        
        if isinstance(doses, np.ndarray):
            doses = torch.from_numpy(doses).to(self.device).to(self.dtype)
        else:
            doses = doses.to(self.device).to(self.dtype)
        
        # Compute resistances and toxicities
        resistances, toxicities, _ = antibiotic_resistances(
            avg_genome, device=self.device, dtype=self.dtype
        )
        
        # Verify shapes
        if doses.shape[0] != toxicities.shape[0]:
            raise ValueError(
                f"dose_vector ({doses.shape[0]}) must match #antibiotics ({toxicities.shape[0]})"
            )
        
        # Compute reward: negative of (aggressiveness × susceptibility)
        # Lower values encourage using susceptible populations with appropriate toxicity
        aggressiveness = doses * toxicities  # [A]
        avg_resistance_by_ab = resistances.mean(dim=0)  # [K, A] -> [A]
        avg_susceptibility = 1.0 - avg_resistance_by_ab  # [A]
        
        # Reward is negative to penalize high toxicity × aggressiveness
        reward_vec = -1.0 * ((aggressiveness * avg_susceptibility) / TOX_TIMES_DOSE_MAX)  # [A]
        genome_term = torch.mean(reward_vec)  # scalar tensor
        
        # Apply age-based decay (older sequencing data is less valuable)
        genome_term = self.age_normalizer(genome_term, age)
        
        # Clip to valid range
        genome_term = torch.clamp(genome_term, min=-1.0, max=1.0)
        
        return float(genome_term.item())


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


class DoseRewardCompound(nn.Module):
    """
    Compound reward module combining population, genome, and cost terms.
    
    Orchestrates all reward components into a single score with weighted aggregation.
    """
    def __init__(
        self,
        target_population: float = 500.0,
        population_norm: float = 1000.0,
        dose_cost_per_unit: float = 0.2,
        w_pop: float = 1.0,
        w_genome: float = 0.5,
        w_cost: float = 0.05,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        aging_type: str = "sqrt",
    ):
        super(DoseRewardCompound, self).__init__()
        self.w_pop = float(w_pop)
        self.w_genome = float(w_genome)
        self.w_cost = float(w_cost)
        
        self.pop_reward = PopulationReward(target_population, population_norm, aging_type)
        self.genome_reward = GenomeReward(device, dtype, aging_type)
        self.cost_reward = CostReward(dose_cost_per_unit)

    def forward(
        self,
        last_count_obs: Optional[Union[int, float]],
        age_pop: Union[int, float],
        avg_genome: Optional[torch.Tensor],
        doses: Union[np.ndarray, torch.Tensor],
        age_genome: Union[int, float],
    ) -> float:
        """
        Compute combined dose reward.
        
        Args:
            last_count_obs: Observed population or None
            age_pop: Age of population measurement
            avg_genome: Average genome matrix or None
            doses: Dose vector
            age_genome: Age of genome measurement
            
        Returns:
            Combined reward as Python float
        """
        pop_term = self.pop_reward(last_count_obs, age_pop)
        genome_term = self.genome_reward(avg_genome, doses, age_genome)
        cost_term = self.cost_reward.dose_cost(doses)
        
        total_reward = (
            self.w_pop * pop_term +
            self.w_genome * genome_term +
            self.w_cost * cost_term
        )
        
        return float(total_reward)


class PopulationMaintenanceReward(nn.Module):
    """
    Per-step population maintenance penalty.
    
    Encourages population to stay near target by penalizing deviation.
    Asymmetric: overshooting target is penalized more than undershooting.
    """
    def __init__(
        self,
        target_population: float,
        population_norm: float,
        asymmetry_factor: float = 3.0,
        weight: float = 0.01,
    ):
        super(PopulationMaintenanceReward, self).__init__()
        self.target_population = float(target_population)
        self.population_norm = float(population_norm)
        self.asymmetry_factor = float(asymmetry_factor)
        self.weight = float(weight)

    def forward(self, true_population: Union[int, float]) -> float:
        """
        Args:
            true_population: Current actual population (from simulator)
            
        Returns:
            Population maintenance penalty as Python float (typically negative)
        """
        pop = float(true_population)
        
        # Asymmetric penalties
        above_target = max(0.0, pop - self.target_population)
        below_target = max(0.0, self.target_population - pop)
        
        # Overshooting is worse than undershooting
        penalty = -(
            self.asymmetry_factor * above_target +
            0.3 * below_target
        ) / max(1.0, self.population_norm) * self.weight
        
        return float(penalty)


