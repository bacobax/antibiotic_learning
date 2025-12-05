"""
Comprehensive configuration loading and validation for PPO training.

Handles loading all hyperparameters from YAML files with validation and defaults.
Provides a unified configuration interface for the entire training pipeline.

Configuration file locations:
  - Default configs: rl/configs/training_config.yaml
  - Fast testing: rl/configs/training_config_fast.yaml
  - Production: rl/configs/training_config_production.yaml

All configurations are loaded from a single YAML file including:
  - Environment parameters
  - Actions configuration
  - Model architecture
  - PPO hyperparameters
  - Training setup
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import copy
from inspect import signature

try:
    import yaml
    HAS_YAML = True

    class _TupleSafeLoader(yaml.SafeLoader):
        """Safe loader that understands python/tuple tags produced by PyYAML dumps."""

    def _construct_python_tuple(loader, node):
        return tuple(loader.construct_sequence(node))

    _TupleSafeLoader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        _construct_python_tuple,
    )
except ImportError:
    HAS_YAML = False
    yaml = None  # type: ignore
    _TupleSafeLoader = None  # type: ignore


@dataclass
class ActionConfig:
    """Configuration for discrete actions."""
    noop_cost: float
    count_cost: float
    sequencing_cost: float
    sequencing_duration: int
    dose_cost: float  # Fixed cost per dose action
    dose_cost_per_unit: float  # Variable cost per unit of antibiotic
    cost_weight: float


@dataclass
class SurvivalBonusConfig:
    """Configuration for survival bonus reward."""
    enabled: bool
    base_bonus: float
    scaling_type: str
    scaling_factor: float
    max_bonus: float


@dataclass
class PopulationMaintenanceConfig:
    """Configuration for simplified kernel-based maintenance reward."""
    enabled: bool = True
    target_population: float = 100.0
    kernel_type: str = "gaussian"
    kernel_peak_reward: float = 1.0
    kernel_max_penalty: float = 0.0
    kernel_zero_distance: float = 100.0


@dataclass
class InformedDosingConfig:
    """Configuration for informed dosing rewards and penalties.
    Defaults provided to allow ignoring this section when not used by the simplified reward system.
    """
    reward_window_steps: int = 0  # Max steps after COUNT where doses remain eligible
    reward_weight: float = 0.0  # Multiplier applied to population drop × dose magnitude
    max_reward_per_dose: float = 0.0  # Hard cap per individual dose reward
    time_decay: bool = False  # Whether to apply time-based decay
    decay_type: str = "linear"  # "linear" or "exponential"
    decay_rate: float = 0.0  # Rate parameter for the chosen decay function
    min_reward_fraction: float = 0.0  # Floor for decay factor
    penalty_dosing_under_target: float = 5.0  # Base penalty when dosing below target
    penalty_dosing_under_target_dose_scale: float = 0.0  # Weight for dose magnitude term
    penalty_dosing_under_target_dose_exponent: float = 1.0  # Exponent applied to dose magnitude
    penalty_dosing_under_target_deficit_scale: float = 0.0  # Weight for normalized population deficit
    penalty_dosing_under_target_deficit_cap: float = 1.0  # Clamp for normalized deficit contribution
    penalty_dosing_under_target_max: Optional[float] = None  # Optional ceiling for total penalty
    reward_dosing_above_with_seq: float = 0.0  # Reward multiplier when dosing above target with fresh seq
    reward_dosing_above_no_seq: float = 0.0  # Reward multiplier when dosing above target without seq
    penalty_blind_dose: float = 3.0  # Base penalty when dosing without a fresh count
    penalty_blind_dose_amount_scale: float = 0.0  # Weight for blind penalty dose magnitude scaling
    penalty_blind_dose_amount_exponent: float = 1.0  # Exponent applied to blind dose magnitude term
    penalty_blind_dose_max: Optional[float] = None  # Optional ceiling for blind penalty


@dataclass
class CountingRewardSimpleConfig:
    """Simplified counting reward configuration for the new reward system."""
    informative_count_reward: float = 1.0


@dataclass
class NoopRewardSimpleConfig:
    """Simplified NOOP reward configuration for the new reward system."""
    strategic_noop_reward: float = 0.5


@dataclass
class TimingWindowConfig:
    """Timing window configuration for specific actions (COUNT/SEQ)."""
    min_elapsed: int
    max_elapsed: int


@dataclass
class TimingConfig:
    """Environment-level timing and freshness thresholds."""
    t_count_freshness: int
    t_seq_freshness: int
    max_count_window: int
    critical_ratio: float
    count_window: TimingWindowConfig
    seq_window: TimingWindowConfig


@dataclass
class CriticalPenaltiesConfig:
    """Configuration for simplified critical penalties used in the reward wrapper."""
    penalty_critical_no_dose: float = 5.0
    penalty_critical_no_count: float = 0.3


@dataclass
class EarlyTerminationConfig:
    """Configuration for early termination on unrecoverable NOOP-only states."""
    enabled: bool = False  # Whether to enable early termination
    penalty: float = 0.0  # Maximum penalty applied when termination happens very early
    min_penalty: Optional[float] = None  # Penalty applied at/near max_steps (defaults to penalty)
    penalty_decay_power: float = 1.0  # Curve exponent controlling how fast the penalty decays with time
    population_threshold: float = 5.0  # Multiplier of target for high-population cutoff
    population_low_threshold: float = 0.2  # Multiplier of target for low-population cutoff
    require_budget_depleted: bool = True  # If True, only trigger when budget is also depleted
    extinction_penalty: float = 0.0  # Penalty applied when population collapses to zero
    noop_only_timeout_steps: Optional[int] = None  # Consecutive steps with only NOOP available before forcing termination


@dataclass
class SequencingRewardConfig:
    """Configuration for sequencing-related rewards."""
    seq_already_pending_penalty: float = 2.0
    informative_seq_reward: float = 1.0
    redundant_penalty: float = 0.001  # Legacy compatibility


@dataclass
class PredictionRewardConfig:
    """Configuration for prediction accuracy rewards."""
    enabled: bool = True
    weight: float = 1.0  # Legacy global weight (used as fallback)
    align_weight: Optional[float] = None  # Weight for aligning prediction to true population
    target_weight: Optional[float] = None  # Weight for aligning prediction to target population
    align_scale: float = 5.0  # Sharpness for true-population potential
    target_scale: float = 2.5  # Sharpness for target-population potential

    def resolved_align_weight(self) -> float:
        """Return the effective alignment weight, honoring legacy configs."""
        return float(self.align_weight if self.align_weight is not None else self.weight)

    def resolved_target_weight(self) -> float:
        """Return the effective target weight, honoring legacy configs."""
        return float(self.target_weight if self.target_weight is not None else self.weight)


@dataclass
class RewardConfig:
    """Unified reward configuration."""
    survival_bonus: SurvivalBonusConfig
    informed_dosing: InformedDosingConfig
    counting: CountingRewardSimpleConfig
    noop: NoopRewardSimpleConfig
    sequencing: SequencingRewardConfig
    prediction: PredictionRewardConfig
    early_termination: EarlyTerminationConfig
    critical_penalties: CriticalPenaltiesConfig
    population_maintenance: Optional[PopulationMaintenanceConfig] = None


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    max_steps: int
    k_doses: int
    device: str
    dtype: str
    rewards: RewardConfig
    timing: Optional[TimingConfig] = None
    initial_bacteria_per_type_range: Optional[Tuple[int, int]] = None
    warmup_skip_steps: int = 0
    enable_individual_tracking: bool = True
    max_individual_history: int = 1000
    max_tracked_individuals: Optional[int] = 2000
    max_history_steps: Optional[int] = 2000
    max_recent_dose_events: int = 256
    population_target: Optional[float] = None
    population_norm: Optional[float] = None
    budget_init: Optional[float] = None
    budget_norm: Optional[float] = None
    use_torch_fields: bool = False
    field_device: Optional[str] = None
    enable_food_diffusion: Optional[bool] = None


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_dim: int
    rnn_layers: int
    n_discrete: int
    dose_action_index: int
    k_doses: int
    sigmoid_scale_factor: float = 1.0


@dataclass
class PPOConfig:
    """PPO algorithm configuration."""
    gamma: float
    gae_lambda: float
    clip_eps: float
    vf_coef: float
    ent_coef: float
    max_grad_norm: float
    rollout_steps: int
    epochs: int
    seq_len: int
    batch_seq_len: int
    lr: float


@dataclass
class TrainingConfig:
    """Training execution configuration."""
    total_updates: int
    seed: int
    checkpoint_interval: int
    log_interval: int
    save_dir: str
    experiment_name: str
    save_checkpoints_per_run: bool = False  # Whether to save checkpoints in timestamped run directories
    log_window_size: Optional[int] = 2000
    log_memory: bool = False
    memory_log_interval: int = 25


@dataclass
class CompleteConfig:
    """Complete configuration combining all subsystems."""
    environment: EnvironmentConfig
    actions: ActionConfig
    model: ModelConfig
    ppo: PPOConfig
    training: TrainingConfig
    
    @property
    def device_type(self) -> str:
        """Get PyTorch-compatible device string."""
        if self.environment.device.lower() in ["mps", "metal"]:
            return "mps"
        return self.environment.device.lower()
    
    @property
    def torch_dtype(self):
        """Get PyTorch dtype from string."""
        import torch
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
        }
        dtype_str = self.environment.dtype.lower()
        if dtype_str not in dtype_map:
            raise ValueError(f"Unknown dtype: {dtype_str}. Choose from {list(dtype_map.keys())}")
        return dtype_map[dtype_str]


def _get_default_config() -> Dict[str, Any]:
    """Get default complete configuration."""
    return {
        "environment": {
            "max_steps": 1000,
            "k_doses": 3,
            "device": "cpu",
            "dtype": "float32",
            "initial_bacteria_per_type_range": None,
            "warmup_skip_steps": 0,
            "enable_individual_tracking": True,
            "max_individual_history": 1000,
            "max_tracked_individuals": 2000,
            "max_history_steps": 2000,
            "max_recent_dose_events": 256,
            "use_torch_fields": False,
            "field_device": None,
            "enable_food_diffusion": None,
            "timing": {
                "t_count_freshness": 5,
                "t_seq_freshness": 8,
                "max_count_window": 30,
                "critical_ratio": 3.0,
                "count_window": {
                    "min_elapsed": 5,
                    "max_elapsed": 30,
                },
                "seq_window": {
                    "min_elapsed": 8,
                    "max_elapsed": 50,
                },
            },
            "population": {
                "target_population": 500,
                "population_norm": 1000.0,
            },
            "budget": {
                "budget_init": 100.0,
                "budget_norm": 100.0,
            },
            "rewards": {
                "population_maintenance": {
                    "enabled": True,
                    "target_population": 500,
                    "kernel_type": "gaussian",
                    "kernel_peak_reward": 1.0,
                    "kernel_max_penalty": 0.0,
                    "kernel_zero_distance": 100.0,
                },
                "survival_bonus": {
                    "enabled": True,
                    "base_bonus": 0.01,
                    "scaling_type": "constant",
                    "scaling_factor": 0.1,
                    "max_bonus": 0.1,
                },
                "informed_dosing": {
                    "reward_window_steps": 5,
                    "reward_weight": 1.0,
                    "max_reward_per_dose": 5.0,
                    "time_decay": True,
                    "decay_type": "linear",
                    "decay_rate": 0.2,
                    "min_reward_fraction": 0.0,
                    "penalty_dosing_under_target": 5.0,
                    "penalty_dosing_under_target_dose_scale": 0.0,
                    "penalty_dosing_under_target_dose_exponent": 1.0,
                    "penalty_dosing_under_target_deficit_scale": 0.0,
                    "penalty_dosing_under_target_deficit_cap": 1.0,
                    "penalty_dosing_under_target_max": None,
                    "reward_dosing_above_with_seq": 0.0,
                    "reward_dosing_above_no_seq": 0.0,
                    "penalty_blind_dose": 3.0,
                    "penalty_blind_dose_amount_scale": 0.0,
                    "penalty_blind_dose_amount_exponent": 1.0,
                    "penalty_blind_dose_max": None,
                },
                "counting": {
                    "informative_count_reward": 1.0,
                },
                "noop": {
                    "strategic_noop_reward": 0.5,
                },
                "critical_penalties": {
                    "penalty_critical_no_dose": 5.0,
                    "penalty_critical_no_count": 2.0,
                },
                "sequencing": {
                    "seq_already_pending_penalty": 2.0,
                    "informative_seq_reward": 1.0,
                    "redundant_penalty": 0.001,
                },
                "prediction": {
                    "enabled": True,
                    "weight": 1.0,
                    "align_weight": None,
                    "target_weight": None,
                    "align_scale": 5.0,
                    "target_scale": 2.5,
                },
                "early_termination": {
                    "enabled": False,
                    "penalty": 0.0,
                    "min_penalty": None,
                    "penalty_decay_power": 1.0,
                    "population_threshold": 5.0,
                    "population_low_threshold": 0.2,
                    "extinction_penalty": 0.0,
                    "require_budget_depleted": True,
                    "noop_only_timeout_steps": None,
                },
            },
        },
        "actions": {
            "weight_cost": 1.0,
            "noop": {
                "cost": 0.0,
                "duration": 0,
            },
            "count_bacteria": {
                "cost": 0.0,
                "duration": 0,
            },
            "sequencing": {
                "cost": 1.0,
                "duration": 5,
            },
            "dose": {
                "cost_per_unit": 0.2,
                "duration": 0,
            },
        },
        "model": {
            "hidden_dim": 256,
            "rnn_layers": 1,
            "n_discrete": 4,
            "dose_action_index": 3,
            "k_doses": 3,
            "sigmoid_scale_factor": 1.0,
        },
        "ppo": {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "max_grad_norm": 1.0,
            "rollout_steps": 2048,
            "epochs": 4,
            "seq_len": 64,
            "batch_seq_len": 64,
            "lr": 3e-4,
        },
        "training": {
            "total_updates": 100,
            "seed": 42,
            "checkpoint_interval": 50,
            "log_interval": 10,
            "save_dir": "./checkpoints",
            "experiment_name": "ppo_training",
            "save_checkpoints_per_run": False,
            "log_window_size": 2000,
        },
    }


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration values.
    
    Args:
        config: Configuration dictionary
    
    Raises:
        ValueError: If configuration is invalid
    """
    env = config.get("environment", {})
    actions = config.get("actions", {})
    model = config.get("model", {})
    ppo = config.get("ppo", {})
    training = config.get("training", {})
    rewards = env.get("rewards", {})
    env_population_cfg = env.get("population", {})
    population_maintenance_cfg = rewards.get("population_maintenance", None)
    
    # Environment validation
    if env.get("k_doses", 1) <= 0:
        raise ValueError("k_doses must be > 0")
    if env.get("device", "cpu").lower() not in ["cpu", "cuda", "mps"]:
        raise ValueError(f"Invalid device: {env.get('device')}")
    if env.get("dtype", "float32").lower() not in ["float32", "float64", "float16"]:
        raise ValueError(f"Invalid dtype: {env.get('dtype')}")

    spawn_range = env.get("initial_bacteria_per_type_range", None)
    if spawn_range is not None:
        if isinstance(spawn_range, dict):
            spawn_min = spawn_range.get("min")
            spawn_max = spawn_range.get("max")
            spawn_values = (spawn_min, spawn_max)
        else:
            spawn_values = spawn_range
        if not isinstance(spawn_values, (list, tuple)) or len(spawn_values) != 2:
            raise ValueError("initial_bacteria_per_type_range must be a 2-element list/tuple or dict with 'min'/'max'")
        spawn_min, spawn_max = spawn_values
        if spawn_min is None or spawn_max is None:
            raise ValueError("initial_bacteria_per_type_range must contain both min and max values")
        try:
            spawn_min = int(spawn_min)
            spawn_max = int(spawn_max)
        except (TypeError, ValueError):
            raise ValueError("initial_bacteria_per_type_range values must be integers")
        if spawn_min <= 0 or spawn_max <= 0:
            raise ValueError("initial_bacteria_per_type_range values must be positive")
        if spawn_min > spawn_max:
            raise ValueError("initial_bacteria_per_type_range min must be <= max")

    if not env_population_cfg:
        raise ValueError("environment.population section is required")
    try:
        target_population = float(env_population_cfg.get("target_population", 0.0))
        population_norm = float(env_population_cfg.get("population_norm", 0.0))
    except (TypeError, ValueError):
        raise ValueError("environment.population target_population and population_norm must be numeric")
    if target_population <= 0:
        raise ValueError("environment.population.target_population must be > 0")
    if population_norm <= 0:
        raise ValueError("environment.population.population_norm must be > 0")

    env_budget_cfg = env.get("budget", {})
    if not env_budget_cfg:
        raise ValueError("environment.budget section is required")
    try:
        budget_init = float(env_budget_cfg.get("budget_init", 0.0))
        budget_norm = float(env_budget_cfg.get("budget_norm", 0.0))
    except (TypeError, ValueError):
        raise ValueError("environment.budget budget_init and budget_norm must be numeric")
    if budget_init <= 0:
        raise ValueError("environment.budget.budget_init must be > 0")
    if budget_norm <= 0:
        raise ValueError("environment.budget.budget_norm must be > 0")

    early_term_cfg = rewards.get("early_termination", {})
    base_penalty = early_term_cfg.get("penalty", 0.0)
    min_penalty = early_term_cfg.get("min_penalty", None)
    penalty_decay_power = early_term_cfg.get("penalty_decay_power", 1.0)
    high_thresh = early_term_cfg.get("population_threshold", 0.0)
    low_thresh = early_term_cfg.get("population_low_threshold", 0.0)
    extinction_penalty = early_term_cfg.get("extinction_penalty", 0.0)
    noop_only_steps = early_term_cfg.get("noop_only_timeout_steps", None)
    if base_penalty < 0.0:
        raise ValueError("early_termination.penalty must be >= 0")
    if min_penalty is None:
        min_penalty = base_penalty
    if min_penalty < 0.0:
        raise ValueError("early_termination.min_penalty must be >= 0")
    if min_penalty > base_penalty:
        raise ValueError("early_termination.min_penalty must be <= penalty")
    if penalty_decay_power <= 0.0:
        raise ValueError("early_termination.penalty_decay_power must be > 0")
    if low_thresh < 0.0:
        raise ValueError("early_termination.population_low_threshold must be >= 0")
    if high_thresh <= 0.0:
        raise ValueError("early_termination.population_threshold must be > 0")
    if low_thresh >= high_thresh:
        raise ValueError("early_termination.population_low_threshold must be < population_threshold")
    if extinction_penalty < 0.0:
        raise ValueError("early_termination.extinction_penalty must be >= 0")
    if noop_only_steps is not None:
        try:
            noop_only_steps_int = int(noop_only_steps)
        except (TypeError, ValueError):
            raise ValueError("early_termination.noop_only_timeout_steps must be an integer or null")
        if noop_only_steps_int < 0:
            raise ValueError("early_termination.noop_only_timeout_steps must be >= 0 or null to disable")

    informed_cfg = rewards.get("informed_dosing", {})
    if informed_cfg.get("reward_window_steps", 0) < 0:
        raise ValueError("informed_dosing.reward_window_steps must be >= 0")
    if informed_cfg.get("max_reward_per_dose", 0.0) < 0.0:
        raise ValueError("informed_dosing.max_reward_per_dose must be >= 0")
    if informed_cfg.get("decay_rate", 0.0) < 0.0:
        raise ValueError("informed_dosing.decay_rate must be >= 0")
    min_fraction = informed_cfg.get("min_reward_fraction", 0.0)
    if not (0.0 <= min_fraction <= 1.0):
        raise ValueError("informed_dosing.min_reward_fraction must be in [0, 1]")
    decay_type = str(informed_cfg.get("decay_type", "linear")).lower()
    if decay_type not in {"linear", "exponential"}:
        raise ValueError("informed_dosing.decay_type must be 'linear' or 'exponential'")
    under_base = informed_cfg.get("penalty_dosing_under_target", 0.0)
    if under_base < 0.0:
        raise ValueError("informed_dosing.penalty_dosing_under_target must be >= 0")
    if informed_cfg.get("penalty_dosing_under_target_dose_scale", 0.0) < 0.0:
        raise ValueError("informed_dosing.penalty_dosing_under_target_dose_scale must be >= 0")
    if informed_cfg.get("penalty_dosing_under_target_dose_exponent", 1.0) <= 0.0:
        raise ValueError("informed_dosing.penalty_dosing_under_target_dose_exponent must be > 0")
    if informed_cfg.get("penalty_dosing_under_target_deficit_scale", 0.0) < 0.0:
        raise ValueError("informed_dosing.penalty_dosing_under_target_deficit_scale must be >= 0")
    deficit_cap = informed_cfg.get("penalty_dosing_under_target_deficit_cap", 1.0)
    if deficit_cap < 0.0:
        raise ValueError("informed_dosing.penalty_dosing_under_target_deficit_cap must be >= 0")
    under_max = informed_cfg.get("penalty_dosing_under_target_max", None)
    if under_max is not None and under_max < under_base:
        raise ValueError("informed_dosing.penalty_dosing_under_target_max must be >= base penalty")
    blind_base = informed_cfg.get("penalty_blind_dose", 0.0)
    if blind_base < 0.0:
        raise ValueError("informed_dosing.penalty_blind_dose must be >= 0")
    if informed_cfg.get("penalty_blind_dose_amount_scale", 0.0) < 0.0:
        raise ValueError("informed_dosing.penalty_blind_dose_amount_scale must be >= 0")
    if informed_cfg.get("penalty_blind_dose_amount_exponent", 1.0) <= 0.0:
        raise ValueError("informed_dosing.penalty_blind_dose_amount_exponent must be > 0")
    blind_max = informed_cfg.get("penalty_blind_dose_max", None)
    if blind_max is not None and blind_max < blind_base:
        raise ValueError("informed_dosing.penalty_blind_dose_max must be >= base penalty")
    if population_maintenance_cfg is not None:
        if population_maintenance_cfg.get("target_population", 0) <= 0:
            raise ValueError("population_maintenance.target_population must be > 0")
        if population_maintenance_cfg.get("kernel_peak_reward", 0.0) < 0.0:
            raise ValueError("population_maintenance.kernel_peak_reward must be >= 0")
        if population_maintenance_cfg.get("kernel_max_penalty", 0.0) < 0.0:
            raise ValueError("population_maintenance.kernel_max_penalty must be >= 0")
        if population_maintenance_cfg.get("kernel_zero_distance", 0.0) <= 0.0:
            raise ValueError("population_maintenance.kernel_zero_distance must be > 0")
    
    # Actions validation
    seq = actions.get("sequencing", {})
    if seq.get("duration", 1) < 0:
        raise ValueError("sequencing duration must be >= 0")
    if seq.get("cost", 0) < 0:
        raise ValueError("sequencing cost must be >= 0")
    
    dose = actions.get("dose", {})
    if dose.get("cost_per_unit", 0) < 0:
        raise ValueError("dose cost_per_unit must be >= 0")
    
    # Model validation
    if model.get("hidden_dim", 1) <= 0:
        raise ValueError("hidden_dim must be > 0")
    if model.get("rnn_layers", 1) <= 0:
        raise ValueError("rnn_layers must be > 0")
    if model.get("n_discrete", 4) != 4:
        raise ValueError("n_discrete should be 4")
    
    # PPO validation
    if not (0 < ppo.get("gamma", 0.99) < 1):
        raise ValueError("gamma must be in (0, 1)")
    if not (0 < ppo.get("gae_lambda", 0.95) <= 1):
        raise ValueError("gae_lambda must be in (0, 1)")
    if ppo.get("rollout_steps", 1) <= 0:
        raise ValueError("rollout_steps must be > 0")
    if ppo.get("epochs", 1) <= 0:
        raise ValueError("epochs must be > 0")
    if ppo.get("seq_len", 1) <= 0:
        raise ValueError("seq_len must be > 0")
    if ppo.get("lr", 1e-5) <= 0:
        raise ValueError("lr must be > 0")
    
    # Training validation
    if training.get("total_updates", 1) <= 0:
        raise ValueError("total_updates must be > 0")
    if training.get("seed", 0) < 0:
        raise ValueError("seed must be >= 0")
    if training.get("checkpoint_interval", 1) <= 0:
        raise ValueError("checkpoint_interval must be > 0")


def _merge_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge provided config with defaults, filling missing values.
    
    Args:
        config: User-provided configuration
    
    Returns:
        Merged configuration with all defaults filled in
    """
    defaults = _get_default_config()
    
    # Deep merge
    for section in ["environment", "actions", "model", "ppo", "training"]:
        if section not in config:
            config[section] = {}
        for key, default_val in defaults[section].items():
            if key not in config[section]:
                config[section][key] = default_val

    # Ensure reward subsections are populated with defaults
    rewards_defaults = defaults["environment"].get("rewards", {})
    rewards_config = config["environment"].setdefault("rewards", {})
    for subsection, subsection_defaults in rewards_defaults.items():
        if isinstance(subsection_defaults, dict):
            user_section = rewards_config.setdefault(subsection, {})
            for key, default_val in subsection_defaults.items():
                if key not in user_section:
                    user_section[key] = default_val
        else:
            rewards_config.setdefault(subsection, subsection_defaults)

    # Ensure timing section is populated with defaults (including nested windows)
    timing_defaults = defaults["environment"].get("timing", {})
    timing_config = config["environment"].setdefault("timing", {})
    for key, default_val in timing_defaults.items():
        if isinstance(default_val, dict):
            user_section = timing_config.setdefault(key, {})
            for inner_key, inner_default in default_val.items():
                user_section.setdefault(inner_key, inner_default)
        else:
            timing_config.setdefault(key, default_val)
    
    return config


def load_config(config_path: Optional[Union[str, Path]] = None) -> CompleteConfig:
    """
    Load configuration from YAML file or use defaults.
    
    Looks for config in rl/configs/ directory if relative path provided.
    
    Args:
        config_path: Path to YAML configuration file, or None for defaults
                     Can be relative or absolute path
    
    Returns:
        CompleteConfig object with all validated parameters
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    if config_path is None:
        config_dict = _get_default_config()
        print("ℹ Using default configuration")
    else:
        if not HAS_YAML:
            raise RuntimeError(
                "PyYAML is required to load config files. "
                "Install it with: pip install pyyaml"
            )
        
        config_file = Path(config_path)
        
        # If path is relative and doesn't exist, try rl/configs directory
        if not config_file.exists() and not config_file.is_absolute():
            alt_path = Path(__file__).parent / "configs" / config_path
            if alt_path.exists():
                config_file = alt_path
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                loader_cls = _TupleSafeLoader if _TupleSafeLoader is not None else yaml.SafeLoader
                config_dict = yaml.load(f, Loader=loader_cls) or {}
        except Exception as e:
            raise ValueError(f"Failed to load YAML config {config_file}: {e}")
        
        print(f"✓ Loaded configuration from: {config_file}")
    
    # Preserve raw user-provided configuration for detecting explicitly-set fields
    raw_config_dict = copy.deepcopy(config_dict)

    # Merge with defaults to fill missing values
    config_dict = _merge_with_defaults(config_dict)

    # Strict validation for the simple rewards config: require all hparams defined
    def _strict_require(keys: list[str], root: Dict[str, Any], context: str) -> None:
        for k in keys:
            if k not in root:
                raise ValueError(f"Missing required hparam '{k}' in {context}")

    if config_path is not None:
        try:
            cfg_name = Path(config_path).name
        except Exception:
            cfg_name = str(config_path)
        if cfg_name == "training_config_simple_rewards.yaml":
            env = config_dict.get("environment", {})
            rewards = env.get("rewards", {})
            actions = config_dict.get("actions", {})
            model = config_dict.get("model", {})
            ppo = config_dict.get("ppo", {})
            training = config_dict.get("training", {})

            # Timing (always under environment.timing in simple rewards config)
            timing = env.get("timing", {})
            _strict_require(["t_count_freshness", "t_seq_freshness", "max_count_window", "critical_ratio"], timing, "environment.timing")
            count_window = timing.get("count_window", {})
            _strict_require(["min_elapsed", "max_elapsed"], count_window, "environment.timing.count_window")
            seq_window = timing.get("seq_window", {})
            _strict_require(["min_elapsed", "max_elapsed"], seq_window, "environment.timing.seq_window")

            # Rewards subsections used by env
            informed = rewards.get("informed_dosing", {})
            _strict_require([
                "penalty_dosing_under_target",
                "penalty_dosing_under_target_dose_scale",
                "penalty_dosing_under_target_dose_exponent",
                "penalty_dosing_under_target_deficit_scale",
                "penalty_dosing_under_target_deficit_cap",
                "penalty_dosing_under_target_max",
                "reward_dosing_above_with_seq",
                "reward_dosing_above_no_seq",
                "penalty_blind_dose",
                "penalty_blind_dose_amount_scale",
                "penalty_blind_dose_amount_exponent",
                "penalty_blind_dose_max",
            ], informed, "environment.rewards.informed_dosing")

            sequencing = rewards.get("sequencing", {})
            _strict_require(["seq_already_pending_penalty", "informative_seq_reward"], sequencing, "environment.rewards.sequencing")

            counting = rewards.get("counting", {})
            _strict_require(["informative_count_reward"], counting, "environment.rewards.counting")

            noop = rewards.get("noop", {})
            _strict_require(["strategic_noop_reward"], noop, "environment.rewards.noop")

            critical = rewards.get("critical_penalties", {})
            _strict_require(["penalty_critical_no_dose", "penalty_critical_no_count"], critical, "environment.rewards.critical_penalties")

            # Kernel-based population maintenance
            pop_maint = rewards.get("population_maintenance", {})
            _strict_require([
                "enabled",
                "target_population",
                "kernel_type",
                "kernel_peak_reward",
                "kernel_max_penalty",
                "kernel_zero_distance",
            ], pop_maint, "environment.rewards.population_maintenance")

            # Survival bonus
            surv = rewards.get("survival_bonus", {})
            _strict_require(["enabled", "base_bonus", "scaling_type", "scaling_factor", "max_bonus"], surv, "environment.rewards.survival_bonus")

            # Prediction
            pred = rewards.get("prediction", {})
            _strict_require([
                "enabled",
                "weight",
                "align_weight",
                "target_weight",
                "align_scale",
                "target_scale",
            ], pred, "environment.rewards.prediction")

            # Early termination
            early = rewards.get("early_termination", {})
            _strict_require([
                "enabled",
                "penalty",
                "min_penalty",
                "penalty_decay_power",
                "population_low_threshold",
                "population_threshold",
                "extinction_penalty",
                "require_budget_depleted",
            ], early, "environment.rewards.early_termination")

            # Budget
            budget = env.get("budget", {})
            _strict_require(["budget_init", "budget_norm"], budget, "environment.budget")

            # Population
            population = env.get("population", {})
            _strict_require(["target_population", "population_norm"], population, "environment.population")

            # Actions
            for act_name in ("noop", "count_bacteria", "sequencing", "dose"):
                act = actions.get(act_name, {})
                _strict_require(["id", "cost", "duration"], act, f"actions.{act_name}")
                if act_name == "dose":
                    if "cost_per_unit" not in act:
                        raise ValueError("Missing required hparam 'cost_per_unit' in actions.dose")
            if "weight_cost" not in actions:
                raise ValueError("Missing required hparam 'weight_cost' in actions")

            # Model
            _strict_require(["hidden_dim", "rnn_layers", "n_discrete", "dose_action_index", "k_doses", "sigmoid_scale_factor"], model, "model")

            # PPO
            _strict_require([
                "gamma","gae_lambda","clip_eps","vf_coef","ent_coef","max_grad_norm",
                "rollout_steps","epochs","seq_len","batch_seq_len","lr"
            ], ppo, "ppo")

            # Training
            _strict_require(["total_updates","seed","checkpoint_interval","log_interval","save_dir","experiment_name","save_checkpoints_per_run"], training, "training")
    
    # Validate configuration
    try:
        _validate_config(config_dict)
    except ValueError as e:
        raise ValueError(f"Configuration validation failed: {e}")
    
    # Extract action configuration
    actions_dict = config_dict.get("actions", {})
    seq = actions_dict.get("sequencing", {})
    dose = actions_dict.get("dose", {})
    noop = actions_dict.get("noop", {})
    count = actions_dict.get("count_bacteria", {})
    
    # Extract environment configuration with nested reward configs
    env_dict = config_dict["environment"]
    rewards_dict = env_dict["rewards"]
    raw_env_dict = raw_config_dict.get("environment", {}) if isinstance(raw_config_dict, dict) else {}
    raw_rewards_dict = raw_env_dict.get("rewards", {}) if isinstance(raw_env_dict, dict) else {}
    
    # Helper: filter dict keys to match dataclass constructor
    def _filter_keys_for(cls, dct: Dict[str, Any]) -> Dict[str, Any]:
        try:
            params = signature(cls).parameters
            allowed = set(params.keys())
            return {k: v for k, v in dct.items() if k in allowed}
        except Exception:
            return dct

    # Create nested reward config dataclasses (filter unknown keys for backward-compat)
    population_maintenance_cfg = None
    if "population_maintenance" in rewards_dict and rewards_dict["population_maintenance"] is not None:
        population_maintenance_section = dict(rewards_dict["population_maintenance"])
        raw_pop_maintenance_section = (
            raw_rewards_dict.get("population_maintenance")
            if isinstance(raw_rewards_dict, dict)
            else None
        )
        user_defined_pop_maint_target = (
            isinstance(raw_pop_maintenance_section, dict)
            and raw_pop_maintenance_section.get("target_population") is not None
        )
        if not user_defined_pop_maint_target:
            env_population_section = env_dict.get("population") or {}
            population_maintenance_section["target_population"] = env_population_section.get("target_population")
        population_maintenance_cfg = PopulationMaintenanceConfig(
            **_filter_keys_for(PopulationMaintenanceConfig, population_maintenance_section)
        )
    survival_bonus_cfg = SurvivalBonusConfig(**_filter_keys_for(SurvivalBonusConfig, rewards_dict["survival_bonus"]))
    informed_dosing_cfg = InformedDosingConfig(**_filter_keys_for(InformedDosingConfig, rewards_dict["informed_dosing"]))
    counting_cfg = CountingRewardSimpleConfig(
        **_filter_keys_for(CountingRewardSimpleConfig, rewards_dict.get("counting", {}))
    )
    noop_cfg = NoopRewardSimpleConfig(
        **_filter_keys_for(NoopRewardSimpleConfig, rewards_dict.get("noop", {}))
    )
    critical_penalties_section = rewards_dict.get("critical_penalties")
    if critical_penalties_section is not None:
        critical_penalties_cfg = CriticalPenaltiesConfig(
            **_filter_keys_for(CriticalPenaltiesConfig, critical_penalties_section)
        )
    else:
        critical_penalties_cfg = CriticalPenaltiesConfig()
    sequencing_cfg = SequencingRewardConfig(**_filter_keys_for(SequencingRewardConfig, rewards_dict["sequencing"]))
    prediction_cfg = PredictionRewardConfig(**_filter_keys_for(PredictionRewardConfig, rewards_dict["prediction"]))
    early_termination_cfg = EarlyTerminationConfig(**_filter_keys_for(EarlyTerminationConfig, rewards_dict["early_termination"]))
    
    reward_cfg = RewardConfig(
        population_maintenance=population_maintenance_cfg,
        survival_bonus=survival_bonus_cfg,
        informed_dosing=informed_dosing_cfg,
        counting=counting_cfg,
        noop=noop_cfg,
        critical_penalties=critical_penalties_cfg,
        sequencing=sequencing_cfg,
        prediction=prediction_cfg,
        early_termination=early_termination_cfg,
    )

    spawn_range_cfg = env_dict.get("initial_bacteria_per_type_range")
    parsed_spawn_range: Optional[Tuple[int, int]] = None
    if spawn_range_cfg is not None:
        if isinstance(spawn_range_cfg, dict):
            spawn_min = spawn_range_cfg.get("min")
            spawn_max = spawn_range_cfg.get("max")
        else:
            spawn_min, spawn_max = spawn_range_cfg
        parsed_spawn_range = (int(spawn_min), int(spawn_max))

    tracking_defaults = {
        "enabled": env_dict.get("enable_individual_tracking", True),
        "max_individual_history": env_dict.get("max_individual_history", 1000),
        "max_tracked_individuals": env_dict.get("max_tracked_individuals", 2000),
    }
    tracking_cfg = env_dict.get("tracking") or {}
    enable_tracking = bool(tracking_cfg.get("enabled", tracking_defaults["enabled"]))
    max_individual_history = int(tracking_cfg.get("max_individual_history", tracking_defaults["max_individual_history"]))
    max_tracked_individuals_val = tracking_cfg.get("max_tracked_individuals", tracking_defaults["max_tracked_individuals"])
    max_tracked_individuals = None if max_tracked_individuals_val is None else int(max_tracked_individuals_val)

    history_defaults = env_dict.get("max_history_steps", 2000)
    history_cfg = env_dict.get("history") or env_dict.get("history_limits") or {}
    max_history_steps_val = history_cfg.get("max_steps", history_defaults)
    max_history_steps = None if max_history_steps_val is None else int(max_history_steps_val)

    reward_buffer_cfg = env_dict.get("reward_buffers") or {}
    max_recent_dose_events = int(
        reward_buffer_cfg.get(
            "max_recent_dose_events",
            env_dict.get("max_recent_dose_events", 256),
        )
    )

    env_population_section = env_dict.get("population") or {}
    population_target_value = float(env_population_section.get("target_population"))
    population_norm_value = float(env_population_section.get("population_norm"))

    env_budget_section = env_dict.get("budget") or {}
    budget_init_value = float(env_budget_section.get("budget_init"))
    budget_norm_value = float(env_budget_section.get("budget_norm"))

    timing_section = env_dict.get("timing")
    timing_cfg: Optional[TimingConfig] = None
    if timing_section is not None:
        count_window_section = timing_section.get("count_window", {})
        seq_window_section = timing_section.get("seq_window", {})
        count_window_cfg = TimingWindowConfig(
            min_elapsed=int(count_window_section.get("min_elapsed", 5)),
            max_elapsed=int(count_window_section.get("max_elapsed", 30)),
        )
        seq_window_cfg = TimingWindowConfig(
            min_elapsed=int(seq_window_section.get("min_elapsed", 8)),
            max_elapsed=int(seq_window_section.get("max_elapsed", 50)),
        )
        timing_cfg = TimingConfig(
            t_count_freshness=int(timing_section.get("t_count_freshness", 5)),
            t_seq_freshness=int(timing_section.get("t_seq_freshness", 8)),
            max_count_window=int(timing_section.get("max_count_window", 30)),
            critical_ratio=float(timing_section.get("critical_ratio", 3.0)),
            count_window=count_window_cfg,
            seq_window=seq_window_cfg,
        )

    # Optional field backend acceleration controls
    use_torch_fields = bool(env_dict.get("use_torch_fields", False))
    field_device = env_dict.get("field_device")
    if field_device is None and use_torch_fields:
        field_device = env_dict.get("device", "cpu")
    enable_food_diffusion = env_dict.get("enable_food_diffusion")

    # Create environment config with nested structures
    env_cfg = EnvironmentConfig(
        max_steps=env_dict["max_steps"],
        k_doses=env_dict["k_doses"],
        device=env_dict["device"],
        dtype=env_dict["dtype"],
        rewards=reward_cfg,
        timing=timing_cfg,
        initial_bacteria_per_type_range=parsed_spawn_range,
        warmup_skip_steps=int(env_dict.get("warmup_skip_steps", 0) or 0),
        enable_individual_tracking=enable_tracking,
        max_individual_history=max_individual_history,
        max_tracked_individuals=max_tracked_individuals,
        max_history_steps=max_history_steps,
        max_recent_dose_events=max_recent_dose_events,
        population_target=population_target_value,
        population_norm=population_norm_value,
        budget_init=budget_init_value,
        budget_norm=budget_norm_value,
        use_torch_fields=use_torch_fields,
        field_device=field_device,
        enable_food_diffusion=enable_food_diffusion,
    )
    
    actions_cfg = ActionConfig(
        noop_cost=noop.get("cost", 0.0),
        count_cost=count.get("cost", 0.0),
        sequencing_cost=seq.get("cost", 1.0),
        sequencing_duration=seq.get("duration", 5),
        dose_cost=dose.get("cost", 2.0),
        dose_cost_per_unit=dose.get("cost_per_unit", 0.2),
        cost_weight=actions_dict.get("weight_cost", 0.0),
    )
    
    model_cfg = ModelConfig(**config_dict["model"])
    ppo_cfg = PPOConfig(**config_dict["ppo"])
    train_cfg = TrainingConfig(**config_dict["training"])
    
    return CompleteConfig(
        environment=env_cfg,
        actions=actions_cfg,
        model=model_cfg,
        ppo=ppo_cfg,
        training=train_cfg,
    )


def save_config(config: Union[CompleteConfig, Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration to save (CompleteConfig or dict)
        output_path: Path to save to
    """
    if not HAS_YAML:
        raise RuntimeError("PyYAML is required to save configs. Install with: pip install pyyaml")
    
    if isinstance(config, CompleteConfig):
        population_target = config.environment.population_target
        population_norm = config.environment.population_norm
        budget_init = config.environment.budget_init
        budget_norm = config.environment.budget_norm

        # Convert dataclasses to dicts
        config_dict = {
            "environment": {
                "max_steps": config.environment.max_steps,
                "k_doses": config.environment.k_doses,
                "device": config.environment.device,
                "dtype": config.environment.dtype,
                "initial_bacteria_per_type_range": config.environment.initial_bacteria_per_type_range,
                "warmup_skip_steps": config.environment.warmup_skip_steps,
                "enable_individual_tracking": config.environment.enable_individual_tracking,
                "max_individual_history": config.environment.max_individual_history,
                "max_tracked_individuals": config.environment.max_tracked_individuals,
                "max_history_steps": config.environment.max_history_steps,
                "max_recent_dose_events": config.environment.max_recent_dose_events,
                "timing": (
                    {
                        "t_count_freshness": config.environment.timing.t_count_freshness,
                        "t_seq_freshness": config.environment.timing.t_seq_freshness,
                        "max_count_window": config.environment.timing.max_count_window,
                        "critical_ratio": config.environment.timing.critical_ratio,
                        "count_window": {
                            "min_elapsed": config.environment.timing.count_window.min_elapsed,
                            "max_elapsed": config.environment.timing.count_window.max_elapsed,
                        },
                        "seq_window": {
                            "min_elapsed": config.environment.timing.seq_window.min_elapsed,
                            "max_elapsed": config.environment.timing.seq_window.max_elapsed,
                        },
                    }
                    if config.environment.timing is not None
                    else None
                ),
                "population": {
                    "target_population": population_target,
                    "population_norm": population_norm,
                },
                "budget": {
                    "budget_init": budget_init,
                    "budget_norm": budget_norm,
                },
                "rewards": {
                    "population_maintenance": (
                        {
                            k: v
                            for k, v in config.environment.rewards.population_maintenance.__dict__.items()
                        }
                        if config.environment.rewards.population_maintenance is not None
                        else None
                    ),
                    "survival_bonus": {
                        k: v for k, v in config.environment.rewards.survival_bonus.__dict__.items()
                    },
                    "informed_dosing": {
                        k: v for k, v in config.environment.rewards.informed_dosing.__dict__.items()
                    },
                    "counting": {
                        k: v for k, v in config.environment.rewards.counting.__dict__.items()
                    },
                    "noop": {
                        k: v for k, v in config.environment.rewards.noop.__dict__.items()
                    },
                    "critical_penalties": (
                        {
                            k: v
                            for k, v in config.environment.rewards.critical_penalties.__dict__.items()
                        }
                        if config.environment.rewards.critical_penalties is not None
                        else None
                    ),
                    "sequencing": {
                        k: v for k, v in config.environment.rewards.sequencing.__dict__.items()
                    },
                    "prediction": {
                        k: v for k, v in config.environment.rewards.prediction.__dict__.items()
                    },
                    "early_termination": {
                        k: v for k, v in config.environment.rewards.early_termination.__dict__.items()
                    },
                },
            },
            "actions": {
                "weight_cost": config.actions.cost_weight,
                "noop": {
                    "cost": config.actions.noop_cost,
                },
                "count_bacteria": {
                    "cost": config.actions.count_cost,
                },
                "sequencing": {
                    "cost": config.actions.sequencing_cost,
                    "duration": config.actions.sequencing_duration,
                },
                "dose": {
                    "cost": config.actions.dose_cost,
                    "cost_per_unit": config.actions.dose_cost_per_unit,
                },
            },
            "model": {
                k: v for k, v in config.model.__dict__.items()
            },
            "ppo": {
                k: v for k, v in config.ppo.__dict__.items()
            },
            "training": {
                k: v for k, v in config.training.__dict__.items()
            },
        }
    else:
        config_dict = config
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Configuration saved to: {output_file}")


if __name__ == "__main__":
    # Example: Create and save default config
    config = load_config()
    save_config(config, "training_config_example.yaml")
