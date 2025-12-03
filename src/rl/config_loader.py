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
class TrackingConfig:
    """Memory safety controls for tracking-heavy runs."""
    enabled: bool
    max_tracked_individuals: int
    max_individual_history: int


@dataclass
class RewardBuffersConfig:
    """Reward buffer configuration."""
    max_recent_dose_events: int


@dataclass
class CountWindowConfig:
    """Timing window for counting."""
    min_elapsed: int
    max_elapsed: int


@dataclass
class SeqWindowConfig:
    """Timing window for sequencing."""
    min_elapsed: int
    max_elapsed: int


@dataclass
class TimingConfig:
    """Timing and freshness thresholds."""
    t_count_freshness: int
    t_seq_freshness: int
    max_count_window: int
    critical_ratio: float
    count_window: CountWindowConfig
    seq_window: SeqWindowConfig


@dataclass
class InformedDosingConfig:
    """Pre-step rewards for informed dosing."""
    penalty_dosing_under_target: float
    penalty_dosing_under_target_dose_scale: float
    penalty_dosing_under_target_dose_exponent: float
    penalty_dosing_under_target_deficit_scale: float
    penalty_dosing_under_target_deficit_cap: float
    penalty_dosing_under_target_max: Optional[float]
    reward_dosing_above_with_seq: float
    reward_dosing_above_no_seq: float
    penalty_blind_dose: float
    penalty_blind_dose_amount_scale: float
    penalty_blind_dose_amount_exponent: float
    penalty_blind_dose_max: Optional[float]


@dataclass
class SequencingRewardConfig:
    """Pre-step rewards for sequencing."""
    seq_already_pending_penalty: float
    informative_seq_reward: float


@dataclass
class CountingRewardConfig:
    """Pre-step rewards for counting."""
    informative_count_reward: float
    cost_penalty: float


@dataclass
class NoopRewardConfig:
    """Pre-step rewards for strategic NOOP."""
    strategic_noop_reward: float


@dataclass
class CriticalPenaltiesConfig:
    """Post-step penalties."""
    penalty_critical_no_dose: float
    penalty_critical_no_count: float


@dataclass
class PopulationMaintenanceConfig:
    """Population maintenance (kernel-based)."""
    enabled: bool
    kernel_type: str
    kernel_peak_reward: float
    kernel_max_penalty: float
    kernel_zero_distance: float
    target_population: int = 0


@dataclass
class SurvivalBonusConfig:
    """Survival bonus (per-step reward for staying alive)."""
    enabled: bool
    base_bonus: float
    scaling_type: str
    scaling_factor: float
    max_bonus: float


@dataclass
class PredictionRewardConfig:
    """Prediction accuracy reward (COUNT-only)."""
    enabled: bool
    weight: float
    align_weight: float
    target_weight: float
    align_scale: float
    target_scale: float
    
    def resolved_align_weight(self) -> float:
        """Return align_weight, or weight if align_weight is None."""
        return self.align_weight if self.align_weight is not None else self.weight
    
    def resolved_target_weight(self) -> float:
        """Return target_weight, or weight if target_weight is None."""
        return self.target_weight if self.target_weight is not None else self.weight


@dataclass
class EarlyTerminationConfig:
    """Early termination on unrecoverable states."""
    enabled: bool
    penalty: float
    min_penalty: float
    penalty_decay_power: float
    population_low_threshold: float
    population_threshold: float
    extinction_penalty: float
    require_budget_depleted: bool


@dataclass
class BudgetConfig:
    """Budget configuration."""
    budget_init: float
    budget_norm: float


@dataclass
class PopulationConfig:
    """Population configuration."""
    target_population: int = 0
    population_norm: float = 1.0


@dataclass
class RewardConfig:
    """Unified reward configuration."""
    informed_dosing: Optional[InformedDosingConfig] = None
    sequencing: Optional[SequencingRewardConfig] = None
    counting: Optional[CountingRewardConfig] = None
    noop: Optional[NoopRewardConfig] = None
    critical_penalties: Optional[CriticalPenaltiesConfig] = None
    population_maintenance: Optional[PopulationMaintenanceConfig] = None
    survival_bonus: Optional[SurvivalBonusConfig] = None
    prediction: Optional[PredictionRewardConfig] = None
    early_termination: Optional[EarlyTerminationConfig] = None
    budget: Optional[BudgetConfig] = None
    population: Optional[PopulationConfig] = None
    tracking: Optional[TrackingConfig] = None
    history: Optional['HistoryConfig'] = None
    reward_buffers: Optional[RewardBuffersConfig] = None


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    max_steps: int
    k_doses: int
    device: str
    dtype: str
    rewards: RewardConfig
    timing: TimingConfig
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
    


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_dim: int
    rnn_layers: int
    n_discrete: int
    dose_action_index: int
    k_doses: int
    sigmoid_scale_factor: float


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
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
    """Training configuration."""
    total_updates: int
    seed: int
    checkpoint_interval: int
    log_interval: int
    save_dir: str
    experiment_name: str
    save_checkpoints_per_run: bool
    log_window_size: int
    log_memory: bool
    memory_log_interval: int

@dataclass
class ActionConfig:
    cost_weight: float
    noop_cost: float
    count_cost: float
    sequencing_cost: float
    sequencing_duration: int
    dose_cost: float
    dose_cost_per_unit: float


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
            "max_steps": None,
            "k_doses": None,
            "device": "cpu",
            "dtype": "float32",
            "initial_bacteria_per_type_range": None,
            "warmup_skip_steps": 0,
            "enable_individual_tracking": True,
            "max_individual_history": 1000,
            "max_tracked_individuals": 2000,
            "max_history_steps": 2000,
            "max_recent_dose_events": 256,
            "population": {
                "target_population": 500,
                "population_norm": 1000.0,
            },
            "budget": {
                "budget_init": 100.0,
                "budget_norm": 100.0,
            },
            "rewards": {
                "population": {
                    "population_norm_reward": 500.0,
                    "w_population_maintenance": 0.01,
                    "count_population_reward": 0.0,
                    "count_population_reward_alpha": 1.0,
                    "count_population_reward_beta": 0.5,
                    "noop_band_factor": 0.02,
                    "noop_reward_magnitude": 0.01,
                },
                "population_maintenance": {
                    "enabled": True,
                    "target_population": 500,
                    "kernel_type": "gaussian",
                    "kernel_peak_reward": 1.0,
                    "kernel_max_penalty": 0.0,
                    "kernel_zero_distance": 100.0,
                },
                "dose": {
                    "w_pop": 1.0,
                    "w_genome": 0.5,
                    "w_cost": 0.05,
                    "missing_feedback_penalty": 0.5,
                },
                "budget": {
                    "budget_init": 100.0,
                    "budget_norm": 100.0,
                    "budget_penalty": 10.0,
                    "unaffordable_action_penalty": 0.0,
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
                "regular_monitoring": {
                    "count_reward": 0.0,
                    "count_interval": 15,
                    "count_min_interval": 3,
                    "safe_nondosing_reward": 0.0,
                },
                "critical_inaction": {
                    "high_population_threshold": 3.0,
                    "no_action_penalty": 0.0,
                    "no_dose_penalty": 0.0,
                    "freshness_window": 5,
                    "noop_penalty": 0.0,
                    "noop_threshold": 15,
                },
                "sequencing": {
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
        config_path: Path to YAML configuration file
    
    Returns:
        CompleteConfig object with all validated parameters
    
    Raises:
        ValueError: If configuration is invalid
    """
    env = config.get("environment", {})
    actions = config.get("actions", {})
    model = config.get("model", {})
    ppo = config.get("ppo", {})
    training = config.get("training", {})
    rewards = env.get("rewards", {})
    population_cfg = rewards.get("population", {})
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

    early_term_cfg = rewards.get("early_termination", {})
    base_penalty = early_term_cfg.get("penalty", 0.0)
    min_penalty = early_term_cfg.get("min_penalty", None)
    penalty_decay_power = early_term_cfg.get("penalty_decay_power", 1.0)
    high_thresh = early_term_cfg.get("population_threshold", 0.0)
    low_thresh = early_term_cfg.get("population_low_threshold", 0.0)
    extinction_penalty = early_term_cfg.get("extinction_penalty", 0.0)
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
    alpha = population_cfg.get("count_population_reward_alpha", 1.0)
    beta = population_cfg.get("count_population_reward_beta", 0.5)
    norm_reward = population_cfg.get(
        "population_norm_reward",
        population_cfg.get("target_population", env_population_cfg.get("target_population", 1.0))
    )
    if norm_reward <= 0.0:
        raise ValueError("population.population_norm_reward must be > 0")
    if alpha <= 0.0:
        raise ValueError("population.count_population_reward_alpha must be > 0")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("population.count_population_reward_beta must be in [0, 1]")
    if population_maintenance_cfg is not None:
        # Validate kernel params only; target population comes from environment.population
        if population_maintenance_cfg.get("kernel_peak_reward", 0.0) < 0.0:
            raise ValueError("population_maintenance.kernel_peak_reward must be >= 0")
        if population_maintenance_cfg.get("kernel_max_penalty", 0.0) < 0.0:
            raise ValueError("population_maintenance.kernel_max_penalty must be >= 0")
        if population_maintenance_cfg.get("kernel_zero_distance", 0.0) <= 0.0:
            raise ValueError("population_maintenance.kernel_zero_distance must be > 0")
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
    
    # Also merge environment.population and environment.budget if they exist as separate sections
    # These get merged into rewards.population and rewards.budget for config construction
    if "population" in config["environment"] and isinstance(config["environment"]["population"], dict):
        pop_defaults = rewards_defaults.get("population", {})
        for key, default_val in pop_defaults.items():
            if key not in rewards_config["population"] and key not in config["environment"]["population"]:
                rewards_config["population"][key] = default_val
    
    if "budget" in config["environment"] and isinstance(config["environment"]["budget"], dict):
        budget_defaults = rewards_defaults.get("budget", {})
        for key, default_val in budget_defaults.items():
            if key not in rewards_config["budget"] and key not in config["environment"]["budget"]:
                rewards_config["budget"][key] = default_val
    
    return config


def load_config(config_path: Optional[Union[str, Path]] = None) -> CompleteConfig:
    """
    Validate that configuration has all required sections and fields.
    Throws errors instead of silently using defaults.
    
    Args:
        config: Configuration dictionary to validate
    
    Raises:
        ValueError: If required fields are missing
    """
    def _require_field(keys: list[str], root: Dict[str, Any], context: str) -> None:
        """Check that all keys exist in root dict."""
        for k in keys:
            if k not in root:
                raise ValueError(f"Missing required field '{k}' in {context}")
    
    # Top-level sections
    _require_field(["environment", "actions", "model", "ppo", "training"], config, "config")
    
    env = config["environment"]
    actions = config["actions"]
    model = config["model"]
    ppo = config["ppo"]
    training = config["training"]
    
    # Environment section
    _require_field(["max_steps", "k_doses", "device", "dtype"], env, "environment")
    
    # Environment.timing (if present, validate structure)
    if "timing" in env:
        timing = env["timing"]
        _require_field(["t_count_freshness", "t_seq_freshness", "max_count_window", "critical_ratio"], 
                      timing, "environment.timing")
        if "count_window" in timing:
            _require_field(["min_elapsed", "max_elapsed"], timing["count_window"], "environment.timing.count_window")
        if "seq_window" in timing:
            _require_field(["min_elapsed", "max_elapsed"], timing["seq_window"], "environment.timing.seq_window")
    
    # Environment.budget
    if "budget" in env:
        _require_field(["budget_init", "budget_norm"], env["budget"], "environment.budget")
    
    # Environment.population
    if "population" in env:
        _require_field(["target_population", "population_norm"], env["population"], "environment.population")
    
    # Environment.rewards section
    if "rewards" not in env:
        raise ValueError("Missing required section 'environment.rewards'")
    
    rewards = env["rewards"]
    
    # Required reward subsections
    _require_field(["informed_dosing", "sequencing", "counting", "noop", "critical_penalties",
                   "population_maintenance", "survival_bonus", "prediction", "early_termination"], 
                  rewards, "environment.rewards")
    
    # Informed dosing
    informed = rewards["informed_dosing"]
    _require_field([
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
    
    # Sequencing
    sequencing = rewards["sequencing"]
    _require_field(["seq_already_pending_penalty", "informative_seq_reward"], 
                  sequencing, "environment.rewards.sequencing")
    
    # Counting
    counting = rewards["counting"]
    _require_field(["informative_count_reward", "cost_penalty"], 
                  counting, "environment.rewards.counting")
    
    # NOOP
    noop_rewards = rewards["noop"]
    _require_field(["strategic_noop_reward"], noop_rewards, "environment.rewards.noop")
    
    # Critical penalties
    critical = rewards["critical_penalties"]
    _require_field(["penalty_critical_no_dose", "penalty_critical_no_count"], 
                  critical, "environment.rewards.critical_penalties")
    
    # Population maintenance (no target_population here; target is in environment.population)
    pop_maint = rewards["population_maintenance"]
    _require_field([
        "enabled",
        "kernel_type",
        "kernel_peak_reward",
        "kernel_max_penalty",
        "kernel_zero_distance",
    ], pop_maint, "environment.rewards.population_maintenance")
    
    # Survival bonus
    surv = rewards["survival_bonus"]
    _require_field(["enabled", "base_bonus", "scaling_type", "scaling_factor", "max_bonus"], 
                  surv, "environment.rewards.survival_bonus")
    
    # Prediction
    pred = rewards["prediction"]
    _require_field([
        "enabled",
        "weight",
        "align_weight",
        "target_weight",
        "align_scale",
        "target_scale",
    ], pred, "environment.rewards.prediction")
    
    # Early termination
    early = rewards["early_termination"]
    _require_field([
        "enabled",
        "penalty",
        "min_penalty",
        "penalty_decay_power",
        "population_low_threshold",
        "population_threshold",
        "extinction_penalty",
        "require_budget_depleted",
    ], early, "environment.rewards.early_termination")
    
    # Actions section
    _require_field(["weight_cost"], actions, "actions")
    for act_name in ["noop", "count_bacteria", "sequencing", "dose"]:
        if act_name not in actions:
            raise ValueError(f"Missing required section 'actions.{act_name}'")
        act = actions[act_name]
        _require_field(["id", "cost", "duration"], act, f"actions.{act_name}")
        if act_name == "dose":
            _require_field(["cost_per_unit"], act, "actions.dose")
    
    # Model section
    _require_field(["hidden_dim", "rnn_layers", "n_discrete", "dose_action_index", "k_doses", "sigmoid_scale_factor"], 
                  model, "model")
    
    # PPO section
    _require_field([
        "gamma", "gae_lambda", "clip_eps", "vf_coef", "ent_coef", "max_grad_norm",
        "rollout_steps", "epochs", "seq_len", "batch_seq_len", "lr"
    ], ppo, "ppo")
    
    # Training section
    _require_field([
        "total_updates", "seed", "checkpoint_interval", "log_interval", 
        "save_dir", "experiment_name", "save_checkpoints_per_run"
    ], training, "training")


def load_config(config_path: Optional[Union[str, Path]] = None) -> CompleteConfig:
    """
    Load configuration from YAML file or use defaults, validate and
    construct typed dataclasses matching the YAML structure used in
    `rl/configs/training_config_simple_rewards.yaml`.

    Args:
        config_path: Path to YAML configuration file, or None to use defaults

    Returns:
        CompleteConfig object with all validated parameters
    """
    # Load raw dict (either defaults or YAML file)
    if config_path is None:
        config_dict = _get_default_config()
        print("ℹ Using default configuration")
    else:
        if not HAS_YAML:
            raise RuntimeError(
                "PyYAML is required to load config files. Install it with: pip install pyyaml"
            )
        config_file = Path(config_path)
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

    # Validate semantics/constraints without applying defaults
    _validate_config(config_dict)

    # Build nested dataclasses directly from YAML (no defaults)
    env = config_dict["environment"]

    # Tracking, reward_buffers and timing
    tracking_cfg = TrackingConfig(**env.get("tracking", {}))
    reward_buffers_cfg = RewardBuffersConfig(**env.get("reward_buffers", {}))

    timing_raw = env.get("timing", {})
    count_window = CountWindowConfig(**timing_raw.get("count_window", {"min_elapsed": 0, "max_elapsed": 0}))
    seq_window = SeqWindowConfig(**timing_raw.get("seq_window", {"min_elapsed": 0, "max_elapsed": 0}))
    timing_cfg = TimingConfig(
        t_count_freshness=timing_raw.get("t_count_freshness", 0),
        t_seq_freshness=timing_raw.get("t_seq_freshness", 0),
        max_count_window=timing_raw.get("max_count_window", 0),
        critical_ratio=timing_raw.get("critical_ratio", 1.0),
        count_window=count_window,
        seq_window=seq_window,
    )

    # Rewards: filter keys to match dataclasses to avoid unexpected fields
    def _filter_keys_for(cls, dct: Dict[str, Any]) -> Dict[str, Any]:
        params = signature(cls).parameters
        allowed = set(params.keys())
        return {k: v for k, v in dct.items() if k in allowed}

    rewards = env.get("rewards", {})
    informed = InformedDosingConfig(**_filter_keys_for(InformedDosingConfig, rewards.get("informed_dosing", {})))
    sequencing = SequencingRewardConfig(**_filter_keys_for(SequencingRewardConfig, rewards.get("sequencing", {})))
    counting = CountingRewardConfig(**_filter_keys_for(CountingRewardConfig, rewards.get("counting", {})))
    noop = NoopRewardConfig(**_filter_keys_for(NoopRewardConfig, rewards.get("noop", {})))
    critical_penalties = CriticalPenaltiesConfig(**_filter_keys_for(CriticalPenaltiesConfig, rewards.get("critical_penalties", {})))
    pop_maint_dict = dict(rewards.get("population_maintenance", {}))
    pop_maint_dict["target_population"] = int(env.get("population", {}).get("target_population", 0))
    population_maintenance = PopulationMaintenanceConfig(**_filter_keys_for(PopulationMaintenanceConfig, pop_maint_dict))
    survival_bonus = SurvivalBonusConfig(**_filter_keys_for(SurvivalBonusConfig, rewards.get("survival_bonus", {})))
    prediction = PredictionRewardConfig(**_filter_keys_for(PredictionRewardConfig, rewards.get("prediction", {})))
    early_termination = EarlyTerminationConfig(**_filter_keys_for(EarlyTerminationConfig, rewards.get("early_termination", {})))
    
    # Budget, tracking, history, reward_buffers, population
    budget_cfg = BudgetConfig(**env.get("budget", {}))
    population_cfg = PopulationConfig(**env.get("population", {}))
    tracking_cfg = TrackingConfig(**env.get("tracking", {}))
    reward_buffers_cfg = RewardBuffersConfig(**env.get("reward_buffers", {}))

    reward_cfg = RewardConfig(
        informed_dosing=informed,
        sequencing=sequencing,
        counting=counting,
        noop=noop,
        critical_penalties=critical_penalties,
        population_maintenance=population_maintenance,
        survival_bonus=survival_bonus,
        prediction=prediction,
        early_termination=early_termination,
        budget=budget_cfg,
        population=population_cfg,
        tracking=tracking_cfg,
        reward_buffers=reward_buffers_cfg,
    )

    # Budget and population
    budget_cfg = BudgetConfig(**env.get("budget", {}))
    population_cfg = PopulationConfig(**env.get("population", {}))

    # initial spawn range
    spawn = env.get("initial_bacteria_per_type_range")
    parsed_spawn = None
    if spawn is not None:
        if isinstance(spawn, dict):
            parsed_spawn = (int(spawn.get("0") or spawn.get("min")), int(spawn.get("1") or spawn.get("max")))
        else:
            parsed_spawn = (int(spawn[0]), int(spawn[1]))

    env_cfg = EnvironmentConfig(
        max_steps=int(env.get("max_steps", 0)),
        k_doses=int(env.get("k_doses", 0)),
        device=str(env.get("device", "cpu")),
        dtype=str(env.get("dtype", "float32")),
        rewards=reward_cfg,
        timing=timing_cfg,
        initial_bacteria_per_type_range=parsed_spawn,
        warmup_skip_steps=int(env.get("warmup_skip_steps", 0)),
        enable_individual_tracking=bool(env.get("tracking", {}).get("enabled", True)),
        max_individual_history=int(env.get("tracking", {}).get("max_individual_history", 100)),
        max_tracked_individuals=env.get("tracking", {}).get("max_tracked_individuals", 2000),
        max_history_steps=int(env.get("history", {}).get("max_steps", 2000)),
        max_recent_dose_events=int(env.get("reward_buffers", {}).get("max_recent_dose_events", 256)),
        population_target=float(env.get("population", {}).get("target_population", population_cfg.target_population)),
        population_norm=float(env.get("population", {}).get("population_norm", population_cfg.population_norm)),
        budget_init=float(env.get("budget", {}).get("budget_init", budget_cfg.budget_init)),
        budget_norm=float(env.get("budget", {}).get("budget_norm", budget_cfg.budget_norm)),
    )

    # Actions (keep raw dicts for action details)
    actions_raw = config_dict.get("actions", {})
    weight = float(actions_raw.get("weight_cost", 0.0))
    noop = actions_raw.get("noop", {})
    count = actions_raw.get("count_bacteria", {})
    seq = actions_raw.get("sequencing", {})
    dose = actions_raw.get("dose", {})
    actions_cfg = ActionConfig(
        cost_weight=weight,
        noop_cost=float(noop.get("cost", 0.0)),
        count_cost=float(count.get("cost", 0.0)),
        sequencing_cost=float(seq.get("cost", 0.0)),
        sequencing_duration=int(seq.get("duration", 0)),
        dose_cost=float(dose.get("cost", 0.0)),
        dose_cost_per_unit=float(dose.get("cost_per_unit", 0.0)),
    )

    model_cfg = ModelConfig(**config_dict.get("model", {}))
    ppo_cfg = PPOConfig(**config_dict.get("ppo", {}))
    training_cfg = TrainingConfig(**config_dict.get("training", {}))

    return CompleteConfig(
        environment=env_cfg,
        actions=actions_cfg,
        model=model_cfg,
        ppo=ppo_cfg,
        training=training_cfg,
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
        population_target = (
            config.environment.population_target
            if config.environment.population_target is not None
            else config.environment.rewards.population.target_population
        )
        population_norm = (
            config.environment.population_norm
            if config.environment.population_norm is not None
            else config.environment.rewards.population.population_norm
        )
        budget_init = (
            config.environment.budget_init
            if config.environment.budget_init is not None
            else config.environment.rewards.budget.budget_init
        )
        budget_norm = (
            config.environment.budget_norm
            if config.environment.budget_norm is not None
            else config.environment.rewards.budget.budget_norm
        )

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
                "population": {
                    "target_population": population_target,
                    "population_norm": population_norm,
                },
                "budget": {
                    "budget_init": budget_init,
                    "budget_norm": budget_norm,
                },
                "rewards": {
                    "population": {
                        k: v
                        for k, v in config.environment.rewards.population.__dict__.items()
                        if k not in {"target_population", "population_norm"}
                    },
                    "population_maintenance": (
                        {
                            k: v
                            for k, v in config.environment.rewards.population_maintenance.__dict__.items()
                        }
                        if config.environment.rewards.population_maintenance is not None
                        else None
                    ),
                    "dose": {
                        k: v for k, v in config.environment.rewards.dose.__dict__.items()
                    },
                    "budget": {
                        k: v
                        for k, v in config.environment.rewards.budget.__dict__.items()
                        if k not in {"budget_init", "budget_norm"}
                    },
                    "survival_bonus": {
                        k: v for k, v in config.environment.rewards.survival_bonus.__dict__.items()
                    },
                    "informed_dosing": {
                        k: v for k, v in config.environment.rewards.informed_dosing.__dict__.items()
                    },
                    "regular_monitoring": {
                        k: v for k, v in config.environment.rewards.regular_monitoring.__dict__.items()
                    },
                    "critical_inaction": {
                        k: v for k, v in config.environment.rewards.critical_inaction.__dict__.items()
                    },
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
