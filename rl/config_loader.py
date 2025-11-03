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
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class ActionConfig:
    """Configuration for discrete actions."""
    noop_cost: float
    count_cost: float
    sequencing_cost: float
    sequencing_duration: int
    dose_cost_per_unit: float


@dataclass
class EnvironmentConfig:
    """Environment and reward configuration."""
    max_steps: int
    k_doses: int
    target_population: float
    population_norm: float
    budget_init: float
    budget_norm: float
    w_pop: float
    w_genome: float
    w_cost: float
    w_population_maintenance: float
    device: str
    dtype: str


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_dim: int
    rnn_layers: int
    n_discrete: int
    dose_action_index: int
    k_doses: int


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
            "target_population": 500,
            "population_norm": 1000.0,
            "budget_init": 100.0,
            "budget_norm": 100.0,
            "w_pop": 1.0,
            "w_genome": 0.5,
            "w_cost": 0.05,
            "w_population_maintenance": 0.01,
            "device": "cpu",
            "dtype": "float32",
        },
        "actions": {
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
    
    # Environment validation
    if env.get("k_doses", 1) <= 0:
        raise ValueError("k_doses must be > 0")
    if env.get("target_population", 1) <= 0:
        raise ValueError("target_population must be > 0")
    if env.get("budget_init", 1) < 0:
        raise ValueError("budget_init must be >= 0")
    if env.get("device", "cpu").lower() not in ["cpu", "cuda", "mps"]:
        raise ValueError(f"Invalid device: {env.get('device')}")
    if env.get("dtype", "float32").lower() not in ["float32", "float64", "float16"]:
        raise ValueError(f"Invalid dtype: {env.get('dtype')}")
    
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
        raise ValueError("gae_lambda must be in (0, 1]")
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
                config_dict = yaml.safe_load(f) or {}
        except Exception as e:
            raise ValueError(f"Failed to load YAML config {config_file}: {e}")
        
        print(f"✓ Loaded configuration from: {config_file}")
    
    # Merge with defaults to fill missing values
    config_dict = _merge_with_defaults(config_dict)
    
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
    
    # Create dataclass instances
    env_cfg = EnvironmentConfig(**config_dict["environment"])
    
    actions_cfg = ActionConfig(
        noop_cost=noop.get("cost", 0.0),
        count_cost=count.get("cost", 0.0),
        sequencing_cost=seq.get("cost", 1.0),
        sequencing_duration=seq.get("duration", 5),
        dose_cost_per_unit=dose.get("cost_per_unit", 0.2),
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
        # Convert dataclasses to dicts
        config_dict = {
            "environment": {
                k: v for k, v in config.environment.__dict__.items()
            },
            "actions": {
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
