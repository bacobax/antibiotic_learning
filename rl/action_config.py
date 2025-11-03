"""
DEPRECATED: Action configuration has been consolidated into training_config.yaml

This module is kept for backwards compatibility only.
All configuration is now handled by config_loader.py and training_config.yaml

For new code, use:
    from rl.config_loader import load_config
    config = load_config("training_config.yaml")
"""

import warnings
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_action_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    DEPRECATED: Use load_config() from config_loader instead.
    
    Load action configuration from YAML file.
    
    If config_path is None, returns default configuration.
    
    Args:
        config_path: Path to YAML config file (or None for defaults)
    
    Returns:
        Dict with action configuration
    """
    warnings.warn(
        "load_action_config() is deprecated. Use config_loader.load_config() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if config_path is None:
        return get_default_action_config()
    
    if not HAS_YAML:
        raise RuntimeError(
            f"PyYAML is required to load config file {config_path}. "
            "Install it with: pip install pyyaml"
        )
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Action config file not found: {config_file}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to load YAML config {config_file}: {e}")
    
    return config


def get_default_action_config() -> Dict[str, Any]:
    """
    Get default action configuration.
    
    DEPRECATED: Use config_loader.load_config() instead.
    
    Returns:
        Default configuration dict
    """
    warnings.warn(
        "get_default_action_config() is deprecated. Use config_loader.load_config() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return {
        "actions": {
            "noop": {
                "id": 0,
                "name": "No operation",
                "cost": 0.01,
                "duration": 0,
                "description": "Wait without taking action"
            },
            "count_bacteria": {
                "id": 1,
                "name": "Count bacteria",
                "cost": 0.0,
                "duration": 0,
                "description": "Measure current bacteria population"
            },
            "sequencing": {
                "id": 2,
                "name": "Genome sequencing",
                "cost": 1.0,
                "duration": 5,
                "description": "Sequence bacteria genome (takes 5 steps)"
            },
            "dose": {
                "id": 3,
                "name": "Administer antibiotics",
                "cost": 0.2,  # per unit
                "duration": 0,
                "description": "Dose antibiotics (continuous action)"
            }
        },
        "environment": {
            "target_population": 500,
            "budget_init": 100.0,
            "max_steps": 1000,
            "w_pop": 1.0,
            "w_genome": 0.5,
            "w_cost": 0.05,
        }
    }


def save_default_config(output_path: str = "actions_config.yaml") -> None:
    """
    DEPRECATED: Use config_loader.save_config() instead.
    
    Save default action configuration to a YAML file for user reference.
    
    Args:
        output_path: Where to save the file
    """
    warnings.warn(
        "save_default_config() is deprecated. Use config_loader.save_config() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if not HAS_YAML:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")
    
    config = get_default_action_config()
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Default action config saved to: {output_path}")


if __name__ == "__main__":
    # Example: Create and save default config
    save_default_config()
