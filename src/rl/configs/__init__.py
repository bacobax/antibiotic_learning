"""
Configuration package for PPO training on bacteria simulation.

Contains all YAML configuration files for training.

Config files:
- training_config.yaml: Default configuration
- training_config_fast.yaml: Quick testing configuration
- training_config_production.yaml: Full production training

To use:
    from rl.config_loader import load_config
    config = load_config("training_config.yaml")  # loads from rl/config/
    
Or with absolute path:
    config = load_config("rl/config/training_config_production.yaml")
"""
