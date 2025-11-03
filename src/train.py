"""
Headless RL Training Entry Point.

Train the RL agent without visualization for maximum performance.
All configuration is loaded from YAML files.

Usage:
    python src/train.py --config src/rl/configs/training_config.yaml
    python src/train.py --config src/rl/configs/training_config_fast.yaml

The training logic is in rl.training_utils, which is also used by:
    - src/train_with_visualization.py (training with live plots)

View TensorBoard during/after training:
    tensorboard --logdir=./checkpoints --port=6006
"""

import argparse
import sys
from pathlib import Path

from rl.config_loader import load_config
from rl.training_config import set_global_seed
from rl.training_utils import (
    train,
    _setup_logger_and_log_startup,
    _create_environment,
    _build_ppo_config,
    _save_configs,
)


def main():
    """Main entry point for headless training."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train Recurrent PPO on bacteria simulation (headless mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use fast config for testing (10 updates)
  python src/train.py --config src/rl/configs/training_config_fast.yaml

  # Use default config
  python src/train.py --config src/rl/configs/training_config.yaml

  # Use production config (200 updates)
  python src/train.py --config src/rl/configs/training_config_production.yaml
  
View TensorBoard:
  tensorboard --logdir=./checkpoints --port=6006
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="src/rl/configs/training_config.yaml",
        help="Path to YAML configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return 1
    
    # Setup
    save_dir = Path(config.training.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = _setup_logger_and_log_startup(save_dir, config)
    
    # Set seed
    set_global_seed(config.training.seed)
    logger.log_debug(f"✓ Random seed set to: {config.training.seed}")
    
    # Create environment
    env = _create_environment(config, logger)
    logger.log_info(f"Observation dimension: {env.get_obs_dim()}")
    
    # Build PPO configuration
    ppo_config = _build_ppo_config(env, config)
    
    # Save all configurations
    _save_configs(save_dir, config, logger)
    
    # Train
    logger.log_info("="*70)
    logger.log_info("Starting Headless Training (No Visualization)")
    logger.log_info("="*70)
    
    train(ppo_config, env, save_dir, config.training.total_updates, logger)
    
    logger.log_info("="*70)
    logger.log_info("✓ Training complete!")
    logger.log_info(f"Logs: {save_dir / 'training.log'}")
    logger.log_info(f"Metrics: {save_dir / 'training_log.json'}")
    logger.log_info(f"TensorBoard: tensorboard --logdir={save_dir / config.training.experiment_name} --port=6006")
    logger.log_info("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
