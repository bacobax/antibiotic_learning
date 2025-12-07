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
from pprint import pprint
from rl.config_loader import load_config
from rl.training_config import set_global_seed
from rl.training_utils import (
    train,
    create_run_directory,
    _setup_logger_and_log_startup,
    _create_environment,
    _build_ppo_config,
    _save_configs,
    _load_checkpoint_into_agent,
    _initialize_agent,
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
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return 1

    
    # Setup
    base_save_dir = Path(config.training.save_dir)
    base_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped run directory
    run_dir = create_run_directory(base_save_dir, config.training.experiment_name)
    logger = _setup_logger_and_log_startup(run_dir, config)
    
    logger.log_info(f"✓ Run directory created: {run_dir}")
    logger.log_info(f"  Structure: {base_save_dir}/{config.training.experiment_name}/{run_dir.name}/")
    
    # Set seed
    set_global_seed(config.training.seed)
    logger.log_debug(f"✓ Random seed set to: {config.training.seed}")
    
    # Create environment
    env = _create_environment(config, logger)
    logger.log_info(f"Observation dimension: {env.get_obs_dim()}")
    
    # Build PPO configuration
    ppo_config = _build_ppo_config(env, config)
    
    # Initialize agent
    agent = _initialize_agent(ppo_config, env)
    
    # Handle checkpoint resumption
    starting_update = 0
    if args.resume:
        logger.log_info(f"Loading checkpoint from: {args.resume}")
        try:
            starting_update = _load_checkpoint_into_agent(agent, args.resume, logger, config)
        except Exception as e:
            logger.log_error(f"Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Determine checkpoint save directory
    if config.training.save_checkpoints_per_run:
        # Save checkpoints in the timestamped run directory
        checkpoint_dir = run_dir
        logger.log_info(f"✓ Checkpoints will be saved in: {checkpoint_dir}")
    else:
        # Save checkpoints in base directory (overwrite each run)
        checkpoint_dir = base_save_dir
        logger.log_info(f"✓ Checkpoints will be saved in: {checkpoint_dir} (shared across runs)")
    
    # Save all configurations to run directory
    _save_configs(run_dir, config, logger)
    
    # Train
    logger.log_info("="*70)
    if starting_update > 0:
        logger.log_info(f"Resuming Headless Training from Update {starting_update}")
    else:
        logger.log_info("Starting Headless Training (No Visualization)")
    logger.log_info("="*70)
    
    train(
        ppo_config,
        env,
        checkpoint_dir,
        config.training.total_updates,
        logger,
        checkpoint_interval=config.training.checkpoint_interval,
        starting_update=starting_update,
        agent=agent,
        log_window_size=config.training.log_window_size,
        log_memory=config.training.log_memory,
        memory_log_interval=config.training.memory_log_interval,
    )
    
    logger.log_info("="*70)
    logger.log_info("✓ Training complete!")
    logger.log_info(f"Logs: {run_dir / 'training.log'}")
    logger.log_info(f"Metrics: {run_dir / 'training_log.json'}")
    logger.log_info(f"TensorBoard: tensorboard --logdir={run_dir / config.training.experiment_name} --port=6006")
    logger.log_info("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
