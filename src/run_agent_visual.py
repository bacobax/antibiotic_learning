"""
Run the trained RL Agent in visual simulation mode.

This script loads a trained agent checkpoint and runs it in the bacteria simulation
with visual feedback. The agent makes all decisions (dosing, sequencing, counting).

Usage:
    python run_agent_visual.py                          # Use default checkpoint
    python run_agent_visual.py --checkpoint path/to/checkpoint.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
from PyQt5 import QtWidgets

from simulation.model import BacteriaModel
from simulation.simulation_ui_agent import AgentSimulatorUI
from rl.config_loader import load_config
from rl.training_utils import _create_environment


DEFAULT_FALLBACK_CONFIG = Path("src/rl/configs/training_config_simple_rewards.yaml")
EMBEDDED_CONFIG_NAME = "complete_config.yaml"


class _ConsoleLogger:
    """Minimal logger compatible with training utilities."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def log_info(self, message: str) -> None:
        print(f"[env] {message}")

    def log_debug(self, message: str) -> None:
        if self.verbose:
            print(f"[env-debug] {message}")


def _resolve_config_path(config_arg: Optional[str], checkpoint_path: Path) -> Path:
    """Determine which config file should drive the playback environment."""
    if config_arg:
        candidate = Path(config_arg).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Specified config not found: {candidate}")
        return candidate

    checkpoint_path = checkpoint_path.expanduser()
    checkpoint_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent

    candidates = [
        checkpoint_dir / EMBEDDED_CONFIG_NAME,
        checkpoint_dir.parent / EMBEDDED_CONFIG_NAME if checkpoint_dir.parent != checkpoint_dir else None,
        Path("checkpoints") / EMBEDDED_CONFIG_NAME,
        DEFAULT_FALLBACK_CONFIG,
    ]

    for candidate in candidates:
        if candidate is None:
            continue
        expanded = candidate.expanduser().resolve()
        if expanded.exists():
            return expanded

    raise FileNotFoundError(
        "Could not locate a configuration file. Pass --config explicitly or ensure "
        "complete_config.yaml exists next to the checkpoint."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run trained RL Agent in visual simulation"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/checkpoint_200.pt",
        help="Path to trained agent checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to the training config or complete_config.yaml. "
            "If omitted, the script looks next to the checkpoint and then uses "
            "src/rl/configs/training_config_simple_rewards.yaml as a fallback"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Verify checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print(f"Available checkpoints:")
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            for ckpt in sorted(checkpoints_dir.glob("checkpoint_*.pt")):
                print(f"  - {ckpt}")
        sys.exit(1)

    # Resolve configuration file
    try:
        config_path = _resolve_config_path(args.config, checkpoint_path)
    except FileNotFoundError as err:
        print(f"Error locating config: {err}")
        sys.exit(1)

    print(f"Loading environment configuration from: {config_path}")
    try:
        config = load_config(str(config_path))
    except Exception as exc:
        print(f"❌ Failed to load configuration: {exc}")
        sys.exit(1)
    
    # Create QApplication
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create model
    print("Initializing bacteria model...")
    model = BacteriaModel()
    
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to {args.seed}")

    env_logger = _ConsoleLogger()
    env = _create_environment(
        config,
        env_logger,
        mesa_model_factory=lambda: model,
    )
    
    # Create UI
    print(f"Loading agent from checkpoint: {checkpoint_path}")
    ui = AgentSimulatorUI(model, env, str(checkpoint_path))
    
    # Setup and run
    ui.run()
    ui.viz_window.show()
    
    print("\nAgent Simulator Ready!")
    print("Controls:")
    print("  - Click 'Start' to begin the simulation")
    print("  - The agent will automatically control antibiotic dosing")
    print("  - Use +/- buttons to adjust simulation speed")
    print("  - Click 'Reset' to start a new episode")
    
    # Run event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
