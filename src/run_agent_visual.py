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
from typing import Optional, Any
from PyQt5 import QtWidgets

from simulation.model import BacteriaModel
from simulation.simulation_ui_agent import AgentSimulatorUI
from rl.config_loader import load_config
from rl.training_utils import _create_environment


DEFAULT_FALLBACK_CONFIG = Path("src/rl/configs/training_config_simple_rewards.yaml")
EMBEDDED_CONFIG_NAME = "complete_config.yaml"


def _make_shared_model_factory(model: BacteriaModel):
    """Return a factory that reuses and resets the provided model instance."""

    def _factory(*factory_args: Any, **factory_kwargs: Any):
        # Support both positional and keyword-based population overrides (N keyword)
        initial_total_bacteria: Optional[int] = None
        if factory_args:
            initial_total_bacteria = factory_args[0]

        keyword_override = factory_kwargs.pop("N", None)
        if keyword_override is not None:
            initial_total_bacteria = keyword_override

        # Other keyword arguments (tracking, field toggles, etc.) are already
        # embedded in the shared model instance, so we intentionally ignore them.

        if initial_total_bacteria is not None:
            try:
                model._initial_bacteria_count = int(initial_total_bacteria)
            except (AttributeError, TypeError, ValueError):
                pass

        model.reset()
        return model

    return _factory


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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override the environment max_steps (set a large value to extend episodes)"
    )
    parser.add_argument(
        "--ignore-max-steps",
        action="store_true",
        help="Ignore the config's max_steps and let budget/early termination end the episode"
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

    if args.ignore_max_steps:
        config.environment.max_steps = None
        print("Ignoring max_steps: episodes will continue until budget depletion or early termination triggers.")
    elif args.max_steps is not None:
        config.environment.max_steps = args.max_steps
        print(f"Overriding max_steps with: {args.max_steps}")
    
    # Create QApplication
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create model aligned with the training configuration so that the shared
    # factory can safely ignore reconfiguration kwargs passed by the environment.
    print("Initializing bacteria model...")
    model = BacteriaModel(
        enable_individual_tracking=config.environment.enable_individual_tracking,
        max_individual_history=config.environment.max_individual_history,
        max_tracked_individuals=config.environment.max_tracked_individuals,
        max_history_steps=config.environment.max_history_steps,
        use_torch_fields=config.environment.use_torch_fields,
        field_device=config.environment.field_device,
        enable_food_diffusion=config.environment.enable_food_diffusion,
    )
    
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed set to {args.seed}")

    env_logger = _ConsoleLogger()
    shared_model_factory = _make_shared_model_factory(model)
    env = _create_environment(
        config,
        env_logger,
        mesa_model_factory=shared_model_factory,
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
