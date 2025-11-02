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
from PyQt5 import QtWidgets

from model import BacteriaModel
from simulation_ui_agent import AgentSimulatorUI


def main():
    parser = argparse.ArgumentParser(
        description="Run trained RL Agent in visual simulation"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/checkpoint_final_30.pt",
        help="Path to trained agent checkpoint"
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
    
    # Create UI
    print(f"Loading agent from checkpoint: {checkpoint_path}")
    ui = AgentSimulatorUI(model, str(checkpoint_path))
    
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
