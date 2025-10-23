"""
Main entry point for the bacteria simulation.

Run: python main.py
Dependencies: mesa, numpy, scipy, matplotlib, tkinter (optional)
"""

from model import BacteriaModel
from ui import SimulatorUI


def main():
    """Initialize and run the bacteria simulation"""
    model = BacteriaModel()
    ui = SimulatorUI(model)
    ui.run()


if __name__ == "__main__":
    main()
