"""
Main entry point for the bacteria simulation - FIXED VERSION.

Properly integrates PyQt5 and Matplotlib to prevent segmentation faults.

Run: python simulation_run.py
Dependencies: mesa, numpy, scipy, matplotlib, PyQt5
"""

from PyQt5 import QtWidgets
import sys

from simulation.model import BacteriaModel
from simulation.ui import SimulatorUI


def main():
    """Initialize and run the bacteria simulation"""
    # Create QApplication ONCE - this manages the event loop
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Initialize simulation model
    model = BacteriaModel()
    
    # Create UI with embedded matplotlib
    ui = SimulatorUI(model)
    
    # Run animation (doesn't block)
    ui.run()
    
    # Show windows
    if ui.control_panel.window:
        ui.control_panel.window.show()
    ui.viz_window.show()
    
    # Run the Qt event loop (this is where everything happens)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
