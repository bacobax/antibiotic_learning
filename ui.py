"""
Simulation UI and visualization.
"""

import matplotlib.animation as animation

from config import (
    DEFAULT_STEPS_PER_SECOND, 
    MIN_STEPS_PER_SECOND, MAX_STEPS_PER_SECOND
)
from tracking import IndividualPlotter
from control_panel import ControlPanel
from visualization import SimulationVisualizer


class SimulatorUI:
    """Main UI for bacteria simulation visualization and control."""
    
    def __init__(self, model):
        self.model = model
        self.paused = True
        self.simulation_started = False
        self.population_extinct = False
        self.latest_dose = 0.0
        
        # Speed control
        self.steps_per_second = DEFAULT_STEPS_PER_SECOND
        self.steps_accumulator = 0.0
        self.animation = None
        
        # Individual tracking
        self.individual_plotter = IndividualPlotter(self.model.individual_tracker)
        
        # Setup visualization
        self.visualizer = SimulationVisualizer(
            model=self.model,
            on_click_callback=self.on_bacterium_click
        )
        
        # Setup control panel
        self.control_panel = ControlPanel(
            model=self.model,
            on_toggle_pause=self.toggle_pause,
            on_reset=self.reset_sim,
            on_apply_antibiotic=self.apply_antibiotic,
            on_toggle_hgt=self.toggle_hgt,
            on_speed_change=self.handle_speed_change,
            on_view_bacterium=self.view_bacterium
        )

    def on_bacterium_click(self, bacterium_id):
        """Handle bacterium click from visualizer"""
        self.individual_plotter.update_plots(bacterium_id)

    def view_bacterium(self, bacterium_id):
        """View selected bacterium from control panel"""
        self.visualizer.set_highlighted_bacterium(bacterium_id)
        self.individual_plotter.update_plots(bacterium_id)

    def toggle_pause(self):
        """Toggle simulation pause/start"""
        if not self.simulation_started:
            # First start
            self.simulation_started = True
            self.paused = False
            self.control_panel.set_pause_button_text("Pause")
            print("Simulation started")
        elif self.population_extinct:
            # Can't resume if extinct
            return
        else:
            # Normal pause/resume
            self.paused = not self.paused
            if self.paused:
                self.control_panel.set_pause_button_text("Resume")
            else:
                self.control_panel.set_pause_button_text("Pause")

    def reset_sim(self):
        """Reset simulation to initial conditions"""
        # Close individual tracker window if it exists
        if hasattr(self, 'individual_plotter'):
            self.individual_plotter.close()
        
        # Reset the model
        self.model.reset()
        
        # Reset UI state
        self.paused = True
        self.simulation_started = False
        self.population_extinct = False
        self.steps_accumulator = 0.0
        
        # Update control panel
        self.control_panel.set_pause_button_text("Start")
        self.control_panel.set_pause_button_state("normal")
        
        # Clear visualization highlight
        self.visualizer.clear_highlight()
        
        # Reset individual plotter
        self.individual_plotter = IndividualPlotter(self.model.individual_tracker)
        
        # Force UI update
        self.visualizer.update_main_plot()
        self.control_panel.update_stats_display(self.model.get_population_stats())
        self.control_panel.update_bacteria_list(force_update=True)
        
        print("Simulation reset - press Start to begin")

    def apply_antibiotic(self, dose):
        """Apply antibiotic with given dose"""
        self.model.apply_antibiotic(dose)
        self.latest_dose = dose

    def toggle_hgt(self, enabled):
        """Toggle horizontal gene transfer"""
        self.model.enable_hgt = enabled

    def handle_speed_change(self, direction):
        """
        Handle speed change from control panel.
        
        Args:
            direction: -1 for slower, 1 for faster, 0 for reset
        """
        if direction == -1:
            self.steps_per_second = max(MIN_STEPS_PER_SECOND, self.steps_per_second - 1)
        elif direction == 1:
            self.steps_per_second = min(MAX_STEPS_PER_SECOND, self.steps_per_second + 1)
        else:  # reset
            self.steps_per_second = DEFAULT_STEPS_PER_SECOND
        
        self.control_panel.update_speed_display(self.steps_per_second)

    def run(self):
        """Run the simulation"""
        self.paused = True
        self.animation = animation.FuncAnimation(
            self.visualizer.fig, 
            self.update, 
            interval=self.visualizer.animation_interval, 
            blit=False, 
            cache_frame_data=False
        )
        self.visualizer.show()

    def update(self, frame):
        """Animation update callback"""
        # Check for population extinction
        population = len(self.model.agent_set)
        if population == 0 and self.simulation_started and not self.population_extinct:
            self.population_extinct = True
            self.paused = True
            self.control_panel.set_pause_button_text("Extinct")
            self.control_panel.set_pause_button_state("disabled")
            print("Population extinct! Simulation paused. Press Reset to restart.")
        
        # Step simulation if not paused
        if not self.paused:
            self.steps_accumulator += self.steps_per_second / self.visualizer.animation_fps
            while self.steps_accumulator >= 1.0:
                self.model.step()
                self.steps_accumulator -= 1.0

        # Update visualizations
        self.visualizer.update_main_plot()
        self.control_panel.update_stats_display(self.model.get_population_stats())
        self.control_panel.update_bacteria_list()
        
        # Update individual plotter if bacterium is selected
        if self.individual_plotter.current_id is not None:
            self.individual_plotter.update_plots(self.individual_plotter.current_id)
        
        # Update UI elements
        self.visualizer.draw()
        self.control_panel.update()
