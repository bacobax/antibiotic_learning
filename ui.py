"""
Simulation UI and visualization.
"""

import matplotlib.animation as animation
import time

from config import (
    DEFAULT_STEPS_PER_FRAME, 
    MIN_STEPS_PER_FRAME, MAX_STEPS_PER_FRAME,
    SLOW_MODE_FRAME_SKIP, PERFORMANCE_MODE,
    STATS_UPDATE_INTERVAL, VISUALIZATION_UPDATE_INTERVAL
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
        
        # Speed control - simpler and more direct
        self.steps_per_frame = DEFAULT_STEPS_PER_FRAME
        self.frame_counter = 0  # For slow mode (when steps_per_frame = 0)
        self.animation = None
        
        # Performance tracking
        self.last_step_time = None
        self.step_times = []  # Track last N step times for averaging
        self.max_step_time_samples = 30
        self.actual_steps_per_second = 0.0
        
        # Performance mode - initialize from config
        self.performance_mode = PERFORMANCE_MODE
        self.stats_update_counter = 0
        self.viz_update_counter = 0
        self.graph_update_counter = 0  # For graph updates
        
        # Individual tracking
        self.individual_plotter = IndividualPlotter(
            self.model.individual_tracker,
            on_close_callback=self.on_individual_window_close
        )
        
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
            on_speed_change=self.handle_speed_change,
            on_view_bacterium=self.view_bacterium
        )

    def on_bacterium_click(self, bacterium_id):
        """Handle bacterium click from visualizer"""
        self.individual_plotter.update_plots(bacterium_id)

    def on_individual_window_close(self):
        """Handle individual tracking window being closed by user"""
        # Clear the highlight in the visualizer
        self.visualizer.clear_highlight()
        print("Individual tracking window closed")

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
        self.frame_counter = 0
        
        # Reset performance tracking
        self.last_step_time = None
        self.step_times = []
        self.actual_steps_per_second = 0.0
        
        # Update control panel
        self.control_panel.set_pause_button_text("Start")
        self.control_panel.set_pause_button_state("normal")
        
        # Clear visualization highlight
        self.visualizer.clear_highlight()
        
        # Reset individual plotter with callback
        self.individual_plotter = IndividualPlotter(
            self.model.individual_tracker,
            on_close_callback=self.on_individual_window_close
        )
        
        # Force UI update
        self.visualizer.update_main_plot()
        self.control_panel.update_stats_display(self.model.get_population_stats())
        self.control_panel.update_bacteria_list(force_update=True)
        
        print("Simulation reset - press Start to begin")

    def apply_antibiotic(self, antibiotic_type, dose):
        """Apply antibiotic of specific type with given dose"""
        self.model.apply_antibiotic(antibiotic_type, dose)
        self.latest_dose = dose

    def toggle_performance_mode(self, enabled):
        """Toggle performance mode"""
        self.performance_mode = enabled
        self.stats_update_counter = 0  # Reset counter
        self.viz_update_counter = 0
        print(f"Performance mode: {'ON' if enabled else 'OFF'}")
        if enabled:
            print(f"  Stats will update every {STATS_UPDATE_INTERVAL} frames instead of every frame")
            print(f"  This significantly speeds up multi-step simulation")

    def handle_speed_change(self, direction):
        """
        Handle speed change from control panel.
        
        Args:
            direction: -1 for slower, 1 for faster, 0 for reset
        """
        if direction == -1:
            self.steps_per_frame = max(MIN_STEPS_PER_FRAME, self.steps_per_frame - 1)
        elif direction == 1:
            self.steps_per_frame = min(MAX_STEPS_PER_FRAME, self.steps_per_frame + 1)
        else:  # reset
            self.steps_per_frame = DEFAULT_STEPS_PER_FRAME
        
        self.control_panel.update_speed_display(self.steps_per_frame)

    def run(self):
        """Run the simulation"""
        self.paused = True
        
        # Pass UI reference to control panel for performance display
        self.control_panel.set_ui_reference(self)
        
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
        steps_executed = 0
        if not self.paused:
            step_start = time.time()
            
            if self.steps_per_frame > 0:
                # Normal mode: run multiple steps per frame
                for _ in range(self.steps_per_frame):
                    self.model.step()  # This now calls _record_history() internally
                    steps_executed += 1
            else:
                # Slow mode: run 1 step every N frames
                self.frame_counter += 1
                if self.frame_counter >= SLOW_MODE_FRAME_SKIP:
                    self.model.step()  # This now calls _record_history() internally
                    steps_executed = 1
                    self.frame_counter = 0

            # Performance tracking
            if steps_executed > 0:
                step_duration = time.time() - step_start
                self.step_times.append(step_duration)
                if len(self.step_times) > self.max_step_time_samples:
                    self.step_times.pop(0)
                
                # Calculate actual steps per second
                avg_duration = sum(self.step_times) / len(self.step_times)
                if avg_duration > 0:
                    self.actual_steps_per_second = steps_executed / avg_duration

        # Update visualizations - with performance mode throttling for graphs
        should_update_graphs = True
        if self.performance_mode:
            self.graph_update_counter += 1
            if self.graph_update_counter >= VISUALIZATION_UPDATE_INTERVAL:
                self.graph_update_counter = 0
            else:
                should_update_graphs = False
        
        # Always update main plot for smooth animation
        self.visualizer.update_main_plot()
        
        # Update graphs conditionally based on performance mode
        if should_update_graphs:
            self.visualizer.update_graphs()
        
        # Update stats panel - with performance mode throttling
        should_update_stats = True
        if self.performance_mode:
            self.stats_update_counter += 1
            if self.stats_update_counter >= STATS_UPDATE_INTERVAL:
                self.stats_update_counter = 0
            else:
                should_update_stats = False
        
        if should_update_stats:
            stats = self.model.get_population_stats()  # Get stats but don't record to history
            self.control_panel.update_stats_display(stats)
            self.control_panel.update_bacteria_list()
        
        # Update individual plotter if bacterium is selected
        if self.individual_plotter.current_id is not None:
            self.individual_plotter.update_plots(self.individual_plotter.current_id)
        
        # Update UI elements
        self.visualizer.draw()
        self.control_panel.update()
