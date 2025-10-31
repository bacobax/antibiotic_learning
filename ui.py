"""
Refactored UI with proper PyQt5 + Matplotlib integration.
Fixes segmentation fault by embedding matplotlib in PyQt5 windows.
"""

import matplotlib.animation as animation
import time
from PyQt5 import QtWidgets, QtCore

from config import (
    DEFAULT_STEPS_PER_FRAME, 
    MIN_STEPS_PER_FRAME, MAX_STEPS_PER_FRAME,
    SLOW_MODE_FRAME_SKIP, PERFORMANCE_MODE,
    STATS_UPDATE_INTERVAL, VISUALIZATION_UPDATE_INTERVAL
)
from tracking import IndividualPlotter
from control_panel import ControlPanel
from visualization import SimulationVisualizer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class VisualizationWindow(QtWidgets.QMainWindow):
    """PyQt5 window containing the embedded matplotlib visualization"""
    
    def __init__(self, visualizer):
        super().__init__()
        self.visualizer = visualizer
        self.setWindowTitle("Bacteria Simulation")
        self.setGeometry(100, 100, 1400, 700)
        
        # Embed matplotlib figure in PyQt5
        self.canvas = FigureCanvas(visualizer.fig)
        self.setCentralWidget(self.canvas)
        
        # Setup toolbar
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.addToolBar(self.toolbar)


class SimulatorUI:
    """Main UI for bacteria simulation visualization and control."""
    
    def __init__(self, model):
        self.model = model
        self.paused = True
        self.simulation_started = False
        self.population_extinct = False
        self.latest_dose = 0.0
        
        # Speed control
        self.steps_per_frame = DEFAULT_STEPS_PER_FRAME
        self.frame_counter = 0
        self.animation = None
        
        # Performance tracking
        self.last_step_time = None
        self.step_times = []
        self.max_step_time_samples = 30
        self.actual_steps_per_second = 0.0
        
        # Performance mode
        self.performance_mode = PERFORMANCE_MODE
        self.stats_update_counter = 0
        self.viz_update_counter = 0
        self.graph_update_counter = 0
        
        # Individual tracking
        self.individual_plotter = IndividualPlotter(
            self.model.individual_tracker,
            on_close_callback=self.on_individual_window_close
        )
        
        # Setup visualization with proper PyQt5 embedding
        self.visualizer = SimulationVisualizer(
            model=self.model,
            on_click_callback=self.on_bacterium_click
        )
        
        # Create visualization window
        self.viz_window = VisualizationWindow(self.visualizer)
        
        # Setup control panel
        self.control_panel = ControlPanel(
            model=self.model,
            on_toggle_pause=self.toggle_pause,
            on_reset=self.reset_sim,
            on_apply_antibiotic=self.apply_antibiotic,
            on_speed_change=self.handle_speed_change,
            on_view_bacterium=self.view_bacterium
        )
        
        # Position windows side by side
        self._arrange_windows()

    def _arrange_windows(self):
        """Arrange visualization and control panel windows"""
        # Control panel first
        if self.control_panel.window:
            self.control_panel.window.move(0, 0)
            self.control_panel.window.resize(400, 900)
        
        # Visualization next to it
        self.viz_window.move(420, 0)
        self.viz_window.show()

    def on_bacterium_click(self, bacterium_id):
        """Handle bacterium click from visualizer"""
        self.individual_plotter.update_plots(bacterium_id)

    def on_individual_window_close(self):
        """Handle individual tracking window being closed by user"""
        self.visualizer.clear_highlight()
        print("Individual tracking window closed")

    def view_bacterium(self, bacterium_id):
        """View selected bacterium from control panel"""
        self.visualizer.set_highlighted_bacterium(bacterium_id)
        self.individual_plotter.update_plots(bacterium_id)

    def toggle_pause(self):
        """Toggle simulation pause/start"""
        if not self.simulation_started:
            self.simulation_started = True
            self.paused = False
            self.control_panel.set_pause_button_text("Pause")
            print("Simulation started")
        elif self.population_extinct:
            return
        else:
            self.paused = not self.paused
            if self.paused:
                self.control_panel.set_pause_button_text("Resume")
            else:
                self.control_panel.set_pause_button_text("Pause")

    def reset_sim(self):
        """Reset simulation to initial conditions"""
        if hasattr(self, 'individual_plotter'):
            self.individual_plotter.close()
        
        self.model.reset()
        
        self.paused = True
        self.simulation_started = False
        self.population_extinct = False
        self.frame_counter = 0
        
        self.last_step_time = None
        self.step_times = []
        self.actual_steps_per_second = 0.0
        
        self.control_panel.set_pause_button_text("Start")
        self.control_panel.set_pause_button_state("normal")
        
        self.visualizer.clear_highlight()
        
        self.individual_plotter = IndividualPlotter(
            self.model.individual_tracker,
            on_close_callback=self.on_individual_window_close
        )
        
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
        self.stats_update_counter = 0
        self.viz_update_counter = 0
        print(f"Performance mode: {'ON' if enabled else 'OFF'}")

    def handle_speed_change(self, direction):
        """Handle speed change from control panel"""
        if direction == -1:
            self.steps_per_frame = max(MIN_STEPS_PER_FRAME, self.steps_per_frame - 1)
        elif direction == 1:
            self.steps_per_frame = min(MAX_STEPS_PER_FRAME, self.steps_per_frame + 1)
        else:
            self.steps_per_frame = DEFAULT_STEPS_PER_FRAME
        
        self.control_panel.update_speed_display(self.steps_per_frame)

    def run(self):
        """Run the simulation"""
        self.paused = True
        
        self.control_panel.set_ui_reference(self)
        
        # Create animation that updates matplotlib figure
        self.animation = animation.FuncAnimation(
            self.visualizer.fig, 
            self.update, 
            interval=self.visualizer.animation_interval, 
            blit=False, 
            cache_frame_data=False
        )
        
        # Show windows - doesn't block because matplotlib is embedded
        if self.control_panel.window:
            self.control_panel.window.show()
        
        # Qt event loop will be run by PyQt5 QApplication
        # No need to call plt.show()

    def update(self, frame):
        """Animation update callback"""
        try:
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
                    for _ in range(self.steps_per_frame):
                        self.model.step()
                        steps_executed += 1
                else:
                    self.frame_counter += 1
                    if self.frame_counter >= SLOW_MODE_FRAME_SKIP:
                        self.model.step()
                        steps_executed = 1
                        self.frame_counter = 0

                # Performance tracking
                if steps_executed > 0:
                    step_duration = time.time() - step_start
                    self.step_times.append(step_duration)
                    if len(self.step_times) > self.max_step_time_samples:
                        self.step_times.pop(0)
                    
                    avg_duration = sum(self.step_times) / len(self.step_times)
                    if avg_duration > 0:
                        self.actual_steps_per_second = steps_executed / avg_duration

            # Update visualizations
            should_update_graphs = True
            if self.performance_mode:
                self.graph_update_counter += 1
                if self.graph_update_counter >= VISUALIZATION_UPDATE_INTERVAL:
                    self.graph_update_counter = 0
                else:
                    should_update_graphs = False
            
            self.visualizer.update_main_plot()
            
            if should_update_graphs:
                self.visualizer.update_graphs()
            
            # Update stats panel
            should_update_stats = True
            if self.performance_mode:
                self.stats_update_counter += 1
                if self.stats_update_counter >= STATS_UPDATE_INTERVAL:
                    self.stats_update_counter = 0
                else:
                    should_update_stats = False
            
            if should_update_stats:
                stats = self.model.get_population_stats()
                self.control_panel.update_stats_display(stats)
                self.control_panel.update_bacteria_list()
            
            # Update individual plotter if bacterium is selected
            if self.individual_plotter.current_id is not None:
                self.individual_plotter.update_plots(self.individual_plotter.current_id)
            
            # Update UI elements
            self.visualizer.draw()
            self.control_panel.update()
            
        except Exception as e:
            print(f"Error in animation update: {e}")
            import traceback
            traceback.print_exc()
