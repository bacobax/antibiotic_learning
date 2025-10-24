"""
Matplotlib visualization for bacteria simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from config import BACTERIAL_TYPES, ANIMATION_FPS


class SimulationVisualizer:
    """Matplotlib visualizer for bacteria simulation."""
    
    def __init__(self, model, on_click_callback):
        """
        Initialize visualizer.
        
        Args:
            model: The simulation model
            on_click_callback: Callback for handling clicks on bacteria
        """
        self.model = model
        self.on_click_callback = on_click_callback
        
        # Color mapping for bacterial types
        self.bacterial_type_names = list(BACTERIAL_TYPES.keys())
        self.color_map = {name: i for i, name in enumerate(self.bacterial_type_names)}
        
        # Animation settings
        self.animation_fps = ANIMATION_FPS
        self.animation_interval = int(1000 / self.animation_fps)
        self.animation = None
        
        # Highlighting
        self.highlighted_bacterium_id = None
        
        # Setup visualization
        self._setup_plots()
        
        # Click handling
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _setup_plots(self):
        """Setup matplotlib figure and subplots"""
        self.fig = plt.figure(figsize=(16, 8))
        gs = self.fig.add_gridspec(2, 2)

        # Main simulation view
        self.ax = self.fig.add_subplot(gs[0, 0])

        # Food level plot
        self.ax_food = self.fig.add_subplot(gs[0, 1])
        self.ax_food.set_xlabel('Steps')
        self.ax_food.set_ylabel('Total Food')
        self.ax_food.grid(True)
        self.line_food, = self.ax_food.plot([], [], label='Food Level', color='green')
        self.ax_food.legend()

        # Population plot
        self.ax_pop = self.fig.add_subplot(gs[1, 0])
        self.ax_pop.set_xlabel('Steps')
        self.ax_pop.set_ylabel('Population')
        self.ax_pop.grid(True)
        self.line_pop, = self.ax_pop.plot([], [], label='Population', color='blue')
        self.ax_pop.legend()

        # Energy plot
        self.ax_energy = self.fig.add_subplot(gs[1, 1])
        self.ax_energy.set_xlabel('Steps')
        self.ax_energy.set_ylabel('Energy (Top 10 Avg)')
        self.ax_energy.grid(True)
        self.line_energy, = self.ax_energy.plot([], [], label='Top 10 Energy', color='red')
        self.ax_energy.legend()

        self.fig.tight_layout()

        # Initialize plot elements
        self.scat = None
        self.highlight_scat = None
        self.im_food = None
        self.im_ab = None

    def _on_click(self, event):
        """Handle mouse clicks to select bacteria"""
        if event.inaxes != self.ax:
            return
            
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
            
        # Find closest bacterium
        min_dist = float('inf')
        closest_bacterium = None
        
        for bacterium in self.model.agent_set:
            dist = np.sqrt((bacterium.pos[0] - x)**2 + (bacterium.pos[1] - y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_bacterium = bacterium
                
        if closest_bacterium and min_dist < 5.0:
            self.highlighted_bacterium_id = closest_bacterium.unique_id
            print(f"Viewing bacterium {closest_bacterium.unique_id} ({closest_bacterium.bacterial_type})")
            self.on_click_callback(closest_bacterium.unique_id)

    def get_bacterial_colors(self, agents):
        """Get numerical colors for bacterial types"""
        return [self.color_map.get(a.bacterial_type, 0) for a in agents]

    def update_history_plots(self):
        """Update history line plots"""
        history = self.model.history
        if len(history['steps']) > 0:
            # Food plot
            self.line_food.set_data(history['steps'], history['total_food'])
            self.ax_food.set_xlim(0, max(10, max(history['steps'])))
            self.ax_food.set_ylim(0, max(10, max(history['total_food']) * 1.1))
            
            # Population plot
            self.line_pop.set_data(history['steps'], history['population'])
            self.ax_pop.set_xlim(0, max(10, max(history['steps'])))
            self.ax_pop.set_ylim(0, max(10, max(history['population']) * 1.1))
            
            # Energy plot
            self.line_energy.set_data(history['steps'], history['avg_energy'])
            self.ax_energy.set_xlim(0, max(10, max(history['steps'])))
            self.ax_energy.set_ylim(0, max(10, max(history['avg_energy']) * 1.1))

    def update_main_plot(self):
        """Update the main simulation plot"""
        agents = list(self.model.agent_set)
        positions = [a.pos for a in agents]
        colors = self.get_bacterial_colors(agents)
        
        # Update bacteria scatter plot
        if self.scat is None:
            self.scat = self.ax.scatter(
                [pos[0] for pos in positions],
                [pos[1] for pos in positions],
                c=colors,
                cmap="viridis",
                s=10,
                edgecolor="k",
                alpha=0.7,
            )
        else:
            if len(positions) > 0:
                self.scat.set_offsets(positions)
                self.scat.set_array(np.array(colors))
            else:
                self.scat.set_offsets(np.empty((0, 2)))
                self.scat.set_array(np.array([]))
        
        # Update field overlays
        if self.im_food is None:
            self.im_food = self.ax.imshow(
                self.model.food_field.T,
                extent=[0, self.model.width, 0, self.model.height],
                origin="lower",
                cmap="Greens",
                alpha=0.3,
            )
        else:
            self.im_food.set_data(self.model.food_field.T)
        
        if self.im_ab is None:
            self.im_ab = self.ax.imshow(
                self.model.antibiotic_field.T,
                extent=[0, self.model.width, 0, self.model.height],
                origin="lower",
                cmap="Reds",
                alpha=0.3,
            )
        else:
            self.im_ab.set_data(self.model.antibiotic_field.T)
        
        # Highlight selected bacterium
        self._update_highlight(agents)
        
        self.ax.set_title(f"Step: {self.model.step_count} Agents: {len(self.model.agent_set)}")
        self.ax.set_xlim(0, self.model.width)
        self.ax.set_ylim(0, self.model.height)

    def _update_highlight(self, agents):
        """Update highlighted bacterium visualization"""
        if self.highlighted_bacterium_id is not None:
            highlighted_bacterium = next(
                (b for b in agents if b.unique_id == self.highlighted_bacterium_id), None
            )
            if highlighted_bacterium:
                highlight_pos = [[highlighted_bacterium.pos[0], highlighted_bacterium.pos[1]]]
                if self.highlight_scat is None:
                    self.highlight_scat = self.ax.scatter(
                        highlight_pos[0][0],
                        highlight_pos[0][1],
                        c='yellow',
                        s=100,
                        edgecolor="black",
                        linewidths=2,
                        alpha=1.0,
                        marker='o',
                        zorder=10
                    )
                else:
                    self.highlight_scat.set_offsets(highlight_pos)
            else:
                # Bacterium died
                if self.highlight_scat is not None:
                    self.highlight_scat.remove()
                    self.highlight_scat = None
                self.highlighted_bacterium_id = None
        elif self.highlight_scat is not None:
            self.highlight_scat.remove()
            self.highlight_scat = None

    def clear_highlight(self):
        """Clear highlighted bacterium"""
        if self.highlight_scat is not None:
            self.highlight_scat.remove()
            self.highlight_scat = None
        self.highlighted_bacterium_id = None

    def set_highlighted_bacterium(self, bacterium_id):
        """Set the highlighted bacterium ID"""
        self.highlighted_bacterium_id = bacterium_id

    def draw(self):
        """Redraw the canvas"""
        try:
            self.update_history_plots()
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Plot update error: {e}")

    def show(self):
        """Show the matplotlib figure"""
        plt.show()
