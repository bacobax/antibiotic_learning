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
        
        # Colors for bacterial types in trait plots
        self.type_colors = {
            "E.coli": "blue",
            "Staph": "red",
            "Pseudomonas": "green"
        }
        
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
        # Larger figure for more content
        self.fig = plt.figure(figsize=(20, 10))
        gs = self.fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # LEFT SIDE: Main simulation view (50% width, full height)
        self.ax = self.fig.add_subplot(gs[:, 0:2])
        self.ax.set_title("Bacteria Simulation")

        # RIGHT TOP: Three existing plots (Food, Population, Energy)
        # Food level plot
        self.ax_food = self.fig.add_subplot(gs[0, 2])
        self.ax_food.set_xlabel('Steps', fontsize=8)
        self.ax_food.set_ylabel('Total Food', fontsize=8)
        self.ax_food.tick_params(labelsize=7)
        self.ax_food.grid(True, alpha=0.3)
        self.line_food, = self.ax_food.plot([], [], label='Food Level', color='green', linewidth=1.5)
        self.ax_food.legend(fontsize=7)
        self.ax_food.set_title('Food Level', fontsize=9)

        # Population plot
        self.ax_pop = self.fig.add_subplot(gs[0, 3])
        self.ax_pop.set_xlabel('Steps', fontsize=8)
        self.ax_pop.set_ylabel('Population', fontsize=8)
        self.ax_pop.tick_params(labelsize=7)
        self.ax_pop.grid(True, alpha=0.3)
        self.line_pop, = self.ax_pop.plot([], [], label='Population', color='blue', linewidth=1.5)
        self.ax_pop.legend(fontsize=7)
        self.ax_pop.set_title('Total Population', fontsize=9)

        # Energy plot
        self.ax_energy = self.fig.add_subplot(gs[1, 2:4])
        self.ax_energy.set_xlabel('Steps', fontsize=8)
        self.ax_energy.set_ylabel('Energy', fontsize=8)
        self.ax_energy.tick_params(labelsize=7)
        self.ax_energy.grid(True, alpha=0.3)
        self.line_energy_avg, = self.ax_energy.plot([], [], label='Avg Energy', color='red', linewidth=1.5)
        self.line_energy_worst, = self.ax_energy.plot([], [], label='Worst 10 Energy', color='green', linewidth=1.5)
        self.line_energy_top, = self.ax_energy.plot([], [], label='Top 10 Energy', color='blue', linewidth=1.5)
        self.ax_energy.legend(fontsize=7)
        self.ax_energy.set_title('Average Energy (Top 10)', fontsize=9)

        # RIGHT BOTTOM: Trait evolution plots per bacterial type
        # We'll create one plot per trait showing all bacterial types
        self.ax_enzyme = self.fig.add_subplot(gs[2, 2])
        self.ax_efflux = self.fig.add_subplot(gs[2, 3])
        self.ax_membrane = self.fig.add_subplot(gs[3, 2])
        self.ax_repair = self.fig.add_subplot(gs[3, 3])
        
        # Store trait axes for easy access
        self.trait_axes = {
            'enzyme': self.ax_enzyme,
            'efflux': self.ax_efflux,
            'membrane': self.ax_membrane,
            'repair': self.ax_repair
        }
        
        # Initialize trait plot lines for each bacterial type
        self.trait_lines = {}
        for trait, ax in self.trait_axes.items():
            ax.set_xlabel('Steps', fontsize=8)
            ax.set_ylabel(f'Avg {trait.capitalize()}', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{trait.capitalize()} Trait Evolution', fontsize=9)
            
            self.trait_lines[trait] = {}
            for btype in self.bacterial_type_names:
                color = self.type_colors.get(btype, 'gray')
                line, = ax.plot([], [], label=btype, color=color, linewidth=1.5, alpha=0.8)
                self.trait_lines[trait][btype] = line
            
            ax.legend(fontsize=6, loc='best')

        # Initialize plot elements
        self.scat = None
        self.scat_persistors = None  # Separate scatter for persistor bacteria
        # Initialize plot elements - now we need separate scatter plots for circles and stars
        self.scat_hgt = None
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
            self.line_energy_avg.set_data(history['steps'], history['avg_energy'])
            self.line_energy_worst.set_data(history['steps'], history['avg_energy_worst'])
            self.line_energy_top.set_data(history['steps'], history['avg_energy_top'])
            self.ax_energy.set_xlim(0, max(10, max(history['steps'])))
            self.ax_energy.set_ylim(0, max(10, max(history['avg_energy']) * 1.1))
            
        # Update trait evolution plots
        self._update_trait_plots()

    def _update_trait_plots(self):
        """Update trait evolution plots for each bacterial type"""
        history = self.model.history
        
        if len(history['steps']) > 0:
            # Update each trait plot
            for trait in ['enzyme', 'efflux', 'membrane', 'repair']:
                ax = self.trait_axes[trait]
                max_val = 0.01
                
                for btype in self.bacterial_type_names:
                    # Get data from history
                    trait_key = f'{btype}_avg_{trait}'
                    
                    if trait_key in history and len(history[trait_key]) > 0:
                        data = history[trait_key]
                        steps = history['steps'][:len(data)]
                        
                        self.trait_lines[trait][btype].set_data(steps, data)
                        
                        if len(data) > 0:
                            max_val = max(max_val, max(data) * 1.1)
                    else:
                        self.trait_lines[trait][btype].set_data([], [])
                
                # Update axis limits
                if len(history['steps']) > 0:
                    ax.set_xlim(0, max(10, max(history['steps'])))
                    ax.set_ylim(0, max(1.0, max_val))

    def update_main_plot(self):
        """Update the main simulation plot"""
        agents = list(self.model.agent_set)
        
        # Separate active and persistor bacteria
        
        persistor_agents = [a for a in agents if a.is_persistor]
        hgt_agents = [a for a in agents if a.has_hgt_gene]
        # rest of the agents
        active_agents = [a for a in agents if (not a.is_persistor and not a.has_hgt_gene)]
        
        active_positions = [a.pos for a in active_agents]
        persistor_positions = [a.pos for a in persistor_agents]
        hgt_positions = [a.pos for a in hgt_agents]
        
        
        active_colors = self.get_bacterial_colors(active_agents)
        persistor_colors = self.get_bacterial_colors(persistor_agents)
        hgt_colors = self.get_bacterial_colors(hgt_agents)
        
        # Update active bacteria scatter plot (normal border)
        if self.scat is None:
            self.scat = self.ax.scatter(
                [pos[0] for pos in active_positions] if active_positions else [],
                [pos[1] for pos in active_positions] if active_positions else [],
                c=active_colors if active_colors else [],
                cmap="viridis",
                s=15,
                marker='o',
                edgecolor="k",
                linewidths=0.5,
                alpha=0.7,
            )
        else:
            if len(active_positions) > 0:
                self.scat.set_offsets(active_positions)
                self.scat.set_array(np.array(active_colors))
            else:
                self.scat.set_offsets(np.empty((0, 2)))
                self.scat.set_array(np.array([]))
        
        # Update bacteria scatter plot for stars (HGT gene)
        if self.scat_hgt is None:
            self.scat_hgt = self.ax.scatter(
                [pos[0] for pos in hgt_positions] if hgt_positions else [],
                [pos[1] for pos in hgt_positions] if hgt_positions else [],
                c=hgt_colors if hgt_colors else [],
                cmap="viridis",
                s=50,  # Slightly larger for visibility
                marker='*',
                edgecolor="k",
                alpha=0.7,
            )
        else:
            if len(hgt_positions) > 0:
                self.scat_hgt.set_offsets(hgt_positions)
                self.scat_hgt.set_array(np.array(hgt_colors))
            else:
                self.scat_hgt.set_offsets(np.empty((0, 2)))
                self.scat_hgt.set_array(np.array([]))
        
        # Update persistor bacteria scatter plot (thicker border)
        if self.scat_persistors is None:
            self.scat_persistors = self.ax.scatter(
                [pos[0] for pos in persistor_positions],
                [pos[1] for pos in persistor_positions],
                c=persistor_colors,
                cmap="viridis",
                s=15,
                edgecolor="purple",  # Distinctive color for persistors
                linewidths=2.5,      # Thicker border for persistors
                alpha=0.7,
                zorder=5  # Draw on top of active bacteria
            )
        else:
            if len(persistor_positions) > 0:
                self.scat_persistors.set_offsets(persistor_positions)
                self.scat_persistors.set_array(np.array(persistor_colors))
            else:
                self.scat_persistors.set_offsets(np.empty((0, 2)))
                self.scat_persistors.set_array(np.array([]))
        
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
        
        # Update title with persistor count
        persistor_count = len(persistor_agents)
        self.ax.set_title(
            f"Step: {self.model.step_count} | Agents: {len(agents)} (Persistors: {persistor_count})", 
            fontsize=11
        )
        self.ax.set_xlim(0, self.model.width)
        self.ax.set_ylim(0, self.model.height)

    def update_graphs(self):
        """Update only the history plots (separate from main plot for performance mode)"""
        self.update_history_plots()

    def _update_highlight(self, agents):
        """Update highlighted bacterium visualization"""
        if self.highlighted_bacterium_id is not None:
            highlighted_bacterium = next(
                (b for b in agents if b.unique_id == self.highlighted_bacterium_id), None
            )
            if highlighted_bacterium:
                highlight_pos = [[highlighted_bacterium.pos[0], highlighted_bacterium.pos[1]]]
                # Use star marker if bacterium has HGT gene, circle otherwise
                marker = '*' if highlighted_bacterium.has_hgt_gene else 'o'
                marker_size = 250 if highlighted_bacterium.has_hgt_gene else 150
                
                if self.highlight_scat is None:
                    self.highlight_scat = self.ax.scatter(
                        highlight_pos[0][0],
                        highlight_pos[0][1],
                        c='yellow',
                        s=marker_size,
                        edgecolor="black",
                        linewidths=2,
                        alpha=1.0,
                        marker=marker,
                        zorder=10
                    )
                else:
                    # Need to remove and recreate if marker changes
                    self.highlight_scat.remove()
                    self.highlight_scat = self.ax.scatter(
                        highlight_pos[0][0],
                        highlight_pos[0][1],
                        c='yellow',
                        s=marker_size,
                        edgecolor="black",
                        linewidths=2,
                        alpha=1.0,
                        marker=marker,
                        zorder=10
                    )
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