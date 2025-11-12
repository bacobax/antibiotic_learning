"""
Matplotlib visualization for bacteria simulation - FIXED for PyQt5 integration.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from simulation.simulation_config import BACTERIAL_TYPES, ANTIBIOTIC_TYPES, ANIMATION_FPS


class SimulationVisualizer:
    """Matplotlib visualizer for bacteria simulation with proper PyQt5 integration."""

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
        self.type_colors = {"E.coli": "blue", "Staph": "red", "Pseudomonas": "green"}

        # Animation settings
        self.animation_fps = ANIMATION_FPS
        self.animation_interval = int(1000 / self.animation_fps)
        self.animation = None

        # Highlighting
        self.highlighted_bacterium_id = None

        # Setup visualization - use Figure() instead of plt.figure()
        # This prevents matplotlib from creating a window manager thread
        self._setup_plots()

        # Click handling - connect AFTER plots are setup
        try:
            self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        except Exception as e:
            print(f"Warning: Could not connect click handler: {e}")

    def _setup_plots(self):
        """Setup matplotlib figure and subplots using non-interactive backend"""
        # Use Figure() directly instead of plt.figure()
        # This prevents matplotlib from managing its own window
        self.fig = Figure(figsize=(20, 10))
        gs = self.fig.add_gridspec(5, 4, hspace=0.3, wspace=0.3)

        # LEFT SIDE: Main simulation view (50% width, full height)
        self.ax = self.fig.add_subplot(gs[:, 0:2])
        self.ax.set_title("Bacteria Simulation")

        # RIGHT TOP: Four plots in 2x2 grid (Food, Population, Energy, Antibiotics)
        # Food level plot
        self.ax_food = self.fig.add_subplot(gs[0, 2])
        self.ax_food.set_xlabel("Steps", fontsize=8)
        self.ax_food.set_ylabel("Total Food", fontsize=8)
        self.ax_food.tick_params(labelsize=7)
        self.ax_food.grid(True, alpha=0.3)
        (self.line_food,) = self.ax_food.plot(
            [], [], label="Food Level", color="green", linewidth=1.5
        )
        self.ax_food.legend(fontsize=7)
        self.ax_food.set_title("Food Level", fontsize=9)

        # Population plot
        self.ax_pop = self.fig.add_subplot(gs[0, 3])
        self.ax_pop.set_xlabel("Steps", fontsize=8)
        self.ax_pop.set_ylabel("Population", fontsize=8)
        self.ax_pop.tick_params(labelsize=7)
        self.ax_pop.grid(True, alpha=0.3)
        (self.line_pop,) = self.ax_pop.plot(
            [], [], label="Population", color="blue", linewidth=1.5
        )
        self.ax_pop.legend(fontsize=7)
        self.ax_pop.set_title("Total Population", fontsize=9)

        # Energy plot
        self.ax_energy = self.fig.add_subplot(gs[1, 2])
        self.ax_energy.set_xlabel("Steps", fontsize=8)
        self.ax_energy.set_ylabel("Energy", fontsize=8)
        self.ax_energy.tick_params(labelsize=7)
        self.ax_energy.grid(True, alpha=0.3)
        (self.line_energy_avg,) = self.ax_energy.plot(
            [], [], label="Avg Energy", color="red", linewidth=1.5
        )
        (self.line_energy_worst,) = self.ax_energy.plot(
            [], [], label="Worst 10", color="green", linewidth=1.5
        )
        (self.line_energy_top,) = self.ax_energy.plot(
            [], [], label="Top 10", color="blue", linewidth=1.5
        )
        self.ax_energy.legend(fontsize=6, loc="best")
        self.ax_energy.set_title("Average Energy", fontsize=9)

        # NEW: Antibiotic concentrations plot
        self.ax_antibiotics = self.fig.add_subplot(gs[1, 3])
        self.ax_antibiotics.set_xlabel("Steps", fontsize=8)
        self.ax_antibiotics.set_ylabel("Concentration", fontsize=8)
        self.ax_antibiotics.tick_params(labelsize=7)
        self.ax_antibiotics.grid(True, alpha=0.3)
        self.ax_antibiotics.set_title("Antibiotic Concentrations", fontsize=9)

        # Create line for each antibiotic type
        self.antibiotic_lines = {}
        for ab_type, ab_config in ANTIBIOTIC_TYPES.items():
            color = ab_config.get("color", "gray")
            (line,) = self.ax_antibiotics.plot(
                [], [], label=ab_type, color=color, linewidth=1.5, alpha=0.8
            )
            self.antibiotic_lines[ab_type] = line
        self.ax_antibiotics.legend(fontsize=6, loc="best")

        # RIGHT BOTTOM: Trait evolution plots per bacterial type (4 plots in 2x2 grid)
        self.ax_enzyme = self.fig.add_subplot(gs[2, 2])
        self.ax_efflux = self.fig.add_subplot(gs[2, 3])
        self.ax_membrane = self.fig.add_subplot(gs[3, 2])
        self.ax_repair = self.fig.add_subplot(gs[3, 3])

        # Store trait axes for easy access
        self.trait_axes = {
            "enzyme": self.ax_enzyme,
            "efflux": self.ax_efflux,
            "membrane": self.ax_membrane,
            "repair": self.ax_repair,
        }

        # Initialize trait plot lines for each bacterial type
        self.trait_lines = {}
        for trait, ax in self.trait_axes.items():
            ax.set_xlabel("Steps", fontsize=8)
            ax.set_ylabel(f"Avg {trait.capitalize()}", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{trait.capitalize()} Trait Evolution", fontsize=9)

            self.trait_lines[trait] = {}
            for btype in self.bacterial_type_names:
                color = self.type_colors.get(btype, "gray")
                (line,) = ax.plot(
                    [], [], label=btype, color=color, linewidth=1.5, alpha=0.8
                )
                self.trait_lines[trait][btype] = line

            ax.legend(fontsize=6, loc="best")

        # Initialize plot elements
        self.scat = None
        self.scat_persistors = None
        self.scat_hgt = None
        self.highlight_scat = None
        self.im_food = None
        self.im_ab = None
        self.im_qs = None  # Quorum sensing field overlay
        self.im_eps = None  # EPS biofilm field overlay
        self.biofilm_lines = []  # List to hold biofilm connection lines


    def _on_click(self, event):
        """Handle mouse clicks to select bacteria"""
        # If no callback is registered, don't process clicks
        if self.on_click_callback is None:
            return
            
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        # Find closest bacterium - with safe copy
        try:
            agents = list(self.model.agent_set)
        except Exception:
            return

        min_dist = float("inf")
        closest_bacterium = None

        for bacterium in agents:
            if bacterium.pos is None:
                continue
            try:
                dist = np.sqrt((bacterium.pos[0] - x) ** 2 + (bacterium.pos[1] - y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_bacterium = bacterium
            except Exception:
                # Skip bacteria with invalid positions
                continue

        if closest_bacterium and min_dist < 5.0:
            self.highlighted_bacterium_id = closest_bacterium.unique_id
            print(
                f"Viewing bacterium {closest_bacterium.unique_id} ({closest_bacterium.bacterial_type})"
            )
            # Call the callback (we already checked it's not None at the top)
            self.on_click_callback(closest_bacterium.unique_id)

    def get_bacterial_colors(self, agents):
        """Get numerical colors for bacterial types"""
        return [self.color_map.get(a.bacterial_type, 0) for a in agents]

    @staticmethod
    def _convex_hull(points):
        """Compute 2D convex hull of a set of points using Andrew's monotone chain.

        points: iterable of (x, y)
        Returns a list of hull vertices in counter-clockwise order (no duplicate endpoint).
        """
        pts = sorted(set(points))
        if len(pts) <= 1:
            return pts

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Concatenate lower and upper to get full hull (omitting last point of each)
        return lower[:-1] + upper[:-1]


    def update_history_plots(self):
        """Update history line plots"""
        try:
            history = self.model.history
            # Only update plots if there's actual data collected
            # At startup, history exists but steps array is empty - this is normal
            if len(history["steps"]) == 0:
                # No data yet - skip update but don't error
                return
            
            # Now we know there's data, proceed with updates
            # Food plot
            self.line_food.set_data(history["steps"], history["total_food"])
            self.ax_food.set_xlim(0, max(10, max(history["steps"])))
            self.ax_food.set_ylim(0, max(10, max(history["total_food"]) * 1.1))

            # Population plot
            self.line_pop.set_data(history["steps"], history["population"])
            self.ax_pop.set_xlim(0, max(10, max(history["steps"])))
            self.ax_pop.set_ylim(0, max(10, max(history["population"]) * 1.1))

            # Energy plot
            self.line_energy_avg.set_data(history["steps"], history["avg_energy"])
            self.line_energy_worst.set_data(
                history["steps"], history["avg_energy_worst"]
            )
            self.line_energy_top.set_data(history["steps"], history["avg_energy_top"])
            self.ax_energy.set_xlim(0, max(10, max(history["steps"])))
            self.ax_energy.set_ylim(0, max(10, max(history["avg_energy"]) * 1.1))

            # Antibiotic concentrations plot
            max_antibiotic_concentration = 0.01
            for ab_type in ANTIBIOTIC_TYPES.keys():
                ab_key = f"antibiotic_{ab_type}"
                if ab_key in history and len(history[ab_key]) > 0:
                    data = history[ab_key]
                    steps = history["steps"][: len(data)]
                    self.antibiotic_lines[ab_type].set_data(steps, data)
                    if len(data) > 0:
                        max_antibiotic_concentration = max(
                            max_antibiotic_concentration, max(data)
                        )
                else:
                    self.antibiotic_lines[ab_type].set_data([], [])

            self.ax_antibiotics.set_xlim(0, max(10, max(history["steps"])))
            self.ax_antibiotics.set_ylim(
                0, max(0.1, max_antibiotic_concentration * 1.1)
            )

            # Update trait evolution plots
            self._update_trait_plots()
        except Exception as e:
            print(f"Error updating history plots: {e}")

    def _update_trait_plots(self):
        """Update trait evolution plots for each bacterial type"""
        try:
            history = self.model.history

            if len(history["steps"]) > 0:
                # Update each trait plot
                for trait in ["enzyme", "efflux", "membrane", "repair"]:
                    ax = self.trait_axes[trait]
                    max_val = 0.01

                    for btype in self.bacterial_type_names:
                        # Get data from history
                        trait_key = f"{btype}_avg_{trait}"

                        if trait_key in history and len(history[trait_key]) > 0:
                            data = history[trait_key]
                            steps = history["steps"][: len(data)]

                            self.trait_lines[trait][btype].set_data(steps, data)

                            if len(data) > 0:
                                max_val = max(max_val, max(data) * 1.1)
                        else:
                            self.trait_lines[trait][btype].set_data([], [])

                    # Update axis limits
                    if len(history["steps"]) > 0:
                        ax.set_xlim(0, max(10, max(history["steps"])))
                        ax.set_ylim(0, max(1.0, max_val))
        except Exception as e:
            print(f"Error updating trait plots: {e}")

    def update_main_plot(self):
        """Update the main simulation plot - with comprehensive error handling"""
        try:
            agents = list(self.model.agent_set)
        except Exception:
            return

        try:
            # Separate active and persistor bacteria
            # Filter agents to separate persistors
            persistor_agents = [a for a in agents if a.is_persister and a.pos is not None]
            hgt_agents = [a for a in agents if a.has_hgt_gene and a.pos is not None]
            active_agents = [
                a
                for a in agents
                if (not a.is_persister and not a.has_hgt_gene and a.pos is not None)
            ]

            active_positions = [a.pos for a in active_agents]
            persistor_positions = [a.pos for a in persistor_agents]
            hgt_positions = [a.pos for a in hgt_agents]

            active_colors = self.get_bacterial_colors(active_agents)
            persistor_colors = self.get_bacterial_colors(persistor_agents)
            hgt_colors = self.get_bacterial_colors(hgt_agents)

            # Update scatter plots with error handling
            self._update_scatter_plot("active", self.scat, active_positions, active_colors)
            self._update_scatter_plot("hgt", self.scat_hgt, hgt_positions, hgt_colors)
            self._update_scatter_plot("persistor", self.scat_persistors, persistor_positions, persistor_colors)

            # Update field overlays
            self._update_field_overlays()

            # Update biofilm
            self._update_biofilm(agents)

            # Highlight selected bacterium
            self._update_highlight(agents)

            # Update title
            persistor_count = len(persistor_agents)
            hgt_gene_count = len(hgt_agents)
            self.ax.set_title(
                f"Step: {self.model.step_count} | Agents: {len(agents)} | Persistors: {persistor_count} | HGT Gene: {hgt_gene_count}",
                fontsize=11,
            )
            self.ax.set_xlim(0, self.model.width)
            self.ax.set_ylim(0, self.model.height)

        except Exception as e:
            print(f"Error in update_main_plot: {e}")
    
    def _update_biofilm(self, agents):
        # Draw biofilm connection lines
        # Remove old lines
        for line in self.biofilm_lines:
            line.remove()
        self.biofilm_lines = []
        
        # Group bacteria by biofilm_id
        self.biofilm_groups = {}
        for agent in agents:
            if hasattr(agent, 'biofilm_id') and agent.biofilm_id is not None and agent.pos is not None:
                if agent.biofilm_id not in self.biofilm_groups:
                    self.biofilm_groups[agent.biofilm_id] = []
                self.biofilm_groups[agent.biofilm_id].append(agent)
        

        for biofilm_id, members in self.biofilm_groups.items():
            n = len(members)
            if n <= 1:
                continue

            pts = [(m.pos[0], m.pos[1]) for m in members]

            if n == 2:
                # Just draw the single segment between the two members
                x_coords = [pts[0][0], pts[1][0]]
                y_coords = [pts[0][1], pts[1][1]]
                line, = self.ax.plot(
                    x_coords,
                    y_coords,
                    color="deepskyblue",
                    linewidth=1.0,
                    alpha=0.4,
                    zorder=1,
                )
                self.biofilm_lines.append(line)
            else:
                # Compute hull and draw polygon edges
                hull = self._convex_hull(pts)
                if len(hull) >= 2:
                    for i in range(len(hull)):
                        a = hull[i]
                        b = hull[(i + 1) % len(hull)]
                        x_coords = [a[0], b[0]]
                        y_coords = [a[1], b[1]]
                        line, = self.ax.plot(
                            x_coords,
                            y_coords,
                            color="deepskyblue",
                            linewidth=1.0,
                            alpha=0.4,
                            zorder=1,
                        )
                        self.biofilm_lines.append(line)


    def _update_scatter_plot(self, plot_type, scat, positions, colors):
        """Helper to safely update scatter plots"""
        try:
            if plot_type == "active":
                if self.scat is None:
                    self.scat = self.ax.scatter(
                        [pos[0] for pos in positions] if positions else [],
                        [pos[1] for pos in positions] if positions else [],
                        c=colors if colors else [],
                        cmap="viridis",
                        s=15,
                        marker="o",
                        edgecolor="k",
                        linewidths=0.5,
                        alpha=0.7,
                    )
                else:
                    if len(positions) > 0:
                        self.scat.set_offsets(positions)
                        self.scat.set_array(np.array(colors))
                    else:
                        self.scat.set_offsets(np.empty((0, 2)))
                        self.scat.set_array(np.array([]))

            elif plot_type == "hgt":
                if self.scat_hgt is None:
                    self.scat_hgt = self.ax.scatter(
                        [pos[0] for pos in positions] if positions else [],
                        [pos[1] for pos in positions] if positions else [],
                        c=colors if colors else [],
                        cmap="viridis",
                        s=50,
                        marker="*",
                        edgecolor="k",
                        alpha=0.7,
                    )
                else:
                    if len(positions) > 0:
                        self.scat_hgt.set_offsets(positions)
                        self.scat_hgt.set_array(np.array(colors))
                    else:
                        self.scat_hgt.set_offsets(np.empty((0, 2)))
                        self.scat_hgt.set_array(np.array([]))

            elif plot_type == "persistor":
                if self.scat_persistors is None:
                    self.scat_persistors = self.ax.scatter(
                        [pos[0] for pos in positions] if positions else [],
                        [pos[1] for pos in positions] if positions else [],
                        c=colors if colors else [],
                        cmap="viridis",
                        s=15,
                        edgecolor="purple",
                        linewidths=2.5,
                        alpha=0.7,
                        zorder=5,
                    )
                else:
                    if len(positions) > 0:
                        self.scat_persistors.set_offsets(positions)
                        self.scat_persistors.set_array(np.array(colors))
                    else:
                        self.scat_persistors.set_offsets(np.empty((0, 2)))
                        self.scat_persistors.set_array(np.array([]))
        except Exception as e:
            print(f"Error updating {plot_type} scatter plot: {e}")

    def _update_field_overlays(self):
        """Update food and antibiotic field overlays"""
        try:
        #     # Debug food field values every 100 steps
        #     if self.model.step_count % 100 == 0:
        #         food_max = np.max(self.model.food_field)
        #         food_min = np.min(self.model.food_field)
        #         food_mean = np.mean(self.model.food_field)
        #         print(f"[Food Field] Step {self.model.step_count}: min={food_min:.6f}, max={food_max:.6f}, mean={food_mean:.6f}")
            
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
        except Exception as e:
            print(f"Error updating food field: {e}")

        # Antibiotic field overlay
        try:
            bg_gray = np.array([0.92, 0.92, 0.92], dtype=float)

            if (
                hasattr(self.model, "antibiotic_fields")
                and len(self.model.antibiotic_fields) > 0
            ):
                h, w = self.model.food_field.shape
                fields = []
                ab_types = []

                for ab_type, ab_field in self.model.antibiotic_fields.items():
                    field = np.array(ab_field, dtype=float)
                    if field.size == 0:
                        continue
                    fields.append(field)
                    ab_types.append(ab_type)

                if len(fields) == 0:
                    rgb_img = np.tile(bg_gray[None, None, :], (h, w, 1))
                else:
                    stacked = np.stack(fields, axis=0)
                    total = np.sum(stacked, axis=0)
                    total_clipped = np.clip(total, 0.0, 1.0)

                    denom = total.copy()
                    denom[denom == 0] = 1.0
                    weights = stacked / denom[None, :, :]

                    base_color = np.zeros((h, w, 3), dtype=float)
                    for i, ab_type in enumerate(ab_types):
                        color = ANTIBIOTIC_TYPES.get(ab_type, {}).get("color", "gray")
                        try:
                            rgb = np.array(mcolors.to_rgb(color), dtype=float)
                        except Exception:
                            rgb = np.array(mcolors.to_rgb("gray"), dtype=float)
                        base_color += weights[i, :, :, None] * rgb[None, None, :]

                    zero_mask = total == 0
                    if zero_mask.any():
                        base_color[zero_mask, :] = bg_gray

                    intensity = total_clipped
                    color_strength = 0.85
                    rgb_img = (
                        bg_gray[None, None, :] * (1.0 - intensity[:, :, None] * color_strength)
                    ) + (base_color * (intensity[:, :, None] * color_strength))
                    rgb_img = np.clip(rgb_img, 0.0, 1.0)

                rgb_display = np.transpose(rgb_img, (1, 0, 2))

                if self.im_ab is None:
                    self.im_ab = self.ax.imshow(
                        rgb_display,
                        extent=[0, self.model.width, 0, self.model.height],
                        origin="lower",
                        alpha=0.6,
                        interpolation="bilinear",
                        zorder=2,
                    )
                else:
                    self.im_ab.set_data(rgb_display)
            else:
                if self.im_ab is None:
                    empty_rgb = np.tile(
                        bg_gray[None, None, :], (self.model.height, self.model.width, 1)
                    )
                    self.im_ab = self.ax.imshow(
                        np.transpose(empty_rgb, (1, 0, 2)),
                        extent=[0, self.model.width, 0, self.model.height],
                        origin="lower",
                        alpha=0.6,
                        interpolation="bilinear",
                        zorder=2,
                    )
                else:
                    empty_rgb = np.tile(
                        bg_gray[None, None, :], (self.model.height, self.model.width, 1)
                    )
                    self.im_ab.set_data(np.transpose(empty_rgb, (1, 0, 2)))
        except Exception as e:
            print(f"Error updating antibiotic field: {e}")

        # Quorum sensing field overlay - with enhanced visualization for low values
        try:
            if hasattr(self.model, "qs_signal_field"):
                # Scale QS field for better visualization
                # Since QS values are typically 0-0.2, we need to amplify them
                # to match the food field scale (0.2-0.8)
                qs_scaled = np.clip(self.model.qs_signal_field * 4.0, 0.0, 1.0)  # Amplify by 4x
                
                if self.im_qs is None:
                    self.im_qs = self.ax.imshow(
                        qs_scaled.T,
                        extent=[0, self.model.width, 0, self.model.height],
                        origin="lower",
                        cmap="Blues",
                        alpha=0.3,  # Increased alpha for visibility
                        vmin=0.0,   # Explicitly set range
                        vmax=1.0,   # after scaling
                    )
                else:
                    self.im_qs.set_data(qs_scaled.T)
        except Exception as e:
            print(f"Error updating QS field: {e}")
        
        # Update EPS biofilm field (orange visualization)
        try:
            if hasattr(self.model, "biofilm_manager") and hasattr(self.model.biofilm_manager, "eps_field"):
                eps_field = self.model.biofilm_manager.eps_field
                # Normalize EPS field for visualization (values typically 0-1)
                eps_normalized = np.clip(eps_field, 0.0, 1.0)
                
                if self.im_eps is None:
                    self.im_eps = self.ax.imshow(
                        eps_normalized.T,
                        extent=[0, self.model.width, 0, self.model.height],
                        origin="lower",
                        cmap="Oranges",  # Orange color palette
                        alpha=0.4,  # Semi-transparent overlay
                        vmin=0.0,
                        vmax=1.0,
                    )
                else:
                    self.im_eps.set_data(eps_normalized.T)
        except Exception as e:
            print(f"Error updating EPS field: {e}")

    def update_graphs(self):
        """Update only the history plots"""
        self.update_history_plots()

    def _update_highlight(self, agents):
        """Update highlighted bacterium visualization"""
        try:
            if self.highlighted_bacterium_id is not None:
                highlighted_bacterium = next(
                    (b for b in agents if b.unique_id == self.highlighted_bacterium_id),
                    None,
                )
                if highlighted_bacterium and highlighted_bacterium.pos is not None:
                    highlight_pos = [
                        [highlighted_bacterium.pos[0], highlighted_bacterium.pos[1]]
                    ]
                    marker = "*" if highlighted_bacterium.has_hgt_gene else "o"
                    marker_size = 250 if highlighted_bacterium.has_hgt_gene else 150

                    if self.highlight_scat is None:
                        self.highlight_scat = self.ax.scatter(
                            highlight_pos[0][0],
                            highlight_pos[0][1],
                            c="yellow",
                            s=marker_size,
                            edgecolor="black",
                            linewidths=2,
                            alpha=1.0,
                            marker=marker,
                            zorder=10,
                        )
                    else:
                        try:
                            self.highlight_scat.set_offsets(highlight_pos)
                            self.highlight_scat.set_sizes([marker_size])
                            self.highlight_scat.set_marker(marker)
                        except Exception:
                            try:
                                self.highlight_scat.remove()
                            except Exception:
                                pass
                            self.highlight_scat = self.ax.scatter(
                                highlight_pos[0][0],
                                highlight_pos[0][1],
                                c="yellow",
                                s=marker_size,
                                edgecolor="black",
                                linewidths=2,
                                alpha=1.0,
                                marker=marker,
                                zorder=10,
                            )
                else:
                    if self.highlight_scat is not None:
                        try:
                            self.highlight_scat.remove()
                        except Exception:
                            pass
                        self.highlight_scat = None
                    self.highlighted_bacterium_id = None
            elif self.highlight_scat is not None:
                try:
                    self.highlight_scat.remove()
                except Exception:
                    pass
                self.highlight_scat = None
        except Exception as e:
            print(f"Error updating highlight: {e}")

    def clear_highlight(self):
        """Clear highlighted bacterium"""
        if self.highlight_scat is not None:
            try:
                self.highlight_scat.remove()
            except Exception:
                pass
            self.highlight_scat = None
        self.highlighted_bacterium_id = None

    def set_highlighted_bacterium(self, bacterium_id):
        """Set the highlighted bacterium ID"""
        self.highlighted_bacterium_id = bacterium_id

    def draw(self):
        """Redraw the canvas"""
        try:
            self.update_history_plots()
            if self.fig and self.fig.canvas:
                try:
                    self.fig.canvas.draw_idle()
                except Exception:
                    # Canvas might not be available if window is closing
                    pass
        except Exception as e:
            print(f"Error in draw: {e}")

    def show(self):
        """Show the matplotlib figure - DOES NOT block when embedded in PyQt5"""
        # Don't call plt.show() - it blocks and causes segfaults
        # Instead, matplotlib will be shown through PyQt5 embedding
        pass
