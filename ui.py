"""
Simulation UI and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    tk = None

from config import (
    BACTERIAL_TYPES, DEFAULT_STEPS_PER_SECOND, 
    MIN_STEPS_PER_SECOND, MAX_STEPS_PER_SECOND, ANIMATION_FPS
)
from tracking import IndividualPlotter


class SimulatorUI:
    """Main UI for bacteria simulation visualization and control."""
    
    def __init__(self, model):
        self.model = model
        self.paused = True
        self.simulation_started = False  # Track if simulation has been started
        self.population_extinct = False  # Track if population is extinct
        self.latest_dose = 0.0
        
        # Color mapping for bacterial types
        self.bacterial_type_names = list(BACTERIAL_TYPES.keys())
        self.color_map = {name: i for i, name in enumerate(self.bacterial_type_names)}
        
        # Speed control
        self.steps_per_second = DEFAULT_STEPS_PER_SECOND
        self.animation_fps = ANIMATION_FPS
        self.animation_interval = int(1000 / self.animation_fps)
        self.steps_accumulator = 0.0
        self.animation = None
        
        # Tracking state
        self.last_bacteria_list_hash = None
        self.highlighted_bacterium_id = None
        
        # Setup visualization
        self._setup_plots()
        
        # Individual tracking
        self.individual_plotter = IndividualPlotter(self.model.individual_tracker)
        
        # Click handling
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Start Tk UI if available
        if tk is not None:
            try:
                self.root = tk.Tk()
                self.root.title("Control Panel")
                self.build_controls()
            except Exception as e:
                print(f"Tk UI init failed: {e}")
                self.root = None
        else:
            self.root = None

    def _setup_plots(self):
        """Setup matplotlib figure and subplots"""
        self.fig = plt.figure(figsize=(16, 4))
        gs = self.fig.add_gridspec(1, 4)
        
        # Main simulation view
        self.ax = self.fig.add_subplot(gs[0])
        
        # Food level plot
        self.ax_food = self.fig.add_subplot(gs[1])
        self.ax_food.set_xlabel('Steps')
        self.ax_food.set_ylabel('Total Food')
        self.ax_food.grid(True)
        self.line_food, = self.ax_food.plot([], [], label='Food Level', color='green')
        self.ax_food.legend()
        
        # Population plot
        self.ax_pop = self.fig.add_subplot(gs[2])
        self.ax_pop.set_xlabel('Steps')
        self.ax_pop.set_ylabel('Population')
        self.ax_pop.grid(True)
        self.line_pop, = self.ax_pop.plot([], [], label='Population', color='blue')
        self.ax_pop.legend()
        
        # Energy plot
        self.ax_energy = self.fig.add_subplot(gs[3])
        self.ax_energy.set_xlabel('Steps')
        self.ax_energy.set_ylabel('Energy (Top 10 Avg)')
        self.ax_energy.grid(True)
        self.line_energy, = self.ax_energy.plot([], [], label='Top 10 Energy', color='red')
        self.ax_energy.legend()
        
        self.fig.tight_layout()
        
        self.scat = None
        self.highlight_scat = None
        self.im_food = None
        self.im_ab = None

    def on_click(self, event):
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
            self.individual_plotter.update_plots(closest_bacterium.unique_id)

    def get_bacterial_colors(self, agents):
        """Get numerical colors for bacterial types"""
        return [self.color_map.get(a.bacterial_type, 0) for a in agents]

    def build_controls(self):
        """Build Tkinter control panel"""
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Left and right columns
        left_frame = ttk.Frame(main_frame, padding=(0, 0, 10, 0))
        left_frame.grid(row=0, column=0, sticky="nsew")
        
        ttk.Separator(main_frame, orient='vertical').grid(row=0, column=1, sticky='ns', padx=5)
        
        right_frame = ttk.Frame(main_frame, padding=(10, 0, 0, 0))
        right_frame.grid(row=0, column=2, sticky="nsew")
        
        self._build_left_controls(left_frame)
        self._build_right_controls(right_frame)

    def _build_left_controls(self, frame):
        """Build left column controls"""
        row = 0
        
        # Title
        ttk.Label(frame, text="Simulation Controls", font=("TkDefaultFont", 10, "bold")).grid(
            column=0, row=row, columnspan=2, pady=(0, 10))
        row += 1

        # Basic controls
        self.pause_btn = ttk.Button(frame, text="Start", command=self.toggle_pause)
        self.pause_btn.grid(column=0, row=row, pady=2)
        ttk.Button(frame, text="Reset", command=self.reset_sim).grid(column=1, row=row, pady=2)
        row += 1

        # Speed controls
        row = self._add_speed_controls(frame, row)

        # Separator
        ttk.Separator(frame, orient='horizontal').grid(column=0, row=row, columnspan=2, sticky='ew', pady=10)
        row += 1

        # Antibiotic controls
        row = self._add_antibiotic_controls(frame, row)

        # HGT toggle
        self.hgt_var = tk.BooleanVar(value=self.model.enable_hgt)
        self.hgt_check = ttk.Checkbutton(
            frame, text="Enable HGT", variable=self.hgt_var, command=self.toggle_hgt
        )
        self.hgt_check.grid(column=0, row=row, columnspan=2, pady=(10, 5))
        row += 1

        # Separator
        ttk.Separator(frame, orient='horizontal').grid(column=0, row=row, columnspan=2, sticky='ew', pady=10)
        row += 1

        # Individual tracking
        self._add_tracking_controls(frame, row)

    def _add_speed_controls(self, frame, row):
        """Add speed control widgets"""
        ttk.Label(frame, text="Speed Control", font=("TkDefaultFont", 9, "bold")).grid(
            column=0, row=row, columnspan=2, pady=(15, 5))
        row += 1
        
        speed_frame = ttk.Frame(frame)
        speed_frame.grid(column=0, row=row, columnspan=2, pady=5)
        row += 1
        
        ttk.Button(speed_frame, text="<<", command=self.speed_slower, width=3).grid(column=0, row=0)
        ttk.Button(speed_frame, text=">>", command=self.speed_faster, width=3).grid(column=1, row=0)
        ttk.Button(speed_frame, text="Reset Speed", command=self.speed_reset).grid(column=2, row=0, padx=(5,0))
        
        self.speed_label = ttk.Label(frame, text=f"Speed: {self.steps_per_second} steps/sec")
        self.speed_label.grid(column=0, row=row, columnspan=2, pady=(0, 5))
        row += 1
        
        return row

    def _add_antibiotic_controls(self, frame, row):
        """Add antibiotic control widgets"""
        ttk.Label(frame, text="Antibiotic Control", font=("TkDefaultFont", 9, "bold")).grid(
            column=0, row=row, columnspan=2, pady=(0, 5))
        row += 1
        
        ttk.Label(frame, text="Type:").grid(column=0, row=row, sticky='w', padx=(0, 5))
        self.antibiotic_var = tk.StringVar(value=self.model.current_antibiotic)
        self.antibiotic_combo = ttk.Combobox(frame, textvariable=self.antibiotic_var, 
                                           values=self.model.available_antibiotics, 
                                           state="readonly", width=12)
        self.antibiotic_combo.grid(column=1, row=row, pady=2)
        self.antibiotic_combo.bind('<<ComboboxSelected>>', self.change_antibiotic)
        row += 1

        ttk.Label(frame, text="Dose:").grid(column=0, row=row, sticky='w')
        self.dose_var = tk.DoubleVar(value=0.5)
        self.dose_entry = ttk.Entry(frame, textvariable=self.dose_var, width=8)
        self.dose_entry.grid(column=1, row=row, pady=2)
        row += 1

        ttk.Button(frame, text="Apply Antibiotic", command=self.apply_antibiotic_ui).grid(
            column=0, row=row, columnspan=2, pady=5)
        row += 1

        ttk.Label(frame, text="Latest dose:").grid(column=0, row=row, sticky='w')
        self.latest_label = ttk.Label(frame, text="0.0")
        self.latest_label.grid(column=1, row=row, sticky='w')
        row += 1
        
        return row

    def _add_tracking_controls(self, frame, row):
        """Add individual tracking controls"""
        ttk.Label(frame, text="Browse Bacteria", font=("TkDefaultFont", 9, "bold")).grid(
            column=0, row=row, columnspan=2, pady=(0, 5))
        row += 1
        
        # Filter options
        ttk.Label(frame, text="Show:").grid(column=0, row=row, sticky='w')
        self.filter_var = tk.StringVar(value="alive")
        
        self.filter_combo = ttk.Combobox(frame, textvariable=self.filter_var, 
                                    values=["alive", "deceased", "all"], 
                                    state="readonly", width=12)
        self.filter_combo.grid(column=1, row=row, pady=2)
        
        def on_filter_change(event):
            selected_value = self.filter_combo.get()
            self.update_bacteria_list(filter_type=selected_value, force_update=True)
        
        self.filter_combo.bind('<<ComboboxSelected>>', on_filter_change)
        self.filter_combo.current(0)
        row += 1
        
        # Bacteria list
        list_frame = ttk.Frame(frame)
        list_frame.grid(column=0, row=row, columnspan=2, pady=5, sticky="ew")
        row += 1
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.bacteria_listbox = tk.Listbox(list_frame, height=8, width=25, 
                                          yscrollcommand=scrollbar.set,
                                          selectmode=tk.SINGLE,
                                          exportselection=False)
        self.bacteria_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.bacteria_listbox.yview)
        
        self.bacteria_listbox.bind('<Double-Button-1>', lambda e: self.view_selected_bacterium())
        
        # Stats
        self.tracking_stats_label = ttk.Label(frame, text="Tracked: 0 alive, 0 deceased", 
                                             font=("TkDefaultFont", 8))
        self.tracking_stats_label.grid(column=0, row=row, columnspan=2, pady=(2, 5))
        row += 1
        
        ttk.Button(frame, text="View Selected Bacterium", command=self.view_selected_bacterium).grid(
            column=0, row=row, columnspan=2, pady=(0, 5))

    def _build_right_controls(self, frame):
        """Build right column (stats display)"""
        ttk.Label(frame, text="Population Stats", font=("TkDefaultFont", 10, "bold")).grid(
            column=0, row=0, columnspan=2, pady=(0, 10), sticky='w')
        
        self.stats_frame = ttk.Frame(frame)
        self.stats_frame.grid(column=0, row=1, columnspan=2, sticky="nsew")
        self.stats_labels = {}

    def change_antibiotic(self, event=None):
        """Change antibiotic type"""
        try:
            new_antibiotic = self.antibiotic_var.get()
            self.model.set_antibiotic_type(new_antibiotic)
        except Exception as e:
            print(f"Error changing antibiotic: {e}")

    def update_stats_display(self):
        """Update population statistics display"""
        if self.root is None:
            return
            
        try:
            stats = self.model.get_population_stats()
            
            if not hasattr(self, '_stats_initialized'):
                self._create_stats_labels()
                self._stats_initialized = True
            
            self._update_stats_values(stats)
                    
        except Exception as e:
            print(f"Error updating stats: {e}")

    def _create_stats_labels(self):
        """Create all stats labels once"""
        self.stats_labels = {}
        row = 0
        
        # Total
        self.stats_labels["total"] = ttk.Label(self.stats_frame, text="Total: 0")
        self.stats_labels["total"].grid(column=0, row=row, columnspan=2, sticky="w")
        row += 1
        
        # Per-type stats
        for btype in BACTERIAL_TYPES.keys():
            self.stats_labels[f"type_{btype}"] = ttk.Label(self.stats_frame, text=f"{btype}: 0", 
                                                         font=("TkDefaultFont", 8, "bold"))
            self.stats_labels[f"type_{btype}"].grid(column=0, row=row, columnspan=2, sticky="w")
            row += 1
            
            for trait in ["enzyme", "efflux", "membrane", "repair", "age"]:
                self.stats_labels[f"{btype}_{trait}"] = ttk.Label(
                    self.stats_frame, 
                    text=f"  {trait}: 0.000" if trait != "age" else "  age: 0.0"
                )
                self.stats_labels[f"{btype}_{trait}"].grid(column=0, row=row, columnspan=2, 
                                                          sticky="w", padx=(10,0))
                row += 1
        
        # Overall averages
        ttk.Label(self.stats_frame, text="Overall Avg:", 
                 font=("TkDefaultFont", 8, "bold")).grid(column=0, row=row, columnspan=2, sticky="w")
        row += 1
        
        for trait in ["enzyme", "efflux", "membrane", "repair"]:
            self.stats_labels[f"avg_{trait}"] = ttk.Label(self.stats_frame, text=f"  {trait}: 0.000")
            self.stats_labels[f"avg_{trait}"].grid(column=0, row=row, columnspan=2, sticky="w")
            row += 1
        
        self.stats_labels["avg_age"] = ttk.Label(self.stats_frame, text="  age: 0.0")
        self.stats_labels["avg_age"].grid(column=0, row=row, columnspan=2, sticky="w")

    def _update_stats_values(self, stats):
        """Update existing stats labels with new values"""
        self.stats_labels["total"].config(text=f"Total: {stats['total']}")
        
        for btype in BACTERIAL_TYPES.keys():
            count = stats["by_type"].get(btype, 0)
            self.stats_labels[f"type_{btype}"].config(text=f"{btype}: {count}")
            
            if count > 0 and btype in stats:
                type_stats = stats[btype]
                self.stats_labels[f"{btype}_enzyme"].config(text=f"  enzyme: {type_stats['enzyme']:.3f}")
                self.stats_labels[f"{btype}_efflux"].config(text=f"  efflux: {type_stats['efflux']:.3f}")
                self.stats_labels[f"{btype}_membrane"].config(text=f"  membrane: {type_stats['membrane']:.3f}")
                self.stats_labels[f"{btype}_repair"].config(text=f"  repair: {type_stats['repair']:.3f}")
                self.stats_labels[f"{btype}_age"].config(text=f"  age: {type_stats['age']:.1f}")
            else:
                self.stats_labels[f"{btype}_enzyme"].config(text="  enzyme: 0.000")
                self.stats_labels[f"{btype}_efflux"].config(text="  efflux: 0.000")
                self.stats_labels[f"{btype}_membrane"].config(text="  membrane: 0.000")
                self.stats_labels[f"{btype}_repair"].config(text="  repair: 0.000")
                self.stats_labels[f"{btype}_age"].config(text="  age: 0.0")
        
        if stats["total"] > 0:
            for trait, value in stats["avg_traits"].items():
                if trait != "age":
                    self.stats_labels[f"avg_{trait}"].config(text=f"  {trait}: {value:.3f}")
            self.stats_labels["avg_age"].config(text=f"  age: {stats['avg_traits']['age']:.1f}")
        else:
            for trait in ["enzyme", "efflux", "membrane", "repair"]:
                self.stats_labels[f"avg_{trait}"].config(text=f"  {trait}: 0.000")
            self.stats_labels["avg_age"].config(text="  age: 0.0")

    def toggle_pause(self):
        """Toggle simulation pause/start"""
        if not self.simulation_started:
            # First start
            self.simulation_started = True
            self.paused = False
            self.pause_btn.config(text="Pause")
            print("Simulation started")
        elif self.population_extinct:
            # Can't resume if extinct
            return
        else:
            # Normal pause/resume
            self.paused = not self.paused
            if self.paused:
                self.pause_btn.config(text="Resume")
            else:
                self.pause_btn.config(text="Pause")

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
        self.highlighted_bacterium_id = None
        self.last_bacteria_list_hash = None
        
        # Update button states
        if self.root is not None:
            self.pause_btn.config(text="Start", state="normal")
            self.latest_label.config(text="0.0")
        
        # Clear highlight
        if self.highlight_scat is not None:
            self.highlight_scat.remove()
            self.highlight_scat = None
        
        # Reset individual plotter
        self.individual_plotter = IndividualPlotter(self.model.individual_tracker)
        
        # Force UI update
        self.update_plot()
        self.update_stats_display()
        self.update_bacteria_list(force_update=True)
        
        print("Simulation reset - press Start to begin")

    def apply_antibiotic_ui(self):
        """Apply antibiotic from UI"""
        try:
            val = float(self.dose_entry.get())
        except Exception:
            val = 0.0
        self.model.apply_antibiotic(val)
        self.latest_dose = val
        if self.root is not None:
            try:
                self.latest_label.config(text=f"{val:.3f}")
            except Exception:
                pass

    def toggle_hgt(self):
        """Toggle horizontal gene transfer"""
        try:
            new_val = bool(self.hgt_var.get())
        except Exception:
            new_val = not self.model.enable_hgt
        self.model.enable_hgt = new_val

    def speed_faster(self):
        """Increase simulation speed"""
        self.steps_per_second = min(MAX_STEPS_PER_SECOND, self.steps_per_second + 1)
        self.update_speed_display()

    def speed_slower(self):
        """Decrease simulation speed"""
        self.steps_per_second = max(MIN_STEPS_PER_SECOND, self.steps_per_second - 1)
        self.update_speed_display()

    def speed_reset(self):
        """Reset speed to default"""
        self.steps_per_second = DEFAULT_STEPS_PER_SECOND
        self.update_speed_display()

    def update_speed_display(self):
        """Update speed label"""
        if self.root is not None:
            try:
                self.speed_label.config(text=f"Speed: {self.steps_per_second} steps/sec")
            except Exception:
                pass

    def update_bacteria_list(self, filter_type=None, force_update=False):
        """Update bacteria listbox based on filter"""
        if self.root is None:
            return
            
        try:
            if filter_type is None:
                filter_type = self.filter_combo.get() if hasattr(self, 'filter_combo') else "alive"
            
            tracker = self.model.individual_tracker
            
            # Get IDs based on filter
            if filter_type == "alive":
                ids = tracker.get_alive_individuals()
            elif filter_type == "deceased":
                ids = tracker.get_deceased_individuals()
            else:
                ids = tracker.get_all_tracked_ids()
            
            # Check if list changed
            current_hash = (filter_type, tuple(sorted(ids)))
            if not force_update and current_hash == self.last_bacteria_list_hash:
                return
                
            self.last_bacteria_list_hash = current_hash
            
            # Save selection
            old_selection = self.bacteria_listbox.curselection()
            old_selected_id = None
            if old_selection:
                try:
                    text = self.bacteria_listbox.get(old_selection[0])
                    id_part = text.split("ID:")[1].strip()
                    old_selected_id = int(id_part.split()[0])
                except:
                    pass
            
            self.bacteria_listbox.delete(0, tk.END)
            ids.sort()
            
            # Add to listbox
            new_selection_index = None
            for i, bacterium_id in enumerate(ids):
                data = tracker.get_tracked_data(bacterium_id)
                if data:
                    btype = data['bacterial_type']
                    text = f"ID:{bacterium_id:3d} {btype}"
                    self.bacteria_listbox.insert(tk.END, text)
                    
                    if bacterium_id == old_selected_id:
                        new_selection_index = i
            
            # Restore selection
            if new_selection_index is not None:
                self.bacteria_listbox.selection_set(new_selection_index)
                self.bacteria_listbox.see(new_selection_index)
            
            # Update stats
            stats = tracker.get_statistics()
            self.tracking_stats_label.config(
                text=f"Tracked: {stats['alive']} alive, {stats['deceased']} deceased (total: {stats['total_tracked']})"
            )
                    
        except Exception as e:
            print(f"Error updating bacteria list: {e}")

    def view_selected_bacterium(self):
        """View selected bacterium plots"""
        try:
            selection = self.bacteria_listbox.curselection()
            
            if not selection or len(selection) == 0:
                print("No bacterium selected")
                return
                
            text = self.bacteria_listbox.get(selection[0])
            id_part = text.split("ID:")[1].strip()
            bacterium_id = int(id_part.split()[0])
            
            self.highlighted_bacterium_id = bacterium_id
            self.individual_plotter.update_plots(bacterium_id)
            
        except Exception as e:
            print(f"Error viewing bacterium: {e}")

    def pump_tk(self):
        """Pump Tk events and update plots"""
        if self.root is not None:
            try:
                self.root.update_idletasks()
                self.root.update()
            except Exception as e:
                print(f"Error pumping Tk events: {e}")
        
        try:
            self._update_history_plots()
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Plot update error: {e}")

    def _update_history_plots(self):
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

    def run(self):
        """Run the simulation"""
        self.paused = True
        self.animation = animation.FuncAnimation(
            self.fig, self.update, interval=self.animation_interval, 
            blit=False, cache_frame_data=False
        )
        plt.show()

    def update(self, frame):
        """Animation update callback"""
        # Check for population extinction
        population = len(self.model.agent_set)
        if population == 0 and self.simulation_started and not self.population_extinct:
            self.population_extinct = True
            self.paused = True
            if self.root is not None:
                self.pause_btn.config(text="Extinct", state="disabled")
            print("Population extinct! Simulation paused. Press Reset to restart.")
        
        if not self.paused:
            self.steps_accumulator += self.steps_per_second / self.animation_fps
            while self.steps_accumulator >= 1.0:
                self.model.step()
                self.steps_accumulator -= 1.0

        self.update_plot()
        self.update_stats_display()
        self.update_bacteria_list()
        
        if self.individual_plotter.current_id is not None:
            self.individual_plotter.update_plots(self.individual_plotter.current_id)
        
        self.pump_tk()

    def update_plot(self):
        """Update the main simulation plot"""
        agents = list(self.model.agent_set)
        positions = [a.pos for a in agents]
        colors = self.get_bacterial_colors(agents)
        
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
        self.fig.canvas.draw_idle()

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
