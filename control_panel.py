"""
Tkinter control panel for bacteria simulation.
"""

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    tk = None

from config import (
    BACTERIAL_TYPES, DEFAULT_STEPS_PER_FRAME, 
    MIN_STEPS_PER_FRAME, MAX_STEPS_PER_FRAME,
    SLOW_MODE_FRAME_SKIP, PERFORMANCE_MODE
)


class ControlPanel:
    """Tkinter control panel for simulation controls and statistics."""
    
    def __init__(self, model, on_toggle_pause, on_reset, on_apply_antibiotic, 
                 on_speed_change, on_view_bacterium):
        """
        Initialize control panel.
        
        Args:
            model: The simulation model
            on_toggle_pause: Callback for pause/resume
            on_reset: Callback for reset
            on_apply_antibiotic: Callback for applying antibiotic
            on_speed_change: Callback for speed changes
            on_view_bacterium: Callback for viewing selected bacterium
        """
        self.model = model
        self.on_toggle_pause = on_toggle_pause
        self.on_reset = on_reset
        self.on_apply_antibiotic = on_apply_antibiotic
        self.on_speed_change = on_speed_change
        self.on_view_bacterium = on_view_bacterium
        
        self.last_bacteria_list_hash = None
        self.ui_ref = None  # Reference to UI for performance stats
        
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
    
    def set_ui_reference(self, ui):
        """Set reference to UI for accessing performance metrics"""
        self.ui_ref = ui

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
        self.pause_btn = ttk.Button(frame, text="Start", command=self.on_toggle_pause)
        self.pause_btn.grid(column=0, row=row, pady=2)
        ttk.Button(frame, text="Reset", command=self.on_reset).grid(column=1, row=row, pady=2)
        row += 1

        # Speed controls
        row = self._add_speed_controls(frame, row)

        # Separator
        ttk.Separator(frame, orient='horizontal').grid(column=0, row=row, columnspan=2, sticky='ew', pady=10)
        row += 1

        # Antibiotic controls
        row = self._add_antibiotic_controls(frame, row)



        # Performance mode toggle - initialize with config value
        self.perf_mode_var = tk.BooleanVar(value=PERFORMANCE_MODE)
        self.perf_mode_check = ttk.Checkbutton(
            frame, text="Performance Mode (Control Panel)", variable=self.perf_mode_var, command=self._toggle_performance_mode
        )
        self.perf_mode_check.grid(column=0, row=row, columnspan=2, pady=(0, 5))
        row += 1

        # Performance info label
        self.perf_info_label = ttk.Label(frame, text="(reduces control panel update frequency, not simulation)", 
                                         font=("TkDefaultFont", 7), foreground="gray")
        self.perf_info_label.grid(column=0, row=row, columnspan=2, pady=(0, 10))
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
        
        ttk.Button(speed_frame, text="<<", command=lambda: self.on_speed_change(-1), width=3).grid(column=0, row=0)
        ttk.Button(speed_frame, text=">>", command=lambda: self.on_speed_change(1), width=3).grid(column=1, row=0)
        ttk.Button(speed_frame, text="Reset Speed", command=lambda: self.on_speed_change(0)).grid(column=2, row=0, padx=(5,0))
        
        self.speed_label = ttk.Label(frame, text=f"Speed: {DEFAULT_STEPS_PER_FRAME} steps/frame")
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
        self.antibiotic_combo.bind('<<ComboboxSelected>>', self._change_antibiotic)
        row += 1

        ttk.Label(frame, text="Dose:").grid(column=0, row=row, sticky='w')
        self.dose_var = tk.DoubleVar(value=0.5)
        self.dose_entry = ttk.Entry(frame, textvariable=self.dose_var, width=8)
        self.dose_entry.grid(column=1, row=row, pady=2)
        row += 1

        ttk.Button(frame, text="Apply Antibiotic", command=self._apply_antibiotic_internal).grid(
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
        
        self.bacteria_listbox.bind('<Double-Button-1>', lambda e: self._view_selected_bacterium())
        
        # Stats
        self.tracking_stats_label = ttk.Label(frame, text="Tracked: 0 alive, 0 deceased", 
                                             font=("TkDefaultFont", 8))
        self.tracking_stats_label.grid(column=0, row=row, columnspan=2, pady=(2, 5))
        row += 1
        
        ttk.Button(frame, text="View Selected Bacterium", command=self._view_selected_bacterium).grid(
            column=0, row=row, columnspan=2, pady=(0, 5))

    def _build_right_controls(self, frame):
        """Build right column (stats display)"""
        ttk.Label(frame, text="Population Stats", font=("TkDefaultFont", 10, "bold")).grid(
            column=0, row=0, columnspan=2, pady=(0, 10), sticky='w')
        
        self.stats_frame = ttk.Frame(frame)
        self.stats_frame.grid(column=0, row=1, columnspan=2, sticky="nsew")
        self.stats_labels = {}

    def _change_antibiotic(self, event=None):
        """Change antibiotic type"""
        try:
            new_antibiotic = self.antibiotic_var.get()
            self.model.set_antibiotic_type(new_antibiotic)
        except Exception as e:
            print(f"Error changing antibiotic: {e}")

    def _apply_antibiotic_internal(self):
        """Internal handler for applying antibiotic"""
        try:
            val = float(self.dose_entry.get())
        except Exception:
            val = 0.0
        self.on_apply_antibiotic(val)
        self.latest_label.config(text=f"{val:.3f}")


    def _toggle_performance_mode(self):
        """Internal handler for performance mode toggle"""
        try:
            # The checkbox value gets updated AFTER this callback is called,
            # so we need to invert the current UI state
            if self.ui_ref is not None:
                current_ui_state = self.ui_ref.performance_mode
                new_val = not current_ui_state
                self.ui_ref.toggle_performance_mode(new_val)
            else:
                print("Warning: UI reference not set, cannot toggle performance mode")
        except Exception as e:
            print(f"Error toggling performance mode: {e}")

    def _view_selected_bacterium(self):
        """View selected bacterium plots"""
        try:
            selection = self.bacteria_listbox.curselection()
            
            if not selection or len(selection) == 0:
                print("No bacterium selected")
                return
                
            text = self.bacteria_listbox.get(selection[0])
            id_part = text.split("ID:")[1].strip()
            bacterium_id = int(id_part.split()[0])
            
            self.on_view_bacterium(bacterium_id)
            
        except Exception as e:
            print(f"Error viewing bacterium: {e}")

    def update_speed_display(self, speed):
        """Update speed label"""
        if self.root is not None:
            try:
                self.speed_label.config(text=f"Speed: {speed} steps/frame")
            except Exception:
                pass

    def update_stats_display(self, stats):
        """Update population statistics display"""
        if self.root is None:
            return
            
        try:
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

    def set_pause_button_text(self, text):
        """Set pause button text"""
        if self.root is not None:
            try:
                self.pause_btn.config(text=text)
            except Exception:
                pass

    def set_pause_button_state(self, state):
        """Set pause button state (normal or disabled)"""
        if self.root is not None:
            try:
                self.pause_btn.config(state=state)
            except Exception:
                pass

    def update(self):
        """Update Tkinter event loop"""
        if self.root is not None:
            try:
                self.root.update_idletasks()
                self.root.update()
            except Exception as e:
                print(f"Error pumping Tk events: {e}")
