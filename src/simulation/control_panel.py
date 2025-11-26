"""
PyQt5 control panel for bacteria simulation controls and statistics.
"""

try:
    from PyQt5 import QtWidgets, QtCore
except Exception:  # pragma: no cover - optional dependency fallback
    QtWidgets = None
    QtCore = None

from simulation.simulation_config import (
    BACTERIAL_TYPES,
    DEFAULT_STEPS_PER_FRAME,
    MIN_STEPS_PER_FRAME,
    MAX_STEPS_PER_FRAME,
    PERFORMANCE_MODE,
)


class ControlPanel:
    """PyQt5 control panel for simulation controls and statistics."""

    def __init__(
        self,
        model,
        on_toggle_pause,
        on_reset,
        on_apply_antibiotic,
        on_speed_change,
        on_view_bacterium,
        on_layer_visibility_change=None,
    ):
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
        self.on_layer_visibility_change = on_layer_visibility_change  # callback for layer toggles

        self.last_bacteria_list_hash = None
        self.ui_ref = None  # Reference to UI for performance stats

        # Track currently selected antibiotic type
        self.selected_antibiotic_type = None

        self.qt_app = None
        self.window = None

        if QtWidgets is not None:
            try:
                self.qt_app = QtWidgets.QApplication.instance()
                if self.qt_app is None:
                    self.qt_app = QtWidgets.QApplication([])
                self.window = QtWidgets.QWidget()
                self.window.setWindowTitle("Control Panel")
                self.build_controls()
                self.window.show()
            except Exception as exc:  # pragma: no cover - UI init errors should not break sim
                print(f"Qt UI init failed: {exc}")
                self.window = None
                self.qt_app = None
        else:
            print("PyQt5 not available - running without control panel UI")

    def set_ui_reference(self, ui):
        """Set reference to UI for accessing performance metrics"""
        self.ui_ref = ui

    # ------------------------------------------------------------------
    # UI construction helpers
    # ------------------------------------------------------------------
    def build_controls(self):
        """Build PyQt5 control panel"""
        main_layout = QtWidgets.QHBoxLayout(self.window)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # Left column - controls
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        title = QtWidgets.QLabel("<b>Simulation Controls</b>")
        left_layout.addWidget(title)

        controls_layout = QtWidgets.QHBoxLayout()
        self.pause_btn = QtWidgets.QPushButton("Start")
        self.pause_btn.clicked.connect(self.on_toggle_pause)
        controls_layout.addWidget(self.pause_btn)

        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.clicked.connect(self.on_reset)
        controls_layout.addWidget(reset_btn)
        controls_layout.addStretch()
        left_layout.addLayout(controls_layout)

        self._add_speed_controls(left_layout)
        left_layout.addWidget(self._create_separator())
        self._add_antibiotic_controls(left_layout)
        left_layout.addWidget(self._create_separator())
        self._add_performance_controls(left_layout)
        left_layout.addWidget(self._create_separator())
        self._add_layer_visibility_group(left_layout)
        left_layout.addWidget(self._create_separator())
        self._add_tracking_controls(left_layout)
        left_layout.addStretch(1)

        # Right column - stats display
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setSpacing(10)

        stats_title = QtWidgets.QLabel("<b>Population Stats</b>")
        right_layout.addWidget(stats_title)

        self.stats_scroll = QtWidgets.QScrollArea()
        self.stats_scroll.setWidgetResizable(True)
        self.stats_container = QtWidgets.QWidget()
        self.stats_layout = QtWidgets.QGridLayout(self.stats_container)
        self.stats_layout.setColumnStretch(0, 1)
        self.stats_scroll.setWidget(self.stats_container)

        right_layout.addWidget(self.stats_scroll)

        main_layout.addWidget(left_widget, stretch=0)
        main_layout.addWidget(right_widget, stretch=1)

        self.stats_labels = {}

    def _create_separator(self):
        frame = QtWidgets.QFrame()
        frame.setFrameShape(QtWidgets.QFrame.HLine)
        frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        return frame

    def _add_speed_controls(self, layout):
        """Add speed control widgets"""
        section_title = QtWidgets.QLabel("<b>Speed Control</b>")
        layout.addWidget(section_title)

        buttons_layout = QtWidgets.QHBoxLayout()
        slower_btn = QtWidgets.QPushButton("<<")
        slower_btn.setFixedWidth(45)
        slower_btn.clicked.connect(lambda: self.on_speed_change(-1))
        buttons_layout.addWidget(slower_btn)

        faster_btn = QtWidgets.QPushButton(">>")
        faster_btn.setFixedWidth(45)
        faster_btn.clicked.connect(lambda: self.on_speed_change(1))
        buttons_layout.addWidget(faster_btn)

        reset_btn = QtWidgets.QPushButton("Reset Speed")
        reset_btn.clicked.connect(lambda: self.on_speed_change(0))
        buttons_layout.addWidget(reset_btn)

        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)

        self.speed_label = QtWidgets.QLabel(
            f"Speed: {DEFAULT_STEPS_PER_FRAME} steps/frame"
        )
        layout.addWidget(self.speed_label)

    def _add_antibiotic_controls(self, layout):
        """Add antibiotic control widgets"""
        layout.addWidget(QtWidgets.QLabel("<b>Antibiotic Control</b>"))

        antibiotic_layout = QtWidgets.QHBoxLayout()
        antibiotic_layout.addWidget(QtWidgets.QLabel("Type:"))

        initial_antibiotic = (
            self.model.current_antibiotic
            if self.model.current_antibiotic
            else self.model.available_antibiotics[0]
        )
        self.selected_antibiotic_type = initial_antibiotic

        self.antibiotic_combo = QtWidgets.QComboBox()
        self.antibiotic_combo.addItems(self.model.available_antibiotics)
        self.antibiotic_combo.setCurrentText(initial_antibiotic)
        self.antibiotic_combo.currentTextChanged.connect(self._change_antibiotic)
        antibiotic_layout.addWidget(self.antibiotic_combo)
        antibiotic_layout.addStretch()
        layout.addLayout(antibiotic_layout)

        dose_layout = QtWidgets.QHBoxLayout()
        dose_layout.addWidget(QtWidgets.QLabel("Dose:"))
        self.dose_spin = QtWidgets.QDoubleSpinBox()
        self.dose_spin.setRange(0.0, 100.0)
        self.dose_spin.setSingleStep(0.1)
        self.dose_spin.setDecimals(3)
        self.dose_spin.setValue(0.5)
        dose_layout.addWidget(self.dose_spin)
        dose_layout.addStretch()
        layout.addLayout(dose_layout)

        apply_btn = QtWidgets.QPushButton("Apply Antibiotic")
        apply_btn.clicked.connect(self._apply_antibiotic_internal)
        layout.addWidget(apply_btn)

        latest_layout = QtWidgets.QHBoxLayout()
        latest_layout.addWidget(QtWidgets.QLabel("Latest dose:"))
        self.latest_label = QtWidgets.QLabel("0.0")
        latest_layout.addWidget(self.latest_label)
        latest_layout.addStretch()
        layout.addLayout(latest_layout)

        # Ensure model matches initial selection only once at startup
        self.model.set_antibiotic_type(initial_antibiotic)

    def _add_performance_controls(self, layout):
        """Add performance mode toggle and info"""
        self.perf_mode_check = QtWidgets.QCheckBox(
            "Performance Mode (Control Panel)"
        )
        self.perf_mode_check.setChecked(PERFORMANCE_MODE)
        self.perf_mode_check.stateChanged.connect(self._toggle_performance_mode)
        layout.addWidget(self.perf_mode_check)

        self.perf_info_label = QtWidgets.QLabel(
            "(reduces control panel update frequency, not simulation)"
        )
        self.perf_info_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.perf_info_label)

    def _add_tracking_controls(self, layout):
        """Add individual tracking controls"""
        layout.addWidget(QtWidgets.QLabel("<b>Browse Bacteria</b>"))

        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.addWidget(QtWidgets.QLabel("Show:"))
        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems(["alive", "deceased", "all"])
        self.filter_combo.setCurrentText("alive")
        self.filter_combo.currentTextChanged.connect(
            lambda text: self.update_bacteria_list(filter_type=text, force_update=True)
        )
        filter_layout.addWidget(self.filter_combo)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        self.bacteria_listbox = QtWidgets.QListWidget()
        self.bacteria_listbox.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection
        )
        self.bacteria_listbox.itemDoubleClicked.connect(
            lambda _: self._view_selected_bacterium()
        )
        self.bacteria_listbox.setMinimumHeight(180)
        layout.addWidget(self.bacteria_listbox)

        self.tracking_stats_label = QtWidgets.QLabel(
            "Tracked: 0 alive, 0 deceased"
        )
        self.tracking_stats_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.tracking_stats_label)

        view_btn = QtWidgets.QPushButton("View Selected Bacterium")
        view_btn.clicked.connect(self._view_selected_bacterium)
        layout.addWidget(view_btn)

    def _add_layer_visibility_group(self, layout):
        """Add layer visibility checkboxes for toggling visualization elements"""
        group = QtWidgets.QGroupBox("Layers")
        grid = QtWidgets.QGridLayout()
        layers = [
            ("agents", "Agents"),
            ("food", "Food"),
            ("antibiotic", "Antibiotic"),
            ("qs", "Quorum"),
            ("biofilm", "Biofilm"),
        ]
        self.layer_checkboxes = {}
        for i, (key, label) in enumerate(layers):
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(True)
            cb.stateChanged.connect(lambda state, k=key: self._on_layer_cb(k, state))
            grid.addWidget(cb, i // 3, i % 3)
            self.layer_checkboxes[key] = cb
        group.setLayout(grid)
        layout.addWidget(group)

    def _on_layer_cb(self, layer, state):
        """Handle layer checkbox toggle."""
        if self.on_layer_visibility_change is not None:
            visible = state == QtCore.Qt.Checked if QtCore is not None else state == 2
            try:
                self.on_layer_visibility_change(layer, visible)
            except Exception as exc:
                print(f"Error toggling layer '{layer}': {exc}")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _change_antibiotic(self, new_value):
        """Change antibiotic type"""
        try:
            self.selected_antibiotic_type = new_value
            self.model.set_antibiotic_type(new_value)
        except Exception as exc:
            print(f"Error changing antibiotic: {exc}")

    def _apply_antibiotic_internal(self):
        """Internal handler for applying antibiotic"""
        try:
            val = float(self.dose_spin.value())
        except Exception:
            val = 0.0

        antibiotic_type = self.selected_antibiotic_type

        if not antibiotic_type and self.model.available_antibiotics:
            antibiotic_type = self.model.available_antibiotics[0]
            self.selected_antibiotic_type = antibiotic_type
            self.antibiotic_combo.setCurrentText(antibiotic_type)

        self.on_apply_antibiotic(antibiotic_type, val)
        self.latest_label.setText(f"{val:.3f}")

    def _toggle_performance_mode(self, _state=None):
        """Internal handler for performance mode toggle"""
        try:
            if self.ui_ref is not None:
                new_val = self.perf_mode_check.isChecked()
                self.ui_ref.toggle_performance_mode(new_val)
            else:
                print("Warning: UI reference not set, cannot toggle performance mode")
        except Exception as exc:
            print(f"Error toggling performance mode: {exc}")

    def _view_selected_bacterium(self):
        """View selected bacterium plots"""
        try:
            current_item = self.bacteria_listbox.currentItem()
            if current_item is None:
                print("No bacterium selected")
                return

            if QtCore is not None:
                bacterium_id = current_item.data(QtCore.Qt.UserRole)
            else:  # pragma: no cover - fallback when QtCore unavailable
                text = current_item.text()
                id_part = text.split("ID:")[1].strip()
                bacterium_id = int(id_part.split()[0])

            if bacterium_id is None:
                print("No bacterium selected")
                return

            self.on_view_bacterium(int(bacterium_id))
        except Exception as exc:
            print(f"Error viewing bacterium: {exc}")

    # ------------------------------------------------------------------
    # External API used by SimulatorUI
    # ------------------------------------------------------------------
    def update_speed_display(self, speed):
        """Update speed label"""
        if self.window is not None:
            try:
                self.speed_label.setText(f"Speed: {speed} steps/frame")
            except Exception:
                pass

    def update_stats_display(self, stats):
        """Update population statistics display"""
        if self.window is None:
            return

        try:
            if not hasattr(self, "_stats_initialized"):
                self._create_stats_labels()
                self._stats_initialized = True

            self._update_stats_values(stats)
        except Exception as exc:
            print(f"Error updating stats: {exc}")

    def _create_stats_labels(self):
        """Create all stats labels once"""
        self.stats_labels = {}
        row = 0

        self.stats_labels["total"] = QtWidgets.QLabel("Total: 0")
        self.stats_layout.addWidget(self.stats_labels["total"], row, 0, 1, 2)
        row += 1

        for btype in BACTERIAL_TYPES.keys():
            type_label = QtWidgets.QLabel(f"<b>{btype}: 0</b>")
            self.stats_labels[f"type_{btype}"] = type_label
            self.stats_layout.addWidget(type_label, row, 0, 1, 2)
            row += 1

            for trait in ["enzyme", "efflux", "membrane", "repair", "age"]:
                text = f"  {trait}: 0.000" if trait != "age" else "  age: 0.0"
                label = QtWidgets.QLabel(text)
                self.stats_labels[f"{btype}_{trait}"] = label
                self.stats_layout.addWidget(label, row, 0, 1, 2)
                row += 1

        overall_label = QtWidgets.QLabel("<b>Overall Avg:</b>")
        self.stats_layout.addWidget(overall_label, row, 0, 1, 2)
        row += 1

        for trait in ["enzyme", "efflux", "membrane", "repair"]:
            label = QtWidgets.QLabel(f"  {trait}: 0.000")
            self.stats_labels[f"avg_{trait}"] = label
            self.stats_layout.addWidget(label, row, 0, 1, 2)
            row += 1

        avg_age_label = QtWidgets.QLabel("  age: 0.0")
        self.stats_labels["avg_age"] = avg_age_label
        self.stats_layout.addWidget(avg_age_label, row, 0, 1, 2)

    def _update_stats_values(self, stats):
        """Update existing stats labels with new values"""
        self.stats_labels["total"].setText(f"Total: {stats['total']}")

        for btype in BACTERIAL_TYPES.keys():
            count = stats["by_type"].get(btype, 0)
            self.stats_labels[f"type_{btype}"].setText(f"{btype}: {count}")

            if count > 0 and btype in stats:
                type_stats = stats[btype]
                self.stats_labels[f"{btype}_enzyme"].setText(
                    f"  enzyme: {type_stats['enzyme']:.3f}"
                )
                self.stats_labels[f"{btype}_efflux"].setText(
                    f"  efflux: {type_stats['efflux']:.3f}"
                )
                self.stats_labels[f"{btype}_membrane"].setText(
                    f"  membrane: {type_stats['membrane']:.3f}"
                )
                self.stats_labels[f"{btype}_repair"].setText(
                    f"  repair: {type_stats['repair']:.3f}"
                )
                self.stats_labels[f"{btype}_age"].setText(
                    f"  age: {type_stats['age']:.1f}"
                )
            else:
                self.stats_labels[f"{btype}_enzyme"].setText("  enzyme: 0.000")
                self.stats_labels[f"{btype}_efflux"].setText("  efflux: 0.000")
                self.stats_labels[f"{btype}_membrane"].setText("  membrane: 0.000")
                self.stats_labels[f"{btype}_repair"].setText("  repair: 0.000")
                self.stats_labels[f"{btype}_age"].setText("  age: 0.0")

        if stats["total"] > 0:
            for trait, value in stats["avg_traits"].items():
                if trait != "age":
                    self.stats_labels[f"avg_{trait}"].setText(
                        f"  {trait}: {value:.3f}"
                    )
            self.stats_labels["avg_age"].setText(
                f"  age: {stats['avg_traits']['age']:.1f}"
            )
        else:
            for trait in ["enzyme", "efflux", "membrane", "repair"]:
                self.stats_labels[f"avg_{trait}"].setText(
                    f"  {trait}: 0.000"
                )
            self.stats_labels["avg_age"].setText("  age: 0.0")

    def update_bacteria_list(self, filter_type=None, force_update=False):
        """Update bacteria listbox based on filter"""
        if self.window is None:
            return

        try:
            if filter_type is None:
                filter_type = (
                    self.filter_combo.currentText()
                    if hasattr(self, "filter_combo")
                    else "alive"
                )

            tracker = self.model.individual_tracker

            if filter_type == "alive":
                ids = tracker.get_alive_individuals()
            elif filter_type == "deceased":
                ids = tracker.get_deceased_individuals()
            else:
                ids = tracker.get_all_tracked_ids()

            current_hash = (filter_type, tuple(sorted(ids)))
            if not force_update and current_hash == self.last_bacteria_list_hash:
                return

            self.last_bacteria_list_hash = current_hash

            current_item = self.bacteria_listbox.currentItem()
            old_selected_id = None
            if current_item is not None:
                if QtCore is not None:
                    old_selected_id = current_item.data(QtCore.Qt.UserRole)
                else:  # pragma: no cover
                    text = current_item.text()
                    id_part = text.split("ID:")[1].strip()
                    old_selected_id = int(id_part.split()[0])

            self.bacteria_listbox.clear()
            ids.sort()

            new_selection_item = None
            for bacterium_id in ids:
                data = tracker.get_tracked_data(bacterium_id)
                if data:
                    btype = data["bacterial_type"]
                    text = f"ID:{bacterium_id:3d} {btype}"
                    item = QtWidgets.QListWidgetItem(text)
                    if QtCore is not None:
                        item.setData(QtCore.Qt.UserRole, bacterium_id)
                    self.bacteria_listbox.addItem(item)
                    if bacterium_id == old_selected_id:
                        new_selection_item = item

            if new_selection_item is not None:
                self.bacteria_listbox.setCurrentItem(new_selection_item)
                self.bacteria_listbox.scrollToItem(
                    new_selection_item,
                    QtWidgets.QAbstractItemView.PositionAtCenter,
                )

            stats = tracker.get_statistics()
            self.tracking_stats_label.setText(
                f"Tracked: {stats['alive']} alive, {stats['deceased']} deceased (total: {stats['total_tracked']})"
            )
        except Exception as exc:
            print(f"Error updating bacteria list: {exc}")

    def set_pause_button_text(self, text):
        """Set pause button text"""
        if self.window is not None:
            try:
                self.pause_btn.setText(text)
            except Exception:
                pass

    def set_pause_button_state(self, state):
        """Set pause button state (normal or disabled)"""
        if self.window is not None:
            try:
                enabled = state != "disabled"
                self.pause_btn.setEnabled(enabled)
            except Exception:
                pass

    def update(self):
        """Update Qt event loop"""
        if self.qt_app is not None:
            try:
                # Process Qt events to keep UI responsive
                # Use a try-except to handle any events that might crash
                self.qt_app.processEvents()
            except RuntimeError as exc:
                # Qt can raise RuntimeError during window close
                print(f"Qt event processing error (window may be closing): {exc}")
            except Exception as exc:
                # Catch any other exceptions to prevent crashes
                print(f"Error pumping Qt events: {exc}")
