"""
Individual bacteria tracking and plotting functionality.
"""

import matplotlib.pyplot as plt
from collections import deque


class IndividualTracker:
    """Tracks individual bacteria throughout their lifecycle with bounded memory."""

    def __init__(self, max_history=1000, max_individuals=2000, enabled=True):
        self.enabled = bool(enabled)
        self.max_history = max_history
        self.max_individuals = None
        if max_individuals is not None:
            max_individuals = int(max_individuals)
            if max_individuals > 0:
                self.max_individuals = max_individuals
        self.tracked_individuals = {}  # {bacterium_id: data_history}
        self.alive_individuals = set()
        self.deceased_individuals = set()
        self._registration_order = deque()

    def register_individual(self, bacterium):
        """Automatically register a new bacterium for tracking"""
        if not self.enabled:
            return

        if bacterium.unique_id not in self.tracked_individuals:
            self.tracked_individuals[bacterium.unique_id] = {
                "steps": deque(maxlen=self.max_history),
                "enzyme": deque(maxlen=self.max_history),
                "efflux": deque(maxlen=self.max_history),
                "membrane": deque(maxlen=self.max_history),
                "repair": deque(maxlen=self.max_history),
                "energy": deque(maxlen=self.max_history),
                "pos_x": deque(maxlen=self.max_history),
                "pos_y": deque(maxlen=self.max_history),
                "bacterial_type": bacterium.bacterial_type,
                "birth_step": None,
                "death_step": None,
                "cause_of_death": None,  # 'starvation', 'antibiotic', 'old_age'
            }
            self.alive_individuals.add(bacterium.unique_id)
            self._registration_order.append(bacterium.unique_id)
            self._evict_if_needed()

    def _evict_if_needed(self):
        if self.max_individuals is None:
            return

        while len(self.tracked_individuals) > self.max_individuals and self._registration_order:
            oldest_id = self._registration_order.popleft()
            if oldest_id in self.tracked_individuals:
                del self.tracked_individuals[oldest_id]
                self.alive_individuals.discard(oldest_id)
                self.deceased_individuals.discard(oldest_id)

    def update_tracked_individuals(self, model, to_remove=None):
        """Update data for all tracked individuals

        Args:
            model: The simulation model
            to_remove: Set of bacteria being removed this step (optional)
        """
        if not self.enabled:
            return

        current_step = model.step_count

        # Get current alive IDs (including those about to be removed)
        current_alive_ids = {b.unique_id for b in model.agent_set}

        # Register any new bacteria (only if they have a valid position)
        for bacterium in model.agent_set:
            # Skip bacteria without positions - they're not properly initialized yet
            if bacterium.pos is None:
                continue

            if bacterium.unique_id not in self.tracked_individuals:
                self.register_individual(bacterium)
                self.tracked_individuals[bacterium.unique_id][
                    "birth_step"
                ] = current_step

        # Update data for all alive bacteria (including those about to die)
        for bacterium in model.agent_set:
            # Skip if bacterium has no position (shouldn't happen, but safety check)
            if bacterium.pos is None:
                continue

            data = self.tracked_individuals[bacterium.unique_id]
            data["steps"].append(current_step)
            data["enzyme"].append(bacterium.enzyme)
            data["efflux"].append(bacterium.efflux)
            data["membrane"].append(bacterium.membrane)
            data["repair"].append(bacterium.repair)
            data["energy"].append(bacterium.energy)
            data["pos_x"].append(bacterium.pos[0])
            data["pos_y"].append(bacterium.pos[1])

        # Detect newly deceased bacteria
        # If to_remove is provided, use it to identify dying bacteria
        if to_remove:
            newly_deceased = {
                b.unique_id for b in to_remove if b.unique_id in self.alive_individuals
            }
        else:
            newly_deceased = self.alive_individuals - current_alive_ids

        for bacterium_id in newly_deceased:
            if bacterium_id in self.tracked_individuals:
                self.tracked_individuals[bacterium_id]["death_step"] = current_step
                self.deceased_individuals.add(bacterium_id)
            self.alive_individuals.discard(bacterium_id)

        # Update alive set (only if to_remove not provided)
        if not to_remove:
            self.alive_individuals = self.alive_individuals.intersection(
                current_alive_ids
            )

    def mark_death(self, bacterium_id, cause):
        """Mark a bacterium as deceased with cause"""
        if not self.enabled:
            return
        if bacterium_id in self.tracked_individuals:
            self.tracked_individuals[bacterium_id]["cause_of_death"] = cause

    def get_tracked_data(self, bacterium_id):
        """Get historical data for a specific bacterium"""
        return self.tracked_individuals.get(bacterium_id, None)

    def get_all_tracked_ids(self):
        """Get all tracked bacterium IDs (alive and dead)"""
        return list(self.tracked_individuals.keys())

    def get_alive_individuals(self):
        """Get list of currently alive individuals"""
        return list(self.alive_individuals)

    def get_deceased_individuals(self):
        """Get list of deceased individuals"""
        return list(self.deceased_individuals)

    def get_statistics(self):
        """Get overall tracking statistics"""
        return {
            "total_tracked": len(self.tracked_individuals),
            "alive": len(self.alive_individuals),
            "deceased": len(self.deceased_individuals),
        }


class IndividualPlotter:
    """Creates and manages individual bacteria plots."""

    def __init__(self, tracker, on_close_callback=None):
        self.tracker = tracker
        self.fig = None
        self.axes = None
        self.current_id = None
        self.on_close_callback = on_close_callback

    def close(self):
        """Close the plot window if it exists"""
        if self.fig is not None:
            try:
                plt.close(self.fig)
            except Exception as e:
                print(f"Error closing individual plot window: {e}")
            finally:
                self.fig = None
                self.axes = None
                self.current_id = None

    def _on_window_close(self, event):
        """Handle window close event"""
        self.fig = None
        self.axes = None
        self.current_id = None
        # Notify callback if provided
        if self.on_close_callback:
            self.on_close_callback()

    def create_plot_window(self):
        """Create a new window for individual plots"""
        # Close existing figure if it's still around but closed
        if self.fig is not None:
            try:
                # Check if figure is still valid
                if not plt.fignum_exists(self.fig.number):
                    self.fig = None
                    self.axes = None
            except:
                self.fig = None
                self.axes = None

        # Create new figure only if needed
        if self.fig is None:
            self.fig, self.axes = plt.subplots(1, 3, figsize=(10, 5))
            self.fig.suptitle("Individual Bacterium Tracking")
            # Register close event handler
            self.fig.canvas.mpl_connect("close_event", self._on_window_close)
            plt.tight_layout()
            plt.show(block=False)  # Show window without blocking

        return self.fig

    def _plot_resistance_traits(self, data, ax):
        """Plot current resistance traits as bar chart"""
        traits = ["enzyme", "efflux", "membrane", "repair"]
        values = [data[trait][-1] for trait in traits]
        ax.bar(traits, values, color=["red", "blue", "green", "orange"])
        ax.set_title("Resistance Traits")
        ax.set_ylabel("Trait Value")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)

    def _plot_energy_over_time(self, data, ax, is_alive):
        """Plot energy over time"""
        steps = list(data["steps"])
        ax.plot(steps, list(data["energy"]), "purple", linewidth=2)
        ax.set_title("Energy Over Time")
        ax.set_xlabel("Simulation Step")
        ax.set_ylabel("Energy")
        ax.axhline(
            y=0, color="r", linestyle="--", alpha=0.3, label="Starvation threshold"
        )
        if not is_alive and data["death_step"]:
            ax.axvline(
                x=data["death_step"],
                color="red",
                linestyle="--",
                alpha=0.5,
                label="Death",
            )
        ax.legend()

    def _plot_movement_trajectory(self, data, ax, is_alive):
        """Plot movement trajectory"""
        ax.plot(
            list(data["pos_x"]), list(data["pos_y"]), "darkblue", linewidth=1, alpha=0.7
        )
        # Mark start and end points
        ax.scatter(
            [data["pos_x"][0]],
            [data["pos_y"][0]],
            color="green",
            s=100,
            marker="o",
            label="Birth",
            zorder=5,
        )
        ax.scatter(
            [data["pos_x"][-1]],
            [data["pos_y"][-1]],
            color="red" if not is_alive else "blue",
            s=100,
            marker="X" if not is_alive else "o",
            label="Death" if not is_alive else "Current",
            zorder=5,
        )
        ax.set_title("Movement Trajectory")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend()

    def update_plots(self, bacterium_id):
        """Update all plots for the given bacterium ID"""
        # Check if figure was closed by user
        if self.fig is not None:
            try:
                if not plt.fignum_exists(self.fig.number):
                    # Window was closed, clear everything and don't reopen
                    self.fig = None
                    self.axes = None
                    self.current_id = None
                    if self.on_close_callback:
                        self.on_close_callback()
                    return
            except:
                self.fig = None
                self.axes = None
                self.current_id = None
                if self.on_close_callback:
                    self.on_close_callback()
                return

        # Create window if needed (only when explicitly called, not from close)
        if self.fig is None:
            self.create_plot_window()

        data = self.tracker.get_tracked_data(bacterium_id)
        if data is None:
            print(f"Bacterium {bacterium_id} not found in tracking data")
            return

        if len(data["steps"]) == 0:
            print(f"No data collected yet for bacterium {bacterium_id}")
            return

        self.current_id = bacterium_id

        # Clear all axes
        for ax in self.axes:
            ax.clear()

        # Determine if bacterium is alive or dead
        is_alive = bacterium_id in self.tracker.alive_individuals
        status = "ALIVE" if is_alive else f"DECEASED (step {data['death_step']})"
        if not is_alive and data["cause_of_death"]:
            status += f" - {data['cause_of_death']}"

        # Update main title
        self.fig.suptitle(
            f"Individual Bacterium Tracking - ID: {bacterium_id} ({data['bacterial_type']}) - {status}"
        )

        # Create all three plots
        self._plot_resistance_traits(data, self.axes[0])
        self._plot_energy_over_time(data, self.axes[1], is_alive)
        self._plot_movement_trajectory(data, self.axes[2], is_alive)

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()  # Force update

        # Bring window to front
        try:
            self.fig.canvas.manager.window.wm_attributes("-topmost", 1)
            self.fig.canvas.manager.window.wm_attributes("-topmost", 0)
        except:
            pass

    def is_window_open(self):
        """Check if the window is currently open"""
        if self.fig is None:
            return False
        try:
            return plt.fignum_exists(self.fig.number)
        except:
            return False
