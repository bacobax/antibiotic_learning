"""
Alternative Simulation UI that integrates the trained RL Agent with visual simulation.

This UI displays the bacterium evolution controlled by an RL agent instead of user input.
The agent uses the trained policy to decide on antibiotic dosing, counting, and sequencing actions.
"""

import numpy as np
import matplotlib.animation as animation
import time
from PyQt5 import QtWidgets, QtCore
from pathlib import Path
from typing import Optional
from matplotlib.figure import Figure

from simulation.simulation_config import (
    DEFAULT_STEPS_PER_FRAME,
    MIN_STEPS_PER_FRAME,
    MAX_STEPS_PER_FRAME,
    SLOW_MODE_FRAME_SKIP,
    PERFORMANCE_MODE,
    STATS_UPDATE_INTERVAL,
    VISUALIZATION_UPDATE_INTERVAL
)
from simulation.tracking import IndividualPlotter
from simulation.visualization import SimulationVisualizer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from rl.agent import RLAgent
from rl.env_wrapper import ACTION_NOOP, ACTION_COUNT_BACTERIA, ACTION_SEQUENCING, ACTION_DOSE


ACTION_NAMES = {
    ACTION_NOOP: "NOOP",
    ACTION_COUNT_BACTERIA: "COUNT",
    ACTION_SEQUENCING: "SEQUENCE",
    ACTION_DOSE: "DOSE"
}

# Ordered metadata describing reward components to visualize
REWARD_COMPONENT_ORDER = [
    ("pre", "reward_pre", "pre", "tab:blue"),
    ("post_penalties", "reward_post_penalties", "post_penalties", "tab:orange"),
    ("kernel_maintenance", "reward_kernel_maintenance", "kernel_maintenance", "tab:green"),
    ("survival_bonus", "reward_survival_bonus", "survival_bonus", "tab:purple"),
    ("prediction", "reward_prediction", "prediction", "tab:pink"),
    ("early_termination_penalty", "reward_early_termination_penalty", "early_termination_penalty", "tab:brown"),
    ("cost_penalty", "reward_cost_penalty", "cost_penalty", "tab:gray"),
    ("total", "reward_total", "total", "black"),
]

REWARD_COMPONENT_INFO_KEYS = {key: info_key for key, info_key, _label, _color in REWARD_COMPONENT_ORDER}
REWARD_COMPONENT_LABELS = {key: label for key, _info, label, _color in REWARD_COMPONENT_ORDER}
REWARD_COMPONENT_COLORS = {key: color for key, _info, _label, color in REWARD_COMPONENT_ORDER}


class VisualizationWindow(QtWidgets.QMainWindow):
    """PyQt5 window containing the embedded matplotlib visualization"""
    
    def __init__(self, visualizer):
        super().__init__()
        self.visualizer = visualizer
        self.setWindowTitle("RL Agent - Bacteria Simulation")
        self.setGeometry(100, 100, 1400, 700)
        
        # Embed matplotlib figure in PyQt5
        self.canvas = FigureCanvas(visualizer.fig)
        self.setCentralWidget(self.canvas)
        
        # Setup toolbar
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.addToolBar(self.toolbar)


class AgentControlPanel(QtWidgets.QMainWindow):
    """Control panel for RL Agent simulation"""
    
    def __init__(self, on_toggle_pause, on_reset, on_speed_change):
        super().__init__()
        self.setWindowTitle("RL Agent - Control Panel")
        self.setGeometry(0, 0, 500, 1100)
        
        self.on_toggle_pause = on_toggle_pause
        self.on_reset = on_reset
        self.on_speed_change = on_speed_change
        
        # Create central widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Title
        title = QtWidgets.QLabel("RL Agent Control")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)
        
        # Simulation controls
        controls_group = QtWidgets.QGroupBox("Simulation Control")
        controls_layout = QtWidgets.QVBoxLayout()
        
        self.pause_button = QtWidgets.QPushButton("Start")
        self.pause_button.clicked.connect(lambda: on_toggle_pause())
        controls_layout.addWidget(self.pause_button)
        
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(lambda: on_reset())
        controls_layout.addWidget(self.reset_button)
        
        # Speed control
        speed_layout = QtWidgets.QHBoxLayout()
        speed_layout.addWidget(QtWidgets.QLabel("Speed:"))
        
        slow_btn = QtWidgets.QPushButton("-")
        slow_btn.setMaximumWidth(40)
        slow_btn.clicked.connect(lambda: on_speed_change(-1))
        speed_layout.addWidget(slow_btn)
        
        self.speed_label = QtWidgets.QLabel("1x")
        self.speed_label.setAlignment(QtCore.Qt.AlignCenter)
        speed_layout.addWidget(self.speed_label)
        
        fast_btn = QtWidgets.QPushButton("+")
        fast_btn.setMaximumWidth(40)
        fast_btn.clicked.connect(lambda: on_speed_change(1))
        speed_layout.addWidget(fast_btn)
        
        default_btn = QtWidgets.QPushButton("Default")
        default_btn.setMaximumWidth(60)
        default_btn.clicked.connect(lambda: on_speed_change(0))
        speed_layout.addWidget(default_btn)
        
        controls_layout.addLayout(speed_layout)
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Agent Statistics
        stats_group = QtWidgets.QGroupBox("Agent Statistics")
        stats_layout = QtWidgets.QVBoxLayout()
        
        self.stats_display = QtWidgets.QTextEdit()
        self.stats_display.setReadOnly(True)
        self.stats_display.setMaximumHeight(300)
        stats_layout.addWidget(self.stats_display)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Episode Graphs
        self._init_episode_graphs(layout)
        
        # Stretch to fill remaining space
        layout.addStretch()

    def _init_episode_graphs(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Create matplotlib graphs for per-episode diagnostics."""
        self._action_ticks = sorted(ACTION_NAMES.items())
        
        graphs_container = QtWidgets.QWidget()
        graphs_layout = QtWidgets.QGridLayout()
        graphs_layout.setContentsMargins(0, 0, 0, 0)
        graphs_layout.setHorizontalSpacing(12)
        graphs_layout.setVerticalSpacing(12)

        # Budget over steps
        budget_group = QtWidgets.QGroupBox("Budget Over Steps (Current Episode)")
        budget_layout = QtWidgets.QVBoxLayout()
        self.budget_fig = Figure(figsize=(4.5, 2.4), dpi=100)
        self.budget_ax = self.budget_fig.add_subplot(111)
        self.budget_canvas = FigureCanvas(self.budget_fig)
        budget_layout.addWidget(self.budget_canvas)
        budget_group.setLayout(budget_layout)
        graphs_layout.addWidget(budget_group, 0, 0)

        # Action history
        action_group = QtWidgets.QGroupBox("Actions Over Steps")
        action_layout = QtWidgets.QVBoxLayout()
        self.action_fig = Figure(figsize=(4.5, 2.4), dpi=100)
        self.action_ax = self.action_fig.add_subplot(111)
        self.action_canvas = FigureCanvas(self.action_fig)
        action_layout.addWidget(self.action_canvas)
        action_group.setLayout(action_layout)
        graphs_layout.addWidget(action_group, 0, 1)

        # Reward evolution
        reward_group = QtWidgets.QGroupBox("Reward Over Steps")
        reward_layout = QtWidgets.QVBoxLayout()
        self.reward_fig = Figure(figsize=(4.5, 2.4), dpi=100)
        self.reward_ax = self.reward_fig.add_subplot(111)
        self.reward_canvas = FigureCanvas(self.reward_fig)
        reward_layout.addWidget(self.reward_canvas)
        reward_group.setLayout(reward_layout)
        graphs_layout.addWidget(reward_group, 1, 0)
        self.reward_component_keys = [key for key, *_ in REWARD_COMPONENT_ORDER]

        # Prediction vs true population
        population_group = QtWidgets.QGroupBox("Prediction vs True Population")
        population_layout = QtWidgets.QVBoxLayout()
        self.population_fig = Figure(figsize=(4.5, 2.6), dpi=100)
        self.population_ax = self.population_fig.add_subplot(111)
        self.population_canvas = FigureCanvas(self.population_fig)
        population_layout.addWidget(self.population_canvas)
        population_group.setLayout(population_layout)
        graphs_layout.addWidget(population_group, 1, 1)

        graphs_layout.setColumnStretch(0, 1)
        graphs_layout.setColumnStretch(1, 1)
        graphs_container.setLayout(graphs_layout)
        layout.addWidget(graphs_container)
        
        self.reset_graphs()
        
    def set_pause_button_text(self, text):
        """Update pause button text"""
        self.pause_button.setText(text)
    
    def set_pause_button_state(self, state):
        """Set pause button state: 'normal' or 'disabled'"""
        self.pause_button.setEnabled(state == "normal")
    
    def update_speed_display(self, steps_per_frame):
        """Update speed display"""
        self.speed_label.setText(f"{steps_per_frame}x")
    
    def update_agent_stats(self, stats):
        """Update agent statistics display"""
        text = f"""Budget: {stats.get('budget', 0):.2f}
Episode Return: {stats.get('episode_return', 0):.2f}
Time Step: {stats.get('t', 0)}
Last Action: {stats.get('last_action', 'N/A')}
Last Reward: {stats.get('last_reward', 0):.4f}
Seq Pending: {stats.get('seq_pending', False)}
Seq ETA: {stats.get('seq_eta', 0)}
Status: {stats.get('termination_reason', 'Running')}
Awaiting Reset: {"Yes (press Reset)" if stats.get('awaiting_reset') else "No"}
"""
        self.stats_display.setText(text)

    def reset_graphs(self) -> None:
        """Clear all episode graphs."""
        self._clear_axis(self.budget_ax, "Budget Over Steps (Current Episode)", "Step", "Budget", zero_min=True)
        self.budget_canvas.draw_idle()
        self._clear_axis(self.action_ax, "Actions Over Steps", "Step", "Action")
        if self._action_ticks:
            positions = [k for k, _ in self._action_ticks]
            labels = [v for _, v in self._action_ticks]
            self.action_ax.set_yticks(positions)
            self.action_ax.set_yticklabels(labels)
        self.action_canvas.draw_idle()
        self._clear_axis(self.reward_ax, "Reward Over Steps", "Step", "Reward")
        self.reward_ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        self.reward_canvas.draw_idle()
        self._clear_axis(self.population_ax, "Prediction vs True Population", "Step", "Population", zero_min=True)
        self.population_canvas.draw_idle()

    def update_episode_graphs(self, history: dict) -> None:
        """Refresh graph canvases with the latest episode history."""
        steps = history.get('steps', [])
        budgets = history.get('budgets', [])
        actions = history.get('actions', [])
        reward_components = history.get('reward_components', {})
        true_population = history.get('true_population', [])
        pred_population = history.get('pred_population', [])

        self._update_budget_plot(steps, budgets)
        self._update_action_plot(steps, actions)
        self._update_reward_plot(steps, reward_components)
        self._update_population_plot(steps, true_population, pred_population)

    def _clear_axis(self, ax, title: str, xlabel: str, ylabel: str, zero_min: bool = False) -> None:
        ax.clear()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if zero_min:
            ax.set_ylim(bottom=0)

    def _update_budget_plot(self, steps, budgets) -> None:
        self._clear_axis(self.budget_ax, "Budget Over Steps (Current Episode)", "Step", "Budget", zero_min=True)
        y_max = 1.0
        if steps and budgets and len(steps) == len(budgets):
            self.budget_ax.plot(steps, budgets, color="tab:green", linewidth=1.5)
            y_max = max(1.0, max(budgets))
        self.budget_ax.set_ylim(0, y_max * 1.05)
        self.budget_fig.tight_layout()
        self.budget_canvas.draw_idle()

    def _update_action_plot(self, steps, actions) -> None:
        self._clear_axis(self.action_ax, "Actions Over Steps", "Step", "Action")
        if self._action_ticks:
            positions = [k for k, _ in self._action_ticks]
            labels = [v for _, v in self._action_ticks]
            self.action_ax.set_yticks(positions)
            self.action_ax.set_yticklabels(labels)
        if steps and actions and len(steps) == len(actions):
            self.action_ax.step(steps, actions, where='post', linewidth=1.2, color='tab:orange')
        self.action_fig.tight_layout()
        self.action_canvas.draw_idle()

    def _update_reward_plot(self, steps, reward_components) -> None:
        self._clear_axis(self.reward_ax, "Reward Over Steps", "Step", "Reward")
        self.reward_ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        plotted = False
        if steps and reward_components:
            for key in self.reward_component_keys:
                values = reward_components.get(key)
                if not values or len(values) != len(steps):
                    continue
                color = REWARD_COMPONENT_COLORS.get(key)
                label = REWARD_COMPONENT_LABELS.get(key, key)
                linewidth = 2.0 if key == "total" else 1.2
                self.reward_ax.plot(
                    steps,
                    values,
                    label=label,
                    color=color,
                    linewidth=linewidth,
                )
                plotted = True
        if plotted:
            self.reward_ax.legend(loc='upper left', fontsize=8)
        self.reward_fig.tight_layout()
        self.reward_canvas.draw_idle()

    def _update_population_plot(self, steps, true_population, pred_population) -> None:
        self._clear_axis(self.population_ax, "Prediction vs True Population", "Step", "Population", zero_min=True)
        has_true = steps and true_population and len(steps) == len(true_population)
        has_pred = steps and pred_population and len(steps) == len(pred_population)
        y_max = 1.0
        if has_true:
            self.population_ax.plot(steps, true_population, label='True Population', color='tab:blue', linewidth=1.5)
            y_max = max(y_max, max(true_population))
        if has_pred:
            self.population_ax.plot(steps, pred_population, label='Predicted Population', color='tab:red', linestyle='--', linewidth=1.2)
            y_max = max(y_max, max(pred_population))
        if has_true or has_pred:
            self.population_ax.legend(loc='upper right', fontsize=8)
        self.population_ax.set_ylim(0, y_max * 1.05)
        self.population_fig.tight_layout()
        self.population_canvas.draw_idle()


class AgentSimulatorUI:
    """
    UI for visualizing RL Agent-controlled bacterium simulation.
    
    The agent controls antibiotic dosing, sequencing, and counting actions
    while the visualization shows the real-time bacterium evolution.
    """
    
    def __init__(self, model, env, checkpoint_path):
        """
        Initialize AgentSimulatorUI
        
        Args:
            model: BacteriaModel instance
            env: Initialized PetriEnvWrapper linked to the provided model
            checkpoint_path: Path to trained agent checkpoint
        """
        self.model = model
        self.env = env
        self.paused = True
        self.simulation_started = False
        self.population_extinct = False
        self.awaiting_manual_reset = False
        self.last_termination_reason: Optional[str] = None
        
        # Load trained agent
        print(f"Loading agent from {checkpoint_path}...")
        self.agent = RLAgent.load_agent_from_checkpoint(checkpoint_path)
        print(f"Agent loaded successfully! Model device: {self.agent.device}")
        
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
        
        # Agent tracking
        self.last_obs = None
        self.last_action = None
        self.last_reward = 0.0
        self.action_history = []
        self.reward_history = []
        self.episode_steps = []
        self.budget_history = []
        self.true_population_history = []
        self.prediction_history = []
        self.reward_component_history = {key: [] for key in REWARD_COMPONENT_INFO_KEYS}
        
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
        
        # Create visualization window
        self.viz_window = VisualizationWindow(self.visualizer)
        
        # Setup control panel
        self.control_panel = AgentControlPanel(
            on_toggle_pause=self.toggle_pause,
            on_reset=self.reset_sim,
            on_speed_change=self.handle_speed_change
        )
        
        # Position windows side by side
        self._arrange_windows()
        
        # Initialize environment and agent
        self._reset_environment()

    def _reset_environment(self):
        """Reset the environment and agent"""
        self.last_obs = self.env.reset()
        self.agent.start_episode()
        self.action_history = []
        self.reward_history = []
        self.episode_steps = []
        self.budget_history = []
        self.true_population_history = []
        self.prediction_history = []
        self.reward_component_history = {key: [] for key in REWARD_COMPONENT_INFO_KEYS}
        print("Environment reset for new episode")
        self.awaiting_manual_reset = False
        self.last_termination_reason = None
        if self.control_panel:
            self.control_panel.set_pause_button_state("normal")
            self.control_panel.set_pause_button_text("Start")
            self.control_panel.reset_graphs()

    def _arrange_windows(self):
        """Arrange visualization and control panel windows"""
        if self.control_panel:
            self.control_panel.move(0, 0)
            self.control_panel.resize(400, 900)
        
        self.viz_window.move(420, 0)
        self.viz_window.show()

    def on_bacterium_click(self, bacterium_id):
        """Handle bacterium click from visualizer"""
        self.individual_plotter.update_plots(bacterium_id)

    def on_individual_window_close(self):
        """Handle individual tracking window being closed"""
        self.visualizer.clear_highlight()
        print("Individual tracking window closed")

    def toggle_pause(self):
        """Toggle simulation pause/start"""
        if self.awaiting_manual_reset:
            print("Episode finished. Press Reset to start a new run.")
            return
        if not self.simulation_started:
            self.simulation_started = True
            self.paused = False
            self.control_panel.set_pause_button_text("Pause")
            print("Simulation started - Agent taking control")
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

        self._reset_environment()
        
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
        
        print("Simulation reset - press Start to begin")

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
        
        # Create animation that updates matplotlib figure
        self.animation = animation.FuncAnimation(
            self.visualizer.fig,
            self.update,
            interval=self.visualizer.animation_interval,
            blit=False,
            cache_frame_data=False
        )
        
        # Show windows
        if self.control_panel:
            self.control_panel.show()

    def update(self, frame):
        """Animation update callback - main simulation loop"""
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
                        self._agent_step()
                        steps_executed += 1
                else:
                    self.frame_counter += 1
                    if self.frame_counter >= SLOW_MODE_FRAME_SKIP:
                        self._agent_step()
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
                # Update agent stats
                agent_stats = {
                    'budget': self.env.budget,
                    'episode_return': self.env.episode_return,
                    't': self.env.t,
                    'last_action': ACTION_NAMES.get(self.last_action, 'N/A') if self.last_action is not None else 'N/A',
                    'last_reward': self.last_reward,
                    'seq_pending': self.env.seq_pending,
                    'seq_eta': self.env.seq_eta,
                    'termination_reason': self._format_termination_reason(self.last_termination_reason),
                    'awaiting_reset': self.awaiting_manual_reset,
                }
                self.control_panel.update_agent_stats(agent_stats)
                self.control_panel.update_episode_graphs(self._collect_episode_history())
                
            
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

    def _agent_step(self):
        """Execute one agent step in the environment"""
        # Get agent action
        (
            a_disc,
            a_cont,
            logp_disc,
            logp_cont,
            value,
            _pred_next_pop,
            h_prev,
            _action_mask,
            _prev_action_onehot,
            _prev_action_cont,
            _prev_pred_next_pop,
        ) = self.agent.select_action(self.last_obs)
        
        # Extract numpy arrays
        a_disc_np = int(a_disc.squeeze().cpu().numpy())
        a_cont_np = a_cont.squeeze().cpu().numpy()
        
        # Format continuous action for logging
        # a_cont_str = str(a_cont_np) if a_cont_np.ndim > 0 else f"{a_cont_np:.2f}"

        # print(f"Agent Action: {ACTION_NAMES.get(a_disc_np, 'N/A')} ({a_cont_str})")
        
        # Store for tracking
        self.last_action = a_disc_np
        
        # Step environment
        obs, reward, done, info = self.env.step(a_disc_np, a_cont_np)
        
        # Store metrics
        self.last_reward = reward
        self.action_history.append(a_disc_np)
        self.reward_history.append(reward)
        self.episode_steps.append(self.env.t)
        self.budget_history.append(self.env.budget)
        self.true_population_history.append(info.get('true_population', np.nan))
        pred_norm = float(_pred_next_pop.squeeze().detach().cpu().item()) if _pred_next_pop is not None else np.nan
        scaled_prediction = pred_norm * max(1.0, getattr(self.env, 'population_norm', 1.0))
        self.prediction_history.append(scaled_prediction)
        for comp_key, info_key in REWARD_COMPONENT_INFO_KEYS.items():
            self.reward_component_history.setdefault(comp_key, []).append(info.get(info_key, 0.0))
        
        # Update observation for next step
        self.last_obs = obs
        
        # Handle episode termination
        if done:
            print(f"Episode done! Total return: {self.env.episode_return:.2f}")
            self.last_termination_reason = info.get('termination_reason')
            if self.last_termination_reason:
                print(f"Termination reason: {self.last_termination_reason}")
            self.population_extinct = self.last_termination_reason == "extinction"
            self.paused = True
            self.simulation_started = False
            self.awaiting_manual_reset = True
            if self.control_panel:
                self.control_panel.set_pause_button_text("Start")
                self.control_panel.set_pause_button_state("disabled")

    def _format_termination_reason(self, reason: Optional[str]) -> str:
        if reason is None:
            return "Running"
        mapping = {
            "extinction": "Extinction (population reached 0)",
            "max_steps": "Max steps reached",
            "budget_depleted": "Budget depleted",
            "unrecoverable_high_population": "Unrecoverable: population too high",
            "unrecoverable_low_population": "Unrecoverable: population too low",
            "unrecoverable_state": "Unrecoverable state",
            "early_termination": "Early termination triggered",
        }
        return mapping.get(reason, reason)

    def _collect_episode_history(self) -> dict:
        return {
            'steps': list(self.episode_steps),
            'budgets': list(self.budget_history),
            'actions': list(self.action_history),
            'rewards': list(self.reward_history),
            'reward_components': {k: list(v) for k, v in self.reward_component_history.items()},
            'true_population': list(self.true_population_history),
            'pred_population': list(self.prediction_history),
        }

    @staticmethod
    def create_and_run(model, checkpoint_path):
        """
        Factory method to create and run the AgentSimulatorUI
        
        Args:
            model: BacteriaModel instance
            checkpoint_path: Path to trained agent checkpoint
        """
        ui = AgentSimulatorUI(model, checkpoint_path)
        ui.run()
        return ui


# Convenience functions for easy integration

def create_agent_ui(model, checkpoint_path):
    """Create an AgentSimulatorUI instance"""
    return AgentSimulatorUI(model, checkpoint_path)


def run_agent_simulation(checkpoint_path="checkpoints/checkpoint_final_30.pt"):
    """
    Run full agent simulation with UI
    
    Args:
        checkpoint_path: Path to trained agent checkpoint
    """
    from PyQt5 import QtWidgets
    import sys
    
    from model import BacteriaModel
    
    # Create QApplication if not exists
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    
    # Create model
    model = BacteriaModel()
    
    # Create and run UI
    ui = AgentSimulatorUI.create_and_run(model, checkpoint_path)
    
    # Show visualization window
    ui.viz_window.show()
    
    # Run event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    # Example usage
    run_agent_simulation()
