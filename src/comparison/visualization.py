"""
Visualization and reporting functions for agent comparison.
"""

from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from .metrics import RunMetrics


def plot_comparison(
    all_metrics: List[RunMetrics],
    target_population: int,
    tolerance: float,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """Generate comparison plots for all agents."""
    n_agents = len(all_metrics)
    
    # Create figure with dynamic layout
    n_cols = min(3, n_agents)
    n_rows = (n_agents + n_cols - 1) // n_cols + 1  # +1 for summary plots
    
    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
    
    # Target band
    lower_band = target_population * (1 - tolerance)
    upper_band = target_population * (1 + tolerance)
    
    # Colors for different agents
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    dose_color = '#e74c3c'
    count_color = '#9b59b6'
    
    # Plot each agent's population
    for i, metrics in enumerate(all_metrics):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        steps = list(range(len(metrics.populations)))
        
        ax.plot(steps, metrics.populations, color=colors[i], linewidth=1.5, label='Population')
        ax.axhline(y=target_population, color='gray', linestyle='--', linewidth=1, label='Target')
        ax.axhspan(lower_band, upper_band, alpha=0.2, color='gray')
        
        # Vertical lines for actions
        for step in metrics.dose_steps:
            ax.axvline(x=step, color=dose_color, alpha=0.6, linewidth=0.8)
        for step in metrics.count_steps:
            ax.axvline(x=step, color=count_color, alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Population')
        title = f'{metrics.agent_name}\n(Gaussian: {metrics.gaussian_kernel_score:.3f})'
        if metrics.early_termination_reason:
            title += f'\n[Early Stop]'
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=[
            Line2D([0], [0], color=colors[i], linewidth=2, label='Population'),
            Line2D([0], [0], color=dose_color, linewidth=2, label='DOSE'),
            Line2D([0], [0], color=count_color, linewidth=2, label='COUNT'),
        ], loc='upper right', fontsize=7)
    
    # Budget comparison plot
    ax_budget = fig.add_subplot(n_rows, n_cols, n_agents + 1)
    for i, metrics in enumerate(all_metrics):
        ax_budget.plot(range(len(metrics.budget_history)), metrics.budget_history,
                      color=colors[i], linewidth=1.5, label=metrics.agent_name)
    ax_budget.set_xlabel('Step')
    ax_budget.set_ylabel('Budget Remaining')
    ax_budget.set_title('Budget Usage Over Time')
    ax_budget.legend(fontsize=8)
    ax_budget.grid(True, alpha=0.3)
    
    # Action distribution comparison
    ax_actions = fig.add_subplot(n_rows, n_cols, n_agents + 2)
    action_types = ['NOOP', 'COUNT', 'SEQUENCE', 'DOSE']
    x = np.arange(len(action_types))
    width = 0.8 / n_agents
    
    for i, metrics in enumerate(all_metrics):
        counts = [metrics.action_counts.get(a, 0) for a in action_types]
        offset = (i - n_agents / 2 + 0.5) * width
        ax_actions.bar(x + offset, counts, width, label=metrics.agent_name, color=colors[i], alpha=0.8)
    
    ax_actions.set_xlabel('Action Type')
    ax_actions.set_ylabel('Count')
    ax_actions.set_title('Action Distribution')
    ax_actions.set_xticks(x)
    ax_actions.set_xticklabels(action_types)
    ax_actions.legend(fontsize=8)
    ax_actions.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig


def print_comparison_table(all_metrics: List[RunMetrics]):
    """Print a formatted comparison table for all agents."""
    print("\n" + "=" * 100)
    print("                              AGENT COMPARISON RESULTS")
    print("=" * 100)
    
    # Header
    agent_names = [m.agent_name[:20] for m in all_metrics]
    header = f"{'Metric':<30}"
    for name in agent_names:
        header += f" {name:>15}"
    print(f"\n{header}")
    print("-" * 100)
    
    def fmt_row(label: str, values: List, fmt: str = ".2f"):
        row = f"{label:<30}"
        for v in values:
            if isinstance(v, float):
                row += f" {v:>15{fmt}}"
            else:
                row += f" {str(v):>15}"
        print(row)
    
    # Population Maintenance
    print("\n📊 POPULATION MAINTENANCE")
    print("-" * 50)
    fmt_row("Initial Population", [m.initial_population for m in all_metrics], "d")
    fmt_row("Final Population", [m.final_population for m in all_metrics], "d")
    fmt_row("Mean Population", [m.mean_population for m in all_metrics], ".1f")
    fmt_row("Std Population", [m.std_population for m in all_metrics], ".1f")
    fmt_row("Min Population", [m.min_population for m in all_metrics], "d")
    fmt_row("Max Population", [m.max_population for m in all_metrics], "d")
    
    # Target tracking
    print("\n🎯 TARGET TRACKING")
    print("-" * 50)
    fmt_row("Target Population", [m.target_population for m in all_metrics], "d")
    fmt_row("Steps in Target Band", [m.steps_in_target_band for m in all_metrics], "d")
    fmt_row("Target Band Ratio (%)", [m.target_band_ratio * 100 for m in all_metrics], ".1f")
    fmt_row("Mean Absolute Error", [m.mean_absolute_error for m in all_metrics], ".2f")
    
    # Kernel scores
    print("\n📈 KERNEL MAINTENANCE SCORES")
    print("-" * 50)
    fmt_row("Gaussian Kernel Score", [m.gaussian_kernel_score for m in all_metrics], ".4f")
    fmt_row("Laplace Kernel Score", [m.laplace_kernel_score for m in all_metrics], ".4f")
    
    # Budget
    print("\n💰 BUDGET")
    print("-" * 50)
    fmt_row("Initial Budget", [m.initial_budget for m in all_metrics], ".1f")
    fmt_row("Final Budget", [m.final_budget for m in all_metrics], ".1f")
    fmt_row("Budget Spent", [m.budget_spent for m in all_metrics], ".1f")
    
    # Actions
    print("\n🎮 ACTIONS")
    print("-" * 50)
    fmt_row("Total Steps", [m.steps for m in all_metrics], "d")
    fmt_row("NOOP Actions", [m.action_counts.get("NOOP", 0) for m in all_metrics], "d")
    fmt_row("COUNT Actions", [m.action_counts.get("COUNT", 0) for m in all_metrics], "d")
    fmt_row("DOSE Actions", [m.action_counts.get("DOSE", 0) for m in all_metrics], "d")
    
    print("\n" + "=" * 100)
    
    # Winner summary based on Gaussian kernel score
    print("\n🏆 RANKING (by Gaussian Kernel Score):")
    sorted_metrics = sorted(all_metrics, key=lambda m: m.gaussian_kernel_score, reverse=True)
    for i, m in enumerate(sorted_metrics, 1):
        print(f"  {i}. {m.agent_name}: {m.gaussian_kernel_score:.4f}")
    print("=" * 100)


def plot_radar_chart(
    all_metrics: List[RunMetrics],
    output_path: Optional[str] = None,
    show: bool = True,
):
    """
    Generate a radar chart comparing all agents across multiple metrics.
    
    All metrics are normalized so that higher values = better performance.
    """
    if not all_metrics:
        return None
    
    # Define metrics to display (all higher=better after normalization)
    # Show metrics: Gaussian Score, Gaussian*Steps (weights steps by gauss score), Low Error, Steps, and Budget AUC
    # Budget AUC is normalized by (tau * B0) per formula:
    # AUC_norm = (1 / (tau * B0)) * sum_{t=0}^{tau-1} B_t
    metric_names = [
        'Gaussian Score',
        'Gaussian*Steps',
        'Low Error',
        'Steps',
        'Budget AUC',
    ]

    # Prepare data for each agent
    # To normalize steps (unbounded) we compute the global max across all metrics
    max_steps = max((m.steps for m in all_metrics), default=1)

    # Precompute max of gaussian_score * steps to normalize the new measure


    agent_data = []
    # Keep track of observed raw maxima for each metric so we can annotate axis with their highest values
    observed_gaussian = []
    observed_gauss_steps = []
    observed_low_error = []
    observed_steps = []
    observed_auc = []
    for metrics in all_metrics:
        # Gaussian kernel score (already 0-1, higher=better)
        gaussian_score = max(0.0, min(1.0, metrics.gaussian_kernel_score))

        # Gaussian*Steps: weight the number of steps by gaussian score, then normalize across agents
        steps_norm = metrics.steps / max_steps if max_steps > 0 else 0.0
        gauss_steps_norm = gaussian_score * steps_norm
        gauss_steps_raw = gaussian_score * metrics.steps
        # Budget efficiency: proportion of budget remaining (0-1, higher=better)
        #budget_efficiency = metrics.final_budget / metrics.initial_budget if metrics.initial_budget > 0 else 0.0
        
        # Population stability: inverse of coefficient of variation (normalized)
        # Lower CV = more stable = better
        cv = metrics.std_population / metrics.mean_population if metrics.mean_population > 0 else 1.0
        #stability = 1.0 / (1.0 + cv)  # Normalize to 0-1, higher=better
        
        # Low error: inverse of normalized MAE (higher=better means lower error)
        # Normalize MAE by target population
        normalized_mae = metrics.mean_absolute_error / metrics.target_population if metrics.target_population > 0 else 1.0
        low_error = 1.0 / (1.0 + normalized_mae)  # Higher=better (lower error)
        
        # Normalize steps (higher = better) relative to the max among all agents
        steps_norm = metrics.steps / max_steps if max_steps > 0 else 0.0

        # Budget AUC normalization per user formula
        auc_norm = 0.0
        if metrics.budget_history:
            # tau is the total number of steps performed by the agent
            tau = metrics.steps if metrics.steps > 0 else len(metrics.budget_history)
            # Use initial budget if available, otherwise take first entry
            B0 = metrics.initial_budget if metrics.initial_budget > 0 else metrics.budget_history[0]
            if B0 > 0 and tau > 0:
                # If history shorter than tau, assume budget stays at last known value
                if len(metrics.budget_history) >= tau:
                    sum_Bt = sum(metrics.budget_history[:tau])
                else:
                    sum_Bt = sum(metrics.budget_history) + metrics.budget_history[-1] * (tau - len(metrics.budget_history))
                auc_norm = float(sum_Bt) / (tau * B0)

        agent_data.append({
            'name': metrics.agent_name,
            'values': [gaussian_score, gauss_steps_norm, low_error, steps_norm, auc_norm]
        })
        # record observed raw values for axis annotations
        observed_gaussian.append(gaussian_score)
        observed_gauss_steps.append(gauss_steps_raw)
        observed_low_error.append(low_error)
        observed_steps.append(metrics.steps)
        observed_auc.append(auc_norm)
    
    # Number of metrics
    num_vars = len(metric_names)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Complete the loop
    angles += angles[:1]
    
    # Create figure (slightly smaller so the chart doesn't overlap window labels/title)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Colors for different agents
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_metrics)))
    
    # Plot each agent
    for i, agent in enumerate(agent_data):
        values = agent['values']
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=agent['name'], color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, size=11)
    # Move x tick labels slightly inward to avoid overlap with max-value annotations
    ax.tick_params(axis='x', pad=-6)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2,  0.6,  1.0])
    ax.set_yticklabels(['0.2', '0.6', '1.0'], size=9, color='gray')
    
    # Add grid
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    # Use suptitle to place the main title above the entire figure, outside the radar area
    plt.suptitle('Agent Performance Comparison\n(Higher = Better for all metrics)',
                 size=15, weight='bold', y=0.98)
    # Remove the subplot title to avoid overlap
    ax.set_title("")
    # Adjust layout so the chart sits lower and doesn't overlap the suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    # Annotate each axis with the highest observed raw value for that metric (placed just outside the rim)
    # Prepare human-readable labels per metric
    axis_max_labels = []
    # Gaussian Score
    max_g = max(observed_gaussian) if observed_gaussian else 0.0
    axis_max_labels.append(f"max: {max_g:.2f}")
    # Gaussian*Steps (raw: gaussian * steps)
    max_gs = max(observed_gauss_steps) if observed_gauss_steps else 0.0
    axis_max_labels.append(f"max: {int(max_gs)} steps")
    # Low Error
    max_le = max(observed_low_error) if observed_low_error else 0.0
    axis_max_labels.append(f"max: {max_le:.2f}")
    # Steps
    max_s = max(observed_steps) if observed_steps else 0
    axis_max_labels.append(f"max: {int(max_s)} steps")
    # Budget AUC
    max_auc = max(observed_auc) if observed_auc else 0.0
    axis_max_labels.append(f"max: {max_auc:.2f}")

    # Place annotations at each axis angle slightly outside the outer radius
    for i, (angle, lbl) in enumerate(zip(angles[:-1], axis_max_labels)):
        # For Budget AUC (axis 0) and Gaussian*Steps (axis 1), rotate and nudge further out
        if i == 0:  # Budget AUC (leftmost)
            ax.text(angle, 1.18, lbl, size=8, ha='right', va='center', color='gray', rotation=angle*180/np.pi-90, rotation_mode='anchor')
        elif i == 1:  # Gaussian*Steps (top-right)
            ax.text(angle, 1.16, lbl, size=8, ha='left', va='center', color='gray', rotation=angle*180/np.pi-90, rotation_mode='anchor')
        else:
            ax.text(angle, 1.12, lbl, size=8, ha='center', va='center', color='gray')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Radar chart saved to: {output_path}")
    
    if show:
        plt.show()
    
    return fig