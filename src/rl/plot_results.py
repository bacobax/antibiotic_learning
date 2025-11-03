"""
Plot training results and bacteria population from simulation data.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_training_metrics(metrics_file: Path, save_dir: Path = None):
    """
    Plot training metrics from JSON file.
    
    Args:
        metrics_file: Path to metrics.json file
        save_dir: Directory to save plots (if None, just show)
    """
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        return
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("No metrics data found")
        return
    
    # Group metrics by name
    metrics_by_name = {}
    for entry in data:
        name = entry['metric']
        update = entry['update']
        value = entry['value']
        
        if name not in metrics_by_name:
            metrics_by_name[name] = {'updates': [], 'values': []}
        
        metrics_by_name[name]['updates'].append(update)
        metrics_by_name[name]['values'].append(value)
    
    # Sort by update
    for name in metrics_by_name:
        updates = metrics_by_name[name]['updates']
        values = metrics_by_name[name]['values']
        sorted_pairs = sorted(zip(updates, values))
        metrics_by_name[name]['updates'] = [p[0] for p in sorted_pairs]
        metrics_by_name[name]['values'] = [p[1] for p in sorted_pairs]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PPO Training Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean Episode Reward
    ax = axes[0, 0]
    if 'mean_episode_reward' in metrics_by_name:
        m = metrics_by_name['mean_episode_reward']
        ax.plot(m['updates'], m['values'], 'b-', linewidth=2, label='Mean Reward')
        ax.set_xlabel('Update')
        ax.set_ylabel('Reward')
        ax.set_title('Mean Episode Reward Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot 2: Actor and Critic Loss
    ax = axes[0, 1]
    if 'loss_actor' in metrics_by_name:
        m = metrics_by_name['loss_actor']
        ax.plot(m['updates'], m['values'], 'r-', linewidth=2, label='Actor Loss')
    if 'loss_critic' in metrics_by_name:
        m = metrics_by_name['loss_critic']
        ax.plot(m['updates'], m['values'], 'g-', linewidth=2, label='Critic Loss')
    ax.set_xlabel('Update')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    
    # Plot 3: Entropy
    ax = axes[1, 0]
    if 'entropy' in metrics_by_name:
        m = metrics_by_name['entropy']
        ax.plot(m['updates'], m['values'], 'purple', linewidth=2)
        ax.set_xlabel('Update')
        ax.set_ylabel('Entropy')
        ax.set_title('Policy Entropy (Exploration)')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Clip Fraction
    ax = axes[1, 1]
    if 'clip_fraction' in metrics_by_name:
        m = metrics_by_name['clip_fraction']
        ax.plot(m['updates'], m['values'], 'orange', linewidth=2)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Warning threshold')
        ax.set_xlabel('Update')
        ax.set_ylabel('Clip Fraction')
        ax.set_title('PPO Clip Fraction')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / 'training_metrics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()


def plot_bacteria_population(log_dir: Path, save_dir: Path = None):
    """
    Plot bacteria population statistics if available.
    
    Args:
        log_dir: Directory containing logs
        save_dir: Directory to save plots (if None, just show)
    """
    metrics_file = log_dir / "metrics.json"
    
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        return
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("No metrics data found")
        return
    
    # Look for bacteria-related metrics
    bacteria_metrics = {}
    for entry in data:
        name = entry['metric']
        if 'bacteria' in name.lower() or 'population' in name.lower():
            update = entry['update']
            value = entry['value']
            
            if name not in bacteria_metrics:
                bacteria_metrics[name] = {'updates': [], 'values': []}
            
            bacteria_metrics[name]['updates'].append(update)
            bacteria_metrics[name]['values'].append(value)
    
    if not bacteria_metrics:
        print("No bacteria population metrics found in data")
        return
    
    # Sort by update
    for name in bacteria_metrics:
        updates = bacteria_metrics[name]['updates']
        values = bacteria_metrics[name]['values']
        sorted_pairs = sorted(zip(updates, values))
        bacteria_metrics[name]['updates'] = [p[0] for p in sorted_pairs]
        bacteria_metrics[name]['values'] = [p[1] for p in sorted_pairs]
    
    # Plot bacteria metrics
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Bacteria Population Dynamics', fontsize=16, fontweight='bold')
    
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink']
    for i, (name, data_dict) in enumerate(bacteria_metrics.items()):
        color = colors[i % len(colors)]
        ax.plot(data_dict['updates'], data_dict['values'], 
               color=color, linewidth=2, label=name, marker='o', markersize=3)
    
    ax.set_xlabel('Update', fontsize=12)
    ax.set_ylabel('Population', fontsize=12)
    ax.set_title('Bacteria Population Over Training')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / 'bacteria_population.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()


def plot_all_metrics(metrics_file: Path, save_dir: Path = None):
    """
    Create an overview plot of all available metrics.
    
    Args:
        metrics_file: Path to metrics.json file
        save_dir: Directory to save plots (if None, just show)
    """
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        return
    
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("No metrics data found")
        return
    
    # Group and sort metrics
    metrics_by_name = {}
    for entry in data:
        name = entry['metric']
        update = entry['update']
        value = entry['value']
        
        if name not in metrics_by_name:
            metrics_by_name[name] = {'updates': [], 'values': []}
        
        metrics_by_name[name]['updates'].append(update)
        metrics_by_name[name]['values'].append(value)
    
    for name in metrics_by_name:
        updates = metrics_by_name[name]['updates']
        values = metrics_by_name[name]['values']
        sorted_pairs = sorted(zip(updates, values))
        metrics_by_name[name]['updates'] = [p[0] for p in sorted_pairs]
        metrics_by_name[name]['values'] = [p[1] for p in sorted_pairs]
    
    # Determine layout
    n_metrics = len(metrics_by_name)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()  # Flatten for easier indexing
    
    fig.suptitle('All Training Metrics', fontsize=16, fontweight='bold')
    
    for idx, (name, data_dict) in enumerate(metrics_by_name.items()):
        ax = axes[idx]
        ax.plot(data_dict['updates'], data_dict['values'], 'b-', linewidth=2)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Update')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(metrics_by_name), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / 'all_metrics.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        log_dir = Path(sys.argv[1])
    else:
        log_dir = Path("logs")
    
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        sys.exit(1)
    
    metrics_file = log_dir / "metrics.json"
    print(f"Plotting metrics from {metrics_file}")
    
    # Create plots and save them
    plot_training_metrics(metrics_file, save_dir=log_dir)
    plot_bacteria_population(log_dir, save_dir=log_dir)
    plot_all_metrics(metrics_file, save_dir=log_dir)
    
    print("✓ All plots saved")
