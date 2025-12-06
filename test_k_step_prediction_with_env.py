"""
Test script to load a checkpoint and visualize K-step population predictions WITH env wrapper.
This test shows how observations evolve based on NOOP actions and compares predictions with actual env counts.
"""
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl.models import RecurrentActorCritic
from rl.env_wrapper import PetriEnvWrapper
from simulation.model import BacteriaModel


def get_plot_params(k_steps):
    """
    Calculate scalable plot parameters based on k_steps.
    Returns dict with adaptive figure size, font sizes, marker sizes, etc.
    """
    # Adaptive figure size based on k_steps
    # Minimum width: 14, scales up with k_steps
    base_width = max(14, min(k_steps * 0.05, 32))  # Cap at 32
    base_height = max(8, min(k_steps * 0.03, 20))   # Cap at 20
    
    # For 2x2 subplots, multiply accordingly
    plot_width = base_width * 1.3
    plot_height = base_height * 1.6
    
    # Font sizes scale inversely with k_steps
    # Larger k = smaller fonts to fit more data
    base_font_size = max(6, min(12, 12 - (k_steps / 100)))
    title_font_size = max(8, min(14, 14 - (k_steps / 100)))
    label_font_size = max(5, min(11, 11 - (k_steps / 100)))
    
    # Marker sizes scale inversely
    # Larger k = smaller markers
    marker_size = max(2, min(9, 9 - (k_steps / 150)))
    marker_size_large = max(3, min(12, 12 - (k_steps / 150)))
    
    # Line widths scale inversely
    line_width = max(0.8, min(3, 3 - (k_steps / 200)))
    line_width_thick = max(1.2, min(3.5, 3.5 - (k_steps / 200)))
    line_width_thin = max(0.5, min(1.5, 1.5 - (k_steps / 200)))
    
    # Decide whether to show all labels or subsample
    show_all_labels = k_steps <= 50
    label_step = max(1, k_steps // 20)  # Show ~20 labels max
    
    # DPI for better quality on large plots
    dpi = 150 if k_steps <= 100 else 100
    
    return {
        'figsize_single': (base_width, base_height),
        'figsize_2x2': (plot_width, plot_height),
        'font_size': base_font_size,
        'title_font_size': title_font_size,
        'label_font_size': label_font_size,
        'marker_size': marker_size,
        'marker_size_large': marker_size_large,
        'line_width': line_width,
        'line_width_thick': line_width_thick,
        'line_width_thin': line_width_thin,
        'show_all_labels': show_all_labels,
        'label_step': label_step,
        'dpi': dpi,
        'k_steps': k_steps,
    }


def get_config_value(config, key, default=None):
    """Get value from config, handling both dict and object."""
    if hasattr(config, key):
        return getattr(config, key)
    elif isinstance(config, dict) and key in config:
        return config[key]
    return default


def load_checkpoint(checkpoint_path):
    """Load model and config from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Try different possible keys
    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        model_state = checkpoint["model_state"]
    elif "actor_critic" in checkpoint:
        model_state = checkpoint["actor_critic"]
    else:
        # Try to find any key that looks like model state
        for key in checkpoint.keys():
            if "state" in key.lower() or "model" in key.lower():
                model_state = checkpoint[key]
                break
        else:
            model_state = checkpoint
    
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        config = {}
    
    return config, model_state


def create_model_from_config(config, device="cpu"):
    """Create RecurrentActorCritic model from config."""
    model = RecurrentActorCritic(
        obs_dim=get_config_value(config, "obs_dim", 28),
        n_discrete=get_config_value(config, "n_discrete", 4),
        k_doses=get_config_value(config, "k_doses", 3),
        hidden_dim=get_config_value(config, "hidden_dim", 256),
        rnn_layers=get_config_value(config, "rnn_layers", 1),
        dose_action_index=get_config_value(config, "dose_action_index", 3),
        sigmoid_scale_factor=get_config_value(config, "sigmoid_scale_factor", 0.2),
    ).to(device)
    return model


def create_dummy_mesa_model():
    """Create a dummy Mesa model for testing."""
    from simulation.model import BacteriaModel
    return BacteriaModel(N=400)


def test_k_step_with_env_wrapper(checkpoint_path, k_steps=10, device="cpu"):
    """
    Test K-step population prediction WITH env wrapper.
    This shows how predictions compare to actual environment evolution.
    
    Args:
        checkpoint_path: Path to checkpoint file
        k_steps: Number of steps to predict ahead
        device: Device to run on
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    config, model_state = load_checkpoint(checkpoint_path)
    
    # Load model
    model = create_model_from_config(config, device=device)
    model.load_state_dict(model_state)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Config: obs_dim={get_config_value(config, 'obs_dim')}, hidden_dim={get_config_value(config, 'hidden_dim')}")
    
    # Create env wrapper
    print("\nCreating environment wrapper...")
    try:
        mesa_factory = create_dummy_mesa_model
        # max_steps must be >= k_steps for prediction to work
        env = PetriEnvWrapper(
            mesa_model_factory=mesa_factory,
            k_doses=get_config_value(config, "k_doses", 3),
            max_steps=max(100, k_steps + 10),  # Add buffer
            target_population=100,
            initial_bacteria_per_type_range=[300,500],
            early_termination_enabled=False
        )
        obs = env.reset()
        print(f"Environment initialized. Observation shape: {obs.shape}, max_steps={max(100, k_steps + 10)}")
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Falling back to test without env wrapper")
        obs = np.random.rand(28).astype(np.float32)
        obs = np.clip(obs, 0, 1)
        env = None
    
    # Convert to torch
    obs_torch = torch.from_numpy(obs).unsqueeze(0).to(device, dtype=torch.float32)
    batch_size = 1
    
    # Initialize hidden state
    h_current = model.init_hidden(batch_size=batch_size, device=device)
    
    # Create dummy actions
    a_disc = torch.tensor([3], device=device, dtype=torch.long)  # DOSE action
    a_cont = torch.full((batch_size, get_config_value(config, "k_doses", 3)), 0.1, device=device, dtype=torch.float32)
    
    print(f"\nRunning K-step prediction with k={k_steps}")
    print(f"Initial action: DOSE with doses={a_cont.cpu().numpy()}")
    print("WITH env wrapper (observations evolve, collecting actual counts)")
    
    # Predict K steps ahead
    with torch.no_grad():
        predictions = model.predict_k_steps_ahead(
            obs=obs_torch,
            a_disc=a_disc,
            a_cont=a_cont,
            h_current=h_current,
            k_steps=k_steps,
            env_wrapper=env,
        )
    
    predictions_np = predictions.cpu().numpy().squeeze()  # [k_steps]
    
    # Now collect actual counts from environment by stepping through
    print(f"\nCollecting actual population counts from environment...")
    actual_counts = []
    
    if env is not None:
        try:
            # Step 0: Get initial population before DOSE
            initial_pop = env._read_true_population()
            actual_counts.append(initial_pop)
            print(f"  Step 0 (before DOSE): actual population={initial_pop:.1f}")
            
            # Step 1: Apply DOSE action
            a_disc_val = int(a_disc[0].cpu())
            a_cont_val = a_cont[0].cpu().numpy()
            next_obs, _, _, info = env.step(a_disc_val, a_cont_val)
            actual_count = info.get('actual_population', -1)
            actual_counts.append(actual_count)
            print(f"  Step 1 (DOSE): actual population={actual_count:.1f}")
            
            # Steps 2-k: Apply NOOP action (index 0)
            for step in range(2, k_steps + 1):
                noop_doses = np.zeros(get_config_value(config, "k_doses", 3))
                next_obs, _, _, info = env.step(0, noop_doses)
                actual_count = info.get('actual_population', -1)
                actual_counts.append(actual_count)
                print(f"  Step {step} (NOOP): actual population={actual_count:.1f}")
        except Exception as e:
            print(f"  Error at step ~{len(actual_counts)}: {type(e).__name__}: {str(e)[:200]}")
            import traceback
            traceback.print_exc()
            actual_counts = None
    else:
        actual_counts = None
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Predictions vs Actual Population:")
    
    # Include Step 0 (initial, before DOSE)
    if actual_counts and len(actual_counts) > 0:
        initial_actual = actual_counts[0]
        print(f"  Step 0 (before action): actual={initial_actual:.1f}, model will predict from this")
    
    for i, pred in enumerate(predictions_np, 1):
        pred_pop = pred * 300  # Denormalize
        if actual_counts and i < len(actual_counts):
            actual = actual_counts[i]
            diff = pred_pop - actual
            error_pct = (abs(diff) / actual * 100) if actual > 0 else 0
            print(f"  Step {i}: predicted={pred_pop:.1f}, actual={actual:.1f}, diff={diff:.1f} ({error_pct:.1f}%)")
        else:
            print(f"  Step {i}: predicted={pred_pop:.1f}, actual=N/A")
    
    # Plot predictions vs actual
    plot_params = get_plot_params(k_steps)
    fig, ax = plt.subplots(1, 1, figsize=plot_params['figsize_single'])
    
    # Separate Step 0 (initial) from Step 1+ (predictions)
    # Step 0 is not predicted by model, it's the starting point
    steps_predicted = np.arange(1, k_steps + 1)
    predictions_actual = predictions_np * 300  # Denormalize to actual population
    
    # Plot Step 0 (initial state)
    if actual_counts and len(actual_counts) > 0:
        ax.plot([0], [actual_counts[0]], 'ko-', linewidth=plot_params['line_width'], markersize=plot_params['marker_size_large'], 
                label='Initial Population (Step 0)', alpha=0.9, zorder=5)
        # Only show Step 0 label always
        ax.text(0, actual_counts[0] + 2, f'{actual_counts[0]:.1f}', ha='center', va='bottom', 
                fontsize=plot_params['label_font_size'], fontweight='bold', color='black')
    
    # Plot predictions
    ax.plot(steps_predicted, predictions_actual, 'g-o', linewidth=plot_params['line_width_thick'], 
            markersize=plot_params['marker_size'], label='Model Prediction (with evolving obs)', alpha=0.85)
    
    # Plot actual counts if available
    if actual_counts:
        actual_counts_arr = np.array(actual_counts[1:k_steps+1])  # Steps 1-10
        ax.plot(steps_predicted, actual_counts_arr, 'r-s', linewidth=plot_params['line_width_thick'], 
                markersize=plot_params['marker_size'], label='Actual Population (from env)', alpha=0.85)
        
        # Add value labels for subsampled steps if k_steps is large
        if plot_params['show_all_labels']:
            for i, (step, val) in enumerate(zip(steps_predicted, actual_counts_arr)):
                ax.text(step, val - 3, f'{val:.0f}', ha='center', va='top', fontsize=plot_params['label_font_size']*0.8, alpha=0.7)
        else:
            # Show every label_step-th label
            for i in range(0, len(steps_predicted), plot_params['label_step']):
                if i < len(actual_counts_arr):
                    step = steps_predicted[i]
                    val = actual_counts_arr[i]
                    ax.text(step, val - 3, f'{val:.0f}', ha='center', va='top', fontsize=plot_params['label_font_size']*0.8, alpha=0.7)
    
    ax.set_xlabel('Steps Ahead', fontsize=plot_params['label_font_size'], fontweight='bold')
    ax.set_ylabel('Population (actual bacteria count)', fontsize=plot_params['label_font_size'], fontweight='bold')
    ax.set_title(f'K-Step Predictions (k={k_steps}): Model vs Actual Environment', fontsize=plot_params['title_font_size'], fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=plot_params['font_size'], loc='best')
    
    # Adaptive x-axis: show all ticks for small k, subsample for large k
    if k_steps <= 50:
        ax.set_xticks(steps_predicted)
    else:
        # Show ~20 ticks max
        tick_step = max(1, k_steps // 20)
        ax.set_xticks(steps_predicted[::tick_step])
    
    # Rotate x-labels for better readability if k_steps is large
    if k_steps > 50:
        ax.tick_params(axis='x', rotation=45)
    
    # Adaptive value labels - only for small k or subsampled for large k
    if plot_params['show_all_labels']:
        # Add value labels on predicted points for small k
        for i, (step, pred) in enumerate(zip(steps_predicted, predictions_actual)):
            ax.text(step, pred + 1.5, f'{pred:.0f}', ha='center', va='bottom', fontsize=plot_params['label_font_size']*0.7, color='green', alpha=0.8)
        
        if actual_counts:
            for i, (step, actual) in enumerate(zip(steps_predicted, actual_counts_arr)):
                ax.text(step, actual - 2.5, f'{actual:.0f}', ha='center', va='top', fontsize=plot_params['label_font_size']*0.7, color='red', alpha=0.8)
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "k_step_prediction_with_env_actual.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.close()
    
    return predictions_np, actual_counts, env


def compare_with_and_without_env(checkpoint_path, k_steps=10, device="cpu"):
    """
    Compare K-step predictions with and without env wrapper, showing actual environment counts.
    """
    print("\n" + "=" * 70)
    print("Comparing predictions WITH and WITHOUT env wrapper")
    print("=" * 70)
    
    config, model_state = load_checkpoint(checkpoint_path)
    model = create_model_from_config(config, device=device)
    model.load_state_dict(model_state)
    model.eval()
    
    obs_dim = get_config_value(config, "obs_dim", 28)
    batch_size = 1
    
    # Create environment to get real observation (same as test_k_step_with_env_wrapper)
    print("\nCreating environment wrapper for comparison...")
    try:
        mesa_factory = create_dummy_mesa_model
        env_for_obs = PetriEnvWrapper(
            mesa_model_factory=mesa_factory,
            k_doses=get_config_value(config, "k_doses", 3),
            max_steps=100,
            target_population=100,
            initial_bacteria_per_type_range=[300,500],
            early_termination_enabled=False

        )
        obs = env_for_obs.reset()
        obs_torch = torch.from_numpy(obs).unsqueeze(0).to(device, dtype=torch.float32)
        print(f"Real observation from environment initialized. Observation shape: {obs.shape}")
    except Exception as e:
        print(f"Error creating environment: {e}")
        # Fallback to fixed observation
        torch.manual_seed(42)
        obs_torch = torch.randn(batch_size, obs_dim, device=device, dtype=torch.float32)
        obs_torch = torch.clamp(obs_torch, 0, 1)
        env_for_obs = None
    
    h_current = model.init_hidden(batch_size=batch_size, device=device)
    
    # Create dummy actions
    a_disc = torch.tensor([3], device=device, dtype=torch.long)  # DOSE
    a_cont = torch.full((batch_size, get_config_value(config, "k_doses", 3)), 0.1, device=device, dtype=torch.float32)
    
    print(f"\nTest 1: WITHOUT env wrapper (fixed observation)")
    with torch.no_grad():
        predictions_without = model.predict_k_steps_ahead(
            obs=obs_torch,
            a_disc=a_disc,
            a_cont=a_cont,
            h_current=h_current,
            k_steps=k_steps,
            env_wrapper=None,
        )
    predictions_without_np = predictions_without.cpu().numpy().squeeze()
    
    print(f"Predictions (fixed observation): {predictions_without_np[:5]}...")
    
    # Reset hidden state for fair comparison
    h_current = model.init_hidden(batch_size=batch_size, device=device)
    
    print(f"\nTest 2: WITH env wrapper (evolving observation)")
    actual_counts = None
    env = env_for_obs  # Reuse the same environment
    try:
        if env is not None:
            with torch.no_grad():
                predictions_with = model.predict_k_steps_ahead(
                    obs=obs_torch,
                    a_disc=a_disc,
                    a_cont=a_cont,
                    h_current=h_current,
                    k_steps=k_steps,
                    env_wrapper=env,
                )
            predictions_with_np = predictions_with.cpu().numpy().squeeze()
            print(f"Predictions (evolving observation): {predictions_with_np[:5]}...")
            
            # Now collect actual counts from the SAME environment
            print(f"\nCollecting actual environment counts for WITH env wrapper...")
            actual_counts = []
            try:
                # Step 0: Get initial population before any action
                initial_pop = env._read_true_population()
                actual_counts.append(initial_pop)
                print(f"  Step 0 (before action): actual={initial_pop:.1f}")
                
                # Step 1: DOSE
                a_disc_val = int(a_disc[0].cpu())
                a_cont_val = a_cont[0].cpu().numpy()
                next_obs, _, _, info = env.step(a_disc_val, a_cont_val)
                actual_count = info.get('actual_population', -1)
                actual_counts.append(actual_count)
                print(f"  Step 1 (DOSE): actual={actual_count:.1f}")
                
                # Steps 2-k: NOOP
                for step in range(2, k_steps + 1):
                    noop_doses = np.zeros(get_config_value(config, "k_doses", 3))
                    next_obs, _, _, info = env.step(0, noop_doses)
                    actual_count = info.get('actual_population', -1)
                    actual_counts.append(actual_count)
                    print(f"  Step {step} (NOOP): actual={actual_count:.1f}")
            except Exception as e:
                print(f"  Error at step ~{len(actual_counts)}: {type(e).__name__}: {str(e)[:200]}")
                import traceback
                traceback.print_exc()
                actual_counts = None
        else:
            predictions_with_np = predictions_without_np
            print("Could not create env wrapper, using fixed observation for both")
    except Exception as e:
        print(f"Error with env wrapper: {e}")
        predictions_with_np = predictions_without_np
    
    # Plot comparison
    plot_params = get_plot_params(k_steps)
    fig, axes = plt.subplots(2, 2, figsize=plot_params['figsize_2x2'])
    
    steps_predicted = np.arange(1, k_steps + 1)
    
    # Denormalize to actual population values
    predictions_without_actual = predictions_without_np * 300
    predictions_with_actual = predictions_with_np * 300
    
    # Plot 1: Predictions overlay (actual population) - EMPHASIS on evolving (trained) prediction
    # Plot Step 0 (initial) first
    if actual_counts and len(actual_counts) > 0:
        axes[0, 0].plot([0], [actual_counts[0]], 'ko-', linewidth=plot_params['line_width'], 
                        markersize=plot_params['marker_size_large'], label='Initial (Step 0)', alpha=0.9, zorder=5)
    
    # Plot actual first (background)
    if actual_counts:
        actual_counts_arr = np.array(actual_counts[1:k_steps+1])
        axes[0, 0].plot(steps_predicted, actual_counts_arr, 'r-^', linewidth=plot_params['line_width_thick'], 
                        markersize=plot_params['marker_size'], label='Actual Env Population', alpha=0.75)
    
    # Plot fixed observation (secondary)
    axes[0, 0].plot(steps_predicted, predictions_without_actual, 'b--o', linewidth=plot_params['line_width_thin'], 
                    markersize=plot_params['marker_size']*0.7, label='Fixed Observation', alpha=0.6, linestyle='--')
    
    # Plot evolving observation PROMINENTLY (this is what the model learned)
    axes[0, 0].plot(steps_predicted, predictions_with_actual, 'g-o', linewidth=plot_params['line_width_thick'], 
                    markersize=plot_params['marker_size'], label='Model Prediction (Evolving Obs - TRAINED)', alpha=0.95, zorder=10)
    
    if actual_counts:
        # Add dashed vertical line after step 0 to show "before action" boundary
        axes[0, 0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        axes[0, 0].text(0.2, axes[0, 0].get_ylim()[1]*0.95, 'DOSE→', fontsize=plot_params['label_font_size'], 
                       style='italic', color='gray', fontweight='bold')
    
    # Adaptive x-axis
    if k_steps <= 50:
        axes[0, 0].set_xticks(steps_predicted)
    else:
        tick_step = max(1, k_steps // 20)
        axes[0, 0].set_xticks(steps_predicted[::tick_step])
    if k_steps > 50:
        axes[0, 0].tick_params(axis='x', rotation=45)
    
    axes[0, 0].set_xlabel('Steps Ahead', fontsize=plot_params['label_font_size'], fontweight='bold')
    axes[0, 0].set_ylabel('Population (bacteria count)', fontsize=plot_params['label_font_size'], fontweight='bold')
    axes[0, 0].set_title(f'All Methods: Actual vs Predictions (k={k_steps})', fontsize=plot_params['title_font_size'], fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=plot_params['font_size']*0.9, loc='best')
    
    # Plot 2: Prediction difference (Without vs With env)
    diff_pred = predictions_with_actual - predictions_without_actual
    colors = ['red' if d < 0 else 'green' for d in diff_pred]
    axes[0, 1].bar(steps_predicted, diff_pred, color=colors, alpha=0.7, edgecolor='black', linewidth=0.8)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Adaptive x-axis
    if k_steps <= 50:
        axes[0, 1].set_xticks(steps_predicted)
    else:
        tick_step = max(1, k_steps // 20)
        axes[0, 1].set_xticks(steps_predicted[::tick_step])
    if k_steps > 50:
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    axes[0, 1].set_xlabel('Steps Ahead', fontsize=plot_params['label_font_size'], fontweight='bold')
    axes[0, 1].set_ylabel('Difference (bacteria count)', fontsize=plot_params['label_font_size'], fontweight='bold')
    axes[0, 1].set_title(f'Prediction Difference: (Evolving - Fixed) (k={k_steps})', fontsize=plot_params['title_font_size'], fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels only for smaller k
    if plot_params['show_all_labels']:
        for i, (step, d) in enumerate(zip(steps_predicted, diff_pred)):
            axes[0, 1].text(step, d + 0.3 if d >= 0 else d - 0.3, f'{d:.0f}', ha='center', va='bottom' if d >= 0 else 'top', 
                           fontsize=plot_params['label_font_size']*0.7)
    else:
        # Show every label_step-th label
        for i in range(0, len(steps_predicted), plot_params['label_step']):
            if i < len(diff_pred):
                step = steps_predicted[i]
                d = diff_pred[i]
                axes[0, 1].text(step, d + 0.3 if d >= 0 else d - 0.3, f'{d:.0f}', ha='center', va='bottom' if d >= 0 else 'top',
                               fontsize=plot_params['label_font_size']*0.7)
    
    # Plot 3: Prediction vs Actual for WITH env wrapper (TRAINED MODEL)
    if actual_counts:
        # Plot Step 0 (initial)
        axes[1, 0].plot([0], [actual_counts[0]], 'ko-', linewidth=plot_params['line_width'], 
                        markersize=plot_params['marker_size_large'], label='Initial (Step 0)', alpha=0.9, zorder=5)
        
        actual_counts_arr = np.array(actual_counts[1:k_steps+1])
        # Plot actual first (background reference)
        axes[1, 0].plot(steps_predicted, actual_counts_arr, 'r-s', linewidth=plot_params['line_width_thick'], 
                        markersize=plot_params['marker_size'], label='Actual Env Count', alpha=0.75)
        # Plot model prediction prominently
        axes[1, 0].plot(steps_predicted, predictions_with_actual, 'g-o', linewidth=plot_params['line_width_thick'], 
                        markersize=plot_params['marker_size'], label='Model Prediction (Evolving Obs - TRAINED)', alpha=0.95, zorder=10)
        
        # Adaptive x-axis
        if k_steps <= 50:
            axes[1, 0].set_xticks(steps_predicted)
        else:
            tick_step = max(1, k_steps // 20)
            axes[1, 0].set_xticks(steps_predicted[::tick_step])
        if k_steps > 50:
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        axes[1, 0].set_xlabel('Steps Ahead', fontsize=plot_params['label_font_size'], fontweight='bold')
        axes[1, 0].set_ylabel('Population (bacteria count)', fontsize=plot_params['label_font_size'], fontweight='bold')
        axes[1, 0].set_title(f'Trained Model: Predictions vs Actual (k={k_steps})', fontsize=plot_params['title_font_size'], fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(fontsize=plot_params['font_size']*0.9, loc='best')
        
        # Error between WITH prediction and actual
        error_with = predictions_with_actual - actual_counts_arr
        colors_err = ['red' if e < 0 else 'green' for e in error_with]
        axes[1, 1].bar(steps_predicted, error_with, color=colors_err, alpha=0.7, edgecolor='black', linewidth=0.8)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Adaptive x-axis
        if k_steps <= 50:
            axes[1, 1].set_xticks(steps_predicted)
        else:
            tick_step = max(1, k_steps // 20)
            axes[1, 1].set_xticks(steps_predicted[::tick_step])
        if k_steps > 50:
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        axes[1, 1].set_xlabel('Steps Ahead', fontsize=plot_params['label_font_size'], fontweight='bold')
        axes[1, 1].set_ylabel('Prediction Error (bacteria count)', fontsize=plot_params['label_font_size'], fontweight='bold')
        axes[1, 1].set_title(f'Model Prediction Error (k={k_steps})', fontsize=plot_params['title_font_size'], fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels only for smaller k
        if plot_params['show_all_labels']:
            for i, (step, e) in enumerate(zip(steps_predicted, error_with)):
                pct_err = (abs(e) / actual_counts_arr[i] * 100) if actual_counts_arr[i] > 0 else 0
                axes[1, 1].text(step, e + 0.3 if e >= 0 else e - 0.3, f'{e:.0f}\n({pct_err:.0f}%)', ha='center', va='bottom' if e >= 0 else 'top', 
                               fontsize=plot_params['label_font_size']*0.6)
        else:
            # Show every label_step-th label
            for i in range(0, len(steps_predicted), plot_params['label_step']):
                if i < len(error_with):
                    step = steps_predicted[i]
                    e = error_with[i]
                    pct_err = (abs(e) / actual_counts_arr[i] * 100) if actual_counts_arr[i] > 0 else 0
                    axes[1, 1].text(step, e + 0.3 if e >= 0 else e - 0.3, f'{e:.0f}\n({pct_err:.0f}%)', ha='center', va='bottom' if e >= 0 else 'top',
                                   fontsize=plot_params['label_font_size']*0.6)
    else:
        # Just show difference if no actual counts
        axes[1, 0].text(0.5, 0.5, 'No actual environment counts collected', ha='center', va='center', transform=axes[1, 0].transAxes, 
                       fontsize=plot_params['label_font_size'])
        axes[1, 1].text(0.5, 0.5, 'No actual environment counts collected', ha='center', va='center', transform=axes[1, 1].transAxes, 
                       fontsize=plot_params['label_font_size'])
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "k_step_prediction_with_vs_without_env.png"
    plt.savefig(output_path, dpi=plot_params['dpi'], bbox_inches='tight')
    print(f"\nComparison plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    checkpoint_path = Path(__file__).parent / "src" / "checkpoints" / "new_expression_computation" / "checkpoint_1000.pt"
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("Test 1: K-step prediction WITH env wrapper (DOSE action)")
    print("=" * 70)
    test_k_step_with_env_wrapper(checkpoint_path, k_steps=100, device="cpu")
    
    print("\n" + "=" * 70)
    print("Test 2: Comparison of WITH vs WITHOUT env wrapper (DOSE action)")
    print("=" * 70)
    compare_with_and_without_env(checkpoint_path, k_steps=100, device="cpu")


def test_k_step_with_noop_action(checkpoint_path, k_steps=50, device="cpu"):
    """
    Test K-step predictions with NOOP action (index 0).
    NOOP doesn't apply antibiotics, so population should grow naturally.
    """
    print(f"\nTesting k-step predictions with NOOP action (k={k_steps})...")
    
    # Load checkpoint (returns config, model_state)
    config, model_state = load_checkpoint(checkpoint_path)
    obs_dim = get_config_value(config, "obs_dim", 47)
    
    # Create model
    model = RecurrentActorCritic(
        obs_dim=obs_dim,
        n_discrete=4,
        k_doses=3,
        hidden_dim=256,
        rnn_layers=2,
    ).to(device)
    
    model.load_state_dict(model_state)
    model.eval()
    print(f"✓ Model loaded from checkpoint")
    
    # Create environment
    env = PetriEnvWrapper(
        mesa_model_factory=lambda: BacteriaModel(),
        k_doses=3,
        max_steps=max(100, k_steps + 10),
    )
    print(f"✓ Environment created (k_steps_ahead will be capped at {max(100, k_steps + 10)})")
    
    # Reset environment
    obs = env.reset()
    print(f"✓ Environment reset")
    
    # Prepare observation tensor
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)  # [1, 47]
    
    # Get initial hidden state
    h_init = torch.zeros(2, 1, 256).to(device)
    
    # Action: NOOP (discrete action 0)
    a_disc = torch.tensor([0], dtype=torch.long).to(device)  # NOOP
    a_cont = torch.zeros(1, 3).to(device)
    
    print(f"\nPredicting {k_steps} steps ahead with NOOP action...")
    
    with torch.no_grad():
        # Predict k steps ahead
        predictions = model.predict_k_steps_ahead(
            obs=obs_tensor,
            a_disc=a_disc,
            a_cont=a_cont,
            h_current=h_init,
            k_steps=k_steps,
            env_wrapper=env
        )
    
    predictions_np = predictions.cpu().numpy().squeeze()
    
    # Collect actual counts by stepping with NOOP
    print(f"\nCollecting actual population counts with NOOP action...")
    actual_counts = []
    
    try:
        initial_pop = env._read_true_population()
        actual_counts.append(initial_pop)
        print(f"  Step 0 (initial): population={initial_pop:.1f}")
        
        # Apply NOOP for k steps
        for step in range(1, k_steps + 1):
            noop_doses = np.zeros(3)
            next_obs, _, _, info = env.step(0, noop_doses)
            actual_pop = info.get('actual_population', -1)
            actual_counts.append(actual_pop)
            if step % 10 == 0 or step < 5:
                print(f"  Step {step} (NOOP): population={actual_pop:.1f}")
    except Exception as e:
        print(f"  Error at step ~{len(actual_counts)}: {type(e).__name__}: {str(e)[:100]}")
        actual_counts = None
    
    # Print comparison
    print(f"\n{'Step':<6} {'Predicted':<15} {'Actual':<15} {'Error':<15}")
    print("-" * 55)
    for i, pred in enumerate(predictions_np):
        pred_pop = pred * 500  # Denormalize (using population_norm=500)
        if actual_counts and i < len(actual_counts):
            actual = actual_counts[i]
            error = pred_pop - actual
            print(f"{i:<6} {pred_pop:<15.1f} {actual:<15.1f} {error:<15.1f}")
        else:
            print(f"{i:<6} {pred_pop:<15.1f} {'N/A':<15} {'N/A':<15}")
    
    # Plot predictions vs actual
    plot_params = get_plot_params(k_steps)
    fig, ax = plt.subplots(1, 1, figsize=plot_params['figsize_single'])
    
    steps_predicted = np.arange(1, k_steps + 1)
    predictions_actual = predictions_np * 500  # Denormalize
    
    # Plot Step 0 (initial state)
    if actual_counts and len(actual_counts) > 0:
        ax.plot([0], [actual_counts[0]], 'ko-', linewidth=plot_params['line_width'], markersize=plot_params['marker_size_large'], 
                label='Initial Population (Step 0)', alpha=0.9, zorder=5)
    
    # Plot predictions (match dimensions: predictions_np has k_steps elements, steps_predicted is 1..k_steps)
    ax.plot(steps_predicted, predictions_actual[:len(steps_predicted)], 'g-o', linewidth=plot_params['line_width_thick'], 
            markersize=plot_params['marker_size'], label='Model Prediction (NOOP actions)', alpha=0.85)
    
    # Plot actual counts if available (actual_counts includes step 0, so indices 1..k_steps match steps 1..k_steps)
    if actual_counts and len(actual_counts) > 1:
        actual_counts_arr = np.array(actual_counts[1:min(k_steps+1, len(actual_counts))])
        ax.plot(steps_predicted[:len(actual_counts_arr)], actual_counts_arr, 'r-s', linewidth=plot_params['line_width_thick'], 
                markersize=plot_params['marker_size'], label='Actual Population (from env)', alpha=0.85)
    
    ax.set_xlabel('Steps Ahead', fontsize=plot_params['label_font_size'], fontweight='bold')
    ax.set_ylabel('Population (actual bacteria count)', fontsize=plot_params['label_font_size'], fontweight='bold')
    ax.set_title(f'K-Step Predictions with NOOP Action (k={k_steps}): Model vs Actual', fontsize=plot_params['title_font_size'], fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=plot_params['font_size'], loc='best')
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "k_step_prediction_noop_action.png"
    plt.savefig(output_path, dpi=plot_params['dpi'], bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_path}")
    plt.close()
    
    print(f"\n✓ NOOP action test completed")


def test_k_step_with_count_action(checkpoint_path, k_steps=50, device="cpu"):
    """
    Test K-step predictions where first action is COUNT (index 1).
    COUNT takes time and doesn't immediately affect population.
    """
    print(f"\nTesting k-step predictions with COUNT action (k={k_steps})...")
    
    # Load checkpoint (returns config, model_state)
    config, model_state = load_checkpoint(checkpoint_path)
    obs_dim = get_config_value(config, "obs_dim", 47)
    
    # Create model
    model = RecurrentActorCritic(
        obs_dim=obs_dim,
        n_discrete=4,
        k_doses=3,
        hidden_dim=256,
        rnn_layers=2,
    ).to(device)
    
    model.load_state_dict(model_state)
    model.eval()
    print(f"✓ Model loaded from checkpoint")
    
    # Create environment
    env = PetriEnvWrapper(
        mesa_model_factory=lambda: BacteriaModel(),
        k_doses=3,
        max_steps=max(100, k_steps + 10),
    )
    print(f"✓ Environment created")
    
    # Reset environment
    obs = env.reset()
    print(f"✓ Environment reset")
    
    # Prepare observation tensor
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
    
    # Get initial hidden state
    h_init = torch.zeros(2, 1, 256).to(device)
    
    # Action: COUNT (discrete action 1)
    a_disc = torch.tensor([1], dtype=torch.long).to(device)  # COUNT
    a_cont = torch.zeros(1, 3).to(device)
    
    print(f"\nPredicting {k_steps} steps ahead with COUNT action followed by NOOPs...")
    
    with torch.no_grad():
        predictions = model.predict_k_steps_ahead(
            obs=obs_tensor,
            a_disc=a_disc,
            a_cont=a_cont,
            h_current=h_init,
            k_steps=k_steps,
            env_wrapper=env
        )
    
    predictions_np = predictions.cpu().numpy().squeeze()
    
    # Collect actual counts
    print(f"\nCollecting actual population counts (COUNT + NOOPs)...")
    actual_counts = []
    
    try:
        initial_pop = env._read_true_population()
        actual_counts.append(initial_pop)
        print(f"  Step 0 (initial): population={initial_pop:.1f}")
        
        # Step 1: COUNT action
        noop_doses = np.zeros(3)
        next_obs, _, _, info = env.step(1, noop_doses)  # COUNT
        actual_pop = info.get('actual_population', -1)
        actual_counts.append(actual_pop)
        print(f"  Step 1 (COUNT): population={actual_pop:.1f}")
        
        # Steps 2+: NOOP
        for step in range(2, k_steps + 1):
            next_obs, _, _, info = env.step(0, noop_doses)
            actual_pop = info.get('actual_population', -1)
            actual_counts.append(actual_pop)
            if step % 10 == 0 or step < 5:
                print(f"  Step {step} (NOOP): population={actual_pop:.1f}")
    except Exception as e:
        print(f"  Error at step ~{len(actual_counts)}: {type(e).__name__}: {str(e)[:100]}")
        actual_counts = None
    
    # Print comparison
    print(f"\n{'Step':<6} {'Predicted':<15} {'Actual':<15} {'Error':<15}")
    print("-" * 55)
    for i, pred in enumerate(predictions_np):
        pred_pop = pred * 500
        if actual_counts and i < len(actual_counts):
            actual = actual_counts[i]
            error = pred_pop - actual
            print(f"{i:<6} {pred_pop:<15.1f} {actual:<15.1f} {error:<15.1f}")
        else:
            print(f"{i:<6} {pred_pop:<15.1f} {'N/A':<15} {'N/A':<15}")
    
    # Plot predictions vs actual
    plot_params = get_plot_params(k_steps)
    fig, ax = plt.subplots(1, 1, figsize=plot_params['figsize_single'])
    
    steps_predicted = np.arange(1, k_steps + 1)
    predictions_actual = predictions_np * 500  # Denormalize
    
    # Plot Step 0 (initial state)
    if actual_counts and len(actual_counts) > 0:
        ax.plot([0], [actual_counts[0]], 'ko-', linewidth=plot_params['line_width'], markersize=plot_params['marker_size_large'], 
                label='Initial Population (Step 0)', alpha=0.9, zorder=5)
    
    # Plot predictions (match dimensions: predictions_np has k_steps elements, steps_predicted is 1..k_steps)
    ax.plot(steps_predicted, predictions_actual[:len(steps_predicted)], 'b-o', linewidth=plot_params['line_width_thick'], 
            markersize=plot_params['marker_size'], label='Model Prediction (COUNT + NOOPs)', alpha=0.85)
    
    # Plot actual counts if available (actual_counts includes step 0, so indices 1..k_steps match steps 1..k_steps)
    if actual_counts and len(actual_counts) > 1:
        actual_counts_arr = np.array(actual_counts[1:min(k_steps+1, len(actual_counts))])
        ax.plot(steps_predicted[:len(actual_counts_arr)], actual_counts_arr, 'r-s', linewidth=plot_params['line_width_thick'], 
                markersize=plot_params['marker_size'], label='Actual Population (from env)', alpha=0.85)
    
    ax.set_xlabel('Steps Ahead', fontsize=plot_params['label_font_size'], fontweight='bold')
    ax.set_ylabel('Population (actual bacteria count)', fontsize=plot_params['label_font_size'], fontweight='bold')
    ax.set_title(f'K-Step Predictions with COUNT Action (k={k_steps}): Model vs Actual', fontsize=plot_params['title_font_size'], fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=plot_params['font_size'], loc='best')
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "k_step_prediction_count_action.png"
    plt.savefig(output_path, dpi=plot_params['dpi'], bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_path}")
    plt.close()
    
    print(f"\n✓ COUNT action test completed")


def test_k_step_with_seq_action(checkpoint_path, k_steps=50, device="cpu"):
    """
    Test K-step predictions where first action is SEQ (index 2).
    SEQ (sequencing) takes time and doesn't affect population directly.
    """
    print(f"\nTesting k-step predictions with SEQ action (k={k_steps})...")
    
    # Load checkpoint (returns config, model_state)
    config, model_state = load_checkpoint(checkpoint_path)
    obs_dim = get_config_value(config, "obs_dim", 47)
    
    # Create model
    model = RecurrentActorCritic(
        obs_dim=obs_dim,
        n_discrete=4,
        k_doses=3,
        hidden_dim=256,
        rnn_layers=2,
    ).to(device)
    
    model.load_state_dict(model_state)
    model.eval()
    print(f"✓ Model loaded from checkpoint")
    
    # Create environment
    env = PetriEnvWrapper(
        mesa_model_factory=lambda: BacteriaModel(),
        k_doses=3,
        max_steps=max(100, k_steps + 10),
    )
    print(f"✓ Environment created")
    
    # Reset environment
    obs = env.reset()
    print(f"✓ Environment reset")
    
    # Prepare observation tensor
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
    
    # Get initial hidden state
    h_init = torch.zeros(2, 1, 256).to(device)
    
    # Action: SEQ (discrete action 2)
    a_disc = torch.tensor([2], dtype=torch.long).to(device)  # SEQ
    a_cont = torch.zeros(1, 3).to(device)
    
    print(f"\nPredicting {k_steps} steps ahead with SEQ action followed by NOOPs...")
    
    with torch.no_grad():
        predictions = model.predict_k_steps_ahead(
            obs=obs_tensor,
            a_disc=a_disc,
            a_cont=a_cont,
            h_current=h_init,
            k_steps=k_steps,
            env_wrapper=env
        )
    
    predictions_np = predictions.cpu().numpy().squeeze()
    
    # Collect actual counts
    print(f"\nCollecting actual population counts (SEQ + NOOPs)...")
    actual_counts = []
    
    try:
        initial_pop = env._read_true_population()
        actual_counts.append(initial_pop)
        print(f"  Step 0 (initial): population={initial_pop:.1f}")
        
        # Step 1: SEQ action
        noop_doses = np.zeros(3)
        next_obs, _, _, info = env.step(2, noop_doses)  # SEQ
        actual_pop = info.get('actual_population', -1)
        actual_counts.append(actual_pop)
        print(f"  Step 1 (SEQ): population={actual_pop:.1f}")
        
        # Steps 2+: NOOP
        for step in range(2, k_steps + 1):
            next_obs, _, _, info = env.step(0, noop_doses)
            actual_pop = info.get('actual_population', -1)
            actual_counts.append(actual_pop)
            if step % 10 == 0 or step < 5:
                print(f"  Step {step} (NOOP): population={actual_pop:.1f}")
    except Exception as e:
        print(f"  Error at step ~{len(actual_counts)}: {type(e).__name__}: {str(e)[:100]}")
        actual_counts = None
    
    # Print comparison
    print(f"\n{'Step':<6} {'Predicted':<15} {'Actual':<15} {'Error':<15}")
    print("-" * 55)
    for i, pred in enumerate(predictions_np):
        pred_pop = pred * 500
        if actual_counts and i < len(actual_counts):
            actual = actual_counts[i]
            error = pred_pop - actual
            print(f"{i:<6} {pred_pop:<15.1f} {actual:<15.1f} {error:<15.1f}")
        else:
            print(f"{i:<6} {pred_pop:<15.1f} {'N/A':<15} {'N/A':<15}")
    
    # Plot predictions vs actual
    plot_params = get_plot_params(k_steps)
    fig, ax = plt.subplots(1, 1, figsize=plot_params['figsize_single'])
    
    steps_predicted = np.arange(1, k_steps + 1)
    predictions_actual = predictions_np * 500  # Denormalize
    
    # Plot Step 0 (initial state)
    if actual_counts and len(actual_counts) > 0:
        ax.plot([0], [actual_counts[0]], 'ko-', linewidth=plot_params['line_width'], markersize=plot_params['marker_size_large'], 
                label='Initial Population (Step 0)', alpha=0.9, zorder=5)
    
    # Plot predictions (match dimensions: predictions_np has k_steps elements, steps_predicted is 1..k_steps)
    ax.plot(steps_predicted, predictions_actual[:len(steps_predicted)], 'm-o', linewidth=plot_params['line_width_thick'], 
            markersize=plot_params['marker_size'], label='Model Prediction (SEQ + NOOPs)', alpha=0.85)
    
    # Plot actual counts if available (actual_counts includes step 0, so indices 1..k_steps match steps 1..k_steps)
    if actual_counts and len(actual_counts) > 1:
        actual_counts_arr = np.array(actual_counts[1:min(k_steps+1, len(actual_counts))])
        ax.plot(steps_predicted[:len(actual_counts_arr)], actual_counts_arr, 'r-s', linewidth=plot_params['line_width_thick'], 
                markersize=plot_params['marker_size'], label='Actual Population (from env)', alpha=0.85)
    
    ax.set_xlabel('Steps Ahead', fontsize=plot_params['label_font_size'], fontweight='bold')
    ax.set_ylabel('Population (actual bacteria count)', fontsize=plot_params['label_font_size'], fontweight='bold')
    ax.set_title(f'K-Step Predictions with SEQ Action (k={k_steps}): Model vs Actual', fontsize=plot_params['title_font_size'], fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=plot_params['font_size'], loc='best')
    
    plt.tight_layout()
    output_path = Path(__file__).parent / "k_step_prediction_seq_action.png"
    plt.savefig(output_path, dpi=plot_params['dpi'], bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_path}")
    plt.close()
    
    print(f"\n✓ SEQ action test completed")


if __name__ == "__main__":
    checkpoint_path = Path(__file__).parent / "src" / "checkpoints" / "new_expression_computation" / "checkpoint_1000.pt"
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("Test 1: K-step prediction WITH env wrapper (DOSE action)")
    print("=" * 70)
    test_k_step_with_env_wrapper(checkpoint_path, k_steps=100, device="cpu")
    
    print("\n" + "=" * 70)
    print("Test 2: Comparison of WITH vs WITHOUT env wrapper (DOSE action)")
    print("=" * 70)
    compare_with_and_without_env(checkpoint_path, k_steps=100, device="cpu")
    
    print("\n" + "=" * 70)
    print("Test 3: K-step prediction with NOOP action")
    print("=" * 70)
    test_k_step_with_noop_action(checkpoint_path, k_steps=50, device="cpu")
    
    print("\n" + "=" * 70)
    print("Test 4: K-step prediction with COUNT action")
    print("=" * 70)
    test_k_step_with_count_action(checkpoint_path, k_steps=50, device="cpu")
    
    print("\n" + "=" * 70)
    print("Test 5: K-step prediction with SEQ action")
    print("=" * 70)
    test_k_step_with_seq_action(checkpoint_path, k_steps=50, device="cpu")
