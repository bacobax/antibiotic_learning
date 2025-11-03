#!/usr/bin/env python3
"""
Test script to verify TensorBoard logging is working correctly.
Run this before starting full training.
"""

import sys
from pathlib import Path

# Add RL module to path
sys.path.insert(0, str(Path(__file__).parent))

from logger import TrainingLogger
import time

def test_tensorboard_logging():
    """Test TensorBoard logging functionality."""
    
    print("=" * 70)
    print("Testing TensorBoard Logging")
    print("=" * 70)
    
    # Create test logger
    log_dir = Path("./test_logs")
    print(f"\n1. Creating logger at: {log_dir}")
    logger = TrainingLogger(log_dir, experiment_name="test_experiment")
    
    # Log some test metrics
    print("\n2. Logging test metrics...")
    for update in range(5):
        metrics = {
            "rollout/mean_episode_reward": 50.0 + update * 10,
            "rollout/std_episode_reward": 5.0,
            "training/loss_actor": 0.5 - update * 0.05,
            "training/loss_critic": 0.3 - update * 0.03,
            "training/entropy": 1.0 - update * 0.1,
            "training/clip_fraction": 0.2 + update * 0.05,
        }
        
        logger.log_update_metrics(update, 
                                 {"mean_episode_reward": metrics["rollout/mean_episode_reward"],
                                  "std_episode_reward": metrics["rollout/std_episode_reward"],
                                  "num_episodes": 5},
                                 {"loss_actor": metrics["training/loss_actor"],
                                  "loss_critic": metrics["training/loss_critic"],
                                  "entropy": metrics["training/entropy"],
                                  "clip_fraction": metrics["training/clip_fraction"],
                                  "grad_norm": 0.8,
                                  "value_mean": 0.5})
        
        print(f"  Update {update}: reward={metrics['rollout/mean_episode_reward']:.1f}, loss={metrics['training/loss_actor']:.4f}")
        time.sleep(0.1)
    
    # Close logger
    print("\n3. Closing logger...")
    logger.close()
    
    # Check output files
    print("\n4. Checking output files...")
    
    files_to_check = [
        ("training.log", "Python logs"),
        ("metrics.json", "JSON metrics"),
    ]
    
    for filename, description in files_to_check:
        filepath = log_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  ✓ {filename} ({description}): {size} bytes")
        else:
            print(f"  ✗ {filename} ({description}): NOT FOUND")
    
    # Check TensorBoard directory
    tb_dir = log_dir / "test_experiment"
    if tb_dir.exists():
        print(f"  ✓ TensorBoard directory: {tb_dir}")
        event_files = list(tb_dir.glob("events.out.tfevents*"))
        if event_files:
            print(f"    ✓ Event files found: {len(event_files)}")
            for ef in event_files:
                print(f"      - {ef.name} ({ef.stat().st_size} bytes)")
        else:
            print(f"    ✗ No event files found in {tb_dir}")
    else:
        print(f"  ✗ TensorBoard directory NOT FOUND: {tb_dir}")
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
    print("\nTo view TensorBoard results, run:")
    print(f"  tensorboard --logdir={log_dir} --port=6006")
    print("\nThen open: http://localhost:6006")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    try:
        test_tensorboard_logging()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
