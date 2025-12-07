#!/usr/bin/env python3
"""
Test script to verify that checkpoint save/load preserves recurrent state.
"""
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl.agent import RLAgent
from rl.models import RecurrentActorCritic
from rl.training_config import PPOConfig
from rl.ppo import PPOTrainer


def test_checkpoint_state_preservation():
    """Test that recurrent state is preserved in checkpoints."""
    print("=" * 70)
    print("Testing Checkpoint State Preservation")
    print("=" * 70)
    
    # Create a simple model and agent
    cfg = PPOConfig(
        obs_dim=50,  # Realistic obs_dim
        n_discrete=4,
        k_doses=3,
        hidden_dim=64,
        rnn_layers=1,
        device="cpu",
        seed=42,
    )
    
    model = RecurrentActorCritic(
        obs_dim=cfg.obs_dim,
        n_discrete=cfg.n_discrete,
        k_doses=cfg.k_doses,
        hidden_dim=cfg.hidden_dim,
        rnn_layers=cfg.rnn_layers,
        sigmoid_scale_factor=1.0,
    )
    
    # Create initial agent
    agent1 = RLAgent(model=model, device=cfg.device)
    
    # Set up trainer (required for save)
    trainer = PPOTrainer(model, cfg)
    agent1.set_trainer(trainer)
    
    # Simulate some steps to get non-zero internal state
    print("\n1. Creating initial agent and simulating steps...")
    obs = torch.randn(1, cfg.obs_dim)
    for _ in range(5):
        with torch.no_grad():
            _ = agent1.select_action(obs.numpy()[0])
    
    # Capture state before saving
    h_state_before = agent1.prev_h_state.clone()
    action_onehot_before = agent1.prev_action_onehot.clone()
    action_cont_before = agent1.prev_action_cont.clone()
    pred_pop_before = agent1.prev_pred_next_pop.clone()
    
    print(f"   ✓ H-state shape: {h_state_before.shape}")
    print(f"   ✓ H-state sample: {h_state_before.view(-1)[:5]}")
    print(f"   ✓ Action onehot shape: {action_onehot_before.shape}")
    print(f"   ✓ Action cont shape: {action_cont_before.shape}")
    print(f"   ✓ Pred pop shape: {pred_pop_before.shape}")
    
    # Save checkpoint
    checkpoint_path = "/tmp/test_checkpoint.pt"
    print(f"\n2. Saving checkpoint to {checkpoint_path}...")
    agent1.save_model(checkpoint_path, extra_info={"update": 100})
    print("   ✓ Checkpoint saved")
    
    # Load checkpoint
    print(f"\n3. Loading checkpoint...")
    agent2 = RLAgent.load_agent_from_checkpoint(checkpoint_path, device=cfg.device)
    # Need to set trainer again for the loaded agent
    trainer2 = PPOTrainer(agent2.model, cfg)
    agent2.set_trainer(trainer2)
    print("   ✓ Checkpoint loaded")
    
    # Verify state matches
    print("\n4. Verifying restored state...")
    
    # Check h_state
    h_match = torch.allclose(h_state_before, agent2.prev_h_state, atol=1e-6)
    print(f"   {'✓' if h_match else '✗'} H-state matches: {h_match}")
    if not h_match:
        print(f"      Max diff: {(h_state_before - agent2.prev_h_state).abs().max().item()}")
    
    # Check action onehot
    action_onehot_match = torch.allclose(action_onehot_before, agent2.prev_action_onehot, atol=1e-6)
    print(f"   {'✓' if action_onehot_match else '✗'} Action onehot matches: {action_onehot_match}")
    
    # Check action cont
    action_cont_match = torch.allclose(action_cont_before, agent2.prev_action_cont, atol=1e-6)
    print(f"   {'✓' if action_cont_match else '✗'} Action cont matches: {action_cont_match}")
    
    # Check pred pop
    pred_pop_match = torch.allclose(pred_pop_before, agent2.prev_pred_next_pop, atol=1e-6)
    print(f"   {'✓' if pred_pop_match else '✗'} Pred pop matches: {pred_pop_match}")
    
    # Verify model parameters match
    print("\n5. Verifying model parameters match...")
    model_match = True
    for p1, p2 in zip(agent1.model.parameters(), agent2.model.parameters()):
        if not torch.allclose(p1, p2, atol=1e-6):
            model_match = False
            break
    print(f"   {'✓' if model_match else '✗'} Model parameters match: {model_match}")
    
    # Final verdict
    all_match = h_match and action_onehot_match and action_cont_match and pred_pop_match and model_match
    print("\n" + "=" * 70)
    if all_match:
        print("✓ ALL TESTS PASSED - Checkpoint state preservation works!")
    else:
        print("✗ SOME TESTS FAILED - State not fully preserved")
    print("=" * 70)
    
    # Cleanup
    Path(checkpoint_path).unlink()
    
    return all_match


if __name__ == "__main__":
    success = test_checkpoint_state_preservation()
    sys.exit(0 if success else 1)
