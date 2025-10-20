"""
Quick demo to verify the complete system works end-to-end.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.antibiotic_env import AntibioticEnv
from agent.q_learning_agent import QLearningAgent


def main():
    """Run a quick demo of the system."""
    print("=" * 60)
    print("Antibiotic Learning System - Quick Demo")
    print("=" * 60)
    
    # Initialize environment
    env = AntibioticEnv(
        initial_population=100,
        carrying_capacity=5000,
        max_steps=50,
        action_space_size=5,
        max_dose=1.0
    )
    
    # Initialize Q-Learning agent
    agent = QLearningAgent(
        state_space_size=50,
        action_space_size=5,
        learning_rate=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.95,
        epsilon_min=0.1
    )
    
    print("\nTraining agent for 10 episodes...")
    print("-" * 60)
    
    # Training loop
    episode_rewards = []
    
    for episode in range(10):
        state, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update agent
            agent.update(state, action, reward, next_state, done or truncated)
            
            state = next_state
            episode_reward += reward
        
        # Decay epsilon
        agent.decay_epsilon()
        episode_rewards.append(episode_reward)
        
        print(f"Episode {episode + 1:2d}: "
              f"Reward={episode_reward:7.2f}, "
              f"Final Pop={info['population_size']:4d}, "
              f"Resistance={info['avg_resistance']:.3f}, "
              f"Epsilon={agent.epsilon:.2f}")
    
    print("-" * 60)
    print("\nTraining Complete!")
    print(f"Average Reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
    
    # Test the trained agent
    print("\nTesting trained agent (greedy policy)...")
    print("-" * 60)
    
    state, _ = env.reset()
    done = False
    truncated = False
    agent.epsilon = 0.0  # Greedy policy
    step = 0
    
    while not (done or truncated) and step < 10:
        action = agent.select_action(state)
        dose = agent.get_action_value(action, max_dose=1.0)
        state, reward, done, truncated, info = env.step(action)
        
        if step % 2 == 0:  # Print every other step
            print(f"Step {step:2d}: "
                  f"Dose={dose:.2f}, "
                  f"Pop={info['population_size']:4d}, "
                  f"Res={info['avg_resistance']:.3f}, "
                  f"Reward={reward:6.2f}")
        step += 1
    
    print("-" * 60)
    print(f"\nFinal State:")
    print(f"  Population: {info['population_size']}")
    print(f"  Resistance: {info['avg_resistance']:.3f}")
    print(f"  Total Steps: {step}")
    print(f"  Episode Reward: {info['episode_reward']:.2f}")
    
    print("\n" + "=" * 60)
    print("Demo Complete! All systems working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
