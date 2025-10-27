"""
Simple Q-Learning training example with tabular representation.

This example is simpler and faster than DQN, good for quick testing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.antibiotic_env import AntibioticEnv
from agent.q_learning_agent import QLearningAgent
from utils.logger import Logger
from utils.visualizer import Visualizer
from utils.metrics import MetricsTracker


def main():
    """Train Q-Learning agent."""
    print("Starting Q-Learning training...")
    
    # Initialize environment
    env = AntibioticEnv(
        initial_population=100,
        carrying_capacity=10000,
        max_steps=200,
        action_space_size=10,
        max_dose=1.0
    )
    
    # Initialize Q-Learning agent
    agent = QLearningAgent(
        state_space_size=100,
        action_space_size=10,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Initialize utilities
    logger = Logger(log_dir='logs', experiment_name='qlearning_training')
    visualizer = Visualizer(output_dir='results/qlearning_training')
    metrics = MetricsTracker()
    
    # Training parameters
    num_episodes = 500
    
    # Training loop
    print(f"\nTraining for {num_episodes} episodes...")
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step = 0
        
        while not (done or truncated):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Update agent
            agent.update(state, action, reward, next_state, done or truncated)
            
            state = next_state
            episode_reward += reward
            step += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record metrics
        episode_rewards.append(episode_reward)
        metrics.record_episode(
            total_reward=episode_reward,
            length=step,
            final_population=info['population_size'],
            final_resistance=info['avg_resistance'],
            total_dose=sum(env.history['doses'])
        )
        
        # Print progress
        if (episode + 1) % 50 == 0:
            stats = metrics.get_statistics(window=50)
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Mean Reward (last 50): {stats['mean_reward']:.2f}")
            print(f"  Mean Population: {stats['mean_population']:.0f}")
            print(f"  Mean Resistance: {stats['mean_resistance']:.3f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
    
    # Final statistics
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    metrics.print_statistics(window=100)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    agent.save('models/qlearning_final.pkl')
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer.plot_training_rewards(episode_rewards, window=50)
    
    # Test the trained agent
    print("\nTesting trained agent...")
    state, _ = env.reset()
    done = False
    truncated = False
    agent.epsilon = 0.0  # Greedy policy
    
    while not (done or truncated):
        action = agent.select_action(state)
        state, reward, done, truncated, info = env.step(action)
    
    # Visualize test episode
    visualizer.plot_combined_metrics(env.history)
    print(f"\nTest episode complete:")
    print(f"  Final population: {info['population_size']}")
    print(f"  Final resistance: {info['avg_resistance']:.3f}")
    print(f"  Total reward: {info['episode_reward']:.2f}")
    
    print("\nAll results saved to results/qlearning_training/!")


if __name__ == "__main__":
    main()
