"""
Train a reinforcement learning agent to optimize antibiotic dosing.

This example demonstrates how to train a DQN agent on the antibiotic environment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.antibiotic_env import AntibioticEnv
from agent.dqn_agent import DQNAgent
from utils.logger import Logger
from utils.visualizer import Visualizer
from utils.metrics import MetricsTracker


def main():
    """Train RL agent on antibiotic dosing task."""
    print("Starting RL training...")
    
    # Initialize environment
    env = AntibioticEnv(
        initial_population=100,
        carrying_capacity=10000,
        max_steps=200,
        action_space_size=10,
        max_dose=1.0
    )
    
    # Initialize agent
    agent = DQNAgent(
        state_dim=4,
        action_space_size=10,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Initialize utilities
    logger = Logger(log_dir='logs', experiment_name='dqn_training')
    visualizer = Visualizer(output_dir='results/dqn_training')
    metrics = MetricsTracker()
    
    # Training parameters
    num_episodes = 1000
    save_interval = 100
    eval_interval = 50
    
    # Log configuration
    config = {
        'environment': {
            'initial_population': 100,
            'carrying_capacity': 10000,
            'max_steps': 200,
            'action_space_size': 10
        },
        'agent': {
            'type': 'DQN',
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_decay': 0.995
        },
        'training': {
            'num_episodes': num_episodes
        }
    }
    logger.log_config(config)
    
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
        
        # Logging
        if (episode + 1) % 10 == 0:
            logger.log_metrics(episode + 1, {
                'episode_reward': episode_reward,
                'epsilon': agent.epsilon,
                'final_population': info['population_size'],
                'final_resistance': info['avg_resistance']
            })
        
        # Print progress
        if (episode + 1) % 50 == 0:
            stats = metrics.get_statistics(window=50)
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Mean Reward (last 50): {stats['mean_reward']:.2f}")
            print(f"  Mean Population: {stats['mean_population']:.0f}")
            print(f"  Mean Resistance: {stats['mean_resistance']:.3f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
        
        # Save model
        if (episode + 1) % save_interval == 0:
            os.makedirs('models', exist_ok=True)
            model_path = f'models/dqn_episode_{episode + 1}.pth'
            agent.save(model_path)
            logger.info(f"Model saved to {model_path}")
    
    # Final statistics
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    metrics.print_statistics(window=100)
    
    # Save final model
    agent.save('models/dqn_final.pth')
    logger.save_metrics()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer.plot_training_rewards(episode_rewards, window=50)
    print("Training plots saved to results/dqn_training/")
    
    # Test the trained agent
    print("\nTesting trained agent...")
    state, _ = env.reset()
    done = False
    truncated = False
    
    while not (done or truncated):
        # Use greedy policy (no exploration)
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0
        action = agent.select_action(state)
        agent.epsilon = old_epsilon
        
        state, reward, done, truncated, info = env.step(action)
    
    # Visualize test episode
    visualizer.plot_combined_metrics(env.history)
    print(f"\nTest episode complete:")
    print(f"  Final population: {info['population_size']}")
    print(f"  Final resistance: {info['avg_resistance']:.3f}")
    print(f"  Total reward: {info['episode_reward']:.2f}")
    
    print("\nAll results saved!")


if __name__ == "__main__":
    main()
