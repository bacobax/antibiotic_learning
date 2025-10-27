"""
Integration tests for the antibiotic environment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest
import numpy as np
from environment.antibiotic_env import AntibioticEnv


def test_env_initialization():
    """Test environment initialization."""
    env = AntibioticEnv(
        initial_population=100,
        carrying_capacity=1000,
        max_steps=50
    )
    
    assert env.initial_population == 100
    assert env.carrying_capacity == 1000
    assert env.max_steps == 50


def test_env_reset():
    """Test environment reset."""
    env = AntibioticEnv()
    obs, info = env.reset()
    
    assert obs is not None
    assert len(obs) == 4  # [population, resistance, growth_rate, previous_dose]
    assert isinstance(info, dict)
    assert env.current_step == 0


def test_env_step():
    """Test environment step."""
    env = AntibioticEnv()
    env.reset()
    
    action = 5  # Some action
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert env.current_step == 1


def test_env_episode():
    """Test complete episode."""
    env = AntibioticEnv(max_steps=10)
    obs, info = env.reset()
    
    done = False
    truncated = False
    steps = 0
    
    while not (done or truncated):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
    
    # Should terminate within max_steps
    assert steps <= 10


def test_action_to_dose_conversion():
    """Test action to dose conversion."""
    env = AntibioticEnv(action_space_size=10, max_dose=1.0)
    
    # Action 0 should give dose 0.0
    dose_0 = env._action_to_dose(0)
    assert dose_0 == 0.0
    
    # Action 9 should give dose 1.0
    dose_max = env._action_to_dose(9)
    assert dose_max == 1.0
    
    # Middle action should give middle dose
    dose_mid = env._action_to_dose(5)
    assert 0.4 < dose_mid < 0.6


def test_reward_calculation():
    """Test reward calculation."""
    env = AntibioticEnv()
    env.reset()
    
    # Create mock stats
    stats = {
        'population_size': 100,
        'avg_resistance': 0.5
    }
    
    reward = env._calculate_reward(stats, dose=0.5)
    assert isinstance(reward, (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
