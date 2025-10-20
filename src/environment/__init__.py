"""
Environment module for RL interaction with bacteria simulation.

This module provides Gymnasium-compatible environment for training RL agents.
"""

from .antibiotic_env import AntibioticEnv

__all__ = ["AntibioticEnv"]
