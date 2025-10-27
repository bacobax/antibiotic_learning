"""
Bacteria module for evolutionary simulation.

This module contains classes and functions for simulating bacterial population
evolution over time using evolutionary algorithms.
"""

from .bacteria_population import BacteriaPopulation
from .bacteria_cell import BacteriaCell
from .evolution import EvolutionEngine

__all__ = ["BacteriaPopulation", "BacteriaCell", "EvolutionEngine"]
