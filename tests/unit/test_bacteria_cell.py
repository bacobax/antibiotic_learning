"""
Unit tests for BacteriaCell class.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest
import numpy as np
from bacteria.bacteria_cell import BacteriaCell


def test_bacteria_cell_initialization():
    """Test basic initialization of bacteria cell."""
    cell = BacteriaCell(resistance=0.5, growth_rate=1.0, mutation_rate=0.01)
    
    assert cell.resistance == 0.5
    assert cell.growth_rate == 1.0
    assert cell.mutation_rate == 0.01
    assert cell.age == 0
    assert cell.alive is True
    assert cell.genome is not None


def test_resistance_bounds():
    """Test that resistance is bounded between 0 and 1."""
    cell1 = BacteriaCell(resistance=-0.5)
    assert cell1.resistance == 0.0
    
    cell2 = BacteriaCell(resistance=1.5)
    assert cell2.resistance == 1.0


def test_reproduce():
    """Test cell reproduction."""
    parent = BacteriaCell(resistance=0.5, growth_rate=1.0)
    offspring = parent.reproduce()
    
    assert isinstance(offspring, BacteriaCell)
    assert offspring.age == 0
    # Offspring should have similar but not identical traits
    assert abs(offspring.resistance - parent.resistance) < 0.5


def test_survive_antibiotic():
    """Test antibiotic survival mechanism."""
    # High resistance cell should survive better
    resistant_cell = BacteriaCell(resistance=0.9)
    susceptible_cell = BacteriaCell(resistance=0.1)
    
    # Test with moderate dose (stochastic, so we test the mechanism works)
    dose = 0.5
    survival_test = resistant_cell.survive_antibiotic(dose)
    assert isinstance(survival_test, (bool, np.bool_))


def test_update_age():
    """Test age increment."""
    cell = BacteriaCell()
    initial_age = cell.age
    
    cell.update_age()
    assert cell.age == initial_age + 1
    
    cell.update_age()
    assert cell.age == initial_age + 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
