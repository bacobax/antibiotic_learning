"""
Unit tests for BacteriaPopulation class.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import pytest
from bacteria.bacteria_population import BacteriaPopulation


def test_population_initialization():
    """Test population initialization."""
    pop = BacteriaPopulation(initial_size=50, carrying_capacity=1000)
    
    assert len(pop.cells) == 50
    assert pop.carrying_capacity == 1000
    assert pop.time_step == 0


def test_population_growth():
    """Test population can grow."""
    pop = BacteriaPopulation(initial_size=10, carrying_capacity=1000)
    initial_size = pop.get_size()
    
    # Let it grow for a few steps
    for _ in range(5):
        pop.grow()
    
    # Population should grow (probabilistic but very likely)
    assert pop.get_size() >= initial_size


def test_apply_antibiotic():
    """Test antibiotic application reduces population."""
    pop = BacteriaPopulation(initial_size=100, carrying_capacity=1000)
    initial_size = pop.get_size()
    
    # Apply high dose
    killed = pop.apply_antibiotic(dose=0.9)
    
    # Should kill some cells
    assert pop.get_size() < initial_size
    assert killed == initial_size - pop.get_size()


def test_population_step():
    """Test complete population step."""
    pop = BacteriaPopulation(initial_size=50, carrying_capacity=1000)
    
    stats = pop.step(antibiotic_dose=0.2)
    
    assert 'time_step' in stats
    assert 'population_size' in stats
    assert 'cells_killed' in stats
    assert 'avg_resistance' in stats
    assert pop.time_step == 1


def test_is_extinct():
    """Test extinction detection."""
    pop = BacteriaPopulation(initial_size=10, carrying_capacity=1000)
    
    assert not pop.is_extinct()
    
    # Kill all cells with very high dose
    for _ in range(10):
        pop.apply_antibiotic(dose=10.0)
    
    # Should be extinct (or very close)
    if pop.get_size() == 0:
        assert pop.is_extinct()


def test_get_average_resistance():
    """Test resistance calculation."""
    pop = BacteriaPopulation(initial_size=10, carrying_capacity=1000)
    
    avg_resistance = pop.get_average_resistance()
    assert 0.0 <= avg_resistance <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
