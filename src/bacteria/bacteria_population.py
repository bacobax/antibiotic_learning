"""
BacteriaPopulation class for managing populations of bacteria.
"""

import numpy as np
from .bacteria_cell import BacteriaCell


class BacteriaPopulation:
    """
    Manages a population of bacteria cells and their collective behavior.
    
    Attributes:
        cells (list): List of BacteriaCell instances
        carrying_capacity (int): Maximum population size
        time_step (int): Current simulation time step
    """
    
    def __init__(self, initial_size=100, carrying_capacity=10000):
        """
        Initialize bacteria population.
        
        Args:
            initial_size (int): Number of cells at initialization
            carrying_capacity (int): Maximum sustainable population
        """
        self.carrying_capacity = carrying_capacity
        self.time_step = 0
        self.cells = self._initialize_population(initial_size)
        self.history = {
            'size': [initial_size],
            'avg_resistance': [self.get_average_resistance()],
            'avg_growth_rate': [self.get_average_growth_rate()]
        }
    
    def _initialize_population(self, size):
        """
        Create initial population with random properties.
        
        Args:
            size (int): Number of cells to create
            
        Returns:
            list: List of BacteriaCell instances
        """
        return [BacteriaCell(
            resistance=np.random.uniform(0.0, 0.3),
            growth_rate=np.random.uniform(0.8, 1.2),
            mutation_rate=0.01
        ) for _ in range(size)]
    
    def grow(self):
        """
        Simulate population growth through cell reproduction.
        
        Growth is limited by carrying capacity using logistic model.
        """
        current_size = len(self.cells)
        growth_factor = 1 - (current_size / self.carrying_capacity)
        
        if growth_factor > 0:
            new_cells = []
            for cell in self.cells:
                # Probability of reproduction based on growth rate and capacity
                reproduction_prob = cell.growth_rate * growth_factor * 0.1
                if np.random.random() < reproduction_prob:
                    new_cells.append(cell.reproduce())
            
            self.cells.extend(new_cells)
    
    def apply_antibiotic(self, dose):
        """
        Apply antibiotic to population and remove susceptible cells.
        
        Args:
            dose (float): Antibiotic concentration (0.0 to 1.0)
            
        Returns:
            int: Number of cells killed
        """
        initial_size = len(self.cells)
        self.cells = [cell for cell in self.cells if cell.survive_antibiotic(dose)]
        killed = initial_size - len(self.cells)
        return killed
    
    def update(self):
        """Update all cells in the population."""
        for cell in self.cells:
            cell.update_age()
        
        # Remove very old cells (natural death)
        max_age = 100
        self.cells = [cell for cell in self.cells if cell.age < max_age]
    
    def step(self, antibiotic_dose=0.0):
        """
        Perform one time step of the simulation.
        
        Args:
            antibiotic_dose (float): Antibiotic dose to apply
            
        Returns:
            dict: Statistics about the population after the step
        """
        self.time_step += 1
        
        # Growth phase
        self.grow()
        
        # Antibiotic application
        killed = self.apply_antibiotic(antibiotic_dose)
        
        # Natural processes
        self.update()
        
        # Record statistics
        stats = {
            'time_step': self.time_step,
            'population_size': len(self.cells),
            'cells_killed': killed,
            'avg_resistance': self.get_average_resistance(),
            'avg_growth_rate': self.get_average_growth_rate()
        }
        
        self.history['size'].append(stats['population_size'])
        self.history['avg_resistance'].append(stats['avg_resistance'])
        self.history['avg_growth_rate'].append(stats['avg_growth_rate'])
        
        return stats
    
    def get_average_resistance(self):
        """
        Calculate average resistance in population.
        
        Returns:
            float: Average resistance level
        """
        if not self.cells:
            return 0.0
        return np.mean([cell.resistance for cell in self.cells])
    
    def get_average_growth_rate(self):
        """
        Calculate average growth rate in population.
        
        Returns:
            float: Average growth rate
        """
        if not self.cells:
            return 0.0
        return np.mean([cell.growth_rate for cell in self.cells])
    
    def get_size(self):
        """
        Get current population size.
        
        Returns:
            int: Number of living cells
        """
        return len(self.cells)
    
    def is_extinct(self):
        """
        Check if population is extinct.
        
        Returns:
            bool: True if no cells remain
        """
        return len(self.cells) == 0
    
    def __repr__(self):
        return f"BacteriaPopulation(size={len(self.cells)}, step={self.time_step}, avg_resistance={self.get_average_resistance():.3f})"
