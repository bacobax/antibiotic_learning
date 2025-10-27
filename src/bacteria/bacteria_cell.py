"""
BacteriaCell class representing individual bacteria with genetic properties.
"""

import numpy as np


class BacteriaCell:
    """
    Represents a single bacterium with genetic properties and resistance traits.
    
    Attributes:
        resistance (float): Resistance level to antibiotics (0.0 to 1.0)
        growth_rate (float): Natural growth rate
        mutation_rate (float): Probability of mutation during reproduction
        genome (np.ndarray): Genetic representation of the cell
        age (int): Age of the cell in time steps
    """
    
    def __init__(self, resistance=0.0, growth_rate=1.0, mutation_rate=0.01, genome=None):
        """
        Initialize a bacteria cell.
        
        Args:
            resistance (float): Initial resistance level (0.0 to 1.0)
            growth_rate (float): Growth rate of the cell
            mutation_rate (float): Probability of mutation
            genome (np.ndarray): Optional genetic representation
        """
        self.resistance = np.clip(resistance, 0.0, 1.0)
        self.growth_rate = growth_rate
        self.mutation_rate = mutation_rate
        self.genome = genome if genome is not None else self._initialize_genome()
        self.age = 0
        self.alive = True
    
    def _initialize_genome(self, size=10):
        """
        Initialize random genome.
        
        Args:
            size (int): Size of the genome array
            
        Returns:
            np.ndarray: Random genome values
        """
        return np.random.rand(size)
    
    def reproduce(self):
        """
        Create a daughter cell with potential mutations.
        
        Returns:
            BacteriaCell: New bacteria cell (offspring)
        """
        # Apply mutation
        new_genome = self.genome.copy()
        if np.random.random() < self.mutation_rate:
            mutation_idx = np.random.randint(len(new_genome))
            new_genome[mutation_idx] += np.random.normal(0, 0.1)
            new_genome = np.clip(new_genome, 0, 1)
        
        # Calculate new traits based on genome
        new_resistance = np.clip(
            self.resistance + np.random.normal(0, 0.05),
            0.0, 1.0
        )
        new_growth_rate = np.clip(
            self.growth_rate + np.random.normal(0, 0.05),
            0.1, 2.0
        )
        
        return BacteriaCell(
            resistance=new_resistance,
            growth_rate=new_growth_rate,
            mutation_rate=self.mutation_rate,
            genome=new_genome
        )
    
    def survive_antibiotic(self, antibiotic_dose):
        """
        Determine if cell survives antibiotic exposure.
        
        Args:
            antibiotic_dose (float): Concentration of antibiotic
            
        Returns:
            bool: True if cell survives, False otherwise
        """
        # Survival probability based on resistance and dose
        survival_prob = self.resistance / (self.resistance + antibiotic_dose + 1e-6)
        return np.random.random() < survival_prob
    
    def update_age(self):
        """Increment cell age."""
        self.age += 1
    
    def __repr__(self):
        return f"BacteriaCell(resistance={self.resistance:.3f}, growth_rate={self.growth_rate:.3f}, age={self.age})"
