"""
EvolutionEngine for managing evolutionary dynamics of bacteria populations.
"""

import numpy as np


class EvolutionEngine:
    """
    Manages evolutionary processes including selection, mutation, and adaptation.
    
    This class implements evolutionary algorithms to simulate how bacteria
    populations evolve in response to environmental pressures.
    """
    
    def __init__(self, selection_pressure=0.5, mutation_rate=0.01):
        """
        Initialize evolution engine.
        
        Args:
            selection_pressure (float): Strength of selection (0.0 to 1.0)
            mutation_rate (float): Base mutation rate
        """
        self.selection_pressure = selection_pressure
        self.mutation_rate = mutation_rate
        self.generation = 0
    
    def select(self, population, fitness_function):
        """
        Apply selection to population based on fitness.
        
        Args:
            population (BacteriaPopulation): Population to select from
            fitness_function (callable): Function to calculate cell fitness
            
        Returns:
            list: Selected cells
        """
        if not population.cells:
            return []
        
        # Calculate fitness for each cell
        fitness_scores = [fitness_function(cell) for cell in population.cells]
        total_fitness = sum(fitness_scores)
        
        if total_fitness == 0:
            return population.cells
        
        # Normalize fitness
        probabilities = [f / total_fitness for f in fitness_scores]
        
        # Selection based on fitness
        selected_indices = np.random.choice(
            len(population.cells),
            size=min(len(population.cells), int(len(population.cells) * self.selection_pressure)),
            replace=False,
            p=probabilities
        )
        
        return [population.cells[i] for i in selected_indices]
    
    def mutate_population(self, population, environmental_stress=0.0):
        """
        Apply mutations to population based on environmental stress.
        
        Args:
            population (BacteriaPopulation): Population to mutate
            environmental_stress (float): Additional stress factor (0.0 to 1.0)
        """
        effective_mutation_rate = self.mutation_rate * (1 + environmental_stress)
        
        for cell in population.cells:
            if np.random.random() < effective_mutation_rate:
                # Random beneficial mutation
                if np.random.random() < 0.3:  # 30% chance of beneficial mutation
                    cell.resistance = np.clip(cell.resistance + 0.05, 0.0, 1.0)
                else:  # Otherwise neutral or slightly deleterious
                    cell.resistance = np.clip(
                        cell.resistance + np.random.normal(0, 0.02),
                        0.0, 1.0
                    )
    
    def calculate_fitness(self, cell, antibiotic_dose=0.0):
        """
        Calculate fitness of a cell given environmental conditions.
        
        Args:
            cell (BacteriaCell): Cell to evaluate
            antibiotic_dose (float): Current antibiotic concentration
            
        Returns:
            float: Fitness score
        """
        # Fitness is combination of growth rate and survival
        survival_fitness = cell.resistance / (antibiotic_dose + 1.0)
        growth_fitness = cell.growth_rate
        
        # Trade-off: high resistance often comes with lower growth rate
        fitness = survival_fitness * growth_fitness
        return max(0.0, fitness)
    
    def evolve_step(self, population, antibiotic_dose=0.0):
        """
        Perform one step of evolution.
        
        Args:
            population (BacteriaPopulation): Population to evolve
            antibiotic_dose (float): Environmental antibiotic concentration
            
        Returns:
            dict: Evolution statistics
        """
        self.generation += 1
        
        initial_resistance = population.get_average_resistance()
        
        # Apply mutations based on environmental stress
        environmental_stress = antibiotic_dose
        self.mutate_population(population, environmental_stress)
        
        final_resistance = population.get_average_resistance()
        
        stats = {
            'generation': self.generation,
            'resistance_change': final_resistance - initial_resistance,
            'avg_resistance': final_resistance,
            'population_size': population.get_size()
        }
        
        return stats
    
    def __repr__(self):
        return f"EvolutionEngine(generation={self.generation}, selection_pressure={self.selection_pressure})"
