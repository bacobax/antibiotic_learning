"""
Basic example of bacteria population simulation without RL.

This demonstrates the evolutionary dynamics of bacteria under antibiotic pressure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bacteria.bacteria_population import BacteriaPopulation
from bacteria.evolution import EvolutionEngine
from utils.visualizer import Visualizer


def main():
    """Run basic bacteria simulation."""
    print("Starting bacteria simulation...")
    
    # Initialize population and evolution
    population = BacteriaPopulation(initial_size=100, carrying_capacity=10000)
    evolution = EvolutionEngine(mutation_rate=0.01)
    visualizer = Visualizer(output_dir='results/basic_simulation')
    
    # Simulation parameters
    num_steps = 200
    antibiotic_schedule = []
    
    # Define antibiotic dosing schedule (example: pulse therapy)
    for step in range(num_steps):
        if 50 <= step < 60 or 120 <= step < 130:
            dose = 0.5  # High dose periods
        elif 60 <= step < 120:
            dose = 0.1  # Low maintenance dose
        else:
            dose = 0.0  # No antibiotic
        antibiotic_schedule.append(dose)
    
    # Run simulation
    print(f"\nRunning {num_steps} simulation steps...")
    for step in range(num_steps):
        dose = antibiotic_schedule[step]
        
        # Simulate one step
        stats = population.step(antibiotic_dose=dose)
        evolution.evolve_step(population, antibiotic_dose=dose)
        
        # Print progress every 20 steps
        if step % 20 == 0:
            print(f"Step {step:3d}: Population={stats['population_size']:5d}, "
                  f"Resistance={stats['avg_resistance']:.3f}, Dose={dose:.2f}")
        
        # Check if population is extinct
        if population.is_extinct():
            print(f"\nPopulation went extinct at step {step}")
            break
    
    # Print final statistics
    print(f"\n{'='*60}")
    print("Final Statistics:")
    print(f"{'='*60}")
    print(f"Final Population: {population.get_size()}")
    print(f"Final Avg Resistance: {population.get_average_resistance():.3f}")
    print(f"Final Avg Growth Rate: {population.get_average_growth_rate():.3f}")
    print(f"Total Steps: {population.time_step}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualizer.plot_population_dynamics(population.history)
    visualizer.plot_resistance_evolution(population.history)
    print("Plots saved to results/basic_simulation/")
    
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
