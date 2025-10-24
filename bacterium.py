"""
Bacterium agent implementation.
"""

import math
import random
import numpy as np
from mesa import Agent

from config import (
    BACTERIAL_TYPES, ANTIBIOTIC_TYPES, ALLOCATION_PARAMS, 
    GROWTH_PARAMS, MUTATION_STD, BACTERIA_SPEED, PERSISTENCE_PARAMS
)


class Bacterium(Agent):
    """Individual bacterium agent with resistance traits and lifecycle."""
    
    def __init__(self, model, bacterial_type="E.coli"):
        super().__init__(model)
        self.unique_id = model.next_id()
        self.pos = None
        self.energy = random.uniform(1.0, 2.0)
        self.bacterial_type = bacterial_type
        self.age = 0.0  # Float to support fractional aging for persistors
        
        # Initialize persistor state flag
        self.is_persistor = False
        self.has_hgt_gene = random.random() < 0.05  # 5% chance of having HGT gene
        
        # Initialize traits
        self._initialize_traits(bacterial_type)
        
        # Expression states (X) start at 0
        self.expression = {
            "membrane": 0.0,
            "efflux": 0.0,
            "enzyme": 0.0,
            "repair": 0.0
        }

    def _initialize_traits(self, bacterial_type):
        """Initialize resistance traits based on bacterial type"""
        type_def = BACTERIAL_TYPES[bacterial_type]
        total = sum(type_def[trait][0] for trait in ['enzyme', 'efflux', 'membrane', 'repair'])
        
        # Normalize initial allocations to budget
        budget = ALLOCATION_PARAMS["total_budget"]
        self.enzyme = max(0.0, min(budget * type_def["enzyme"][0] / total, budget))
        self.efflux = max(0.0, min(budget * type_def["efflux"][0] / total, budget))
        self.membrane = max(0.0, min(budget * type_def["membrane"][0] / total, budget))
        self.repair = max(0.0, min(budget * type_def["repair"][0] / total, budget))
        
        # Other initializations
        self.max_age = max(10, int(random.gauss(*type_def["max_age"])))
        self.speed = max(0.1, random.gauss(*type_def["base_speed"]))

    def mutate_offspring_traits(self):
        """Implement budget-allocation mutation model"""
        traits = ["enzyme", "efflux", "membrane", "repair"]
        current_allocations = np.array([getattr(self, trait) for trait in traits])
        budget = ALLOCATION_PARAMS["total_budget"]
        
        # 1. Small reallocations (correlated Gaussian)
        deltas = np.random.normal(0, ALLOCATION_PARAMS["reallocation_std"], len(traits))
        deltas -= deltas.mean()  # ensure sum of changes is zero
        
        # 2. Rare amplifications
        if random.random() < ALLOCATION_PARAMS["amplification_prob"]:
            amp_idx = random.randrange(len(traits))
            amp_amount = random.uniform(0, 0.2)
            deltas[amp_idx] += amp_amount
            self.energy -= amp_amount * ALLOCATION_PARAMS["amplification_cost"]
        
        # Apply changes while ensuring constraints
        new_allocations = current_allocations + deltas
        new_allocations = np.maximum(new_allocations, 0)
        new_allocations *= budget / new_allocations.sum()
        
        return {trait: float(alloc) for trait, alloc in zip(traits, new_allocations)}

    def _update_expression_states(self, local_antibiotics):
        """Update gene expression based on antibiotic presence
        
        Args:
            local_antibiotics: dict mapping antibiotic_type -> concentration
        
        TODO: Currently uses sum of all antibiotics as temporary solution.
        Should be updated to respond differently to each antibiotic type.
        
        Persistors maintain minimal expression states to reduce energy costs.
        """
        # TEMPORARY: Sum all antibiotic concentrations
        # TODO: Each antibiotic should induce different expression patterns
        local_ab = sum(local_antibiotics.values()) if local_antibiotics else 0.0
        
        # Persistors don't actively respond to antibiotics (dormant metabolism)
        if self.is_persistor:
            # Slow decay of existing expression
            for k in self.expression:
                X = self.expression[k]
                dX = -GROWTH_PARAMS["kd"][k] * X * GROWTH_PARAMS["dt"] * 0.5  # Slower decay
                self.expression[k] = max(0.0, X + dX)
            return
        
        # Normal expression dynamics for active bacteria
        if local_ab > 0:
            S_A = (local_ab**GROWTH_PARAMS["n_ind"]) / \
                  (GROWTH_PARAMS["k_i"]**GROWTH_PARAMS["n_ind"] + local_ab**GROWTH_PARAMS["n_ind"])
        else:
            S_A = 0.0
        
        for k in self.expression:
            g_i = getattr(self, k)
            X = self.expression[k]
            dX = (GROWTH_PARAMS["ks"][k] * g_i * S_A - GROWTH_PARAMS["kd"][k] * X) * GROWTH_PARAMS["dt"]
            self.expression[k] = max(0.0, min(1.0, X + dX))

    def _consume_nutrients(self, local_food, fx, fy):
        """Consume nutrients and update energy
        
        Persistors cannot actively consume nutrients but lose energy slowly.
        """
        if self.is_persistor:
            # Persistors don't actively consume nutrients (dormant metabolism)
            # Only lose energy through minimal maintenance
            self.energy -= PERSISTENCE_PARAMS["energy_decay_rate"] * GROWTH_PARAMS["dt"]
            return
        
        # Normal nutrient consumption for active bacteria
        # Monod kinetics
        uptake = GROWTH_PARAMS["u_max"] * (local_food / (GROWTH_PARAMS["k_s"] + local_food))
        self.model.subtract_from_field(self.model.food_field, fx, fy, uptake * GROWTH_PARAMS["dt"])
        
        # Expression cost
        expr_cost = GROWTH_PARAMS["c_prod"] * sum(
            GROWTH_PARAMS["expression_weights"][k] * self.expression[k] 
            for k in self.expression
        )
        
        # Energy update
        dE = (GROWTH_PARAMS["eta"] * uptake - GROWTH_PARAMS["m0"] - expr_cost) * GROWTH_PARAMS["dt"]
        self.energy = max(0.0, self.energy + dE)

    def _check_antibiotic_death(self, local_antibiotics):
        """Check if bacterium dies from antibiotic exposure
        
        Processes each antibiotic type separately with its own parameters,
        then combines the kill probabilities.
        
        Args:
            local_antibiotics: dict mapping antibiotic_type -> concentration
        
        Persistors have greatly reduced susceptibility to antibiotics.
        """
        if not local_antibiotics or sum(local_antibiotics.values()) <= 0:
            return False
        
        # Calculate effective concentration and kill probability for each antibiotic
        total_kappa = 0.0
        
        for ab_type, ab_concentration in local_antibiotics.items():
            if ab_concentration <= 0:
                continue
                
            # Get antibiotic-specific parameters
            ab_params = ANTIBIOTIC_TYPES[ab_type]
            
            # Calculate resistance effectiveness for this specific antibiotic
            # Each antibiotic has different weights for how effective each resistance mechanism is
            resistance_reduction = (
                ab_params["enzyme_weight"] * GROWTH_PARAMS["alpha"]["enzyme"] * self.expression["enzyme"] +
                ab_params["efflux_weight"] * GROWTH_PARAMS["alpha"]["efflux"] * self.expression["efflux"] +
                ab_params["membrane_weight"] * GROWTH_PARAMS["alpha"]["membrane"] * self.expression["membrane"]
            )
            
            # Calculate effective antibiotic concentration after resistance
            A_eff = ab_concentration * (1 - resistance_reduction)
            A_eff = max(0.0, A_eff)
            
            # Calculate kill rate for this antibiotic using its toxicity constant
            # Higher toxicity_constant = more deadly antibiotic
            ec50_adjusted = GROWTH_PARAMS["ec50"] / ab_params["toxicity_constant"]
            
            kappa = GROWTH_PARAMS["emax"] * (A_eff**GROWTH_PARAMS["h"]) / \
                    (ec50_adjusted**GROWTH_PARAMS["h"] + A_eff**GROWTH_PARAMS["h"])
            
            # Apply repair mechanism (weighted by antibiotic's repair_weight)
            kappa *= (1 - ab_params["repair_weight"] * GROWTH_PARAMS["beta_r"] * self.expression["repair"])
            
            # Accumulate kill rates (additive effect of multiple antibiotics)
            total_kappa += kappa
        
        # Convert accumulated kill rate to death probability
        p_death = 1 - math.exp(-total_kappa * GROWTH_PARAMS["dt"])
        
        # Persistors have dramatically reduced kill probability (dormant cells less affected)
        if self.is_persistor:
            p_death *= PERSISTENCE_PARAMS["antibiotic_resistance_factor"]
        
        return random.random() < p_death

    def _move_towards_nutrients(self, fx, fy):
        """Move bacterium towards nutrient gradient
        
        Persistors move randomly with significantly reduced speed (no chemotaxis).
        """
        # Persistors move randomly without chemotaxis
        if self.is_persistor:
            # Pure random walk for persistors
            rand_dir = np.random.normal(size=2)
            rand_dir /= np.linalg.norm(rand_dir) + 1e-9
            direction = rand_dir
            
            # Significantly reduced speed for persistors
            effective_speed = self.speed * BACTERIA_SPEED * PERSISTENCE_PARAMS["movement_speed_factor"]
        else:
            # Normal chemotactic movement for active bacteria
            grad = self.model.compute_gradient_at_field(fx, fy)
            g = np.array(grad, dtype=float)
            
            if np.linalg.norm(g) > 1e-8:
                g = g / (np.linalg.norm(g) + 1e-9)
            else:
                g = np.zeros(2)
                
            rand_dir = np.random.normal(size=2)
            rand_dir /= np.linalg.norm(rand_dir) + 1e-9
            
            alpha = 0.8
            direction = alpha * g + (1 - alpha) * rand_dir
            direction /= np.linalg.norm(direction) + 1e-9
            
            effective_speed = self.speed * BACTERIA_SPEED

        # Calculate new position with proper boundary clamping
        new_x = self.pos[0] + direction[0] * effective_speed
        new_y = self.pos[1] + direction[1] * effective_speed
        
        # Clamp to valid range [0, width) and [0, height)
        epsilon = 1e-6
        new_x = max(0, min(self.model.width - epsilon, new_x))
        new_y = max(0, min(self.model.height - epsilon, new_y))

        # Update position
        new_pos = (float(new_x), float(new_y))
        try:
            self.model.space.move_agent(self, new_pos)
            self.pos = new_pos
        except Exception as e:
            print(f"Error moving agent {self.unique_id} to position {new_pos}: {e}")
            # Keep old position if move fails
            pass

    def _try_reproduce(self):
        """Attempt reproduction if energy threshold is met
        
        Persistors cannot reproduce (dormant state).
        """
        # Persistors cannot reproduce
        if self.is_persistor:
            return
            
        if self.energy < GROWTH_PARAMS["e_div"]:
            return
            
        self.energy /= 2.0
        
        # Create child with mutations
        mutated_traits = self.mutate_offspring_traits()
        
        # Calculate child position
        offset_x = random.uniform(-0.5, 0.5)
        offset_y = random.uniform(-0.5, 0.5)
        child_x = max(0, min(self.model.width, self.pos[0] + offset_x))
        child_y = max(0, min(self.model.height, self.pos[1] + offset_y))
        child_pos = (child_x, child_y)
        
        # Create child
        child = Bacterium(self.model, bacterial_type=self.bacterial_type)
        
        # Apply mutations with repair-dependent rate
        mu_eff = MUTATION_STD * (1 - 0.5 * self.expression["repair"])
        for trait in mutated_traits:
            delta = random.gauss(0, mu_eff)
            setattr(child, trait, max(0.0, min(1.0, mutated_traits[trait] + delta)))
        
        child.energy = self.energy
        self.model.new_agents.append((child, child_pos))

    def _check_persistor_entry(self, local_antibiotics):
        """Determine if bacterium should enter persistor state
        
        Args:
            local_antibiotics: dict mapping antibiotic_type -> concentration
        
        TODO: Currently uses sum of all antibiotics as temporary solution.
        Should be updated to have antibiotic-specific stress responses.
        
        Entry probability increases continuously with stress conditions:
        - Scales with antibiotic concentration (dose-dependent)
        - Inversely proportional to energy (starvation stress)
        
        Returns:
            bool: True if bacterium enters persistor state
        """
        if self.is_persistor:
            return False  # Already a persistor
        
        # TEMPORARY: Sum all antibiotic concentrations
        # TODO: Different antibiotics should cause different stress levels
        local_ab = sum(local_antibiotics.values()) if local_antibiotics else 0.0
        
        # Base probability
        prob = PERSISTENCE_PARAMS["base_entry_prob"]
        
        # Antibiotic stress: Hill-function style dose-dependent response
        # Probability increases sigmoidally with antibiotic concentration
        if local_ab > 0:
            ab_factor = (local_ab ** 2) / (PERSISTENCE_PARAMS["antibiotic_stress_threshold"] ** 2 + local_ab ** 2)
            prob += ab_factor * PERSISTENCE_PARAMS["stress_entry_multiplier"] * PERSISTENCE_PARAMS["base_entry_prob"]
        
        # Energy stress: probability increases as energy decreases
        # Normalized energy (0 = no energy, 1 = well-fed at e_div threshold)
        normalized_energy = self.energy / GROWTH_PARAMS["e_div"]
        if normalized_energy < 1.0:
            # Inverse relationship: lower energy -> higher probability
            energy_stress_factor = (1.0 - normalized_energy) ** 2  # Quadratic to emphasize low energy
            prob += energy_stress_factor * PERSISTENCE_PARAMS["stress_entry_multiplier"] * PERSISTENCE_PARAMS["base_entry_prob"]
        
        # Cap probability at reasonable maximum
        prob = min(prob, PERSISTENCE_PARAMS["max_entry_prob"])
        
        return random.random() < prob

    def _check_persistor_exit(self, local_antibiotics):
        """Determine if persistor should revert to normal state
        
        Args:
            local_antibiotics: dict mapping antibiotic_type -> concentration
        
        TODO: Currently uses sum of all antibiotics as temporary solution.
        Should be updated to consider specific antibiotic threats.
        
        Exit probability increases under favorable conditions:
        - No antibiotics present
        - Sufficient energy reserves
        
        Returns:
            bool: True if persistor exits to normal state
        """
        if not self.is_persistor:
            return False  # Not a persistor
        
        # TEMPORARY: Sum all antibiotic concentrations
        # TODO: Different antibiotics should have different thresholds for "safe" exit
        local_ab = sum(local_antibiotics.values()) if local_antibiotics else 0.0
        
        # Force exit if energy too low (starvation override)
        if self.energy < PERSISTENCE_PARAMS["min_persistor_energy"]:
            return True
        
        # Calculate favorable conditions
        no_antibiotics = local_ab < PERSISTENCE_PARAMS["antibiotic_stress_threshold"]
        sufficient_energy = self.energy > PERSISTENCE_PARAMS["energy_favorable_threshold"]
        
        # Base probability with favorable condition multiplier
        prob = PERSISTENCE_PARAMS["base_exit_prob"]
        if no_antibiotics and sufficient_energy:
            prob *= PERSISTENCE_PARAMS["favorable_exit_multiplier"]
        
        return random.random() < prob

    def step(self):
        """Execute one step of the bacterium lifecycle"""
        # Safety check: skip if position is None (being removed)
        if self.pos is None:
            return
            
        # Aging mechanism - persistors age slower
        if self.is_persistor:
            # Accumulate fractional aging for persistors
            self.age += PERSISTENCE_PARAMS["aging_rate_factor"]
        else:
            # Normal aging for active bacteria
            self.age += 1
        
        # Check for death by old age
        if self.age >= self.max_age:
            self.model.individual_tracker.mark_death(self.unique_id, 'old_age')
            self.model.to_remove.add(self)
            return

        # Get local conditions
        fx, fy = self.model.nutrient_to_field_coords(self.pos)
        local_food = self.model.sample_field(self.model.food_field, fx, fy)
        
        # Get individual antibiotic concentrations (dict: antibiotic_type -> concentration)
        local_antibiotics = self.model.get_antibiotic_concentrations_at_position(fx, fy)
        
        # Check for persistor state transitions (before other actions)
        if self._check_persistor_entry(local_antibiotics):
            self.is_persistor = True
        elif self._check_persistor_exit(local_antibiotics):
            self.is_persistor = False

        # Update expression states
        self._update_expression_states(local_antibiotics)

        # Consume nutrients and update energy
        self._consume_nutrients(local_food, fx, fy)

        # Check for starvation
        if self.energy <= 0:
            self.model.individual_tracker.mark_death(self.unique_id, 'starvation')
            self.model.to_remove.add(self)
            return

        # Check antibiotic death - now passes individual concentrations
        if self._check_antibiotic_death(local_antibiotics):
            self.model.individual_tracker.mark_death(self.unique_id, 'antibiotic')
            self.model.to_remove.add(self)
            return

        # Move towards nutrients
        self._move_towards_nutrients(fx, fy)

        # Try to reproduce
        self._try_reproduce()
