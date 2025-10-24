"""
Bacterium agent implementation.
"""

import math
import random
import numpy as np
from mesa import Agent

from config import (
    BACTERIAL_TYPES, ANTIBIOTIC_TYPES, ALLOCATION_PARAMS, 
    GROWTH_PARAMS, MUTATION_STD, BACTERIA_SPEED
)


class Bacterium(Agent):
    """Individual bacterium agent with resistance traits and lifecycle."""
    
    def __init__(self, model, bacterial_type="E.coli"):
        super().__init__(model)
        self.unique_id = model.next_id()
        self.pos = None
        self.energy = random.uniform(1.0, 2.0)
        self.bacterial_type = bacterial_type
        self.age = 0
        
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

    def _update_expression_states(self, local_ab):
        """Update gene expression based on antibiotic presence"""
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
        """Consume nutrients and update energy"""
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

    def _check_antibiotic_death(self, local_ab):
        """Check if bacterium dies from antibiotic exposure"""
        if local_ab <= 0:
            return False
            
        # Calculate effective antibiotic concentration
        A_eff = local_ab * (1 - sum(
            GROWTH_PARAMS["alpha"][k] * self.expression[k] 
            for k in ["efflux", "enzyme", "membrane"]
        ))
        A_eff = max(0.0, A_eff)
        
        # Calculate kill probability
        kappa = GROWTH_PARAMS["emax"] * (A_eff**GROWTH_PARAMS["h"]) / \
                (GROWTH_PARAMS["ec50"]**GROWTH_PARAMS["h"] + A_eff**GROWTH_PARAMS["h"])
        kappa *= (1 - GROWTH_PARAMS["beta_r"] * self.expression["repair"])
        p_death = 1 - math.exp(-kappa * GROWTH_PARAMS["dt"])
        
        return random.random() < p_death

    def _move_towards_nutrients(self, fx, fy):
        """Move bacterium towards nutrient gradient"""
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

        # Calculate new position with proper boundary clamping
        # Use slightly smaller bounds to avoid edge cases (exclusive upper bound)
        new_x = self.pos[0] + direction[0] * self.speed * BACTERIA_SPEED
        new_y = self.pos[1] + direction[1] * self.speed * BACTERIA_SPEED
        
        # Clamp to valid range [0, width) and [0, height)
        # Use a small epsilon to keep away from exact boundary
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
        """Attempt reproduction if energy threshold is met"""
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

    def step(self):
        """Execute one step of the bacterium lifecycle"""
        self.age += 1
        
        # Check for death by old age
        if self.age >= self.max_age:
            self.model.individual_tracker.mark_death(self.unique_id, 'old_age')
            self.model.to_remove.add(self)
            return

        # Get local conditions
        fx, fy = self.model.nutrient_to_field_coords(self.pos)
        local_food = self.model.sample_field(self.model.food_field, fx, fy)
        local_ab = self.model.sample_field(self.model.antibiotic_field, fx, fy)

        # Update expression states
        self._update_expression_states(local_ab)

        # Consume nutrients and update energy
        self._consume_nutrients(local_food, fx, fy)

        # Check for starvation
        if self.energy <= 0:
            self.model.individual_tracker.mark_death(self.unique_id, 'starvation')
            self.model.to_remove.add(self)
            return

        # Check antibiotic death
        if self._check_antibiotic_death(local_ab):
            self.model.individual_tracker.mark_death(self.unique_id, 'antibiotic')
            self.model.to_remove.add(self)
            return

        # Move towards nutrients
        self._move_towards_nutrients(fx, fy)

        # Try to reproduce
        self._try_reproduce()
