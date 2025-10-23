"""
Updated Bacteria simulation (Mesa model) with continuous space.
- Reworked to avoid deprecated Mesa schedulers (manage agents manually)
- Proper Model and Agent initialization
- Toggleable horizontal gene transfer (HGT) from UI
- Tk UI event pumping integrated into Matplotlib animation (no separate Tk thread)
- cache_frame_data disabled for FuncAnimation to avoid unbounded cache warning

Run: python mesa_bacteria_simulation.py
Dependencies: mesa, numpy, scipy, matplotlib, tkinter (optional)
"""

import sys
import time
import math
import random

import numpy as np
from scipy.ndimage import gaussian_filter

from mesa import Model, Agent
from mesa.space import ContinuousSpace

import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    tk = None

# -----------------------
# Configuration / Params
# -----------------------
WIDTH = 100.0                       # continuous width
HEIGHT = 100.0
GRID_RES = 200                      # resolution for nutrient & antibiotic fields (square grid)

INITIAL_BACTERIA = 20
FOOD_DIFFUSION_SIGMA = 1.0          # for gaussian_filter diffusion approximation
FOOD_DECAY = 0.3

# Growth model parameters
GROWTH_PARAMS = {
    "dt": 0.1,               # integration timestep
    "u_max": 1.0,            # max uptake rate
    "k_s": 0.5,              # half-saturation constant
    "eta": 1.0,              # conversion efficiency
    "m0": 0.01,              # maintenance cost
    "c_prod": 0.3,           # expression cost scale
    "k_i": 0.2,              # induction constant
    "n_ind": 1,              # Hill coefficient
    "emax": 1.5,             # max kill rate
    "ec50": 0.3,             # kill rate half-max
    "h": 1.0,                # kill rate Hill coefficient
    "beta_r": 0.5,           # repair effectiveness
    "e_div": 3.0,            # division threshold
    
    # Expression kinetics
    "ks": {"membrane": 0.2, "efflux": 1.0, "enzyme": 0.8, "repair": 0.5},
    "kd": {"membrane": 0.05, "efflux": 0.2, "enzyme": 0.2, "repair": 0.1},
    
    # Expression costs
    "expression_weights": {
        "membrane": 0.1,
        "efflux": 0.25,
        "enzyme": 0.35,
        "repair": 0.1
    },
    
    # Resistance effectiveness
    "alpha": {"efflux": 0.6, "enzyme": 0.7, "membrane": 0.4}
}

# Budget allocation model parameters
ALLOCATION_PARAMS = {
    "total_budget": 1.0,           # Total resource budget per bacterium
    "reallocation_std": 0.02,      # Standard deviation for small reallocations
    "amplification_prob": 0.05,    # Probability of trait amplification
    "amplification_cost": 0.2,     # Energy cost per unit of amplification
    "diminishing_return_alpha": 0.7 # Exponent for diminishing returns
}

BACTERIA_SPEED = 0.3               # scaling factor for bacterium movement speed
MUTATION_STD = 0.03
HGT_RADIUS = 1.5                  # horizontal gene transfer radius
HGT_PROB = 0.001                  # probability of HGT per neighbor per step

ANTIBIOTIC_DECAY = 0.05  # per sim step

CONTROL_INTERVAL = 20  # timesteps between control checks (UI applies when pressed)

# Simulation speed settings
DEFAULT_STEPS_PER_SECOND = 5    # simulation steps per second
MIN_STEPS_PER_SECOND = 1        # minimum speed (slowest)
MAX_STEPS_PER_SECOND = 20       # maximum speed (fastest)
ANIMATION_FPS = 30              # visual update rate (frames per second)

# -----------------------
# Bacterial Types and Antibiotic Definitions
# -----------------------

# Define bacterial types with resistance trait ranges
BACTERIAL_TYPES = {
    "E.coli": {
        "enzyme": (0.2, 0.1),      # mean, std
        "efflux": (0.3, 0.15),
        "membrane": (0.25, 0.1),
        "repair": (0.4, 0.2),
        "max_age": (50, 10),       # timesteps
        "base_speed": (0.8, 0.2),
        "color": "blue"
    },
    "Staph": {
        "enzyme": (0.4, 0.15),
        "efflux": (0.2, 0.1),
        "membrane": (0.5, 0.2),
        "repair": (0.3, 0.15),
        "max_age": (80, 15),
        "base_speed": (0.6, 0.15),
        "color": "red"
    },
    "Pseudomonas": {
        "enzyme": (0.6, 0.2),
        "efflux": (0.7, 0.25),
        "membrane": (0.3, 0.1),
        "repair": (0.5, 0.2),
        "max_age": (60, 12),
        "base_speed": (1.0, 0.3),
        "color": "green"
    }
}

# Define antibiotic types with their effectiveness weights
ANTIBIOTIC_TYPES = {
    "penicillin": {
        "enzyme_weight": 0.8,      # β-lactamase effectiveness
        "efflux_weight": 0.2,
        "membrane_weight": 0.3,
        "repair_weight": 0.4,
        "toxicity_constant": 2.0,
        "color": "red"
    },
    "tetracycline": {
        "enzyme_weight": 0.1,
        "efflux_weight": 0.9,      # efflux pumps very effective
        "membrane_weight": 0.4,
        "repair_weight": 0.3,
        "toxicity_constant": 1.5,
        "color": "orange"
    },
    "vancomycin": {
        "enzyme_weight": 0.2,
        "efflux_weight": 0.3,
        "membrane_weight": 0.8,    # membrane changes very effective
        "repair_weight": 0.6,
        "toxicity_constant": 3.0,
        "color": "purple"
    }
}

# Number of bacteria per type at initialization
BACTERIA_PER_TYPE = 1

# -----------------------
# Agent definition
# -----------------------
class Bacterium(Agent):
    def __init__(self, model, pos, bacterial_type="E.coli"):
        # Correct Agent initialization signature: Agent(unique_id, model)
        super().__init__(model)
        self.unique_id = model.next_id()
        self.pos = pos
        self.energy = random.uniform(1.0, 2.0)
        self.bacterial_type = bacterial_type
        self.age = 0
        
        # Generate initial trait allocations based on bacterial type
        type_def = BACTERIAL_TYPES[bacterial_type]
        total = sum(type_def[trait][0] for trait in ['enzyme', 'efflux', 'membrane', 'repair'])
        
        # Normalize initial allocations to budget
        budget = ALLOCATION_PARAMS["total_budget"]
        self.enzyme = max(0.0, min(budget * type_def["enzyme"][0] / total, budget))
        self.efflux = max(0.0, min(budget * type_def["efflux"][0] / total, budget))
        self.membrane = max(0.0, min(budget * type_def["membrane"][0] / total, budget))
        self.repair = max(0.0, min(budget * type_def["repair"][0] / total, budget))
        
        # Other initializations remain the same
        self.max_age = max(10, int(random.gauss(*type_def["max_age"])))
        self.speed = max(0.1, random.gauss(*type_def["base_speed"]))

        # Expression states (X) start at 0
        self.expression = {
            "membrane": 0.0,
            "efflux": 0.0,
            "enzyme": 0.0,
            "repair": 0.0
        }

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
            # Choose random trait for amplification
            amp_idx = random.randrange(len(traits))
            amp_amount = random.uniform(0, 0.2)  # up to 20% amplification
            deltas[amp_idx] += amp_amount
            
            # Apply metabolic cost
            self.energy -= amp_amount * ALLOCATION_PARAMS["amplification_cost"]
        
        # Apply changes while ensuring constraints
        new_allocations = current_allocations + deltas
        
        # Ensure non-negativity
        new_allocations = np.maximum(new_allocations, 0)
        
        # Normalize to maintain budget
        new_allocations *= budget / new_allocations.sum()
        
        # Convert to trait dictionary
        mutated_traits = {
            trait: float(alloc) for trait, alloc in zip(traits, new_allocations)
        }
        
        return mutated_traits

    def calculate_survival_probability(self, antibiotic_conc, antibiotic_type):
        """Modified to include diminishing returns on trait effectiveness"""
        if antibiotic_conc <= 0:
            return 1.0
            
        ab_def = ANTIBIOTIC_TYPES[antibiotic_type]
        alpha = ALLOCATION_PARAMS["diminishing_return_alpha"]
        
        # Apply diminishing returns to trait effectiveness
        effective_traits = {
            "efflux": self.efflux ** alpha,
            "enzyme": self.enzyme ** alpha,
            "membrane": self.membrane ** alpha,
            "repair": self.repair ** alpha
        }
        
        # Calculate effective antibiotic concentration
        A_eff = antibiotic_conc * \
                (1 - ab_def["efflux_weight"] * effective_traits["efflux"]) * \
                (1 - ab_def["enzyme_weight"] * effective_traits["enzyme"]) * \
                (1 - ab_def["membrane_weight"] * effective_traits["membrane"])
        
        A_eff = max(0.0, A_eff)
        
        # Calculate survival probability with diminishing returns on repair
        damage_factor = A_eff * (1 - ab_def["repair_weight"] * effective_traits["repair"])
        survival_prob = math.exp(-ab_def["toxicity_constant"] * damage_factor)
        
        return min(1.0, max(0.0, survival_prob))

    def step(self):
        # Age the bacterium
        self.age += 1
        
        # Check for death by old age
        if self.age >= self.max_age:
            self.model.to_remove.add(self)
            return

        # Get local conditions
        fx, fy = self.model.nutrient_to_field_coords(self.pos)
        local_food = self.model.sample_field(self.model.food_field, fx, fy)
        local_ab = self.model.sample_field(self.model.antibiotic_field, fx, fy)

        # 1) Update expression states based on antibiotic presence
        if local_ab > 0:
            S_A = (local_ab**GROWTH_PARAMS["n_ind"]) / (GROWTH_PARAMS["k_i"]**GROWTH_PARAMS["n_ind"] + local_ab**GROWTH_PARAMS["n_ind"])
        else:
            S_A = 0.0
        
        for k in self.expression:
            g_i = getattr(self, k)  # get genome capacity
            X = self.expression[k]   # current expression
            dX = (GROWTH_PARAMS["ks"][k] * g_i * S_A - GROWTH_PARAMS["kd"][k] * X) * GROWTH_PARAMS["dt"]
            self.expression[k] = max(0.0, min(1.0, X + dX))

        # 2) Compute nutrient uptake (Monod kinetics)
        uptake = GROWTH_PARAMS["u_max"] * (local_food / (GROWTH_PARAMS["k_s"] + local_food))
        
        # Subtract consumed nutrients from field
        self.model.subtract_from_field(self.model.food_field, fx, fy, uptake * GROWTH_PARAMS["dt"])
        
        # 3) Expression cost
        expr_cost = GROWTH_PARAMS["c_prod"] * sum(GROWTH_PARAMS["expression_weights"][k] * self.expression[k] 
                                for k in self.expression)
        
        # 4) Energy update
        dE = (GROWTH_PARAMS["eta"] * uptake - GROWTH_PARAMS["m0"] - expr_cost) * GROWTH_PARAMS["dt"]
        self.energy = max(0.0, self.energy + dE)

        # Death by starvation
        if self.energy <= 0:
            self.model.to_remove.add(self)
            return

        # 5) Antibiotic kill effect
        if local_ab > 0:
            # Calculate effective antibiotic concentration
            A_eff = local_ab * (1 - sum(GROWTH_PARAMS["alpha"][k] * self.expression[k] 
                              for k in ["efflux", "enzyme", "membrane"]))
            A_eff = max(0.0, A_eff)
            
            # Calculate kill probability
            kappa = GROWTH_PARAMS["emax"] * (A_eff**GROWTH_PARAMS["h"]) / (GROWTH_PARAMS["ec50"]**GROWTH_PARAMS["h"] + A_eff**GROWTH_PARAMS["h"])
            kappa *= (1 - GROWTH_PARAMS["beta_r"] * self.expression["repair"])
            p_death = 1 - math.exp(-kappa * GROWTH_PARAMS["dt"])
            
            if random.random() < p_death:
                self.model.to_remove.add(self)
                return

        # 6) Movement: biased random walk toward nutrient gradient
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

        # Move
        new_x = self.pos[0] + direction[0] * self.speed * BACTERIA_SPEED
        new_y = self.pos[1] + direction[1] * self.speed * BACTERIA_SPEED
        new_x = max(0, min(self.model.space.x_max, new_x))
        new_y = max(0, min(self.model.space.y_max, new_y))

        # Update position using ContinuousSpace API
        self.pos = (new_x, new_y)
        try:
            self.model.space.move_agent(self, self.pos)
        except Exception:
            raise Exception("Agent movement failed")

        # 7) Reproduction based on energy threshold
        if self.energy >= GROWTH_PARAMS["e_div"]:
            self.energy /= 2.0  # Split energy between parent and child
            
            # Create child with mutations
            mutated_traits = self.mutate_offspring_traits()
            
            # Give child a slightly offset position
            offset_x = random.uniform(-0.5, 0.5)
            offset_y = random.uniform(-0.5, 0.5)
            child_x = max(0, min(self.model.width, self.pos[0] + offset_x))
            child_y = max(0, min(self.model.height, self.pos[1] + offset_y))
            child_pos = (child_x, child_y)
            
            child = Bacterium(self.model, pos=child_pos, bacterial_type=self.bacterial_type)
            
            # Apply mutations with repair-dependent rate
            mu_eff = MUTATION_STD * (1 - 0.5 * self.expression["repair"])
            for trait in mutated_traits:
                delta = random.gauss(0, mu_eff)
                setattr(child, trait, max(0.0, min(1.0, mutated_traits[trait] + delta)))
            
            child.energy = self.energy  # Give child half energy
            
            self.model.new_agents.append(child)

    def advance(self):
        # placeholder if later one wants two-phase updates
        pass


# -----------------------
# Model definition
# -----------------------
class BacteriaModel(Model):
    def __init__(self, N=None, width=WIDTH, height=HEIGHT, enable_hgt=True):
        super().__init__()  # explicit model init to avoid FutureWarning
        self.width = width
        self.height = height
        self.space = ContinuousSpace(width, height, torus=False)
        self.random = random.Random()

        # agent container instead of deprecated scheduler
        self.agent_set = set()
        self._next_id = 0

        # Antibiotic management
        self.current_antibiotic = "penicillin"  # Default antibiotic type
        self.available_antibiotics = list(ANTIBIOTIC_TYPES.keys())

        # fields
        self.field_w = GRID_RES
        self.field_h = GRID_RES
        self.food_field = np.zeros((self.field_w, self.field_h), dtype=float)
        self.antibiotic_field = np.zeros_like(self.food_field)

        # initialize food with several gaussian patches
        for _ in range(6):
            cx = random.uniform(0, self.field_w - 1)
            cy = random.uniform(0, self.field_h - 1)
            sigma = random.uniform(6, 18)
            amplitude = random.uniform(2.0, 5.0)
            self.add_gaussian_patch(self.food_field, cx, cy, sigma, amplitude)

        self.to_remove = set()
        self.new_agents = []

        # Create agents based on bacterial types
        if N is None:
            # Create Z bacteria per type
            total_bacteria = len(BACTERIAL_TYPES) * BACTERIA_PER_TYPE
        else:
            total_bacteria = N
            
        bacteria_per_type = total_bacteria // len(BACTERIAL_TYPES)
        remainder = total_bacteria % len(BACTERIAL_TYPES)
        
        for i, bacterial_type in enumerate(BACTERIAL_TYPES.keys()):
            # Distribute remainder among first types
            count = bacteria_per_type + (1 if i < remainder else 0)
            
            for _ in range(count):
                x, y = random.uniform(0, width), random.uniform(0, height)
                bacterium = Bacterium(self, (x, y), bacterial_type=bacterial_type)
                self.agent_set.add(bacterium)
                # Place agent in the space
                self.space.place_agent(bacterium, (x, y))

        self.running = True
        self.step_count = 0

        # HGT toggle
        self.enable_hgt = bool(enable_hgt)

        # Add tracking arrays for plotting
        self.history = {
            'steps': [],
            'population': [],
            'total_food': [],
            'avg_energy': []
        }

    def next_id(self):
        nid = self._next_id
        self._next_id += 1
        return nid

    def set_antibiotic_type(self, antibiotic_type):
        """Change the current antibiotic being used"""
        if antibiotic_type in ANTIBIOTIC_TYPES:
            self.current_antibiotic = antibiotic_type
            print(f"Switched to {antibiotic_type}")
        else:
            print(f"Unknown antibiotic type: {antibiotic_type}")

    def get_population_stats(self):
        """Get statistics about the current population"""
        stats = {
            "total": len(self.agent_set),
            "by_type": {},
            "avg_traits": {},
            "avg_age": 0
        }
        
        if len(self.agent_set) == 0:
            return stats
            
        # Count by type and collect traits
        trait_sums = {"enzyme": 0, "efflux": 0, "membrane": 0, "repair": 0, "age": 0}
        
        for bacterium in self.agent_set:
            btype = bacterium.bacterial_type
            if btype not in stats["by_type"]:
                stats["by_type"][btype] = 0
                stats[btype] = {"enzyme": 0, "efflux": 0, "membrane": 0, "repair": 0, "age": 0}
            stats["by_type"][btype] += 1
            stats[btype]["enzyme"] += bacterium.enzyme
            stats[btype]["efflux"] += bacterium.efflux
            stats[btype]["membrane"] += bacterium.membrane
            stats[btype]["repair"] += bacterium.repair
            stats[btype]["age"] += bacterium.age
            
            
            trait_sums["enzyme"] += bacterium.enzyme
            trait_sums["efflux"] += bacterium.efflux
            trait_sums["membrane"] += bacterium.membrane
            trait_sums["repair"] += bacterium.repair
            trait_sums["age"] += bacterium.age
        
        # Calculate averages
        total = len(self.agent_set)
        for trait in trait_sums:
            stats["avg_traits"][trait] = trait_sums[trait] / total
        for btype in stats["by_type"]:
            count = stats["by_type"][btype]
            stats[btype]["enzyme"] /= count
            stats[btype]["efflux"] /= count
            stats[btype]["membrane"] /= count
            stats[btype]["repair"] /= count
            stats[btype]["age"] /= count
        
        # Add food and energy tracking
        total_food = np.sum(self.food_field)
        avg_energy = np.mean([a.energy for a in self.agent_set]) if self.agent_set else 0
        
        # Calculate top 10 energy instead of average
        top_energies = sorted([a.energy for a in self.agent_set], reverse=True)[:10]
        avg_top_energy = np.mean(top_energies) if top_energies else 0
        
        # Record history
        self.history['steps'].append(self.step_count)
        self.history['population'].append(len(self.agent_set))
        self.history['total_food'].append(total_food)
        self.history['avg_energy'].append(avg_top_energy)  # Now stores top 10 average
        
        # Add to stats
        stats['total_food'] = total_food
        stats['avg_energy'] = avg_top_energy
        stats['top_energies'] = top_energies  # Add full list for detailed display
        
        return stats

    # ---------------------
    # Field utilities
    # ---------------------
    def add_gaussian_patch(self, field, cx, cy, sigma, amplitude):
        X, Y = np.meshgrid(
            np.arange(self.field_w), np.arange(self.field_h), indexing="ij"
        )
        patch = amplitude * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))
        field += patch

    def nutrient_to_field_coords(self, pos):
        fx = (pos[0] / self.width) * (self.field_w - 1)
        fy = (pos[1] / self.height) * (self.field_h - 1)
        return fx, fy

    def sample_field(self, field, fx, fy):
        x0 = int(np.floor(fx))
        y0 = int(np.floor(fy))
        x1 = min(x0 + 1, self.field_w - 1)
        y1 = min(y0 + 1, self.field_h - 1)
        dx = fx - x0
        dy = fy - y0
        v00 = field[x0, y0]
        v10 = field[x1, y0]
        v01 = field[x0, y1]
        v11 = field[x1, y1]
        v = (
            v00 * (1 - dx) * (1 - dy)
            + v10 * dx * (1 - dy)
            + v01 * (1 - dx) * dy
            + v11 * dx * dy
        )
        return v

    def subtract_from_field(self, field, fx, fy, amount):
        x = int(round(fx))
        y = int(round(fy))
        x = min(max(x, 0), self.field_w - 1)
        y = min(max(y, 0), self.field_h - 1)
        field[x, y] = max(0.0, field[x, y] - amount)

    def compute_gradient_at_field(self, fx, fy):
        x = int(round(fx))
        y = int(round(fy))
        x0 = min(max(x - 1, 0), self.field_w - 1)
        x1 = min(max(x + 1, 0), self.field_w - 1)
        y0 = min(max(y - 1, 0), self.field_h - 1)
        y1 = min(max(y + 1, 0), self.field_h - 1)
        gx = self.food_field[x1, y] - self.food_field[x0, y]
        gy = self.food_field[x, y1] - self.food_field[x, y0]
        gx *= self.field_w / self.width
        gy *= self.field_h / self.height
        return gx, gy

    # ---------------------
    # Antibiotic control
    # ---------------------
    def apply_antibiotic(self, amount):
        if amount <= 0:
            return
        self.antibiotic_field += float(amount)

    # ---------------------
    # HGT: exchange resistance traits when bacteria are close
    # ---------------------
    def horizontal_gene_transfer(self):
        agents = list(self.agent_set)
        for i, a in enumerate(agents):
            # use ContinuousSpace neighbors lookup
            try:
                neighbors = self.space.get_neighbors(
                    a.pos, HGT_RADIUS, include_center=False
                )
            except Exception:
                # fallback: brute force
                neighbors = [
                    b
                    for b in agents
                    if b is not a
                    and np.hypot(b.pos[0] - a.pos[0], b.pos[1] - a.pos[1]) <= HGT_RADIUS
                ]
            for nb in neighbors:
                if random.random() < HGT_PROB:
                    mix = 0.3  # How much to exchange
                    # Exchange each resistance trait
                    traits = ['enzyme', 'efflux', 'membrane', 'repair']
                    for trait in traits:
                        a_val = getattr(a, trait)
                        nb_val = getattr(nb, trait)
                        new_a_val = a_val * (1 - mix) + nb_val * mix
                        new_nb_val = nb_val * (1 - mix) + a_val * mix
                        setattr(a, trait, float(min(max(new_a_val, 0.0), 1.0)))
                        setattr(nb, trait, float(min(max(new_nb_val, 0.0), 1.0)))

    # ---------------------
    # Step
    # ---------------------
    def step(self):
        # Update fields: diffuse nutrient and antibiotic
        self.food_field = gaussian_filter(self.food_field, sigma=FOOD_DIFFUSION_SIGMA)
        self.antibiotic_field *= 1 - ANTIBIOTIC_DECAY

        # Prepare collections
        self.to_remove.clear()
        self.new_agents.clear()

        # Step each agent
        for a in list(self.agent_set):
            # print(f"Stepping bacterium {a.unique_id}, age: {a.age}, energy: {a.energy:.3f}")
            try:
                a.step()
            except Exception as e:
                print(f"Exception during step for bacterium {a.unique_id}: {e}")
                # Avoid one agent failing stopping the sim
                pass
        # print(f"Agents to remove this step: {len(self.to_remove)}")
        # print(f"New agents this step: {len(self.new_agents)}")

        # Remove dead agents
        for a in list(self.to_remove):
            try:
                # remove from space and agent set
                try:
                    self.space.remove_agent(a)
                except Exception:
                    pass
                if a in self.agent_set:
                    self.agent_set.remove(a)
            except Exception:
                pass

        # Add newborns
        for child in self.new_agents:
            try:
                self.space.place_agent(child, child.pos)
            except Exception:
                pass
            self.agent_set.add(child)

        # Horizontal gene transfer (toggleable)
        if self.enable_hgt:
            try:
                self.horizontal_gene_transfer()
            except Exception:
                pass

        self.step_count += 1


# -----------------------
# Visualization + Control UI
# -----------------------
class SimulatorUI:
    def __init__(self, model):
        self.model = model
        self.paused = False
        self.latest_dose = 0.0
        
        # Create dynamic color mapping for bacterial types
        self.bacterial_type_names = list(BACTERIAL_TYPES.keys())
        self.color_map = {name: i for i, name in enumerate(self.bacterial_type_names)}
        
        # Speed control variables
        self.steps_per_second = DEFAULT_STEPS_PER_SECOND
        self.animation_fps = ANIMATION_FPS
        self.animation_interval = int(1000 / self.animation_fps)
        self.steps_accumulator = 0.0
        self.animation = None
        
        # Setup matplotlib figure with four subplots (1x4 layout)
        self.fig = plt.figure(figsize=(16, 4))
        gs = self.fig.add_gridspec(1, 4)
        
        # Main simulation view (leftmost)
        self.ax = self.fig.add_subplot(gs[0])
        
        # Food level plot
        self.ax_food = self.fig.add_subplot(gs[1])
        self.ax_food.set_xlabel('Steps')
        self.ax_food.set_ylabel('Total Food')
        self.ax_food.grid(True)
        self.line_food, = self.ax_food.plot([], [], label='Food Level', color='green')
        self.ax_food.legend()
        
        # Population plot
        self.ax_pop = self.fig.add_subplot(gs[2])
        self.ax_pop.set_xlabel('Steps')
        self.ax_pop.set_ylabel('Population')
        self.ax_pop.grid(True)
        self.line_pop, = self.ax_pop.plot([], [], label='Population', color='blue')
        self.ax_pop.legend()
        
        # Energy plot (modified label)
        self.ax_energy = self.fig.add_subplot(gs[3])
        self.ax_energy.set_xlabel('Steps')
        self.ax_energy.set_ylabel('Energy (Top 10 Avg)')
        self.ax_energy.grid(True)
        self.line_energy, = self.ax_energy.plot([], [], label='Top 10 Energy', color='red')
        self.ax_energy.legend()
        
        # Adjust layout
        self.fig.tight_layout()
        
        self.scat = None
        self.im_food = None
        self.im_ab = None

        # Start Tk UI if available
        if tk is not None:
            try:
                self.root = tk.Tk()
                self.root.title("Control Panel")
                self.build_controls()
            except Exception as e:
                print(f"Tk UI init failed: {e}")
                self.root = None
        else:
            self.root = None

    def get_bacterial_colors(self, agents):
        """Get numerical colors for bacterial types"""
        return [self.color_map.get(a.bacterial_type, 0) for a in agents]

    def build_controls(self):
        frm = ttk.Frame(self.root, padding=8)
        frm.grid()
        ttk.Label(frm, text="Simulation Controls").grid(column=0, row=0, columnspan=2)

        self.pause_btn = ttk.Button(frm, text="Pause", command=self.toggle_pause)
        self.pause_btn.grid(column=0, row=1)
        ttk.Button(frm, text="Reset", command=self.reset_sim).grid(column=1, row=1)

        # Speed controls
        ttk.Label(frm, text="Speed Control").grid(column=0, row=2, columnspan=2, pady=(10,5))
        
        speed_frame = ttk.Frame(frm)
        speed_frame.grid(column=0, row=3, columnspan=2, pady=5)
        
        ttk.Button(speed_frame, text="<<", command=self.speed_slower, width=3).grid(column=0, row=0)
        ttk.Button(speed_frame, text=">>", command=self.speed_faster, width=3).grid(column=1, row=0)
        ttk.Button(speed_frame, text="Reset Speed", command=self.speed_reset).grid(column=2, row=0, padx=(5,0))
        
        self.speed_label = ttk.Label(frm, text=f"Speed: {self.steps_per_second} steps/sec")
        self.speed_label.grid(column=0, row=4, columnspan=2)

        # Antibiotic type selection
        ttk.Label(frm, text="Antibiotic Type:").grid(column=0, row=5, pady=(10,5))
        self.antibiotic_var = tk.StringVar(value=self.model.current_antibiotic)
        self.antibiotic_combo = ttk.Combobox(frm, textvariable=self.antibiotic_var, 
                                           values=self.model.available_antibiotics, 
                                           state="readonly", width=12)
        self.antibiotic_combo.grid(column=1, row=5, pady=(10,5))
        self.antibiotic_combo.bind('<<ComboboxSelected>>', self.change_antibiotic)

        ttk.Label(frm, text="Antibiotic dose:").grid(column=0, row=6)
        self.dose_var = tk.DoubleVar(value=0.5)
        self.dose_entry = ttk.Entry(frm, textvariable=self.dose_var, width=8)
        self.dose_entry.grid(column=1, row=6)

        ttk.Button(frm, text="Apply antibiotic", command=self.apply_antibiotic_ui).grid(
            column=0, row=7, columnspan=2, pady=(4, 4)
        )

        ttk.Label(frm, text="Latest dose applied:").grid(column=0, row=8)
        self.latest_label = ttk.Label(frm, text="0.0")
        self.latest_label.grid(column=1, row=8)

        # HGT toggle
        self.hgt_var = tk.BooleanVar(value=self.model.enable_hgt)
        self.hgt_check = ttk.Checkbutton(
            frm, text="Enable HGT", variable=self.hgt_var, command=self.toggle_hgt
        )
        self.hgt_check.grid(column=0, row=9, columnspan=2)

        # Population stats display
        ttk.Label(frm, text="Population Stats", font=("TkDefaultFont", 9, "bold")).grid(
            column=0, row=10, columnspan=2, pady=(10,5))
        
        self.stats_frame = ttk.Frame(frm)
        self.stats_frame.grid(column=0, row=11, columnspan=2, sticky="ew")
        
        self.stats_labels = {}

    def change_antibiotic(self, event=None):
        """Change the antibiotic type when user selects from dropdown"""
        try:
            new_antibiotic = self.antibiotic_var.get()
            self.model.set_antibiotic_type(new_antibiotic)
        except Exception as e:
            print(f"Error changing antibiotic: {e}")

    def update_stats_display(self):
        """Update the population statistics display without flickering"""
        if self.root is None:
            return
            
        try:
            stats = self.model.get_population_stats()
            
            # Create labels only if they don't exist
            if not hasattr(self, '_stats_initialized'):
                self._create_stats_labels()
                self._stats_initialized = True
            
            # Update existing labels with new values
            self.stats_labels["total"].config(text=f"Total: {stats['total']}")
            
            # Update population by type
            for btype in BACTERIAL_TYPES.keys():
                count = stats["by_type"].get(btype, 0)
                self.stats_labels[f"type_{btype}"].config(text=f"{btype}: {count}")
                
                # Update per-type traits if the type has population
                if count > 0 and btype in stats:
                    type_stats = stats[btype]
                    self.stats_labels[f"{btype}_enzyme"].config(text=f"  enzyme: {type_stats['enzyme']:.3f}")
                    self.stats_labels[f"{btype}_efflux"].config(text=f"  efflux: {type_stats['efflux']:.3f}")
                    self.stats_labels[f"{btype}_membrane"].config(text=f"  membrane: {type_stats['membrane']:.3f}")
                    self.stats_labels[f"{btype}_repair"].config(text=f"  repair: {type_stats['repair']:.3f}")
                    self.stats_labels[f"{btype}_age"].config(text=f"  age: {type_stats['age']:.1f}")
                else:
                    # Clear per-type traits when no population of this type
                    self.stats_labels[f"{btype}_enzyme"].config(text="  enzyme: 0.000")
                    self.stats_labels[f"{btype}_efflux"].config(text="  efflux: 0.000")
                    self.stats_labels[f"{btype}_membrane"].config(text="  membrane: 0.000")
                    self.stats_labels[f"{btype}_repair"].config(text="  repair: 0.000")
                    self.stats_labels[f"{btype}_age"].config(text="  age: 0.0")
            
            # Update overall average traits (if population exists)
            if stats["total"] > 0:
                for trait, value in stats["avg_traits"].items():
                    if trait != "age":  # age is handled separately
                        self.stats_labels[f"avg_{trait}"].config(text=f"  {trait}: {value:.3f}")
                self.stats_labels["avg_age"].config(text=f"  age: {stats['avg_traits']['age']:.1f}")
            else:
                # Clear overall traits when no population
                for trait in ["enzyme", "efflux", "membrane", "repair"]:
                    self.stats_labels[f"avg_{trait}"].config(text=f"  {trait}: 0.000")
                self.stats_labels["avg_age"].config(text="  age: 0.0")
                
        except Exception as e:
            print(f"Error updating stats: {e}")

    def _create_stats_labels(self):
        """Create all stats labels once"""
        self.stats_labels = {}
        
        row = 0
        # Total population
        self.stats_labels["total"] = ttk.Label(self.stats_frame, text="Total: 0")
        self.stats_labels["total"].grid(column=0, row=row, columnspan=2, sticky="w")
        row += 1
        
        # Population by type with per-type stats
        for btype in BACTERIAL_TYPES.keys():
            # Type count
            self.stats_labels[f"type_{btype}"] = ttk.Label(self.stats_frame, text=f"{btype}: 0", 
                                                         font=("TkDefaultFont", 8, "bold"))
            self.stats_labels[f"type_{btype}"].grid(column=0, row=row, columnspan=2, sticky="w")
            row += 1
            
            # Per-type traits (indented)
            for trait in ["enzyme", "efflux", "membrane", "repair", "age"]:
                self.stats_labels[f"{btype}_{trait}"] = ttk.Label(self.stats_frame, 
                                                                text=f"  {trait}: 0.000" if trait != "age" else "  age: 0.0")
                self.stats_labels[f"{btype}_{trait}"].grid(column=0, row=row, columnspan=2, sticky="w", padx=(10,0))
                row += 1
        
        # Overall average traits header
        ttk.Label(self.stats_frame, text="Overall Avg:", 
                 font=("TkDefaultFont", 8, "bold")).grid(column=0, row=row, columnspan=2, sticky="w")
        row += 1
        
        # Overall average traits
        for trait in ["enzyme", "efflux", "membrane", "repair"]:
            self.stats_labels[f"avg_{trait}"] = ttk.Label(self.stats_frame, text=f"  {trait}: 0.000")
            self.stats_labels[f"avg_{trait}"].grid(column=0, row=row, columnspan=2, sticky="w")
            row += 1
        
        # Overall average age
        self.stats_labels["avg_age"] = ttk.Label(self.stats_frame, text="  age: 0.0")
        self.stats_labels["avg_age"].grid(column=0, row=row, columnspan=2, sticky="w")

    def toggle_pause(self):
        self.paused = not self.paused
        print("Paused" if self.paused else "Resumed")
        if self.paused:
            self.pause_btn.config(text="Resume")
        else:
            self.pause_btn.config(text="Pause")

    def reset_sim(self):
        print("Reset not implemented in this prototype")

    def apply_antibiotic_ui(self):
        try:
            # Get value directly from the entry widget instead of relying on textvariable
            val = float(self.dose_entry.get())
        except Exception:
            val = 0.0
        self.model.apply_antibiotic(val)
        self.latest_dose = val
        if self.root is not None:
            try:
                self.latest_label.config(text=f"{val:.3f}")
            except Exception:
                pass

    def toggle_hgt(self):
        try:
            new_val = bool(self.hgt_var.get())
        except Exception:
            new_val = not self.model.enable_hgt
        self.model.enable_hgt = new_val

    def speed_faster(self):
        """Increase simulation steps per second"""
        self.steps_per_second = min(MAX_STEPS_PER_SECOND, self.steps_per_second + 1)
        self.update_speed_display()

    def speed_slower(self):
        """Decrease simulation steps per second"""
        self.steps_per_second = max(MIN_STEPS_PER_SECOND, self.steps_per_second - 1)
        self.update_speed_display()

    def speed_reset(self):
        """Reset speed to default"""
        self.steps_per_second = DEFAULT_STEPS_PER_SECOND
        self.update_speed_display()

    def update_speed_display(self):
        """Update the speed label"""
        if self.root is not None:
            try:
                self.speed_label.config(text=f"Speed: {self.steps_per_second} steps/sec")
            except Exception:
                pass

    def init_plot(self):
        self.ax.set_xlim(0, self.model.width)
        self.ax.set_ylim(0, self.model.height)
        self.ax.set_aspect("equal")
        food_img = np.rot90(self.model.food_field)
        self.im_food = self.ax.imshow(
            food_img,
            extent=[0, self.model.width, 0, self.model.height],
            alpha=0.6,
            cmap="Greens",
        )
        ab_img = np.rot90(self.model.antibiotic_field)
        self.im_ab = self.ax.imshow(
            ab_img,
            extent=[0, self.model.width, 0, self.model.height],
            alpha=0.35,
            cmap="Reds",
        )
        xs = [a.pos[0] for a in self.model.agent_set]
        ys = [a.pos[1] for a in self.model.agent_set]
        # Use numerical color mapping
        colors = self.get_bacterial_colors(self.model.agent_set)
        self.scat = self.ax.scatter(xs, ys, c=colors, s=20, cmap="tab10")
        return (self.scat,)

    def pump_tk(self):
        # Pump tkinter events from the main thread (safe) so we don't run mainloop in another thread
        if self.root is not None:
            try:
                self.root.update_idletasks()
                self.root.update()
            except tk.TclError:
                # If the window has been closed or errors occur, ignore
                pass
            except Exception:
                pass

    def update_plot(self, frame):
        self.pump_tk()

        if not self.paused:
            # Calculate how many simulation steps to run this frame
            steps_per_frame = self.steps_per_second / self.animation_fps
            self.steps_accumulator += steps_per_frame
            
            # Run integer number of steps
            steps_to_run = int(self.steps_accumulator)
            self.steps_accumulator -= steps_to_run
            
            # Run the simulation steps
            for _ in range(steps_to_run):
                try:
                    self.model.step()
                except Exception:
                    pass

        # Update main simulation view
        try:
            food_img = np.rot90(self.model.food_field)
            self.im_food.set_data(food_img)
            ab_img = np.rot90(self.model.antibiotic_field)
            self.im_ab.set_data(ab_img)
            
            xs = [a.pos[0] for a in self.model.agent_set]
            ys = [a.pos[1] for a in self.model.agent_set]
            
            if len(xs) == 0:
                self.scat.set_offsets(np.empty((0, 2)))
                self.scat.set_array(np.array([]))
            else:
                colors = self.get_bacterial_colors(self.model.agent_set)
                self.scat.set_offsets(np.c_[xs, ys])
                self.scat.set_array(np.array(colors))
                
            # Update all plots separately
            history = self.model.history
            steps = history['steps']
            if len(steps) > 0:
                # Update food plot
                self.line_food.set_data(steps, history['total_food'])
                self.ax_food.set_xlim(0, max(steps))
                self.ax_food.set_ylim(0, max(history['total_food']) * 1.1)
                
                # Update population plot
                self.line_pop.set_data(steps, history['population'])
                self.ax_pop.set_xlim(0, max(steps))
                self.ax_pop.set_ylim(0, max(history['population']) * 1.1)
                
                # Update energy plot
                self.line_energy.set_data(steps, history['avg_energy'])
                self.ax_energy.set_xlim(0, max(steps))
                self.ax_energy.set_ylim(0, max(history['avg_energy']) * 1.1)
                
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"Plot update error: {e}")
            pass

        self.ax.set_title(
            f"Step: {self.model.step_count}  Agents: {len(self.model.agent_set)}  Antibiotic: {self.model.current_antibiotic}"
        )
        self.update_stats_display()
        return (self.scat,)

    def run(self):
        self.animation = animation.FuncAnimation(
            self.fig,
            self.update_plot,
            init_func=self.init_plot,
            interval=self.animation_interval,  # Use configurable interval
            blit=False,
            cache_frame_data=False,  # avoid unbounded cache warning
        )
        plt.show()


# -----------------------
# Entrypoint
# -----------------------
def main():
    model = BacteriaModel(N=INITIAL_BACTERIA)
    ui = SimulatorUI(model)
    ui.run()


if __name__ == "__main__":
    main()
# Entrypoint
# -----------------------
def main():
    model = BacteriaModel(N=INITIAL_BACTERIA)
    ui = SimulatorUI(model)
    ui.run()


if __name__ == "__main__":
    main()
