"""
Updated Bacteria simulation (Mesa model) with continuous space.
- Reworked to avoid deprecated Mesa schedulers (manage agents manually)
- Proper Model and Agent initialization
- Toggleable horizontal gene transfer (HGT) from UI
- Tk UI event pumping integrated into Matplotlib animation (no separate Tk thread)
- cache_frame_data disabled for FuncAnimation to avoid unbounded cache warning
- Individual bacteria tracking and plotting system

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
from collections import defaultdict, deque

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

INITIAL_BACTERIA = 1
FOOD_DIFFUSION_SIGMA = 1.0          # for gaussian_filter diffusion approximation
FOOD_DECAY = 0.1
FOOD_CONSUMPTION_PER_STEP = 0.1     # Increased from 0.01
BACTERIA_SPEED = 0.3                # scaling factor for bacterium movement speed (each bacterium has its own speed multiplier)
REPRODUCTION_ENERGY_THRESHOLD = 3   
ENERGY_FROM_FOOD_SCALE = 1.0        # energy gained per unit food consumed
MUTATION_STD = 0.03
HGT_RADIUS = 1.5                    # horizontal gene transfer radius
HGT_PROB = 0.001                    # probability of HGT per neighbor per step  

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
BACTERIA_PER_TYPE = 7

# -----------------------
# Individual Tracking Classes
# -----------------------
class IndividualTracker:
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.tracked_individuals = {}  # {bacterium_id: data_history}
        self.alive_individuals = set()
        self.deceased_individuals = set()
        
    def register_individual(self, bacterium):
        """Automatically register a new bacterium for tracking"""
        if bacterium.unique_id not in self.tracked_individuals:
            self.tracked_individuals[bacterium.unique_id] = {
                'steps': deque(maxlen=self.max_history),
                'enzyme': deque(maxlen=self.max_history),
                'efflux': deque(maxlen=self.max_history),
                'membrane': deque(maxlen=self.max_history),
                'repair': deque(maxlen=self.max_history),
                'energy': deque(maxlen=self.max_history),
                'pos_x': deque(maxlen=self.max_history),
                'pos_y': deque(maxlen=self.max_history),
                'bacterial_type': bacterium.bacterial_type,
                'birth_step': None,
                'death_step': None,
                'cause_of_death': None  # 'starvation', 'antibiotic', 'old_age'
            }
            self.alive_individuals.add(bacterium.unique_id)
        
    def update_tracked_individuals(self, model):
        """Update data for all tracked individuals"""
        current_step = model.step_count
        current_alive_ids = {b.unique_id for b in model.agent_set}
        
        # Register any new bacteria
        for bacterium in model.agent_set:
            if bacterium.unique_id not in self.tracked_individuals:
                self.register_individual(bacterium)
                self.tracked_individuals[bacterium.unique_id]['birth_step'] = current_step
        
        # Update data for all alive bacteria
        for bacterium in model.agent_set:
            data = self.tracked_individuals[bacterium.unique_id]
            data['steps'].append(current_step)
            data['enzyme'].append(bacterium.enzyme)
            data['efflux'].append(bacterium.efflux)
            data['membrane'].append(bacterium.membrane)
            data['repair'].append(bacterium.repair)
            data['energy'].append(bacterium.energy)
            data['pos_x'].append(bacterium.pos[0])
            data['pos_y'].append(bacterium.pos[1])
        
        # Detect newly deceased bacteria
        newly_deceased = self.alive_individuals - current_alive_ids
        for bacterium_id in newly_deceased:
            self.tracked_individuals[bacterium_id]['death_step'] = current_step
            self.deceased_individuals.add(bacterium_id)
        
        # Update alive set
        self.alive_individuals = self.alive_individuals.intersection(current_alive_ids)
    
    def mark_death(self, bacterium_id, cause):
        """Mark a bacterium as deceased with cause"""
        if bacterium_id in self.tracked_individuals:
            self.tracked_individuals[bacterium_id]['cause_of_death'] = cause
    
    def get_tracked_data(self, bacterium_id):
        """Get historical data for a specific bacterium"""
        return self.tracked_individuals.get(bacterium_id, None)
    
    def get_all_tracked_ids(self):
        """Get all tracked bacterium IDs (alive and dead)"""
        return list(self.tracked_individuals.keys())
    
    def get_alive_individuals(self):
        """Get list of currently alive individuals"""
        return list(self.alive_individuals)
    
    def get_deceased_individuals(self):
        """Get list of deceased individuals"""
        return list(self.deceased_individuals)
    
    def get_statistics(self):
        """Get overall tracking statistics"""
        return {
            'total_tracked': len(self.tracked_individuals),
            'alive': len(self.alive_individuals),
            'deceased': len(self.deceased_individuals)
        }

class IndividualPlotter:
    def __init__(self, tracker):
        self.tracker = tracker
        self.fig = None
        self.axes = None
        self.current_id = None
        
    def create_plot_window(self):
        """Create a new window for individual plots"""
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8))
        self.fig.suptitle("Individual Bacterium Tracking")
        
        # Flatten axes for easier indexing
        self.axes = self.axes.flatten()
        
        plt.tight_layout()
        plt.show(block=False)  # Show window without blocking
        return self.fig
        
    def update_plots(self, bacterium_id):
        """Update all plots for the given bacterium ID"""
        if self.fig is None:
            self.create_plot_window()
            
        data = self.tracker.get_tracked_data(bacterium_id)
        if data is None:
            print(f"Bacterium {bacterium_id} not found in tracking data")
            return
            
        if len(data['steps']) == 0:
            print(f"No data collected yet for bacterium {bacterium_id}")
            return
            
        self.current_id = bacterium_id
        
        # Clear all axes
        for ax in self.axes:
            ax.clear()
            
        steps = list(data['steps'])
        print(f"Plotting {len(steps)} data points for bacterium {bacterium_id}")
        
        # Determine if bacterium is alive or dead
        is_alive = bacterium_id in self.tracker.alive_individuals
        status = "ALIVE" if is_alive else f"DECEASED (step {data['death_step']})"
        if not is_alive and data['cause_of_death']:
            status += f" - {data['cause_of_death']}"
        
        # Update main title
        self.fig.suptitle(f"Individual Bacterium Tracking - ID: {bacterium_id} ({data['bacterial_type']}) - {status}")
        
        # Plot 1: Current resistance traits (bar chart)
        traits = ['enzyme', 'efflux', 'membrane', 'repair']
        values = [data[trait][-1] for trait in traits]
        self.axes[0].bar(traits, values, color=['red', 'blue', 'green', 'orange'])
        self.axes[0].set_title(f"Final Resistance Traits")
        self.axes[0].set_ylabel("Trait Value")
        self.axes[0].set_ylim(0, 1)
        self.axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Energy over time
        self.axes[1].plot(steps, list(data['energy']), 'purple', linewidth=2)
        self.axes[1].set_title("Energy Over Time")
        self.axes[1].set_xlabel("Simulation Step")
        self.axes[1].set_ylabel("Energy")
        self.axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Starvation threshold')
        if not is_alive and data['death_step']:
            self.axes[1].axvline(x=data['death_step'], color='red', linestyle='--', alpha=0.5, label='Death')
        self.axes[1].legend()
        
        # Plot 3: All resistance traits over time
        colors = ['red', 'blue', 'green', 'orange']
        traits = ['enzyme', 'efflux', 'membrane', 'repair']
        for trait, color in zip(traits, colors):
            self.axes[2].plot(steps, list(data[trait]), color=color, label=trait, linewidth=2)
        self.axes[2].set_title("Resistance Traits Evolution")
        self.axes[2].set_xlabel("Simulation Step")
        self.axes[2].set_ylabel("Trait Value")
        self.axes[2].legend()
        self.axes[2].set_ylim(0, 1)
        if not is_alive and data['death_step']:
            self.axes[2].axvline(x=data['death_step'], color='red', linestyle='--', alpha=0.5)
        
        # Plot 4: Position X over time
        self.axes[3].plot(steps, list(data['pos_x']), 'cyan', linewidth=2)
        self.axes[3].set_title("X Position Over Time")
        self.axes[3].set_xlabel("Simulation Step")
        self.axes[3].set_ylabel("X Position")
        if not is_alive and data['death_step']:
            self.axes[3].axvline(x=data['death_step'], color='red', linestyle='--', alpha=0.5)
        
        # Plot 5: Position Y over time
        self.axes[4].plot(steps, list(data['pos_y']), 'magenta', linewidth=2)
        self.axes[4].set_title("Y Position Over Time")
        self.axes[4].set_xlabel("Simulation Step")
        self.axes[4].set_ylabel("Y Position")
        if not is_alive and data['death_step']:
            self.axes[4].axvline(x=data['death_step'], color='red', linestyle='--', alpha=0.5)
        
        # Plot 6: Combined position trajectory
        self.axes[5].plot(list(data['pos_x']), list(data['pos_y']), 'darkblue', linewidth=1, alpha=0.7)
        # Mark start and end points
        self.axes[5].scatter([data['pos_x'][0]], [data['pos_y'][0]], color='green', s=100, marker='o', label='Birth', zorder=5)
        self.axes[5].scatter([data['pos_x'][-1]], [data['pos_y'][-1]], 
                            color='red' if not is_alive else 'blue', 
                            s=100, marker='X' if not is_alive else 'o', 
                            label='Death' if not is_alive else 'Current', zorder=5)
        self.axes[5].set_title("Movement Trajectory")
        self.axes[5].set_xlabel("X Position")
        self.axes[5].set_ylabel("Y Position")
        self.axes[5].legend()
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()  # Force update
        
        # Bring window to front
        try:
            self.fig.canvas.manager.window.wm_attributes('-topmost', 1)
            self.fig.canvas.manager.window.wm_attributes('-topmost', 0)
        except:
            pass

# -----------------------
# Agent definition
# -----------------------
class Bacterium(Agent):
    def __init__(self, model, bacterial_type="E.coli"):
        # Correct Agent initialization signature: Agent(unique_id, model)
        super().__init__(model)
        self.unique_id = model.next_id()
        self.pos = None
        self.energy = random.uniform(1.0, 2.0)
        self.bacterial_type = bacterial_type
        self.age = 0
        
        # Generate resistance traits based on bacterial type
        type_def = BACTERIAL_TYPES[bacterial_type]
        self.enzyme = max(0.0, min(1.0, random.gauss(*type_def["enzyme"])))
        self.efflux = max(0.0, min(1.0, random.gauss(*type_def["efflux"])))
        self.membrane = max(0.0, min(1.0, random.gauss(*type_def["membrane"])))
        self.repair = max(0.0, min(1.0, random.gauss(*type_def["repair"])))
        self.max_age = max(10, int(random.gauss(*type_def["max_age"])))
        self.speed = max(0.1, random.gauss(*type_def["base_speed"]))

    def calculate_survival_probability(self, antibiotic_conc, antibiotic_type):
        """Calculate survival probability using the new resistance formula"""
        if antibiotic_conc <= 0:
            return 1.0
            
        ab_def = ANTIBIOTIC_TYPES[antibiotic_type]
        
        # Calculate effective antibiotic concentration
        A_eff = antibiotic_conc * \
                (1 - ab_def["efflux_weight"] * self.efflux) * \
                (1 - ab_def["enzyme_weight"] * self.enzyme) * \
                (1 - ab_def["membrane_weight"] * self.membrane)
        
        A_eff = max(0.0, A_eff)  # Can't be negative
        
        # Calculate survival probability
        damage_factor = A_eff * (1 - ab_def["repair_weight"] * self.repair)
        survival_prob = math.exp(-ab_def["toxicity_constant"] * damage_factor)
        
        return min(1.0, max(0.0, survival_prob))

    def mutate_offspring_traits(self):
        """Create mutated traits for offspring"""
        traits = {
            "enzyme": max(0.0, min(1.0, self.enzyme + random.gauss(0, MUTATION_STD))),
            "efflux": max(0.0, min(1.0, self.efflux + random.gauss(0, MUTATION_STD))),
            "membrane": max(0.0, min(1.0, self.membrane + random.gauss(0, MUTATION_STD))),
            "repair": max(0.0, min(1.0, self.repair + random.gauss(0, MUTATION_STD)))
        }
        return traits

    def step(self):
        # Age the bacterium
        self.age += 1
        
        # Check for death by old age
        if self.age >= self.max_age:
            self.model.individual_tracker.mark_death(self.unique_id, 'old_age')
            self.model.to_remove.add(self)
            return

        # Movement: biased random walk toward nutrient gradient
        nx, ny = self.model.nutrient_to_field_coords(self.pos)
        grad = self.model.compute_gradient_at_field(nx, ny)
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
            # print("exception:", sys.exc_info())
            raise Exception("Agent movement failed")

        # Consume food at location (sampled from field)
        fx, fy = self.model.nutrient_to_field_coords(self.pos)
        food_amount = self.model.sample_field(self.model.food_field, fx, fy)
        consumed = min(food_amount, FOOD_CONSUMPTION_PER_STEP)
        self.model.subtract_from_field(self.model.food_field, fx, fy, consumed)
        self.energy += consumed * ENERGY_FROM_FOOD_SCALE

        # Energy decay - bacteria need constant food to survive
        self.energy -= 0.05  # Base metabolic cost
        
        # Death by starvation
        if self.energy <= 0:
            self.model.individual_tracker.mark_death(self.unique_id, 'starvation')
            self.model.to_remove.add(self)
            return

        # Antibiotic effect using new survival formula
        a_conc = self.model.sample_field(self.model.antibiotic_field, fx, fy)
        # print(f"Antibiotic concentration at bacterium {self.unique_id}: {a_conc:.3f}")
        if a_conc > 0:
            survival_prob = self.calculate_survival_probability(a_conc, self.model.current_antibiotic)
            if(survival_prob < 0.0):
                print("Warning: negative survival probability computed.")
            elif(survival_prob > 1.0):
                print("Warning: survival probability greater than 1 computed.")
            if random.random() > survival_prob:
                self.model.individual_tracker.mark_death(self.unique_id, f'antibiotic_{self.model.current_antibiotic}')
                self.model.to_remove.add(self)
                return

        # Reproduction
        if self.energy >= REPRODUCTION_ENERGY_THRESHOLD:
            self.energy /= 2.0
            mutated_traits = self.mutate_offspring_traits()
            
            # Give child a slightly offset position to avoid placing at same location as parent
            offset_x = random.uniform(-0.5, 0.5)
            offset_y = random.uniform(-0.5, 0.5)
            child_x = max(0, min(self.model.width, self.pos[0] + offset_x))
            child_y = max(0, min(self.model.height, self.pos[1] + offset_y))
            child_pos = (child_x, child_y)
            
            child = Bacterium(self.model, bacterial_type=self.bacterial_type)
            # Apply mutations
            child.enzyme = mutated_traits["enzyme"]
            child.efflux = mutated_traits["efflux"]
            child.membrane = mutated_traits["membrane"]
            child.repair = mutated_traits["repair"]
            # Defer adding to model until after stepping through all agents
            self.model.new_agents.append((child, child_pos))

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
        self.current_antibiotic = None  # Default antibiotic type
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
                bacterium = Bacterium(self, bacterial_type=bacterial_type)
                self.agent_set.add(bacterium)
                # Place agent in the space - only call this once
                self.space.place_agent(bacterium, (x, y))

        self.running = True
        self.step_count = 0

        # Tracking system
        self.individual_tracker = IndividualTracker()

        # HGT toggle
        self.enable_hgt = bool(enable_hgt)

        # Tracking system
        self.tracked_bacteria = set()
        self.tracking_data = defaultdict(lambda: defaultdict(deque))

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
        
        return stats

    def track_bacterium(self, bacterium):
        """Start tracking a bacterium"""
        self.tracked_bacteria.add(bacterium)

    def untrack_bacterium(self, bacterium):
        """Stop tracking a bacterium"""
        if bacterium in self.tracked_bacteria:
            self.tracked_bacteria.remove(bacterium)

    def collect_tracking_data(self):
        """Collect data for tracked bacteria"""
        for bacterium in self.tracked_bacteria:
            self.tracking_data[bacterium]["age"].append(bacterium.age)
            self.tracking_data[bacterium]["energy"].append(bacterium.energy)
            self.tracking_data[bacterium]["enzyme"].append(bacterium.enzyme)
            self.tracking_data[bacterium]["efflux"].append(bacterium.efflux)
            self.tracking_data[bacterium]["membrane"].append(bacterium.membrane)
            self.tracking_data[bacterium]["repair"].append(bacterium.repair)

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
        for child,child_pos in self.new_agents:
            try:
                self.space.place_agent(child, child_pos)
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

        # Update individual tracking data
        self.individual_tracker.update_tracked_individuals(self)

        # Collect tracking data
        self.collect_tracking_data()


# -----------------------
# Visualization + Control UI
# -----------------------
class SimulatorUI:
    def __init__(self, model):
        self.model = model
        self.paused = True
        self.latest_dose = 0.0
        
        # Create dynamic color mapping for bacterial types
        self.bacterial_type_names = list(BACTERIAL_TYPES.keys())
        self.color_map = {name: i for i, name in enumerate(self.bacterial_type_names)}
        
        # Speed control variables
        self.steps_per_second = DEFAULT_STEPS_PER_SECOND
        self.animation_fps = ANIMATION_FPS
        self.animation_interval = int(1000 / self.animation_fps)  # Convert FPS to milliseconds
        self.steps_accumulator = 0.0  # For fractional step timing
        self.animation = None
        
        # Tracking list state - to avoid unnecessary updates
        self.last_bacteria_list_hash = None
        
        # Setup matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.scat = None
        self.im_food = None
        self.im_ab = None

        # Individual tracking
        self.individual_plotter = IndividualPlotter(self.model.individual_tracker)
        
        # Click handling for selecting individuals
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

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

    def on_click(self, event):
        """Handle mouse clicks to select bacteria for tracking"""
        if event.inaxes != self.ax:
            return
            
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
            
        # Find closest bacterium
        min_dist = float('inf')
        closest_bacterium = None
        
        for bacterium in self.model.agent_set:
            dist = np.sqrt((bacterium.pos[0] - x)**2 + (bacterium.pos[1] - y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_bacterium = bacterium
                
        if closest_bacterium and min_dist < 5.0:  # Within 5 units
            # The bacterium is already being tracked automatically, just view its plots
            print(f"Viewing bacterium {closest_bacterium.unique_id} ({closest_bacterium.bacterial_type})")
            
            # Open/update plot window
            self.individual_plotter.update_plots(closest_bacterium.unique_id)

    def get_bacterial_colors(self, agents):
        """Get numerical colors for bacterial types"""
        return [self.color_map.get(a.bacterial_type, 0) for a in agents]

    def build_controls(self):
        frm = ttk.Frame(self.root, padding=8)
        frm.grid()
        
        row = 0
        
        # Title
        ttk.Label(frm, text="Simulation Controls", font=("TkDefaultFont", 10, "bold")).grid(
            column=0, row=row, columnspan=2, pady=(0, 10))
        row += 1

        # Basic controls
        self.pause_btn = ttk.Button(frm, text="Resume", command=self.toggle_pause)
        self.pause_btn.grid(column=0, row=row, pady=2)
        ttk.Button(frm, text="Reset", command=self.reset_sim).grid(column=1, row=row, pady=2)
        row += 1

        # Speed controls
        ttk.Label(frm, text="Speed Control", font=("TkDefaultFont", 9, "bold")).grid(
            column=0, row=row, columnspan=2, pady=(15, 5))
        row += 1
        
        speed_frame = ttk.Frame(frm)
        speed_frame.grid(column=0, row=row, columnspan=2, pady=5)
        row += 1
        
        ttk.Button(speed_frame, text="<<", command=self.speed_slower, width=3).grid(column=0, row=0)
        ttk.Button(speed_frame, text=">>", command=self.speed_faster, width=3).grid(column=1, row=0)
        ttk.Button(speed_frame, text="Reset Speed", command=self.speed_reset).grid(column=2, row=0, padx=(5,0))
        
        self.speed_label = ttk.Label(frm, text=f"Speed: {self.steps_per_second} steps/sec")
        self.speed_label.grid(column=0, row=row, columnspan=2, pady=(0, 5))
        row += 1

        # Separator
        ttk.Separator(frm, orient='horizontal').grid(column=0, row=row, columnspan=2, sticky='ew', pady=10)
        row += 1

        # Antibiotic controls
        ttk.Label(frm, text="Antibiotic Control", font=("TkDefaultFont", 9, "bold")).grid(
            column=0, row=row, columnspan=2, pady=(0, 5))
        row += 1
        
        ttk.Label(frm, text="Type:").grid(column=0, row=row, sticky='w', padx=(0, 5))
        self.antibiotic_var = tk.StringVar(value=self.model.current_antibiotic)
        self.antibiotic_combo = ttk.Combobox(frm, textvariable=self.antibiotic_var, 
                                           values=self.model.available_antibiotics, 
                                           state="readonly", width=12)
        self.antibiotic_combo.grid(column=1, row=row, pady=2)
        self.antibiotic_combo.bind('<<ComboboxSelected>>', self.change_antibiotic)
        row += 1

        ttk.Label(frm, text="Dose:").grid(column=0, row=row, sticky='w')
        self.dose_var = tk.DoubleVar(value=0.5)
        self.dose_entry = ttk.Entry(frm, textvariable=self.dose_var, width=8)
        self.dose_entry.grid(column=1, row=row, pady=2)
        row += 1

        ttk.Button(frm, text="Apply Antibiotic", command=self.apply_antibiotic_ui).grid(
            column=0, row=row, columnspan=2, pady=5)
        row += 1

        ttk.Label(frm, text="Latest dose:").grid(column=0, row=row, sticky='w')
        self.latest_label = ttk.Label(frm, text="0.0")
        self.latest_label.grid(column=1, row=row, sticky='w')
        row += 1

        # HGT toggle
        self.hgt_var = tk.BooleanVar(value=self.model.enable_hgt)
        self.hgt_check = ttk.Checkbutton(
            frm, text="Enable Horizontal Gene Transfer", variable=self.hgt_var, command=self.toggle_hgt
        )
        self.hgt_check.grid(column=0, row=row, columnspan=2, pady=(10, 5))
        row += 1

        # Separator
        ttk.Separator(frm, orient='horizontal').grid(column=0, row=row, columnspan=2, sticky='ew', pady=10)
        row += 1

        # Individual tracking controls
        ttk.Label(frm, text="Browse Bacteria", font=("TkDefaultFont", 9, "bold")).grid(
            column=0, row=row, columnspan=2, pady=(0, 5))
        row += 1
        
        # Filter options
        ttk.Label(frm, text="Show:").grid(column=0, row=row, sticky='w')
        self.filter_var = tk.StringVar(value="alive")
        
        self.filter_combo = ttk.Combobox(frm, textvariable=self.filter_var, 
                                    values=["alive", "deceased", "all"], 
                                    state="readonly", width=12)
        self.filter_combo.grid(column=1, row=row, pady=2)
        
        # Pass the combobox value directly to avoid StringVar update issues
        def on_filter_change(event):
            selected_value = self.filter_combo.get()
            self.update_bacteria_list(filter_type=selected_value, force_update=True)
        
        self.filter_combo.bind('<<ComboboxSelected>>', on_filter_change)
        self.filter_combo.current(0)  # Set to first item (alive) by default
        row += 1
        
        # Bacteria list with scrollbar
        list_frame = ttk.Frame(frm)
        list_frame.grid(column=0, row=row, columnspan=2, pady=5, sticky="ew")
        row += 1
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.bacteria_listbox = tk.Listbox(list_frame, height=8, width=30, 
                                          yscrollcommand=scrollbar.set,
                                          selectmode=tk.SINGLE,  # Allow single selection
                                          exportselection=False)  # Keep selection when clicking outside
        self.bacteria_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.bacteria_listbox.yview)
        
        # Double-click to view (in addition to button)
        self.bacteria_listbox.bind('<Double-Button-1>', lambda e: self.view_selected_bacterium())
        
        # Tracking stats
        self.tracking_stats_label = ttk.Label(frm, text="Tracked: 0 alive, 0 deceased", font=("TkDefaultFont", 8))
        self.tracking_stats_label.grid(column=0, row=row, columnspan=2, pady=(2, 5))
        row += 1
        
        ttk.Button(frm, text="View Selected Bacterium", command=self.view_selected_bacterium).grid(
            column=0, row=row, columnspan=2, pady=(0, 5))
        row += 1

        # Separator
        ttk.Separator(frm, orient='horizontal').grid(column=0, row=row, columnspan=2, sticky='ew', pady=10)
        row += 1

        # Population stats display
        ttk.Label(frm, text="Population Stats", font=("TkDefaultFont", 9, "bold")).grid(
            column=0, row=row, columnspan=2, pady=(0, 5))
        row += 1

        self.stats_frame = ttk.Frame(frm)
        self.stats_frame.grid(column=0, row=row, columnspan=2, sticky="ew")
        row += 1
        
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

    def track_bacterium_ui(self):
        """Track a bacterium by ID from the UI"""
        try:
            bacterium_id = self.track_id_var.get()
            bacterium = next((b for b in self.model.agent_set if b.unique_id == bacterium_id), None)
            if bacterium:
                self.model.track_bacterium(bacterium)
                print(f"Tracking bacterium {bacterium_id}")
            else:
                print(f"Bacterium {bacterium_id} not found")
        except Exception as e:
            print(f"Error tracking bacterium: {e}")

    def untrack_bacterium_ui(self):
        """Untrack a bacterium by ID from the UI"""
        try:
            bacterium_id = self.track_id_var.get()
            bacterium = next((b for b in self.model.agent_set if b.unique_id == bacterium_id), None)
            if bacterium:
                self.model.untrack_bacterium(bacterium)
                print(f"Untracking bacterium {bacterium_id}")
            else:
                print(f"Bacterium {bacterium_id} not found")
        except Exception as e:
            print(f"Error untracking bacterium: {e}")

    def show_individual_plot(self, event=None):
        """Show plot for selected individual from dropdown"""
        try:
            selected = self.tracked_var.get()
            if selected and selected != "":
                bacterium_id = int(selected.split(":")[1].split(" ")[0])
                self.individual_plotter.update_plots(bacterium_id)
        except Exception as e:
            print(f"Error showing individual plot: {e}")

    def refresh_individual_plots(self):
        """Refresh the individual plots for currently selected bacterium"""
        if self.individual_plotter.current_id is not None:
            self.individual_plotter.update_plots(self.individual_plotter.current_id)

    def update_tracked_dropdown(self):
        """Update the dropdown with currently tracked individuals"""
        if self.root is None:
            return
            
        try:
            active_ids = self.model.individual_tracker.get_active_individuals()
            if len(active_ids) == 0:
                self.tracked_combo['values'] = ["No tracked individuals"]
                return
                
            # Create descriptive names for tracked individuals
            options = []
            for bacterium_id in active_ids:
                # Find the bacterium to get its type
                bacterium = None
                for b in self.model.agent_set:
                    if b.unique_id == bacterium_id:
                        bacterium = b
                        break
                        
                if bacterium:
                    options.append(f"ID: {bacterium_id} ({bacterium.bacterial_type})")
                else:
                    options.append(f"ID: {bacterium_id} (unknown)")
                    
            self.tracked_combo['values'] = options
        except Exception as e:
            print(f"Error updating tracked dropdown: {e}")

    def update_bacteria_list(self, event=None, filter_type=None, force_update=False):
        """Update the listbox with bacteria based on filter - only when list changes"""
        if self.root is None:
            return
            
        try:
            # Use the passed filter_type parameter, or get it from the combobox directly
            if filter_type is None:
                filter_type = self.filter_combo.get() if hasattr(self, 'filter_combo') else "alive"
            
            # print(f"Updating bacteria list with filter: {filter_type}") 
            tracker = self.model.individual_tracker
            
            # Create a hash of the current state to detect changes
            if filter_type == "alive":
                ids = tracker.get_alive_individuals()
            elif filter_type == "deceased":
                ids = tracker.get_deceased_individuals()
            else:  # all
                ids = tracker.get_all_tracked_ids()
            
            # Create hash from the list state
            current_hash = (filter_type, tuple(sorted(ids)))
            
            # Only update if the list has changed or force_update is True
            if not force_update and current_hash == self.last_bacteria_list_hash:
                return
                
            self.last_bacteria_list_hash = current_hash
            
            # Save current selection if any
            old_selection = self.bacteria_listbox.curselection()
            old_selected_id = None
            if old_selection:
                try:
                    text = self.bacteria_listbox.get(old_selection[0])
                    id_part = text.split("ID:")[1].strip()
                    old_selected_id = int(id_part.split()[0])
                except:
                    pass
            
            self.bacteria_listbox.delete(0, tk.END)
            
            # Sort by ID
            ids.sort()
            
            # Add to listbox with descriptive text
            new_selection_index = None
            for i, bacterium_id in enumerate(ids):
                data = tracker.get_tracked_data(bacterium_id)
                if data:
                    status = "✓" if bacterium_id in tracker.alive_individuals else "✗"
                    btype = data['bacterial_type'][:8]  # Truncate long names
                    lifespan = len(data['steps'])
                    text = f"{status} ID:{bacterium_id:3d} {btype:8s} ({lifespan} steps)"
                    self.bacteria_listbox.insert(tk.END, text)
                    
                    # Track index if this was the previously selected bacterium
                    if bacterium_id == old_selected_id:
                        new_selection_index = i
            
            # Restore selection if the bacterium is still in the list
            if new_selection_index is not None:
                self.bacteria_listbox.selection_set(new_selection_index)
                self.bacteria_listbox.see(new_selection_index)
            
            # Update stats label
            stats = tracker.get_statistics()
            self.tracking_stats_label.config(
                text=f"Tracked: {stats['alive']} alive, {stats['deceased']} deceased (total: {stats['total_tracked']})"
            )
                    
        except Exception as e:
            print(f"Error updating bacteria list: {e}")

    def on_bacteria_select(self, event):
        """Handle bacteria selection from listbox"""
        # Selection is handled by view button
        pass

    def view_selected_bacterium(self):
        """View the selected bacterium's plots"""
        try:
            selection = self.bacteria_listbox.curselection()
            print(f"DEBUG: curselection() returned: {selection}")
            
            if not selection or len(selection) == 0:
                print("No bacterium selected")
                return
                
            # Extract ID from the listbox text
            text = self.bacteria_listbox.get(selection[0])
            print(f"DEBUG: Selected text: '{text}'")
            
            # Format is "✓ ID:  1 E.coli   (50 steps)" or "✗ ID:  1 E.coli   (50 steps)"
            # Split by "ID:" and get the number part
            id_part = text.split("ID:")[1].strip()  # Get part after "ID:"
            print(f"DEBUG: ID part: '{id_part}'")
            
            bacterium_id = int(id_part.split()[0])  # Get first number
            print(f"DEBUG: Extracted bacterium ID: {bacterium_id}")
            
            self.individual_plotter.update_plots(bacterium_id)
            
        except Exception as e:
            print(f"Error viewing bacterium: {e}")
            import traceback
            traceback.print_exc()

    def pump_tk(self):
        """Pump Tk events to keep UI responsive"""
        if self.root is not None:
            try:
                self.root.update_idletasks()
                self.root.update()
            except Exception as e:
                print(f"Error pumping Tk events: {e}")

    def run(self):
        """Run the simulation with visualization"""
        self.paused = True
        self.animation = animation.FuncAnimation(
            self.fig, self.update, interval=self.animation_interval, blit=False, cache_frame_data=False
        )
        plt.show()

    def update(self, frame):
        """Update the visualization and model"""
        if not self.paused:
            self.steps_accumulator += self.steps_per_second / self.animation_fps
            while self.steps_accumulator >= 1.0:
                self.model.step()
                self.steps_accumulator -= 1.0

        self.update_plot()
        self.update_stats_display()
        self.update_bacteria_list()
        self.pump_tk()

    def update_plot(self):
        """Update the plot with current model state"""
        agents = list(self.model.agent_set)
        positions = [a.pos for a in agents]
        colors = self.get_bacterial_colors(agents)
        
        if self.scat is None:
            self.scat = self.ax.scatter(
                [pos[0] for pos in positions],
                [pos[1] for pos in positions],
                c=colors,
                cmap="viridis",
                s=10,
                edgecolor="k",
                alpha=0.7,
            )
        else:
            self.scat.set_offsets(positions)
            self.scat.set_array(np.array(colors))
        
        # Update food field visualization
        if self.im_food is None:
            self.im_food = self.ax.imshow(
                self.model.food_field.T,
                extent=[0, self.model.width, 0, self.model.height],
                origin="lower",
                cmap="Greens",
                alpha=0.3,
            )
        else:
            self.im_food.set_data(self.model.food_field.T)
        
        # Update antibiotic field visualization
        if self.im_ab is None:
            self.im_ab = self.ax.imshow(
                self.model.antibiotic_field.T,
                extent=[0, self.model.width, 0, self.model.height],
                origin="lower",
                cmap="Reds",
                alpha=0.3,
            )
        else:
            self.im_ab.set_data(self.model.antibiotic_field.T)
        
        self.ax.set_title(f"Step: {self.model.step_count}")
        self.ax.set_xlim(0, self.model.width)
        self.ax.set_ylim(0, self.model.height)
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    model = BacteriaModel()
    ui = SimulatorUI(model)
    ui.run()
