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
DEFAULT_ANIMATION_INTERVAL = 200  # milliseconds between frames (lower = faster)
MIN_ANIMATION_INTERVAL = 50      # minimum interval (maximum speed)
MAX_ANIMATION_INTERVAL = 1000    # maximum interval (minimum speed)
STEPS_PER_FRAME = 1              # number of simulation steps per animation frame

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
            
            child = Bacterium(self.model, pos=child_pos, bacterial_type=self.bacterial_type)
            # Apply mutations
            child.enzyme = mutated_traits["enzyme"]
            child.efflux = mutated_traits["efflux"]
            child.membrane = mutated_traits["membrane"]
            child.repair = mutated_traits["repair"]
            # Defer adding to model until after stepping through all agents
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
        trait_sums = {"enzyme": 0, "efflux": 0, "membrane": 0, "repair": 0}
        age_sum = 0
        
        for bacterium in self.agent_set:
            btype = bacterium.bacterial_type
            if btype not in stats["by_type"]:
                stats["by_type"][btype] = 0
            stats["by_type"][btype] += 1
            
            trait_sums["enzyme"] += bacterium.enzyme
            trait_sums["efflux"] += bacterium.efflux
            trait_sums["membrane"] += bacterium.membrane
            trait_sums["repair"] += bacterium.repair
            age_sum += bacterium.age
        
        # Calculate averages
        total = len(self.agent_set)
        for trait in trait_sums:
            stats["avg_traits"][trait] = trait_sums[trait] / total
        stats["avg_age"] = age_sum / total
        
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
        self.animation_interval = DEFAULT_ANIMATION_INTERVAL
        self.steps_per_frame = STEPS_PER_FRAME
        self.animation = None  # Store reference to animation
        self.step_counter = 0  # Counter to skip frames for slower speeds

        # Setup matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.scat = None
        self.im_food = None
        self.im_ab = None

        # Start Tk UI if available, but DO NOT start mainloop in a separate thread.
        # We'll pump Tk events from the animation loop to avoid cross-thread Tcl calls.
        if tk is not None:
            try:
                self.root = tk.Tk()
                self.root.title("Control Panel")
                self.build_controls()
                # do NOT call self.root.mainloop() in another thread
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
        
        self.speed_label = ttk.Label(frm, text=f"Interval: {self.animation_interval}ms")
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
        """Update the population statistics display"""
        if self.root is None:
            return
            
        try:
            stats = self.model.get_population_stats()
            
            # Clear old labels
            for label in self.stats_labels.values():
                label.destroy()
            self.stats_labels.clear()
            
            row = 0
            # Total population
            self.stats_labels["total"] = ttk.Label(self.stats_frame, 
                                                  text=f"Total: {stats['total']}")
            self.stats_labels["total"].grid(column=0, row=row, columnspan=2, sticky="w")
            row += 1
            
            # Population by type
            for btype, count in stats["by_type"].items():
                color = BACTERIAL_TYPES[btype]["color"]
                self.stats_labels[f"type_{btype}"] = ttk.Label(self.stats_frame, 
                                                              text=f"{btype}: {count}")
                self.stats_labels[f"type_{btype}"].grid(column=0, row=row, columnspan=2, sticky="w")
                row += 1
            
            # Average traits (if population exists)
            if stats["total"] > 0:
                ttk.Label(self.stats_frame, text="Avg Traits:", 
                         font=("TkDefaultFont", 8, "bold")).grid(column=0, row=row, columnspan=2, sticky="w")
                row += 1
                
                for trait, value in stats["avg_traits"].items():
                    self.stats_labels[f"trait_{trait}"] = ttk.Label(self.stats_frame, 
                                                                   text=f"  {trait}: {value:.3f}")
                    self.stats_labels[f"trait_{trait}"].grid(column=0, row=row, columnspan=2, sticky="w")
                    row += 1
                
                self.stats_labels["age"] = ttk.Label(self.stats_frame, 
                                                    text=f"  avg age: {stats['avg_age']:.1f}")
                self.stats_labels["age"].grid(column=0, row=row, columnspan=2, sticky="w")
                
        except Exception as e:
            print(f"Error updating stats: {e}")

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
        """Decrease interval to make animation faster"""
        self.animation_interval = max(MIN_ANIMATION_INTERVAL, self.animation_interval - 50)
        self.update_speed_display()
        self.update_animation_speed()

    def speed_slower(self):
        """Increase interval to make animation slower"""
        self.animation_interval = min(MAX_ANIMATION_INTERVAL, self.animation_interval + 50)
        self.update_speed_display()
        self.update_animation_speed()

    def speed_reset(self):
        """Reset speed to default"""
        self.animation_interval = DEFAULT_ANIMATION_INTERVAL
        self.update_speed_display()
        self.update_animation_speed()

    def update_speed_display(self):
        """Update the speed label"""
        if self.root is not None:
            try:
                self.speed_label.config(text=f"Interval: {self.animation_interval}ms")
            except Exception:
                pass

    def update_animation_speed(self):
        """Update the animation interval if animation is running"""
        if self.animation is not None:
            try:
                self.animation.event_source.interval = self.animation_interval
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
        # pump tk events so controls are responsive without separate Tk thread
        self.pump_tk()

        if not self.paused:
            if self.animation_interval <= DEFAULT_ANIMATION_INTERVAL:
                # Faster than default: run multiple steps per frame
                steps_to_run = max(1, int(DEFAULT_ANIMATION_INTERVAL / self.animation_interval))
                for _ in range(steps_to_run):
                    try:
                        self.model.step()
                    except Exception:
                        pass
            else:
                # Slower than default: skip frames to run fewer steps
                frames_per_step = int(self.animation_interval / DEFAULT_ANIMATION_INTERVAL)
                self.step_counter += 1
                if self.step_counter >= frames_per_step:
                    self.step_counter = 0
                    try:
                        self.model.step()
                    except Exception:
                        pass

        # update images and scatter
        try:
            food_img = np.rot90(self.model.food_field)
            self.im_food.set_data(food_img)
            ab_img = np.rot90(self.model.antibiotic_field)
            self.im_ab.set_data(ab_img)
        except Exception:
            pass

        xs = [a.pos[0] for a in self.model.agent_set]
        ys = [a.pos[1] for a in self.model.agent_set]
        
        if len(xs) == 0:
            self.scat.set_offsets(np.empty((0, 2)))
            self.scat.set_array(np.array([]))
        else:
            # Use numerical color mapping
            colors = self.get_bacterial_colors(self.model.agent_set)
            self.scat.set_offsets(np.c_[xs, ys])
            self.scat.set_array(np.array(colors))
        
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
