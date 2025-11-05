"""
Configuration parameters for the bacteria simulation.
Contains all constants, growth parameters, and bacterial/antibiotic definitions.
"""

# -----------------------
# Spatial Configuration
# -----------------------
from typing import Tuple
import numpy as np
import torch

WIDTH = 100.0                       # continuous width
HEIGHT = 100.0
GRID_RES = 200                      # resolution for nutrient & antibiotic fields (square grid)

# -----------------------
# Simulation Parameters
# -----------------------
# Number of bacteria per type at initialization
BACTERIA_PER_TYPE = 50
INITIAL_BACTERIA = None
FOOD_DIFFUSION_SIGMA = 1.0 #UNUSED          # for gaussian_filter diffusion approximation 
BACTERIA_SPEED = 1                # scaling factor for bacterium movement speed
MUTATION_STD = 0.03

# NOOP Action Reward Shaping
NOOP_BAND_FACTOR = 0.02           # deadband around target as fraction of population_norm (~2%)
NOOP_REWARD_MAGNITUDE = 0.0      # small shaping magnitude for NOOP action

# Food Patch Configuration
FOOD_PATCH_COUNT = 120                # number of food patches to spawn
FOOD_PATCH_AMPLITUDE_MIN = 0.2      # minimum food concentration per patch
FOOD_PATCH_AMPLITUDE_MAX = 0.8      # maximum food concentration per patch
FOOD_PATCH_SIGMA_MIN = 3            # minimum patch size (smaller = more concentrated)
FOOD_PATCH_SIGMA_MAX = 15            # maximum patch size (larger = more spread out)

# Gene Transfer Parameters
HGT_RADIUS = 1.5                    # gene transfer radius between bacteria
HGT_PROB = 0.001                    # probability of gene transfer between neighbors with HGT gene

# Simulation speed settings
DEFAULT_STEPS_PER_FRAME = 1         # simulation steps to run per visual frame (more direct control)
MIN_STEPS_PER_FRAME = 0             # 0 means skip some frames (slower than 1 step/frame)
MAX_STEPS_PER_FRAME = 10            # maximum steps per frame (can go higher if needed)
ANIMATION_FPS = 30                  # visual update rate (frames per second)

# When MIN_STEPS_PER_FRAME = 0, this controls how often to step
# e.g., SLOW_MODE_FRAME_SKIP = 3 means step every 3rd frame
SLOW_MODE_FRAME_SKIP = 3            # step every Nth frame when in "skip" mode

# Performance mode settings
PERFORMANCE_MODE = False            # When enabled, reduces UI update frequency for better performance
STATS_UPDATE_INTERVAL = 5           # Update stats every N frames when performance mode is on
VISUALIZATION_UPDATE_INTERVAL = 1   # Update visualization every N frames (1 = every frame)

# Antibiotic degradation (legacy - now per antibiotic type, see ANTIBIOTIC_TYPES)
ANTIBIOTIC_DECAY = 0.05             # per sim step (default if not specified)

# -----------------------
# Growth Model Parameters
# -----------------------
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

# -----------------------
# Budget Allocation Parameters
# -----------------------
ALLOCATION_PARAMS = {
    "total_budget": 1.0,           # Total resource budget per bacterium
    "reallocation_std": 0.02,      # Standard deviation for small reallocations
    "amplification_prob": 0.05,    # Probability of trait amplification
    "amplification_cost": 0.2,     # Energy cost per unit of amplification
    "diminishing_return_alpha": 0.7 # Exponent for diminishing returns
}

# -----------------------
# Persistence Parameters
# -----------------------
PERSISTENCE_PARAMS = {
    # Entry probabilities (per timestep)
    "base_entry_prob": 0.001,           # Baseline probability of entering persistor state
    "stress_entry_multiplier": 10.0,    # Multiplier when under stress (low energy or antibiotics)
    "energy_stress_threshold": 1.0,     # Energy below this is considered stress
    "antibiotic_stress_threshold": 0.1, # Antibiotic concentration above this is stress
    
    # Exit probabilities (per timestep)
    "base_exit_prob": 0.005,            # Baseline probability of exiting persistor state
    "favorable_exit_multiplier": 5.0,   # Multiplier when conditions are favorable
    "energy_favorable_threshold": 0.5,  # Minimum energy to consider conditions favorable
    
    # Persistor state modifiers
    "energy_decay_rate": 0.002,         # Energy loss per timestep while dormant
    "antibiotic_resistance_factor": 0.1,# Multiply kill probability by this (90% reduction)
    "movement_speed_factor": 0.2,       # Movement speed multiplier (80% reduction)
    "min_persistor_energy": 0.1,        # Minimum energy to remain in persistor state
    "aging_rate_factor": 0.1,           # Aging rate multiplier for persistors (age 10x slower)
    
    "max_entry_prob": 0.5,               # Cap on entry probability
}

# -----------------------
# Bacterial Type Definitions
# -----------------------
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

# -----------------------
# Antibiotic Type Definitions
# -----------------------
ANTIBIOTIC_TYPES = {
    "penicillin": {
        "enzyme_weight": 0.8,      # β-lactamase effectiveness
        "efflux_weight": 0.2,
        "membrane_weight": 0.3,
        "repair_weight": 0.4,
        "toxicity_constant": 2.0,
        "color": "red",
        "decay_rate": 0.05         # per sim step
    },
    "tetracycline": {
        "enzyme_weight": 0.1,
        "efflux_weight": 0.9,      # efflux pumps very effective
        "membrane_weight": 0.4,
        "repair_weight": 0.3,
        "toxicity_constant": 1.5,
        "color": "orange",
        "decay_rate": 0.03         # per sim step
    },
    "vancomycin": {
        "enzyme_weight": 0.2,
        "efflux_weight": 0.3,
        "membrane_weight": 0.8,    # membrane changes very effective
        "repair_weight": 0.6,
        "toxicity_constant": 3.0,
        "color": "purple",
        "decay_rate": 0.07         # per sim step
    }
}

TRAIT_KEYS = ["enzyme_weight" , "efflux_weight", "membrane_weight", "repair_weight"]
N_TRAITS = len(TRAIT_KEYS)
N_BACTERIA_TYPES = len(BACTERIAL_TYPES)
TOXICITY_MAX = max([ANTIBIOTIC_TYPES[ab]["toxicity_constant"] for ab in ANTIBIOTIC_TYPES.keys()])
DOSE_MAX = 3
TOX_TIMES_DOSE_MAX = TOXICITY_MAX * DOSE_MAX
def antibiotic_resistances(sequencing_results, dtype, device)->Tuple[torch.Tensor, torch.Tensor, list[str]]:
    # sequency_results -> [K, M]
    ab_names = list(ANTIBIOTIC_TYPES.keys()) # [A]
    G = torch.as_tensor(sequencing_results, dtype=dtype, device=device)
    G = torch.clamp(G, 0.0, 1.0)  # optional safety
    W_rows = []
    tox_list = []
    for name in ab_names:
        ab = ANTIBIOTIC_TYPES[name]
        w = torch.tensor([ab[k] for k in TRAIT_KEYS], dtype=dtype, device=device)
        # normalize so sum=1 to keep dot products in [0,1]
        w = w / (w.sum() + 1e-8)
        W_rows.append(w)
        tox_list.append(ab["toxicity_constant"])
    W = torch.stack(W_rows, dim=0) 

    resistances = torch.clamp(G @ W.T, 0.0, 1.0)
    toxicities = torch.tensor(tox_list, dtype=dtype, device=device)  # [A]

    return resistances, toxicities, ab_names



if __name__ == "__main__":
    seq_res_test = np.array([
        [1,2,3,4],
        [4,3,2,1],
        [2,2,2,2],
        [1,1,1,1]
    ])
    antibiotic_resistencess(seq_res_test)
# -----------------------
# Biofilm Parameters
# -----------------------
BIOFILM_PARAMS = {
    # Formation parameters
    "formation_radius": 3.0,        # Detection radius for biofilm formation
    "min_neighbors": 3,             # Minimum neighbors required to create biofilm (including self)
    "formation_base_prob": 0.01,    # Base probability of creating biofilm per step
    "formation_stress_bonus": 1, # Additional formation probability under antibiotic stress
    
    # Protection and costs
    "base_protection": 1.5,         # Minimum antibiotic protection multiplier
    "max_protection": 3.0,          # Maximum antibiotic protection multiplier (at optimal size)
    "optimal_size": 10,             # Optimal cluster size for maximum protection
    "energy_cost": 0.03,            # Energy cost per step for maintaining biofilm
    "movement_penalty": 0.3,        # Movement speed multiplier (0.3 = 70% slower)
    
    # Exit parameters
    "exit_energy_threshold": 0.15,  # Exit if energy below this
    "exit_food_threshold": 0.1,     # Exit if local food below this
    "exit_threat_threshold": 0.05,   # Exit if antibiotic threat below this (low danger)
    "detachment_cost": 0.02         # Energy cost for detaching from biofilm
}
