"""
Configuration parameters for the bacteria simulation.
Contains all constants, growth parameters, and bacterial/antibiotic definitions.
"""

# -----------------------
# Spatial Configuration
# -----------------------
WIDTH = 100.0                       # continuous width
HEIGHT = 100.0
GRID_RES = 200                      # resolution for nutrient & antibiotic fields (square grid)

# -----------------------
# Simulation Parameters
# -----------------------
INITIAL_BACTERIA = 1
FOOD_DIFFUSION_SIGMA = 1.0          # for gaussian_filter diffusion approximation
FOOD_DECAY = 0.3
BACTERIA_SPEED = 0.3                # scaling factor for bacterium movement speed
MUTATION_STD = 0.03
ANTIBIOTIC_DECAY = 0.05             # per sim step
CONTROL_INTERVAL = 20               # timesteps between control checks

# HGT Parameters
HGT_RADIUS = 1.5                    # horizontal gene transfer radius
HGT_PROB = 0.001                    # probability of HGT per neighbor per step

# Simulation speed settings
DEFAULT_STEPS_PER_SECOND = 5        # simulation steps per second
MIN_STEPS_PER_SECOND = 1            # minimum speed (slowest)
MAX_STEPS_PER_SECOND = 20           # maximum speed (fastest)
ANIMATION_FPS = 30                  # visual update rate (frames per second)

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

# Number of bacteria per type at initialization
BACTERIA_PER_TYPE = 1

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
