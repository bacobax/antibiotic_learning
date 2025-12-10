"""
Bacteria simulation model implementation.
"""

import math
import random
import numpy as np
from mesa import Model
from mesa.space import ContinuousSpace
import heapq
from typing import Optional


from simulation.simulation_config import (
    WIDTH,
    HEIGHT,
    GRID_RES,
    FOOD_DIFFUSION_SIGMA,
    ANTIBIOTIC_DECAY,
    BACTERIAL_TYPES,
    ANTIBIOTIC_TYPES,
    BACTERIA_PER_TYPE,
    HGT_RADIUS,
    HGT_PROB,
    FOOD_PATCH_COUNT,
    FOOD_PATCH_AMPLITUDE_MIN,
    FOOD_PATCH_AMPLITUDE_MAX,
    FOOD_PATCH_SIGMA_MIN,
    FOOD_PATCH_SIGMA_MAX,
    BIOFILM_PARAMS,
    QUORUM_SENSING_PARAMS,
    FOOD_REPLENISHMENT,
    FIELD_ACCELERATION,
)
from simulation.bacterium import Bacterium
from simulation.tracking import IndividualTracker
from simulation.biofilm_manager import BiofilmManager
from simulation.field_backend import FieldBackend



class BacteriaModel(Model):
    """Main simulation model for bacteria population dynamics."""

    def __init__(
        self,
        N=None,
        width=WIDTH,
        height=HEIGHT,
        *,
        enable_individual_tracking: bool = True,
        max_individual_history: int = 1000,
        max_tracked_individuals: Optional[int] = 2000,
        max_history_steps: Optional[int] = 2000,
        use_torch_fields: Optional[bool] = None,
        field_device: Optional[str] = None,
        enable_food_diffusion: Optional[bool] = None,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.space = ContinuousSpace(width, height, torus=False)
        self.random = random.Random()

        if use_torch_fields is None:
            use_torch_fields = FIELD_ACCELERATION.get("enabled", False)
        if field_device is None:
            field_device = FIELD_ACCELERATION.get("device", "cuda")
        self.field_backend = FieldBackend(enabled=use_torch_fields, device=field_device)
        backend_status = "ON" if self.field_backend.enabled else "OFF"
        backend_device = getattr(self.field_backend, "device", None)
        backend_device_str = str(backend_device) if backend_device is not None else "cpu"
        print(f"[FieldBackend] acceleration {backend_status} (device={backend_device_str})")

        self._food_diffusion_sigma = FOOD_DIFFUSION_SIGMA
        if enable_food_diffusion is None:
            enable_food_diffusion = self.field_backend.enabled
        self._food_diffusion_enabled = bool(enable_food_diffusion) and self._food_diffusion_sigma > 0.0
        if self._food_diffusion_enabled and not self.field_backend.enabled:
            print("[FieldBackend] Food diffusion enabled without Torch acceleration; falling back to SciPy (slower).")

        # Agent management
        self.agent_set = set()
        self._next_id = 0

        # Antibiotic management - now supports multiple simultaneous antibiotics
        self.available_antibiotics = list(ANTIBIOTIC_TYPES.keys())
        self.current_antibiotic = (
            self.available_antibiotics[0] if self.available_antibiotics else None
        )  # Default to first antibiotic

        # Initialize fields
        self.field_w = GRID_RES
        self.field_h = GRID_RES
        self.food_field = np.zeros((self.field_w, self.field_h), dtype=float)

        # Multiple antibiotic fields - one per antibiotic type
        self.antibiotic_fields = {
            ab_type: np.zeros((self.field_w, self.field_h), dtype=float)
            for ab_type in ANTIBIOTIC_TYPES.keys()
        }
        
        # Quorum sensing signal field
        self.qs_signal_field = np.zeros((self.field_w, self.field_h), dtype=float)

        self._initialize_food_patches()
        # Store the starting total food amount so replenishment can restore to this
        self.initial_food_total = float(np.sum(self.food_field))

        # Agent tracking
        self.to_remove = set()
        self.new_agents = []

        # Store initial bacteria count for reset
        self._initial_bacteria_count = N

        # Initialize bacteria population
        self._create_initial_population(N)

        self.running = True
        self.step_count = 0

        # Tracking configuration and system
        self._tracker_enabled = bool(enable_individual_tracking)
        self._tracker_history_len = max(1, int(max_individual_history))
        if max_tracked_individuals is not None and max_tracked_individuals > 0:
            self._max_tracked_individuals = int(max_tracked_individuals)
        else:
            self._max_tracked_individuals = None
        if max_history_steps is not None and max_history_steps > 0:
            self.max_history_steps = int(max_history_steps)
        else:
            self.max_history_steps = None

        self.individual_tracker = IndividualTracker(
            max_history=self._tracker_history_len,
            max_individuals=self._max_tracked_individuals,
            enabled=self._tracker_enabled,
        )
        
        # Biofilm management system
        self.biofilm_manager = BiofilmManager(
            self,
            BIOFILM_PARAMS,
            field_backend=self.field_backend,
        )
        
        # Biofilm tracking (legacy - kept for compatibility)
        self._next_biofilm_id = 0  # Counter for unique biofilm IDs

        # History for plotting
        self.history = self._init_history()

    def _initialize_food_patches(self):
        """Initialize food field with Gaussian patches"""
        for _ in range(FOOD_PATCH_COUNT):
            cx = random.uniform(0, self.field_w - 1)
            cy = random.uniform(0, self.field_h - 1)
            sigma = random.uniform(FOOD_PATCH_SIGMA_MIN, FOOD_PATCH_SIGMA_MAX)
            amplitude = random.uniform(
                FOOD_PATCH_AMPLITUDE_MIN, FOOD_PATCH_AMPLITUDE_MAX
            )
            self.add_gaussian_patch(self.food_field, cx, cy, sigma, amplitude)

    def _create_initial_population(self, N):
        """Create initial bacteria population"""
        if N is None:
            total_bacteria = len(BACTERIAL_TYPES) * BACTERIA_PER_TYPE
        else:
            total_bacteria = N

        bacteria_per_type = total_bacteria // len(BACTERIAL_TYPES)
        remainder = total_bacteria % len(BACTERIAL_TYPES)
        epsilon = 1e-6
        for i, bacterial_type in enumerate(BACTERIAL_TYPES.keys()):
            count = bacteria_per_type + (1 if i < remainder else 0)

            for _ in range(count):
                # ensure positions are strictly within bounds (ContinuousSpace is non-toroidal)
                x = random.uniform(0.0, max(0.0, self.width - epsilon))
                y = random.uniform(0.0, max(0.0, self.height - epsilon))
                bacterium = Bacterium(self, bacterial_type=bacterial_type)
                self.agent_set.add(bacterium)
                self.space.place_agent(bacterium, (x, y))

    def next_id(self):
        """Generate next unique ID"""
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
        """Get statistics about the current population
        
        Returns:
            stats (dict): Dictionary containing overall statistics and per-type information
            traits_matrix (np.ndarray): Matrix of shape (K, T) where:
                - K = number of bacterial types (from BACTERIAL_TYPES)
                - T = number of traits (4: enzyme, efflux, membrane, repair)
                - Each row contains average trait values for that bacterial type
                - Rows are ordered according to BACTERIAL_TYPES.keys()
        """
        stats = {
            "total": len(self.agent_set),
            "by_type": {},
            "avg_traits": {},
            "avg_age": 0,
        }

        # Always add food and energy tracking to stats, even if population is 0
        stats["total_food"] = float(np.sum(self.food_field))
        stats["avg_energy"] = 0.0
        stats["avg_energy_top"] = 0.0
        stats["avg_energy_worst"] = 0.0

        # Initialize traits matrix (K x T)
        # K = number of bacterial types, T = 4 traits
        bacterial_types_list = list(BACTERIAL_TYPES.keys())
        traits_list = ["enzyme", "efflux", "membrane", "repair"]
        K = len(bacterial_types_list)
        T = len(traits_list)
        traits_matrix = np.zeros((K, T), dtype=float)

        if len(self.agent_set) == 0:
            stats["traits_matrix"] = traits_matrix
            return stats

        # Collect statistics
        trait_sums = {"enzyme": 0, "efflux": 0, "membrane": 0, "repair": 0, "age": 0}
        trait_arrays = {trait: [] for trait in trait_sums}
        energy_array = []

        for bacterium in self.agent_set:
            btype = bacterium.bacterial_type
            if btype not in stats["by_type"]:
                stats["by_type"][btype] = 0
                stats[btype] = {
                    "enzyme": 0,
                    "efflux": 0,
                    "membrane": 0,
                    "repair": 0,
                    "age": 0,
                }

            stats["by_type"][btype] += 1
            for trait in trait_sums:
                value = getattr(bacterium, trait)
                stats[btype][trait] += value
                trait_sums[trait] += value
                trait_arrays[trait].append(value)
            energy_array.append(bacterium.energy)

        # Calculate averages
        total = len(self.agent_set)
        for trait in trait_sums:
            stats["avg_traits"][trait] = trait_sums[trait] / total

        for btype in stats["by_type"]:
            count = stats["by_type"][btype]
            for trait in ["enzyme", "efflux", "membrane", "repair", "age"]:
                stats[btype][trait] /= count

        # Populate traits matrix
        for i, btype in enumerate(bacterial_types_list):
            for j, trait in enumerate(traits_list):
                if btype in stats["by_type"] and stats["by_type"][btype] > 0:
                    traits_matrix[i, j] = stats[btype][trait]
                else:
                    traits_matrix[i, j] = 0.0

        # Update energy tracking with actual values
        stats["avg_energy"] = float(np.mean(energy_array)) if energy_array else 0.0
        stats["avg_energy_top"] = (
            float(np.mean(heapq.nlargest(10, energy_array))) if energy_array else 0.0
        )
        stats["avg_energy_worst"] = (
            float(np.mean(heapq.nsmallest(10, energy_array))) if energy_array else 0.0
        )

        # Add traits matrix to stats dictionary
        stats["traits_matrix"] = traits_matrix

        return stats

    def _record_history(self):
        """Record current stats to history - called every step"""
        stats = self.get_population_stats()

        # Record basic stats
        self.history["steps"].append(self.step_count)
        self.history["population"].append(len(self.agent_set))
        self.history["total_food"].append(stats["total_food"])
        self.history["avg_energy"].append(stats["avg_energy"])
        self.history["avg_energy_top"].append(stats["avg_energy_top"])
        self.history["avg_energy_worst"].append(stats["avg_energy_worst"])

        # Record antibiotic concentrations (average across field)
        for ab_type in ANTIBIOTIC_TYPES.keys():
            avg_concentration = float(np.mean(self.antibiotic_fields[ab_type]))
            self.history[f"antibiotic_{ab_type}"].append(avg_concentration)

        # Record per-type trait averages
        for btype in BACTERIAL_TYPES.keys():
            if btype in stats["by_type"] and stats["by_type"][btype] > 0:
                for trait in ["enzyme", "efflux", "membrane", "repair"]:
                    self.history[f"{btype}_avg_{trait}"].append(stats[btype][trait])
            else:
                # No bacteria of this type, append 0
                for trait in ["enzyme", "efflux", "membrane", "repair"]:
                    self.history[f"{btype}_avg_{trait}"].append(0.0)

        self._trim_history()

    def _trim_history(self) -> None:
        if self.max_history_steps is None:
            return

        max_len = self.max_history_steps
        for key, series in self.history.items():
            excess = len(series) - max_len
            if excess > 0:
                del series[:excess]

    # Field utilities
    def add_gaussian_patch(self, field, cx, cy, sigma, amplitude):
        """Add a Gaussian patch to the field using a clipped window to reduce allocations."""
        if amplitude == 0.0:
            return

        if sigma <= 0.0:
            x = int(round(np.clip(cx, 0, self.field_w - 1)))
            y = int(round(np.clip(cy, 0, self.field_h - 1)))
            field[x, y] += amplitude
            return

        radius = max(1, int(math.ceil(3.0 * sigma)))
        x0 = max(0, int(np.floor(cx)) - radius)
        x1 = min(self.field_w, int(np.floor(cx)) + radius + 1)
        y0 = max(0, int(np.floor(cy)) - radius)
        y1 = min(self.field_h, int(np.floor(cy)) + radius + 1)

        if x0 >= x1 or y0 >= y1:
            return

        xs = np.arange(x0, x1, dtype=np.float32) - float(cx)
        ys = np.arange(y0, y1, dtype=np.float32) - float(cy)
        xs_sq = xs[:, None] ** 2
        ys_sq = ys[None, :] ** 2
        inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)
        patch = amplitude * np.exp(-(xs_sq + ys_sq) * inv_two_sigma_sq)
        field[x0:x1, y0:y1] += patch.astype(field.dtype, copy=False)

    def nutrient_to_field_coords(self, pos):
        """Convert position to field coordinates"""
        fx = (pos[0] / self.width) * (self.field_w - 1)
        fy = (pos[1] / self.height) * (self.field_h - 1)
        return fx, fy

    def sample_field(self, field, fx, fy):
        """Sample field value using bilinear interpolation"""
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
        """Subtract amount from field at position"""
        x = int(round(fx))
        y = int(round(fy))
        x = min(max(x, 0), self.field_w - 1)
        y = min(max(y, 0), self.field_h - 1)
        field[x, y] = max(0.0, field[x, y] - amount)

    def compute_gradient_at_field(self, fx, fy):
        """Compute nutrient gradient at field position"""
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

    def apply_antibiotic(self, antibiotic_type, amount, verbose=False):
        """Apply antibiotic of specific type to the field

        Args:
            antibiotic_type: The type of antibiotic to apply (must be in ANTIBIOTIC_TYPES)
            amount: The concentration to add to the field
            verbose: If True, print application details (default: False for cleaner logs)
        """
        if amount <= 0:
            return

        if antibiotic_type not in ANTIBIOTIC_TYPES:
            if verbose:
                print(f"Warning: Unknown antibiotic type '{antibiotic_type}'")
            return

        self.antibiotic_fields[antibiotic_type] += float(amount)
        
        # Only log if explicitly requested to avoid console spam during RL training
        if verbose:
            avg_conc = float(np.mean(self.antibiotic_fields[antibiotic_type]))
            print(f"Applied {amount:.3f} of {antibiotic_type} (total avg: {avg_conc:.3f})")

    def get_antibiotic_concentrations_at_position(self, fx, fy):
        """Get all antibiotic concentrations at a position as a dictionary

        Returns:
            dict: Mapping of antibiotic_type -> concentration
        """
        concentrations = {}
        for ab_type, ab_field in self.antibiotic_fields.items():
            concentrations[ab_type] = self.sample_field(ab_field, fx, fy)
        return concentrations

    def get_total_antibiotic_at_position(self, fx, fy):
        """Get combined antibiotic concentration at a position

        Returns the sum of all antibiotic types at the given position.
        This is used for stress calculations and general antibiotic presence.
        """
        total = 0.0
        for ab_field in self.antibiotic_fields.values():
            total += self.sample_field(ab_field, fx, fy)
        return total

    def create_biofilm(self, initiator):
        """Create a new biofilm cluster with initiator and nearby bacteria
        
        Args:
            initiator: The bacterium that triggered biofilm formation
        """
        
        # Generate unique biofilm ID
        biofilm_id = self._next_biofilm_id
        self._next_biofilm_id += 1
        
        # Get neighbors within formation radius
        try:
            neighbors = self.space.get_neighbors(
                initiator.pos,
                BIOFILM_PARAMS["formation_radius"],
                include_center=True  # Include initiator
            )
        except:
            neighbors = [initiator]
        
        # Assign all neighbors to this biofilm
        biofilm_members = []
        for bacterium in neighbors:
            if hasattr(bacterium, 'biofilm_id') and bacterium.biofilm_id is None:
                bacterium.biofilm_id = biofilm_id
                biofilm_members.append(bacterium)
    
    def horizontal_gene_transfer(self):
        """Exchange resistance traits between nearby bacteria"""
        agents = list(self.agent_set)
        for a in agents:
            try:
                neighbors = self.space.get_neighbors(
                    a.pos, HGT_RADIUS, include_center=False
                )
            except Exception:
                neighbors = [
                    b
                    for b in agents
                    if b is not a
                    and np.hypot(b.pos[0] - a.pos[0], b.pos[1] - a.pos[1]) <= HGT_RADIUS
                ]

            for nb in neighbors:
                if random.random() < HGT_PROB and a.has_hgt_gene and nb.has_hgt_gene:
                    mix = 0.3
                    traits = ["enzyme", "efflux", "membrane", "repair"]
                    for trait in traits:
                        a_val = getattr(a, trait)
                        nb_val = getattr(nb, trait)
                        new_a_val = a_val * (1 - mix) + nb_val * mix
                        new_nb_val = nb_val * (1 - mix) + a_val * mix
                        setattr(a, trait, float(min(max(new_a_val, 0.0), 1.0)))
                        setattr(nb, trait, float(min(max(new_nb_val, 0.0), 1.0)))
    
    # -----------------------
    # Quorum Sensing Methods
    # -----------------------
    
    def add_qs_signal(self, fx, fy, amount):
        """Add quorum sensing signal to the field at a specific position
        
        Args:
            fx, fy: Field coordinates
            amount: Amount of signal to add
        """
        x = int(round(fx))
        y = int(round(fy))
        x = min(max(x, 0), self.field_w - 1)
        y = min(max(y, 0), self.field_h - 1)
        
        self.qs_signal_field[x, y] += amount
    
    def get_qs_concentration(self, fx, fy):
        """Get quorum sensing signal concentration at a position
        
        Args:
            fx, fy: Field coordinates
            
        Returns:
            Concentration of QS signal at the position
        """
        return self.sample_field(self.qs_signal_field, fx, fy)
    
    def update_qs_field(self):
        """Update quorum sensing signal field with diffusion and decay
        
        Applies:
        1. Diffusion (spreading of signals)
        2. Decay (degradation of signals)
        """
        
        diffusion_coef = QUORUM_SENSING_PARAMS["diffusion_coefficient"]
        decay_rate = QUORUM_SENSING_PARAMS["decay_rate"]
        
        # Apply diffusion using Gaussian filter
        if diffusion_coef > 0:
            self.qs_signal_field = self.field_backend.gaussian_filter(
                self.qs_signal_field,
                sigma=diffusion_coef,
                mode='constant',
                cval=0.0,
            )
        
        # Apply decay
        self.qs_signal_field *= (1.0 - decay_rate)
        
        # Ensure non-negative
        self.qs_signal_field = np.maximum(self.qs_signal_field, 0.0)

    def replenish_food(self):
        """Add periodic food impulses to prevent starvation
        
        Creates new Gaussian food patches at random locations to simulate
        continuous nutrient availability in the environment.
        """
        if not FOOD_REPLENISHMENT["enabled"]:
            return

        # 1) Add new random patches (locations and spreads may differ each time)
        patch_count = FOOD_REPLENISHMENT["patch_count"]

        # Accumulate new patches into a temporary field so we can adjust after
        add_field = np.zeros_like(self.food_field)
        for _ in range(patch_count):
            cx = random.uniform(0, self.field_w - 1)
            cy = random.uniform(0, self.field_h - 1)
            sigma = random.uniform(
                FOOD_REPLENISHMENT["sigma_min"],
                FOOD_REPLENISHMENT["sigma_max"],
            )
            amplitude = random.uniform(
                FOOD_REPLENISHMENT["amplitude_min"],
                FOOD_REPLENISHMENT["amplitude_max"],
            )
            self.add_gaussian_patch(add_field, cx, cy, sigma, amplitude)

        # Apply additions
        self.food_field += add_field

        # 2) Rescale total food to match the starting amount, preserving timing
        target_total = getattr(self, "initial_food_total", float(np.sum(self.food_field)))
        current_total = float(np.sum(self.food_field))
        if current_total <= 1e-12:
            # Degenerate case: distribute uniformly
            uniform_value = target_total / (self.field_w * self.field_h)
            self.food_field[:] = uniform_value
        else:
            scale = target_total / current_total
            self.food_field *= float(scale)

        # # Optional: Log replenishment event (only occasionally to avoid spam)
        # if self.step_count % 200 == 0:
        #     total_food = float(np.sum(self.food_field))
        #     print(
        #         f"[Food] Step {self.step_count}: Added {patch_count} patches and normalized total to {total_food:.2f}"
        #     )

    def step(self):
        """Execute one simulation step"""
        # Check if it's time to replenish food
        if (FOOD_REPLENISHMENT["enabled"] and 
            self.step_count > 0 and 
            self.step_count % FOOD_REPLENISHMENT["period"] == 0):
            self.replenish_food()

        if self._food_diffusion_enabled:
            self.food_field = self.field_backend.gaussian_filter(
                self.food_field,
                sigma=self._food_diffusion_sigma,
                mode='constant',
                cval=0.0,
            )
        
        # Update fields - decay each antibiotic independently
        for ab_type, ab_field in self.antibiotic_fields.items():
            decay_rate = ANTIBIOTIC_TYPES[ab_type].get("decay_rate", ANTIBIOTIC_DECAY)
            self.antibiotic_fields[ab_type] *= 1 - decay_rate
        
        # Update quorum sensing signal field (diffusion + decay)
        self.update_qs_field()
        
        # Update biofilm EPS field (diffusion + decay + production)
        self.biofilm_manager.update_eps_field()
        
        # Update biofilm cell timers and transitions
        self.biofilm_manager.update_biofilm_cells()

        # Prepare collections
        self.to_remove.clear()
        self.new_agents.clear()

        # Step each agent
        for a in list(self.agent_set):
            try:
                a.step()
            except Exception as e:
                print(f"Exception during step for bacterium {a.unique_id}: {e}")

        # Update tracking BEFORE removing dead agents (to capture final state)
        self.individual_tracker.update_tracked_individuals(self, self.to_remove)

        # Remove dead agents
        for a in list(self.to_remove):
            try:
                self.space.remove_agent(a)
            except Exception:
                pass
            if a in self.agent_set:
                self.agent_set.remove(a)

        # Add newborns
        for child, child_pos in self.new_agents:
            try:
                self.space.place_agent(child, child_pos)
                self.agent_set.add(child)
            except Exception as e:
                # If placement fails, don't add to agent_set
                print(f"Failed to place child bacterium {child.unique_id}: {e}")

        # Horizontal gene transfer
        try:
            self.horizontal_gene_transfer()
        except Exception:
            pass

        self.step_count += 1

        #print(f"Step {self.step_count}: Population = {len(self.agent_set)}", flush=True)
        # Record history every step
        self._record_history()

    def reset(self, N: Optional[int] = None):
        """Reset simulation to initial conditions.

        Args:
            N: Optional override for the total bacteria count. When None, the
               constructor-provided count is reused.
        """
        if N is not None:
            self._initial_bacteria_count = N

        # Clear all agents
        for agent in list(self.agent_set):
            try:
                self.space.remove_agent(agent)
            except Exception:
                pass
        self.agent_set.clear()

        # Reset ID counter
        self._next_id = 0

        # Reset fields
        self.food_field = np.zeros((self.field_w, self.field_h), dtype=float)

        # Reset all antibiotic fields
        self.antibiotic_fields = {
            ab_type: np.zeros((self.field_w, self.field_h), dtype=float)
            for ab_type in ANTIBIOTIC_TYPES.keys()
        }

        # Reset quorum sensing signal field
        self.qs_signal_field = np.zeros((self.field_w, self.field_h), dtype=float)

        self._initialize_food_patches()
        # Recompute initial food total after reinitializing patches
        self.initial_food_total = float(np.sum(self.food_field))

        # Clear tracking collections
        self.to_remove.clear()
        self.new_agents.clear()

        # Reset step counter and runtime flags
        self.step_count = 0
        self.running = True

        # Reset history
        self.history = self._init_history()

        # Reset tracking system
        self.individual_tracker = IndividualTracker(
            max_history=self._tracker_history_len,
            max_individuals=self._max_tracked_individuals,
            enabled=self._tracker_enabled,
        )
        self.biofilm_manager = BiofilmManager(
            self,
            BIOFILM_PARAMS,
            field_backend=self.field_backend,
        )
        self._next_biofilm_id = 0

        # Recreate initial population
        self._create_initial_population(self._initial_bacteria_count)

        # Avoid spamming stdout during RL training

    def _init_history(self):
        history = {
            "steps": [],
            "population": [],
            "total_food": [],
            "avg_energy": [],
            "avg_energy_top": [],
            "avg_energy_worst": [],
        }

        for ab_type in ANTIBIOTIC_TYPES.keys():
            history[f"antibiotic_{ab_type}"] = []

        for btype in BACTERIAL_TYPES.keys():
            for trait in ["enzyme", "efflux", "membrane", "repair"]:
                history[f"{btype}_avg_{trait}"] = []

        return history
