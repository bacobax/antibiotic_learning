"""
Bacteria simulation model implementation.
"""

import random
import numpy as np
from scipy.ndimage import gaussian_filter
from mesa import Model
from mesa.space import ContinuousSpace
from collections import defaultdict, deque

from config import (
    WIDTH, HEIGHT, GRID_RES, FOOD_DIFFUSION_SIGMA, ANTIBIOTIC_DECAY,
    BACTERIAL_TYPES, ANTIBIOTIC_TYPES, BACTERIA_PER_TYPE, HGT_RADIUS, HGT_PROB
)
from bacterium import Bacterium
from tracking import IndividualTracker


class BacteriaModel(Model):
    """Main simulation model for bacteria population dynamics."""
    
    def __init__(self, N=None, width=WIDTH, height=HEIGHT, enable_hgt=True):
        super().__init__()
        self.width = width
        self.height = height
        self.space = ContinuousSpace(width, height, torus=False)
        self.random = random.Random()

        # Agent management
        self.agent_set = set()
        self._next_id = 0

        # Antibiotic management
        self.current_antibiotic = None
        self.available_antibiotics = list(ANTIBIOTIC_TYPES.keys())

        # Initialize fields
        self.field_w = GRID_RES
        self.field_h = GRID_RES
        self.food_field = np.zeros((self.field_w, self.field_h), dtype=float)
        self.antibiotic_field = np.zeros_like(self.food_field)
        self._initialize_food_patches()

        # Agent tracking
        self.to_remove = set()
        self.new_agents = []

        # Initialize bacteria population
        self._create_initial_population(N)

        self.running = True
        self.step_count = 0

        # Tracking system
        self.individual_tracker = IndividualTracker()
        self.enable_hgt = bool(enable_hgt)
        
        # History for plotting
        self.history = {
            'steps': [],
            'population': [],
            'total_food': [],
            'avg_energy': []
        }

    def _initialize_food_patches(self):
        """Initialize food field with Gaussian patches"""
        for _ in range(6):
            cx = random.uniform(0, self.field_w - 1)
            cy = random.uniform(0, self.field_h - 1)
            sigma = random.uniform(6, 18)
            amplitude = random.uniform(2.0, 5.0)
            self.add_gaussian_patch(self.food_field, cx, cy, sigma, amplitude)

    def _create_initial_population(self, N):
        """Create initial bacteria population"""
        if N is None:
            total_bacteria = len(BACTERIAL_TYPES) * BACTERIA_PER_TYPE
        else:
            total_bacteria = N
            
        bacteria_per_type = total_bacteria // len(BACTERIAL_TYPES)
        remainder = total_bacteria % len(BACTERIAL_TYPES)
        
        for i, bacterial_type in enumerate(BACTERIAL_TYPES.keys()):
            count = bacteria_per_type + (1 if i < remainder else 0)
            
            for _ in range(count):
                x, y = random.uniform(0, self.width), random.uniform(0, self.height)
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
        """Get statistics about the current population"""
        stats = {
            "total": len(self.agent_set),
            "by_type": {},
            "avg_traits": {},
            "avg_age": 0
        }
        
        if len(self.agent_set) == 0:
            return stats
        
        # Collect statistics
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
            for trait in ["enzyme", "efflux", "membrane", "repair", "age"]:
                stats[btype][trait] /= count
        
        # Add food and energy tracking
        total_food = np.sum(self.food_field)
        top_energies = sorted([a.energy for a in self.agent_set], reverse=True)[:10]
        avg_top_energy = np.mean(top_energies) if top_energies else 0
        
        # Record history
        self.history['steps'].append(self.step_count)
        self.history['population'].append(len(self.agent_set))
        self.history['total_food'].append(total_food)
        self.history['avg_energy'].append(avg_top_energy)
        
        stats['total_food'] = total_food
        stats['avg_energy'] = avg_top_energy
        stats['top_energies'] = top_energies
        
        return stats

    # Field utilities
    def add_gaussian_patch(self, field, cx, cy, sigma, amplitude):
        """Add a Gaussian patch to the field"""
        X, Y = np.meshgrid(
            np.arange(self.field_w), np.arange(self.field_h), indexing="ij"
        )
        patch = amplitude * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))
        field += patch

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

    def apply_antibiotic(self, amount):
        """Apply antibiotic to the field"""
        if amount <= 0:
            return
        self.antibiotic_field += float(amount)

    def horizontal_gene_transfer(self):
        """Exchange resistance traits between nearby bacteria"""
        agents = list(self.agent_set)
        for a in agents:
            try:
                neighbors = self.space.get_neighbors(a.pos, HGT_RADIUS, include_center=False)
            except Exception:
                neighbors = [
                    b for b in agents
                    if b is not a and np.hypot(b.pos[0] - a.pos[0], b.pos[1] - a.pos[1]) <= HGT_RADIUS
                ]
            
            for nb in neighbors:
                if random.random() < HGT_PROB:
                    mix = 0.3
                    traits = ['enzyme', 'efflux', 'membrane', 'repair']
                    for trait in traits:
                        a_val = getattr(a, trait)
                        nb_val = getattr(nb, trait)
                        new_a_val = a_val * (1 - mix) + nb_val * mix
                        new_nb_val = nb_val * (1 - mix) + a_val * mix
                        setattr(a, trait, float(min(max(new_a_val, 0.0), 1.0)))
                        setattr(nb, trait, float(min(max(new_nb_val, 0.0), 1.0)))

    def step(self):
        """Execute one simulation step"""
        # Update fields
        self.food_field = gaussian_filter(self.food_field, sigma=FOOD_DIFFUSION_SIGMA)
        self.antibiotic_field *= 1 - ANTIBIOTIC_DECAY

        # Prepare collections
        self.to_remove.clear()
        self.new_agents.clear()

        # Step each agent
        for a in list(self.agent_set):
            try:
                a.step()
            except Exception as e:
                print(f"Exception during step for bacterium {a.unique_id}: {e}")

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
            except Exception:
                pass
            self.agent_set.add(child)

        # Horizontal gene transfer
        if self.enable_hgt:
            try:
                self.horizontal_gene_transfer()
            except Exception:
                pass

        self.step_count += 1

        # Update tracking
        self.individual_tracker.update_tracked_individuals(self)
