"""
Biofilm formation and management system with QS integration.

This module implements realistic biofilm dynamics with:
- State-based transitions (planktonic → reversible → irreversible → mature)
- QS-dependent activation
- EPS (extracellular polymeric substance) field dynamics
- Persister exclusion (persisters cannot form or contribute to biofilms)

Biological Context:
    Biofilm formation is a multi-stage process where bacteria attach to surfaces,
    produce protective matrix (EPS), and form structured communities. This process
    is coordinated by quorum sensing (cell-cell communication via autoinducers).
    Persister cells are dormant and do not participate in biofilm formation.

Computational Complexity:
    - Neighbor counting: O(N) per cell using spatial hashing
    - EPS field update: O(grid_size) with vectorized operations
    - State transitions: O(N) where N = number of bacteria
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from enum import Enum
from typing import Set, Dict, Tuple, Optional


class BiofilmState(Enum):
    """Biofilm developmental states for bacterial agents.
    
    States:
        PLANKTONIC: Free-swimming, no attachment
        REVERSIBLE_ATTACH: Initial surface contact, can detach easily
        IRREVERSIBLE_ATTACH: Firmly attached, producing EPS
        MATURE: Fully developed biofilm cell with maximum EPS/protection
        PERSISTER: Dormant state (excluded from biofilm processes)
    """
    PLANKTONIC = "planktonic"
    REVERSIBLE_ATTACH = "reversible_attach"
    IRREVERSIBLE_ATTACH = "irreversible_attach"
    MATURE = "mature"


class BiofilmManager:
    """Manages biofilm formation, EPS dynamics, and state transitions.
    
    This class coordinates biofilm-related processes across the simulation:
    - Tracking EPS concentration in a continuous 2D field
    - Managing bacterial state transitions through biofilm lifecycle
    - Enforcing persister exclusion from biofilm processes
    - Computing protection and growth modifiers based on biofilm state
    
    The EPS field represents the extracellular polymeric substance matrix
    that biofilms produce. High EPS concentrations provide protection from
    antibiotics and reduce diffusion rates.
    
    Attributes:
        model: Reference to the main simulation model
        eps_field: 2D numpy array of EPS concentrations [grid_w, grid_h]
        params: Dictionary of biofilm parameters from config
    """
    
    def __init__(self, model, params: dict):
        """Initialize biofilm manager.
        
        Args:
            model: Main simulation model (provides grid dimensions, agents)
            params: Biofilm parameters dictionary (BIOFILM_PARAMS from config)
        """
        self.model = model
        self.params = params
        
        # Initialize EPS field (same resolution as nutrient/antibiotic fields)
        self.eps_field = np.zeros((model.field_w, model.field_h), dtype=float)
        
    def update_eps_field(self):
        """Update EPS field with production, diffusion, and decay.
        
        Process:
            1. Add EPS from irreversibly attached and mature cells
            2. Apply Gaussian diffusion to spread EPS locally
            3. Apply exponential decay to degrade EPS over time
            
        Biological meaning:
            EPS is constantly produced by biofilm cells and degrades over time.
            It diffuses slowly through the environment, creating gradients that
            attract new recruits.
            
        Computational complexity: O(grid_size) for vectorized operations
        """
        # Reset production (will accumulate from bacteria)
        eps_production = np.zeros_like(self.eps_field)
        
        # Accumulate EPS production from biofilm cells
        for bacterium in self.model.agent_set:
            # Skip persisters and planktonic cells
            if not hasattr(bacterium, 'biofilm_state'):
                continue
                
            if bacterium.is_persister:
                continue
                
            # Only irreversibly attached and mature cells produce EPS
            if bacterium.biofilm_state not in [BiofilmState.IRREVERSIBLE_ATTACH, BiofilmState.MATURE]:
                continue
            
            if bacterium.pos is None:
                continue
                
            # Get grid coordinates
            fx, fy = self.model.nutrient_to_field_coords(bacterium.pos)
            x = int(round(fx))
            y = int(round(fy))
            x = np.clip(x, 0, self.model.field_w - 1)
            y = np.clip(y, 0, self.model.field_h - 1)
            
            # Calculate production rate
            base_rate = self.params["eps_production_base"]
            
            # QS boost for EPS production
            if bacterium.qs_active:
                production = base_rate * self.params["eps_production_qs_mult"]
            else:
                production = base_rate
            
            # Add to field
            eps_production[x, y] += production
        
        # Update field: add production
        self.eps_field += eps_production
        
        # Apply diffusion (EPS spreads slowly)
        diffusion_sigma = 0.3  # Slower than AI diffusion
        self.eps_field = gaussian_filter(
            self.eps_field,
            sigma=diffusion_sigma,
            mode='constant',
            cval=0.0
        )
        
        # Apply decay
        decay_rate = self.params["eps_decay_rate"]
        self.eps_field *= (1.0 - decay_rate)
        
        # Ensure non-negative
        self.eps_field = np.maximum(self.eps_field, 0.0)
    
    def get_eps_concentration(self, fx: float, fy: float) -> float:
        """Get EPS concentration at a field position.
        
        Args:
            fx, fy: Field coordinates (continuous)
            
        Returns:
            EPS concentration at the position (bilinear interpolation)
        """
        return self.model.sample_field(self.eps_field, fx, fy)
    
    def count_nonpersister_neighbors(self, bacterium, radius: float) -> int:
        """Count nearby non-persister bacteria within radius.
        
        Args:
            bacterium: The bacterium to count neighbors for
            radius: Search radius
            
        Returns:
            Number of non-persister neighbors (excluding self)
            
        Biological meaning:
            Biofilm formation requires sufficient cell density. Only active
            (non-persister) cells contribute to the quorum needed for attachment.
            
        Computational complexity: O(N) worst case, but spatial hashing in Mesa
            makes this approximately O(k) where k is local neighbor count.
        """
        if bacterium.pos is None:
            return 0
        
        try:
            neighbors = self.model.space.get_neighbors(
                bacterium.pos,
                radius,
                include_center=False
            )
        except Exception:
            return 0
        
        # Count only non-persisters
        count = sum(
            1 for n in neighbors
            if hasattr(n, 'is_persister') and not n.is_persister
        )
        
        return count
    
    def try_attach(self, bacterium, local_antibiotics: Dict[str, float]) -> bool:
        """Attempt initial (reversible) attachment for planktonic cell.
        
        Args:
            bacterium: The bacterium attempting attachment
            local_antibiotics: Dict of antibiotic concentrations at position
            
        Returns:
            True if attachment occurs, False otherwise
            
        Biological meaning:
            Initial attachment is reversible and depends on:
            - Local cell density (need neighbors for biofilm formation)
            - Quorum sensing activation (coordinated response)
            - Environmental stress (antibiotics trigger attachment)
            
        Requirements:
            - Must be planktonic (not already attached)
            - Must NOT be a persister
            - Must have sufficient non-persister neighbors
            - Probabilistic check with stress/QS bonuses
        """
        # Persisters cannot attach
        if bacterium.is_persister:
            return False
        
        # Must be planktonic
        if bacterium.biofilm_state != BiofilmState.PLANKTONIC:
            return False
        
        # Count non-persister neighbors
        neighbor_count = self.count_nonpersister_neighbors(
            bacterium,
            self.params["formation_radius"]
        )
        
        # Need minimum neighbors
        if neighbor_count < self.params["min_neighbors"]:
            return False
        
        # Calculate attachment probability
        prob = self.params["attachment_base_prob"]
        
        # QS bonus
        if bacterium.qs_active:
            prob += self.params["attachment_qs_bonus"]
        
        # Stress bonus (antibiotic pressure)
        from simulation.simulation_config import ANTIBIOTIC_TYPES
        total_stress = sum(
            conc * ANTIBIOTIC_TYPES[ab_type]["toxicity_constant"]
            for ab_type, conc in local_antibiotics.items()
            if conc > 0
        )
        if total_stress > 0.1:
            prob += self.params["attachment_stress_bonus"]
        
        # Probabilistic attachment
        if np.random.random() < prob:
            # Transition to reversible attachment
            bacterium.biofilm_state = BiofilmState.REVERSIBLE_ATTACH
            bacterium.adhesion_timer = 0
            bacterium.maturation_timer = 0
            return True
        
        return False
    
    def try_irreversible_attach(self, bacterium) -> bool:
        """Transition from reversible to irreversible attachment.
        
        Args:
            bacterium: The bacterium attempting irreversible attachment
            
        Returns:
            True if transition occurs, False otherwise
            
        Biological meaning:
            After sufficient time in reversible attachment, cells produce
            adhesins and commit to biofilm formation. This transition is
            accelerated by QS activation, which coordinates the community.
            
        Requirements:
            - Must be reversibly attached
            - Must NOT be a persister
            - Timer must exceed minimum duration
            - QS must be active (coordinated commitment)
        """
        # Persisters cannot progress
        if bacterium.is_persister:
            return False
        
        # Must be reversibly attached
        if bacterium.biofilm_state != BiofilmState.REVERSIBLE_ATTACH:
            return False
        
        # Check timer
        if bacterium.adhesion_timer < self.params["reversible_duration_min"]:
            return False
        
        # Require QS activation for irreversible commitment
        if not bacterium.qs_active:
            return False
        
        # Transition to irreversible attachment
        bacterium.biofilm_state = BiofilmState.IRREVERSIBLE_ATTACH
        bacterium.maturation_timer = 0
        return True
    
    def try_mature(self, bacterium) -> bool:
        """Transition from irreversible attachment to mature biofilm.
        
        Args:
            bacterium: The bacterium attempting maturation
            
        Returns:
            True if maturation occurs, False otherwise
            
        Biological meaning:
            Fully mature biofilm cells have maximum EPS production and
            protection. Maturation takes time and represents the final
            developmental stage where cells are embedded in thick matrix.
        """
        # Persisters cannot mature
        if bacterium.is_persister:
            return False
        
        # Must be irreversibly attached
        if bacterium.biofilm_state != BiofilmState.IRREVERSIBLE_ATTACH:
            return False
        
        # Check maturation timer
        if bacterium.maturation_timer < self.params["maturation_time"]:
            return False
        
        # Transition to mature
        bacterium.biofilm_state = BiofilmState.MATURE
        return True
    
    def maybe_detach(self, bacterium, local_food: float, local_antibiotics: Dict[str, float]) -> bool:
        """Check if bacterium should detach from biofilm.
        
        Args:
            bacterium: The bacterium potentially detaching
            local_food: Local nutrient concentration
            local_antibiotics: Dict of antibiotic concentrations
            
        Returns:
            True if detachment occurs, False otherwise
            
        Biological meaning:
            Biofilm cells can detach due to:
            - Nutrient starvation (seek better resources)
            - Energy depletion (cannot maintain attachment)
            - Excessive antibiotic stress (flee toxic environment)
            - Random dispersal events (explore new niches)
            - Environmental shear forces
            
            Mature cells are most resistant to detachment due to strong
            adhesion and EPS protection. Reversibly attached cells detach easily.
        """
        # Persisters and planktonic cells don't detach
        if bacterium.is_persister or bacterium.biofilm_state == BiofilmState.PLANKTONIC:
            return False
        
        # Calculate detachment resistance based on state
        if bacterium.biofilm_state == BiofilmState.REVERSIBLE_ATTACH:
            detach_resistance = 0.2  # Easy to detach
        elif bacterium.biofilm_state == BiofilmState.IRREVERSIBLE_ATTACH:
            detach_resistance = 0.5  # Moderate resistance
        else:  # MATURE
            detach_resistance = self.params["maturation_detach_resist"]
        
        # Energy-based detachment (starvation)
        if bacterium.energy < self.params["detach_energy_threshold"]:
            if np.random.random() > detach_resistance:
                self._detach_cell(bacterium)
                return True
        
        # Nutrient-based detachment (seek food)
        if local_food < self.params["detach_food_threshold"]:
            if np.random.random() > detach_resistance * 1.5:  # Slightly harder with food
                self._detach_cell(bacterium)
                return True
        
        # Stress-based detachment (flee antibiotics)
        from simulation.simulation_config import ANTIBIOTIC_TYPES
        total_stress = sum(
            conc * ANTIBIOTIC_TYPES[ab_type]["toxicity_constant"]
            for ab_type, conc in local_antibiotics.items()
            if conc > 0
        )
        if total_stress > self.params["detach_stress_threshold"]:
            if np.random.random() > detach_resistance:
                self._detach_cell(bacterium)
                return True
        
        # Random detachment (dispersal)
        if np.random.random() < self.params["detach_base_prob"] * (1 - detach_resistance):
            self._detach_cell(bacterium)
            return True
        
        # Shear-based detachment (environmental forces)
        if np.random.random() < self.params["detach_shear_prob"] * (1 - detach_resistance):
            self._detach_cell(bacterium)
            return True
        
        return False
    
    def _detach_cell(self, bacterium):
        """Detach a cell from biofilm, returning it to planktonic state.
        
        Args:
            bacterium: The bacterium to detach
        """
        bacterium.biofilm_state = BiofilmState.PLANKTONIC
        bacterium.adhesion_timer = 0
        bacterium.maturation_timer = 0
    
    def get_protection_factor(self, bacterium) -> float:
        """Calculate antibiotic protection factor based on biofilm state.
        
        Args:
            bacterium: The bacterium to calculate protection for
            
        Returns:
            Protection multiplier (1.0 = no protection, higher = more protection)
            
        Biological meaning:
            EPS matrix and cell aggregation provide physical barriers to
            antibiotics, reducing effective concentration. Protection increases
            with biofilm maturity.
        """
        if bacterium.is_persister or not hasattr(bacterium, 'biofilm_state'):
            return 1.0
        
        state = bacterium.biofilm_state
        
        if state == BiofilmState.PLANKTONIC:
            return 1.0
        elif state == BiofilmState.REVERSIBLE_ATTACH:
            return self.params["reversible_protection"]
        elif state == BiofilmState.IRREVERSIBLE_ATTACH:
            return self.params["irreversible_protection"]
        elif state == BiofilmState.MATURE:
            return self.params["mature_protection"]
        
        return 1.0
    
    def get_growth_penalty(self, bacterium) -> float:
        """Calculate growth rate penalty based on biofilm state.
        
        Args:
            bacterium: The bacterium to calculate penalty for
            
        Returns:
            Growth rate multiplier (1.0 = no penalty, lower = slower growth)
            
        Biological meaning:
            Biofilm cells grow slower due to:
            - Reduced nutrient access (matrix diffusion barrier)
            - Energy costs of EPS production
            - Physical constraints in dense aggregates
        """
        if bacterium.is_persister or not hasattr(bacterium, 'biofilm_state'):
            return 1.0
        
        state = bacterium.biofilm_state
        
        if state == BiofilmState.PLANKTONIC:
            return 1.0
        elif state == BiofilmState.REVERSIBLE_ATTACH:
            return self.params["reversible_growth_penalty"]
        elif state == BiofilmState.IRREVERSIBLE_ATTACH:
            return self.params["irreversible_growth_penalty"]
        elif state == BiofilmState.MATURE:
            return self.params["mature_growth_penalty"]
        
        return 1.0
    
    def get_speed_multiplier(self, bacterium) -> float:
        """Calculate movement speed multiplier based on biofilm state.
        
        Args:
            bacterium: The bacterium to calculate speed for
            
        Returns:
            Speed multiplier (1.0 = normal, 0.0 = immobile)
            
        Biological meaning:<f
            Attached cells have reduced motility due to adhesion and
            matrix constraints. Mature cells are effectively immobile.
        """
        if bacterium.is_persister or not hasattr(bacterium, 'biofilm_state'):
            return 1.0
        
        state = bacterium.biofilm_state
        
        if state == BiofilmState.PLANKTONIC:
            return 1.0
        elif state == BiofilmState.REVERSIBLE_ATTACH:
            return self.params["reversible_speed_mult"]
        elif state == BiofilmState.IRREVERSIBLE_ATTACH:
            return self.params["irreversible_speed_mult"]
        elif state == BiofilmState.MATURE:
            return self.params["mature_speed_mult"]
        
        return 1.0
    
    def get_energy_cost(self, bacterium) -> float:
        """Calculate energy cost per step based on biofilm state.
        
        Args:
            bacterium: The bacterium to calculate cost for
            
        Returns:
            Energy cost per timestep
            
        Biological meaning:
            Biofilm formation and maintenance require ATP:
            - Adhesin production for attachment
            - EPS synthesis and secretion
            - Maintenance of matrix structure
        """
        if bacterium.is_persister or not hasattr(bacterium, 'biofilm_state'):
            return 0.0
        
        state = bacterium.biofilm_state
        
        if state == BiofilmState.PLANKTONIC:
            return 0.0
        elif state == BiofilmState.REVERSIBLE_ATTACH:
            return self.params["attachment_cost"]
        elif state == BiofilmState.IRREVERSIBLE_ATTACH:
            return self.params["attachment_cost"] + self.params["eps_production_cost"]
        elif state == BiofilmState.MATURE:
            return self.params["maturation_cost"]
        
        return 0.0
    
    def update_biofilm_cells(self):
        """Update all biofilm-related timers and state transitions.
        
        Called once per simulation step to:
        - Increment adhesion and maturation timers
        - Check for state transitions (reversible → irreversible → mature)
        - Apply energy costs
        
        Computational complexity: O(N) where N = number of bacteria
        """
        for bacterium in self.model.agent_set:
            # Skip if no biofilm state attribute
            if not hasattr(bacterium, 'biofilm_state'):
                continue
            
            # Persisters skip all biofilm logic
            if bacterium.is_persister:
                continue
            
            # Increment timers for attached cells
            if bacterium.biofilm_state != BiofilmState.PLANKTONIC:
                bacterium.adhesion_timer += 1
            
            if bacterium.biofilm_state in [BiofilmState.IRREVERSIBLE_ATTACH, BiofilmState.MATURE]:
                bacterium.maturation_timer += 1
            
            # Apply energy costs
            energy_cost = self.get_energy_cost(bacterium)
            bacterium.energy = max(0.0, bacterium.energy - energy_cost)
