from typing import Any, Callable, Dict, Tuple, Optional, Union
import numpy as np
import torch
from config import ANTIBIOTIC_TYPES, antibiotic_resistances, TOX_TIMES_DOSE_MAX, N_TRAITS, N_BACTERIA_TYPES
from rl.reward import (
    PopulationReward,
    GenomeReward,
    CostReward,
    DoseRewardCompound,
    PopulationMaintenanceReward,
)

# Discrete actions
ACTION_NOOP = 0
ACTION_COUNT_BACTERIA = 1
ACTION_SEQUENCING = 2
ACTION_DOSE = 3

class PetriEnvWrapper:
    """
    Thin wrapper around a Mesa bacteria simulation for RL:
      - Partial observability: agent only "knows" what it measures.
      - Action durations: sequencing has latency; count is instant.
      - Delayed rewards: dose efficacy is evaluated when a measurement lands.
    
    Observation vector (gated, not the true state):
      [ budget_norm,
        last_count_norm,                          # -1 if never observed
        last_seq_avg_enzyme, last_seq_avg_efflux,
        last_seq_avg_repair, last_seq_avg_membrane,  # 0 if never observed
        last_seq_prop_0 .. last_seq_prop_{K-1},      # 0 if never observed
        time_since_last_measure_norm,
        is_seq_pending (0/1),
        steps_until_seq_result_norm
      ]
    Length = 1 (budget) + 1 (count) + 4 (genome avgs) + K (proportions) + 3 meta = 9 + K
    """

    def __init__(
        self,
        mesa_model_factory: Callable[[], Any],
        k_doses: int,
        scale_dose: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        max_steps: int = 1000,
        # costs & durations
        sequencing_cost: float = 1.0,
        sequencing_duration: int = 5,   # steps to finish
        dose_cost_per_unit: float = 0.2,
        # shaping & norms
        target_population: int = 500,   # P*
        w_pop: float = 1.0,             # weight for population term in dose reward
        w_genome: float = 0.5,          # weight for resistance term in dose reward
        w_cost: float = 0.05,           # weight for monetary penalty in dose reward
        w_population_maintenance: float = 0.01,  # per-step penalty for being far from target
        budget_init: float = 100.0,
        budget_norm: float = 100.0,     # divisor for budget normalization
        population_norm: float = 1000.0, # to map counts to ~[0,1]
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.mesa_model_factory = mesa_model_factory
        self.k_doses = k_doses
        self.scale_dose = scale_dose if scale_dose is not None else (lambda x: x)
        self.max_steps = max_steps

        # economics & timing
        self.sequencing_cost = sequencing_cost
        self.sequencing_duration = sequencing_duration
        self.dose_cost_per_unit = dose_cost_per_unit

        # reward shaping
        self.target_population = target_population
        self.w_pop = w_pop
        self.w_genome = w_genome
        self.w_cost = w_cost
        self.w_population_maintenance = w_population_maintenance

        # normalization/display
        self.budget_init = budget_init
        self.budget_norm = budget_norm
        self.population_norm = population_norm
        
        # device & dtype for reward modules
        self.device = device
        self.dtype = dtype

        # runtime state
        self.model: Any = None
        self.t = 0
        self.episode_return = 0.0
        self.budget = budget_init

        # observation cache (what the agent "knows")
        self.last_count_obs: Optional[int] = None
        self.last_seq_obs: Optional[Dict[str, Any]] = None
        self.ts_last_seq: Optional[int] = None
        self.ts_last_count: Optional[int] = None

        # sequencing pipeline
        self.seq_pending = False
        self.seq_eta = 0  # steps until result is ready

        # pending dose ledger (evaluated when a measurement lands)

        noop_band = 0.02 * population_norm   # ~2% deadband around target
        noop_mag  = 0.01                     # small shaping magnitude
        self.noop_band = noop_band
        self.noop_mag  = noop_mag
        
        # ========== Reward Modules ==========
        # Initialize reward computation modules
        self.dose_reward_compound = DoseRewardCompound(
            target_population=target_population,
            population_norm=population_norm,
            dose_cost_per_unit=dose_cost_per_unit,
            w_pop=w_pop,
            w_genome=w_genome,
            w_cost=w_cost,
            device=device,
            dtype=dtype,
            aging_type="sqrt",
        )
        
        self.pop_maintenance_reward = PopulationMaintenanceReward(
            target_population=target_population,
            population_norm=population_norm,
            asymmetry_factor=3.0,
            weight=w_population_maintenance,
        )

    # -------------------------
    # Public API
    # -------------------------

    def reset(self) -> np.ndarray:
        self.model = self.mesa_model_factory()
        self.t = 0
        self.episode_return = 0.0
        self.budget = self.budget_init

        # clear caches
        self.last_count_obs = None
        self.last_seq_obs = None
        self.ts_last_count = None
        self.ts_last_seq = None

        # clear pipelines
        self.seq_pending = False
        self.seq_eta = 0


        return self._build_observation()

    def step(self, a_discrete: int, a_cont: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert 0 <= a_discrete <= 3, f"a_discrete out of range: {a_discrete}"
        assert isinstance(a_cont, np.ndarray) and a_cont.shape == (self.k_doses,), \
            f"a_cont must be np.ndarray shape ({self.k_doses},)"

        # 1) Execute action: computes immediate reward (dose reward computed here now)
        immediate_reward = self._execute_action(a_discrete, a_cont)

        # 2) Advance simulation one base step
        self.model.step()
        self.t += 1

        # 3) Progress sequencing countdown; when it finishes, publish result
        # Note: We no longer use delayed dose rewards here
        if self.seq_pending:
            self.seq_eta -= 1
            if self.seq_eta <= 0:
                # Sequencing result lands NOW
                seq_result = self._read_true_sequencing()
                self._cache_sequencing_obs(seq_result)
                self.seq_pending = False
                self.seq_eta = 0

        # COUNT has duration 0 → if the agent performed COUNT this step, cache count obs immediately
        if a_discrete == ACTION_COUNT_BACTERIA or a_discrete == ACTION_DOSE:
            # We just performed a count → cache count obs immediately
            count_now = self._read_true_population()
            self._cache_count_obs(count_now)

        # 4) Build obs from what the agent knows (cached), never from the hidden true state directly
        # 4) Build obs from what the agent knows (cached), never from the hidden true state directly
        obs = self._build_observation()

        # 5) Termination conditions
        true_population = self._read_true_population()
        done = (true_population == 0) or (self.t >= self.max_steps) or (self.budget < 0.0)

        # 6) Compute total reward: immediate action reward + step-wise population maintenance penalty
        # Use PopulationMaintenanceReward module for consistent asymmetric penalty
        maintenance_penalty = self.pop_maintenance_reward(true_population)
        
        reward = immediate_reward + maintenance_penalty
        self.episode_return += reward

        info = {
            "t": self.t,
            "true_population": true_population,  # diagnostic only (agent doesn't see it)
            "episode_return": self.episode_return,
            "budget": self.budget,
            "seq_pending": self.seq_pending,
            "seq_eta": self.seq_eta,
        }
        return obs, float(reward), bool(done), info

    # -------------------------
    # Action execution
    # -------------------------

    def _execute_action(self, a_discrete: int, a_cont: np.ndarray) -> float:
        """
        Applies the chosen action. Returns *immediate* reward as float.
        
        For ACTION_DOSE: reward is computed immediately using cached observations,
        aligning credit assignment with the timestep where the decision was made.
        This improves PPO training stability vs. delayed rewards.
        
        Returns:
            Immediate reward (float)
        """
        if a_discrete == ACTION_NOOP:
            bonus = 0.0
            if self.last_count_obs is not None:
                diff = self.last_count_obs - self.target_population
                if abs(diff) <= self.noop_band:
                    bonus = 0.0  # in band: neutral
                elif diff < 0:
                    bonus = +self.noop_mag   # below target → doing nothing is good
                else:
                    bonus = -self.noop_mag   # above target → doing nothing is bad
            # if we have no measurement yet, stay neutral
            return bonus

        if a_discrete == ACTION_COUNT_BACTERIA:
            # As per spec: 0 cost, duration 0, reward 0 now
            return 0.0

        if a_discrete == ACTION_SEQUENCING:
            # Cost now, reward 0 now; result later
            self.budget -= self.sequencing_cost
            if not self.seq_pending:
                self.seq_pending = True
                self.seq_eta = int(self.sequencing_duration)
            else:
                # If sequencing is already pending, we just paid again
                return -0.001
            return 0.0

        if a_discrete == ACTION_DOSE:
            # Cost now; efficacy reward computed immediately using cached obs
            scaled = self.scale_dose(np.clip(a_cont, 0.0, 1.0))
            self._apply_antibiotics(scaled)

            cost = float(np.sum(scaled) * self.dose_cost_per_unit)
            self.budget -= cost

            # ✅ Compute efficacy reward IMMEDIATELY using cached observations
            # This aligns credit with the timestep where the dose decision was made
            efficacy_reward = self._compute_dose_reward_immediate(scaled)

            # Store for diagnosis/debugging (but don't wait for delayed settling)

            return float(efficacy_reward)
        
        raise ValueError(f"Unknown discrete action: {a_discrete}")

    def _compute_dose_reward_immediate(self, dose_vector: np.ndarray) -> float:
        """
        Compute dose efficacy reward using CURRENT cached observations.
        
        This reward is assigned immediately to the timestep where the dose was administered.
        Uses the DoseRewardCompound module to orchestrate all reward components.
        
        Reward the agent for *approaching* target, not for massive kill-offs.
        
        This design encourages the agent to:
          1. Dose when it has fresh population & genome information
          2. Avoid dosing on stale/blind data (via staleness penalty)
          3. Trade off measurement costs against effectiveness
        
        Args:
            dose_vector: scaled dose amounts [0, 1] for each antibiotic
            
        Returns:
            Immediate reward signal (float)
        """
        # Compute age of measurements
        age_pop = 0 if self.ts_last_count is None else (self.t - self.ts_last_count)
        age_genome = 0 if self.ts_last_seq is None else (self.t - self.ts_last_seq)
        
        # Prepare genome tensor for reward computation
        avg_genome = None
        if self.last_seq_obs is not None:
            avg_genome = torch.tensor(
                self.last_seq_obs["avg_genome"],
                dtype=self.dtype,
                device=self.device,
            )
        
        # Prepare dose tensor
        doses = torch.tensor(dose_vector, dtype=self.dtype, device=self.device)
        
        # Use compound reward module to compute total reward
        reward = self.dose_reward_compound(
            last_count_obs=self.last_count_obs,
            age_pop=age_pop,
            avg_genome=avg_genome,
            doses=doses,
            age_genome=age_genome,
        )
        
        return float(reward)
    
    # -------------------------
    # Reading true state (hidden from agent)
    # -------------------------

    def _read_true_population(self) -> int:
        # Assumes the Mesa model exposes agent set
        return int(len(self.model.agent_set))

    def _read_true_sequencing(self) -> Dict[str, Any]:
        """
        Build a sequencing summary from the true state (hidden),
        shaped like:
          { "avg_genome": np.array([enzyme, efflux, repair, membrane], dtype=float32),
            "proportions": np.array([p_0..p_{K-1}], dtype=float32) }
        """
        # --- Average genome traits ---
        genome_matrix = self.model.get_population_stats()["traits_matrix"]

        # --- Type proportions ---
        # You can compute these from your model taxonomy; here we default to zeros
        proportions = np.zeros((self.k_doses,), dtype=np.float32)

        return {
            "avg_genome": np.array(genome_matrix, dtype=np.float32),
            "proportions": proportions,
        }

    def _apply_antibiotics(self, scaled_doses: np.ndarray) -> None:
        """
        Applies antibiotics into the Mesa model. Expects the model to have:
          - model.antibiotic_fields: Dict[name -> field]
          - model.apply_antibiotic(name, amount)
        """
        # print(f"APPLY ANTIBIOTICS: ", scaled_doses)
        ab_names = list(self.model.antibiotic_fields.keys())
        K = min(self.k_doses, len(ab_names))
        for i in range(K):
            amt = float(scaled_doses[i])
            self.model.apply_antibiotic(ab_names[i], amt)

    # -------------------------
    # Observation management (gated)
    # -------------------------

    def _cache_count_obs(self, population: int) -> None:
        self.last_count_obs = int(population)
        self.ts_last_count = self.t

    def _cache_sequencing_obs(self, seq: Dict[str, Any]) -> None:
        self.last_seq_obs = {
            "avg_genome": seq["avg_genome"].astype(np.float32),
            "proportions": seq["proportions"].astype(np.float32),
            "t": self.t,
        }
        self.ts_last_seq = self.t

    def _build_observation(self) -> np.ndarray:
        """
        Assemble what the agent is allowed to see (cached measurements + meta).
        """
        budget_norm = np.clip(self.budget / max(1.0, self.budget_norm), -1.0, 1.0)

        # Count
        if self.last_count_obs is None:
            last_count_norm = -1.0  # sentinel for unknown
        else:
            last_count_norm = float(self.last_count_obs) / max(1.0, self.population_norm)

        # Sequencing
        if self.last_seq_obs is None:
            avg_genome = np.zeros((N_BACTERIA_TYPES, N_TRAITS), dtype=np.float32)
            props = np.zeros((self.k_doses,), dtype=np.float32)
        else:
            avg_genome = self.last_seq_obs["avg_genome"]
            props = self.last_seq_obs["proportions"]

        # Meta
        if self.ts_last_seq is None:
            ts_since_seq = 0
        else:
            ts_since_seq = self.t - self.ts_last_seq
        if self.ts_last_count is None:
            ts_since_count = 0
        else:
            ts_since_count = self.t - self.ts_last_count

        ts_since_measure = min(ts_since_seq, ts_since_count)

        ts_since =  self.t - ts_since_measure
        time_since_last_measure_norm = min(1.0, ts_since / 100.0)
        seq_pending_flag = 1.0 if self.seq_pending else 0.0
        seq_eta_norm = min(1.0, max(0, self.seq_eta) / max(1, self.sequencing_duration))


        avg_genome = avg_genome.flatten()


        obs_parts = [
            budget_norm,
            last_count_norm,
            *avg_genome.tolist(),
            *props.tolist(),
            time_since_last_measure_norm,
            seq_pending_flag,
            seq_eta_norm,
        ]
        return np.asarray(obs_parts, dtype=np.float32)

    # -------------------------
    # Convenience
    # -------------------------

    def get_obs_dim(self) -> int:
        # Build once without stepping sim
        dummy = self._build_observation()
        return int(dummy.shape[0])
    
    def get_bacteria_population(self) -> int:
        """Get the current bacteria population count from the model."""
        if self.model is None:
            return 0
        return self._read_true_population()