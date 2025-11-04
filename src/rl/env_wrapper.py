from typing import Any, Callable, Dict, Tuple, Optional, Union, List
import numpy as np
import torch
from simulation.simulation_config import ANTIBIOTIC_TYPES, antibiotic_resistances, TOX_TIMES_DOSE_MAX, N_TRAITS, N_BACTERIA_TYPES
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
        count_cost: float = 0.0,        # cost for COUNT action
        # shaping & norms
        target_population: int = 500,   # P*
        w_pop: float = 1.0,             # weight for population term in dose reward
        w_genome: float = 0.5,          # weight for resistance term in dose reward
        w_cost: float = 0.05,           # weight for monetary penalty in dose reward
        w_population_maintenance: float = 0.01,  # per-step penalty for being far from target
        budget_init: float = 100.0,
        budget_norm: float = 100.0,     # divisor for budget normalization
        population_norm: float = 1000.0, # to map counts to ~[0,1]
        budget_penalty: float = 10.0,   # big penalty when budget reaches 0
        # NOOP action reward shaping
        noop_band_factor: float = 0.02,      # deadband around target as fraction of population_norm
        noop_reward_magnitude: float = 0.01, # small shaping magnitude for NOOP action
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
        self.count_cost = count_cost
        self.budget_penalty = budget_penalty

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

        # Budget tracking per episode
        self.episode_start_budget = budget_init
        self.episode_budget_spent = 0.0

        # observation cache (what the agent "knows")
        self.last_count_obs: Optional[int] = None
        self.last_seq_obs: Optional[Dict[str, Any]] = None
        self.ts_last_seq: Optional[int] = None
        self.ts_last_count: Optional[int] = None

        # sequencing pipeline
        self.seq_pending = False
        self.seq_eta = 0  # steps until result is ready

        # pending dose ledger (evaluated when a measurement lands)
        self.pending_dose_events: List[Dict[str, Any]] = []

        # NOOP action reward shaping
        noop_band = noop_band_factor * population_norm
        self.noop_band = noop_band
        self.noop_mag = noop_reward_magnitude
        
        # ========== Reward Modules ==========
        # Initialize reward computation modules
        # Removed DoseRewardCompound - using simple cost penalty instead
        # Natural population changes captured by maintenance reward provide delayed feedback
        
        self.pop_maintenance_reward = PopulationMaintenanceReward(
            target_population=target_population,
            population_norm=population_norm,
            asymmetry_factor=3.0,
            weight=w_population_maintenance,
        )

    # -------------------------
    # Public API
    # -------------------------
    
    def reset(self)-> np.ndarray:
        self.model = self.mesa_model_factory()
        self.t = 0
        self.episode_return = 0.0
        self.budget = self.budget_init
        
        # Reset budget tracking for new episode
        self.episode_start_budget = self.budget_init
        self.episode_budget_spent = 0.0

        # clear caches
        self.last_count_obs = None
        self.last_seq_obs = None
        self.ts_last_count = None
        self.ts_last_seq = None

        # clear pipelines
        self.seq_pending = False
        self.seq_eta = 0
        self.pending_dose_events.clear()


        return self._build_observation()

    def step(self, a_discrete: int, a_cont: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert 0 <= a_discrete <= 3, f"a_discrete out of range: {a_discrete}"
        assert isinstance(a_cont, np.ndarray) and a_cont.shape == (self.k_doses,), \
            f"a_cont must be np.ndarray shape ({self.k_doses},)"

        # 1) Execute action: computes immediate reward (only instant penalties/shaping)
        immediate_reward = self._execute_action(a_discrete, a_cont)

        # 2) Advance simulation one base step
        self.model.step()
        self.t += 1

        # 3) Progress sequencing countdown; when it finishes, publish result
        sequencing_result_landed = False
        if self.seq_pending:
            self.seq_eta -= 1
            if self.seq_eta <= 0:
                # Sequencing result lands NOW
                seq_result = self._read_true_sequencing()
                self._cache_sequencing_obs(seq_result)
                self.seq_pending = False
                self.seq_eta = 0
                sequencing_result_landed = True

        # COUNT has duration 0 → if the agent performed COUNT this step, cache count obs immediately
        count_result_landed = False
        if a_discrete == ACTION_COUNT_BACTERIA or a_discrete == ACTION_DOSE:
            # We just performed a count → cache count obs immediately
            count_now = self._read_true_population()
            self._cache_count_obs(count_now)
            count_result_landed = True

        # 4) Build obs from what the agent knows (cached), never from the hidden true state directly
        # 4) Build obs from what the agent knows (cached), never from the hidden true state directly
        obs = self._build_observation()

        # 5) Termination conditions
        true_population = self._read_true_population()
        done = (true_population == 0) or (self.t >= self.max_steps) or (self.budget < 0.0)

        # 6) Release any pending dose rewards when a measurement lands
        delayed_reward = 0.0
        if count_result_landed:
            delayed_reward += self._collect_pending_dose_rewards(self.last_count_obs)
        elif sequencing_result_landed:
            delayed_reward += self._collect_pending_dose_rewards(None)

        # 7) Compute total reward: immediate penalties + delayed efficacy + maintenance
        # Use PopulationMaintenanceReward module for consistent asymmetric penalty
        maintenance_penalty = 0.0
        if a_discrete in (ACTION_COUNT_BACTERIA, ACTION_DOSE):
            maintenance_penalty = self.pop_maintenance_reward(true_population)
        
        # 6b) Add big penalty if budget reaches 0
        budget_penalty = 0.0
        if self.budget <= 0.0:
            budget_penalty = -self.budget_penalty
        
        reward = immediate_reward + maintenance_penalty + budget_penalty + delayed_reward
        self.episode_return += reward

        info = {
            "t": self.t,
            "true_population": true_population,  # diagnostic only (agent doesn't see it)
            "episode_return": self.episode_return,
            "budget": self.budget,
            "seq_pending": self.seq_pending,
            "seq_eta": self.seq_eta,
            "delayed_reward": delayed_reward,
        }
        return obs, float(reward), bool(done), info

    # -------------------------
    # Action execution
    # -------------------------

    def _execute_action(self, a_discrete: int, a_cont: np.ndarray) -> float:
        """
        Applies the chosen action. Returns *immediate* reward as float.
        
        Simplified approach: Let natural consequences (population changes) 
        drive learning through maintenance reward, rather than predicting
        efficacy at dose time.
        
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
            # Apply count cost from action config
            self.budget -= self.count_cost
            self.episode_budget_spent += self.count_cost
            return -self.count_cost  # Penalize the COUNT action with its cost

        if a_discrete == ACTION_SEQUENCING:
            # Cost now, reward 0 now; result later
            self.budget -= self.sequencing_cost
            self.episode_budget_spent += self.sequencing_cost
            if not self.seq_pending:
                self.seq_pending = True
                self.seq_eta = int(self.sequencing_duration)
            else:
                return -0.001  # Small penalty for redundant sequencing
            return 0.0

        if a_discrete == ACTION_DOSE:
            # Cost now; efficacy reward computed later when a measurement lands
            scaled = self.scale_dose(np.clip(a_cont, 0.0, 1.0))
            self._apply_antibiotics(scaled)

            cost = float(np.sum(scaled) * self.dose_cost_per_unit)
            self.budget -= cost
            self.episode_budget_spent += cost

            # ✅ SIMPLIFIED: Just return cost penalty
            # Let population maintenance reward (computed every step) capture efficacy
            # PPO's TD learning will connect: dose → future population drops → better rewards
            return -cost * self.w_cost
        
        raise ValueError(f"Unknown discrete action: {a_discrete}")

    def _register_pending_dose(self, dose_vector: np.ndarray) -> None:
        """Record dose metadata so efficacy can be scored when feedback arrives."""
        age_pop = None if self.ts_last_count is None else (self.t - self.ts_last_count)
        age_genome = None if self.ts_last_seq is None else (self.t - self.ts_last_seq)

        avg_genome = None
        if self.last_seq_obs is not None:
            avg_genome = np.copy(self.last_seq_obs["avg_genome"])

        event: Dict[str, Any] = {
            "doses": np.copy(dose_vector),
            "pre_count": None if self.last_count_obs is None else int(self.last_count_obs),
            "age_pop": age_pop,
            "avg_genome": avg_genome,
            "age_genome": age_genome,
        }
        self.pending_dose_events.append(event)
    
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

    def _collect_pending_dose_rewards(self, post_count: Optional[int]) -> float:
        if not self.pending_dose_events:
            return 0.0

        total = 0.0
        for event in self.pending_dose_events:
            total += self._evaluate_dose_event(event, post_count)
        self.pending_dose_events.clear()
        return float(total)

    def _evaluate_dose_event(self, event: Dict[str, Any], post_count: Optional[int]) -> float:
        doses = torch.tensor(event["doses"], dtype=self.dtype, device=self.device)

        genome_term = self.dose_reward_compound.genome_reward(
            event.get("avg_genome"),
            doses,
            0 if event.get("age_genome") is None else event["age_genome"],
        )

        pre_count = event.get("pre_count")
        if post_count is None:
            pop_term_raw = -0.5
        elif pre_count is None:
            gap = abs(float(post_count) - self.target_population)
            pop_term_raw = -gap / max(1.0, self.population_norm)
        else:
            pre_gap = abs(float(pre_count) - self.target_population)
            post_gap = abs(float(post_count) - self.target_population)
            improvement = pre_gap - post_gap
            pop_term_raw = improvement / max(1.0, self.population_norm)

        pop_term_tensor = torch.tensor(pop_term_raw, dtype=self.dtype, device=self.device)
        age_pop = event.get("age_pop", 0) or 0
        pop_term_tensor = self.dose_reward_compound.pop_reward.age_normalizer(pop_term_tensor, age_pop)
        pop_term_tensor = torch.clamp(pop_term_tensor, min=-1.0, max=1.0)
        pop_term = float(pop_term_tensor.item())

        total = (
            self.dose_reward_compound.w_pop * pop_term
            + self.dose_reward_compound.w_genome * genome_term
        )

        return float(total)

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
    
    def get_episode_budget_metrics(self) -> Dict[str, float]:
        """
        Get budget metrics for the current episode.
        
        Returns:
            Dict with:
                - start_budget: Budget at episode start
                - current_budget: Current remaining budget
                - budget_spent: Total budget spent this episode
                - budget_per_step: Average budget spent per step
        """
        budget_per_step = self.episode_budget_spent / max(1, self.t)
        return {
            "start_budget": float(self.episode_start_budget),
            "current_budget": float(self.budget),
            "budget_spent": float(self.episode_budget_spent),
            "budget_per_step": float(budget_per_step),
        }