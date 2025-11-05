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
    SurvivalBonusReward,
    BudgetConservationReward,
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
    
    REWARD COMPONENTS BREAKDOWN:
    ============================
    
    Total reward = immediate + maintenance + budget_penalty + delayed + survival_bonus + budget_conservation
    
    1. IMMEDIATE REWARD (returned from _execute_action):
       Composed of:
       a) Action cost penalty: -cost * w_cost (for COUNT, SEQUENCING, DOSE actions)
       b) NOOP shaping bonus/penalty: rewards staying in deadband around target population
       c) Regular count bonus: rewards counting at regular intervals (every ~15 steps)
       d) Safe behavior bonus: rewards NOT dosing when population is below target
       e) Informed dosing bonus/penalty:
          - Positive bonus: dosing with recent count AND sequencing data
          - Negative penalty: dosing without recent data (blind dosing)
          - Large negative penalty: dosing when population is already below target
    
    2. MAINTENANCE REWARD:
       Per-step reward/penalty based on distance from target population.
       Computed every step using PopulationMaintenanceReward module.
       Encourages keeping bacteria count near target_population.
    
    3. BUDGET PENALTY:
       Large negative penalty when budget reaches 0.
       Prevents running out of resources.
    
    4. DELAYED REWARD:
       Evaluates efficacy of past DOSE actions when new measurements land.
       Only computed when COUNT or SEQUENCING results become available.
    
    5. SURVIVAL BONUS:
       Per-step bonus for staying alive (configurable: constant/linear/exponential).
       Encourages longer episodes and survival.
    
    6. BUDGET CONSERVATION:
       Optional reward for efficient budget usage (typically disabled).
       Encourages saving resources when possible.
    
    All components are tracked separately in the info dict for analysis and TensorBoard logging.
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
        dose_cost: float = 2.0,         # fixed cost per dose action
        dose_cost_per_unit: float = 0.2,  # variable cost per unit of antibiotic
        count_cost: float = 0.0,        # cost for COUNT action
        # informed dosing params
        informed_dosing_reward: float = 0.0,    # bonus for dosing after recent count AND sequencing
        informed_dosing_above_target_reward: float = 0.0,  # additional bonus for informed dosing when pop above target
        informed_dosing_window: int = 10,       # steps window for "recent" count
        informed_sequencing_window: int = 50,   # steps window for "recent" sequencing
        blind_dosing_penalty: float = 0.0,      # penalty for dosing without count/sequencing
        dosing_low_population_penalty: float = 0.0,  # BIG penalty for dosing when pop below target
        # regular monitoring rewards
        regular_count_reward: float = 0.0,      # reward for counting regularly
        regular_count_interval: int = 15,       # target interval for regular counting
        regular_count_min_interval: int = 3,    # minimum interval to avoid spam-counting
        safe_nondosing_reward: float = 0.0,     # reward for NOT dosing when pop is low
        count_population_reward: float = 0.0,   # reward based on distance from target after COUNT
        # critical inaction penalties
        critical_high_population_threshold: float = 3.0,  # multiplier of target for critical level
        critical_no_action_penalty: float = 0.0,  # penalty for not seq/dosing when count shows critical pop
        critical_no_dose_penalty: float = 0.0,    # penalty for not dosing when count+seq fresh and critical
        critical_freshness_window: int = 5,       # steps to consider data "fresh"
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
        self.dose_cost = dose_cost
        self.dose_cost_per_unit = dose_cost_per_unit
        self.count_cost = count_cost
        self.budget_penalty = budget_penalty
        
        # informed dosing parameters
        self.informed_dosing_reward = informed_dosing_reward
        self.informed_dosing_above_target_reward = informed_dosing_above_target_reward
        self.informed_dosing_window = informed_dosing_window
        self.informed_sequencing_window = informed_sequencing_window
        self.blind_dosing_penalty = blind_dosing_penalty
        self.dosing_low_population_penalty = dosing_low_population_penalty
        
        # regular monitoring parameters
        self.regular_count_reward = regular_count_reward
        self.regular_count_interval = regular_count_interval
        self.regular_count_min_interval = regular_count_min_interval
        self.safe_nondosing_reward = safe_nondosing_reward
        self.count_population_reward = count_population_reward
        
        # critical inaction penalties
        self.critical_high_population_threshold = critical_high_population_threshold
        self.critical_no_action_penalty = critical_no_action_penalty
        self.critical_no_dose_penalty = critical_no_dose_penalty
        self.critical_freshness_window = critical_freshness_window

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
        
        # Survival bonus reward module (encourage longer episodes)
        self.survival_bonus_reward = None  # Will be set if enabled
        
        # Budget conservation reward module (encourage efficient spending)
        self.budget_conservation_reward = None  # Will be set if enabled
        
        # Track last step's budget for computing spending rate
        self.last_step_budget = budget_init

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
        
        # Reset reward component tracking
        self.last_regular_count_bonus = 0.0
        self.last_safe_behavior_bonus = 0.0
        self.last_informed_dosing_bonus = 0.0
        self.last_count_population_reward = 0.0

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

        # Reset reward component tracking (will be set if applicable)
        self.last_safe_behavior_bonus = 0.0
        self.last_informed_dosing_bonus = 0.0
        
        # Check if action is affordable BEFORE executing
        # If not affordable, silently convert to NOOP
        # Also calculate cost to avoid recalculating in _execute_action
        original_action = a_discrete
        a_discrete, action_cost = self._check_action_affordability(a_discrete, a_cont)

        # 1) Execute action: computes immediate reward (only instant penalties/shaping)
        # This now includes regular_count_reward for COUNT actions and informed_dosing_bonus for DOSE actions
        immediate_reward = self._execute_action(a_discrete, a_cont, action_cost)

        # 1b) Safe non-dosing reward: reward for NOT dosing when population is below target
        safe_behavior_bonus = 0.0
        if a_discrete != ACTION_DOSE:
            # Check if we have recent count data
            steps_since_count = self.t - self.ts_last_count if self.ts_last_count is not None else float('inf')
            has_recent_count = steps_since_count <= self.informed_dosing_window
            
            # If count is recent and population is below target, reward for NOT dosing
            if has_recent_count and self.last_count_obs is not None:
                if self.last_count_obs < self.target_population:
                    safe_behavior_bonus = self.safe_nondosing_reward
                    # Debug output (only first few episodes)
                    # if self.t < 100:
                    #     print(f"[SAFE BEHAVIOR BONUS] t={self.t}, action={a_discrete}, pop={self.last_count_obs}, target={self.target_population}, bonus={safe_behavior_bonus}")
        
        # Store for tracking
        self.last_safe_behavior_bonus = safe_behavior_bonus
        immediate_reward += safe_behavior_bonus
        
        # 1c) Critical inaction penalties: penalize inaction when population is dangerously high
        critical_inaction_penalty = 0.0
        
        # Check if we have fresh count data showing critical population
        steps_since_count = self.t - self.ts_last_count if self.ts_last_count is not None else float('inf')
        has_fresh_count = steps_since_count <= self.critical_freshness_window
        critical_threshold = self.target_population * self.critical_high_population_threshold
        
        if has_fresh_count and self.last_count_obs is not None:
            population_is_critical = self.last_count_obs >= critical_threshold
            
            if population_is_critical:
                # Penalty 1: Not taking SEQUENCING or DOSE when count shows critical population
                if a_discrete not in [ACTION_SEQUENCING, ACTION_DOSE]:
                    critical_inaction_penalty -= self.critical_no_action_penalty
                    # # Debug output
                    # if self.t < 100:
                    #     print(f"[CRITICAL INACTION] t={self.t}, pop={self.last_count_obs}, threshold={critical_threshold}, action={a_discrete}, penalty={critical_inaction_penalty}")
                
                # Penalty 2: Not dosing when BOTH count and sequencing are fresh and population is critical
                steps_since_seq = self.t - self.ts_last_seq if self.ts_last_seq is not None else float('inf')
                has_fresh_seq = steps_since_seq <= self.critical_freshness_window
                
                if has_fresh_seq and a_discrete != ACTION_DOSE:
                    # Both count and sequencing are fresh, population is critical, but agent isn't dosing
                    critical_inaction_penalty -= self.critical_no_dose_penalty
                    # # Debug output
                    # if self.t < 100:
                    #     print(f"[CRITICAL NO DOSE] t={self.t}, pop={self.last_count_obs}, threshold={critical_threshold}, count_age={steps_since_count}, seq_age={steps_since_seq}, action={a_discrete}, penalty={critical_inaction_penalty}")
        
        immediate_reward += critical_inaction_penalty

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
        if a_discrete == ACTION_COUNT_BACTERIA:
            # Count action was executed (we already checked affordability upfront)
            count_now = self._read_true_population()
            self._cache_count_obs(count_now)
            count_result_landed = True

        # 4) Build obs from what the agent knows (cached), never from the hidden true state directly
        # 4) Build obs from what the agent knows (cached), never from the hidden true state directly
        obs = self._build_observation()

        # 5) Termination conditions
        true_population = self._read_true_population()
        # 5) Termination conditions
        true_population = self._read_true_population()
        done = (true_population == 0) or (self.t >= self.max_steps) or (self.budget <= 0.0)

        # 6) Release any pending dose rewards when a measurement lands
        delayed_reward = 0.0
        if count_result_landed:
            delayed_reward += self._collect_pending_dose_rewards(self.last_count_obs)
        elif sequencing_result_landed:
            delayed_reward += self._collect_pending_dose_rewards(None)

        # 7) Compute total reward: immediate penalties + delayed efficacy + maintenance
        # Use PopulationMaintenanceReward module for consistent asymmetric penalty
        # DISABLED: Only apply maintenance reward when w_population_maintenance > 0
        maintenance_penalty = 0.0
        if self.w_population_maintenance > 0.0:
            maintenance_penalty = self.pop_maintenance_reward(true_population)
        
        # 7b) Add survival bonus reward if enabled
        survival_bonus = 0.0
        if self.survival_bonus_reward is not None:
            survival_bonus = self.survival_bonus_reward(self.t)
        
        # 7c) Add budget conservation reward if enabled
        budget_conservation_bonus = 0.0
        if self.budget_conservation_reward is not None:
            budget_spent_this_step = self.last_step_budget - self.budget
            budget_conservation_bonus = self.budget_conservation_reward(
                budget_spent_this_step=budget_spent_this_step,
                current_budget=self.budget,
                initial_budget=self.episode_start_budget,
                timestep=self.t,
            )
            self.last_step_budget = self.budget
        
        # 7d) Add budget penalty if budget reaches 0 (configurable via budget_penalty weight)
        budget_penalty = 0.0
        if self.budget <= 0.0 and self.budget_penalty > 0.0:
            budget_penalty = -self.budget_penalty
        
        reward = (
            immediate_reward + 
            maintenance_penalty + 
            budget_penalty + 
            delayed_reward +
            survival_bonus +
            budget_conservation_bonus
        )
        self.episode_return += reward

        info = {
            "t": self.t,
            "true_population": true_population,  # diagnostic only (agent doesn't see it)
            "episode_return": self.episode_return,
            "budget": self.budget,
            "seq_pending": self.seq_pending,
            "seq_eta": self.seq_eta,
            # Detailed reward breakdown
            "reward_immediate": immediate_reward,
            "reward_maintenance": maintenance_penalty,
            "reward_budget_penalty": budget_penalty,
            "reward_delayed": delayed_reward,
            "reward_survival_bonus": survival_bonus,
            "reward_budget_conservation": budget_conservation_bonus,
            "reward_regular_count_bonus": self.last_regular_count_bonus,
            "reward_safe_behavior_bonus": self.last_safe_behavior_bonus,
            "reward_informed_dosing_bonus": self.last_informed_dosing_bonus,
            "reward_count_population": self.last_count_population_reward,
            "reward_critical_inaction_penalty": critical_inaction_penalty,
            "reward_total": reward,
        }
        return obs, float(reward), bool(done), info

    # -------------------------
    # Action execution
    # -------------------------

    def _check_action_affordability(self, a_discrete: int, a_cont: np.ndarray) -> Tuple[int, float]:
        """
        Check if the agent can afford the requested action.
        If not affordable, return ACTION_NOOP instead.
        
        This ensures that budget constraints are respected without breaking
        the action execution logic - the agent simply performs NOOP when broke.
        
        Args:
            a_discrete: Requested discrete action
            a_cont: Continuous action parameters
            
        Returns:
            Tuple of (action_to_execute, action_cost)
        """
        if a_discrete == ACTION_NOOP:
            return ACTION_NOOP, 0.0
        
        if a_discrete == ACTION_COUNT_BACTERIA:
            if self.budget < self.count_cost:
                return ACTION_NOOP, 0.0
            return ACTION_COUNT_BACTERIA, self.count_cost
        
        if a_discrete == ACTION_SEQUENCING:
            if self.budget < self.sequencing_cost:
                return ACTION_NOOP, 0.0
            return ACTION_SEQUENCING, self.sequencing_cost
        
        if a_discrete == ACTION_DOSE:
            # Calculate total dose cost
            scaled = self.scale_dose(np.clip(a_cont, 0.0, 1.0))
            variable_cost = float(np.sum(scaled) * self.dose_cost_per_unit)
            total_cost = self.dose_cost + variable_cost
            
            if self.budget < total_cost:
                return ACTION_NOOP, 0.0
            return ACTION_DOSE, total_cost
        
        return a_discrete, 0.0

    def _execute_action(self, a_discrete: int, a_cont: np.ndarray, action_cost: float) -> float:
        """
        Applies the chosen action. Returns *immediate* reward as float.
        
        Simplified approach: Let natural consequences (population changes) 
        drive learning through maintenance reward, rather than predicting
        efficacy at dose time.
        
        Returns:
            Immediate reward (float)
        """
        # Reset tracking variables (will be set if applicable)
        self.last_regular_count_bonus = 0.0
        self.last_informed_dosing_bonus = 0.0
        self.last_count_population_reward = 0.0
        
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
            # Apply count cost (pre-calculated and passed in)
            self.budget -= action_cost
            self.episode_budget_spent += action_cost
            
            # Get current population to evaluate distance from target
            current_population = self._read_true_population()
            
            # Count population reward: immediate feedback based on distance from target
            # Positive reward when close to target, negative when far
            count_pop_reward = 0.0
            if self.count_population_reward > 0.0:
                distance = abs(current_population - self.target_population)
                normalized_distance = distance / self.population_norm
                # Use exponential decay: reward decreases as distance increases
                # When distance = 0: reward = count_population_reward
                # When distance = population_norm: reward ≈ -count_population_reward
                count_pop_reward = self.count_population_reward * (1.0 - 2.0 * normalized_distance)
                count_pop_reward = np.clip(count_pop_reward, -self.count_population_reward, self.count_population_reward)
            
            # Store for tracking
            self.last_count_population_reward = count_pop_reward
            
            # Regular monitoring reward: encourage counting at regular intervals
            regular_monitoring_bonus = 0.0
            if self.ts_last_count is not None:
                time_since_last_count = self.t - self.ts_last_count
                # Reward if counting within optimal window (not too soon, not too late)
                min_interval = self.regular_count_min_interval  # Don't spam-count too frequently
                max_interval = self.regular_count_interval  # Don't wait too long
                
                if min_interval <= time_since_last_count <= max_interval:
                    regular_monitoring_bonus = self.regular_count_reward
                    # # Debug output (only first few episodes)
                    # if self.t < 100:
                    #     print(f"[REGULAR COUNT BONUS] t={self.t}, time_since_last={time_since_last_count}, window=[{min_interval}, {max_interval}], bonus={regular_monitoring_bonus}")
                # elif self.t < 100 and time_since_last_count < min_interval:
                #     print(f"[COUNT TOO SOON] t={self.t}, time_since_last={time_since_last_count} < {min_interval}, no bonus")
                # elif self.t < 100 and time_since_last_count > max_interval:
                #     print(f"[COUNT TOO LATE] t={self.t}, time_since_last={time_since_last_count} > {max_interval}, no bonus")
            else:
                # First count in episode - give reward to encourage starting monitoring
                regular_monitoring_bonus = self.regular_count_reward
                count_pop_reward = 0.0  # No population reward on first count
                # Debug output
                # if self.t < 100:
                #     print(f"[FIRST COUNT BONUS] t={self.t}, bonus={regular_monitoring_bonus}")
            
            # Store for tracking
            self.last_regular_count_bonus = regular_monitoring_bonus
            return -action_cost + regular_monitoring_bonus + count_pop_reward

        if a_discrete == ACTION_SEQUENCING:
            # Cost now, reward 0 now; result later (pre-calculated cost)
            self.budget -= action_cost
            self.episode_budget_spent += action_cost
            if not self.seq_pending:
                self.seq_pending = True
                self.seq_eta = int(self.sequencing_duration)
            else:
                return -0.001  # Small penalty for redundant sequencing
            return 0.0

        if a_discrete == ACTION_DOSE:
            # Apply antibiotics (cost already calculated and checked)
            scaled = self.scale_dose(np.clip(a_cont, 0.0, 1.0))
            self._apply_antibiotics(scaled)
            
            # Deduct pre-calculated cost
            self.budget -= action_cost
            self.episode_budget_spent += action_cost

            # Informed dosing reward/penalty system
            dosing_bonus = 0.0
            
            # Check 1: Do we have recent COUNT data?
            has_recent_count = False
            if self.ts_last_count is not None:
                steps_since_count = self.t - self.ts_last_count
                has_recent_count = (steps_since_count <= self.informed_dosing_window)
            
            # Check 2: Do we have recent SEQUENCING data?
            has_recent_sequencing = False
            if self.ts_last_seq is not None:
                steps_since_seq = self.t - self.ts_last_seq
                has_recent_sequencing = (steps_since_seq <= self.informed_sequencing_window)
            
            # Check 3: Is population BELOW target? (CRITICAL CHECK)
            population_below_target = None
            if self.last_count_obs is not None and has_recent_count:
                population_below_target = (self.last_count_obs < self.target_population)
            
            # Apply penalties/rewards based on checks
            if population_below_target:
                # CRITICAL: BIG penalty for dosing when population is already below target
                # This is dangerous - you're killing bacteria that are already too few!
                dosing_bonus = -self.dosing_low_population_penalty
                # # Debug output (only first few episodes)
                # if self.t < 200:
                #     print(f"[DOSING LOW POP] t={self.t}, pop={self.last_count_obs}, target={self.target_population}, penalty={dosing_bonus}")
            elif has_recent_count and has_recent_sequencing:
                # GOOD: Have both recent count AND sequencing data
                if population_below_target is False:
                    # Population is ABOVE target - give additional bonus for informed dosing
                    dosing_bonus = self.informed_dosing_reward + self.informed_dosing_above_target_reward
                    # # Debug output (only first few episodes)
                    # if self.t < 200:
                    #     print(f"[INFORMED DOSING ABOVE TARGET] t={self.t}, pop={self.last_count_obs}, target={self.target_population}, bonus={dosing_bonus}")
                else:
                    # Population is close to or at target - give base informed dosing reward
                    dosing_bonus = self.informed_dosing_reward
                # # Debug output (only first few episodes)
                # if self.t < 200:
                #     print(f"[INFORMED DOSING] t={self.t}, count_age={self.t - self.ts_last_count}, seq_age={self.t - self.ts_last_seq}, bonus={dosing_bonus}")
            elif not has_recent_count or not has_recent_sequencing:
                # BAD: Missing recent count OR sequencing data (blind dosing)
                dosing_bonus = -self.blind_dosing_penalty
                # # Debug output (only first few episodes)
                # if self.t < 200:
                #     count_age = self.t - self.ts_last_count if self.ts_last_count is not None else "None"
                #     seq_age = self.t - self.ts_last_seq if self.ts_last_seq is not None else "None"
                #     print(f"[BLIND DOSING] t={self.t}, has_count={has_recent_count} (age={count_age}), has_seq={has_recent_sequencing} (age={seq_age}), penalty={dosing_bonus}")
            
            # Store for tracking
            self.last_informed_dosing_bonus = dosing_bonus
            
            # ✅ SIMPLIFIED: Return cost penalty + informed dosing bonus/penalty
            # Let population maintenance reward (computed every step) capture efficacy
            # PPO's TD learning will connect: dose → future population drops → better rewards
            return -action_cost * self.w_cost + dosing_bonus
        
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
        
        # Target population (normalized) - gives agent explicit goal
        target_norm = float(self.target_population) / max(1.0, self.population_norm)

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
            ts_since_seq = float('inf')  # Never sequenced yet
        else:
            ts_since_seq = self.t - self.ts_last_seq
        if self.ts_last_count is None:
            ts_since_count = float('inf')  # Never counted yet
        else:
            ts_since_count = self.t - self.ts_last_count

        ts_since_measure = min(ts_since_seq, ts_since_count)

        # If never measured, use maximum staleness value
        if ts_since_measure == float('inf'):
            time_since_last_measure_norm = 1.0  # Maximum value (never measured)
        else:
            time_since_last_measure_norm = min(1.0, ts_since_measure / 100.0)
        
        seq_pending_flag = 1.0 if self.seq_pending else 0.0
        seq_eta_norm = min(1.0, max(0, self.seq_eta) / max(1, self.sequencing_duration))


        avg_genome = avg_genome.flatten()


        obs_parts = [
            budget_norm,
            target_norm,
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

    def enable_survival_bonus(
        self,
        base_bonus: float = 0.01,
        scaling_type: str = "constant",
        scaling_factor: float = 0.1,
        max_bonus: float = 0.1,
    ) -> None:
        """
        Enable survival bonus reward to encourage longer episodes.
        
        Args:
            base_bonus: Base bonus per step
            scaling_type: "constant", "linear", or "exponential"
            scaling_factor: Multiplier for linear/exponential scaling
            max_bonus: Maximum bonus cap
        """
        self.survival_bonus_reward = SurvivalBonusReward(
            base_bonus=base_bonus,
            scaling_type=scaling_type,
            scaling_factor=scaling_factor,
            max_bonus=max_bonus,
        )
        print(f"✓ Survival bonus reward enabled: base={base_bonus}, type={scaling_type}")
    
    def enable_budget_conservation(
        self,
        weight: float = 0.01,
        spending_penalty_factor: float = 1.0,
        reserve_bonus_threshold: float = 0.5,
        reserve_bonus_magnitude: float = 0.005,
    ) -> None:
        """
        Enable budget conservation reward to encourage efficient spending.
        
        Args:
            weight: Overall weight for budget conservation
            spending_penalty_factor: Penalty multiplier for spending rate
            reserve_bonus_threshold: Budget fraction for reserve bonus (0.5 = 50%)
            reserve_bonus_magnitude: Bonus magnitude when above threshold
        """
        self.budget_conservation_reward = BudgetConservationReward(
            weight=weight,
            spending_penalty_factor=spending_penalty_factor,
            reserve_bonus_threshold=reserve_bonus_threshold,
            reserve_bonus_magnitude=reserve_bonus_magnitude,
        )
        print(f"✓ Budget conservation reward enabled: weight={weight}, threshold={reserve_bonus_threshold}")
    
    def disable_survival_bonus(self) -> None:
        """Disable survival bonus reward."""
        self.survival_bonus_reward = None
        print("✓ Survival bonus reward disabled")
    
    def disable_budget_conservation(self) -> None:
        """Disable budget conservation reward."""
        self.budget_conservation_reward = None
        print("✓ Budget conservation reward disabled")

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