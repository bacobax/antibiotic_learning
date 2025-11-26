from typing import Any, Callable, Dict, Tuple, Optional, Union, List
import numpy as np
import torch
from simulation.simulation_config import ANTIBIOTIC_TYPES, antibiotic_resistances, TOX_TIMES_DOSE_MAX, N_TRAITS, N_BACTERIA_TYPES
from rl.reward import (
    SurvivalBonusReward,
    KernelPopulationMaintenanceReward,
    InformedDosingReward,
    SequencingReward,
    CountReward,
    StrategicNoopReward,
    CriticalNoDosePenalty,
    CriticalNoCountPenalty,
    ExtinctionPenalty,
)

# Discrete actions
ACTION_NOOP = 0
ACTION_COUNT_BACTERIA = 1
ACTION_SEQUENCING = 2
ACTION_DOSE = 3

class PetriEnvWrapper:
    MAX_AGE = 100.0
    """
    Simplified reward-based wrapper for Mesa bacteria simulation.
    
    Follows a clean pre-step/post-step reward structure:
    - Pre-step rewards: evaluate action quality before execution
    - Post-step penalties: evaluate critical states after environment step
    
    Observation vector (float32, mask aware):
        [ last_count_norm,
            has_last_count,
            last_count_age_norm,
            avg_genome_flat (12 values),
            has_last_seq,
            last_seq_age_norm,
            measure_age_norm,
            dose_history_K (has, norm, age),
            dose_history_I (has, norm, age),
            dose_history_A (has, norm, age),
            t_norm
        ]
    Length = 28. Missing values are zeroed with companion mask bits.
    
    REWARD STRUCTURE:
    =================
    
    Total reward = pre_reward + post_penalties + kernel_maintenance + survival_bonus + prediction_reward + early_termination_penalty
    
    PRE-STEP REWARDS:
    - DOSE: informed vs blind dosing, above vs below target
    - SEQ: informative vs redundant sequencing
    - COUNT: timing-based counting rewards
    - NOOP: strategic waiting rewards
    
    POST-STEP PENALTIES:
    - Critical no-dose: penalty for not dosing when population is critical
    - Critical no-count: penalty for stale count data
    - Extinction: penalty for population collapse
    
    KERNEL MAINTENANCE:
    - Gaussian or Laplace kernel-based population maintenance reward
    
    SURVIVAL BONUS:
    - Per-step reward for staying alive (constant/linear/exponential)
    
    PREDICTION REWARD:
    - Accuracy reward for population prediction (COUNT-only)
    
    EARLY TERMINATION:
    - Handles unrecoverable states and extinction
    """

    def __init__(
        self,
        mesa_model_factory: Callable[[], Any],
        k_doses: int,
        scale_dose: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        max_steps: int = 1000,
        
        # ===== Timing and freshness thresholds =====
        t_count_freshness: int = 5,            # Steps for count to remain "fresh"
        t_seq_freshness: int = 8,              # Steps for sequencing to remain "fresh"  
        max_count_window: int = 30,            # Max steps without counting before penalty
        critical_ratio: float = 3.0,           # Population ratio for critical state
        
        # Timing windows for informative actions
        t_min_elapsed_time_count: int = 5,     # Min time between counts
        t_max_elapsed_time_count: int = 30,    # Max time for informative count
        t_min_elapsed_time_seq: int = 8,       # Min time between sequencing
        t_max_elapsed_time_seq: int = 50,      # Max time for informative sequencing
        
        # ===== Action costs and durations =====
        sequencing_cost: float = 2.5,
        sequencing_duration: int = 5,
        dose_cost: float = 2.0,
        dose_cost_per_unit: float = 2.0,
        count_cost: float = 0.5,
        
        # ===== Pre-step reward scalars =====
        # Informed dosing
        penalty_informed_dosing_under: float = 5.0,
        reward_informed_dosing_above: float = 2.0,
        reward_informed_dosing_above_without_seq: float = 1.0,
        penalty_blind_dose: float = 3.0,
        
        # Sequencing
        seq_already_pending_penalty: float = 2.0,
        informative_seq_reward: float = 1.0,
        
        # Counting
        cost_penalty: float = 0.5,
        informative_count_reward: float = 1.0,
        
        # Strategic NOOP
        strategic_noop_reward: float = 0.5,
        
        # ===== Post-step penalty scalars =====
        penalty_critical_no_dose: float = 5.0,
        penalty_critical_no_count: float = 2.0,
        big_penalty: float = 50.0,              # Extinction penalty
        
        # ===== Population maintenance (kernel-based) =====
        kernel_maintenance_enabled: bool = True,
        kernel_type: str = "gaussian",          # "gaussian" or "laplace"
        kernel_bandwidth: float = 50.0,
        kernel_weight: float = 1.0,
        
        # ===== Survival bonus =====
        survival_bonus_enabled: bool = True,
        survival_bonus_base: float = 0.1,
        survival_bonus_scaling_type: str = "linear",
        survival_bonus_scaling_factor: float = 0.001,
        survival_bonus_max: float = 2.0,
        
        # ===== Prediction reward =====
        prediction_reward_enabled: bool = True,
        prediction_reward_weight: float = 0.4,
        
        # ===== Early termination =====
        early_termination_enabled: bool = True,
        early_termination_penalty: float = 25.0,
        early_termination_min_penalty: Optional[float] = 10.0,
        early_termination_penalty_decay_power: float = 1.0,
        early_termination_population_threshold: float = 1.5,
        early_termination_population_low_threshold: float = 0.2,
        early_termination_extinction_penalty: float = 12.0,
        early_termination_require_budget_depleted: bool = False,
        
        # ===== Environment parameters =====
        target_population: int = 100,
        population_norm: float = 500.0,
        budget_init: float = 100.0,
        budget_norm: float = 100.0,
        
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.mesa_model_factory = mesa_model_factory
        self.k_doses = k_doses
        self.scale_dose = scale_dose if scale_dose is not None else (lambda x: x)
        self.max_steps = max_steps
        self.episode_length = max_steps
        
        # ===== Timing and freshness thresholds =====
        self.t_count_freshness = int(t_count_freshness)
        self.t_seq_freshness = int(t_seq_freshness)
        self.max_count_window = int(max_count_window)
        self.critical_ratio = float(critical_ratio)
        
        self.t_min_elapsed_time_count = int(t_min_elapsed_time_count)
        self.t_max_elapsed_time_count = int(t_max_elapsed_time_count)
        self.t_min_elapsed_time_seq = int(t_min_elapsed_time_seq)
        self.t_max_elapsed_time_seq = int(t_max_elapsed_time_seq)
        
        # ===== Action costs and durations =====
        self.sequencing_cost = float(sequencing_cost)
        self.sequencing_duration = int(sequencing_duration)
        self.dose_cost = float(dose_cost)
        self.dose_cost_per_unit = float(dose_cost_per_unit)
        self.count_cost = float(count_cost)
        
        # ===== Environment parameters =====
        self.target_population = float(target_population)
        self.population_norm = float(population_norm)
        self.budget_init = float(budget_init)
        self.budget_norm = float(budget_norm)
        self.device = device
        self.dtype = dtype
        
        # ===== Prediction reward =====
        self.prediction_reward_enabled = bool(prediction_reward_enabled)
        self.prediction_reward_weight = float(prediction_reward_weight)
        
        # ===== Early termination parameters =====
        self.early_termination_enabled = bool(early_termination_enabled)
        self.early_termination_penalty = float(max(0.0, early_termination_penalty))
        min_penalty = early_termination_min_penalty if early_termination_min_penalty is not None else early_termination_penalty
        self.early_termination_penalty_min = float(max(0.0, min_penalty))
        if self.early_termination_penalty_min > self.early_termination_penalty:
            self.early_termination_penalty_min = self.early_termination_penalty
        self.early_termination_penalty_decay_power = float(max(1e-6, early_termination_penalty_decay_power))
        self._early_termination_penalty_span = (
            self.early_termination_penalty - self.early_termination_penalty_min
        )
        self.early_termination_population_threshold = float(early_termination_population_threshold)
        self.early_termination_population_low_threshold = float(max(0.0, early_termination_population_low_threshold))
        self.early_termination_extinction_penalty = float(max(0.0, early_termination_extinction_penalty))
        self.early_termination_require_budget_depleted = bool(early_termination_require_budget_depleted)
        
        # ===== Reward modules =====
        # Pre-step rewards
        self.informed_dosing_reward = InformedDosingReward(
            penalty_dosing_under_target=penalty_informed_dosing_under,
            reward_dosing_above_with_seq=reward_informed_dosing_above,
            reward_dosing_above_no_seq=reward_informed_dosing_above_without_seq,
            penalty_blind_dose=penalty_blind_dose,
        )
        
        self.sequencing_reward = SequencingReward(
            seq_already_pending_penalty=seq_already_pending_penalty,
            informative_seq_reward=informative_seq_reward,
        )
        
        self.count_reward = CountReward(
            cost_penalty=cost_penalty,
            informative_count_reward=informative_count_reward,
        )
        
        self.strategic_noop_reward = StrategicNoopReward(
            strategic_noop_reward=strategic_noop_reward,
        )
        
        # Post-step penalties
        self.critical_no_dose_penalty = CriticalNoDosePenalty(
            penalty_critical_no_dose=penalty_critical_no_dose,
        )
        
        self.critical_no_count_penalty = CriticalNoCountPenalty(
            penalty_critical_no_count=penalty_critical_no_count,
        )
        
        self.extinction_penalty = ExtinctionPenalty(
            big_penalty=big_penalty,
        )
        
        # Kernel-based population maintenance
        self.kernel_maintenance_enabled = bool(kernel_maintenance_enabled)
        if self.kernel_maintenance_enabled:
            self.kernel_maintenance_reward = KernelPopulationMaintenanceReward(
                target_population=target_population,
                kernel_type=kernel_type,
                bandwidth=kernel_bandwidth,
                weight=kernel_weight,
            )
        else:
            self.kernel_maintenance_reward = None
        
        # Survival bonus
        self.survival_bonus_reward = None
        if survival_bonus_enabled:
            self.survival_bonus_reward = SurvivalBonusReward(
                base_bonus=survival_bonus_base,
                scaling_type=survival_bonus_scaling_type,
                scaling_factor=survival_bonus_scaling_factor,
                max_bonus=survival_bonus_max,
            )
        
        # ===== Runtime state =====
        self.model: Any = None
        self.t = 0
        self.MAX_AGE = float(self.__class__.MAX_AGE)
        self.episode_return = 0.0
        self.budget = budget_init
        # Budget tracking per episode
        self.episode_budget_spent = 0.0
        self.episode_start_budget = float(budget_init)
        self.last_step_budget = float(budget_init)
        
        # ===== Timer state (key to the new reward system) =====
        # These track time since last action (0.0 = never performed or just performed)
        self.t_since_last_count = 0.0
        self.t_since_last_dose = 0.0
        self.t_since_last_seq = 0.0
        
        # Tracking flags: has this action ever been performed?
        self.has_ever_counted = False
        self.has_ever_dosed = False
        self.has_ever_sequenced = False
        
        # Sequencing state
        self.seq_pending = False
        self.seq_eta = 0
        self.recent_sequencing = False

        # Legacy dose event tracking (kept inert; initialized to avoid attribute errors)
        self.recent_dose_events = []
        self.pending_dose_events = []
        self.last_regular_count_bonus = 0.0
        self.last_informed_dose_reward = 0.0
        self.last_count_population_reward = 0.0
        self.last_action_cost_penalty = 0.0
        # Informed dosing legacy params (not used in simplified rewards)
        self.informed_reward_window_steps = 0
        self.informed_reward_weight = 0.0
        self.informed_time_decay_enabled = False
        self.informed_decay_type = "linear"
        self.informed_decay_rate = 0.0
        self.informed_min_reward_fraction = 0.0
        
        # ===== Observation cache (what the agent "knows") =====
        self.last_count_obs: Optional[int] = None
        self.prev_count_obs: Optional[int] = None
        self.last_seq_obs: Optional[Dict[str, Any]] = None
        self.ts_last_seq: Optional[int] = None
        self.ts_last_count: Optional[int] = None
        self.prev_count_step: Optional[int] = None
        
        # Measurement/state caches
        self.avg_genome = np.zeros((N_BACTERIA_TYPES, N_TRAITS), dtype=np.float32)
        
        # Dosing history (K, I, A) - still needed for observation
        self.last_dose_K = 0.0
        self.last_dose_I = 0.0
        self.last_dose_A = 0.0
        self.ts_last_dose_K: Optional[int] = None
        self.ts_last_dose_I: Optional[int] = None
        self.ts_last_dose_A: Optional[int] = None
        self._dose_update_buffer: Optional[np.ndarray] = None
        self.max_dose_values = self._infer_max_dose_values()
        
        # ===== Reward tracking (for logging) =====
        self.last_pre_reward = 0.0
        self.last_post_penalties = 0.0
        self.last_kernel_maintenance_reward = 0.0
        self.last_survival_bonus = 0.0
        self.last_prediction_reward = 0.0
        self.last_early_termination_penalty = 0.0
        self.early_termination_triggered = False

    # -------------------------
    # Public API
    # -------------------------
    
    def reset(self)-> np.ndarray:
        self.model = self.mesa_model_factory()
        self.t = 0
        self.episode_return = 0.0
        self.budget = self.budget_init
        self.episode_budget_spent = 0.0
        self.episode_start_budget = float(self.budget_init)
        self.last_step_budget = float(self.budget)
        
        # Reset timer state
        self.t_since_last_count = 0.0
        self.t_since_last_dose = 0.0
        self.t_since_last_seq = 0.0
        
        # Reset tracking flags
        self.has_ever_counted = False
        self.has_ever_dosed = False
        self.has_ever_sequenced = False
        
        # Reset sequencing state
        self.seq_pending = False
        self.seq_eta = 0
        self.recent_sequencing = False
        
        # Reset reward component tracking
        self.last_pre_reward = 0.0
        self.last_post_penalties = 0.0
        self.last_kernel_maintenance_reward = 0.0
        self.last_survival_bonus = 0.0
        self.last_prediction_reward = 0.0
        self.last_early_termination_penalty = 0.0
        self.early_termination_triggered = False

        # Clear observation caches
        self.last_count_obs = None
        self.prev_count_obs = None
        self.last_seq_obs = None
        self.ts_last_count = None
        self.ts_last_seq = None
        self.prev_count_step = None
        self.avg_genome.fill(0.0)

        # Reset dosing history
        self.last_dose_K = 0.0
        self.last_dose_I = 0.0
        self.last_dose_A = 0.0
        self.ts_last_dose_K = None
        self.ts_last_dose_I = None
        self.ts_last_dose_A = None
        self._dose_update_buffer = None
        self.max_dose_values = self._infer_max_dose_values()

        return self._build_observation()

    # -------------------------
    # Freshness helpers (following pseudo-code)
    # -------------------------

    def _count_fresh(self) -> bool:
        """
        COUNT_FRESH = (has counted before) AND (t_since_last_count < t_count_freshness)
                      AND (t_since_last_dose > t_since_last_count)
        If never counted (last_count_obs is None), return False.
        """
        if self.last_count_obs is None or not self.has_ever_counted:
            return False
        return (self.t_since_last_count < self.t_count_freshness and
                self.t_since_last_dose > self.t_since_last_count)
    
    def _seq_fresh(self) -> bool:
        """
        SEQ_FRESH = (has sequenced before) AND (t_since_last_seq < t_seq_freshness)
        If never sequenced, return False.
        """
        if self.last_seq_obs is None or not self.has_ever_sequenced:
            return False
        return self.t_since_last_seq < self.t_seq_freshness

    # -------------------------
    # Reward computation (following pseudo-code)
    # -------------------------

    def _compute_pre_reward(self, action: int) -> float:
        """
        Compute pre-step reward based on action quality.
        
        Args:
            action: Discrete action (NOOP=0, COUNT=1, SEQ=2, DOSE=3)
            
        Returns:
            Pre-step reward as float
        """
        count_fresh = self._count_fresh()
        
        if action == ACTION_DOSE:
            return self.informed_dosing_reward(
                count_fresh=count_fresh,
                last_count_pop=float(self.last_count_obs) if self.last_count_obs is not None else None,
                target_pop=self.target_population,
                recent_sequencing=self.recent_sequencing,
            )
        
        elif action == ACTION_SEQUENCING:
            return self.sequencing_reward(
                seq_pending=self.seq_pending,
                t_since_last_seq=self.t_since_last_seq,
                t_min_elapsed_time_seq=float(self.t_min_elapsed_time_seq),
                t_max_elapsed_time_seq=float(self.t_max_elapsed_time_seq),
            )
        
        elif action == ACTION_COUNT_BACTERIA:
            return self.count_reward(
                t_since_last_count=self.t_since_last_count,
                t_min_elapsed_time_count=float(self.t_min_elapsed_time_count),
                t_max_elapsed_time_count=float(self.t_max_elapsed_time_count),
            )
        
        elif action == ACTION_NOOP:
            return self.strategic_noop_reward(
                count_fresh=count_fresh,
                last_count_pop=float(self.last_count_obs) if self.last_count_obs is not None else None,
                target_pop=self.target_population,
            )
        
        return 0.0
    
    def _compute_post_penalties(self, action: int, true_population: int) -> float:
        """
        Compute post-step penalties after environment step.
        
        Args:
            action: Discrete action taken
            true_population: Current true population
            
        Returns:
            Post-step penalties as float (typically negative)
        """
        penalties = 0.0
        count_fresh = self._count_fresh()
        
        # Critical no-dose penalty
        penalties += self.critical_no_dose_penalty(
            count_fresh=count_fresh,
            last_count_pop=float(self.last_count_obs) if self.last_count_obs is not None else None,
            target_pop=self.target_population,
            critical_ratio=self.critical_ratio,
            action_was_dose=(action == ACTION_DOSE),
        )
        
        # Critical no-count penalty
        penalties += self.critical_no_count_penalty(
            t_since_last_count=self.t_since_last_count,
            max_count_window=float(self.max_count_window),
        )
        
        # Extinction penalty
        penalties += self.extinction_penalty(population=true_population)
        
        return penalties
    
    def _update_timers_after_env_step(self, action: int, dt: float = 1.0) -> None:
        """
        Update internal timer state AFTER env_step() has been called.
        Following the pseudo-code structure.
        
        Args:
            action: Discrete action taken
            dt: Time increment (default 1.0)
        """
        # COUNT
        if action == ACTION_COUNT_BACTERIA:
            # Count is instant (duration 0), so result lands immediately
            # The count observation is cached in step() after env_step()
            self.t_since_last_count = 0.0
            self.has_ever_counted = True
        else:
            self.t_since_last_count += dt
        
        # DOSE
        if action == ACTION_DOSE:
            self.t_since_last_dose = 0.0
            self.has_ever_dosed = True
        else:
            self.t_since_last_dose += dt
        
        # SEQ
        if action == ACTION_SEQUENCING:
            self.t_since_last_seq = 0.0
            self.has_ever_sequenced = True
        else:
            self.t_since_last_seq += dt
        
        # Decay recent_sequencing flag based on window
        if self.has_ever_sequenced and self.t_since_last_seq > self.t_seq_freshness:
            self.recent_sequencing = False

    def _apply_internal_agent_updates(self, action: int) -> None:
        """
        Internal agent state updates BEFORE env_step().
        Following pseudo-code structure.
        
        Args:
            action: Discrete action about to be executed
        """
        # SEQ: mark as pending
        if action == ACTION_SEQUENCING:
            if not self.seq_pending:
                self.seq_pending = True
                self.recent_sequencing = True
                # Timer reset happens after env_step() in _update_timers_after_env_step

    def step(self, a_discrete: int, a_cont: np.ndarray, pred_population: Optional[float] = None) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step following the pseudo-code structure:
        1) Compute pre-reward
        2) Apply internal agent updates
        3) Advance environment (env_step)
        4) Update timers after env_step
        5) Compute post-penalties
        6) Return observation + total reward
        
        Args:
            a_discrete: Discrete action (NOOP=0, COUNT=1, SEQ=2, DOSE=3)
            a_cont: Continuous action (dose amounts)
            pred_population: Optional population prediction (for prediction reward)
            
        Returns:
            obs, reward, done, info
        """
        assert 0 <= a_discrete <= 3, f"a_discrete out of range: {a_discrete}"
        assert isinstance(a_cont, np.ndarray) and a_cont.shape == (self.k_doses,)
        
        # Reset reward tracking
        self.last_pre_reward = 0.0
        self.last_post_penalties = 0.0
        self.last_kernel_maintenance_reward = 0.0
        self.last_survival_bonus = 0.0
        self.last_prediction_reward = 0.0
        self.last_early_termination_penalty = 0.0
        self.early_termination_triggered = False
        
        # ==============================================
        # STEP 1: PRE-STEP REWARD
        # ==============================================
        pre_reward = self._compute_pre_reward(a_discrete)
        self.last_pre_reward = pre_reward
        
        # ==============================================
        # STEP 2: INTERNAL AGENT UPDATES (before env step)
        # ==============================================
        self._apply_internal_agent_updates(a_discrete)
        
        # ==============================================
        # STEP 3: EXECUTE ACTION & ADVANCE ENVIRONMENT
        # ==============================================
        # Deduct action costs from budget
        action_cost = 0.0
        if a_discrete == ACTION_COUNT_BACTERIA:
            action_cost = self.count_cost
        elif a_discrete == ACTION_SEQUENCING:
            action_cost = self.sequencing_cost
        elif a_discrete == ACTION_DOSE:
            scaled_doses = self.scale_dose(a_cont)
            action_cost = self.dose_cost + np.sum(scaled_doses) * self.dose_cost_per_unit
            # Apply antibiotics to environment
            self._apply_antibiotics(scaled_doses)
            # Update dose history for observation
            self._update_dose_history(scaled_doses)
        
        self.budget -= action_cost
        
        # Advance the biological simulation by one step
        self.model.step()
        self.t += 1
        
        # ==============================================
        # STEP 4: POST-ENV UPDATES & OBSERVATION CACHING
        # ==============================================
        # Update timers based on action taken
        self._update_timers_after_env_step(a_discrete, dt=1.0)
        
        # Handle sequencing countdown (if pending)
        if self.seq_pending:
            self.seq_eta -= 1
            if self.seq_eta <= 0:
                # Sequencing result lands NOW
                seq_result = self._read_true_sequencing()
                self._cache_sequencing_obs(seq_result)
                self.seq_pending = False
                self.seq_eta = 0
        
        # Handle COUNT action (instant, duration=0)
        count_result_landed = False
        population_counted_norm = 0.0
        if a_discrete == ACTION_COUNT_BACTERIA:
            true_pop = self._read_true_population()
            self._cache_count_obs(true_pop)
            count_result_landed = True
            population_counted_norm = float(true_pop) / max(1.0, self.population_norm)
        
        # Read true population for reward computation
        true_population = self._read_true_population()
        
        # Build observation from cached agent knowledge
        obs = self._build_observation()
        
        # ==============================================
        # STEP 5: POST-STEP PENALTIES
        # ==============================================
        post_penalties = self._compute_post_penalties(a_discrete, true_population)
        self.last_post_penalties = post_penalties
        
        # ==============================================
        # STEP 6: ADDITIONAL REWARDS
        # ==============================================
        
        # Kernel-based population maintenance reward
        kernel_maintenance_reward = 0.0
        if self.kernel_maintenance_enabled and self.kernel_maintenance_reward is not None:
            kernel_maintenance_reward = self.kernel_maintenance_reward(true_population)
            self.last_kernel_maintenance_reward = kernel_maintenance_reward
        
        # Survival bonus
        survival_bonus = 0.0
        if self.survival_bonus_reward is not None:
            survival_bonus = self.survival_bonus_reward(self.t)
            self.last_survival_bonus = survival_bonus
        
        # Prediction accuracy reward (COUNT-only)
        prediction_reward = 0.0
        if count_result_landed and pred_population is not None and self.prediction_reward_enabled:
            # Compute accuracy reward based on prediction error
            predicted_norm = float(pred_population)
            actual_norm = population_counted_norm
            error = abs(predicted_norm - actual_norm)
            # Reward is higher for lower error (exponential decay)
            import math
            prediction_reward = self.prediction_reward_weight * math.exp(-5.0 * error)
            self.last_prediction_reward = prediction_reward
        
        # ==============================================
        # STEP 7: EARLY TERMINATION CHECK
        # ==============================================
        base_done = (self.t >= self.max_steps) or (self.budget <= 0.0)
        done = base_done
        early_termination_penalty = 0.0
        
        # Immediate termination on extinction
        if true_population <= 0:
            done = True
            early_termination_penalty = -self.early_termination_extinction_penalty
            self.early_termination_triggered = True
        
        # Check for unrecoverable states (only NOOP available)
        if self.early_termination_enabled and not base_done and true_population > 0:
            # Unrecoverable if:
            # 1) Population is very high OR very low
            # 2) Budget depleted (if required)
            population_high = true_population > (self.target_population * self.early_termination_population_threshold)
            population_low = true_population < (self.target_population * self.early_termination_population_low_threshold)
            budget_depleted = self.budget <= 0.0
            
            is_unrecoverable = (population_high or population_low)
            if self.early_termination_require_budget_depleted:
                is_unrecoverable = is_unrecoverable and budget_depleted
            
            if is_unrecoverable:
                # Trigger early termination
                done = True
                early_termination_penalty = -self._compute_step_scaled_early_termination_penalty()
                self.early_termination_triggered = True
        
        self.last_early_termination_penalty = early_termination_penalty
        
        # ==============================================
        # STEP 8: TOTAL REWARD
        # ==============================================
        total_reward = (
            pre_reward +
            post_penalties +
            kernel_maintenance_reward +
            survival_bonus +
            prediction_reward +
            early_termination_penalty
        )
        
        self.episode_return += total_reward
        
        # ==============================================
        # STEP 9: INFO DICT
        # ==============================================
        info = {
            # Episode tracking
            "t": self.t,
            "episode_return": self.episode_return,
            "budget": self.budget,
            "true_population": true_population,
            
            # Timer state (for debugging)
            "t_since_last_count": self.t_since_last_count,
            "t_since_last_dose": self.t_since_last_dose,
            "t_since_last_seq": self.t_since_last_seq,
            "count_fresh": self._count_fresh(),
            "seq_fresh": self._seq_fresh(),
            "recent_sequencing": self.recent_sequencing,
            
            # Reward breakdown
            "reward_pre": pre_reward,
            "reward_post_penalties": post_penalties,
            "reward_kernel_maintenance": kernel_maintenance_reward,
            "reward_survival_bonus": survival_bonus,
            "reward_prediction": prediction_reward,
            "reward_early_termination_penalty": early_termination_penalty,
            "reward_total": total_reward,
            
            # Early termination
            "early_termination_triggered": self.early_termination_triggered,
            
            # Prediction supervision signal
            "population_next_norm": population_counted_norm,
            "count_was_performed": count_result_landed,
        }
        
        return obs, float(total_reward), bool(done), info

    # -------------------------
    # Early termination helpers
    # -------------------------


    def _compute_step_scaled_early_termination_penalty(self) -> float:
        
        """Scale the early termination penalty based on how far the episode has progressed."""
        if self.max_steps <= 0:
            remaining_fraction = 1.0
        else:
            remaining_fraction = max(0.0, min(1.0, (self.max_steps - self.t) / float(self.max_steps)))

        if self._early_termination_penalty_span <= 0.0:
            scaled_penalty = self.early_termination_penalty_min
        else:
            scaled_penalty = self.early_termination_penalty_min + (
                self._early_termination_penalty_span
                * (remaining_fraction ** self.early_termination_penalty_decay_power)
            )

        # Clamp to configured bounds to avoid floating point drift
        scaled_penalty = min(self.early_termination_penalty, max(self.early_termination_penalty_min, scaled_penalty))
        return float(scaled_penalty)

    # -------------------------
    # Action execution
    # -------------------------

    def _execute_action(self, a_discrete: int, a_cont: np.ndarray) -> float:
        """
        Applies the chosen action. Returns *immediate* reward as float.
        
        NOTE: With hybrid action masking (Option C), all actions are guaranteed
        to be affordable when called. No need to check budget here.
        
        Simplified approach: Let natural consequences (population changes) 
        drive learning through maintenance reward, rather than predicting
        efficacy at dose time.
        
        Returns:
            Immediate reward (float)
        """
        # Reset tracking variables (will be set if applicable)
        self.last_regular_count_bonus = 0.0
        self.last_informed_dose_reward = 0.0
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
            # Calculate and apply count cost
            action_cost = self.count_cost
            self.budget -= action_cost
            self.episode_budget_spent += action_cost
            
            # Get current population to evaluate distance from target
            current_population = self._read_true_population()
            
            # Count population reward: immediate feedback based on distance from target
            # Positive reward when close to target, negative when far
            count_pop_reward = 0.0
            if self.count_population_reward > 0.0:
                distance = abs(current_population - self.target_population)
                denom = max(1.0, float(self.population_norm_reward))
                normalized_distance = min(distance / denom, 1.0)
                # Exponential shaping: reward = R * (exp(-alpha * norm_dist) - beta)
                exp_term = np.exp(-self.count_population_reward_alpha * normalized_distance)
                shifted = exp_term - self.count_population_reward_beta
                # Normalize to [-1, 1] before scaling by the configured magnitude
                scale_bound = max(self.count_population_reward_beta, 1.0 - self.count_population_reward_beta, 1e-6)
                normalized_reward = shifted / scale_bound
                count_pop_reward = self.count_population_reward * np.clip(normalized_reward, -1.0, 1.0)
                count_pop_reward = float(np.clip(count_pop_reward, -self.count_population_reward, self.count_population_reward))
            
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
            self.last_action_cost_penalty = -self.count_cost * self.w_cost
            return -self.count_cost * self.w_cost + regular_monitoring_bonus + count_pop_reward

        if a_discrete == ACTION_SEQUENCING:
            # Calculate and apply sequencing cost
            action_cost = self.sequencing_cost
            self.budget -= action_cost
            self.episode_budget_spent += action_cost
            if not self.seq_pending:
                self.seq_pending = True
                self.seq_eta = int(self.sequencing_duration)
                self.last_action_cost_penalty = -action_cost * self.w_cost
                return -action_cost * self.w_cost
            else:
                self.last_action_cost_penalty = -action_cost * self.w_cost
                return -action_cost * self.w_cost - float(self.redundant_sequencing_penalty)

        if a_discrete == ACTION_DOSE:
            # Calculate dose cost from continuous action
            scaled = self.scale_dose(np.clip(a_cont, 0.0, 1.0))
            variable_cost = float(np.sum(scaled) * self.dose_cost_per_unit)
            action_cost = self.dose_cost + variable_cost
            
            # Apply antibiotics
            self._apply_antibiotics(scaled)
            
            # Deduct cost
            self.budget -= action_cost
            self.episode_budget_spent += action_cost
            self._dose_update_buffer = np.array(scaled, dtype=np.float32, copy=True)
            self.last_action_cost_penalty = -action_cost * self.w_cost

            return -action_cost * self.w_cost
        
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

    def _infer_max_dose_values(self) -> np.ndarray:
        """Estimate per-antibiotic max doses using the scaling function."""
        try:
            ones = np.ones(self.k_doses, dtype=np.float32)
            scaled = np.asarray(self.scale_dose(ones), dtype=np.float32)
        except Exception:
            scaled = np.ones(self.k_doses, dtype=np.float32)
        if scaled.shape[0] < self.k_doses:
            padded = np.ones(self.k_doses, dtype=np.float32)
            padded[:scaled.shape[0]] = scaled
            scaled = padded
        elif scaled.shape[0] > self.k_doses:
            scaled = scaled[: self.k_doses]
        return np.maximum(scaled, 1e-6)

    # -------------------------
    # Observation management (gated)
    # -------------------------

    def _cache_count_obs(self, population: int) -> None:
        self.prev_count_obs = None if self.last_count_obs is None else int(self.last_count_obs)
        self.prev_count_step = None if self.ts_last_count is None else int(self.ts_last_count)
        self.last_count_obs = int(population)
        self.ts_last_count = self.t

    def _update_dose_history(self, doses: np.ndarray) -> None:
        """Persist last dose magnitudes and timestamps for antibiotics K, I, A."""
        mapping = ("K", "I", "A")
        for idx, label in enumerate(mapping):
            if idx >= doses.shape[0]:
                continue
            value = float(doses[idx])
            setattr(self, f"last_dose_{label}", value)
            setattr(self, f"ts_last_dose_{label}", self.t)
        self._record_dose_event(doses)

    def _record_dose_event(self, doses: np.ndarray) -> None:
        """Store dose information so it can be rewarded the next time a COUNT lands."""
        if self.ts_last_count is None:
            return  # No baseline COUNT yet → skip

        dose_magnitude = float(np.sum(doses))
        if dose_magnitude <= 0.0:
            return

        event = {
            "step": int(self.t),
            "doses": np.array(doses, dtype=np.float32, copy=True),
        }
        self.recent_dose_events.append(event)

    def _score_recent_doses(self) -> float:
        """Compute reward for doses taken since the previous COUNT once a new COUNT arrives."""
        if not self.recent_dose_events:
            self.last_informed_dose_reward = 0.0
            return 0.0

        if self.prev_count_obs is None or self.last_count_obs is None:
            self.recent_dose_events.clear()
            self.last_informed_dose_reward = 0.0
            return 0.0

        population_drop = max(0.0, float(self.prev_count_obs - self.last_count_obs))
        if population_drop <= 0.0:
            self.recent_dose_events.clear()
            self.last_informed_dose_reward = 0.0
            return 0.0

        total_reward = 0.0
        baseline_step = self.prev_count_step

        for event in self.recent_dose_events:
            event_step = int(event.get("step", self.t))
            steps_since_count = 0
            if baseline_step is not None:
                steps_since_count = max(0, event_step - int(baseline_step) - 1)

            if steps_since_count > self.informed_reward_window_steps:
                continue

            dose_magnitude = float(np.sum(event["doses"]))
            if dose_magnitude <= 0.0:
                continue

            decay_factor = self._dose_time_decay_factor(steps_since_count)
            reward = population_drop * dose_magnitude * self.informed_reward_weight * decay_factor
            reward = min(reward, self.informed_max_reward_per_dose)
            total_reward += reward

        self.recent_dose_events.clear()
        self.last_informed_dose_reward = float(total_reward)
        return self.last_informed_dose_reward

    def _dose_time_decay_factor(self, steps_since_count: int) -> float:
        if not self.informed_time_decay_enabled:
            return 1.0

        if steps_since_count <= 0:
            factor = 1.0
        elif self.informed_decay_type == "linear":
            factor = max(0.0, 1.0 - self.informed_decay_rate * steps_since_count)
        else:
            factor = float(self.informed_decay_rate ** steps_since_count)

        factor = max(self.informed_min_reward_fraction, factor)
        return float(min(1.0, factor))

    def _cache_sequencing_obs(self, seq: Dict[str, Any]) -> None:
        self.last_seq_obs = {
            "avg_genome": seq["avg_genome"].astype(np.float32),
            "proportions": seq["proportions"].astype(np.float32),
            "t": self.t,
        }
        self.ts_last_seq = self.t
        self.avg_genome = np.array(seq["avg_genome"], dtype=np.float32, copy=True)

    def _age_norm(self, timestamp: Optional[int]) -> float:
        if timestamp is None:
            return 0.0
        age = max(0, self.t - timestamp)
        clipped = min(age, self.MAX_AGE)
        return float(clipped) / self.MAX_AGE if self.MAX_AGE > 0 else 0.0

    def _build_observation(self) -> np.ndarray:
        """
        Assemble what the agent is allowed to see (cached measurements + meta).
        """
        last_count_norm = 0.0 if self.last_count_obs is None else float(self.last_count_obs) / max(1.0, self.population_norm)
        has_last_count = 1.0 if self.last_count_obs is not None else 0.0
        last_count_age_norm = self._age_norm(self.ts_last_count)

        genome_values = self.avg_genome.astype(np.float32, copy=False).reshape(-1)

        has_last_seq = 1.0 if self.ts_last_seq is not None else 0.0
        last_seq_age_norm = self._age_norm(self.ts_last_seq)

        age_norms = []
        if self.ts_last_count is not None:
            age_norms.append(last_count_age_norm)
        if self.ts_last_seq is not None:
            age_norms.append(last_seq_age_norm)
        measure_age_norm = min(age_norms) if age_norms else 0.0

        dose_features: List[float] = []
        dose_state = [
            (self.last_dose_K, self.ts_last_dose_K, 0),
            (self.last_dose_I, self.ts_last_dose_I, 1),
            (self.last_dose_A, self.ts_last_dose_A, 2),
        ]
        for value, timestamp, idx in dose_state:
            ts_exists = timestamp is not None
            has_last_dose = 1.0 if ts_exists else 0.0
            max_dose = self.max_dose_values[idx] if idx < self.max_dose_values.shape[0] else 1.0
            norm = (float(value) / max_dose) if ts_exists and max_dose > 0 else 0.0
            norm = float(np.clip(norm, 0.0, 1.0))
            age_norm = self._age_norm(timestamp)
            if not ts_exists:
                norm = 0.0
                age_norm = 0.0
            dose_features.extend([has_last_dose, norm, age_norm])

        t_norm = float(self.t) / max(1.0, float(self.episode_length))
        t_norm = float(np.clip(t_norm, 0.0, 1.0))

        obs_parts = [
            last_count_norm,
            has_last_count,
            last_count_age_norm,
            *genome_values.tolist(),
            has_last_seq,
            last_seq_age_norm,
            measure_age_norm,
            *dose_features,
            t_norm,
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
    
    def disable_survival_bonus(self) -> None:
        """Disable survival bonus reward."""
        self.survival_bonus_reward = None
        print("✓ Survival bonus reward disabled")

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
    
    def get_action_mask(self, a_cont: np.ndarray) -> np.ndarray:
        """
        Compute action mask based on budget and continuous dose action.
        
        This implements Option C: continuous-dependent dose masking.
        The continuous dose is used to compute the exact cost of DOSE action,
        and DOSE is masked out if this specific dose is unaffordable.
        
        Args:
            a_cont: Continuous action (dose amounts), shape [K], values in [0,1]
        
        Returns:
            mask: Binary mask of shape [4], where:
                [0] = NOOP (always 1.0)
                [1] = COUNT (1.0 if affordable, 0.0 otherwise)
                [2] = SEQUENCING (1.0 if affordable, 0.0 otherwise)
                [3] = DOSE (1.0 if this specific dose is affordable, 0.0 otherwise)
        """
        mask = np.zeros(4, dtype=np.float32)
        
        # NOOP is always valid
        mask[ACTION_NOOP] = 1.0
        
        # COUNT validity (fixed cost)
        if self.budget >= self.count_cost:
            mask[ACTION_COUNT_BACTERIA] = 1.0
        
        # SEQUENCING validity (fixed cost)
        if self.budget >= self.sequencing_cost:
            mask[ACTION_SEQUENCING] = 1.0
        
        # DOSE validity (depends on the specific continuous action)
        # Clip continuous action to [0,1] and scale
        a_cont_clipped = np.clip(a_cont, 0.0, 1.0)
        scaled = self.scale_dose(a_cont_clipped)
        variable_cost = float(np.sum(scaled) * self.dose_cost_per_unit)
        total_dose_cost = self.dose_cost + variable_cost
        
        if self.budget >= total_dose_cost:
            mask[ACTION_DOSE] = 1.0
        
        # Safety: if all actions are invalid (shouldn't happen), force NOOP
        if np.sum(mask) == 0.0:
            mask[ACTION_NOOP] = 1.0
        
        return mask