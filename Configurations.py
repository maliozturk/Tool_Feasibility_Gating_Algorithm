# =============================================================================
#  TOOL FEASIBILITY GATING ALGORITHM (TFG)
#  Product Signature: TFG
# ------------------------------------------------------------------------------
#  File: Configurations.py
#  Purpose: Define configuration dataclasses for simulation components.
#  Author: Muhammet Ali Ozturk
#  Generated: 2026-01-18
#  Environment: Python 3.9.13
# =============================================================================

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Arrival_Config:

    lambda_rate_f64 : float
    max_tasks_i32_opt : Optional[int] = None


@dataclass(frozen=True)
class Ttl_Config:

    ttl_seconds_f64 : float
    ttl_std_f64 : float = 0.5
    ttl_min_f64 : float = 0.05

    def __post_init__(self) -> None:
        if self.ttl_seconds_f64 <= 0.0:
            raise ValueError("ttl_seconds_f64 must be > 0.")
        if self.ttl_std_f64 < 0.0:
            raise ValueError("ttl_std_f64 must be >= 0.")
        if self.ttl_min_f64 <= 0.0:
            raise ValueError("ttl_min_f64 must be > 0.")


@dataclass(frozen=True)
class Service_Config:

    slow_logn_mu_f64    : float
    slow_logn_sigma_f64 : float
    fast_logn_mu_f64    : float
    fast_logn_sigma_f64 : float

    service_time_source : str = "lognormal"                          
    trace_csv_path_str  : str = "Tool_Caller_Agent/trace_results_counterfactual.csv"
    trace_fast_column_str : str = "fast_generation_only_sec"
    trace_slow_column_str : str = "slow_generation_only_sec"
    trace_drop_error_rows_bool : bool = True
    trace_prompt_type_filter_opt : Optional[str] = None                               

    ewma_alpha_f64        : float = 0.10
    ewma_warmup_count_i32 : int = 500
    ewma_enabled_bool     : bool = False


@dataclass(frozen=True)
class Utility_Config:

    firm_deadline_bool         : bool  = True
    slow_success_utility_f64   : float = 1.0                         
    slow_success_std_f64       : float = 0.10
    slow_success_min_f64       : float = 0.0
    slow_success_max_f64       : Optional[float] = 1.2

    fast_success_utility_f64   : float = 0.5                         
    fast_success_std_f64       : float = 0.05
    fast_success_min_f64       : float = 0.0
    fast_success_max_f64       : Optional[float] = None

                                                                   
    high_priority_slow_success_utility_f64 : float = 1.2
    high_priority_fast_success_utility_f64 : float = 0.6

    missed_deadline_utility_f64: float = 0.0


@dataclass(frozen=True)
class Policy_Config:

    L_threshold_i32 : int   = 5
    alpha_f64       : float = 0.9

                                                
    enable_mode_switching_bool : bool  = False
    switch_check_quantum_f64   : float = 0.05
    switch_alpha_f64           : float = 0.95

                                                 
    static_mix_p: float = 0.5                                

                                                          
    queue_threshold_tau: int = 5

                                                                
    drift_V: float = 1.0                                              

                                                       
                                                                                                      
    tfg_slack_factor: float = 1.0                               
                                                                                                       
    tfg_epsilon_f64: float = 0.0

                                                                          
    tfg_adaptive_epsilon_enabled_bool: bool = False
    tfg_adaptive_epsilon_window_i32: int = 25
    tfg_adaptive_epsilon_kp_f64: float = 0.02
    tfg_adaptive_epsilon_ki_f64: float = 0.002
    tfg_adaptive_epsilon_min_f64: float = -.10
    tfg_adaptive_epsilon_max_f64: float =  .10

                                      
                                                                      
                                                                   
    tfg_wait_estimator: str = "mix"                           

                                              
    tfg_queue_slow_mix_p: float = 0.55                                                     

                                                                            
    tfg_include_in_service: bool = True

                                            
    tfg_trace_enabled_bool: bool = False

                                               
    tfg_quantile_q_f64: float = 0.9



@dataclass(frozen=True)
class Simulation_Config:

    seed_i32                      : int   = 1453
    until_time_f64                : float = 100_000.0
    warmup_time_f64               : float = 1000.0
    drop_expired_in_queue_bool    : bool  = True
    priority_task_rate_f64        : float = 0.10
    priority_first_enabled_bool   : bool  = True

    arrival_config : Arrival_Config = Arrival_Config(
        lambda_rate_f64=0.05,
        max_tasks_i32_opt=None,
    )

    ttl_ttl_config : Ttl_Config = Ttl_Config(
        ttl_seconds_f64=35.0,
        ttl_std_f64=10.0,
        ttl_min_f64=0.05,
    )

                                                                                  
    service_config : Service_Config = Service_Config(
        slow_logn_mu_f64=3.065581979824905,
        slow_logn_sigma_f64=0.3492635134873723,
        fast_logn_mu_f64=-0.17131692726202313,
        fast_logn_sigma_f64=0.748885523221923,
        service_time_source="trace",
    )

    utility_config : Utility_Config = Utility_Config()
    policy_config   : Policy_Config  = Policy_Config()

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.priority_task_rate_f64) <= 1.0):
            raise ValueError("priority_task_rate_f64 must be in [0, 1].")


"""
Notes (implementation choices embedded in config):

Service times use lognormal to avoid negative durations and to model heavy tails (typical in inference latency).

warmup_time is included for steady-state estimates (common in academic simulation studies).
"""
