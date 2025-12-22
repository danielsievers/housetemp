import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize
from .run_model import run_model_continuous
from .utils import upsample_dataframe, get_effective_hvac_state
from .energy import calculate_energy_vectorized

_LOGGER = logging.getLogger(__name__)

# --- CONFIGURATION (Tuning Parameters) ---
# Continuity penalty guides solver towards integer setpoints
SNAP_REG_WEIGHT = 0.1  # Canonical value for continuity penalty

try:
    from .constants import (
        DEFAULT_SWING_TEMP,
        DEFAULT_MIN_CYCLE_MINUTES,
        DEFAULT_OFF_INTENT_EPS,
        DEFAULT_MIN_SETPOINT,
        DEFAULT_MAX_SETPOINT,
        DEFAULT_EFFICIENCY_DERATE
    )
except ImportError:
    # Fallbacks for standalone usage
    DEFAULT_SWING_TEMP = 1.0
    DEFAULT_MIN_CYCLE_MINUTES = 15.0
    DEFAULT_OFF_INTENT_EPS = 0.1
    DEFAULT_MIN_SETPOINT = 60.0
    DEFAULT_MAX_SETPOINT = 75.0
    DEFAULT_EFFICIENCY_DERATE = 0.75

# --- OPTIMIZER-SPECIFIC DEFAULTS ---
DEFAULT_CENTER_PREFERENCE = 1.0  # User preference for hitting the exact target
DEFAULT_DEADBAND_SLACK = 1.5     # Degrees of freedom without penalty
DEFAULT_COMFORT_MODE = 'quadratic'
DEFAULT_AVOID_DEFROST = False

# Optimization Solver Defaults
DEFAULT_SOLVER_MAXITER = 500
DEFAULT_SOLVER_FTOL = 1e-4
DEFAULT_SOLVER_GTOL = 1e-4
DEFAULT_SOLVER_EPS = 0.5  # Large step size to jump over deadbands

# Coarse Pass Defaults (Multi-Scale)
DEFAULT_COARSE_BLOCK_SIZE = 120  # Minutes
DEFAULT_COARSE_MAXITER = 50
DEFAULT_COARSE_FTOL = 1e-2
DEFAULT_COARSE_GTOL = 1e-2
DEFAULT_COARSE_EPS = 1.0

# --- TRUE CONSTANTS (Physical/Mathematical) ---
# Parameter Optimization Bounds
# See Design.md Section 4 (Parameter Optimization)
BOUNDS_MASS_C = (4000, 20000)      # BTU/F
BOUNDS_UA = (200, 1500)            # BTU/hr/F
BOUNDS_UA_ACTIVE = (200, 600)      # Tighter bounds for active optimization
BOUNDS_K_SOLAR = (700, 2000)       # BTU/hr per kW/m^2
BOUNDS_Q_INT = (200, 10000)        # BTU/hr
BOUNDS_H_FACTOR = (5000, 20000)    # BTU/hr/F (Aggressiveness)
BOUNDS_H_FACTOR_ACTIVE = (2000, 30000) # Wider bounds for active
BOUNDS_EFFICIENCY = (0.5, 1.2)     # Derate factor

# Initial Guesses
GUESS_FULL_OPT = [4732, 213, 836, 600, 10000] # [C, UA, K, Q, H]
GUESS_ACTIVE_H = 10000
GUESS_ACTIVE_EFF = 0.9

# Soft Constraint Penalties (Physics Violations)
PENALTY_PHYSICS_VIOLATION = 1e6
MIN_MASS_C = 1000
MIN_UA = 50

_LOGGER = logging.getLogger(__name__)

# No change to loss_function needed as it calls run_model wrapper which we updated?
# Wait, loss_function calls run_model.run_model. We updated the wrapper signature to have defaults.
# But does run_model wrapper logic handle the default args?
# Yes, `run_model` definition I wrote has `swing_temp=1.0, min_cycle_minutes=15` defaults.
# So loss_function is safe if it doesn't pass them (uses defaults).

def loss_function(active_params, data, hw, fixed_passive_params=None, fixed_efficiency_derate=None):
    # Construct full params vector
    if fixed_passive_params:
        # fixed_passive_params comes in as [C, UA, K, Q]
        # active_params = [UA, H_factor] (efficiency_derate is FIXED)
        
        ua_opt = active_params[0]
        h_factor = active_params[1]
        
        # Fixed Derate (Duct Efficiency) - NOT OPTIMIZED
        # Use value passed from run_optimization (which comes from file or default)
        eff_derate = float(fixed_efficiency_derate if fixed_efficiency_derate is not None else DEFAULT_EFFICIENCY_DERATE)
        
        # Construct using fixed C, K, Q
        c = fixed_passive_params[0]
        # ua is optimized
        k = fixed_passive_params[2]
        q = fixed_passive_params[3]
        
        full_params = [c, ua_opt, k, q, h_factor, eff_derate]
    else:
        # Optimizing EVERYTHING
        # active_params = [C, UA, K_solar, Q_int, H_factor]
        # efficiency_derate is appended as fixed constant
        eff_derate = float(fixed_efficiency_derate if fixed_efficiency_derate is not None else DEFAULT_EFFICIENCY_DERATE)
        full_params = list(active_params) + [eff_derate]

    # 1. Run Simulation
    # Note: run_model returns 5 values now (temps, rmse, delivered, produced, actual_state)
    # But loss_function only needs error.
    # We should unpack carefully.
    # Unpack for continuous model
    max_caps_list = hw.get_max_capacity(data.t_out).tolist() if hw else [0.0]*len(data.t_out)
    
    sim_temps, _, _ = run_model_continuous(
        full_params,
        t_out_list=data.t_out.tolist(),
        solar_kw_list=data.solar_kw.tolist(),
        dt_hours_list=data.dt_hours.tolist(),
        setpoint_list=data.setpoint.tolist(),
        hvac_state_list=data.hvac_state.tolist(),
        max_caps_list=max_caps_list,
        min_output=hw.min_output_btu_hr if hw else 0,
        max_cool=hw.max_cool_btu_hr if hw else 0,
        eff_derate=full_params[5] if len(full_params) > 5 else 1.0,
        start_temp=float(data.t_in[0])
    )
    
    # Calculate RMSE manually
    sim_temps_arr = np.array(sim_temps)
    actual_temps = data.t_in[:len(sim_temps_arr)]
    mse = np.mean((sim_temps_arr - actual_temps)**2)
    sim_error = np.sqrt(mse)
    
    # 2. Compare to Reality (RMSE)
    error = sim_error
    
    # 3. Penalize Physics Violations (Soft Constraints)
    # If fixing passive params, we don't check them here (assumed good)
    if not fixed_passive_params:
        if full_params[0] < MIN_MASS_C: return PENALTY_PHYSICS_VIOLATION # Mass too low
        if full_params[1] < MIN_UA: return PENALTY_PHYSICS_VIOLATION   # UA too low
    
    return error

def run_optimization(data, hw, initial_guess=None, fixed_passive_params=None, fixed_efficiency_derate=None):
    # hw is now passed in
    
    if fixed_passive_params:
        print(f"--- ACTIVE PARAMETER OPTIMIZATION MODE ---")
        print("Optimizing Active Parameters + Floating UA (Occupied Infiltration).")
         # Initial Guess for [UA, H_factor]
         # Start UA at fixed value, H at 20k
        initial_guess = [fixed_passive_params[1], GUESS_ACTIVE_H]
        
        bounds = [
            BOUNDS_UA_ACTIVE,
            BOUNDS_H_FACTOR_ACTIVE
        ]
    else:
        # Full Optimization
        # [C, UA, K_solar, Q_int, H_factor]
        if initial_guess is None:
            initial_guess = GUESS_FULL_OPT
        
        bounds = [
            BOUNDS_MASS_C,
            BOUNDS_UA,
            BOUNDS_K_SOLAR,
            BOUNDS_Q_INT,
            BOUNDS_H_FACTOR
        ]
        
    print("Starting Optimization...")
    eff = float(fixed_efficiency_derate if fixed_efficiency_derate is not None else DEFAULT_EFFICIENCY_DERATE)
    
    result = minimize(
        loss_function, 
        initial_guess, 
        args=(data, hw, fixed_passive_params, eff), 
        method='L-BFGS-B', 
        bounds=bounds,
        options={'disp': True}
    )

    if result.success:
        # Reconstruct full parameter set for return
        best_active = result.x
        
        if fixed_passive_params:
            # best_active = [UA, H]
            ua = best_active[0]
            h = best_active[1]
            c = fixed_passive_params[0]
            k = fixed_passive_params[2]
            q = fixed_passive_params[3]
             
            full_params = [c, ua, k, q, h, eff]
            
            # Result type needs to mock the scipy result object or we just modify result.x
            result.x = np.array(full_params)
        else:
            # Full Optimization
            # best_active = [C, UA, K, Q, H]
            
            # Append fixed efficiency for Result (restoring 6-param structure)
            full_params = list(best_active) + [eff]
            result.x = np.array(full_params)
            
    return result

def optimize_hvac_schedule(data, params, hw, target_temps, comfort_config, block_size_minutes=30, fixed_mask=None, enable_multiscale=True, rate_per_step=None):
    """
    Finds the optimal setpoint schedule to minimize Energy Cost + Comfort Penalty.
    
    Args:
        data, params, hw, target_temps, comfort_config: Core optimization inputs.
        block_size_minutes: Optimization resolution (e.g. 30).
        fixed_mask: Optional boolean mask for mandatory setpoints.
        enable_multiscale: If True, runs a coarse pass first for warm-start.
        rate_per_step: Optional array of TOU rate scale factors (e.g. 1.0 = baseline, 1.3 = peak).
                       Must be aligned 1:1 with data.timestamps.
                       If provided, optimizer minimizes total cost: sum(kWh_i * rate_i).
                       If rate_per_step is unitless (relative multipliers), cost is weighted-kWh.
                       If rate_per_step is currency ($/kWh), cost is in dollars.
    
    Returns:
        (final_setpoints, debug_info)
        where debug_info is a dict: {'success': bool, 'cost': float, 'message': str, 'iterations': int}
    """
    
    # Force Mode based on Config
    mode_str = comfort_config.get('mode', '').lower()
    if mode_str == 'heat':
        hvac_mode_val = 1
    elif mode_str == 'cool':
        hvac_mode_val = -1
    else:
        raise ValueError("Comfort config must specify 'mode': 'heat' or 'cool'. Auto mode is no longer supported.")
        
    # --- Defensive Copy Pattern (Exception-Safe) ---
    # Instead of mutating data.hvac_state, create a working copy.
    # This eliminates the need for try/finally restoration.
    working_hvac_state = np.full_like(data.hvac_state, hvac_mode_val, dtype=data.hvac_state.dtype)
    working_hvac_state_list = working_hvac_state.tolist()
    
    # --- TOU Robustness Check ---
    if rate_per_step is not None:
        if len(rate_per_step) != len(data.timestamps):
            _LOGGER.warning(f"TOU rate alignment mismatch: Expected {len(data.timestamps)} steps, got {len(rate_per_step)}. Falling back to uniform rate.")
            rate_per_step = None
        else:
            # Ensure it is a numpy array for vectorized math
            rate_per_step = np.array(rate_per_step)
            
            # --- Robustness Value Checks ---
            if np.any(rate_per_step < 0):
                _LOGGER.error("TOU rates cannot be negative. Clipping to 0.")
                rate_per_step = np.maximum(0, rate_per_step)
            
            if not np.all(np.isfinite(rate_per_step)):
                _LOGGER.error("TOU rates contain NaN or Inf. Falling back to uniform rate.")
                rate_per_step = None
    
    # Fallback: If no explicit fixed_mask passed, check data object
    if fixed_mask is None and data.is_setpoint_fixed is not None:
         fixed_mask = data.is_setpoint_fixed
    
    # --- 1. PRE-EXTRACT RAW ARRAYS (Hoisting) ---
    # Convert everything to Python lists ONCE for the kernel
    t_out_list = data.t_out.tolist()
    solar_kw_list = data.solar_kw.tolist()
    dt_hours_list = data.dt_hours.tolist()
    
    # Pre-calculate hardware limits (List format for kernel)
    max_caps_list = hw.get_max_capacity(data.t_out).tolist()
    min_output = hw.min_output_btu_hr
    max_cool = hw.max_cool_btu_hr
    
    # For Cost Calculation (Vectorized NumPy still fastest for arrays outside the loop)
    # We keep these as numpy arrays for the cost function math
    max_caps_np = hw.get_max_capacity(data.t_out)
    if hvac_mode_val < 0:
        base_cops_np = hw.get_cooling_cop(data.t_out)
    else:
        base_cops_np = hw.get_cop(data.t_out)
    dt_hours_np = data.dt_hours
    
    # Shared config & Vectorized Time Mapping
    start_ts = pd.Timestamp(data.timestamps[0])
    
    # Vectorized timestamp conversion (User optimization)
    dt_arr = data.dt_hours
    if len(dt_arr) > 1 and np.max(dt_arr) - np.min(dt_arr) < 1e-6:
        step_min = dt_arr[0] * 60.0
        sim_minutes = np.arange(len(data.timestamps)) * step_min
    else:
        ts_series = pd.to_datetime(data.timestamps)
        sim_minutes = (ts_series - start_ts).total_seconds().values / 60.0
    
    # Unpack Optimization Params (Needed for kernel)
    eff_derate = DEFAULT_EFFICIENCY_DERATE
    if len(params) > 5:
        eff_derate = params[5]
    start_temp = float(data.t_in[0])
    
    # Hoist setpoint bounds for use in both _run_pass and post-processing
    min_setpoint = comfort_config.get('min_setpoint', DEFAULT_MIN_SETPOINT)
    max_setpoint = comfort_config.get('max_setpoint', DEFAULT_MAX_SETPOINT)

    def _run_pass(current_block_size, initial_guess_blocks=None, optimization_options=None):
        """Helper to run a single optimization pass at a specific resolution."""
        print(f"  Running Optimization Pass ({current_block_size}-min blocks)...")
        
        # --- Setup Control Blocks ---
        freq = f"{current_block_size}min"
        # Align end to cover the full range
        end_ts = pd.Timestamp(data.timestamps[-1])
        aligned_start = start_ts.floor(freq)
        aligned_end = end_ts.ceil(freq)
        
        control_ts = pd.date_range(start=aligned_start, end=aligned_end, freq=freq)
        control_times = (control_ts - start_ts).total_seconds().values / 60.0
        num_blocks = len(control_times)
        
        # --- Hoisted Upsampling Map ---
        sim_to_block_map = np.searchsorted(control_times, sim_minutes, side='right') - 1
        sim_to_block_map = np.clip(sim_to_block_map, 0, len(control_times) - 1)
        
        # --- Initial Guess ---
        if initial_guess_blocks is None:
            # Downsample target temps
            current_guess = np.interp(control_times, sim_minutes, target_temps)
        else:
            current_guess = initial_guess_blocks

        # --- Fixed Constraints ---
        block_fixed_flags = np.zeros(num_blocks, dtype=bool)
        if fixed_mask is not None:
             indices = np.searchsorted(sim_minutes, control_times, side='left')
             indices = np.clip(indices, 0, len(fixed_mask) - 1)
             block_fixed_flags = fixed_mask[indices]
             current_guess[block_fixed_flags] = np.interp(control_times[block_fixed_flags], sim_minutes, target_temps)

        # --- Bounds ---
        min_setpoint = comfort_config.get('min_setpoint', DEFAULT_MIN_SETPOINT)
        max_setpoint = comfort_config.get('max_setpoint', DEFAULT_MAX_SETPOINT)
        bounds = []
        for i in range(num_blocks):
            if block_fixed_flags[i]:
                val = current_guess[i]
                bounds.append((val, val))
            else:
                bounds.append((min_setpoint, max_setpoint))
        
        # --- Loss Function ---
        # Dual-weight CP mapping: single user knob u∈[0,1] maps to two weights
        # cp_outside: penalty for leaving deadband (strong at high u)
        # cp_inside: pull toward target within deadband (moderate at high u)
        u = np.clip(comfort_config.get('center_preference', DEFAULT_CENTER_PREFERENCE), 0.0, 1.0)
        cp_outside = 0.1 + 49.9 * (u ** 3)   # Range: 0.1 (eco) → 50.0 (comfort)
        cp_inside = 5.0 * (u ** 2)              # Range: 0.0 (eco) → 5.0 (comfort)
        
        comfort_mode = comfort_config.get('comfort_mode', DEFAULT_COMFORT_MODE)
        deadband_slack = comfort_config.get('deadband_slack', DEFAULT_DEADBAND_SLACK)
        avoid_defrost = comfort_config.get('avoid_defrost', DEFAULT_AVOID_DEFROST)
        
        # Thermostat Parity Config
        # Hardcoding defaults for now as they are not yet in comfort_config
        # But should probably pull from there if available
        # Using const defaults
        swing_temp = DEFAULT_SWING_TEMP
        min_cycle_minutes = DEFAULT_MIN_CYCLE_MINUTES


        def schedule_loss(candidate_blocks):
            # 1. Update Setpoints via Hoisted Map (Fast)
            full_res_setpoints = candidate_blocks[sim_to_block_map]
            
            # --- Late Rounding Strategy ---
            # We use RAW setpoints for physics simulation to maintain gradients.
            # However, for True-Off accounting (Idle/Blower detection), we use 
            # quantized (effective_setpoints) to match real thermostat behavior.
            effective_setpoints = np.round(full_res_setpoints)
            
            # Continuity Penalty (L2 on Rounding Error): 
            # Nudges the solver towards integer setpoints during the search to 
            # minimize discrepancy between the "gradient-friendly" continuous 
            # simulation and the "reality" of a discrete thermostat.
            continuity_penalty = SNAP_REG_WEIGHT * np.sum((full_res_setpoints - effective_setpoints)**2)
            
            # Physics sees continuous values for gradients
            setpoint_list = full_res_setpoints.tolist()
            
            # --- Use Raw Intent for Physics & Energy ---
            # We no longer "soft gate" hvac_state in the optimizer.
            # We pass the raw intent (+1/-1) and let energy.py handle "True Off" accounting based on setpoints.
            hvac_state_list = working_hvac_state_list  # Use working copy (not data.hvac_state)

            # 2. Run THIN Kernel
            start_temp = float(data.t_in[0]) # Ensure start_temp is float

            # --- Upstream True Off (for Optimization Loop) ---
            # We calculate this DYNAMICALLY inside the loop using helper
            effective_hvac_state = get_effective_hvac_state(
                working_hvac_state, 
                effective_setpoints, 
                min_setpoint, 
                max_setpoint, 
                DEFAULT_OFF_INTENT_EPS
            )
            
            # 3. Run Physics Model (Continuous)
            # Pass effective_hvac_state (0 for True Off) so physics naturally shuts down.
            sim_temps_list, hvac_outputs_list, hvac_produced_list = run_model_continuous(
                params, 
                t_out_list=t_out_list, 
                solar_kw_list=solar_kw_list, 
                dt_hours_list=dt_hours_list, 
                setpoint_list=setpoint_list, 
                hvac_state_list=effective_hvac_state.tolist(),
                max_caps_list=max_caps_np.tolist(), 
                min_output=min_output, 
                max_cool=max_cool, 
                eff_derate=eff_derate, 
                start_temp=start_temp
            )
            
            sim_temps = np.array(sim_temps_list)
            hvac_produced = np.array(hvac_produced_list)
            
            # --- 4. Calculate Costs ---
            
            # A. Comfort Cost (RMSE from Preference Curve)
            
            # B. Energy Cost
            # Pass effective_hvac_state so energy calculator sees 0 enabled, charging 0 idle.
            res = calculate_energy_vectorized(
                hvac_produced, 
                dt_hours_np, 
                max_caps_np, 
                base_cops_np, 
                hw, 
                eff_derate=1.0, # Produced is Gross (Pre-Derate)
                hvac_states=effective_hvac_state
            )
            
            if rate_per_step is not None:
                kwh_steps = res['kwh_steps']
                energy_cost = np.sum(kwh_steps * rate_per_step)
            else:
                energy_cost = res['kwh']  # Fallback to pure kWh minimization
            
            load_ratios = res['load_ratios']
            
            # Comfort Cost: Two separate terms (dual-weight approach)
            # - outside_cost: penalty for violating deadband floor/ceiling (strong constraint)
            # - inside_cost: pull toward target within deadband (preference)
            safe_slack = max(deadband_slack, 1e-6)  # Guard against divide-by-zero
            
            if hvac_mode_val > 0:  # Heating
                if comfort_mode == 'deadband':
                    floor = target_temps - deadband_slack
                    # Outside violation (too cold)
                    outside_errors = np.minimum(0, sim_temps - floor)
                    # Inside gap (normalized 0-1: 0 at target, 1 at floor)
                    inside_gap = np.clip(target_temps - sim_temps, 0, deadband_slack)
                    inside_norm = inside_gap / safe_slack
                    # Zero out inside term when outside band
                    inside_norm = np.where(outside_errors < 0, 0, inside_norm)
                else:
                    outside_errors = np.minimum(0, sim_temps - target_temps)
                    inside_norm = np.zeros_like(sim_temps)
            else:  # Cooling
                if comfort_mode == 'deadband':
                    ceiling = target_temps + deadband_slack
                    # Outside violation (too warm)
                    outside_errors = np.maximum(0, sim_temps - ceiling)
                    # Inside gap (normalized 0-1: 0 at target, 1 at ceiling)
                    inside_gap = np.clip(sim_temps - target_temps, 0, deadband_slack)
                    inside_norm = inside_gap / safe_slack
                    # Zero out inside term when outside band
                    inside_norm = np.where(outside_errors > 0, 0, inside_norm)
                else:
                    outside_errors = np.maximum(0, sim_temps - target_temps)
                    inside_norm = np.zeros_like(sim_temps)

            # Comfort cost per step (two separate terms)
            outside_cost_term = cp_outside * (outside_errors ** 2)
            inside_cost_term = cp_inside * (inside_norm ** 2)
            comfort_cost = outside_cost_term + inside_cost_term
            
            total_penalty = np.sum(comfort_cost * dt_hours_np)
            
            # Defrost
            defrost_cost = 0.0
            if hasattr(hw, 'defrost_risk_zone') and hw.defrost_risk_zone is not None:
                risk_min, risk_max = hw.defrost_risk_zone
                duration_hr = hw.defrost_duration_min / 60.0
                interval_hr = hw.defrost_interval_min / 60.0
                power_kw = hw.defrost_power_kw
                in_frost_zone = (data.t_out >= risk_min) & (data.t_out <= risk_max)
                runtime_in_zone = np.sum(load_ratios * in_frost_zone * dt_hours_np)
                expected_cycles = runtime_in_zone / interval_hr
                defrost_cost_val = expected_cycles * duration_hr * power_kw
                if avoid_defrost:
                    defrost_cost_val *= 10.0
                defrost_cost = defrost_cost_val

            return energy_cost + total_penalty + defrost_cost + continuity_penalty


        # --- Run Minimize ---
        # Settings for optimization
        if optimization_options:
             opts = optimization_options
        else:
             opts = {
                 'disp': False, 
                 'maxiter': DEFAULT_SOLVER_MAXITER,  
                 'ftol': DEFAULT_SOLVER_FTOL,    
                 'gtol': DEFAULT_SOLVER_GTOL,    
                 'eps': DEFAULT_SOLVER_EPS       
             }
        
        result = minimize(
            schedule_loss,
            current_guess,
            bounds=bounds,
            method='L-BFGS-B',
            options=opts
        )
        return result, control_times
    
    # === Main Optimization Logic (Multi-Scale) ===
    
    if enable_multiscale:
        # Pass 1: Coarse (120-min blocks)
        coarse_block_size = DEFAULT_COARSE_BLOCK_SIZE
        if block_size_minutes >= coarse_block_size:
            print("Skipping Multi-Scale: Target resolution is already coarse.")
            final_result, final_times = _run_pass(block_size_minutes)
        else:
            print("Strategy: Multi-Scale (Coarse -> Fine)")
            
            # Coarse Options
            coarse_opts = {
                'disp': False,
                'maxiter': DEFAULT_COARSE_MAXITER,
                'ftol': DEFAULT_COARSE_FTOL,
                'gtol': DEFAULT_COARSE_GTOL,
                'eps': DEFAULT_COARSE_EPS
            }
            coarse_res, coarse_times = _run_pass(coarse_block_size, optimization_options=coarse_opts)
            
            # Pass 2: Fine (Warm Start)
            # Create Fine Times needed for interpolation
            freq = f"{block_size_minutes}min"
            end_ts = pd.Timestamp(data.timestamps[-1])
            aligned_start = start_ts.floor(freq)
            aligned_end = end_ts.ceil(freq)
            fine_ts = pd.date_range(start=aligned_start, end=aligned_end, freq=freq)
            fine_times = (fine_ts - start_ts).total_seconds().values / 60.0
            
            # Check success OR abnormal termination (often usable)
            msg = str(coarse_res.message)
            if coarse_res.success or "ABNORMAL" in msg:
                if not coarse_res.success:
                    print(f"Note: Coarse pass finished with warning ({msg}), but using result for warm start.")
                warm_start_guess = np.interp(fine_times, coarse_times, coarse_res.x)
            else:
                print(f"Warning: Coarse pass failed ({coarse_res.message}). Using cold start for fine pass.")
                warm_start_guess = None
            
            # Fine Options (Precision)
            # Re-use default solver settings
            fine_opts = {
                'disp': False,
                'maxiter': DEFAULT_SOLVER_MAXITER,
                'ftol': DEFAULT_SOLVER_FTOL,
                'gtol': DEFAULT_SOLVER_GTOL,
                'eps': DEFAULT_SOLVER_EPS
            }
            final_result, final_times = _run_pass(block_size_minutes, initial_guess_blocks=warm_start_guess, optimization_options=fine_opts)
    else:
        # Legacy Single Pass
        print("Strategy: Single-Scale (Cold Start)")
        legacy_opts = {
            'disp': False,
            'maxiter': DEFAULT_SOLVER_MAXITER,
            'ftol': DEFAULT_SOLVER_FTOL,
            'gtol': DEFAULT_SOLVER_GTOL,
            'eps': DEFAULT_SOLVER_EPS
        }
        final_result, final_times = _run_pass(block_size_minutes, optimization_options=legacy_opts)

    # --- Final Output Processing ---
    result = final_result
    control_times = final_times
    
    if not result.success:
        msg = result.message.decode('utf-8') if isinstance(result.message, bytes) else str(result.message)
        if "ABNORMAL" in msg:
             print(f"Optimization Finished with Warning: {msg}")
        else:
             print(f"Optimization Failed: {msg!r}")
    else:
        print(f"Optimization Converged Successfully (Cost: {result.fun:.4f})")
    
    # Return Upsampled Schedule
    final_blocks = np.round(result.x) # Round to nearest integer (thermostats don't do floats)
    indices = np.searchsorted(control_times, sim_minutes, side='right') - 1
    indices = np.clip(indices, 0, len(final_blocks) - 1)
    
    final_setpoints = final_blocks[indices]
    
    # --- Post-Processing: Snap-to-Boundary (Visual/UI & True-Off Consistency) ---
    # Instead of a wide "Target - X" heuristic (which overrides legitimate setbacks),
    # we use a local "hover" snap. If the solver gets close to the min/max (within 0.5F),
    # we snap it exactly to the limit to ensure "True Off" logic triggers downstream.
    SNAP_EPS = 0.5
    
    if hvac_mode_val > 0:  # Heating
        # If hovering near min_setpoint
        is_near_min = final_setpoints <= (min_setpoint + SNAP_EPS)
        final_setpoints = np.where(is_near_min, min_setpoint, final_setpoints)
    else:  # Cooling
        # If hovering near max_setpoint
        is_near_max = final_setpoints >= (max_setpoint - SNAP_EPS)
        final_setpoints = np.where(is_near_max, max_setpoint, final_setpoints)
    
    # Metadata
    debug_info = {
        'success': result.success,
        'message': str(result.message),
        'cost': float(result.fun),
        'iterations': result.nit,
        'evaluations': result.nfev
    }
    
    # Check for "ABNORMAL" (Deadband success)
    msg_str = str(result.message)
    if not result.success and "ABNORMAL" in msg_str:
        debug_info['success'] = True # Weak success
        debug_info['message'] = "Converged (Deadband)"

    # Strict Failure Handling
    if not debug_info['success']:
        _LOGGER.error(f"Optimization FAILED: {debug_info['message']}")
        return None, debug_info

    return final_setpoints, debug_info
