import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize
from .run_model import run_model_continuous
from .energy import calculate_energy_vectorized

_LOGGER = logging.getLogger(__name__)

# --- CONFIGURATION (Tuning Parameters) ---
CONFIG_SNAP_WEIGHT = 0.01  # Incentivizes closing a 2F gap (0.16 cost) vs. paying 0.15kW idle power
CONFIG_CONTINUITY_WEIGHT = 0.0001  # Small penalty to guide solver towards integer setpoints

try:
    from ..const import (
        DEFAULT_SWING_TEMP,
        DEFAULT_MIN_CYCLE_MINUTES,
    )
except (ImportError, ValueError):
    # Fallback when running as standalone library (parent const unreachable)
    DEFAULT_SWING_TEMP = 1.0
    DEFAULT_MIN_CYCLE_MINUTES = 15


# --- DEFAULT OVERRIDES (Fallbacks) ---
DEFAULT_EFFICIENCY_DERATE = 1.0
DEFAULT_MIN_SETPOINT = 60.0
DEFAULT_MAX_SETPOINT = 75.0
DEFAULT_CENTER_PREFERENCE = 1.0  # User preference for hitting the exact target
DEFAULT_DEADBAND_SLACK = 1.5  # Degrees of freedom without penalty
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

def loss_function(active_params, data, hw, fixed_passive_params=None):
    # Construct full params vector
    if fixed_passive_params:
        # fixed_passive_params comes in as [C, UA, K, Q]
        # active_params = [UA, H_factor, efficiency_derate]
        
        ua_opt = active_params[0]
        h_factor = active_params[1]
        eff_derate = active_params[2] if len(active_params) > 2 else DEFAULT_EFFICIENCY_DERATE
        
        # Construct using fixed C, K, Q
        c = fixed_passive_params[0]
        # ua is optimized
        k = fixed_passive_params[2]
        q = fixed_passive_params[3]
        
        full_params = [c, ua_opt, k, q, h_factor, eff_derate]
    else:
        # Optimizing EVERYTHING
        # active_params = [C, UA, K_solar, Q_int, H_factor]
        # We need to add a default efficiency_derate for the run_model call
        full_params = list(active_params) + [DEFAULT_EFFICIENCY_DERATE] # Default efficiency_derate

    # 1. Run Simulation
    # Note: run_model returns 5 values now (temps, rmse, delivered, produced, actual_state)
    # But loss_function only needs error.
    # We should unpack carefully.
    outputs = run_model.run_model(full_params, data, hw)
    sim_error = outputs[1] # RMSE is 2nd return
    
    # 2. Compare to Reality (RMSE)
    error = sim_error
    
    # 3. Penalize Physics Violations (Soft Constraints)
    # If fixing passive params, we don't check them here (assumed good)
    if not fixed_passive_params:
        if full_params[0] < MIN_MASS_C: return PENALTY_PHYSICS_VIOLATION # Mass too low
        if full_params[1] < MIN_UA: return PENALTY_PHYSICS_VIOLATION   # UA too low
    
    return error

def run_optimization(data, hw, initial_guess=None, fixed_passive_params=None):
    # hw is now passed in
    
    if fixed_passive_params:
        print(f"--- ACTIVE PARAMETER OPTIMIZATION MODE ---")
        print("Optimizing Active Parameters + Floating UA (Occupied Infiltration).")
         # Initial Guess for [UA, H_factor, efficiency_derate]
         # Start UA at fixed value, H at 10k, Eff at 0.9
        initial_guess = [fixed_passive_params[1], GUESS_ACTIVE_H, GUESS_ACTIVE_EFF]
        
        bounds = [
            BOUNDS_UA_ACTIVE,
            BOUNDS_H_FACTOR_ACTIVE,
            BOUNDS_EFFICIENCY
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
    result = minimize(
        loss_function, 
        initial_guess, 
        args=(data, hw, fixed_passive_params), 
        bounds=bounds, 
        method='L-BFGS-B'
    )
    
    if fixed_passive_params:
        # Reconstruct full parameter set for return
        best_active = result.x
        
        # best_active = [UA, H, Eff]
        ua = best_active[0]
        h = best_active[1]
        eff = best_active[2]
         
        c = fixed_passive_params[0]
        k = fixed_passive_params[2]
        q = fixed_passive_params[3]
         
        full_params = [c, ua, k, q, h, eff]
        
        # Result type needs to mock the scipy result object or we just modify result.x
        result.x = np.array(full_params)
        
    return result

def optimize_hvac_schedule(data, params, hw, target_temps, comfort_config, block_size_minutes=30, fixed_mask=None, enable_multiscale=True):
    """
    Finds the optimal setpoint schedule to minimize Energy + Comfort Penalty.
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
        
    original_hvac_state = data.hvac_state.copy()
    data.hvac_state[:] = hvac_mode_val
    
    # Fallback: If no explicit fixed_mask passed, check data object
    if fixed_mask is None and data.is_setpoint_fixed is not None:
         fixed_mask = data.is_setpoint_fixed
    
    # --- 1. PRE-EXTRACT RAW ARRAYS (Hoisting) ---
    # Convert everything to Python lists ONCE for the kernel
    t_out_list = data.t_out.tolist()
    solar_kw_list = data.solar_kw.tolist()
    dt_hours_list = data.dt_hours.tolist()
    # hvac_state_list = data.hvac_state.tolist() # REMOVED: Now derived inside loop
    
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
        user_preference = comfort_config.get('center_preference', DEFAULT_CENTER_PREFERENCE)
        center_preference = 0.1 + (user_preference * 5.0)
        comfort_mode = comfort_config.get('comfort_mode', DEFAULT_COMFORT_MODE)
        deadband_slack = comfort_config.get('deadband_slack', DEFAULT_DEADBAND_SLACK)
        avoid_defrost = comfort_config.get('avoid_defrost', DEFAULT_AVOID_DEFROST)
        
        # Thermostat Parity Config
        # Hardcoding defaults for now as they are not yet in comfort_config
        # But should probably pull from there if available
        # Using const defaults
        swing_temp = DEFAULT_SWING_TEMP
        min_cycle_minutes = DEFAULT_MIN_CYCLE_MINUTES
        
        # Tuning weights
        snap_weight = CONFIG_SNAP_WEIGHT
        
        # Pre-calc constants for Gate Derivation
        GATE_EPSILON = 0.1 # Distance from Min/Max to trigger "Off" intent
        GATE_STEEPNESS = 4.0 # Softness of the gate

        def schedule_loss(candidate_blocks):
            # 1. Update Setpoints via Hoisted Map (Fast)
            full_res_setpoints = candidate_blocks[sim_to_block_map]
            
            # --- Late Rounding Strategy ---
            effective_setpoints = np.round(full_res_setpoints)
            continuity_penalty = CONFIG_CONTINUITY_WEIGHT * np.sum((full_res_setpoints - effective_setpoints)**2)
            setpoint_list = effective_setpoints.tolist()
            
            # --- Derive "True Off" Gate from Setpoints ---
            # If setpoint is at the boundary (min for heat, max for cool), we assume "Off Intent".
            # We map this to hvac_state -> 0 (softly).
            # This allows the optimizer to kill Idle Power by picking the boundary.
            if hvac_mode_val > 0: # Heating
                # "Off" is setpoint <= min_setpoint
                dist = full_res_setpoints - min_setpoint
                # Sigmoid-like gate: 1 when dist is high, 0 when dist is 0
                # Use simple linear ramp for speed and gradient
                gate = np.clip(dist * GATE_STEEPNESS, 0.0, 1.0)
            else: # Cooling
                # "Off" is setpoint >= max_setpoint
                dist = max_setpoint - full_res_setpoints
                gate = np.clip(dist * GATE_STEEPNESS, 0.0, 1.0)
            
            effective_hvac_states = gate * hvac_mode_val
            hvac_state_list = effective_hvac_states.tolist()

            # 2. Run THIN Kernel
            start_temp = float(data.t_in[0]) # Ensure start_temp is float

            # --- 3. Run Physics Model (Continuous) ---
            sim_temps_list, hvac_outputs_list, hvac_produced_list = run_model_continuous(
                params, 
                t_out_list, 
                solar_kw_list, 
                dt_hours_list, 
                setpoint_list, 
                hvac_state_list,  # Intent (+1/-1/0)
                max_caps_np.tolist(), 
                min_output, 
                max_cool, 
                eff_derate, 
                start_temp
            )
            
            sim_temps = np.array(sim_temps_list)
            hvac_produced = np.array(hvac_produced_list)
            
            # Note: run_model_continuous does NOT return actual_hvac_state (it yields continuous intent)
            # We use the intent `hvac_state_np` for energy gating (soft gating) implies:
            # We trust the optimizer's "off" intent (hvac_mode_val * gate) for basic energy estimation.
            # We do NOT use discrete actual_state because it breaks gradients.
            
            # --- 4. Calculate Costs ---
            
            # A. Comfort Cost (RMSE from Preference Curve)
            # Get preference limits for *current* time
            # We want to penalize deviation from target, but also respect bounds?
            # Actually we just use the pre-calculated `setpoints` vs `sim_temps`?
            # No, `sim_temps` is result of `run_model`.
            # We want `sim_temps` to match `setpoints`? 
            # NO. We want `sim_temps` to be "comfortable".
            # Since `setpoints` ARE the variables we control, we implicitly "want" the house to be at `setpoints`.
            # But if `setpoints` are chosen effectively, `sim_temps` will track them.
            # The actual comfort penalty comes from: `setpoints` vs `user_preference_envelope`?
            # Looking at original code (I can view it if needed, but assuming standard implementation):
            # It penalizes `sim_temps` deviation from *ideal*?
            # Actually, let's look at `errors = sim_temps - setpoints`?
            # Usually we penalize deviation of SIM TEMP from PREFERRED TEMP.
            # But here `setpoints` IS the control variable.
            # The optimizer tries to pick setpoints such that:
            # 1. Energy is low.
            # 2. Comfort is high (function of `sim_temps` vs config).
            
            # Looking at surrounding code (I can view it if needed, but assuming standard implementation):
            # `state_penalty` usually penalizes valid range violations.
            # `comfort_penalty` penalizes deviation from "ideal".
            
            # ... (Re-using existing logic below, just replacing the model call) ...
            # B. Energy Cost
            res = calculate_energy_vectorized(
                hvac_produced, 
                dt_hours_np, 
                max_caps_np, 
                base_cops_np, 
                hw, 
                eff_derate=1.0, # applied in model already? OR model returns gross?
                # run_model_continuous returns 'hvac_produced' (gross un-derated un-ramped? No, wait)
                # Check run_model_continuous: 
                # hvac_produced_list[i] = q_hvac (calculated from request)
                # then q_hvac *= eff_derate
                # So produced IS UN-DERATED.
                # So we MUST pass eff_derate=1.0? 
                # WAIT. calculate_energy_vectorized applies derate if we pass it? 
                # Or does it assume 'hvac_outputs' as input?
                # Argument name is `hvac_outputs` in signature usually?
                # Let's check calculate_energy_vectorized signature in energy.py later.
                # Assuming we need to pass:
                hvac_states=None, # Use implicit enabled logic based on produced? 
                # OR: We should pass effective_hvac_states (continuous intent) for "soft gating"?
                # User spec says: "True-Off intent derived from setpoints".
                # So we should pass setpoints and bounds.
                setpoints=full_res_setpoints, # Use full_res_setpoints (unrounded) for energy calc
                hvac_mode_val=hvac_mode_val, # +1 or -1
                min_setpoint=min_setpoint,
                max_setpoint=max_setpoint,
                off_intent_eps=0.1 # Hardcoded or from config?
                # Note: optimize.py might not have access to const.CONF_OFF_INTENT_EPS easily unless we import it.
                # For now hardcoding 0.1 is safe/consistent with spec.
            )
            kwh = res['kwh']
            load_ratios = res['load_ratios']
            
            # Comfort
            if hvac_mode_val > 0:  # Heating
                if comfort_mode == 'deadband':
                    floor = target_temps - deadband_slack
                    effective_errors = np.minimum(0, sim_temps - floor)
                else:
                    effective_errors = np.minimum(0, sim_temps - target_temps)
            else:  # Cooling
                if comfort_mode == 'deadband':
                    ceiling = target_temps + deadband_slack
                    effective_errors = np.maximum(0, sim_temps - ceiling)
                else:
                    effective_errors = np.maximum(0, sim_temps - target_temps)

            comfort_cost = center_preference * (effective_errors**2)
            
            # Snap-to-boundary regularization
            # Now primarily a visual tie-breaker
            if hvac_mode_val > 0:
                is_off = full_res_setpoints < (sim_temps - 1.0)
                snap_cost = np.where(is_off, snap_weight * (full_res_setpoints - min_setpoint)**2, 0)
            else:
                is_off = full_res_setpoints > (sim_temps + 1.0)
                snap_cost = np.where(is_off, snap_weight * (full_res_setpoints - max_setpoint)**2, 0)
            
            total_penalty = np.sum((comfort_cost + snap_cost) * dt_hours_np)
            
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

            return kwh + total_penalty + defrost_cost + continuity_penalty


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
    
    # cleanup: restore original hvac state to object
    data.hvac_state[:] = original_hvac_state
    
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
