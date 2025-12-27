import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize
from .run_model import run_model_continuous
from .utils import upsample_dataframe, get_effective_hvac_state, quantize_setpoint, is_off_intent
from .energy import calculate_energy_vectorized

_LOGGER = logging.getLogger(__name__)

try:
    from .constants import (
        DEFAULT_SWING_TEMP,
        DEFAULT_MIN_CYCLE_MINUTES,
        DEFAULT_OFF_INTENT_EPS,
        DEFAULT_MIN_SETPOINT,
        DEFAULT_MAX_SETPOINT,
        DEFAULT_EFFICIENCY_DERATE,
        W_BOUNDARY_PULL_HEAT,
        W_BOUNDARY_PULL_COOL,
        OFF_BLOCK_MINUTES,
        S_GAP_SMOOTH
    )
except ImportError:
    # Fallbacks for standalone usage
    DEFAULT_SWING_TEMP = 1.0
    DEFAULT_MIN_CYCLE_MINUTES = 15.0
    DEFAULT_OFF_INTENT_EPS = 0.1
    DEFAULT_MIN_SETPOINT = 60.0
    DEFAULT_MAX_SETPOINT = 75.0
    DEFAULT_EFFICIENCY_DERATE = 0.75
    W_BOUNDARY_PULL_HEAT = 0.05
    W_BOUNDARY_PULL_COOL = 0.0
    OFF_BLOCK_MINUTES = 60
    S_GAP_SMOOTH = 0.2

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
        
        # Diagnostic / Snapping Term Weights
        # Use mode-specific constants
        W_WEIGHT = W_BOUNDARY_PULL_HEAT if hvac_mode_val > 0 else W_BOUNDARY_PULL_COOL
        
        # Extract swing for (optional) OFF-safety gating.
        # IMPORTANT: swing_temp is interpreted as a HALF-BAND (±swing_temp around target), not full width.
        swing_temp = float(comfort_config.get('swing_temp', DEFAULT_SWING_TEMP))
        swing_temp = max(0.1, swing_temp)  # guard against misconfig/0
        
        min_cycle_minutes = DEFAULT_MIN_CYCLE_MINUTES


        def schedule_loss(candidate_blocks):
            # 1. Update Setpoints via Hoisted Map (Fast)
            full_res_setpoints = candidate_blocks[sim_to_block_map]
            
            # --- True Off Snapping Redesign ---
            # NO ROUNDING HERE - use continuous setpoints for smooth gradients.
            # Quantization happens ONLY post-solve.
            
            # Physics sees continuous values for gradients
            setpoint_list = full_res_setpoints.tolist()
            
            # --- Use Raw Intent for Physics & Energy ---
            # We no longer "soft gate" hvac_state in the optimizer.
            # We pass the raw intent (+1/-1) and let energy.py handle "True Off" accounting based on setpoints.
            hvac_state_list = working_hvac_state_list  # Use working copy (not data.hvac_state)

            # 2. Run THIN Kernel
            start_temp = float(data.t_in[0]) # Ensure start_temp is float

            # --- Upstream True Off (for Optimization Loop) ---
            # Use continuous setpoints for get_effective_hvac_state to maintain gradients
            effective_hvac_state = get_effective_hvac_state(
                working_hvac_state, 
                full_res_setpoints,  # Continuous, not rounded
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
                    # Outside violation (too hot)
                    outside_errors = np.maximum(0, sim_temps - ceiling)
                    # Inside gap (normalized 0-1: 0 at target, 1 at ceiling)
                    inside_gap = np.clip(sim_temps - target_temps, 0, deadband_slack)
                    inside_norm = inside_gap / safe_slack
                    # Zero out inside term when outside band
                    inside_norm = np.where(outside_errors > 0, 0, inside_norm)
                else:
                    outside_errors = np.maximum(0, sim_temps - target_temps)
                    inside_norm = np.zeros_like(sim_temps)

            comfort_cost = cp_outside * (outside_errors**2) + \
                          cp_inside * (inside_norm**2)
            
            total_penalty = np.sum(comfort_cost * dt_hours_np)
            
            # --- Gated Boundary Pull (True Off Snapping Redesign) ---
            # Only applies in plateau regions where HVAC output is near zero.
            # This breaks ties when the objective is flat, directing the solver to min/max.
            
            # Internal config override (not exposed to HASS/JSON)
            boundary_pull_weight = comfort_config.get('boundary_pull_weight', W_WEIGHT)
            
            # Explicitly form step-aligned temperature vectors to avoid length mismatch.
            # run_model_continuous may return N+1 temps (start + end of each step).
            N = len(dt_hours_np)
            if len(sim_temps) == N + 1:
                T_step = sim_temps[1:]      # end-of-step temps (for comfort)
                T_start = sim_temps[:-1]    # start-of-step temps (for gap gating)
            else:
                T_step = sim_temps[:N]
                T_start = sim_temps[:N]
            
            # Use T_start for gap calculation (pre-control reference)
            temps_np = T_start
            
            if hvac_mode_val > 0:  # Heating
                gap = full_res_setpoints - temps_np
            else:  # Cooling
                gap = temps_np - full_res_setpoints
            
            # Sigmoid Gating with numerical stability (prevent exp overflow)
            z = np.clip(gap / S_GAP_SMOOTH, -50, 50)
            sigmoid = 1 / (1 + np.exp(-z))
            w_idle = 1 - sigmoid  # w_idle ≈ 1 when gap <= 0 (plateau), ≈ 0 when gap >> 0 (active)

            # --- Optional: Cheap OFF-Safety Gate (Block-Level) ---
            # Only relevant if boundary pull is enabled (W_WEIGHT > 0). If W=0, skip for speed.
            # Purpose: prevent boundary_pull from pinning to min/max in cases where being OFF for the
            # full OFF_BLOCK_MINUTES window would violate comfort.
            #
            # We approximate "HVAC OFF" end-of-block temp using passive drift:
            #   dT/dt ≈ ( UA*(T_out - T) + K_solar*solar + Q_int ) / C
            #
            # w_safe_block ~= 1 if OFF is safe for the whole block, ~= 0 if unsafe.
            # We apply: w_idle <- w_idle * w_safe_per_step
            if boundary_pull_weight > 0.0 and OFF_BLOCK_MINUTES and OFF_BLOCK_MINUTES > 0:
                dt_minutes = float(np.median(dt_hours_np) * 60.0)
                steps_per_off_block = max(1, int(round(OFF_BLOCK_MINUTES / dt_minutes)))

                n_steps = len(full_res_setpoints)
                off_block_ids = (np.arange(n_steps) // steps_per_off_block).astype(int)
                n_off_blocks = int(off_block_ids[-1]) + 1 if n_steps > 0 else 0

                # swing_temp is HALF-BAND (±swing). Keep margin < slack so "safe" doesn't collapse.
                safe_margin = 0.5 * swing_temp
                safe_smooth = max(0.05, 0.25 * swing_temp)

                C_thermal = float(params[0])
                UA = float(params[1])
                K_solar = float(params[2])
                Q_int = float(params[3])
                inv_C = 1.0 / max(C_thermal, 1e-9)

                block_hours = float(OFF_BLOCK_MINUTES) / 60.0
                w_safe_blocks = np.ones(n_off_blocks, dtype=float)

                for b in range(n_off_blocks):
                    i0 = b * steps_per_off_block
                    if i0 >= n_steps:
                        break
                    i1 = min(i0 + steps_per_off_block, n_steps)

                    T0 = float(temps_np[i0])
                    
                    # Conservative block extrema for OFF drift estimate
                    Tout_slice = np.asarray(t_out_list[i0:i1], dtype=float)
                    Sol_slice = np.asarray(solar_kw_list[i0:i1], dtype=float)
                    if Tout_slice.size == 0:
                        Tout_slice = np.asarray([float(t_out_list[i0])], dtype=float)
                    if Sol_slice.size == 0:
                        Sol_slice = np.asarray([float(solar_kw_list[i0])], dtype=float)

                    # For heating safety: use min (coldest) outdoor conditions
                    # For cooling safety: use max (hottest) outdoor conditions
                    if hvac_mode_val > 0:
                        Tout_blk = float(np.min(Tout_slice))
                        Sol_blk = float(np.min(Sol_slice))
                    else:
                        Tout_blk = float(np.max(Tout_slice))
                        Sol_blk = float(np.max(Sol_slice))

                    q_passive = UA * (Tout_blk - T0) + K_solar * Sol_blk + Q_int
                    dTdt_off = q_passive * inv_C  # °F/hr
                    T_end_off = T0 + dTdt_off * block_hours

                    targ0 = float(target_temps[i0])
                    
                    # Slack: use deadband_slack in deadband mode, else swing_temp (half-band = ±swing)
                    if comfort_mode == 'deadband':
                        slack = float(deadband_slack)
                    else:
                        slack = swing_temp  # HALF-BAND semantics (±swing)

                    if hvac_mode_val > 0:  # Heating: safe if OFF stays above floor+margin
                        floor0 = targ0 - slack
                        safe_gap = T_end_off - (floor0 + safe_margin)
                    else:  # Cooling: safe if OFF stays below ceiling-margin
                        ceil0 = targ0 + slack
                        safe_gap = (ceil0 - safe_margin) - T_end_off

                    zsafe = np.clip(safe_gap / safe_smooth, -50, 50)
                    w_safe_blocks[b] = 1.0 / (1.0 + np.exp(-zsafe))

                w_safe = w_safe_blocks[off_block_ids] if n_off_blocks > 0 else np.ones(n_steps)
                w_idle = w_idle * w_safe

                if _LOGGER.isEnabledFor(logging.DEBUG):
                    _LOGGER.debug(
                        f"w_safe: min={w_safe.min():.2f}, max={w_safe.max():.2f}, mean={w_safe.mean():.2f}, "
                        f"steps_per_off_block={steps_per_off_block}, dt_minutes={dt_minutes:.2f}, "
                        f"swing_temp={swing_temp:.2f}, safe_margin={safe_margin:.2f}, safe_smooth={safe_smooth:.2f}"
                    )

            # Apply Weighted Pull (only in plateau regions)
            if hvac_mode_val > 0:
                boundary_pull = boundary_pull_weight * np.sum(dt_hours_np * w_idle * (full_res_setpoints - min_setpoint)**2)
            else:
                boundary_pull = boundary_pull_weight * np.sum(dt_hours_np * w_idle * (max_setpoint - full_res_setpoints)**2)
            
            # Diagnostic logging
            if _LOGGER.isEnabledFor(logging.DEBUG):
                _LOGGER.debug(f"boundary_pull contribution: {boundary_pull:.4f}")
                _LOGGER.debug(f"w_idle: min={w_idle.min():.2f}, max={w_idle.max():.2f}, mean={w_idle.mean():.2f}")
                _LOGGER.debug(f"gap: min={gap.min():.1f}°F, max={gap.max():.1f}°F")
            
            # Defrost Energy & Penalty
            # Calculate defrost profile once (vectorized)
            defrost_kwh_steps = np.zeros_like(dt_hours_np)
            defrost_penalty = 0.0
            
            if hasattr(hw, 'defrost_risk_zone') and hw.defrost_risk_zone is not None:
                risk_min, risk_max = hw.defrost_risk_zone
                duration_hr = hw.defrost_duration_min / 60.0
                interval_hr = hw.defrost_interval_min / 60.0
                power_kw = hw.defrost_power_kw
                
                # Identify risk steps
                in_frost_zone = (data.t_out >= risk_min) & (data.t_out <= risk_max)
                # Defrost only happens if we are actually heating (load > 0)
                # (Assuming reverse cycle needed to heat. If idle, coils don't freeze as fast?)
                # Logic: If unit is running significant load in risk zone.
                # Use load_ratios > 0 as trigger, or just is_active?
                # Simple model: if running heat in zone.
                is_heating_active = (load_ratios > 1e-3) & (hvac_mode_val > 0)
                
                # Ratio of time spent in defrost vs heating
                # If interval=60 and duration=10, we spend 10/60 of time defrosting?
                # Or adds overhead? Usually overhead.
                overhead_ratio = duration_hr / interval_hr
                
                # Added kWh = Power * dt * ratio * mask
                defrost_kwh_steps = np.where(in_frost_zone & is_heating_active,
                                           power_kw * overhead_ratio * dt_hours_np,
                                           0.0)
                                           
                if avoid_defrost:
                    # Avoidance Penalty: Pure Nuisance (Unitless / Virtual Cost)
                    # Weighting: 10.0 (Strong signal relative to roughly $0.20-0.50 energy cost)
                    # We penalize the Defrost kWh magnitude itself as a proxy for 'amount of defrosting'
                    defrost_penalty = np.sum(defrost_kwh_steps) * 10.0

            # Total Energy (Base + Defrost)
            total_kwh_steps = res['kwh_steps'] + defrost_kwh_steps

            if rate_per_step is not None:
                # Financial Cost (Weighted by TOU)
                energy_cost = np.sum(total_kwh_steps * rate_per_step)
            else:
                # Pure kWh minimization
                energy_cost = np.sum(total_kwh_steps)
            
            return energy_cost + total_penalty + defrost_penalty + boundary_pull


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
    
    # --- Post-Solve Processing (True Off Snapping Redesign) ---
    
    # 1. Quantize using canonical helper (half-up rounding)
    final_blocks = quantize_setpoint(result.x)
    indices = np.searchsorted(control_times, sim_minutes, side='right') - 1
    indices = np.clip(indices, 0, len(final_blocks) - 1)
    final_setpoints = final_blocks[indices]
    
    # Clamp to bounds
    final_setpoints = np.clip(final_setpoints, min_setpoint, max_setpoint).astype(int)
    
    # 2. Verification Sim with quantized schedule
    # Ensures user-visible metrics match what will actually be commanded
    verify_hvac_state = get_effective_hvac_state(
        working_hvac_state, 
        final_setpoints.astype(float),
        min_setpoint, 
        max_setpoint, 
        DEFAULT_OFF_INTENT_EPS
    )
    
    verify_temps, _, verify_produced = run_model_continuous(
        params, 
        t_out_list=t_out_list, 
        solar_kw_list=solar_kw_list, 
        dt_hours_list=dt_hours_list, 
        setpoint_list=final_setpoints.astype(float).tolist(), 
        hvac_state_list=verify_hvac_state.tolist(),
        max_caps_list=max_caps_np.tolist(), 
        min_output=min_output, 
        max_cool=max_cool, 
        eff_derate=eff_derate, 
        start_temp=float(data.t_in[0])
    )
    
    verify_energy = calculate_energy_vectorized(
        np.array(verify_produced), 
        dt_hours_np, 
        max_caps_np, 
        base_cops_np, 
        hw, 
        eff_derate=1.0,
        hvac_states=verify_hvac_state
    )
    
    # 3. Block-Level OFF Recommendation (Broadcast 1:1)
    dt_minutes = np.mean(dt_hours_np) * 60
    steps_per_block = max(1, int(round(OFF_BLOCK_MINUTES / dt_minutes)))
    
    # Warn if block size doesn't align cleanly
    expected_minutes = steps_per_block * dt_minutes
    if abs(expected_minutes - OFF_BLOCK_MINUTES) > 1.0:
        _LOGGER.warning(
            f"OFF block size ({expected_minutes:.1f}m) differs from target ({OFF_BLOCK_MINUTES}m). "
            f"Consider adjusting timestep for even division."
        )
    
    # Metadata
    debug_info = {
        'success': result.success,
        'message': str(result.message),
        'cost': float(result.fun),
        'iterations': result.nit,
        'evaluations': result.nfev,
        'verify_energy_kwh': float(verify_energy['kwh']),
        'verify_temps': np.array(verify_temps).tolist(),
        'verify_produced': np.array(verify_produced).tolist(),
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
