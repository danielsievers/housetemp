import numpy as np
import pandas as pd
from scipy.optimize import minimize
from . import run_model
from .energy import calculate_energy_vectorized

def loss_function(active_params, data, hw, fixed_passive_params=None):
    # Construct full params vector
    if fixed_passive_params:
        # fixed_passive_params comes in as [C, UA, K, Q]
        # active_params = [UA, H_factor, efficiency_derate]
        
        ua_opt = active_params[0]
        h_factor = active_params[1]
        eff_derate = active_params[2] if len(active_params) > 2 else 1.0
        
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
        full_params = list(active_params) + [1.0] # Default efficiency_derate

    # 1. Run Simulation
    predicted_temps, sim_error, _, _ = run_model.run_model(full_params, data, hw)
    
    # 2. Compare to Reality (RMSE)
    error = sim_error
    
    # 3. Penalize Physics Violations (Soft Constraints)
    # If fixing passive params, we don't check them here (assumed good)
    if not fixed_passive_params:
        if full_params[0] < 1000: return 1e6 # Mass too low
        if full_params[1] < 50: return 1e6   # UA too low
    
    return error

def run_optimization(data, hw, initial_guess=None, fixed_passive_params=None):
    # hw is now passed in
    
    if fixed_passive_params:
        print(f"--- ACTIVE PARAMETER OPTIMIZATION MODE ---")
        print("Optimizing Active Parameters + Floating UA (Occupied Infiltration).")
         # Initial Guess for [UA, H_factor, efficiency_derate]
         # Start UA at fixed value (189), H at 10k, Eff at 0.9
        initial_guess = [fixed_passive_params[1], 10000, 0.9]
        
        bounds = [
            (200, 600),    # UA (Constrain slightly to help convergence: 200-600)
            (2000, 30000), # H_factor
            (0.5, 1.2)     # Efficiency
        ]
    else:
        # Full Optimization
        # [C, UA, K_solar, Q_int, H_factor]
        if initial_guess is None:
            initial_guess = [4732, 213, 836, 600, 10000]
        
        bounds = [
            (4000, 20000),  # C (Mass)
            (200, 1500),    # UA (Leakage)
            (700, 2000),    # K_solar
            (200, 10000),   # Q_int
            (5000, 20000)   # H_factor
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
    
    # --- 1. PRE-EXTRACT RAW ARRAYS (Hoisting) ---
    # Convert everything to Python lists ONCE for the kernel
    t_out_list = data.t_out.tolist()
    solar_kw_list = data.solar_kw.tolist()
    dt_hours_list = data.dt_hours.tolist()
    hvac_state_list = data.hvac_state.tolist()
    
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
    # Check if we can use arange (fast path)
    # dt_hours is usually uniform coming from input_handler
    # Check first few elements variance or just assume uniform if configured
    # Safety: check if max variance is tiny.
    dt_arr = data.dt_hours
    if len(dt_arr) > 1 and np.max(dt_arr) - np.min(dt_arr) < 1e-6:
        # Uniform steps
        step_min = dt_arr[0] * 60.0
        sim_minutes = np.arange(len(data.timestamps)) * step_min
    else:
        # Fallback to Pandas (slow path)
        ts_series = pd.to_datetime(data.timestamps)
        sim_minutes = (ts_series - start_ts).total_seconds().values / 60.0
    
    # Unpack Optimization Params (Needed for kernel)
    # We pass the full params tuple to kernel
    # Params: [C, UA, K, Q, H, (eff)]
    eff_derate = 1.0
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
        # Calculate indices once per pass!
        # Maps every simulation step to the control block that governs it
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
             # Map fixed mask to control blocks
             # Any fixed time in a block fixes the block? Or majority?
             # Safer: Center sample
             indices = np.searchsorted(sim_minutes, control_times, side='left')
             indices = np.clip(indices, 0, len(fixed_mask) - 1)
             block_fixed_flags = fixed_mask[indices]
             current_guess[block_fixed_flags] = np.interp(control_times[block_fixed_flags], sim_minutes, target_temps)

        # --- Bounds ---
        min_setpoint = comfort_config.get('min_setpoint', 60.0)
        max_setpoint = comfort_config.get('max_setpoint', 75.0)
        bounds = []
        for i in range(num_blocks):
            if block_fixed_flags[i]:
                val = current_guess[i]
                bounds.append((val, val))
            else:
                bounds.append((min_setpoint, max_setpoint))
        
        # --- Loss Function ---
        user_preference = comfort_config.get('center_preference', 1.0)
        # Map 0-1 input to a 0.1-5.1 range to compete with high energy costs (0.75 derate)
        center_preference = 0.1 + (user_preference * 5.0)
        comfort_mode = comfort_config.get('comfort_mode', 'quadratic')
        deadband_slack = comfort_config.get('deadband_slack', 1.5)
        avoid_defrost = comfort_config.get('avoid_defrost', False)
        snap_weight = 0.001  # Tie-breaker weight for snap-to-boundary

        def schedule_loss(candidate_blocks):
            # 1. Update Setpoints via Hoisted Map (Fast)
            # data.setpoint (numpy) -> list conversion overhead? 
            # run_model_fast needs a list for setpoints. 
            # Generating a massive list from numpy array every iteration is slow.
            # OPTIMIZATION: operate on the list directly?
            # Python lists don't support vectorized indexing.
            # So we MUST generate the full resolution array first.
            
            # Using numpy indexing is fast:
            full_res_setpoints = candidate_blocks[sim_to_block_map]
            
            # --- Late Rounding Strategy ---
            # 1. Round setpoints for honest physics simulation
            effective_setpoints = np.round(full_res_setpoints)
            
            # 2. Add continuity penalty so solver sees a gradient on the floats
            # This guides the solver towards integers without stalling on flat plateaus
            continuity_penalty = 0.0001 * np.sum((full_res_setpoints - effective_setpoints)**2)
            
            # Convert to list for the kernel (Overhead here is inevitable if kernel uses lists)
            # But converting simple float array to list is reasonable (~ms for 1000 items)
            # Use EFFECTIVE (rounded) setpoints for physics
            setpoint_list = effective_setpoints.tolist()
            
            # 2. Run THIN Kernel
            sim_temps_list, hvac_delivered_list, hvac_produced_list = run_model.run_model_fast(
                params, t_out_list, solar_kw_list, dt_hours_list, setpoint_list, hvac_state_list,
                max_caps_list, min_output, max_cool, eff_derate, start_temp
            )
            
            # 3. Vectorized Cost Calculation (NumPy)
            # Convert results back to numpy for vectorized math
            hvac_produced = np.array(hvac_produced_list) # Unramped for Energy Bill
            sim_temps = np.array(sim_temps_list)         # Ramped for Comfort
            
            # Energy (Use PRODUCED/UNRAMPED heat for cost calculation)
            # hvac_produced is GROSS (Pre-Derate). Pass eff_derate=1.0 to avoid double-division.
            res = calculate_energy_vectorized(hvac_produced, dt_hours_np, max_caps_np, base_cops_np, hw, eff_derate=1.0, hvac_states=data.hvac_state)
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
            
            # Snap-to-boundary regularization for "off" state visualization
            # When setpoint is well below (heating) or above (cooling) room temp,
            # nudge it toward the cap boundary for cleaner UI presentation
            if hvac_mode_val > 0:  # Heating
                # "Off" = setpoint well below room temp (won't trigger heat)
                is_off = full_res_setpoints < (sim_temps - 1.0)
                snap_cost = np.where(is_off, snap_weight * (full_res_setpoints - min_setpoint)**2, 0)
            else:  # Cooling
                # "Off" = setpoint well above room temp (won't trigger cool)
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
        # User suggested tuning 'eps' (step size) and 'ftol'
        # Landscape is noisy due to discrete minute steps and time-aliasing.
        # Relax tolerance to prevent excessive evaluations.
        if optimization_options:
             opts = optimization_options
        else:
             # Default Fallback (should typically be passed)
             opts = {
                 'disp': False, 
                 'maxiter': 500,  
                 'ftol': 1e-4,    
                 'gtol': 1e-4,    
                 'eps': 0.5       
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
        coarse_block_size = 120
        if block_size_minutes >= coarse_block_size:
            print("Skipping Multi-Scale: Target resolution is already coarse.")
            final_result, final_times = _run_pass(block_size_minutes)
        else:
            print("Strategy: Multi-Scale (Coarse -> Fine)")
            
            # Coarse Options
            coarse_opts = {'disp': False, 'maxiter': 50, 'ftol': 1e-2, 'gtol': 1e-2, 'eps': 1.0}
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
            fine_opts = {'disp': False, 'maxiter': 500, 'ftol': 1e-4, 'gtol': 1e-4, 'eps': 0.5}
            final_result, final_times = _run_pass(block_size_minutes, initial_guess_blocks=warm_start_guess, optimization_options=fine_opts)
    else:
        # Legacy Single Pass
        print("Strategy: Single-Scale (Cold Start)")
        # Legacy Single Pass
        print("Strategy: Single-Scale (Cold Start)")
        # Legacy opts (same as fine)
        legacy_opts = {'disp': False, 'maxiter': 500, 'ftol': 1e-4, 'gtol': 1e-4, 'eps': 0.5}
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
