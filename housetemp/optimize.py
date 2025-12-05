import numpy as np
import pandas as pd
from scipy.optimize import minimize
from . import run_model
from . import run_model

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
    predicted_temps, sim_error, _ = run_model.run_model(full_params, data, hw)
    
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

def optimize_hvac_schedule(data, params, hw, target_temps, comfort_config, block_size_minutes=30):
    """
    Finds the optimal setpoint schedule to minimize Energy + Comfort Penalty.
    Uses block_size_minutes control blocks to reduce dimensionality.
    """
    print(f"Optimizing HVAC Schedule ({block_size_minutes}-min blocks)...")
    
    # Config
    center_preference = comfort_config.get('center_preference', 0.5)
    
    # --- 1. Setup Control Blocks (Aligned) ---
    # Convert start/end to pandas Timestamp for easy flooring
    start_ts = pd.Timestamp(data.timestamps[0])
    end_ts = pd.Timestamp(data.timestamps[-1])
    
    # Floor start to block boundary (e.g. 10:15 -> 10:00 for 60min)
    freq = f"{block_size_minutes}min"
    aligned_start = start_ts.floor(freq)
    aligned_end = end_ts.ceil(freq)
    
    # Generate control timestamps
    control_timestamps = pd.date_range(start=aligned_start, end=aligned_end, freq=freq)
    
    # Convert to sim_minutes (relative to data start)
    control_times = (control_timestamps - start_ts).total_seconds() / 60.0
    
    num_blocks = len(control_times)
    
    # Map simulation timestamps to "minutes from start"
    sim_minutes = (data.timestamps - data.timestamps[0]) / np.timedelta64(1, 'm')
    
    # Initial Guess: Downsample target_temps to blocks
    initial_guess = np.interp(control_times, sim_minutes, target_temps)
    
    # Bounds for optimization variables (thermostat limits)
    bounds = [(50.0, 90.0) for _ in range(num_blocks)]
    
    # Pre-calculate hardware limits for speed
    max_caps = hw.get_max_capacity(data.t_out)
    base_cops = hw.get_cop(data.t_out)
    dt_hours = data.dt_hours
    
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
    
    def schedule_loss(candidate_blocks):
        # 1. Upsample Blocks to Simulation Resolution
        # Use Linear interpolation for smoother gradients during optimization
        candidate_setpoints = np.interp(sim_minutes, control_times, candidate_blocks)
        
        # 2. Update Data
        data.setpoint[:] = candidate_setpoints
        
        # 3. Run Simulation
        sim_temps, _, hvac_outputs = run_model.run_model(params, data, hw)
        
        # 4. Calculate Energy Cost (kWh)
        safe_max_caps = np.where(max_caps > 0, max_caps, 1.0)
        load_ratios = np.abs(hvac_outputs) / safe_max_caps
        plf_corrections = 1.4 - (0.4 * load_ratios)
        final_cops = base_cops * plf_corrections
        
        watts = np.abs(hvac_outputs) / final_cops
        kwh = np.sum((watts / 1000) * dt_hours)
        
        # 5. Calculate Comfort Penalty (Squared Error)
        # Asymmetric Penalty:
        # - Heating: No penalty if T_sim > T_target (overshoot is free/good, energy cost limits it)
        # - Cooling: No penalty if T_sim < T_target (undershoot is free/good)
        
        errors = sim_temps - target_temps
        
        if hvac_mode_val > 0: # Heating
            # We want simn >= target. Error is negative if sim < target.
            # If sim > target, error is positive -> clamp to 0 (no penalty)
            effective_errors = np.minimum(0, errors) 
        else: # Cooling
            # We want sim <= target. Error is positive if sim > target.
            # If sim < target, error is negative -> clamp to 0 (no penalty)
            effective_errors = np.maximum(0, errors)

        comfort_cost = center_preference * (effective_errors**2)
        
        # Sum and Normalize by Time
        total_penalty = np.sum(comfort_cost * data.dt_hours)
        
        # 6. Defrost Energy Cost (Optional)
        defrost_cost = 0.0
        
        if hasattr(hw, 'defrost_risk_zone') and hw.defrost_risk_zone is not None:
            # Get defrost params from heat pump
            risk_min, risk_max = hw.defrost_risk_zone
            duration_hr = hw.defrost_duration_min / 60.0
            interval_hr = hw.defrost_interval_min / 60.0
            power_kw = hw.defrost_power_kw
            
            # Mask: Are we in the frost zone?
            in_frost_zone = (data.t_out >= risk_min) & (data.t_out <= risk_max)
            
            # Weight by load ratio (high load = more defrost needed)
            runtime_in_zone = np.sum(load_ratios * in_frost_zone * dt_hours)
            expected_cycles = runtime_in_zone / interval_hr
            defrost_energy_kwh = expected_cycles * duration_hr * power_kw
            
            # Add to energy cost (1:1 with kWh)
            defrost_cost = defrost_energy_kwh
            
            # Aggressive avoidance mode
            avoid_defrost = comfort_config.get('avoid_defrost', False)
            if avoid_defrost:
                defrost_cost *= 10.0
            
        return kwh + total_penalty + defrost_cost

    # Run Optimization
    print(f"Solving for {num_blocks} control variables (30-min blocks)...")
    result = minimize(
        schedule_loss,
        initial_guess,
        bounds=bounds,
        method='L-BFGS-B',
        options={'disp': True, 'maxiter': 1000}
    )
    
    # Return the upsampled full-resolution schedule
    # 1. Round to nearest integer (thermostats don't do floats)
    final_blocks = np.round(result.x)
    
    # 2. Upsample using "Step" interpolation (hold value)
    # Thermostats hold the setting until the next change.
    indices = np.searchsorted(control_times, sim_minutes, side='right') - 1
    indices = np.clip(indices, 0, len(final_blocks) - 1)
    
    final_setpoints = final_blocks[indices]
    
    return final_setpoints
