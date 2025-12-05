import numpy as np
import pandas as pd
from scipy.optimize import minimize
from . import run_model
from . import run_model

def loss_function(params, data, hw):
    # 1. Run Simulation with current parameter guess
    predicted_temps, sim_error, _ = run_model.run_model(params, data, hw)
    
    # 2. Compare to Reality (Root Mean Square Error)
    # We use the error returned by run_model directly, but let's verify:
    manual_error = np.sqrt(np.mean((predicted_temps - data.t_in)**2))
    
    # Print both (as requested) - Note: This will spam the console during optimization
    # print(f"DEBUG: Sim Error={sim_error:.6f}, Manual Error={manual_error:.6f}")
    
    # Assert that they match (sanity check)
    if not np.isclose(sim_error, manual_error, atol=1e-5):
        raise ValueError(f"Error mismatch! Sim: {sim_error}, Manual: {manual_error}")
    
    error = sim_error
    
    # 3. Penalize Physics Violations (Soft Constraints)
    if params[0] < 1000: return 1e6 # Mass too low
    if params[1] < 50: return 1e6   # UA too low
    
    return error

def run_optimization(data, hw, initial_guess=None):
    # hw is now passed in
    
    # [C, UA, K_solar, Q_int, H_factor]
    # If no linear fit provided, fall back to hardcoded defaults
    if initial_guess is None:
        initial_guess = [4732, 213, 836, 600, 10000]
    
    # Bounds for the solver
    # (min, max)
    bounds = [
# vacation run:
#        (4000, 20000),  # C (Mass) - Widen this. Let it go higher if it wants.
#        (150, 800),     # UA (Leakage) - TIGHTEN THIS. Cap at 500 (Reasonable Max).
#        (700, 2000),  # K_solar (Window Factor) - Lower cap (Low-E glass confirmed).
#        (200, 2500),    # Q_int (Internal Heat) - TIGHTEN THIS. Cap at 2500 (730 Watts).
#        (1000, 30000)   # H_factor (Inverter Ramp)

        (4000, 20000),  # C (Mass) - Widen this. Let it go higher if it wants.
        (200, 1500),     # UA (Leakage) - TIGHTEN THIS. Cap at 500 (Reasonable Max).
        (700, 2000),  # K_solar (Window Factor) - Lower cap (Low-E glass confirmed).
        (200, 10000),    # Q_int (Internal Heat) - TIGHTEN THIS. Cap at 2500 (730 Watts).
        (5000, 20000)   # H_factor (Inverter Ramp)
    ]   
    print("Starting Optimization (this may take a few seconds)...")
    result = minimize(
        loss_function, 
        initial_guess, 
        args=(data, hw), 
        bounds=bounds, 
        method='L-BFGS-B'
    )
    
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
        # Cost = center_preference * (T_sim - T_target)^2
        errors = sim_temps - target_temps
        comfort_cost = center_preference * (errors**2)
        
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
