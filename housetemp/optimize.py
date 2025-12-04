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

def optimize_hvac_schedule(data, params, hw, target_temps, min_bounds, max_bounds, comfort_config, block_size_minutes=30):
    """
    Finds the optimal setpoint schedule to minimize Energy + Comfort Penalty.
    Uses block_size_minutes control blocks to reduce dimensionality.
    """
    print(f"Optimizing HVAC Schedule ({block_size_minutes}-min blocks)...")
    
    # Config
    center_preference = comfort_config.get('center_preference', 0.5)
    # Outer penalty is implicitly much steeper (soft wall)
    w_outer = center_preference * 10000.0
    
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
    # data.timestamps[0] is the reference 0
    control_times = (control_timestamps - start_ts).total_seconds() / 60.0
    
    num_blocks = len(control_times)
    
    # Map simulation timestamps to "minutes from start"
    sim_minutes = (data.timestamps - data.timestamps[0]) / np.timedelta64(1, 'm')
    
    # Initial Guess: Downsample target_temps to blocks
    # We take the target temp at the start of each block
    # Use np.interp to sample the target_temps at control_times
    initial_guess = np.interp(control_times, sim_minutes, target_temps)
    
    # Bounds for optimization variables (thermostat limits)
    bounds = [(50.0, 90.0) for _ in range(num_blocks)]
    
    # Pre-calculate hardware limits for speed
    max_caps = hw.get_max_capacity(data.t_out)
    base_cops = hw.get_cop(data.t_out)
    dt_hours = data.dt_hours
    
    # Force Mode based on Config
    mode_str = comfort_config.get('mode', 'auto').lower()
    hvac_mode_val = 2 # Default Auto
    
    if mode_str == 'heat':
        hvac_mode_val = 1
    elif mode_str == 'cool':
        hvac_mode_val = -1
        
    original_hvac_state = data.hvac_state.copy()
    data.hvac_state[:] = hvac_mode_val
    
    def schedule_loss(candidate_blocks):
        # 1. Upsample Blocks to Simulation Resolution
        # Use Linear interpolation for smoother gradients during optimization
        # Thermostats usually hold the setpoint until changed. So 'previous' (step) is more realistic.
        # But 'linear' is smoother for the optimizer.
        # Let's use Linear for now as it helps gradients.
        
        # We need to handle the end of the simulation. 
        # interp will extrapolate constant if outside range? 
        # np.interp uses constant extrapolation by default.
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
        
        # 5. Calculate Comfort Penalty (Envelope)
        # Inner Cost: Parabola centered at Target
        # We want cost at Min/Max to be center_preference.
        # Cost = center_preference * ((T - Target) / (Target - Min))^2  (if T < Target)
        # Cost = center_preference * ((T - Target) / (Max - Target))^2  (if T > Target)
        
        # Calculate distances to bounds
        dist_to_min = target_temps - min_bounds
        dist_to_max = max_bounds - target_temps
        
        # Avoid division by zero (if min == target or max == target)
        dist_to_min = np.maximum(dist_to_min, 0.1)
        dist_to_max = np.maximum(dist_to_max, 0.1)
        
        errors = sim_temps - target_temps
        
        # Inner Logic
        # If T < Target: use dist_to_min
        # If T > Target: use dist_to_max
        scale_factors = np.where(errors < 0, dist_to_min, dist_to_max)
        normalized_error = errors / scale_factors
        
        # Inner Mask: Inside the envelope?
        # i.e. min <= sim <= max  =>  -dist_to_min <= error <= dist_to_max
        # equivalent to |normalized_error| <= 1.0
        abs_norm_error = np.abs(normalized_error)
        inner_mask = abs_norm_error <= 1.0
        
        cost_inner = center_preference * (normalized_error[inner_mask])**2
        
        # Outer Logic (Violation)
        # Smooth transition at boundary (slope match)
        # Slope of inner at boundary (norm_error=1) is: 2 * center_preference
        # But this is slope w.r.t normalized error.
        # Actual slope w.r.t temperature is: 2 * center_preference / scale_factor
        
        outer_mask = ~inner_mask
        excess_norm_error = abs_norm_error[outer_mask] - 1.0
        
        # We can just use the normalized error for the outer penalty too for simplicity
        # Cost = center_preference + slope*excess + w_outer*excess^2
        # slope = 2 * center_preference
        cost_outer = center_preference + (2 * center_preference * excess_norm_error) + (w_outer * excess_norm_error**2)
        
        total_cost = np.zeros(len(errors))
        total_cost[inner_mask] = cost_inner
        total_cost[outer_mask] = cost_outer
        
        # Sum and Normalize by Time
        # We multiply by dt_hours so that weights are "per hour" costs
        # This makes them independent of simulation resolution
        # and comparable to kWh (which is also an integral over time).
        # Assuming uniform dt for simplicity in vectorization, or use data.dt_hours
        
        # cost_inner and cost_outer are arrays of values.
        # We need to map them back to the full timeline to multiply by dt_hours properly
        # But since we split them, let's just use mean dt or assume uniform for the penalty scaling
        # or better:
        
        total_penalty = np.sum(total_cost * data.dt_hours)
        
        return kwh + total_penalty

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
    # We use searchsorted to find the block index for each simulation step
    # side='right' - 1 gives us the index of the control point <= current time
    indices = np.searchsorted(control_times, sim_minutes, side='right') - 1
    # Clamp to valid range (0 to num_blocks-1)
    indices = np.clip(indices, 0, len(final_blocks) - 1)
    
    final_setpoints = final_blocks[indices]
    
    return final_setpoints
