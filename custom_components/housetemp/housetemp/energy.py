import numpy as np
from . import run_model
from . import run_model

def estimate_consumption(data, params, hw, cost_per_kwh=0.45):
    """
    Calculates estimated kWh usage and cost based on the thermal model fits.
    Applies Mitsubishi-specific Part-Load Efficiency corrections.
    """
    # hw is passed in
    
    # 1. Re-Run Simulation to get the modeled indoor temperatures
    # We use the simulated temps because they reflect the steady-state physics
    # rather than noisy sensor jitter.
    # 1. Re-Run Simulation to get the modeled indoor temperatures
    # We use the simulated temps because they reflect the steady-state physics
    # rather than noisy sensor jitter.
    sim_temps, _, hvac_outputs = run_model.run_model(params, data, hw)
    
    if hw is None:
        print("Warning: No Heat Pump model provided. Skipping energy calculation.")
        return 0.0, 0.0

    
    return calculate_energy_stats(hvac_outputs, data, hw, h_factor=params[4], cost_per_kwh=cost_per_kwh)


def calculate_energy_stats(hvac_outputs, data, hw, h_factor=None, cost_per_kwh=0.45):
    """
    Calculates energy stats from known HVAC outputs.
    Avoids re-running simulation.
    """
    if hw is None:
        return {'total_kwh': 0.0, 'total_cost': 0.0}

    total_kwh = 0.0
    hourly_kwh = np.zeros(len(data))
    
    # Pre-calculate Limits for vectorization speed
    max_caps = hw.get_max_capacity(data.t_out)
    base_cops = hw.get_cop(data.t_out)
    
    # We can only calculate energy for the steps we have outputs for
    num_steps = min(len(data), len(hvac_outputs))
    
    for i in range(num_steps):
        q_output = hvac_outputs[i]

        if q_output == 0:
            continue

        # --- B. CALCULATE EFFICIENCY (COP) ---
        # 1. Get Rated COP for this outdoor temp (High Static Ducted Curve)
        rated_cop = base_cops[i]
        
        # 2. Calculate Part-Load Ratio (PLF)
        # How hard is the unit working relative to max capacity?
        # Avoid division by zero if max_caps is 0 (though q_output should be 0 then)
        if max_caps[i] > 0:
            load_ratio = abs(q_output) / max_caps[i]
        else:
            load_ratio = 1.0 # Fallback
        
        # Mitsubishi Inverter Correction Approximation:
        # Low speed (30% load) is ~40% more efficient than Max speed.
        # Curve: 1.4 at low load, tapering to 1.0 at full load.
        plf_correction = 1.4 - (0.4 * load_ratio)
        
        # Real-world COP
        final_cop = rated_cop * plf_correction
        
        # --- C. CONVERT TO KWH ---
        # Watts = BTU / COP
        watts_input = abs(q_output) / final_cop
        
        # kWh = (Watts / 1000) * Time Step Hours
        kwh_step = (watts_input / 1000) * data.dt_hours[i]
        
        hourly_kwh[i] = kwh_step
        total_kwh += kwh_step

    # --- REPORTING ---
    total_cost = total_kwh * cost_per_kwh
    
    return {'total_kwh': total_kwh, 'total_cost': total_cost}
