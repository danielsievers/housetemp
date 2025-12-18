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

    
    if len(params) > 5:
        eff_derate = params[5]
    else:
        eff_derate = 1.0
        
    return calculate_energy_stats(hvac_outputs, data, hw, h_factor=params[4], eff_derate=eff_derate, cost_per_kwh=cost_per_kwh)


def calculate_energy_vectorized(hvac_outputs, dt_hours, max_caps, base_cops, hw, eff_derate=1.0):
    """
    Vectorized energy calculation used by both sensor and optimizer.
    Returns: Total kWh
    """
    # Avoid div/0 by masking/forcing safe max_caps
    # If max_caps is 0, load_ratio is 0 because output must be 0 (enforced by model/optimizer)
    # But if output is non-zero and max_cap is 0, we have physics violation. 
    # Safety: use a tiny number or 1.0 for division if max_caps is 0.
    safe_max_caps = np.where(max_caps > 0, max_caps, 1.0)
    
    # Load Ratio = Output / Max Capacity
    # Note: Output here is Delivered Capacity.
    # To get Produced Capacity (which defines load ratio), we technically should divide by eff_derate too.
    # However, max_caps is usually defined at the unit (Produced).
    # So Load Ratio = (Delivered / Derate) / MaxCap
    produced_output = np.abs(hvac_outputs) / eff_derate
    load_ratios = produced_output / safe_max_caps
    
    # Efficiency Correction (PLF)
    # PLF = Base + (Slope * LoadMultiplier) -> No, standard model is usually linear relation
    # From RunModel/Tests: plf = low_load - (slope * ratio)
    plf_corrections = hw.plf_low_load - (hw.plf_slope * load_ratios)
    
    # Final COP
    final_cops = base_cops * plf_corrections
    
    # Input Watts = Output BTU / COP / 3.412
    # 3.412 BTU = 1 Watt-hour. So Watts = (BTU/hr) / 3.412
    # But Output is derived from COP: Input * COP = Output
    # So Input = Output / COP
    # Input is in BTU/hr. Need Watts. 
    # Watts = (Input BTU/hr) / 3.412
    
    # Corrected for Derate: Input = (Delivered / Derate) / COP
    watts = (produced_output / final_cops) / 3.412
    
    # kWh = (Watts / 1000) * Hours
    kwh_steps = (watts / 1000.0) * dt_hours
    
    return {
        'kwh': np.sum(kwh_steps),
        'load_ratios': load_ratios
    }


def calculate_energy_stats(hvac_outputs, data, hw, h_factor=None, eff_derate=1.0, cost_per_kwh=0.45):
    """
    Calculates energy stats from known HVAC outputs.
    Avoids re-running simulation.
    """
    if hw is None:
        return {'total_kwh': 0.0, 'total_cost': 0.0}

    # Pre-calculate Limits for vectorization speed
    max_caps = hw.get_max_capacity(data.t_out)
    base_cops = hw.get_cop(data.t_out)
    
    # Truncate if needed
    num_steps = min(len(data), len(hvac_outputs))
    
    hvac_out_slice = hvac_outputs[:num_steps]
    dt_slice = data.dt_hours[:num_steps]
    max_slice = max_caps[:num_steps]
    cop_slice = base_cops[:num_steps]
    
    res = calculate_energy_vectorized(hvac_out_slice, dt_slice, max_slice, cop_slice, hw, eff_derate=eff_derate)
    total_kwh = res['kwh']

    # --- REPORTING ---
    total_cost = total_kwh * cost_per_kwh
    
    return {'total_kwh': total_kwh, 'total_cost': total_cost}
