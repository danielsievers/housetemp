import numpy as np
import logging
from . import run_model

_LOGGER = logging.getLogger(__name__)

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
    # Note: hvac_outputs is the DELIVERED heat (ramped).
    # hvac_produced is the PRODUCED heat (unramped) -> Correct for Energy Bill.
    sim_temps, _, hvac_delivered, hvac_produced = run_model.run_model(params, data, hw)
    
    if hw is None:
        _LOGGER.warning("No Heat Pump model provided. Skipping energy calculation.")
        return {'total_kwh': 0.0, 'total_cost': 0.0}

    
    if len(params) > 5:
        eff_derate = params[5]
    else:
        eff_derate = 1.0
        
    # hvac_produced is GROSS (Pre-Derate). Pass eff_derate=1.0.
    return calculate_energy_stats(hvac_produced, data, hw, h_factor=params[4], eff_derate=1.0, cost_per_kwh=cost_per_kwh)


def calculate_energy_vectorized(hvac_outputs, dt_hours, max_caps, base_cops, hw, eff_derate=1.0, hvac_states=None, t_out=None, include_defrost=False):
    """
    Vectorized energy calculation used by both sensor and optimizer.
    Returns: Total kWh
    """
    # Safety Check: Warn if max capacity is missing (0)
    zero_cap_mask = max_caps <= 0
    if np.any(zero_cap_mask):
        # Check if we actually tried to output heat/cool during these steps
        violating_mask = zero_cap_mask & (np.abs(hvac_outputs) > 1.0) # >1 BTU tolerance
        violation_count = np.sum(violating_mask)
        
        if violation_count > 0:
            max_viol_output = np.max(np.abs(hvac_outputs[violating_mask]))
            # Get indices of first 5 violations for context
            viol_indices = np.where(violating_mask)[0][:5]
            _LOGGER.error(f"CRITICAL: Zero Max Capacity at {violation_count} steps (Idx: {viol_indices}...)! Max Load: {max_viol_output:.1f} BTU/hr.")
        elif np.sum(zero_cap_mask) > 0:
            # Harmless zero (e.g. unit off or T_out out of bounds but no request)
            pass

    # Avoid div/0 by masking/forcing safe max_caps
    safe_max_caps = np.where(max_caps > 0, max_caps, 1.0)
    
    # Load Ratio = Output / Max Capacity
    produced_output = np.abs(hvac_outputs) / eff_derate
    load_ratios = produced_output / safe_max_caps
    
    # Efficiency Correction (PLF)
    plf_corrections = hw.plf_low_load - (hw.plf_slope * load_ratios)
    
    # Clip PLF to reasonable physics bounds
    plf_min = getattr(hw, 'plf_min', 0.5)
    
    clipped_plf = np.clip(plf_corrections, plf_min, hw.plf_low_load)
    
    # Check for clipping (Logging)
    if np.any(clipped_plf != plf_corrections):
        clip_count = np.sum(clipped_plf != plf_corrections)
        clip_frac = clip_count / len(clipped_plf)
        if clip_count > 0:
             _LOGGER.debug(f"PLF Clipping Active: {clip_count} steps ({clip_frac:.1%} of time) clamped to [{plf_min}, {hw.plf_low_load}].")
             
    plf_corrections = clipped_plf
    
    # Final COP
    final_cops = base_cops * plf_corrections
    
    # Safety: Ensure COP is never near zero (Div/0 protection)
    # 0.1 is Physical Plausibility warning.
    # 1e-3 is Numerical Safety floor.
    if np.any(final_cops < 0.1):
        low_cop_count = np.sum(final_cops < 0.1)
        _LOGGER.warning(f"Extreme Low COP detected: {low_cop_count} steps < 0.1 (Implausible). Clamped to safe floor.")
        
    final_cops = np.maximum(final_cops, 1e-3)
    
    # Blower & Idle Power Selection
    # If system is Active (Output > 0), use Blower Power (High Static).
    # If system is Enabled but Idle (Output == 0), use Sampling Power.
    
    # 1. Base Input Watts from Compressor COP
    watts = (produced_output / final_cops) / 3.412
    
    # 2. Add Fan/Control Power
    if hasattr(hw, 'idle_power_kw') or hasattr(hw, 'blower_active_kw'):
        idle_kw = getattr(hw, 'idle_power_kw', 0.0)
        active_kw = getattr(hw, 'blower_active_kw', 0.0)
        
        # Mask for Active vs Idle (Enabled)
        # Note: hvac_states=None implies we don't know enabling, but typically we do
        if hvac_states is not None:
            is_enabled = hvac_states != 0
            is_active = (np.abs(hvac_outputs) > 1e-3) & is_enabled
            is_idle = (~is_active) & is_enabled
            
            # Apply Active Fan Power
            # Note: If blower_active_kw is set, we assume it captures the EXTRA fan load.
            # If COP included "standard" fan, adding 0.9kW might be aggressive, but user requested it.
            if active_kw > 0:
                watts = np.where(is_active, watts + (active_kw * 1000.0), watts)
                
            # Apply Idle Power
            if idle_kw > 0:
                watts = np.where(is_idle, watts + (idle_kw * 1000.0), watts)
    
    # 3. Defrost Penalty (Reporting Only)
    if include_defrost and hasattr(hw, 'defrost_risk_zone') and hw.defrost_risk_zone and t_out is not None:
        risk_min, risk_max = hw.defrost_risk_zone
        in_risk = (t_out >= risk_min) & (t_out <= risk_max)
        
        # Defrost happens when Heating is Active in Risk Zone
        # Penalty: Reversing cycle uses power (hw.defrost_power_kw) for duration/interval fraction
        # e.g. 10 mins every 60 mins = 1/6th of time
        ratio = hw.defrost_duration_min / hw.defrost_interval_min
        defrost_kw = hw.defrost_power_kw * ratio
        
        # Apply only when Heating (Output > 0)
        is_heating = hvac_outputs > 0
        
        # Add to watts
        watts = np.where(in_risk & is_heating, watts + (defrost_kw * 1000.0), watts)

    # kWh = (Watts / 1000) * Hours
    kwh_steps = (watts / 1000.0) * dt_hours
    
    return {
        'kwh': np.sum(kwh_steps),
        'kwh_steps': kwh_steps,
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
    
    # Mode-Aware COP Selection
    # If hvac_state < 0 (Cooling), use Cooling Curve. Else (Heating/Off), use Heating Curve.
    heat_cops = hw.get_cop(data.t_out)
    cool_cops = hw.get_cooling_cop(data.t_out)
    is_cooling = data.hvac_state < 0
    base_cops = np.where(is_cooling, cool_cops, heat_cops)
    
    # Truncate if needed
    num_steps = min(len(data), len(hvac_outputs))
    
    hvac_out_slice = hvac_outputs[:num_steps]
    dt_slice = data.dt_hours[:num_steps]
    max_slice = max_caps[:num_steps]
    cop_slice = base_cops[:num_steps]
    hvac_state_slice = data.hvac_state[:num_steps]
    
    res = calculate_energy_vectorized(hvac_out_slice, dt_slice, max_slice, cop_slice, hw, eff_derate=eff_derate, hvac_states=hvac_state_slice, t_out=data.t_out[:num_steps], include_defrost=True)
    total_kwh = res['kwh']

    # --- REPORTING ---
    total_cost = total_kwh * cost_per_kwh
    
    return {
        'total_kwh': total_kwh, 
        'total_cost': total_cost,
        'kwh_steps': res['kwh_steps']
    }
