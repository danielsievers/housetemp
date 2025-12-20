import numpy as np
import logging
from . import run_model

try:
    from ..const import TOLERANCE_BTU_ACTIVE
except (ImportError, ValueError):
    # Fallback for standalone library usage
    TOLERANCE_BTU_ACTIVE = 1.0

_LOGGER = logging.getLogger(__name__)

# --- DEFAULT OVERRIDES (Fallbacks) ---
DEFAULT_COST_PER_KWH = 0.45
DEFAULT_EFFICIENCY_DERATE = 0.75

# --- TRUE CONSTANTS (Physical/Mathematical) ---
KW_TO_WATTS = 1000.0
BTU_TO_WATTS = 0.293071 # 1 / 3.412
BTU_TO_KWH = 0.000293071

# Tolerance Thresholds (use shared constant for "active" detection)
TOLERANCE_BTU = TOLERANCE_BTU_ACTIVE  # Alias for backward compatibility
TOLERANCE_COP_FLOOR = 1e-3  # Numerical safety floor for COP
TOLERANCE_COP_WARN = 0.1    # Physical plausibility warning threshold

# PLF Constants
PLF_MIN_DEFAULT = 0.5   # Default minimum Part-Load Factor if not on HW

def estimate_consumption(data, params, hw, cost_per_kwh=DEFAULT_COST_PER_KWH, setpoints=None, hvac_mode_val=0, min_setpoint=-999, max_setpoint=999, off_intent_eps=0.1):
    """
    Calculates estimated kWh usage and cost based on the thermal model fits.
    Applies Mitsubishi-specific Part-Load Efficiency corrections.
    
    Args:
        setpoints: If provided, enables True-Off accounting. If None, uses data.setpoint.
        hvac_mode_val: +1 for heat, -1 for cool.
        min_setpoint, max_setpoint: Bounds for True-Off detection.
        off_intent_eps: Tolerance for boundary detection.
    """
    # hw is passed in
    
    # 1. Re-Run Simulation to get the modeled indoor temperatures
    # We use the simulated temps because they reflect the steady-state physics
    # rather than noisy sensor jitter.
    # Note: hvac_outputs is the DELIVERED heat (ramped).
    # hvac_produced is the PRODUCED heat (unramped) -> Correct for Energy Bill.
    sim_temps, _, hvac_produced = run_model.run_model_continuous(
        params, 
        t_out_list=data.t_out.flatten().tolist(),
        solar_kw_list=data.solar_kw.flatten().tolist(),
        dt_hours_list=data.dt_hours.flatten().tolist(),
        setpoint_list=data.setpoint.flatten().tolist(),
        hvac_state_list=data.hvac_state.flatten().tolist(),
        max_caps_list=hw.get_max_capacity(data.t_out).flatten().tolist(),
        min_output=hw.min_output_btu_hr,
        max_cool=hw.max_cool_btu_hr,
        eff_derate=params[5] if len(params) > 5 else 1.0,
        start_temp=float(data.t_in[0])
    )
    # run_model_continuous returns (temps, delivered, produced)
    # It does not return actual_hvac_state (uses intent) or rmse.
    # We only need hvac_produced for energy.
    
    if hw is None:
        _LOGGER.warning("No Heat Pump model provided. Skipping energy calculation.")
        return {'total_kwh': 0.0, 'total_cost': 0.0}
    
    # Cast to numpy array for vectorized calculation
    hvac_produced = np.array(hvac_produced)
    if len(params) > 5:
        eff_derate = params[5]
    else:
        eff_derate = 1.0
    
    # Use provided setpoints or fall back to data.setpoint for True-Off accounting
    effective_setpoints = setpoints if setpoints is not None else data.setpoint
        
    # hvac_produced is GROSS (Pre-Derate). Pass eff_derate=1.0.
    return calculate_energy_stats(
        hvac_produced, data, hw, 
        h_factor=params[4], 
        eff_derate=1.0, 
        cost_per_kwh=cost_per_kwh,
        setpoints=effective_setpoints,
        hvac_mode_val=hvac_mode_val,
        min_setpoint=min_setpoint,
        max_setpoint=max_setpoint,
        off_intent_eps=off_intent_eps
    )


def calculate_energy_vectorized(hvac_outputs, dt_hours, max_caps, base_cops, hw, eff_derate=1.0, hvac_states=None, setpoints=None, hvac_mode_val=0, min_setpoint=-999, max_setpoint=999, off_intent_eps=0.1, t_out=None, include_defrost=False):
    """
    Vectorized energy calculation with True-Off accounting (Epsilon-tolerant).
    
    Args:
        hvac_outputs: Produced BTU/hr (Gross/Pre-derate).
        hvac_states: Optional 'enabled intent' (-1/0/1).
        setpoints: Array of setpoints (required for True Off).
        hvac_mode_val: +1 (Heat) or -1 (Cool).
        min_setpoint, max_setpoint: Bounds.
        off_intent_eps: Tolerance for off-intent detection.
    """
    # --- Robustness Shape Checks ---
    assert len(hvac_outputs) == len(dt_hours) == len(max_caps) == len(base_cops), \
        f"Input shape mismatch in energy calc: outputs={len(hvac_outputs)}, dt={len(dt_hours)}, caps={len(max_caps)}, cops={len(base_cops)}"
    
    if setpoints is not None:
        assert len(setpoints) == len(hvac_outputs), "Setpoint array must match simulation length"

    # 1) Compressor watts from physics ALWAYS
    # 1) Determine Effective Operating Point (Duty Cycle Logic)
    produced_output = np.abs(hvac_outputs)
    
    # Get Min Output from HW
    min_output_val = float(getattr(hw, 'min_output_btu_hr', 0.0))
    # Safety floor
    safe_min = np.maximum(min_output_val, TOLERANCE_BTU)
    
    # Effective Capacity: The unit runs at LEAST at min_output when ON.
    # If produced < min, we interpret as cycling min_output at D = produced/min.
    effective_capacity = np.maximum(produced_output, safe_min)
    
    # Duty Cycle: Fraction of the hour the unit is ON
    # If produced > min, D=1.0. If produced < min, D < 1.0.
    duty_cycle = np.clip(produced_output / effective_capacity, 0.0, 1.0)
    
    # Calculate Load Ratio & PLF based on EFFECTIVE capacity
    # (Because efficiency depends on how it RUNS, not the average output)
    # Handle zero max_cap gracefully (avoid div/0)
    safe_max_caps = np.where(max_caps > 1e-3, max_caps, 1.0)
    
    load_ratios = effective_capacity / safe_max_caps
    
    # Get PLF Model Parameters
    # Default to 1.0 (no effect) if missing
    plf_low = float(getattr(hw, 'plf_low_load', 1.0))
    plf_slope = float(getattr(hw, 'plf_slope', 0.0))
    plf_min = float(getattr(hw, 'plf_min', PLF_MIN_DEFAULT))
    
    # Linear Model: PLF = Intercept - (Slope * LoadRatio)
    plf = plf_low - (plf_slope * load_ratios)
    # Clamp
    plf = np.maximum(plf, plf_min)
    
    # Determine final COP
    final_cops = base_cops * plf * eff_derate
    # Protect against div/0
    final_cops = np.maximum(final_cops, TOLERANCE_COP_FLOOR) 
    
    produced_output = np.abs(hvac_outputs)
    
    # Instantaneous Watts (When ON)
    watts_inst = (effective_capacity / final_cops) * BTU_TO_WATTS
    
    # Average Watts (Scaled by Duty)
    watts = watts_inst * duty_cycle
    
    # 2) Determine enabled / active / idle
    if hvac_states is None:
        is_enabled_intent = np.ones_like(produced_output, dtype=bool)
    else:
        # allow float states; treat near-zero as off
        is_enabled_intent = np.abs(hvac_states) > 1e-6
        
    # Active means the physics is actually producing meaningful output AND enabled
    is_active = (produced_output > TOLERANCE_BTU) & is_enabled_intent
    
    # --- True-Off intent derived from setpoints ---
    if setpoints is not None:
        if hvac_mode_val > 0:
            off_intent = setpoints <= (min_setpoint + off_intent_eps)
        elif hvac_mode_val < 0:
            off_intent = setpoints >= (max_setpoint - off_intent_eps)
        else:
             off_intent = np.zeros_like(produced_output, dtype=bool)
             
        # If off-intent: do NOT charge idle/blower adders.
        is_enabled = is_enabled_intent & (~off_intent)
    else:
        is_enabled = is_enabled_intent
        off_intent = np.zeros_like(produced_output, dtype=bool)

    # Recompute active/idle under enabled mask
    is_active = (produced_output > TOLERANCE_BTU) & is_enabled
    is_idle = (~is_active) & is_enabled
    
    # 3) Apply adders
    idle_kw = float(getattr(hw, "idle_power_kw", 0.0) or 0.0)
    active_kw = float(getattr(hw, "blower_active_kw", 0.0) or 0.0)
    
    if idle_kw > 0:
        # Idle applies during the OFF portion of the duty cycle (if enabled)
        # If D=0.25, we are idle 0.75 of the time.
        idle_duty = (1.0 - duty_cycle)
        watts = np.where(is_enabled, watts + (idle_kw * KW_TO_WATTS * idle_duty), watts)
        
    if active_kw > 0:
        # Active adder applies during the ON portion (Duty)
        watts = np.where(is_enabled, watts + (active_kw * KW_TO_WATTS * duty_cycle), watts)
        
    # 4) Validation warning
    mismatch = off_intent & (produced_output > TOLERANCE_BTU)
    if np.any(mismatch):
        mismatch_count = np.sum(mismatch)
        max_output = np.max(produced_output[mismatch])
        _LOGGER.debug(f"True-Off mismatch: {mismatch_count} steps have output (max {max_output:.0f} BTU/hr) despite off-intent setpoint.")

    # 5. Defrost Penalty (Reporting Only)
    # Re-using logic if needed, but keeping it simple for now to fix syntax.
    # If include_defrost requested:
    if include_defrost and hasattr(hw, 'defrost_risk_zone') and hw.defrost_risk_zone and t_out is not None:
        risk_min, risk_max = hw.defrost_risk_zone
        in_risk = (t_out >= risk_min) & (t_out <= risk_max)
        
        defrost_interval = getattr(hw, 'defrost_interval_min', 0)
        defrost_duration = getattr(hw, 'defrost_duration_min', 0)
        defrost_power = getattr(hw, 'defrost_power_kw', 0)
        
        if defrost_interval > 0 and defrost_duration > 0 and defrost_power > 0:
            ratio = defrost_duration / defrost_interval
            defrost_kw = defrost_power * ratio
            is_heating = hvac_outputs > 0
            watts = np.where(in_risk & is_heating, watts + (defrost_kw * KW_TO_WATTS), watts)
            
    kwh_steps = (watts / KW_TO_WATTS) * dt_hours
    
    # Backcalc load ratios for logging/debugging
    safe_max_caps = np.where(max_caps > 0, max_caps, 1.0)
    load_ratios = produced_output / safe_max_caps

    return {
        'kwh': np.sum(kwh_steps),
        'kwh_steps': kwh_steps,
        'load_ratios': load_ratios
    }



def calculate_energy_stats(hvac_outputs, data, hw, h_factor=None, eff_derate=DEFAULT_EFFICIENCY_DERATE, cost_per_kwh=DEFAULT_COST_PER_KWH, setpoints=None, hvac_mode_val=0, min_setpoint=-999, max_setpoint=999, off_intent_eps=0.1):
    """
    Calculates energy stats from known HVAC outputs.
    Avoids re-running simulation.
    
    Args:
        setpoints: If provided, enables True-Off accounting (setpoint at floor/ceiling = no idle power).
        hvac_mode_val: +1 for heat, -1 for cool.
        min_setpoint, max_setpoint: Bounds for True-Off detection.
        off_intent_eps: Tolerance for boundary detection.
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
    setpoint_slice = setpoints[:num_steps] if setpoints is not None else None
    
    res = calculate_energy_vectorized(
        hvac_out_slice, dt_slice, max_slice, cop_slice, hw, 
        eff_derate=eff_derate, 
        hvac_states=hvac_state_slice, 
        setpoints=setpoint_slice,
        hvac_mode_val=hvac_mode_val,
        min_setpoint=min_setpoint,
        max_setpoint=max_setpoint,
        off_intent_eps=off_intent_eps,
        t_out=data.t_out[:num_steps], 
        include_defrost=True
    )
    total_kwh = res['kwh']

    # --- REPORTING ---
    total_cost = total_kwh * cost_per_kwh
    
    return {
        'total_kwh': total_kwh, 
        'total_cost': total_cost,
        'kwh_steps': res['kwh_steps']
    }
