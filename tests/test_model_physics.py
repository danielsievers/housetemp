
import numpy as np
import pytest
from housetemp.run_model import run_model_continuous, run_model_discrete
from housetemp.measurements import Measurements

# Mock Constants
C_THERMAL = 10000.0
UA = 750.0
K_SOLAR = 3000.0
Q_INT = 2000.0
H_FACTOR = 5000.0
EFF_DERATE = 1.0 # Simplify
PARAMS = [C_THERMAL, UA, K_SOLAR, Q_INT, H_FACTOR, EFF_DERATE]

MIN_OUTPUT = 5000.0
MAX_COOL = 20000.0

@pytest.fixture
def mock_inputs():
    steps = 100
    t_out = np.full(steps, 60.0)
    solar = np.zeros(steps)
    dt_hours = np.full(steps, 5.0/60.0) # 5 min steps
    setpoints = np.full(steps, 70.0)
    hvac_state = np.zeros(steps)
    max_caps = np.full(steps, 30000.0)
    start_temp = 70.0
    return {
        "params": PARAMS,
        "t_out_list": t_out.tolist(),
        "solar_kw_list": solar.tolist(),
        "dt_hours_list": dt_hours.tolist(),
        "setpoint_list": setpoints.tolist(),
        "hvac_state_list": hvac_state.tolist(),
        "max_caps_list": max_caps.tolist(),
        "min_output": MIN_OUTPUT,
        "max_cool": MAX_COOL,
        "eff_derate": EFF_DERATE,
        "start_temp": start_temp
    }

def test_continuous_small_perturbation_is_bounded_smoke(mock_inputs):
    """
    Verify that run_model_continuous is strictly continuous/differentiable-like.
    A small perturbation in setpoint should result in a small, bounded change in output.
    This ensures no discrete jumps (soft gating is OK, but hard discrete logic is banned).
    """
    inputs = mock_inputs.copy()
    inputs["hvac_state_list"] = [1] * 100 # Always Enabled (Heat)
    
    # Baseline
    res1_temps, _, res1_produced = run_model_continuous(**inputs)
    
    # Perturbed
    inputs_Mod = inputs.copy()
    inputs_Mod["setpoint_list"] = [s + 0.1 for s in inputs["setpoint_list"]] # +0.1F
    res2_temps, _, res2_produced = run_model_continuous(**inputs_Mod)
    
    # Check max difference
    diff_temps = np.abs(np.array(res1_temps) - np.array(res2_temps))
    diff_prod = np.abs(np.array(res1_produced) - np.array(res2_produced))
    
    assert np.max(diff_temps) < 0.5, "Temperature output diverged too much for small perturbation"
    assert np.max(diff_prod) < 5000.0, "Production output jump was too large (discontinuity?)" 
    # Note: 0.1F change with H=5000 implies ~500 BTU change. If it jumps 20,000, that's a discrete switch.

def test_discrete_scenarios_hysteresis(mock_inputs):
    """
    Verify Hysteresis Logic:
    - Should turn ON when < Setpoint - HalfSwing
    - Should turn OFF when > Setpoint + HalfSwing
    """
    inputs = mock_inputs.copy()
    swing = 2.0
    half_swing = 1.0
    min_cycle = 0 # Disable timers for pure hysteresis check
    
    # Create a temperature ramp down then up
    # We cheat: we force t_out low to drive temp down, then high to drive up.
    # Actually, easier to just mock internal temp? No, model calculates it.
    # Let's use a very small C_thermal to make it responsive.
    fast_params = list(PARAMS)
    fast_params[0] = 100.0 # Tiny thermal mass
    inputs["params"] = fast_params
    
    setpoint = 70.0
    inputs["setpoint_list"] = [70.0] * 100
    inputs["hvac_state_list"] = [1] * 100 # Heat Enabled
    
    # Cold outside, then Warm outside
    t_out = [50.0] * 50 + [90.0] * 50
    inputs["t_out_list"] = t_out
    
    res_temps, _, _, actual_state, diag = run_model_discrete(
        **inputs, swing_temp=swing, min_cycle_minutes=min_cycle
    )
    
    # Check transitions
    # Temp input starts at 70.
    # 50F outside -> Temp drops.
    # Should turn ON when < 69.0
    
    max_output = MAX_COOL # Just checking
    
    was_on = False
    turn_on_temp = None
    turn_off_temp = None
    
    for i, temp in enumerate(res_temps):
        state = actual_state[i]
        if not was_on and state == 1:
            was_on = True
            turn_on_temp = res_temps[i-1] # Temp *before* switch? Or current? 
            # Logic: checks current_temp at start of step vs threshold.
            turn_on_temp = temp
            
        if was_on and state == 0:
            was_on = False
            turn_off_temp = res_temps[i-1] # Temp at previous step (checking logic consistency) or current?
            turn_off_temp = temp
            
    # Verify thresholds
    # Note: Initial temp 70 is > 69, so it starts OFF.
    if turn_on_temp is not None:
        assert turn_on_temp <= (setpoint - half_swing + 0.1), f"Turned on too late: {turn_on_temp:.2f}"
    
    if turn_off_temp is not None:
        # Note: In discrete step, it might overshoot slightly.
        assert turn_off_temp >= (setpoint + half_swing - 0.5), f"Turned off too early: {turn_off_temp:.2f}" # Allow slop

def test_discrete_min_cycle_timer(mock_inputs):
    """
    Verify Min-On Timer:
    - Once ON, cannot turn OFF for X minutes even if temp satisfied.
    """
    inputs = mock_inputs.copy()
    min_cycle = 15.0
    swing = 1.0
    
    # 20 steps only
    steps = 20
    
    # Start Cold, Turn On immediately
    inputs["start_temp"] = 65.0 # Setpoint 70
    inputs["hvac_state_list"] = [1] * steps # 20 steps * 5 min = 100 min
    
    # Truncate lists to match
    inputs["t_out_list"] = inputs["t_out_list"][:steps]
    inputs["solar_kw_list"] = inputs["solar_kw_list"][:steps]
    inputs["dt_hours_list"] = inputs["dt_hours_list"][:steps]
    inputs["setpoint_list"] = inputs["setpoint_list"][:steps]
    inputs["max_caps_list"] = inputs["max_caps_list"][:steps]
    
    # Force Heat Up FAST
    inputs["min_output"] = 50000.0 
    inputs["max_caps_list"] = [50000.0] * steps
    
    # Use small thermal mass
    fast_params = list(PARAMS)
    fast_params[0] = 500.0 
    inputs["params"] = fast_params
    
    res_temps, _, _, actual_state, diag = run_model_discrete(
        **inputs, swing_temp=swing, min_cycle_minutes=min_cycle
    )
    
    # Should start ON.
    # Should heat up rapidly past 70.5 (Setpoint + HalfSwing).
    # But should stay ON for at least 3 steps (15 min).
    
    assert actual_state[0] == 1
    assert actual_state[1] == 1
    assert actual_state[2] == 1 # 10-15 min
    
    # Check temps
    # Step 0: 65 -> Huge Heat -> Maybe 75?
    # Step 1: 75 -> Huge Heat -> 85?
    # Step 2: 85 -> ...
    # Even if temp > 70.5, State must be 1 due to timer.
    
    assert res_temps[1] > 70.5, "Setup failed: didn't heat up enough to trigger off"
    assert actual_state[1] == 1, "Min Cycle Violation: Turned off before 15 min"

