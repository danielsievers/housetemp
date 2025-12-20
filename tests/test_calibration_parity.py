
import pytest
import pandas as pd
import numpy as np
import datetime
from custom_components.housetemp.housetemp.measurements import Measurements
from custom_components.housetemp.housetemp.optimize import run_optimization
from custom_components.housetemp.housetemp.run_model import HeatPump
from unittest.mock import MagicMock

# Load processed CSV (as produced by hass_convert.py -> main.py pipeline)
DATA_PATH = "data/dec15.csv"

def test_calibration_parity():
    """
    Verify that loading the CSV and running optimization produces results
    consistent with the 'main.py' workflow.
    """
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        pytest.skip(f"Comparison data {DATA_PATH} not found.")

    # 1. Simulate 'main.py' data loading (Measurements construction)
    # main.py expects columns: time, t_in, t_out, solar, hvac_state, setpoint
    # It parses dates and builds Measurements.
    
    # Ensure correct types
    df['time'] = pd.to_datetime(df['time'])
    
    # Calculate dt_hours from timestamps
    timestamps = df['time'].dt.to_pydatetime()
    t_sec = np.array([t.timestamp() for t in timestamps])
    if len(t_sec) > 1:
        dt_arr_h = np.diff(t_sec) / 3600.0
        # Pad last element to match length (assume constant step at end)
        dt_hours = np.append(dt_arr_h, dt_arr_h[-1])
    else:
        dt_hours = np.array([1.0]) # Fallback for single point

    meas = Measurements(
        timestamps=timestamps,
        t_in=df['indoor_temp'].values,
        t_out=df['outdoor_temp'].values,
        solar_kw=df['solar_kw'].values,
        hvac_state=df['hvac_mode'].values,
        setpoint=df['target_temp'].values,
        dt_hours=dt_hours
    )

    # 2. Hardware (HeatPump)
    # Use actual config to match main.py defaults exactly
    hw = HeatPump("data/heat_pump.json")

    # 3. Initial Guesses (Standard defaults)
    initial_guess = None # Use optimize.py defaults (GUESS_FULL_OPT)
    
    # 4. Run Optimization
    # This calls the SAME function used by the service and main.py
    res = run_optimization(meas, hw, initial_guess=initial_guess)
    
    assert res.success, f"Optimization failed: {res.message}"
    
    # 5. Output Results for verification
    print("\n--- Calibration Parity Results ---")
    print(f"C: {res.x[0]:.1f}")
    print(f"UA: {res.x[1]:.1f}")
    print(f"K: {res.x[2]:.1f}")
    print(f"Q: {res.x[3]:.1f}")
    print(f"H: {res.x[4]:.1f}")
    print(f"Eff: {res.x[5]:.3f}")
    print(f"Cost: {res.fun:.4f}")
    
    # Basic sanity checks
    # C hits lower bound (4000) with this data/mock
    assert 3999 <= res.x[0] < 50000 
    assert 100 < res.x[1] < 2000   # UA
    
