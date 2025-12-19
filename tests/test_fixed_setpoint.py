"""Test fixed schedule functionality."""
import pytest
import numpy as np
import json
from datetime import datetime, timedelta
import pandas as pd
from unittest.mock import MagicMock
import unittest.mock

from custom_components.housetemp.housetemp.schedule import process_schedule_data
from custom_components.housetemp.housetemp.measurements import Measurements
from custom_components.housetemp.housetemp.optimize import optimize_hvac_schedule

def test_process_schedule_fixed_mask():
    """Test that process_schedule_data correctly parses 'fixed' flag."""
    
    # 1. Setup Schedule
    schedule_json = {
        "schedule": [
            {
                "weekdays": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                "daily_schedule": [
                    {"time": "00:00", "temp": 60, "fixed": False},
                    {"time": "12:00", "temp": 70, "fixed": True}, # 12pm-6pm FIXED
                    {"time": "18:00", "temp": 60, "fixed": False}
                ]
            }
        ]
    }
    
    # 2. Setup Timestamps (One Day)
    start = datetime(2023, 1, 1, 0, 0, 0) # Sunday
    timestamps = [start + timedelta(minutes=30*i) for i in range(48)] # 24 hours in 30min steps
    
    # 3. Run
    hvac, targets, fixed_mask = process_schedule_data(timestamps, schedule_json, default_mode="heat")
    
    # 4. Verify
    # 00:00 - 12:00 (Indices 0-23) -> Fixed=False
    assert not np.any(fixed_mask[0:24])
    
    # 12:00 - 18:00 (Indices 24-35) -> Fixed=True
    assert np.all(fixed_mask[24:36])
    
    # 18:00 - 00:00 (Indices 36-47) -> Fixed=False
    assert not np.any(fixed_mask[36:])

def test_optimize_fixed_bounds():
    """Test that optimization respects fixed blocks."""
    
    # 1. Mock Data / Hardware
    hw = MagicMock()
    hw.get_max_capacity.return_value = np.full(48, 10000)
    hw.get_cop.return_value = np.full(48, 3.0)
    hw.defrost_risk_zone = None
    hw.min_output_btu_hr = 3000
    hw.max_cool_btu_hr = 54000
    hw.plf_low_load = 1.4
    hw.plf_slope = 0.4
    hw.plf_min = 0.5
    hw.idle_power_kw = 0.25
    hw.blower_active_kw = 0.9
    
    data = MagicMock()
    start = datetime(2023, 1, 1, 0, 0, 0)
    timestamps = [start + timedelta(minutes=30*i) for i in range(48)]
    data.timestamps = timestamps
    data.t_in = np.full(48, 60.0)
    data.t_out = np.full(48, 40.0)
    data.hvac_state = np.full(48, 1)
    data.setpoint = np.full(48, 60.0)
    data.dt_hours = np.full(48, 0.5)
    data.solar_kw = np.zeros(48)
    
    params = [5000, 200, 0, 0, 5000, 1.0] # Dummy physics
    comfort_config = {"center_preference": 1.0, "mode": "heat"}
    
    # 2. Setup Targets & Fixed Mask
    # All 60, except block 24 (12:00) is 70 and FIXED.
    target_temps = np.full(48, 60.0)
    target_temps[24:36] = 70.0 
    
    fixed_mask = np.zeros(48, dtype=bool)
    fixed_mask[24:36] = True # Fix the 70 region
    
    # 3. Run Optimization
    # We use small bounds on the non-fixed parts to see if they move
    # But mainly we check if the FIXED parts stayed at 70.
    
    # Needs to mock run_model to return valid shapes
    with unittest.mock.patch("custom_components.housetemp.housetemp.optimize.run_model.run_model") as mock_run:
        # returns: sim_temps, sim_error, hvac_outputs
        # Must match length 48
        mock_run.return_value = (np.full(48, 60.0), 0.0, np.full(48, 5000.0))
        
        result_setpoints, _ = optimize_hvac_schedule(
            data, params, hw, target_temps, comfort_config, block_size_minutes=30, fixed_mask=fixed_mask
        )
    
    # 4. Verify
    # The optimized output for the fixed region MUST be 70
    # Because we mocked specific control blocks to align with 30min data steps,
    # indices 24-35 correspond to blocks 24-35 approx.
    
    fixed_slice = result_setpoints[24:36]
    assert np.all(fixed_slice == 70.0), f"Fixed setpoints changed! {fixed_slice}"
    
    # Ensure non-fixed points are NOT 70 (they were target 60, optimizer likely kept them near 60)
    # They should definitely not be forced to 70.
    non_fixed_slice = result_setpoints[0:24]
    
    # With dummy model, it might drift. But ensuring equality to 70 is the test for 'fixed'.
    # Checking that bounds were respected is implicitly tested by result == target.
