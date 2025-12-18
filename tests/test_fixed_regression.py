
import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from custom_components.housetemp.housetemp.measurements import Measurements
from custom_components.housetemp.housetemp import optimize

class TestFixedRegression(unittest.TestCase):
    def test_optimization_respects_measurement_fixed_mask(self):
        """
        Regression Test: Ensure optimize_hvac_schedule respects data.is_setpoint_fixed
        even if fixed_mask argument is NOT explicitly passed (fixing coordinator regression).
        """
        # 1. Setup Data (10 steps)
        timestamps = pd.date_range("2023-01-01 00:00", periods=10, freq="30min")
        steps = len(timestamps)
        
        # Targets: All 70
        target_temps = np.full(steps, 70.0)
        
        # Fixed Mask: Fix indices 4,5,6 to 80 (Simulating a "Boost" or mandatory hold)
        fixed_mask = np.zeros(steps, dtype=bool)
        fixed_mask[4:7] = True
        target_temps[4:7] = 80.0
        
        data = Measurements(
            timestamps=timestamps,
            t_in=np.full(steps, 60.0),
            t_out=np.full(steps, 40.0), # Cold outside
            solar_kw=np.zeros(steps),
            hvac_state=np.zeros(steps),
            setpoint=target_temps,      # Original setpoints
            dt_hours=np.full(steps, 0.5),
            is_setpoint_fixed=fixed_mask # <--- The critical field
        )
        
        # 2. Setup Mock HW & Params
        hw = MagicMock()
        hw.get_max_capacity.return_value = np.full(steps, 10000)
        hw.get_cop.return_value = np.full(steps, 3.0)
        hw.min_output_btu_hr = 1000
        hw.max_cool_btu_hr = 10000
        hw.plf_slope = 0.4
        hw.plf_low_load = 1.4
        hw.plf_min = 0.5
        hw.idle_power_kw = 0.0
        hw.blower_active_kw = 0.0
        hw.defrost_risk_zone = None
        
        # 3. Optim config
        comfort_config = {"mode": "heat", "center_preference": 0.0} # Low comfort pref to encourage drift
        params = [5000, 200, 0, 0, 5000, 1.0]

        # 4. Mock run_model to prevent actual simulation overhead/errors
        # We just need it to return *something* so optimizer runs. 
        # But wait, optimizer needs gradients? 
        # Actually with L-BFGS-B it might run a few iterations. 
        # To strictly test the "FIXED" logic, we rely on the implementation detail 
        # that 'fixed' values set bounds to (val, val).
        
        
        # Patch both run_model (used for initial verification?) and run_model_fast (used in loop)
        # Use full module path consistent with other tests
        with unittest.mock.patch("custom_components.housetemp.housetemp.optimize.run_model.run_model") as mock_run, \
             unittest.mock.patch("custom_components.housetemp.housetemp.optimize.run_model.run_model_fast") as mock_fast:
             
            # run_model returns: sim_temps, sim_error, hvac_delivered, hvac_produced
            mock_run.return_value = (np.full(steps, 70.0), 0.0, np.full(steps, 5000.0), np.full(steps, 5000.0))
            
            # run_model_fast returns lists: sim_temps, hvac_delivered, hvac_produced
            mock_fast.return_value = (
                [70.0]*steps, 
                [5000.0]*steps, 
                [5000.0]*steps
            )
            
            # --- THE CALL ---
            # Crucially: We DO NOT pass 'fixed_mask=...' here.
            final_setpoints, _ = optimize.optimize_hvac_schedule(
                data, params, hw, target_temps, comfort_config, block_size_minutes=30
            )

        # 5. Verify
        # Indices 4,5,6 MUST be 80.0 because they are fixed.
        # Check middle (Index 5)
        self.assertEqual(final_setpoints[5], 80.0, "Fixed setpoint was not respected!")
        
        # Verify others are NOT forced to 80 (Bounds should default to min/max caps)
        # With preference 0, optimizer might drift, but they shouldn't be LOCKED to 80
        # Wait, target_temps outside fixed region is 70.
        # The test is that 80 is preserved.
