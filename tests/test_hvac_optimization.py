import unittest
import numpy as np
import pandas as pd
import json
import os
from unittest.mock import MagicMock
from housetemp import schedule, optimize, run_model
from housetemp.measurements import Measurements

class TestHvacOptimization(unittest.TestCase):
    def setUp(self):
        # Create dummy data (24 hours)
        self.timestamps = pd.date_range("2023-01-01 00:00", periods=24, freq="H")
        self.dt_hours = np.ones(24)
        self.t_out = np.full(24, 50.0) # Cold outside
        self.t_in = np.full(24, 60.0) # Start cold
        self.solar = np.zeros(24)
        self.hvac_state = np.zeros(24)
        self.setpoint = np.full(24, 70.0)
        
        self.measurements = Measurements(
            timestamps=self.timestamps,
            t_in=self.t_in,
            t_out=self.t_out,
            solar_kw=self.solar,
            hvac_state=self.hvac_state,
            setpoint=self.setpoint,
            dt_hours=self.dt_hours
        )
        
        # Mock HeatPump
        self.hw = MagicMock()
        self.hw.get_max_capacity.return_value = np.full(24, 20000.0) # 20k BTU max
        self.hw.get_cop.return_value = np.full(24, 3.0) # COP 3
        
        # Params: C, UA, K, Q_int, H_fac
        self.params = [5000, 200, 1000, 0, 5000]

    def test_load_comfort_schedule(self):
        # Create a dummy comfort.json
        comfort_data = {
            "schedule": [
                {"time": "00:00", "temp": 60},
                {"time": "08:00", "temp": 70, "min": 68, "max": 72},
                {"time": "20:00", "temp": 60}
            ]
        }
        with open("test_comfort.json", "w") as f:
            json.dump(comfort_data, f)
            
        timestamps = pd.date_range("2023-01-01 00:00", "2023-01-01 23:59", freq="30T")
        targets, min_bounds, max_bounds, config = schedule.load_comfort_schedule("test_comfort.json", timestamps)
        
        # Check lengths
        self.assertEqual(len(targets), len(timestamps))
        self.assertEqual(len(min_bounds), len(timestamps))
        self.assertEqual(len(max_bounds), len(timestamps))
        
        # Check values at specific times
        # 00:00 -> 60 (default swing 2 -> min 58, max 62)
        self.assertEqual(targets[0], 60)
        self.assertEqual(min_bounds[0], 58)
        self.assertEqual(max_bounds[0], 62)
        
        # 08:00 -> 70 (explicit min 68, max 72)
        idx_8am = 16 # 8 * 2
        self.assertEqual(targets[idx_8am], 70)
        self.assertEqual(min_bounds[idx_8am], 68)
        self.assertEqual(max_bounds[idx_8am], 72)
        
        os.remove("test_comfort.json")

    def test_optimize_hvac_schedule(self):
        # Create dummy data
        timestamps = pd.date_range("2023-01-01 00:00", periods=24, freq="1H")
        data = Measurements(
            timestamps=timestamps,
            t_in=np.full(24, 70.0),
            t_out=np.full(24, 50.0),
            solar_kw=np.zeros(24),
            hvac_state=np.zeros(24),
            setpoint=np.full(24, 70.0),
            dt_hours=np.full(24, 1.0)
        )
        
        target_temps = np.full(24, 70.0)
        min_bounds = np.full(24, 68.0)
        max_bounds = np.full(24, 72.0)
        comfort_config = {"center_preference": 1.0}
        
        optimized_setpoints = optimize.optimize_hvac_schedule(data, self.params, self.hw, target_temps, min_bounds, max_bounds, comfort_config, block_size_minutes=30)
        
        # Check that setpoints are reasonable
        # Since it's cold outside (50F) and target is 70F, it should heat.
        # To save energy, it might lower the setpoint within the swing (e.g. 68F).
        
        # Verify result shape
        # The optimizer returns the UPSAMPLED setpoints, so it should match input length (24)
        self.assertEqual(len(optimized_setpoints), 24)
        
        # Verify it didn't go crazy
        self.assertTrue(np.all(optimized_setpoints >= 50))
        self.assertTrue(np.all(optimized_setpoints <= 90))
        
        # Verify it tries to stay near target (within swing)
        avg_setpoint = np.mean(optimized_setpoints)
        print(f"Average Optimized Setpoint: {avg_setpoint}")
        
        # It should be lower than 70 (to save energy)
        self.assertLess(avg_setpoint, 70.0)
        self.assertGreaterEqual(avg_setpoint, 67.0) # Allow some tolerance

    def test_run_model_duration(self):
        # Test that duration_minutes works correctly with 1-minute data
        # Create 60 minutes of data (1-min intervals)
        timestamps = pd.date_range("2023-01-01 00:00", periods=60, freq="T")
        dt_hours = np.full(60, 1/60) # 1 minute
        data = Measurements(
            timestamps=timestamps,
            t_in=np.full(60, 70.0),
            t_out=np.full(60, 50.0),
            solar_kw=np.zeros(60),
            hvac_state=np.zeros(60),
            setpoint=np.full(60, 70.0),
            dt_hours=dt_hours
        )
        
        # Run for 30 minutes
        sim_temps, _, _ = run_model.run_model(self.params, data, self.hw, duration_minutes=30)
        
        # Should return 30 steps
        self.assertEqual(len(sim_temps), 30)
        
        # Run for full duration (0)
        sim_temps, _, _ = run_model.run_model(self.params, data, self.hw, duration_minutes=0)
        self.assertEqual(len(sim_temps), 60)

if __name__ == '__main__':
    unittest.main()
