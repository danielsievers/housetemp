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
        self.hw.get_max_capacity.return_value = np.full(24, 20000.0) # 20k BTU max
        self.hw.get_cop.return_value = np.full(24, 3.0) # COP 3
        self.hw.defrost_risk_zone = None # Disable defrost for tests
        
        # Params: C, UA, K, Q_int, H_fac
        self.params = [5000, 200, 1000, 0, 5000]

    def test_load_comfort_schedule(self):
        # Create a dummy comfort.json
        comfort_data = {
            "schedule": [
                {
                    "weekdays": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                    "daily_schedule": [
                        {"time": "00:00", "temp": 60},
                        {"time": "08:00", "temp": 70},
                        {"time": "20:00", "temp": 60}
                    ]
                }
            ]
        }
        with open("test_comfort.json", "w") as f:
            json.dump(comfort_data, f)
            
        timestamps = pd.date_range("2023-01-01 00:00", "2023-01-01 23:59", freq="30T", tz="America/Los_Angeles")
        targets, config = schedule.load_comfort_schedule("test_comfort.json", timestamps)
        
        # Check lengths
        self.assertEqual(len(targets), len(timestamps))
        
        # Check values at specific times
        # 00:00 -> 60
        self.assertEqual(targets[0], 60)
        
        # 08:00 -> 70
        idx_8am = 16 # 8 * 2
        self.assertEqual(targets[idx_8am], 70)
        
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
        comfort_config = {"center_preference": 1.0, "mode": "heat"}
        
        optimized_setpoints = optimize.optimize_hvac_schedule(data, self.params, self.hw, target_temps, comfort_config, block_size_minutes=30)
        
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

    def test_setpoints_only_change_at_block_boundaries(self):
        """
        Behavioral test: Optimized setpoints should only change at control block 
        boundaries (e.g., every 30 minutes aligned to :00/:30), not at intermediate times.
        
        This ensures that the sensor output at 15-min intervals uses the value
        from the previous control block, not an interpolated value.
        """
        # Create 2-hour window at 5-minute resolution (24 points)
        # Starting at a non-aligned time to verify alignment behavior
        timestamps = pd.date_range("2023-01-01 00:07", periods=24, freq="5min")
        
        # Varying outdoor temp to encourage different setpoints per block
        t_out = np.concatenate([
            np.full(12, 40.0),  # First hour: cold (40F)
            np.full(12, 55.0)   # Second hour: warmer (55F)
        ])
        
        data = Measurements(
            timestamps=timestamps,
            t_in=np.full(24, 68.0),
            t_out=t_out,
            solar_kw=np.zeros(24),
            hvac_state=np.zeros(24),
            setpoint=np.full(24, 70.0),
            dt_hours=np.full(24, 5/60)  # 5-min steps
        )
        
        target_temps = np.full(24, 70.0)
        comfort_config = {"center_preference": 1.0, "mode": "heat"}
        
        optimized = optimize.optimize_hvac_schedule(
            data, self.params, self.hw, target_temps, comfort_config, 
            block_size_minutes=30
        )
        
        # Verify output length matches input
        self.assertEqual(len(optimized), len(timestamps))
        
        # Key behavioral check: Within each 30-minute block, all values should be identical.
        # Group indices by their 30-min block (floor to nearest 30 min)
        block_values = {}
        for i, ts in enumerate(timestamps):
            block_key = ts.floor("30min")
            if block_key not in block_values:
                block_values[block_key] = []
            block_values[block_key].append(optimized[i])
        
        # Within each block, all optimized setpoints must be the same
        for block, values in block_values.items():
            unique_vals = set(values)
            self.assertEqual(
                len(unique_vals), 1, 
                f"Block {block} has varying values {unique_vals} - should be constant within block"
            )
    
    def test_stepped_interpolation_no_linear_blending(self):
        """
        Behavioral test: Setpoints should use stepped (hold) interpolation,
        not linear interpolation between control points.
        
        If we have blocks A=68F and B=72F, intermediate simulation points
        should be exactly 68 or 72, never 70 (midpoint).
        """
        # Create 1-hour window at 5-minute resolution (12 points)
        timestamps = pd.date_range("2023-01-01 00:00", periods=12, freq="5min")
        
        # Create conditions that should result in different block setpoints
        # Cold at start, warmer at end
        t_out = np.linspace(35.0, 50.0, 12)
        
        data = Measurements(
            timestamps=timestamps,
            t_in=np.full(12, 65.0),
            t_out=t_out,
            solar_kw=np.zeros(12),
            hvac_state=np.zeros(12),
            setpoint=np.full(12, 70.0),
            dt_hours=np.full(12, 5/60)
        )
        
        target_temps = np.full(12, 70.0)
        comfort_config = {"center_preference": 1.0, "mode": "heat"}
        
        # Local mock with correct size
        hw = MagicMock()
        hw.get_max_capacity.return_value = np.full(12, 20000.0)
        hw.get_cop.return_value = np.full(12, 3.0)
        hw.defrost_risk_zone = None
        
        optimized = optimize.optimize_hvac_schedule(
            data, self.params, hw, target_temps, comfort_config,
            block_size_minutes=30
        )
        
        # All output values should be integers (rounded thermostat values)
        for val in optimized:
            self.assertEqual(val, round(val), f"Value {val} is not an integer")
        
        # Count unique values - should be small (1 per 30-min block, so max 2 for 1 hour)
        unique_values = set(optimized)
        self.assertLessEqual(
            len(unique_values), 2,
            f"Expected at most 2 unique values (one per 30-min block), got {len(unique_values)}: {unique_values}"
        )

    def test_output_aligned_to_clock_boundaries(self):
        """
        Behavioral test: Even with non-aligned input timestamps, the control
        blocks should align to natural clock boundaries (:00, :30).
        
        All simulation points within the same clock-aligned 30-min block
        should have identical setpoint values.
        """
        # Start at an odd time: 10:17
        timestamps = pd.date_range("2023-01-01 10:17", periods=18, freq="5min")
        # This spans 10:17 to 11:42 (85 minutes)
        # Block boundaries at clock times: 10:00, 10:30, 11:00, 11:30, 12:00
        # Our data covers blocks: 10:00-10:30, 10:30-11:00, 11:00-11:30, 11:30-12:00
        
        data = Measurements(
            timestamps=timestamps,
            t_in=np.full(18, 68.0),
            t_out=np.full(18, 45.0),
            solar_kw=np.zeros(18),
            hvac_state=np.zeros(18),
            setpoint=np.full(18, 70.0),
            dt_hours=np.full(18, 5/60)
        )
        
        target_temps = np.full(18, 70.0)
        comfort_config = {"center_preference": 1.0, "mode": "heat"}
        
        # Local mock with correct size
        hw = MagicMock()
        hw.get_max_capacity.return_value = np.full(18, 20000.0)
        hw.get_cop.return_value = np.full(18, 3.0)
        hw.defrost_risk_zone = None
        
        optimized = optimize.optimize_hvac_schedule(
            data, self.params, hw, target_temps, comfort_config,
            block_size_minutes=30
        )
        
        # Group simulation points by their clock-aligned 30-min block
        # Each group should have identical setpoint values
        block_values = {}
        for i, ts in enumerate(timestamps):
            # Floor to 30-min boundary (clock-aligned)
            block_key = ts.floor("30min")
            if block_key not in block_values:
                block_values[block_key] = []
            block_values[block_key].append(optimized[i])
        
        # Verify all points in each block have identical values
        for block, values in block_values.items():
            unique_vals = set(values)
            self.assertEqual(
                len(unique_vals), 1,
                f"Block {block} has varying values {unique_vals} - "
                f"all points within a clock-aligned block should be identical"
            )

if __name__ == '__main__':
    unittest.main()
