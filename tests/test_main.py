import unittest
import os
import shutil
import tempfile
import pandas as pd
import numpy as np
import json
from unittest.mock import patch
import sys

# Import the main module to test
# We need to make sure the parent directory is in sys.path if running from tests/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import main

class TestMainIntegration(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.test_dir, 'test_data.csv')
        self.csv_path = os.path.join(self.test_dir, 'test_data.csv')
        self.model_path = os.path.join(self.test_dir, 'test_model.json')
        self.hp_path = os.path.join(self.test_dir, 'test_hp.json')
        
        # Create Dummy Heat Pump Config
        hp_data = {
            "description": "Test Heat Pump",
            "max_capacity": {
                "x_outdoor_f": [-10, 100],
                "y_btu_hr": [50000, 50000]
            },
            "cop": {
                "x_outdoor_f": [-10, 100],
                "y_cop": [3.0, 3.0]
            }
        }
        with open(self.hp_path, 'w') as f:
            json.dump(hp_data, f)
        self.hp_path = os.path.join(self.test_dir, 'test_hp.json')
        
        # Create Dummy Heat Pump Config
        hp_data = {
            "description": "Test Heat Pump",
            "max_capacity": {
                "x_outdoor_f": [-10, 100],
                "y_btu_hr": [50000, 50000]
            },
            "cop": {
                "x_outdoor_f": [-10, 100],
                "y_cop": [3.0, 3.0]
            }
        }
        with open(self.hp_path, 'w') as f:
            json.dump(hp_data, f)
        
        # Generate Test Data (Ported from generate_csv.py)
        self.generate_test_data()

        # Conditional Plot Suppression
        # If SHOW_PLOTS env var is NOT set, suppress plots.
        if not os.environ.get('SHOW_PLOTS'):
            self.plot_patcher = patch('matplotlib.pyplot.show')
            self.mock_show = self.plot_patcher.start()
        else:
            self.plot_patcher = None

    def tearDown(self):
        # Stop patcher if it was started
        if self.plot_patcher:
            self.plot_patcher.stop()
            
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def generate_test_data(self):
        # --- CONFIGURATION ---
        # Create 48 hours of data (30 min intervals)
        timestamps = pd.date_range(start="2025-01-15 00:00", periods=96, freq="30min")

        # 1. Outdoor Temp: Sinusoidal wave between 40F (night) and 60F (day)
        hours = timestamps.hour + (timestamps.minute / 60)
        t_out = 50 - 10 * np.cos((hours - 3) * np.pi / 12)

        # 2. Solar KW: Bell curve active between 7am and 5pm
        solar = np.zeros(len(timestamps))
        daylight_mask = (hours > 7) & (hours < 17)
        solar[daylight_mask] = 4.0 * np.sin((hours[daylight_mask] - 7) * np.pi / 10)
        solar = np.maximum(0, solar)

        # 3. Target Temp (Thermostat Schedule)
        targets = np.where((hours >= 7) & (hours < 21), 68.0, 62.0)

        # 4. Simulate Indoor Temp & HVAC State
        t_in = np.zeros(len(timestamps))
        hvac_mode = np.zeros(len(timestamps))
        current_temp = 66.0

        for i in range(len(timestamps)):
            t_in[i] = current_temp
            
            if current_temp < targets[i] - 0.5:
                mode = 1 # Heat
            elif current_temp > targets[i] + 0.5:
                mode = 0 # Off
            else:
                mode = 0 # Deadband
                
            hvac_mode[i] = mode
            
            # Physics
            loss = 310 * (t_out[i] - current_temp)
            sun = solar[i] * 17000
            heat = mode * 45000
            delta_T = (loss + sun + heat) * 0.5 / 8000
            current_temp += delta_T

        # Export to CSV
        df = pd.DataFrame({
            'time': timestamps,
            'indoor_temp': np.round(t_in, 2),
            'outdoor_temp': np.round(t_out, 2),
            'solar_kw': np.round(solar, 3),
            'hvac_mode': hvac_mode.astype(int),
            'target_temp': targets
        })
        
        df.to_csv(self.csv_path, index=False)

    def test_use_regression(self):
        """Test running with -r flag (Initialize with Regression)"""
        # Suppress stdout/stderr to keep test output clean
        with open(os.devnull, 'w') as devnull:
            with patch('sys.stdout', new=devnull):
                # This should now run regression AND optimization
                status = main.run_main([self.csv_path, '-r', '-o', self.model_path, '--heat-pump', self.hp_path])
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(self.model_path))

    def test_full_optimization_defaults(self):
        """Test full optimization run (Defaults, No Regression)"""
        with open(os.devnull, 'w') as devnull:
            with patch('sys.stdout', new=devnull):
                status = main.run_main([self.csv_path, '-o', self.model_path, '--heat-pump', self.hp_path])
        
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(self.model_path), "Model file should be created")
        
        # Verify model content
        with open(self.model_path, 'r') as f:
            data = json.load(f)
            self.assertIn('C_thermal', data)
            self.assertIn('UA_overall', data)
            self.assertIn('K_solar', data)
            self.assertIn('Q_int', data)
            self.assertIn('H_factor', data)

    def test_prediction(self):
        """Test prediction with -p flag"""
        # First create a model
        with open(os.devnull, 'w') as devnull:
            with patch('sys.stdout', new=devnull):
                main.run_main([self.csv_path, '-o', self.model_path, '--heat-pump', self.hp_path])
        
        # Now run prediction
        with open(os.devnull, 'w') as devnull:
            with patch('sys.stdout', new=devnull):
                status = main.run_main([self.csv_path, '-p', self.model_path, '--heat-pump', self.hp_path])
        
        self.assertEqual(status, 0)

    def test_prediction_forecast(self):
        """Test prediction with forecast data - Requires --start-temp"""
        forecast_csv = os.path.join(self.test_dir, 'forecast.csv')
        df = pd.DataFrame({
            'time': pd.date_range(start="2025-02-01 00:00", periods=48, freq="30min"),
            'outdoor_temp': np.full(48, 50.0),
            'solar_kw': np.zeros(48)
        })
        df.to_csv(forecast_csv, index=False)
        
        # Create model
        with open(os.devnull, 'w') as devnull:
            with patch('sys.stdout', new=devnull):
                main.run_main([self.csv_path, '-o', self.model_path, '--heat-pump', self.hp_path])

        # Run with --start-temp (Should Succeed)
        with open(os.devnull, 'w') as devnull:
            with patch('sys.stdout', new=devnull):
                status = main.run_main([forecast_csv, '-p', self.model_path, '--start-temp', '68.0', '--heat-pump', self.hp_path])
        self.assertEqual(status, 0)

    def test_prediction_forecast_missing_start_temp(self):
        """Test prediction with forecast data MISSING --start-temp (Should Fail)"""
        forecast_csv = os.path.join(self.test_dir, 'forecast.csv')
        df = pd.DataFrame({
            'time': pd.date_range(start="2025-02-01 00:00", periods=48, freq="30min"),
            'outdoor_temp': np.full(48, 50.0),
            'solar_kw': np.zeros(48)
        })
        df.to_csv(forecast_csv, index=False)
        
        # Create model
        with open(os.devnull, 'w') as devnull:
            with patch('sys.stdout', new=devnull):
                main.run_main([self.csv_path, '-o', self.model_path, '--heat-pump', self.hp_path])

        # Run WITHOUT --start-temp (Should Fail)
        # We expect ValueError from load_csv, which main.py doesn't catch explicitly, so it crashes.
        # unittest handles crashes as errors, but we want to verify it raises ValueError.
        with self.assertRaises(ValueError):
             with open(os.devnull, 'w') as devnull:
                with patch('sys.stdout', new=devnull):
                    main.run_main([forecast_csv, '-p', self.model_path, '--heat-pump', self.hp_path])

    def test_prediction_duration(self):
        """Test prediction with --duration argument"""
        # Create model
        with open(os.devnull, 'w') as devnull:
            with patch('sys.stdout', new=devnull):
                main.run_main([self.csv_path, '-o', self.model_path, '--heat-pump', self.hp_path])
        
        # Run prediction with duration=60 minutes (2 steps)
        # We can't easily check the output length from integration test without capturing stdout or inspecting internals
        # But we can check it runs successfully
        with open(os.devnull, 'w') as devnull:
            with patch('sys.stdout', new=devnull):
                status = main.run_main([self.csv_path, '-p', self.model_path, '--duration', '60', '--heat-pump', self.hp_path])
        self.assertEqual(status, 0)

    def test_rolling_evaluation(self):
        """Test rolling evaluation with -e flag"""
        # Create model
        with open(os.devnull, 'w') as devnull:
            with patch('sys.stdout', new=devnull):
                main.run_main([self.csv_path, '-o', self.model_path, '--heat-pump', self.hp_path])
        
        # Run rolling evaluation
        # We need enough data for at least one 12h window.
        # Our test data is 48 hours, so it should be fine.
        with open(os.devnull, 'w') as devnull:
            with patch('sys.stdout', new=devnull):
                status = main.run_main([self.csv_path, '-e', self.model_path, '--heat-pump', self.hp_path])
        self.assertEqual(status, 0)

if __name__ == '__main__':
    unittest.main()
