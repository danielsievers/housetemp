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
        self.model_path = os.path.join(self.test_dir, 'test_model.json')
        
        # Generate Test Data (Ported from generate_csv.py)
        self.generate_test_data()

    def tearDown(self):
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

    def test_linear_regression_only(self):
        """Test running with -r flag (Regression Only)"""
        # Suppress stdout/stderr to keep test output clean
        with patch('sys.stdout', new=open(os.devnull, 'w')):
            status = main.run_main([self.csv_path, '-r'])
        self.assertEqual(status, 0)

    def test_full_optimization(self):
        """Test full optimization run"""
        with patch('sys.stdout', new=open(os.devnull, 'w')):
            status = main.run_main([self.csv_path, '-o', self.model_path])
        
        self.assertEqual(status, 0)
        self.assertTrue(os.path.exists(self.model_path), "Model file should be created")
        
        # Verify model content
        with open(self.model_path, 'r') as f:
            data = json.load(f)
            self.assertIn('raw_params', data)
            self.assertEqual(len(data['raw_params']), 5)

    def test_energy_estimation(self):
        """Test energy estimation with -e flag"""
        # First create a model
        with patch('sys.stdout', new=open(os.devnull, 'w')):
            main.run_main([self.csv_path, '-o', self.model_path])
        
        # Now run estimation
        with patch('sys.stdout', new=open(os.devnull, 'w')):
            status = main.run_main([self.csv_path, '-e', self.model_path])
        
        self.assertEqual(status, 0)

if __name__ == '__main__':
    unittest.main()
