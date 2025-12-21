import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from housetemp import optimize, energy

class TestTrueOffSemantics(unittest.TestCase):
    def setUp(self):
        # Create minimal 1-hour data
        self.timestamps = pd.date_range("2023-01-01 00:00", periods=2, freq="30min")
        self.dt_hours = np.full(2, 0.5)
        self.t_out = np.full(2, 30.0) # Cold outside
        self.t_in = np.full(2, 60.0) # Start cold
        
        # Hardware with SIGNIFICANT idle/blower power
        self.hw = MagicMock()
        self.hw.get_max_capacity.return_value = np.full(2, 20000.0)
        self.hw.get_cop.return_value = np.full(2, 3.0)
        self.hw.get_cooling_cop.return_value = np.full(2, 3.0)
        self.hw.min_output_btu_hr = 3000
        self.hw.max_cool_btu_hr = 54000
        self.hw.idle_power_kw = 0.5 # High idle power for visibility
        self.hw.blower_active_kw = 0.5 # High blower power
        self.hw.defrost_risk_zone = None

    def test_true_off_suppresses_idle_power(self):
        """
        Verify that passing hvac_states=0 suppresses idle power,
        mirroring the upstream True-Off logic.
        """
        # Scenario: Heating Mode
        
        # Case A: Enabled (+1) -> Should Charge Idle/Blower
        hvac_states_enabled = np.array([1, 1]) 
        hvac_produced_zero = np.zeros(2) 
        
        res_active = energy.calculate_energy_vectorized(
            hvac_outputs=hvac_produced_zero, 
            dt_hours=self.dt_hours,
            max_caps=self.hw.get_max_capacity(self.t_out),
            base_cops=self.hw.get_cop(self.t_out),
            hw=self.hw,
            hvac_states=hvac_states_enabled
        )
        
        # Expectation: 0.5 kWh (Idle)
        self.assertAlmostEqual(res_active['kwh'], 0.5, delta=0.01, 
            msg="Failed to charge idle power when enabled")

        # Case B: True Off (0) -> Should NOT Charge Idle
        hvac_states_off = np.array([0, 0]) # Manually set to 0 (simulating upstream logic)
        
        res_true_off = energy.calculate_energy_vectorized(
            hvac_outputs=hvac_produced_zero,
            dt_hours=self.dt_hours,
            max_caps=self.hw.get_max_capacity(self.t_out),
            base_cops=self.hw.get_cop(self.t_out),
            hw=self.hw,
            hvac_states=hvac_states_off
        )
        
        self.assertAlmostEqual(res_true_off['kwh'], 0.0, delta=0.0001,
            msg="Failed to suppress idle power when hvac_states=0")

    def test_true_off_cooling_mode(self):
        """Verify True-Off works symmetrically for Cooling (0 State)."""
        hvac_produced_zero = np.zeros(2)
        
        # Case A: Active (-1) -> Idle Charge
        hvac_states_cool = np.array([-1, -1])
        res_active = energy.calculate_energy_vectorized(
            hvac_outputs=hvac_produced_zero,
            dt_hours=self.dt_hours,
            max_caps=self.hw.get_max_capacity(self.t_out),
            base_cops=self.hw.get_cooling_cop(self.t_out),
            hw=self.hw,
            hvac_states=hvac_states_cool
        )
        self.assertAlmostEqual(res_active['kwh'], 0.5, delta=0.01)

        # Case B: True Off (0) -> No Charge
        hvac_states_off = np.array([0, 0])
        res_true_off = energy.calculate_energy_vectorized(
            hvac_outputs=hvac_produced_zero,
            dt_hours=self.dt_hours,
            max_caps=self.hw.get_max_capacity(self.t_out),
            base_cops=self.hw.get_cooling_cop(self.t_out),
            hw=self.hw,
            hvac_states=hvac_states_off
        )
        self.assertAlmostEqual(res_true_off['kwh'], 0.0, delta=0.0001)

if __name__ == '__main__':
    unittest.main()
