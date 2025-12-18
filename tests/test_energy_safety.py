import unittest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../custom_components/housetemp')))

from housetemp import energy

class TestEnergySafety(unittest.TestCase):
    def test_zero_max_cap_fallback(self):
        """Test fallback to 1.0 BTU max_cap when 0.0 is provided."""
        # Setup: 1 step, Output=5000, MaxCap=0
        hvac_outputs = np.array([5000.0])
        dt_hours = np.array([1.0])
        max_caps_bad = np.array([0.0]) 
        base_cops = np.array([3.0])
        hvac_states = np.array([1])
        
        div_hw = MagicMock()
        div_hw.plf_low_load = 1.4
        div_hw.plf_slope = 0.4
        div_hw.plf_min = 0.5
        div_hw.idle_power_kw = 0.0
        div_hw.blower_active_kw = 0.0

        # Run
        res = energy.calculate_energy_vectorized(
            hvac_outputs, dt_hours, max_caps_bad, base_cops, div_hw, 
            eff_derate=1.0, hvac_states=hvac_states
        )
        
        # Verify Fallback Logic:
        # Load Ratio = 5000 / 1.0 = 5000
        # PLF = 1.4 - 2000 = -Huge -> Clamped to 0.5
        # COP = 3.0 * 0.5 = 1.5
        # Watts = (5000 / 1.5) / 3.412 = 3333.3 / 3.412 = 976.9 W
        # kWh = 0.9769
        
        self.assertAlmostEqual(res['kwh'], 0.9769, places=3)
        self.assertFalse(np.isnan(res['kwh']))

    def test_low_cop_handling(self):
        """
        Verify:
        1. Warns but allows 0.068 (since > 1e-3).
        2. Clamps 0.0001 to 1e-3.
        """
        # Case A: 0.068 (Implausible but Numeric Valid)
        # LoadRatio=0.1, PLF=1.36. Base=0.05 -> Final=0.068
        hvac_outputs = np.array([1000.0])
        dt_hours = np.array([1.0])
        max_caps = np.array([10000.0])
        base_cops_warn = np.array([0.05]) 
        
        hw = MagicMock()
        hw.plf_low_load = 1.4
        hw.plf_slope = 0.4
        hw.plf_min = 0.5 # Defaults
        hw.idle_power_kw = 0.0
        hw.blower_active_kw = 0.0
        
        res_warn = energy.calculate_energy_vectorized(hvac_outputs, dt_hours, max_caps, base_cops_warn, hw)
        
        # Exp Watts: (1000 / 0.068) / 3.412 = 4310 W
        exp_kwh_warn = (4310.2 / 1000.0)
        self.assertAlmostEqual(res_warn['kwh'], exp_kwh_warn, places=2)

        # Case B: 0.00001 (Numeric Hazard)
        # Base=0.00001 -> Final ~ 0.0000136. Should clamp to 0.001.
        base_cops_crit = np.array([1e-5])
        res_crit = energy.calculate_energy_vectorized(hvac_outputs, dt_hours, max_caps, base_cops_crit, hw)
        
        # Exp Watts: (1000 / 0.001) / 3.412 = 293083 W = 293 kW
        exp_kwh_crit = 293.08
        self.assertAlmostEqual(res_crit['kwh'], exp_kwh_crit, places=1)

    def test_plf_clipping(self):
        """Verify PLF clamps when Load Ratio is Extreme."""
        # Case: Massive Overload (Load Ratio >> 1)
        hvac_outputs = np.array([50000.0]) # 50k
        max_caps = np.array([10000.0])     # 10k -> Ratio 5.0
        # PLF = 1.4 - (0.4 * 5.0) = 1.4 - 2.0 = -0.6
        # Start with default min 0.5
        
        hw = MagicMock()
        hw.plf_low_load = 1.4
        hw.plf_slope = 0.4
        hw.plf_min = 0.5
        hw.idle_power_kw = 0.0
        hw.blower_active_kw = 0.0
        
        base_cops = np.array([3.0])
        dt = np.array([1.0])
        
        res = energy.calculate_energy_vectorized(hvac_outputs, dt, max_caps, base_cops, hw)
        
        # Expect PLF = 0.5 (Clamped)
        # COP = 3.0 * 0.5 = 1.5
        # Watts = (50000 / 1.5) / 3.412 = 9769 W
        self.assertAlmostEqual(res['kwh'], 9.769, places=2)
        
        # Case: Change Min to 0.1
        hw.plf_min = 0.1
        res2 = energy.calculate_energy_vectorized(hvac_outputs, dt, max_caps, base_cops, hw)
        
        # Expect PLF = 0.1 (Clamped)
        # COP = 0.3
        # Watts = (50000 / 0.3) / 3.412 = 48847 W
        self.assertAlmostEqual(res2['kwh'], 48.85, places=1)

if __name__ == '__main__':
    unittest.main()
