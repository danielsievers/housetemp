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
        Verify that pinning setpoint to min_setpoint (Heat) suppresses idle & blower power
        even when hvac_states is 'enabled' (+1).
        """
        # Scenario: Heating Mode
        hvac_mode_val = 1
        min_setpoint = 60.0
        max_setpoint = 75.0
        off_eps = 0.1
        
        # Case A: Enabled (+1) and Active Setpoint (70F) -> Should Charge Idle/Blower
        # ---------------------------------------------------------------------------
        hvac_states = np.array([1, 1]) # Fully Enabled Intent
        setpoints_active = np.array([70.0, 70.0]) # Well above min
        # Produced heat must be > 0 ideally, but for this accounting test 
        # we can just zero it to check *idle* adder specifically?
        # Energy calc logic:
        # is_active = (produced > TOL) & enabled
        # is_idle = (~is_active) & enabled
        # If we pass produced=0, it should be IDLE.
        hvac_produced_zero = np.zeros(2) 
        
        res_active = energy.calculate_energy_vectorized(
            hvac_outputs=hvac_produced_zero, # 0 output -> Idle
            dt_hours=self.dt_hours,
            max_caps=self.hw.get_max_capacity(self.t_out),
            base_cops=self.hw.get_cop(self.t_out),
            hw=self.hw,
            hvac_states=hvac_states,
            setpoints=setpoints_active, # 70F
            hvac_mode_val=hvac_mode_val,
            min_setpoint=min_setpoint,
            max_setpoint=max_setpoint,
            off_intent_eps=off_eps
        )
        
        # Expectation: 
        # Produced = 0 -> Watts = 0
        # Idle = True (Enabled & Not Active) -> Watts += 0.5 kW
        # Total kWh = 0.5 kW * 1 hr (2x30min) = 0.5 kWh
        self.assertAlmostEqual(res_active['kwh'], 0.5, delta=0.01, 
            msg="Failed to charge idle power when enabled and setpoint is active")

        # Case B: Enabled (+1) BUT Pinned Setpoint (60F) -> Should BE TRUE OFF (0 kWh)
        # ---------------------------------------------------------------------------
        setpoints_pinned = np.array([60.0, 60.0]) # Exactly min_setpoint
        
        res_true_off = energy.calculate_energy_vectorized(
            hvac_outputs=hvac_produced_zero,
            dt_hours=self.dt_hours,
            max_caps=self.hw.get_max_capacity(self.t_out),
            base_cops=self.hw.get_cop(self.t_out),
            hw=self.hw,
            hvac_states=hvac_states, # STILL ENABLED INTENT!
            setpoints=setpoints_pinned, # Pinned -> Off Intent
            hvac_mode_val=hvac_mode_val,
            min_setpoint=min_setpoint,
            max_setpoint=max_setpoint,
            off_intent_eps=off_eps
        )
        
        # Expectation:
        # Pinned setpoint triggers off_intent.
        # off_intent suppresses is_enabled.
        # So is_idle becomes False.
        # Total kWh should be 0.0
        self.assertAlmostEqual(res_true_off['kwh'], 0.0, delta=0.0001,
            msg="Failed to suppress idle power when setpoint pinned to boundary (True Off)")

    def test_true_off_cooling_mode(self):
        """Verify True-Off works symmetrically for Cooling (Pinned to Max)."""
        hvac_mode_val = -1
        min_setpoint = 60.0
        max_setpoint = 75.0
        off_eps = 0.1
        
        hvac_states = np.array([-1, -1])
        hvac_produced_zero = np.zeros(2)
        
        # Case A: Active Setpoint (70F) -> Idle Charge
        setpoints_active = np.array([70.0, 70.0]) 
        res_active = energy.calculate_energy_vectorized(
            hvac_outputs=hvac_produced_zero,
            dt_hours=self.dt_hours,
            max_caps=self.hw.get_max_capacity(self.t_out),
            base_cops=self.hw.get_cooling_cop(self.t_out), # Use cooling COP mock
            hw=self.hw,
            hvac_states=hvac_states,
            setpoints=setpoints_active,
            hvac_mode_val=hvac_mode_val,
            min_setpoint=min_setpoint,
            max_setpoint=max_setpoint,
            off_intent_eps=off_eps
        )
        self.assertAlmostEqual(res_active['kwh'], 0.5, delta=0.01)

        # Case B: Pinned Setpoint (75F) -> True Off
        setpoints_pinned = np.array([75.0, 75.0])
        res_true_off = energy.calculate_energy_vectorized(
            hvac_outputs=hvac_produced_zero,
            dt_hours=self.dt_hours,
            max_caps=self.hw.get_max_capacity(self.t_out),
            base_cops=self.hw.get_cooling_cop(self.t_out),
            hw=self.hw,
            hvac_states=hvac_states,
            setpoints=setpoints_pinned, # Pinned to Max
            hvac_mode_val=hvac_mode_val,
            min_setpoint=min_setpoint,
            max_setpoint=max_setpoint,
            off_intent_eps=off_eps
        )
        self.assertAlmostEqual(res_true_off['kwh'], 0.0, delta=0.0001)

if __name__ == '__main__':
    unittest.main()
