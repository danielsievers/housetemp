
import numpy as np
import pytest
from unittest.mock import MagicMock
from custom_components.housetemp.housetemp.energy import calculate_energy_stats
from custom_components.housetemp.housetemp.measurements import Measurements

def test_energy_conversion_factor():
    """
    Verify that BTU/hr is correctly converted to Watts/kWh.
    We test with inputs that should yield exactly 1.0 kWh if the 3.412 divisor is present.
    """
    # 1. Setup Data
    # 1 hour duration
    dt_hours = np.array([1.0])
    # Outdoor temp (arbitrary, used for COP lookup)
    t_out = np.array([50.0])
    
    # Create dummy Measurements (most fields unused by calculate_energy_stats except t_out, dt_hours)
    data = Measurements(
        timestamps=np.array([0]),
        t_in=np.array([70]),
        t_out=t_out,
        solar_kw=np.array([0]),
        hvac_state=np.array([1]),
        setpoint=np.array([72]),
        dt_hours=dt_hours,
        is_setpoint_fixed=None
    )
    
    # 2. Setup Mock Hardware
    hw = MagicMock()
    # Mock Max Capacity to be large so Part Load Ratio is small -> PLF ~ 1.4
    target_q = 3412.14 # 1 kWh = 3412.14 BTU
    
    hw.get_max_capacity.return_value = np.array([target_q]) 
    # Set Rated COP = 1.0
    hw.get_cop.return_value = np.array([1.0])
    hw.get_cooling_cop.return_value = np.array([1.0])
    # PLF curve: at full load (ratio=1), plf = 1.4 - 0.4 = 1.0
    hw.plf_low_load = 1.4
    hw.plf_slope = 0.4
    hw.idle_power_kw = 0.0
    hw.blower_active_kw = 0.0
    hw.defrost_risk_zone = None
    hw.plf_min = 0.5
    
    # 3. Setup Outputs
    hvac_outputs = np.array([target_q])
    
    # 4. Run Calculation
    # Expected Logic:
    #   load_ratio = 3412.14 / 3412.14 = 1.0
    #   plf = 1.4 - 0.4(1.0) = 1.0
    #   final_cop = 1.0 * 1.0 = 1.0
    #   watts_input = (3412.14 / 1.0) / 3.412 = 1000.0 Watts
    #   kwh = (1000.0 / 1000.0) * 1.0 hr = 1.0 kWh
    
    result = calculate_energy_stats(hvac_outputs, data, hw, h_factor=1000, cost_per_kwh=1.0, eff_derate=1.0)
    
    # Allow small float error
    assert result['total_kwh'] == pytest.approx(1.0, rel=1e-3)

def test_energy_partial_load():
    """ Verify calculation with partial load efficiency boost """
    dt_hours = np.array([1.0])
    t_out = np.array([50.0])
    data = Measurements(
        timestamps=np.array([0]),
        t_in=np.array([70]),
        t_out=t_out,
        solar_kw=np.array([0]),
        hvac_state=np.array([1]),
        setpoint=np.array([72]),
        dt_hours=dt_hours
    )
    
    hw = MagicMock()
    target_q = 3412.14
    
    # Set Max Cap = 4 * output => 25% load
    # load_ratio = 0.25
    # plf = 1.4 - 0.4(0.25) = 1.3
    hw.get_max_capacity.return_value = np.array([target_q * 4])
    hw.get_cop.return_value = np.array([1.0])
    hw.get_cooling_cop.return_value = np.array([1.0])
    hw.plf_low_load = 1.4
    hw.plf_slope = 0.4
    hw.idle_power_kw = 0.0
    hw.blower_active_kw = 0.0
    hw.defrost_risk_zone = None
    hw.plf_min = 0.5
    
    hvac_outputs = np.array([target_q])
    
    result = calculate_energy_stats(hvac_outputs, data, hw, cost_per_kwh=1.0, eff_derate=1.0)
    
    # Expected:
    # final_cop = 1.0 * 1.3 = 1.3
    # watts_input = 3412.14 / 1.3 / 3.412 = 1000 / 1.3 = 769.23 Watts
    # kwh = 0.76923
    
    expected_kwh = 1.0 / 1.3
    assert result['total_kwh'] == pytest.approx(expected_kwh, rel=1e-3)

def test_energy_derate():
    """Test with an efficiency derate (e.g., duct losses)."""
    # Create output that requires 10,000 BTU input (if COP=1)
    # With COP=3 and derate=0.8, input should be higher.
    hvac_outputs = np.array([3000.0]) # 1 hour
    data = MagicMock()
    data.dt_hours = np.array([1.0])
    data.t_out = np.array([50.0])
    data.hvac_state = np.array([1])
    data.__len__.return_value = 1
    
    hw = MagicMock()
    hw.get_max_capacity.return_value = np.array([20000.0])
    hw.get_cop.return_value = np.array([3.0])
    hw.get_cooling_cop.return_value = np.array([3.0])
    hw.plf_low_load = 1.0 # Simplify (no PLF)
    hw.plf_slope = 0.0
    hw.idle_power_kw = 0.0
    hw.blower_active_kw = 0.0
    hw.defrost_risk_zone = None
    hw.plf_min = 0.5
    
    # 1. Base case: Derate = 1.0
    res_base = calculate_energy_stats(hvac_outputs, data, hw, h_factor=5000, eff_derate=1.0)
    kwh_base = res_base['total_kwh']
    
    # 2. Derated case: Derate = 0.8
    res_derate = calculate_energy_stats(hvac_outputs, data, hw, h_factor=5000, eff_derate=0.8)
    kwh_derate = res_derate['total_kwh']
    
    # With derate=0.8, we expect usage to be 1/0.8 = 1.25x higher
    ratio = kwh_derate / kwh_base
    assert ratio == pytest.approx(1.25, abs=1e-5)

def test_energy_steps_returned():
    """Verify that per-step energy is returned and sums to total."""
    dt_hours = np.array([1.0, 1.0])
    hvac_outputs = np.array([3412.14, 3412.14 * 2])
    data = MagicMock()
    data.dt_hours = dt_hours
    data.t_out = np.array([50.0, 50.0])
    data.hvac_state = np.array([1, 1])
    data.__len__.return_value = 2
    
    hw = MagicMock()
    hw.get_max_capacity.return_value = np.array([10000.0, 10000.0])
    # COP=1.0 for easy math
    hw.get_cop.return_value = np.array([1.0, 1.0])
    hw.get_cooling_cop.return_value = np.array([1.0, 1.0])
    hw.plf_low_load = 1.0
    hw.plf_slope = 0.0
    hw.idle_power_kw = 0.0
    hw.blower_active_kw = 0.0
    hw.defrost_risk_zone = None
    hw.plf_min = 0.5
    
    result = calculate_energy_stats(hvac_outputs, data, hw, h_factor=1000, cost_per_kwh=1.0, eff_derate=1.0)
    
    # Check keys
    assert 'kwh_steps' in result
    assert 'total_kwh' in result
    
    kwh_steps = result['kwh_steps']
    total_kwh = result['total_kwh']
    
    # Verify length
    assert len(kwh_steps) == 2
    
    # Verify values
    # Step 1: 3412.14 BTU = 1 kWh
    # Step 2: 6824.28 BTU = 2 kWh
    assert kwh_steps[0] == pytest.approx(1.0, rel=1e-3)
    assert kwh_steps[1] == pytest.approx(2.0, rel=1e-3)
    
    # Verify sum integrity
    assert np.sum(kwh_steps) == pytest.approx(total_kwh, rel=1e-6)
