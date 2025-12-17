
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
    # Wait, we want controlled COP.
    # Logic: 
    #   load_ratio = output / max_cap
    #   plf = 1.4 - 0.4 * load_ratio
    #   final_cop = rated_cop * plf
    # Let's make load_ratio = 1.0 so plf = 1.0.
    # Set max_cap = output.
    
    target_q = 3412.14 # 1 kWh = 3412.14 BTU
    
    hw.get_max_capacity.return_value = np.array([target_q]) 
    # Set Rated COP = 1.0
    hw.get_cop.return_value = np.array([1.0])
    # PLF curve: at full load (ratio=1), plf = 1.4 - 0.4 = 1.0
    hw.plf_low_load = 1.4
    hw.plf_slope = 0.4
    
    # 3. Setup Outputs
    hvac_outputs = np.array([target_q])
    
    # 4. Run Calculation
    # Expected Logic:
    #   load_ratio = 3412.14 / 3412.14 = 1.0
    #   plf = 1.4 - 0.4(1.0) = 1.0
    #   final_cop = 1.0 * 1.0 = 1.0
    #   watts_input = (3412.14 / 1.0) / 3.412 = 1000.0 Watts
    #   kwh = (1000.0 / 1000.0) * 1.0 hr = 1.0 kWh
    
    result = calculate_energy_stats(hvac_outputs, data, hw, h_factor=1000, cost_per_kwh=1.0)
    
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
    hw.plf_low_load = 1.4
    hw.plf_slope = 0.4
    
    hvac_outputs = np.array([target_q])
    
    result = calculate_energy_stats(hvac_outputs, data, hw, cost_per_kwh=1.0)
    
    # Expected:
    # final_cop = 1.0 * 1.3 = 1.3
    # watts_input = 3412.14 / 1.3 / 3.412 = 1000 / 1.3 = 769.23 Watts
    # kwh = 0.76923
    
    expected_kwh = 1.0 / 1.3
    assert result['total_kwh'] == pytest.approx(expected_kwh, rel=1e-3)
