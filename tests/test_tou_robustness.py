import pytest
import numpy as np
from datetime import datetime, timezone
from custom_components.housetemp.housetemp.optimize import optimize_hvac_schedule
from custom_components.housetemp.housetemp.measurements import Measurements
from custom_components.housetemp.housetemp.run_model import HeatPump

@pytest.fixture
def mock_context():
    # 4 steps = 1 hour at 15 min steps
    ts = [datetime(2023, 1, 1, i, 0, tzinfo=timezone.utc) for i in range(4)]
    data = Measurements(
        timestamps=np.array(ts),
        t_in=np.array([70.0, 70.0, 70.0, 70.0]),
        t_out=np.array([40.0, 40.0, 40.0, 40.0]),
        solar_kw=np.zeros(4),
        hvac_state=np.array([1, 1, 1, 1]),
        setpoint=np.array([70.0, 70.0, 70.0, 70.0]),
        dt_hours=np.array([0.25, 0.25, 0.25, 0.25]),
        is_setpoint_fixed=np.zeros(4, dtype=bool),
        target_temp=np.array([70.0, 70.0, 70.0, 70.0])
    )
    params = [10000.0, 750.0, 3000.0, 2000.0, 5000.0, 1.0]
    from unittest.mock import MagicMock
    hw = MagicMock(spec=HeatPump)
    hw.min_output_btu_hr = 1000.0
    hw.max_cool_btu_hr = 30000.0
    hw.get_max_capacity.return_value = np.array([30000.0, 30000.0, 30000.0, 30000.0])
    hw.get_cop.return_value = np.array([3.0, 3.0, 3.0, 3.0])
    hw.get_cooling_cop.return_value = np.array([3.0, 3.0, 3.0, 3.0])
    hw.defrost_risk_zone = None
    comfort = {'mode': 'heat', 'center_preference': 1.0}
    return data, params, hw, comfort

def test_optimize_handles_mismatched_rate_length(mock_context):
    data, params, hw, comfort = mock_context
    # Input has 4 steps, give it 3 rates
    bad_rates = np.array([1.0, 1.3, 1.0])
    
    # Should not crash. Should log warning and fall back to 1.0.
    result, info = optimize_hvac_schedule(data, params, hw, data.target_temp, comfort, rate_per_step=bad_rates)
    
    assert result is not None
    assert info['success'] is True

def test_optimize_handles_negative_rates(mock_context):
    data, params, hw, comfort = mock_context
    # Give it a negative rate
    bad_rates = np.array([1.0, -1.0, 1.0, 1.0])
    
    # Should not crash. Should clip to 0.
    result, info = optimize_hvac_schedule(data, params, hw, data.target_temp, comfort, rate_per_step=bad_rates)
    
    assert result is not None
    assert info['success'] is True

def test_optimize_handles_nan_rates(mock_context):
    data, params, hw, comfort = mock_context
    # Give it a NaN rate
    bad_rates = np.array([1.0, np.nan, 1.0, 1.0])
    
    # Should not crash. Should fall back to uniform.
    result, info = optimize_hvac_schedule(data, params, hw, data.target_temp, comfort, rate_per_step=bad_rates)
    
    assert result is not None
    assert info['success'] is True
