
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from scipy.optimize import OptimizeResult
from custom_components.housetemp.housetemp.optimize import optimize_hvac_schedule
from custom_components.housetemp.const import DOMAIN

class MockHeatPump:
    def __init__(self):
        self.min_output_btu_hr = 5000
        self.max_cool_btu_hr = 12000
        self.defrost_risk_zone = None
    
    def get_max_capacity(self, temps):
        return np.full_like(temps, 12000.0)
    
    def get_cop(self, temps):
        return np.full_like(temps, 3.5)
    
    def get_cooling_cop(self, temps):
        return np.full_like(temps, 3.0)

@pytest.fixture
def mock_heat_pump():
    return MockHeatPump()

@pytest.fixture
def mock_data():
    steps = 10
    return MagicMock(
        timestamps=np.array([0]*steps),
        t_out=np.full(steps, 50.0),
        solar_kw=np.zeros(steps),
        dt_hours=np.full(steps, 0.5), # 30 min
        t_in=np.array([70.0]),
        hvac_state=np.ones(steps, dtype=int),
        setpoint=np.full(steps, 70.0),
        is_setpoint_fixed=None
    )

def test_optimize_crash_missing_nit(mock_data, mock_heat_pump):
    """
    Test that optimize_hvac_schedule handles a scipy result missing 'nit'
    (which happens when L-BFGS-B determines all variables are fixed by bounds).
    """
    
    # 1. Setup Inputs
    # 5 params: C, UA, K, Q, H, Eff
    params = [5000, 500, 1000, 2000, 10000, 0.9]
    target_temps = np.full(10, 70.0)
    comfort_config = {
        'mode': 'heat', 
        'min_setpoint': 60, 
        'max_setpoint': 80
    }
    
    # 2. Mock 'minimize' to return a result WITHOUT 'nit'
    # This simulates the exact crash condition
    mock_res = OptimizeResult(
        x=np.full(10, 70.0),
        success=True,
        fun=0.0,
        message=b'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL',
        nfev=1
        # NO 'nit' here!
    )
    
    with patch("custom_components.housetemp.housetemp.optimize.minimize", return_value=mock_res):
        # 3. Method Under Test
        # We pass fixed_mask=None, but the mock minimize is what matters.
        # Actually, optimize_hvac_schedule calls run_model inside, 
        # so we need to ensure run_model_continuous doesn't crash or return garbage.
        # But we are mocking minimize, so the cost function might not even run if we return immediately?
        # Wait, optimize_hvac_schedule calls minimize. minimize calls the cost function.
        # If we mock minimize, the cost function isn't called by scipy.
        # So we just test that optimize_hvac_schedule handles the return value of minimize.
        
        # We need to mock run_model_continuous because verification step calls it AFTER optimization
        with patch("custom_components.housetemp.housetemp.optimize.run_model_continuous") as mock_run:
            # Mock verification return (temps, output, produced)
            mock_run.return_value = (
                [70.0]*10, # temps
                [0.0]*10,  # output
                [0.0]*10   # produced
            )
            
            result, debug_info = optimize_hvac_schedule(
                mock_data, 
                params, 
                mock_heat_pump, 
                target_temps, 
                comfort_config,
                block_size_minutes=30,
                enable_multiscale=False
            )
            
            # 4. Assertions
            assert result is not None
            assert debug_info['success'] is True
            # Check that it safely defaulted
            assert debug_info.get('iterations') is None or debug_info.get('iterations') == -1 or debug_info.get('iterations') == 0
            
