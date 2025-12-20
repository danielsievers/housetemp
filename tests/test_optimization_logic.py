"""Tests for optimization logic in the coordinator."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta, timezone
import numpy as np

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.housetemp.const import (
    DOMAIN,
    CONF_SENSOR_INDOOR_TEMP,
    CONF_WEATHER_ENTITY,
    CONF_C_THERMAL,
    CONF_UA,
    CONF_K_SOLAR,
    CONF_Q_INT,
    CONF_H_FACTOR,
    CONF_HVAC_MODE,
    CONF_CENTER_PREFERENCE,
    CONF_AVOID_DEFROST,
)
from custom_components.housetemp.coordinator import HouseTempCoordinator

VALID_CONFIG = {
    CONF_SENSOR_INDOOR_TEMP: "sensor.indoor",
    CONF_WEATHER_ENTITY: "weather.home",
    CONF_C_THERMAL: 10000.0,
    CONF_UA: 500.0,
    CONF_K_SOLAR: 50.0,
    CONF_Q_INT: 500.0,
    CONF_H_FACTOR: 1000.0,
}

@pytest.fixture
def coordinator(hass):
    """Create a coordinator instance."""
    options = {
        CONF_C_THERMAL: 10000.0,
        CONF_UA: 500.0,
        CONF_K_SOLAR: 50.0,
        CONF_Q_INT: 500.0,
        CONF_H_FACTOR: 1000.0,
        CONF_HVAC_MODE: "heat",
        CONF_CENTER_PREFERENCE: 1.0,
        CONF_AVOID_DEFROST: True,
    }
    entry = MockConfigEntry(domain=DOMAIN, data=VALID_CONFIG, options=options)
    entry.add_to_hass(hass)
    
    with patch("custom_components.housetemp.coordinator.HeatPump") as mock_hp:
        # Mock the instance returned by the class constructor
        instance = MagicMock()
        # Return proper numpy arrays matching input shape
        instance.get_cop.side_effect = lambda x: np.full(len(x), 3.0)
        instance.get_cooling_cop.side_effect = lambda x: np.full(len(x), 3.0)
        instance.get_max_capacity.side_effect = lambda x: np.full(len(x), 10000.0)
        instance.min_output_btu_hr = 3000
        instance.max_cool_btu_hr = 54000
        instance.plf_low_load = 1.4
        instance.plf_slope = 0.4
        instance.plf_min = 0.5
        instance.idle_power_kw = 0.25
        instance.blower_active_kw = 0.9
        instance.defrost_risk_zone = None
        mock_hp.return_value = instance
        coord = HouseTempCoordinator(hass, entry)
        # Manually ensure heat_pump is set if setup fails silently or we need to force it
        coord.heat_pump = instance
        return coord

@pytest.fixture
def mock_data(coordinator):
    """Mock the data preparation helper."""
    measurements = MagicMock()
    measurements.timestamps = [datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc) + timedelta(minutes=i*15) for i in range(4)]
    measurements.setpoint = np.array([70.0, 70.0, 70.0, 70.0])
    measurements.target_temp = np.array([70.0, 70.0, 70.0, 70.0])
    measurements.t_in = np.array([68.0, 68.0, 68.0, 68.0])
    measurements.t_out = np.array([50.0, 50.0, 50.0, 50.0])
    measurements.solar_kw = np.array([0.0, 0.0, 0.0, 0.0])
    measurements.hvac_state = np.array([0, 0, 0, 0])
    # Add numerical dt_hours for run_model validation
    measurements.dt_hours = np.array([0.25, 0.25, 0.25, 0.25])
    
    params = [10000.0, 750.0, 3000.0, 2000.0, 5000.0, 1.0]
    start_time = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)
    
    return measurements, params, start_time

@pytest.mark.asyncio
async def test_auto_update_does_not_optimize(hass, coordinator, mock_data):
    """Test that atomic periodic update does NOT run optimization."""
    ms, params, start_time = mock_data
    
    with patch.object(coordinator, "_prepare_simulation_inputs", return_value=(ms, params, start_time)), \
         patch("custom_components.housetemp.coordinator.run_model_continuous") as mock_run, \
         patch("custom_components.housetemp.coordinator.estimate_consumption") as mock_estimate, \
         patch("custom_components.housetemp.coordinator.optimize_hvac_schedule") as mock_opt:
        
        mock_run.return_value = ([0.0]*4, [0.0]*4, [0.0]*4)
        mock_estimate.return_value = {'total_kwh': 0.0}
        
        await coordinator._async_update_data()
        
        # Should NOT run optimization
        assert not mock_opt.called
        # Should run simulation
        assert mock_run.called

@pytest.mark.asyncio
async def test_manual_trigger_optimizes(hass, coordinator, mock_data):
    """Test that manual trigger runs optimization and updates cache."""
    ms, params, start_time = mock_data
    
    # Fake optimized result (different from schedule)
    optimized_setpoints = np.array([72.0, 72.0, 71.0, 71.0])
    
    # We must patch run_model because async_trigger_optimization calls it now,
    # and if we mocked _prepare_simulation_inputs but not run_model, it might fail logic
    # or return invalid outputs.
    # The previous failure was due to run_model executing logic on MagicMock.
    with patch.object(coordinator, "_prepare_simulation_inputs", return_value=(ms, params, start_time)), \
         patch("custom_components.housetemp.coordinator.optimize_hvac_schedule", return_value=(optimized_setpoints, {"success": True})) as mock_opt, \
         patch.object(coordinator, "async_request_refresh") as mock_refresh, \
         patch.object(coordinator, "async_set_updated_data") as mock_set_data, \
         patch("custom_components.housetemp.coordinator.estimate_consumption", return_value={'total_kwh': 0.0}), \
         patch("custom_components.housetemp.coordinator.run_model_continuous", return_value=([68.0]*4, [0.0]*4, [0.0]*4)) as mock_run_model:
        
        await coordinator.async_trigger_optimization()
        
        assert mock_opt.called
        # Check cache update
        # timestamp for first point is mock_data[0].timestamps[0]
        ts0_int = int(ms.timestamps[0].timestamp())
        assert coordinator.optimized_setpoints_map[ts0_int] == 72.0
        
        # Should NOT trigger refresh (we do direct update now)
        assert not mock_refresh.called
        
        # Should trigger direct data update
        assert mock_set_data.called
        
        # Should run simulation for response
        assert mock_run_model.called

@pytest.mark.asyncio
async def test_cache_application_in_update(hass, coordinator, mock_data):
    """Test that cached optimized setpoints are applied during update."""
    ms, params, start_time = mock_data
    
    # Pre-populate cache
    # Initialize last_config_id to prevent clearing
    coordinator._last_config_id = "static_id"
    
    ts0 = ms.timestamps[0]
    ts0_int = int(ts0.timestamp())
    coordinator.optimized_setpoints_map[ts0_int] = 75.0 # Cached value
    
    # Mock run_model to capture what was passed
    # Ensure current time is BEFORE the cache timestamps so they aren't expired
    fake_now = ms.timestamps[0] - timedelta(hours=1)
    
    # Track what setpoint was used (measurements.setpoint is modified in-place)
    captured_setpoints = None
    def capture_run(*args, **kwargs):
        nonlocal captured_setpoints
        # Capture setpoint_list from kwargs (new API) or from measurements
        captured_setpoints = kwargs.get('setpoint_list', list(ms.setpoint))
        return ([0.0]*4, [0.0]*4, [0.0]*4)
    
    with patch("homeassistant.util.dt.now", return_value=fake_now), \
         patch.object(coordinator, "_prepare_simulation_inputs", return_value=(ms, params, start_time)), \
         patch.object(coordinator, "_get_config_id", return_value="static_id"), \
         patch.object(coordinator, "_expire_cache"), \
         patch("custom_components.housetemp.coordinator.estimate_consumption", return_value={'total_kwh': 0.0}), \
         patch("custom_components.housetemp.coordinator.run_model_continuous", side_effect=capture_run) as mock_run:
    
        result = await coordinator._async_update_data()
        
        # Verify measurements.setpoint was modified to include cached value
        # The coordinator applies cache to ms.setpoint before calling run_model
        assert ms.setpoint[0] == 75.0  # From cache
        assert ms.setpoint[1] == 70.0  # From schedule (fallback)
        
        # Verify result contains optimized setpoint array
        assert "optimized_setpoint" in result
        assert result["optimized_setpoint"][0] == 75.0
        assert result["optimized_setpoint"][1] is None
@pytest.mark.asyncio
async def test_gap_neutrality_in_simulation(hass, coordinator, mock_data):
    """Test that gaps in both cache and schedule are forced to HVAC off and indoor temp."""
    ms, params, start_time = mock_data
    
    # Copy arrays to avoid mutating shared fixture state across tests
    ms.target_temp = ms.target_temp.copy()
    ms.hvac_state = ms.hvac_state.copy()
    ms.setpoint = ms.setpoint.copy()
    
    # Introduce a gap in the schedule (NaN) at index 2
    ms.target_temp[2] = np.nan
    ms.hvac_state[2] = 1 # Would be heating if it weren't for the gap
    
    # Cache is empty for index 2
    coordinator.optimized_setpoints_map = {}
    
    fake_now = ms.timestamps[0] - timedelta(hours=1)
    fake_temps = np.array([68.0, 68.0, 68.0, 68.0], dtype=float)
    fake_hvac_out = np.zeros_like(fake_temps)
    
    with patch("custom_components.housetemp.coordinator.dt_util.now", return_value=fake_now), \
         patch.object(coordinator, "_prepare_simulation_inputs", return_value=(ms, params, start_time)), \
         patch("custom_components.housetemp.coordinator.estimate_consumption", return_value={'total_kwh': 0.0}), \
         patch("custom_components.housetemp.coordinator.run_model_continuous", return_value=(fake_temps, fake_hvac_out, fake_hvac_out)) as mock_run:
         
        await coordinator._async_update_data()
        
        # Check measurements.setpoint and measurements.hvac_state directly
        # The coordinator modifies these in-place before calling run_model
        
        # Index 2 had a gap in both cache and schedule.
        # 1. Should be clamped to current indoor temp (ms.t_in[0] == 68.0)
        assert float(ms.setpoint[2]) == 68.0
        # 2. HVAC state should be forced to 0 for this index, even though original schedule had 1
        assert int(ms.hvac_state[2]) == 0
        
        # Index 0 was not a gap (target_temp exists).
        # It should preserve its original hvac_state and use target_temp for setpoint.
        assert float(ms.setpoint[0]) == 70.0

@pytest.mark.asyncio
async def test_optimization_includes_energy_steps(hass, coordinator, mock_data):
    """Test that optimization result includes per-step energy for hourly aggregation.
    
    This prevents regressions where energy_per_hour disappears from sensor attributes.
    """
    ms, params, start_time = mock_data
    
    optimized_setpoints = np.array([72.0, 72.0, 71.0, 71.0])
    
    # Mock energy calculation to return kwh_steps
    mock_energy_result = {
        'kwh': 1.5,
        'kwh_steps': np.array([0.4, 0.4, 0.35, 0.35]),  # Per-step energy
        'load_ratios': np.array([0.5, 0.5, 0.4, 0.4])
    }
    
    with patch.object(coordinator, "_prepare_simulation_inputs", return_value=(ms, params, start_time)), \
         patch("custom_components.housetemp.coordinator.optimize_hvac_schedule", return_value=(optimized_setpoints, {"success": True})), \
         patch("custom_components.housetemp.coordinator.estimate_consumption", return_value={'total_kwh': 2.0}), \
         patch("custom_components.housetemp.coordinator.run_model_continuous", return_value=([68.0]*4, [1000.0]*4, [1000.0]*4)), \
         patch("custom_components.housetemp.coordinator.run_model_discrete", return_value=([68.0]*4, [1000.0]*4, [1000.0]*4, [1]*4, {})), \
         patch("custom_components.housetemp.coordinator.calculate_energy_vectorized", return_value=mock_energy_result), \
         patch.object(coordinator, "async_set_updated_data") as mock_set_data:
        
        await coordinator.async_trigger_optimization()
        
        # Verify async_set_updated_data was called with data containing energy_kwh_steps
        assert mock_set_data.called
        call_args = mock_set_data.call_args[0][0]  # First positional arg
        
        assert "energy_kwh_steps" in call_args, "energy_kwh_steps missing from coordinator data after optimization"
        assert call_args["energy_kwh_steps"] is not None, "energy_kwh_steps should not be None"
        # Verify it has the correct length
        assert len(call_args["energy_kwh_steps"]) == len(ms.timestamps), "energy_kwh_steps length mismatch"
