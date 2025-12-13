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
    measurements.t_out = np.array([50.0, 50.0, 50.0, 50.0])
    measurements.solar_kw = np.array([0.0, 0.0, 0.0, 0.0])
    measurements.hvac_state = np.array([0, 0, 0, 0])
    # Add numerical dt_hours for run_model validation
    measurements.dt_hours = np.array([0.25, 0.25, 0.25, 0.25])
    
    params = [10000.0, 750.0, 3000.0, 2000.0, 5000.0]
    start_time = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)
    
    return measurements, params, start_time

@pytest.mark.asyncio
async def test_auto_update_does_not_optimize(hass, coordinator, mock_data):
    """Test that atomic periodic update does NOT run optimization."""
    ms, params, start_time = mock_data
    
    with patch.object(coordinator, "_prepare_simulation_inputs", return_value=(ms, params, start_time)), \
         patch("custom_components.housetemp.coordinator.run_model") as mock_run, \
         patch("custom_components.housetemp.coordinator.optimize_hvac_schedule") as mock_opt:
        
        mock_run.return_value = ([], 0.0, [])
        
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
         patch("custom_components.housetemp.coordinator.optimize_hvac_schedule", return_value=optimized_setpoints) as mock_opt, \
         patch.object(coordinator, "async_request_refresh") as mock_refresh, \
         patch.object(coordinator, "async_set_updated_data") as mock_set_data, \
         patch("custom_components.housetemp.coordinator.run_model", return_value=([68.0]*4, 0.0, [])) as mock_run_model:
        
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
    ts0 = ms.timestamps[0]
    ts0_int = int(ts0.timestamp())
    coordinator.optimized_setpoints_map[ts0_int] = 75.0 # Cached value
    
    # Mock run_model to capture what was passed
    # Ensure current time is BEFORE the cache timestamps so they aren't expired
    fake_now = ms.timestamps[0] - timedelta(hours=1)
    
    with patch("homeassistant.util.dt.now", return_value=fake_now), \
         patch.object(coordinator, "_prepare_simulation_inputs", return_value=(ms, params, start_time)), \
         patch("custom_components.housetemp.coordinator.run_model", return_value=([], 0.0, [])) as mock_run:
    
        result = await coordinator._async_update_data()
        
        # Verify measurements passed to run_model has the cached setpoint
        args, _ = mock_run.call_args
        measurements_arg = args[1]
        assert measurements_arg.setpoint[0] == 75.0 # From cache
        assert measurements_arg.setpoint[1] == 70.0 # From schedule (fallback)
        
        # Verify result contains optimized setpoint array
        assert "optimized_setpoint" in result
        assert result["optimized_setpoint"][0] == 75.0
        assert result["optimized_setpoint"][1] is None
