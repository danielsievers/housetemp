"""Test the sensor."""
from unittest.mock import patch, MagicMock
import pytest
from datetime import datetime, timedelta
import numpy as np

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.housetemp.const import (
    DOMAIN,
    CONF_SENSOR_INDOOR_TEMP,
    CONF_WEATHER_ENTITY,
    CONF_SOLAR_ENTITY,
    CONF_C_THERMAL,
    CONF_UA,
    CONF_K_SOLAR,
    CONF_Q_INT,
    CONF_H_FACTOR,
    CONF_HEAT_PUMP_CONFIG,
    CONF_SCHEDULE_CONFIG,
    CONF_FORECAST_DURATION,
    CONF_UPDATE_INTERVAL,
)

@pytest.mark.asyncio
async def test_sensor_setup_and_state(hass: HomeAssistant):
    """Test sensor setup and state update."""
    # Set to Imperial
    from homeassistant.util.unit_system import US_CUSTOMARY_SYSTEM
    hass.config.units = US_CUSTOMARY_SYSTEM

    # 1. Setup Config Entry
    config_data = {
        CONF_SENSOR_INDOOR_TEMP: "sensor.indoor",
        CONF_WEATHER_ENTITY: "weather.home",
        CONF_SOLAR_ENTITY: "sensor.solar",
        CONF_C_THERMAL: 10000.0,
        CONF_UA: 500.0,
        CONF_K_SOLAR: 50.0,
        CONF_Q_INT: 500.0,
        CONF_H_FACTOR: 1000.0,
        CONF_HEAT_PUMP_CONFIG: "{}",
        CONF_SCHEDULE_CONFIG: '[{"time": "00:00", "mode": "heat", "setpoint": 70}]',
        CONF_FORECAST_DURATION: 8,
        CONF_UPDATE_INTERVAL: 15,
    }
    
    entry = MockConfigEntry(domain=DOMAIN, data=config_data)
    entry.add_to_hass(hass)

    # 2. Mock Dependencies
    # Mock States
    hass.states.async_set("sensor.indoor", "68.0")
    hass.states.async_set("weather.home", "50.0", {"forecast": [
        {"datetime": (datetime.now() + timedelta(hours=1)).isoformat(), "temperature": 55.0}
    ]})
    hass.states.async_set("sensor.solar", "0.0", {"forecast": []})

    # Mock run_model to return dummy data
    # We mock the import in coordinator.py
    with patch("custom_components.housetemp.coordinator.run_model") as mock_run_model, \
         patch("custom_components.housetemp.coordinator.HeatPump") as mock_hp_cls:
        
        # Setup Mock Return
        # run_model returns (sim_temps, rmse)
        # sim_temps should be array of length steps
        steps = 8 * 2 # 8 hours * 2 per hour
        mock_run_model.return_value = (np.full(steps, 72.5), 0.0, np.zeros(steps))
        
        # 3. Setup Integration
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        # 4. Verify Sensor State
        state = hass.states.get("sensor.indoor_temperature_forecast")
        assert state is not None
        assert state.state == "72.5"
        
        # Verify Attributes
        attrs = state.attributes
        assert "forecast" in attrs
        # Now resampled to 15-min intervals: 8 hours * 4 per hour = 32 points
        assert len(attrs["forecast"]) == 32
        assert attrs["forecast"][0]["temperature"] == 72.5
        assert attrs["forecast"][0]["target_temp"] == 70.0  # Renamed from setpoint
        assert "forecast_points" in attrs  # Original resolution count
        pass # mock return is repeated or real model runs with constant inputs
        
    # 5. Verify Unload
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()

@pytest.mark.asyncio
async def test_sensor_initially_unavailable(hass: HomeAssistant):
    """Test sensor state when coordinator has no data."""
    # Setup Config Entry
    config_data = {
        CONF_SENSOR_INDOOR_TEMP: "sensor.indoor",
        CONF_WEATHER_ENTITY: "weather.home",
        CONF_SOLAR_ENTITY: "sensor.solar",
        CONF_C_THERMAL: 1000.0,
        CONF_UA: 100.0,
        CONF_K_SOLAR: 10.0,
        CONF_Q_INT: 100.0,
        CONF_H_FACTOR: 100.0,
        CONF_HEAT_PUMP_CONFIG: "{}",
        CONF_SCHEDULE_CONFIG: "[]",
    }
    
    entry = MockConfigEntry(domain=DOMAIN, data=config_data)
    entry.add_to_hass(hass)
    
    # Mock data to return None initially or empty
    with patch("custom_components.housetemp.coordinator.HouseTempCoordinator._async_update_data", side_effect=lambda: None):
        # We need update to succeed but data to be empty? 
        # Easier: Setup with successful init, but then clear data?
        pass

    # Actually, easiest way to test "no data" path in sensor is to catch it before first update?
    # Or just mock coordinator data directly
    
    # Let's use a simpler approach: Instantiate sensor manually or mock coordinator
    
    from custom_components.housetemp.sensor import HouseTempPredictionSensor
    
    mock_coord = MagicMock()
    mock_coord.data = None
    
    sensor = HouseTempPredictionSensor(mock_coord, entry)
    assert sensor.native_value is None
    assert sensor.extra_state_attributes == {}

@pytest.mark.asyncio
async def test_sensor_empty_prediction(hass):
    """Test sensor with data but empty prediction array."""
    config_data = {
        CONF_SENSOR_INDOOR_TEMP: "sensor.indoor",
        CONF_C_THERMAL: 1000.0,
        # Minimal required
    }
    
    entry = MockConfigEntry(domain=DOMAIN, data=config_data)
    entry.add_to_hass(hass)
    
    from custom_components.housetemp.sensor import HouseTempPredictionSensor
    
    mock_coord = MagicMock()
    mock_coord.data = {
        "predicted_temp": [], # Empty
        "timestamps": []
    }
    
    sensor = HouseTempPredictionSensor(mock_coord, entry)
    assert sensor.native_value is None

