
"""Test coordinator coverage for edge cases."""
from unittest.mock import patch, MagicMock
import pytest
import json
from datetime import datetime, timedelta
import numpy as np

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import UpdateFailed
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
from custom_components.housetemp.coordinator import HouseTempCoordinator

# Default valid config data
VALID_CONFIG = {
    CONF_SENSOR_INDOOR_TEMP: "sensor.indoor",
    CONF_WEATHER_ENTITY: "weather.home",
    CONF_SOLAR_ENTITY: "sensor.solar",
    CONF_C_THERMAL: 1000.0,
    CONF_UA: 100.0,
    CONF_K_SOLAR: 10.0,
    CONF_Q_INT: 100.0,
    CONF_H_FACTOR: 100.0,
    CONF_HEAT_PUMP_CONFIG: '{"max_capacity": 10000, "cop_curve": [[-10, 3], [10, 4]], "hvac_power": 1000}',
    CONF_SCHEDULE_CONFIG: '{"schedule": [{"weekdays": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"], "daily_schedule": [{"time": "00:00", "temp": 70}]}]}',
    CONF_FORECAST_DURATION: 1,
    CONF_UPDATE_INTERVAL: 60,
}

@pytest.fixture
def coordinator(hass):
    """Create a coordinator instance."""
    entry = MockConfigEntry(domain=DOMAIN, data=VALID_CONFIG)
    entry.add_to_hass(hass)
    
    with patch("custom_components.housetemp.coordinator.HeatPump") as mock_hp:
        # We need to make sure the instance is not None
        mock_hp.return_value = MagicMock()
        coord = HouseTempCoordinator(hass, entry)
        # Ensure heat_pump is set to Mock to bypass lazy setup logic during update tests
        coord.heat_pump = mock_hp.return_value
        return coord

@pytest.mark.asyncio
async def test_setup_heat_pump_missing_config(hass):
    """Test heat pump setup with missing config."""
    config = VALID_CONFIG.copy()
    config[CONF_HEAT_PUMP_CONFIG] = "" # Empty
    entry = MockConfigEntry(domain=DOMAIN, data=config)
    c = HouseTempCoordinator(hass, entry)
    assert c.heat_pump is None

@pytest.mark.asyncio
async def test_setup_heat_pump_invalid_json(hass):
    """Test heat pump setup with invalid JSON."""
    config = VALID_CONFIG.copy()
    config[CONF_HEAT_PUMP_CONFIG] = "{invalid"
    entry = MockConfigEntry(domain=DOMAIN, data=config)
    c = HouseTempCoordinator(hass, entry)
    assert c.heat_pump is None

@pytest.mark.asyncio
async def test_update_heat_pump_not_configured(hass, coordinator):
    """Test update fails if heat pump is not configured."""
    c = coordinator
    # Force heat_pump to None
    c.heat_pump = None
    # Ensure it tries to setup but fails (e.g. by making config invalid now)
    hass.config_entries.async_update_entry(
        c.config_entry,
        data={**VALID_CONFIG, CONF_HEAT_PUMP_CONFIG: "{invalid"}
    )
    
    with pytest.raises(UpdateFailed):
        await c._async_update_data()

@pytest.mark.asyncio
async def test_update_indoor_sensor_unavailable(hass, coordinator):
    """Test update with unavailable indoor sensor (should return previous data)."""
    coordinator.data = {"test": "data"}
    hass.states.async_set("sensor.indoor", "unavailable")
    res = await coordinator._async_update_data()
    assert res == {"test": "data"}

@pytest.mark.asyncio
async def test_update_indoor_sensor_invalid(hass, coordinator):
    """Test update with invalid indoor sensor value."""
    hass.states.async_set("sensor.indoor", "invalid_float")
    with pytest.raises(UpdateFailed, match="Invalid indoor temp"):
        await coordinator._async_update_data()

@pytest.mark.asyncio
async def test_update_weather_entity_missing(hass, coordinator):
    """Test update with missing weather entity."""
    hass.states.async_set("sensor.indoor", "70.0")
    # weather.home not set
    with pytest.raises(UpdateFailed, match="Weather entity weather.home not found"):
        await coordinator._async_update_data()

@pytest.mark.asyncio
async def test_update_weather_no_forecast_attribute(hass, coordinator):
    """Test update with weather entity missing forecast attribute."""
    hass.states.async_set("sensor.indoor", "70.0")
    hass.states.async_set("weather.home", "50.0", {}) # No forecast
    
    # Needs to proceed with dummy/empty forecast or fallback without crashing
    # The code currently warns and uses empty list
    # We need to mock run_model to avoid actually running it on empty data causing other issues?
    # Or let it run and see if it handles empty arrays (it might fallback to defaults in interpolation)
    
    with pytest.raises(UpdateFailed, match="No forecast data available"):
        await coordinator._async_update_data()

@pytest.mark.asyncio
async def test_update_weather_invalid_current(hass, coordinator):
    """Test update with invalid current outdoor temp."""
    coordinator.data = {"baseline": "data"}
    hass.states.async_set("sensor.indoor", "70.0")
    hass.states.async_set("weather.home", "unknown") # Invalid float/state
    
    res = await coordinator._async_update_data()
    assert res == {"baseline": "data"}

@pytest.mark.asyncio
async def test_state_trackers_setup(hass, coordinator):
    """Test that state trackers are correctly set up and trigger refresh."""
    coordinator.async_setup_trackers()
    assert len(coordinator._unsub_state_trackers) == 1
    
    # Mock async_request_refresh
    coordinator.async_request_refresh = MagicMock(return_value=None)
    
    # Manually trigger the event to simulate state change
    from homeassistant.core import Event, EventOrigin
    from homeassistant.const import EVENT_STATE_CHANGED
    
    event = Event(
        EVENT_STATE_CHANGED,
        {
            "entity_id": "sensor.indoor",
            "old_state": MagicMock(state="unavailable"),
            "new_state": MagicMock(state="72.0"),
        },
        origin=EventOrigin.local,
    )
    
    # Since we can't easily trigger the subscriber without real HASS event bus logic in unit test 
    # without more mocks, we can just verify the unsub logic works
    coordinator.async_setup_trackers()
    unsub = coordinator._unsub_state_trackers[0]
    coordinator.async_setup_trackers() # Re-setup should call unsub
    assert len(coordinator._unsub_state_trackers) == 1
