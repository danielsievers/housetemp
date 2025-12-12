
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
    """Test update with unavailable indoor sensor."""
    hass.states.async_set("sensor.indoor", "unavailable")
    with pytest.raises(UpdateFailed, match="Indoor sensor sensor.indoor unavailable"):
        await coordinator._async_update_data()

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
    hass.states.async_set("sensor.indoor", "70.0")
    hass.states.async_set("weather.home", "unknown") # Invalid float
    
    with pytest.raises(UpdateFailed, match="No forecast data available"):
        await coordinator._async_update_data()

@pytest.mark.asyncio
async def test_update_schedule_invalid_json(hass, coordinator):
    """Test update with invalid schedule JSON in options."""
    # Mock input handler to return valid simulation data so we reach schedule processing
    from datetime import datetime, timezone
    
    start = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)
    mock_sim_data = (
        [start], # timestamps
        np.array([50.0]), # t_out
        np.array([0.0]),  # solar
        np.array([1.0])   # dt
    )
    
    # We must patch the input_handler on the coordinator instance
    coordinator.input_handler = MagicMock()
    coordinator.input_handler.prepare_simulation_data = MagicMock(return_value=mock_sim_data)
    
    # Set invalid schedule in options
    hass.config_entries.async_update_entry(
        coordinator.config_entry,
        options={CONF_SCHEDULE_CONFIG: "{invalid"}
    )
    
    # Should raise UpdateFailed because ValueError from json.loads is caught in _async_update_data
    with pytest.raises(UpdateFailed, match="Error preparing simulation inputs"):
        await coordinator._async_update_data()
