
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
    CONF_HEAT_PUMP_CONFIG: "{}",
    CONF_SCHEDULE_CONFIG: "[]",
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
        # Ensure heat_pump is set (it should be if mock works)
        # But setup catches exception. If Mock is successful, it sets self.heat_pump = mock_hp()
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
    c.config_entry.data = {**VALID_CONFIG, CONF_HEAT_PUMP_CONFIG: "{invalid"}
    
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
    
    with patch("custom_components.housetemp.coordinator.run_model") as mock_run:
        mock_run.return_value = (np.array([70.0, 70.0]), 0.0, np.array([0, 0]))
        res = await coordinator._async_update_data()
        assert res["outdoor"][0] == 70.0 # Fallback to passed current_temp (indoor) as per code logic

@pytest.mark.asyncio
async def test_update_weather_invalid_current(hass, coordinator):
    """Test update with invalid current outdoor temp."""
    hass.states.async_set("sensor.indoor", "70.0")
    hass.states.async_set("weather.home", "unknown") # Invalid float
    
    with patch("custom_components.housetemp.coordinator.run_model") as mock_run:
        mock_run.return_value = (np.array([70.0, 70.0]), 0.0, np.array([0, 0]))
        res = await coordinator._async_update_data()
        # Should fallback to default 50.0 if interpolation also fails/empty
        assert len(res["outdoor"]) > 0

@pytest.mark.asyncio
async def test_interpolation_malformed_items(hass, coordinator):
    """Test weather interpolation with malformed items."""
    forecast = [
        {"datetime": "invalid_date", "temperature": 60},
        {"datetime": None, "temperature": 60},
        {"no_date": "value"},
    ]
    timestamps = [datetime.now()]
    res = coordinator._get_interpolated_weather(timestamps, forecast)
    assert len(res) == 1
    assert res[0] == 50.0 # Default fallback

@pytest.mark.asyncio
async def test_solar_interpolation_malformed(hass, coordinator):
    """Test solar interpolation with malformed items."""
    forecast = [
        {"period_end": "invalid", "pv_estimate": 10},
    ]
    timestamps = [datetime.now()]
    res = coordinator._get_interpolated_solar(timestamps, forecast)
    assert len(res) == 1
    assert res[0] == 0.0

@pytest.mark.asyncio
async def test_process_schedule_invalid_json(hass, coordinator):
    """Test schedule processing with invalid JSON."""
    timestamps = [datetime.now()]
    hvac, setpoints = coordinator._process_schedule(timestamps, "{invalid")
    assert hvac[0] == 0
    assert setpoints[0] == 70.0

@pytest.mark.asyncio
async def test_process_schedule_modes(hass, coordinator):
    """Test schedule processing for cool mode and time logic."""
    schedule = json.dumps([
        {"time": "00:00", "mode": "heat", "setpoint": 68},
        {"time": "12:00", "mode": "cool", "setpoint": 75},
    ])
    
    # Create timestamps for 10am and 2pm
    t1 = datetime(2023, 1, 1, 10, 0, 0)
    t2 = datetime(2023, 1, 1, 14, 0, 0)
    timestamps = [t1, t2]
    
    hvac, setpoints = coordinator._process_schedule(timestamps, schedule)
    
    # 10am -> matches 00:00 -> heat (1)
    assert hvac[0] == 1 
    assert setpoints[0] == 68.0
    
    # 2pm -> matches 12:00 -> cool (-1)
    assert hvac[1] == -1
    assert setpoints[1] == 75.0

@pytest.mark.asyncio
async def test_interpolation_exception_trigger(hass, coordinator):
    """Test interpolation exception handling (not just None return)."""
    # Trigger exception by passing type that causes error
    forecast = [{"datetime": object(), "temperature": 10}] # object() as datetime string might raise TypeError in parse_datetime or get
    # Note: parse_datetime(None) returns None
    timestamps = [datetime.now()]
    res = coordinator._get_interpolated_weather(timestamps, forecast)
    assert len(res) == 1

@pytest.mark.asyncio
async def test_solar_interpolation_exception_trigger(hass, coordinator):
    """Test solar interpolation exception handling."""
    forecast = [{"datetime": "2023-01-01", "value": "invalid_float"}]
    timestamps = [datetime.now()]
    res = coordinator._get_interpolated_solar(timestamps, forecast)
    assert len(res) == 1
    assert res[0] == 0.0

@pytest.mark.asyncio
async def test_solar_interpolation_valid(hass, coordinator):
    """Test solar interpolation with valid data."""
    # Use generic format
    dt_str = datetime.now().isoformat()
    forecast = [{"datetime": dt_str, "value": 5.0}]
    timestamps = [datetime.now()]
    res = coordinator._get_interpolated_solar(timestamps, forecast)
    assert len(res) == 1
    assert res[0] == 5.0
