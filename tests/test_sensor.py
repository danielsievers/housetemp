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
    # 1. Setup Config Entry
    # Fixed parameters in data
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
    }
    # Adjustable parameters in options
    config_options = {
        CONF_SCHEDULE_CONFIG: '{"schedule": [{"weekdays": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"], "daily_schedule": [{"time": "00:00", "temp": 70}]}]}',
        CONF_FORECAST_DURATION: 8,
        CONF_UPDATE_INTERVAL: 15,
    }
    
    entry = MockConfigEntry(domain=DOMAIN, data=config_data, options=config_options)
    entry.add_to_hass(hass)

    # 2. Mock Dependencies
    # Mock States
    from homeassistant.util import dt as dt_util
    now = dt_util.now()
    hass.states.async_set("sensor.indoor", "68.0")
    hass.states.async_set("weather.home", "50.0", {"forecast": [
        {"datetime": (now - timedelta(hours=1)).isoformat(), "temperature": 55.0} # Start 1 hour ago to be safe
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
        if state is None:
             # print("Available entities:", [s.entity_id for s in hass.states.async_all()])
             pass
        assert state is not None
        # Now returns current target temp (70.0 from schedule), not predicted (72.5)
        assert state.state == "70.0"
        
        # Verify Attributes
        attrs = state.attributes
        assert "forecast" in attrs
        # Now resampled to 15-min intervals: 8 hours * 4 per hour = 32 points
        assert len(attrs["forecast"]) == 32
        assert attrs["forecast"][0]["temperature"] == 72.5
        assert attrs["forecast"][0]["target_temp"] == 70.0  # From schedule
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
        CONF_SCHEDULE_CONFIG: '{"schedule": []}',
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


@pytest.mark.asyncio
async def test_sensor_with_optimized_setpoint(hass):
    """Test sensor with optimized_setpoint (numpy array)."""
    from custom_components.housetemp.sensor import HouseTempPredictionSensor
    from datetime import datetime, timedelta, timezone
    
    config_data = {
        CONF_SENSOR_INDOOR_TEMP: "sensor.indoor",
        CONF_C_THERMAL: 1000.0,
    }
    
    entry = MockConfigEntry(domain=DOMAIN, data=config_data)
    entry.add_to_hass(hass)
    
    # Create mock data with numpy arrays (simulating real coordinator data)
    now = datetime.now(timezone.utc)
    timestamps = [now + timedelta(minutes=i*5) for i in range(24)]  # 2 hours of 5-min data
    
    mock_coord = MagicMock()
    mock_coord.data = {
        "timestamps": timestamps,
        "predicted_temp": np.array([70.0] * 24),
        "setpoint": np.array([68.0] * 24),
        "optimized_setpoint": np.array([69.0] * 24),  # Optimization result
    }
    
    sensor = HouseTempPredictionSensor(mock_coord, entry)
    attrs = sensor.extra_state_attributes
    
    assert "forecast" in attrs
    # Should have resampled to 15-min intervals
    assert len(attrs["forecast"]) > 0
    # Each item should have ideal_setpoint
    assert "ideal_setpoint" in attrs["forecast"][0]
    assert attrs["forecast"][0]["ideal_setpoint"] == 69.0


@pytest.mark.asyncio
async def test_sensor_without_optimized_setpoint(hass):
    """Test sensor without optimized_setpoint (no optimization ran)."""
    from custom_components.housetemp.sensor import HouseTempPredictionSensor
    from datetime import datetime, timedelta, timezone
    
    config_data = {
        CONF_SENSOR_INDOOR_TEMP: "sensor.indoor",
        CONF_C_THERMAL: 1000.0,
    }
    
    entry = MockConfigEntry(domain=DOMAIN, data=config_data)
    entry.add_to_hass(hass)
    
    now = datetime.now(timezone.utc)
    timestamps = [now + timedelta(minutes=i*5) for i in range(24)]
    
    mock_coord = MagicMock()
    mock_coord.data = {
        "timestamps": timestamps,
        "predicted_temp": np.array([70.0] * 24),
        "setpoint": np.array([68.0] * 24),
        # No optimized_setpoint - optimization didn't run
    }
    
    sensor = HouseTempPredictionSensor(mock_coord, entry)
    attrs = sensor.extra_state_attributes
    
    assert "forecast" in attrs
    assert len(attrs["forecast"]) > 0
    # Should NOT have ideal_setpoint when no optimization
    assert "ideal_setpoint" not in attrs["forecast"][0]
    # But should have setpoint (renamed to target_temp)
    assert attrs["forecast"][0]["target_temp"] == 68.0

@pytest.mark.asyncio
async def test_sensor_preference_optimized_over_schedule(hass: HomeAssistant):
    """Test that sensor prefers optimized_setpoint over schedule setpoint."""
    from custom_components.housetemp.sensor import HouseTempPredictionSensor
    from datetime import datetime, timedelta, timezone
    
    config_data = {
        CONF_SENSOR_INDOOR_TEMP: "sensor.indoor",
        CONF_C_THERMAL: 1000.0,
    }
    
    entry = MockConfigEntry(domain=DOMAIN, data=config_data)
    entry.add_to_hass(hass)
    
    now = datetime.now(timezone.utc)
    timestamps = [now + timedelta(minutes=i*5) for i in range(24)]
    
    mock_coord = MagicMock()
    mock_coord.data = {
        "timestamps": timestamps,
        "predicted_temp": np.array([72.5] * 24),
        "setpoint": np.array([65.0] * 24),            # Schedule says 65
        "optimized_setpoint": np.array([68.0] * 24),  # Optimization says 68
    }
    
    sensor = HouseTempPredictionSensor(mock_coord, entry)
    
    # helper to check state
    val = sensor.native_value
    assert val == 68.0  # Should pick optimized (68) over schedule (65)

@pytest.mark.asyncio
async def test_sensor_away_expires_in_forecast(hass: HomeAssistant):
    """Test that ideal_setpoint stops using away temp after expiration."""
    from custom_components.housetemp.sensor import HouseTempPredictionSensor
    from datetime import datetime, timedelta, timezone
    from homeassistant.util import dt as dt_util
    
    config_data = {
        CONF_SENSOR_INDOOR_TEMP: "sensor.indoor",
        CONF_C_THERMAL: 1000.0,
    }
    
    entry = MockConfigEntry(domain=DOMAIN, data=config_data)
    entry.add_to_hass(hass)
    
    now = datetime.now(timezone.utc)
    # 4 hours of data
    timestamps = [now + timedelta(minutes=i*15) for i in range(16)] 
    
    # Away ends after 1 hour (4 points)
    away_end = now + timedelta(hours=1)
    
    mock_coord = MagicMock()
    mock_coord.hass = hass
    
    # Simulate data WITHOUT optimized setpoints (forcing fallback logic)
    mock_coord.data = {
        "timestamps": timestamps,
        "predicted_temp": np.array([70.0] * 16),
        "setpoint": np.array([68.0] * 16),
        "optimized_setpoint": [], # Empty/Missing optimization
        "away_info": {
            "active": True,
            "temp": 55.0,
            "end": away_end.isoformat()
        }
    }
    
    sensor = HouseTempPredictionSensor(mock_coord, entry)
    attrs = sensor.extra_state_attributes
    forecast = attrs["forecast"]
    
    # Point 0 (Now): Should be Away Temp (55)
    # Note: Sensor realigns to 15min grid, so we just check values, not exact timestamp strings unless calculated
    assert forecast[0]["ideal_setpoint"] == 55.0
    
    # Point 4 (1 hour later = away_end): Should NOT be Away Temp (None or fall through)
    # Since optimized_setpoint is empty, it should simply not be present or match logic
    
    # Check 5th point (index 4) which is exactly AT away_end (or slightly after depending on < vs <=)
    # Logic is: if current_dt < away_end. 
    # timestamps[4] == away_end. So 55 should NOT apply.
    
    # With no optimized data, ideal_setpoint should be missing for this point
    assert "ideal_setpoint" not in forecast[4]
