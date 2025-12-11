"""Tests for optimization logic in the coordinator."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta, timezone

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.housetemp.const import (
    DOMAIN,
    CONF_OPTIMIZATION_ENABLED,
    CONF_OPTIMIZATION_INTERVAL,
    CONF_SENSOR_INDOOR_TEMP,
    CONF_WEATHER_ENTITY,
    CONF_C_THERMAL,
    CONF_UA,
    CONF_K_SOLAR,
    CONF_Q_INT,
    CONF_H_FACTOR,
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
    entry = MockConfigEntry(domain=DOMAIN, data=VALID_CONFIG)
    entry.add_to_hass(hass)
    
    with patch("custom_components.housetemp.coordinator.HeatPump") as mock_hp:
        # Mock the instance returned by the class constructor
        instance = MagicMock()
        mock_hp.return_value = instance
        coord = HouseTempCoordinator(hass, entry)
        # Manually ensure heat_pump is set if setup fails silently or we need to force it
        coord.heat_pump = instance
        return coord

@pytest.mark.asyncio
async def test_optimization_disabled_by_default(hass, coordinator):
    """Test optimization does not run by default."""
    hass.states.async_set("sensor.indoor", "70.0")
    today = datetime.now(timezone.utc).isoformat()
    mock_forecast = [{"datetime": today, "temperature": 60.0}]
    hass.states.async_set("weather.home", "50.0", {"forecast": mock_forecast})

    with patch("custom_components.housetemp.coordinator.run_model") as mock_run:
        mock_run.return_value = ([], 0.0, [])
        await coordinator._async_update_data()
        
        # Check that optimization logic (logging for now) didn't run
        # We can check if last_optimization_time is still None
        assert coordinator.last_optimization_time is None

@pytest.mark.asyncio
async def test_optimization_enabled_runs(hass, coordinator):
    """Test optimization runs when enabled."""
    hass.states.async_set("sensor.indoor", "70.0")
    today = datetime.now(timezone.utc).isoformat()
    mock_forecast = [{"datetime": today, "temperature": 60.0}]
    hass.states.async_set("weather.home", "50.0", {"forecast": mock_forecast})
    
    # Enable optimization in options using async_update_entry
    hass.config_entries.async_update_entry(
        coordinator.config_entry,
        options={
            CONF_OPTIMIZATION_ENABLED: True,
            CONF_OPTIMIZATION_INTERVAL: 60
        }
    )

    with patch("custom_components.housetemp.coordinator.run_model") as mock_run, \
         patch("custom_components.housetemp.coordinator.optimize_hvac_schedule") as mock_opt:
        
        mock_run.return_value = ([], 0.0, [])
        mock_opt.return_value = [72.0] # Dummy optimized setpoint

        # First run
        await coordinator._async_update_data()
        
        # Verify optimization was called
        assert mock_opt.called
        assert coordinator.last_optimization_time is not None

@pytest.mark.asyncio
async def test_optimization_throttling(hass, coordinator):
    """Test optimization respects the interval."""
    hass.states.async_set("sensor.indoor", "70.0")
    today = datetime.now(timezone.utc).isoformat()
    mock_forecast = [{"datetime": today, "temperature": 60.0}]
    hass.states.async_set("weather.home", "50.0", {"forecast": mock_forecast})
    
    # Enable optimization, 60 min interval
    hass.config_entries.async_update_entry(
        coordinator.config_entry,
        options={
            CONF_OPTIMIZATION_ENABLED: True,
            CONF_OPTIMIZATION_INTERVAL: 60
        }
    )

    with patch("custom_components.housetemp.coordinator.run_model") as mock_run, \
         patch("custom_components.housetemp.coordinator.optimize_hvac_schedule") as mock_opt:
        
        mock_run.return_value = ([], 0.0, [])
        mock_opt.return_value = [72.0]

        # 1. First run: Should run
        await coordinator._async_update_data()
        assert mock_opt.call_count == 1
        t1 = coordinator.last_optimization_time
        
        # 2. Second run immediately: Should NOT run (throttled)
        await coordinator._async_update_data()
        assert mock_opt.call_count == 1 # Still 1
        t2 = coordinator.last_optimization_time
        assert t1 == t2 # Unchanged
        
        # 3. Third run after 61 minutes: Should run
        future_time = datetime.now(timezone.utc) + timedelta(minutes=61)
        with patch("custom_components.housetemp.coordinator.dt_util.now", return_value=future_time):
             await coordinator._async_update_data()
             assert mock_opt.call_count == 2 # Now 2
             t3 = coordinator.last_optimization_time
             assert t3 > t2

@pytest.mark.asyncio
async def test_optimization_trigger_on_config_update(hass):
    """Test that configuration update triggers a reload (which resets everything)."""
    from custom_components.housetemp.const import (
        DOMAIN, 
        CONF_SENSOR_INDOOR_TEMP, 
        CONF_C_THERMAL, 
        CONF_OPTIMIZATION_ENABLED, 
        CONF_OPTIMIZATION_INTERVAL,
        CONF_SCHEDULE_CONFIG
    )
    from unittest.mock import patch, MagicMock
    from pytest_homeassistant_custom_component.common import MockConfigEntry
    
    # Setup entry
    entry = MockConfigEntry(
        domain=DOMAIN, 
        data={CONF_SENSOR_INDOOR_TEMP: "sensor.indoor"},
        options={CONF_C_THERMAL: 1000.0, CONF_OPTIMIZATION_ENABLED: True}
    )
    entry.add_to_hass(hass)
    
    # Mock setup
    with patch("custom_components.housetemp.coordinator.HouseTempCoordinator._async_update_data", return_value={"timestamps": [], "predicted_temp": []}):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    # Patch async_reload to verify it is called
    with patch.object(hass.config_entries, "async_reload", return_value=None) as mock_reload:
        # Update options
        result = await hass.config_entries.options.async_init(entry.entry_id)
        await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                CONF_SCHEDULE_CONFIG: "[]",
                CONF_C_THERMAL: 2000.0,
                CONF_OPTIMIZATION_ENABLED: True
            }
        )
        await hass.async_block_till_done()
        
        # Verify reload was called
        mock_reload.assert_called_once_with(entry.entry_id)
