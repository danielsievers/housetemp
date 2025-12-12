"""Test the config flow."""
from unittest.mock import patch
import pytest
from homeassistant import config_entries, data_entry_flow
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

# Import using the path we set up in conftest
from custom_components.housetemp.const import (
    DOMAIN,
    CONF_SENSOR_INDOOR_TEMP,
    CONF_WEATHER_ENTITY,
    CONF_SOLAR_ENTITY,
    CONF_C_THERMAL,
    CONF_HEAT_PUMP_CONFIG,
    CONF_SCHEDULE_CONFIG,
)

@pytest.mark.asyncio
async def test_flow_user_init(hass: HomeAssistant):
    """Test the initialization of the user step."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"

@pytest.mark.asyncio
async def test_flow_full_path(hass: HomeAssistant):
    """Test the full config flow path."""
    # 1. User Step (Identity)
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"

    user_input = {
        CONF_SENSOR_INDOOR_TEMP: "sensor.indoor",
        CONF_WEATHER_ENTITY: "weather.home",
        CONF_SOLAR_ENTITY: ["sensor.solar"],
        CONF_HEAT_PUMP_CONFIG: "{}",
    }
    
    # 2. Advance to Settings
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], user_input=user_input
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "model_settings"
    
    # 3. Model Settings Step
    settings_input = {
        CONF_C_THERMAL: 15000.0,
        CONF_SCHEDULE_CONFIG: '{"mode": "heat", "schedule": [{"weekdays": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"], "daily_schedule": [{"time": "00:00", "temp": 70}]}]}',
        # ... other defaults are handled by vol.Optional/Required defaults
    }
    
    with patch("custom_components.housetemp.async_setup_entry", return_value=True) as mock_setup:
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input=settings_input
        )
    
    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "House Temp Prediction"
    
    # Verify Split Storage
    assert result["data"][CONF_SENSOR_INDOOR_TEMP] == "sensor.indoor"
    assert result["options"][CONF_C_THERMAL] == 15000.0
