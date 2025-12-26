"""Test to reproduce the config flow 500 error and verify fix."""
import pytest
from unittest.mock import patch

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

# Import constants
from custom_components.housetemp.const import (
    DOMAIN,
    CONF_SENSOR_INDOOR_TEMP,
    CONF_WEATHER_ENTITY,
    CONF_HEAT_PUMP_CONFIG,
    CONF_UA,
    CONF_C_THERMAL,
)

@pytest.mark.asyncio
async def test_options_flow_init(hass: HomeAssistant):
    """Test accessing options flow via the flow manager."""
    
    # 1. Create a Mock Config Entry
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_SENSOR_INDOOR_TEMP: "sensor.indoor",
            CONF_WEATHER_ENTITY: "weather.home",
            CONF_HEAT_PUMP_CONFIG: "{}",
        },
        options={},
        entry_id="test_opt_entry"
    )
    config_entry.add_to_hass(hass)
    
    # 2. Setup (loads integration)
    with patch("custom_components.housetemp.async_setup_entry", return_value=True):
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()

    # 3. Trigger Options Flow
    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "init"

@pytest.mark.asyncio
async def test_options_flow_migration_fallback(hass: HomeAssistant):
    """Test options flow falls back to data if options are missing (migration)."""
    
    # 1. Create Entry with DATA but NO OPTIONS
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_SENSOR_INDOOR_TEMP: "sensor.indoor",
            CONF_WEATHER_ENTITY: "weather.home",
            CONF_HEAT_PUMP_CONFIG: "{}",
            # Legacy settings in data:
            CONF_UA: 999.0,
            CONF_C_THERMAL: 888.0,
        },
        options={},
        entry_id="test_migration_entry"
    )
    config_entry.add_to_hass(hass)

    # 2. Setup (loads integration)
    with patch("custom_components.housetemp.async_setup_entry", return_value=True):
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()

    # 3. Init Options Flow
    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    assert result["step_id"] == "init"
    
    # Advance to Step 2 (Model Params) where UA is defined
    # We pass empty input to use defaults for Step 1
    result = await hass.config_entries.options.async_configure(
         result["flow_id"], user_input={}
    )
    assert result["step_id"] == "model_params"
    
    schema = result["data_schema"]
    
    # 3. Verify defaults are pulled from data
    ua_key = next(k for k in schema.schema if k == CONF_UA)
    c_thermal_key = next(k for k in schema.schema if k == CONF_C_THERMAL)
    
    assert ua_key.default() == 999.0
    assert c_thermal_key.default() == 888.0
