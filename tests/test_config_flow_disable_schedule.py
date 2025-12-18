
from unittest.mock import patch, MagicMock
import pytest
from homeassistant import config_entries, data_entry_flow
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.housetemp.const import (
    DOMAIN, 
    CONF_SCHEDULE_CONFIG, 
    CONF_SCHEDULE_ENABLED, 
    CONF_HEAT_PUMP_CONFIG,
    DEFAULT_SCHEDULE_ENABLED,
    DEFAULT_HEAT_PUMP_CONFIG,
)
from custom_components.housetemp.config_flow import ConfigFlow

@pytest.mark.asyncio
async def test_options_flow_disable_schedule(hass: HomeAssistant):
    """Test disabling the schedule via options flow."""
    
    # Create a mock config entry
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        title="House Temp Prediction",
        data={},
        source="user",
        options={
            CONF_SCHEDULE_CONFIG: '{"schedule": []}',
            CONF_SCHEDULE_ENABLED: True
        },
        entry_id="test_entry_id",
    )
    config_entry.add_to_hass(hass)
    
    # Setup integration (mocked)
    with patch("custom_components.housetemp.async_setup_entry", return_value=True):
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()

    # Initialize Options Flow
    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    assert result["type"] == data_entry_flow.FlowResultType.FORM
    assert result["step_id"] == "init"
    
    # Verify Schema has the toggle
    schema = result["data_schema"]
    
    # 1. Test Submitting DISABLED schedule with INVALID JSON
    # This should pass because validation is skipped
    user_input = {
        CONF_SCHEDULE_ENABLED: False,
        CONF_SCHEDULE_CONFIG: "INVALID JSON []",
        # Add other required fields with defaults
        "hvac_mode": "heat",
        "avoid_defrost": True,
        "ua": 750.0,
        "c_thermal": 10000.0,
        "k_solar": 3000.0,
        "q_int": 2000.0,
        "h_factor": 5000.0,
        "center_preference": 1.0,
        "forecast_duration": 8,
        "update_interval": 15
    }
    
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input=user_input,
    )
    
    assert result["type"] == data_entry_flow.FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_SCHEDULE_ENABLED] is False
    assert result["data"][CONF_SCHEDULE_CONFIG] == "INVALID JSON []"


@pytest.mark.asyncio
async def test_options_flow_enable_schedule_validation(hass: HomeAssistant):
    """Test that enabling schedule enforces validation."""
    
    # Create a mock config entry
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        title="House Temp Prediction",
        data={},
        source="user",
        options={CONF_SCHEDULE_ENABLED: False},
        entry_id="test_entry_id_2",
    )
    config_entry.add_to_hass(hass)
    
    # Setup integration (mocked)
    with patch("custom_components.housetemp.async_setup_entry", return_value=True):
        await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
    
    # Initialize Options Flow
    result = await hass.config_entries.options.async_init(config_entry.entry_id)
    
    # 2. Test Submitting ENABLED schedule with INVALID JSON
    # This should FAIL
    user_input = {
        CONF_SCHEDULE_ENABLED: True,
        CONF_SCHEDULE_CONFIG: "INVALID JSON",
        CONF_HEAT_PUMP_CONFIG: DEFAULT_HEAT_PUMP_CONFIG,
        # Add other required fields
        "hvac_mode": "heat",
        "avoid_defrost": True,
        "ua": 750.0,
        "c_thermal": 10000.0,
        "k_solar": 3000.0,
        "q_int": 2000.0,
        "h_factor": 5000.0,
        "center_preference": 1.0,
        "forecast_duration": 8,
        "update_interval": 15
    }
    
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input=user_input,
    )
    
    assert result["type"] == data_entry_flow.FlowResultType.FORM
    assert result["errors"] == {CONF_SCHEDULE_CONFIG: "invalid_json"}

