"""Test the dynamic config flow defaults."""
from unittest.mock import Mock
import pytest
import json
from custom_components.housetemp.const import (
    DOMAIN,
    CONF_SCHEDULE_CONFIG,
    CONF_SCHEDULE_ENABLED,
    CONF_HVAC_MODE,
    CONF_COMFORT_MODE,
    CONF_DEADBAND_SLACK,
    CONF_SWING_TEMP,
    CONF_MIN_CYCLE_DURATION,
    CONF_MIN_SETPOINT,
    CONF_MAX_SETPOINT,
    DEFAULT_MAX_SETPOINT,
    DEFAULT_MIN_SETPOINT,
    CONF_AVOID_DEFROST,
)
from custom_components.housetemp.config_flow import OptionsFlowHandler

# Helper to mock the handler with a settable config_entry
class MockOptionsHandler(OptionsFlowHandler):
    def __init__(self, config_entry):
        self._config_entry = config_entry
        super().__init__()

    @property
    def config_entry(self):
        return self._config_entry

@pytest.mark.asyncio
async def test_options_flow_dynamic_cap_heating():
    """Test that MAX setpoint is capped in Heating mode based on schedule."""
    
    # 1. Setup Mock Entry
    config_entry = Mock()
    config_entry.data = {}
    # Simulate existing config with wide range so we can prove clamping works
    config_entry.options = {CONF_MAX_SETPOINT: 90.0}
    
    handler = MockOptionsHandler(config_entry)
    
    # 2. Step 1: Input a schedule with Max Temp = 72.0
    # Logic: smart_max = 72 + slack(2) + 2 = 76.0
    schedule_data = {
        "schedule": [{
            "weekdays": ["monday", "tuesday", "wednesday", "thursday", 
                        "friday", "saturday", "sunday"],
            "daily_schedule": [{"time": "00:00", "temp": 72.0}]
        }]
    }
    
    step1_input = {
        CONF_SCHEDULE_ENABLED: True,
        CONF_SCHEDULE_CONFIG: json.dumps(schedule_data),
        CONF_HVAC_MODE: "heat",
        CONF_COMFORT_MODE: "deadband",
        CONF_DEADBAND_SLACK: 2.0,
        CONF_SWING_TEMP: 1.0,
        CONF_MIN_CYCLE_DURATION: 10,
        CONF_AVOID_DEFROST: True,
    }
    
    # Mock async_show_form to capture the schema
    handler.async_show_form = Mock(return_value={"type": "form"})
    
    # Run Step 1
    await handler.async_step_init(user_input=step1_input)
    
    # Verify Step 1 data stored
    assert handler.step1_data == step1_input
    
    # Verify Schema in Step 2 call
    assert handler.async_show_form.called
    args, kwargs = handler.async_show_form.call_args
    schema = kwargs["data_schema"]
    
    # 3. Verify Dynamic Default for MAX SETPOINT
    # Expect 76.0
    
    max_sp_key = None
    for k in schema.schema:
        if k == CONF_MAX_SETPOINT:
            max_sp_key = k
            break
            
    assert max_sp_key is not None
    assert max_sp_key.default() == 76.0
    
    # Verify MIN SETPOINT is NOT capped (Asymmetric Rule)
    # Should correspond to global default since we have no existing config
    min_sp_key = None
    for k in schema.schema:
        if k == CONF_MIN_SETPOINT:
            min_sp_key = k
            break
    
    assert min_sp_key.default() == DEFAULT_MIN_SETPOINT

@pytest.mark.asyncio
async def test_dynamic_setpoint_capping_cooling():
    """Test that MIN setpoint is capped in Cooling mode based on schedule."""
    
    # Existing config has WIDE range
    config_entry = Mock()
    config_entry.data = {
        CONF_MIN_SETPOINT: 60.0,
        CONF_MAX_SETPOINT: 90.0
    }
    config_entry.options = {}
    
    handler = MockOptionsHandler(config_entry)
    handler.async_show_form = Mock(return_value={"type": "form"})
    
    # Schedule Min Temp = 74.0
    # Logic: smart_min = 74 - slack(2) - 2 = 70.0
    schedule_data = {
        "schedule": [{
            "weekdays": ["monday", "tuesday", "wednesday", "thursday", 
                        "friday", "saturday", "sunday"],
            "daily_schedule": [{"time": "00:00", "temp": 74.0}]
        }]
    }
    
    step1_input = {
        CONF_SCHEDULE_ENABLED: True,
        CONF_SCHEDULE_CONFIG: json.dumps(schedule_data),
        CONF_HVAC_MODE: "cool",
        CONF_DEADBAND_SLACK: 2.0,
        CONF_SWING_TEMP: 1.0,
        CONF_MIN_CYCLE_DURATION: 10,
        CONF_COMFORT_MODE: "deadband",
        CONF_AVOID_DEFROST: True
    }
    
    await handler.async_step_init(user_input=step1_input)
    
    args, kwargs = handler.async_show_form.call_args
    schema = kwargs["data_schema"]
    
    # Check MIN SETPOINT
    # Current config is 60.0. Smart Min is 70.0.
    # Logic: max(current, smart) -> max(60, 70) = 70.0
    
    min_sp_key = None
    for k in schema.schema:
        if k == CONF_MIN_SETPOINT:
            min_sp_key = k
            break
    
    print(f"DEBUG: Found min_sp default: {min_sp_key.default()}")
    assert min_sp_key.default() == 70.0
    
    # Check MAX SETPOINT - Unchanged (Asymmetric)
    max_sp_key = None
    for k in schema.schema:
        if k == CONF_MAX_SETPOINT:
            max_sp_key = k
            break
    
    # Should use the existing config value (90.0)
    assert max_sp_key.default() == 90.0

