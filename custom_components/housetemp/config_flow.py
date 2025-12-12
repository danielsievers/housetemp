"""Config flow for House Temp Prediction integration."""
from __future__ import annotations

import logging
import json
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    CONF_C_THERMAL,
    CONF_UA,
    CONF_K_SOLAR,
    CONF_Q_INT,
    CONF_H_FACTOR,
    CONF_CENTER_PREFERENCE,
    CONF_SENSOR_INDOOR_TEMP,
    CONF_WEATHER_ENTITY,
    CONF_SOLAR_ENTITY,
    CONF_HEAT_PUMP_CONFIG,
    CONF_SCHEDULE_CONFIG,
    CONF_FORECAST_DURATION,
    CONF_UPDATE_INTERVAL,
    DEFAULT_FORECAST_DURATION,
    DEFAULT_UPDATE_INTERVAL,
    CONF_MODEL_TIMESTEP,
    DEFAULT_MODEL_TIMESTEP,
    MIN_MODEL_TIMESTEP,
    CONF_CONTROL_TIMESTEP,
    DEFAULT_CONTROL_TIMESTEP,
    MIN_CONTROL_TIMESTEP,
)
_LOGGER = logging.getLogger(DOMAIN)

# Validation Helpers
SCHEDULE_SCHEMA = vol.Schema({
    vol.Required("mode"): vol.In(["heat", "cool", "auto"]),
    vol.Optional("center_preference", default=1.0): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
    vol.Optional("avoid_defrost", default=True): bool,
    vol.Required("schedule"): [
        {
            vol.Required("weekdays"): [vol.In([
                "monday", "tuesday", "wednesday", "thursday", 
                "friday", "saturday", "sunday"
            ])],
            vol.Required("daily_schedule"): [
                {
                    vol.Required("time"): vol.Match(r"^([0-1][0-9]|2[0-3]):[0-5][0-9]$"),  # HH:MM
                    vol.Required("temp"): vol.All(vol.Coerce(float), vol.Range(min=40, max=95))
                }
            ]
        }
    ]
}, extra=vol.ALLOW_EXTRA)

def validate_schedule_timeline(schedule_data):
    """Ensure schedule creates a valid, gapless timeline."""
    weekday_names = ["monday", "tuesday", "wednesday", "thursday", 
                    "friday", "saturday", "sunday"]
    covered = set()
    
    # Check 1: Weekday Coverage & Duplicates
    if not isinstance(schedule_data.get("schedule"), list):
        raise ValueError("Schedule must contain a list under 'schedule' key")

    for rule in schedule_data["schedule"]:
        for day in rule["weekdays"]:
            day_clean = day.lower()
            if day_clean in covered:
                raise ValueError(f"Weekday '{day}' is defined in multiple schedule blocks")
            covered.add(day_clean)
    
    if len(covered) != 7:
        missing = set(weekday_names) - covered
        raise ValueError(f"Schedule must cover all 7 days. Missing: {missing}")
    
    # Check 2: Daily Schedule Format
    for rule in schedule_data["schedule"]:
        daily = rule["daily_schedule"]
        if not daily:
            raise ValueError("daily_schedule cannot be empty")
        
        # Verify chronological order
        times = [item["time"] for item in daily]
        if times != sorted(times):
            raise ValueError(f"Daily schedule times must be sorted chronologically: {times}")

    return True

# Step 1: Fixed Identity (Stored in DATA)
STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_SENSOR_INDOOR_TEMP): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="sensor")
        ),
        vol.Required(CONF_WEATHER_ENTITY): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="weather")
        ),
        vol.Optional(CONF_SOLAR_ENTITY): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="sensor", device_class="power", multiple=True)
        ),
        vol.Required(CONF_HEAT_PUMP_CONFIG): selector.TextSelector(
            selector.TextSelectorConfig(multiline=True)
        ),
    }
)

# Step 2: Modifiable Settings (Stored in OPTIONS)
# Physics Defaults: UA=750, C=10000, K=3000, Q=2000, H=5000
STEP_MODEL_SETTINGS_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_SCHEDULE_CONFIG, default="[]"): selector.TextSelector(
            selector.TextSelectorConfig(multiline=True)
        ),
        vol.Required(CONF_UA, default=750.0): vol.Coerce(float),
        vol.Required(CONF_C_THERMAL, default=10000.0): vol.Coerce(float),
        vol.Required(CONF_K_SOLAR, default=3000.0): vol.Coerce(float),
        vol.Required(CONF_Q_INT, default=2000.0): vol.Coerce(float),
        vol.Required(CONF_H_FACTOR, default=5000.0): vol.Coerce(float),
        vol.Required(CONF_CENTER_PREFERENCE, default=1.0): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
        vol.Required(CONF_FORECAST_DURATION, default=DEFAULT_FORECAST_DURATION): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=24)
        ),
        vol.Required(CONF_UPDATE_INTERVAL, default=DEFAULT_UPDATE_INTERVAL): vol.All(
             vol.Coerce(int), vol.Range(min=1)
        ),
    }
)

class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for House Temp Prediction."""

    VERSION = 1
    
    def __init__(self):
        self._data = {}
        self._options = {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step (Fixed Identity)."""
        errors = {}

        if user_input is not None:
             # Validate Heat Pump JSON
             try:
                 json.loads(user_input[CONF_HEAT_PUMP_CONFIG])
                 self._data.update(user_input)
                 return await self.async_step_model_settings()
             except ValueError:
                 errors[CONF_HEAT_PUMP_CONFIG] = "invalid_json"

        return self.async_show_form(
            step_id="user", 
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors
        )

    async def async_step_model_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the Settings step (Modifiable Options)."""
        errors = {}
        
        if user_input is not None:
            # Validate Schedule JSON
            try:
                schedule_data = json.loads(user_input[CONF_SCHEDULE_CONFIG])
                SCHEDULE_SCHEMA(schedule_data)        # Validate Types/Structure
                validate_schedule_timeline(schedule_data) # Validate Logic
                
                self._options.update(user_input)
                
                return self.async_create_entry(
                    title="House Temp Prediction", 
                    data=self._data,
                    options=self._options
                )
            except (ValueError, vol.Invalid) as e:
                errors[CONF_SCHEDULE_CONFIG] = "invalid_json"
                _LOGGER.warning("Invalid schedule config: %s", e)

        return self.async_show_form(
            step_id="model_settings", 
            data_schema=STEP_MODEL_SETTINGS_SCHEMA,
            errors=errors
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return OptionsFlowHandler(config_entry)

class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options."""

    def __init__(self, config_entry):
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(self, user_input=None):
        """Manage the options - main menu."""
        if user_input is not None:
             # Validate Schedule JSON
            try:
                schedule_data = json.loads(user_input[CONF_SCHEDULE_CONFIG])
                SCHEDULE_SCHEMA(schedule_data)        # Validate Types/Structure
                validate_schedule_timeline(schedule_data) # Validate Logic
                
                return self.async_create_entry(title="", data=user_input)
            except (ValueError, vol.Invalid) as e:
                errors = {CONF_SCHEDULE_CONFIG: "invalid_json"}
                _LOGGER.warning("Invalid schedule config in options: %s", e)
                # Fall through to show form with errors
                pass
        else:
            errors = {}

        # Get current values from OPTIONS (tunable)
        # Fallback to DATA only if migrating (optional, but good for safety)
        # Note: In this strict refactor, we assume options are populated.
        opts = self.config_entry.options
        
        # Re-create schema with current values as defaults
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_SCHEDULE_CONFIG,
                    default=opts.get(CONF_SCHEDULE_CONFIG, "[]"),
                ): selector.TextSelector(
                    selector.TextSelectorConfig(multiline=True)
                ),
                vol.Required(
                    CONF_UA,
                    default=opts.get(CONF_UA, 750.0),
                ): vol.Coerce(float),
                vol.Required(
                    CONF_C_THERMAL,
                    default=opts.get(CONF_C_THERMAL, 10000.0),
                ): vol.Coerce(float),
                vol.Required(
                    CONF_K_SOLAR,
                    default=opts.get(CONF_K_SOLAR, 3000.0),
                ): vol.Coerce(float),
                vol.Required(
                    CONF_Q_INT,
                    default=opts.get(CONF_Q_INT, 2000.0),
                ): vol.Coerce(float),
                vol.Required(
                    CONF_H_FACTOR,
                    default=opts.get(CONF_H_FACTOR, 5000.0),
                ): vol.Coerce(float),
                vol.Required(
                    CONF_CENTER_PREFERENCE,
                    default=opts.get(CONF_CENTER_PREFERENCE, 1.0),
                ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
                vol.Required(
                    CONF_FORECAST_DURATION,
                    default=opts.get(CONF_FORECAST_DURATION, DEFAULT_FORECAST_DURATION),
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=24)),
                vol.Required(
                    CONF_UPDATE_INTERVAL,
                    default=opts.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL),
                ): vol.All(vol.Coerce(int), vol.Range(min=1)),
                
                # Advanced Optimization Toggles (Keep specific optimization flags if they were there)
                vol.Optional(
                    CONF_MODEL_TIMESTEP,
                    default=opts.get(CONF_MODEL_TIMESTEP, DEFAULT_MODEL_TIMESTEP),
                ): vol.All(vol.Coerce(int), vol.Range(min=MIN_MODEL_TIMESTEP)),
                vol.Optional(
                    CONF_CONTROL_TIMESTEP,
                    default=opts.get(CONF_CONTROL_TIMESTEP, DEFAULT_CONTROL_TIMESTEP),
                ): vol.All(vol.Coerce(int), vol.Range(min=MIN_CONTROL_TIMESTEP)),
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema, errors=errors)
