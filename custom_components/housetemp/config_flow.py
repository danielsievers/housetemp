"""Config flow for House Temp Prediction integration."""
from __future__ import annotations

import logging
import json
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
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
    CONF_SENSOR_INDOOR_TEMP,
    CONF_WEATHER_ENTITY,
    CONF_SOLAR_ENTITY,
    CONF_HEAT_PUMP_CONFIG,
    CONF_SCHEDULE_CONFIG,
    CONF_FORECAST_DURATION,
    CONF_UPDATE_INTERVAL,
    DEFAULT_FORECAST_DURATION,
    DEFAULT_FORECAST_DURATION,
    DEFAULT_UPDATE_INTERVAL,
    CONF_OPTIMIZATION_ENABLED,
    CONF_OPTIMIZATION_INTERVAL,
    DEFAULT_OPTIMIZATION_INTERVAL,
    MIN_OPTIMIZATION_INTERVAL,
    CONF_MODEL_TIMESTEP,
    DEFAULT_MODEL_TIMESTEP,
    MIN_MODEL_TIMESTEP,
    CONF_CONTROL_TIMESTEP,
    DEFAULT_CONTROL_TIMESTEP,
    MIN_CONTROL_TIMESTEP,
)
from homeassistant.core import callback

_LOGGER = logging.getLogger(DOMAIN)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_SENSOR_INDOOR_TEMP): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="sensor")
        ),
        vol.Required(CONF_WEATHER_ENTITY): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="weather")
        ),
        vol.Optional(CONF_SOLAR_ENTITY): selector.EntitySelector(
            selector.EntitySelectorConfig(domain="sensor", device_class=["power", "energy"], multiple=True)
        ),
    }
)

STEP_PARAMS_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_C_THERMAL, default=10000.0): vol.Coerce(float),
        vol.Required(CONF_UA, default=500.0): vol.Coerce(float),
        vol.Required(CONF_K_SOLAR, default=50.0): vol.Coerce(float),
        vol.Required(CONF_Q_INT, default=500.0): vol.Coerce(float),
        vol.Required(CONF_H_FACTOR, default=1000.0): vol.Coerce(float),
    }
)

STEP_CONFIGS_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_HEAT_PUMP_CONFIG): selector.TextSelector(
            selector.TextSelectorConfig(multiline=True)
        ),
        vol.Required(CONF_SCHEDULE_CONFIG): selector.TextSelector(
            selector.TextSelectorConfig(multiline=True)
        ),
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

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step (Entities)."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        self._data.update(user_input)
        return await self.async_step_params()

    async def async_step_params(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the parameters step."""
        if user_input is None:
            return self.async_show_form(
                step_id="params", data_schema=STEP_PARAMS_SCHEMA
            )

        self._data.update(user_input)
        return await self.async_step_configs()

    async def async_step_configs(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the configs step."""
        errors = {}

        if user_input is not None:
             # Validate JSON
             try:
                 json.loads(user_input[CONF_HEAT_PUMP_CONFIG])
             except ValueError:
                 errors[CONF_HEAT_PUMP_CONFIG] = "invalid_json"
                 
             try:
                 json.loads(user_input[CONF_SCHEDULE_CONFIG])
             except ValueError:
                 errors[CONF_SCHEDULE_CONFIG] = "invalid_json"

             if not errors:
                 self._data.update(user_input)
                 return self.async_create_entry(title="House Temp Prediction", data=self._data)

        return self.async_show_form(
             step_id="configs", 
             data_schema=STEP_CONFIGS_SCHEMA,
             errors=errors
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return OptionsFlowHandler(config_entry)

class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options."""

    async def async_step_init(self, user_input=None):
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        # Schema for options
        # We can add more advanced options here
        options_schema = vol.Schema(
            {
                vol.Optional(
                    CONF_OPTIMIZATION_ENABLED,
                    default=self.config_entry.options.get(
                        CONF_OPTIMIZATION_ENABLED, False
                    ),
                ): bool,
                vol.Optional(
                    CONF_OPTIMIZATION_INTERVAL,
                    default=self.config_entry.options.get(
                        CONF_OPTIMIZATION_INTERVAL, DEFAULT_OPTIMIZATION_INTERVAL
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=MIN_OPTIMIZATION_INTERVAL)),
                
                vol.Optional(
                    CONF_MODEL_TIMESTEP,
                    default=self.config_entry.options.get(
                        CONF_MODEL_TIMESTEP, DEFAULT_MODEL_TIMESTEP
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=MIN_MODEL_TIMESTEP)),
                
                vol.Optional(
                    CONF_CONTROL_TIMESTEP,
                    default=self.config_entry.options.get(
                        CONF_CONTROL_TIMESTEP, DEFAULT_CONTROL_TIMESTEP
                    ),
                ): vol.All(vol.Coerce(int), vol.Range(min=MIN_CONTROL_TIMESTEP)),
            }
        )

        return self.async_show_form(step_id="init", data_schema=options_schema)
