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
    CONF_HVAC_MODE,
    CONF_AVOID_DEFROST,
    CONF_K_SOLAR,
    CONF_Q_INT,
    CONF_H_FACTOR,
    CONF_CENTER_PREFERENCE,
    CONF_SENSOR_INDOOR_TEMP,
    CONF_WEATHER_ENTITY,
    CONF_SOLAR_ENTITY,
    CONF_HEAT_PUMP_CONFIG,
    CONF_SCHEDULE_CONFIG,
    CONF_SCHEDULE_ENABLED,
    CONF_FORECAST_DURATION,
    CONF_UPDATE_INTERVAL,
    DEFAULT_FORECAST_DURATION,
    DEFAULT_UPDATE_INTERVAL,
    DEFAULT_C_THERMAL,
    DEFAULT_UA,
    DEFAULT_K_SOLAR,
    DEFAULT_Q_INT,
    DEFAULT_H_FACTOR,
    CONF_EFF_DERATE,
    DEFAULT_EFF_DERATE,
    DEFAULT_CENTER_PREFERENCE,
    DEFAULT_SCHEDULE_CONFIG,
    DEFAULT_SCHEDULE_ENABLED,
    CONF_MODEL_TIMESTEP,
    DEFAULT_MODEL_TIMESTEP,
    MIN_MODEL_TIMESTEP,
    CONF_CONTROL_TIMESTEP,
    DEFAULT_CONTROL_TIMESTEP,
    MIN_CONTROL_TIMESTEP,
    CONF_COMFORT_MODE,
    CONF_DEADBAND_SLACK,
    DEFAULT_COMFORT_MODE,
    DEFAULT_DEADBAND_SLACK,
    CONF_ENABLE_MULTISCALE,
    DEFAULT_ENABLE_MULTISCALE,
    DEFAULT_HEAT_PUMP_CONFIG,
    CONF_MIN_SETPOINT,
    CONF_MAX_SETPOINT,
    DEFAULT_MIN_SETPOINT,
    DEFAULT_MAX_SETPOINT,
    CONF_SWING_TEMP,
    DEFAULT_SWING_TEMP,
    CONF_MIN_CYCLE_DURATION,
    DEFAULT_MIN_CYCLE_MINUTES,
)
_LOGGER = logging.getLogger(DOMAIN)

# Validation Helpers
SCHEDULE_SCHEMA = vol.Schema({
    vol.Required("schedule"): [
        {
            vol.Required("weekdays"): [vol.In([
                "monday", "tuesday", "wednesday", "thursday", 
                "friday", "saturday", "sunday"
            ])],
            vol.Required("daily_schedule"): [
                {
                    vol.Required("time"): vol.Match(r"^([0-1][0-9]|2[0-3]):[0-5][0-9]$"),  # HH:MM
                    vol.Required("temp"): vol.All(vol.Coerce(float), vol.Range(min=40, max=95)),
                    vol.Optional("fixed", default=False): bool
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
# Note: HEAT_PUMP_CONFIG is now options-only (configured after setup, with default)
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

# Step 2: Modifiable Settings (Stored in OPTIONS)
# Physics Defaults: UA=750, C=10000, K=3000, Q=2000, H=5000
# Step 2: Modifiable Settings (Previously used in setup, now Options Only)
# Note: OptionsFlow uses dynamic schema, so this constant is technically unused
# but we leave it commented or remove it. We will remove it.

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
            self._data.update(user_input)
            
            # Generate Unique ID and Title
            sensor_entity = user_input[CONF_SENSOR_INDOOR_TEMP]
            await self.async_set_unique_id(sensor_entity)
            self._abort_if_unique_id_configured()
            
            # Dynamic Title: "SmartAss Thermostat (Friendly Name)"
            # Get the friendly name from the state machine
            friendly_name = sensor_entity
            state = self.hass.states.get(sensor_entity)
            if state and state.name:
                friendly_name = state.name
                
            title = f"SmartAss Thermostat ({friendly_name})"
            
            # DIRECT CREATION: Skip Model Settings Step
            # Heat pump config is now options-only (defaults used until configured)
            return self.async_create_entry(
                title=title, 
                data=self._data,
                # Initialize options as empty (Coordinator will use defaults)
                options={} 
            )

        return self.async_show_form(
            step_id="user", 
            data_schema=STEP_USER_DATA_SCHEMA,
            errors=errors
        )

    # async_step_model_settings Removed (Dead Code)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return OptionsFlowHandler()

class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options."""



    async def async_step_init(self, user_input=None):
        """Step 1: General Settings (Schedule, Comfort, Mode)."""
        errors = {} 

        if user_input is not None:
            # Validate Schedule JSON (if enabled)
            schedule_enabled = user_input.get(CONF_SCHEDULE_ENABLED, True)
            
            if schedule_enabled:
                try:
                    schedule_data = json.loads(user_input[CONF_SCHEDULE_CONFIG])
                    SCHEDULE_SCHEMA(schedule_data)        # Validate Types/Structure
                    validate_schedule_timeline(schedule_data) # Validate Logic
                except (ValueError, vol.Invalid) as e:
                    errors[CONF_SCHEDULE_CONFIG] = "invalid_json"
                    _LOGGER.warning("Invalid schedule configuration: %s", e)
            
            # Only proceed if no errors
            if not errors:
                self.step1_data = user_input
                return await self.async_step_model_params()
        
        # Get current values from OPTIONS (tunable)
        # Fallback to DATA only if migrating (optional, but good for safety)
        opts = self.config_entry.options
        data = self.config_entry.data
        
        # Helper to get value from opts -> data -> default
        def get_opt(key, default_val):
            return opts.get(key, data.get(key, default_val))
        
        # Step 1 Schema: General & Comfort
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_SCHEDULE_ENABLED,
                    default=get_opt(CONF_SCHEDULE_ENABLED, DEFAULT_SCHEDULE_ENABLED),
                ): selector.BooleanSelector(),
                vol.Required(
                    CONF_SCHEDULE_CONFIG,
                    default=get_opt(CONF_SCHEDULE_CONFIG, DEFAULT_SCHEDULE_CONFIG),
                ): selector.TextSelector(
                    selector.TextSelectorConfig(multiline=True)
                ),
                vol.Required(
                    CONF_HVAC_MODE,
                    default=get_opt(CONF_HVAC_MODE, "heat"),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=["heat", "cool"])
                ),
                vol.Required(
                    CONF_AVOID_DEFROST,
                    default=get_opt(CONF_AVOID_DEFROST, True),
                ): selector.BooleanSelector(),
                vol.Required(
                    CONF_COMFORT_MODE,
                    default=get_opt(CONF_COMFORT_MODE, DEFAULT_COMFORT_MODE),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=["quadratic", "deadband"])
                ),
                vol.Optional(
                    CONF_DEADBAND_SLACK,
                    default=get_opt(CONF_DEADBAND_SLACK, DEFAULT_DEADBAND_SLACK),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0.0, max=5.0, step=0.5, unit_of_measurement="°F")
                ),
                vol.Required(
                    CONF_SWING_TEMP,
                    default=get_opt(CONF_SWING_TEMP, DEFAULT_SWING_TEMP),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0.1, max=5.0, step=0.1, unit_of_measurement="°F")
                ),
                vol.Required(
                    CONF_MIN_CYCLE_DURATION,
                    default=get_opt(CONF_MIN_CYCLE_DURATION, DEFAULT_MIN_CYCLE_MINUTES),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0, max=60, step=1, unit_of_measurement="min")
                ),
            }
        )

        return self.async_show_form(step_id="init", data_schema=schema, errors=errors)

    async def async_step_model_params(self, user_input=None):
        """Step 2: Model & Physics (with Dynamic Setpoint Capping)."""
        errors = {}

        if user_input is not None:
            # Validate Heat Pump JSON
            hp_config = user_input.get(CONF_HEAT_PUMP_CONFIG)
            if hp_config:
                try:
                    json.loads(hp_config)
                except ValueError as e:
                    errors[CONF_HEAT_PUMP_CONFIG] = "invalid_json"
                    _LOGGER.warning("Invalid heat pump configuration: %s", e)
            
            if not errors:
                # Merge Step 1 and Step 2 data
                final_options = {**self.step1_data, **user_input}
                return self.async_create_entry(title="", data=final_options)

        # Get existing values
        opts = self.config_entry.options
        data = self.config_entry.data
        def get_opt(key, default_val):
            return opts.get(key, data.get(key, default_val))

        # --- Dynamic Default Logic ---
        # 1. Get Limits from CURRENT config (or static default) as fallback
        current_min = get_opt(CONF_MIN_SETPOINT, DEFAULT_MIN_SETPOINT)
        current_max = get_opt(CONF_MAX_SETPOINT, DEFAULT_MAX_SETPOINT)
        
        default_min_sp = current_min
        default_max_sp = current_max

        # 2. Try to calculate "Smart Caps" from Step 1 Schedule
        try:
            # Get data from Step 1
            if hasattr(self, "step1_data"):
                schedule_enabled = self.step1_data.get(CONF_SCHEDULE_ENABLED, True)
                schedule_json = self.step1_data.get(CONF_SCHEDULE_CONFIG, "")
                hvac_mode = self.step1_data.get(CONF_HVAC_MODE, "heat")
                slack = self.step1_data.get(CONF_DEADBAND_SLACK, DEFAULT_DEADBAND_SLACK)
                
                if schedule_enabled and schedule_json:
                    schedule_data = json.loads(schedule_json)
                    # Extract all temps
                    all_temps = []
                    for rule in schedule_data.get("schedule", []):
                        for item in rule.get("daily_schedule", []):
                            all_temps.append(float(item["temp"]))
                    
                    if all_temps:
                        min_sched = min(all_temps)
                        max_sched = max(all_temps)
                        
                        # Apply Asymmetric Logic:
                        # HEATING: Cap Max Setpoint. Min SP is untouched.
                        if hvac_mode == "heat":
                            smart_max = max_sched + slack + 2.0
                            # Propose the tighter constraint
                            default_max_sp = min(current_max, smart_max)
                            
                        # COOLING: Cap Min Setpoint. Max SP is untouched.
                        elif hvac_mode == "cool":
                            smart_min = min_sched - slack - 2.0
                            # Propose the tighter constraint (higher min)
                            default_min_sp = max(current_min, smart_min)

        except Exception as e:
            # Robustness: Fallback to existing values if anything fails
            _LOGGER.warning("Failed to calculate dynamic setpoint defaults: %s", e)

        # Step 2 Schema
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_HEAT_PUMP_CONFIG,
                    default=get_opt(CONF_HEAT_PUMP_CONFIG, DEFAULT_HEAT_PUMP_CONFIG),
                ): selector.TextSelector(
                    selector.TextSelectorConfig(multiline=True)
                ),
                 vol.Required(
                    CONF_UA,
                    default=get_opt(CONF_UA, DEFAULT_UA),
                ): vol.Coerce(float),
                vol.Required(
                    CONF_C_THERMAL,
                    default=get_opt(CONF_C_THERMAL, DEFAULT_C_THERMAL),
                ): vol.Coerce(float),
                vol.Required(
                    CONF_K_SOLAR,
                    default=get_opt(CONF_K_SOLAR, DEFAULT_K_SOLAR),
                ): vol.Coerce(float),
                vol.Required(
                    CONF_Q_INT,
                    default=get_opt(CONF_Q_INT, DEFAULT_Q_INT),
                ): vol.Coerce(float),
                vol.Required(
                    CONF_H_FACTOR,
                    default=get_opt(CONF_H_FACTOR, DEFAULT_H_FACTOR),
                ): vol.Coerce(float),
                vol.Optional(
                    CONF_EFF_DERATE,
                    default=get_opt(CONF_EFF_DERATE, DEFAULT_EFF_DERATE),
                ): vol.All(vol.Coerce(float), vol.Range(min=0.5, max=1.0)),
                vol.Required(
                    CONF_CENTER_PREFERENCE,
                    default=get_opt(CONF_CENTER_PREFERENCE, DEFAULT_CENTER_PREFERENCE),
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0.0, max=1.0, step=0.05, mode=selector.NumberSelectorMode.SLIDER)
                ),
                # Dynamic Defaults applied here
                vol.Optional(
                    CONF_MIN_SETPOINT,
                    default=default_min_sp, 
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=40, max=80, step=1, unit_of_measurement="°F", mode=selector.NumberSelectorMode.BOX)
                ),
                vol.Optional(
                    CONF_MAX_SETPOINT,
                    default=default_max_sp,
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=50, max=95, step=1, unit_of_measurement="°F", mode=selector.NumberSelectorMode.BOX)
                ),
                vol.Required(
                    CONF_FORECAST_DURATION,
                    default=get_opt(CONF_FORECAST_DURATION, DEFAULT_FORECAST_DURATION),
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=24)),
                vol.Required(
                    CONF_UPDATE_INTERVAL,
                    default=get_opt(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL),
                ): vol.All(vol.Coerce(int), vol.Range(min=1)),
                
                vol.Optional(
                    CONF_MODEL_TIMESTEP,
                    default=get_opt(CONF_MODEL_TIMESTEP, DEFAULT_MODEL_TIMESTEP),
                ): vol.All(vol.Coerce(int), vol.Range(min=MIN_MODEL_TIMESTEP)),
                vol.Optional(
                    CONF_CONTROL_TIMESTEP,
                    default=str(get_opt(CONF_CONTROL_TIMESTEP, DEFAULT_CONTROL_TIMESTEP)),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"value": "15", "label": "15 minutes"},
                            {"value": "30", "label": "30 minutes (Default)"},
                            {"value": "60", "label": "1 hour"},
                            {"value": "120", "label": "2 hours"},
                        ],
                        mode=selector.SelectSelectorMode.DROPDOWN
                    )
                ),
                vol.Required(
                    CONF_ENABLE_MULTISCALE,
                    default=get_opt(CONF_ENABLE_MULTISCALE, DEFAULT_ENABLE_MULTISCALE),
                ): selector.BooleanSelector(),
            }
        )

        return self.async_show_form(step_id="model_params", data_schema=schema, errors=errors)
