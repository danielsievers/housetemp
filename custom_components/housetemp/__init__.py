"""The House Temp Prediction integration."""
from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, SupportsResponse
from homeassistant.helpers import config_validation as cv
import voluptuous as vol

from .const import DOMAIN, DEFAULT_AWAY_TEMP
from .coordinator import HouseTempCoordinator

PLATFORMS: list[Platform] = [Platform.SENSOR]

_LOGGER = logging.getLogger(__name__)


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the housetemp domain (services)."""
    
    # Register run_hvac_optimization service
    async def async_service_handler(call):
        """Handle run_hvac_optimization service call."""
        duration = call.data.get("duration")
        current_entries = hass.config_entries.async_entries(DOMAIN)
        
        results = {}
        
        for entry in current_entries:
            if entry.entry_id in hass.data.get(DOMAIN, {}):
                coord = hass.data[DOMAIN][entry.entry_id]
                try:
                    res = await coord.async_trigger_optimization(duration_hours=duration)
                    results[entry.title] = res
                except Exception as e:
                    results[entry.title] = {"error": str(e)}
        
        if call.return_response:
            return results
    
    hass.services.async_register(
        DOMAIN, 
        "run_hvac_optimization", 
        async_service_handler,
        supports_response=SupportsResponse.OPTIONAL
    )
    
    # Register set_away service
    async def async_handle_set_away(call):
        """Handle set_away service call."""
        from homeassistant.exceptions import ServiceValidationError
        
        duration_data = call.data.get("duration")
        
        try:
            if isinstance(duration_data, dict):
                duration = cv.time_period(duration_data)
            elif isinstance(duration_data, str):
                duration = cv.time_period_str(duration_data)
            else:
                duration = cv.time_period(duration_data)
        except (vol.Invalid, ValueError, TypeError) as e:
            raise ServiceValidationError(
                f"Invalid duration format: {duration_data}. Expected dict or string."
            ) from e

        safety_temp = call.data.get("safety_temp", DEFAULT_AWAY_TEMP)
        
        current_entries = hass.config_entries.async_entries(DOMAIN)
        
        results = {}
        for entry in current_entries:
            if entry.entry_id in hass.data.get(DOMAIN, {}):
                coord = hass.data[DOMAIN][entry.entry_id]
                try:
                    opt_result = await coord.async_set_away_mode(duration, safety_temp)
                    
                    response = {"success": True}
                    
                    # check if we should return energy stats
                    # opt_result is the full structure returned by async_trigger_optimization
                    # which contains "optimization_summary"
                    if opt_result and "optimization_summary" in opt_result:
                        summary = opt_result["optimization_summary"]
                        
                        # Check if away_end is within the optimization horizon
                        # We can roughly check if duration < forecast_duration
                        # Or better, check the timestamps.
                        # The optimization result has 'points' and 'duration', but not explicit end time easily accessible 
                        # without parsing.
                        # However, we know 'duration' passed to set_away is the away duration.
                        # And we know the forecast duration from config.
                        
                        forecast_duration_hours = coord.config_entry.options.get("forecast_duration", 48) # default 48
                        away_duration_hours = duration.total_seconds() / 3600.0
                        
                        # Requirement: "only do this calculation if the away end is within the 12h window that we did the optimization for"
                        # The user prompt said: "within the 12h window that we did the optimization for"
                        # But wait, optimization is done for 'forecast_duration' (default 48h).
                        # Let's assume "window" means the optimization horizon.
                        
                        if away_duration_hours <= forecast_duration_hours:
                            response["energy_used_schedule_kwh"] = summary.get("total_energy_use_kwh")
                            response["energy_used_optimized_kwh"] = summary.get("total_energy_use_optimized_kwh")

                    results[entry.title] = response
                except Exception as e:
                    _LOGGER.error("Failed to set away mode for %s: %s", entry.title, e)
                    results[entry.title] = {"success": False, "error": str(e)}
        
        if call.return_response:
            return results

    hass.services.async_register(
        DOMAIN,
        "set_away",
        async_handle_set_away,
        supports_response=SupportsResponse.OPTIONAL
    )
    
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up House Temp Prediction from a config entry."""

    coordinator = HouseTempCoordinator(hass, entry)
    
    # Fetch initial data so we have data when entities subscribe
    await coordinator.async_config_entry_first_refresh()

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = coordinator

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Reload entry when options update
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Update options."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    
    # Cancel away timer if exists
    if hasattr(coordinator, "_away_timer_unsub") and coordinator._away_timer_unsub:
        coordinator._away_timer_unsub()
    
    # Unload platforms
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)
        
        # Unregister services if this is the last entry
        remaining = hass.config_entries.async_entries(DOMAIN)
        if not remaining:
            if hass.services.has_service(DOMAIN, "run_hvac_optimization"):
                hass.services.async_remove(DOMAIN, "run_hvac_optimization")
            if hass.services.has_service(DOMAIN, "set_away"):
                hass.services.async_remove(DOMAIN, "set_away")
    
    return unload_ok
