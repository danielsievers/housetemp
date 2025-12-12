"""The House Temp Prediction integration."""
from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, SupportsResponse
from homeassistant.helpers import config_validation as cv
import voluptuous as vol

from .const import DOMAIN
from .coordinator import HouseTempCoordinator

PLATFORMS: list[Platform] = [Platform.SENSOR]


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
        duration_data = call.data.get("duration")
        
        try:
            if isinstance(duration_data, dict):
                duration = cv.time_period(duration_data)
            elif isinstance(duration_data, str):
                duration = cv.time_period_str(duration_data)
            else:
                duration = cv.time_period(duration_data)
        except Exception:
            import logging
            logging.getLogger(DOMAIN).warning("Invalid duration format: %s", duration_data)
            return

        safety_temp = call.data.get("safety_temp", 50.0)
        
        current_entries = hass.config_entries.async_entries(DOMAIN)
        
        results = {}
        for entry in current_entries:
            if entry.entry_id in hass.data.get(DOMAIN, {}):
                coord = hass.data[DOMAIN][entry.entry_id]
                try:
                    await coord.async_set_away_mode(duration, safety_temp)
                    results[entry.title] = "OK"
                except Exception as e:
                    results[entry.title] = {"error": str(e)}
        
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
