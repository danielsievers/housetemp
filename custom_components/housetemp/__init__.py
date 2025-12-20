"""The House Temp Prediction integration."""
from __future__ import annotations

import logging
from datetime import timedelta

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, SupportsResponse
from homeassistant.helpers import config_validation as cv
import voluptuous as vol
from homeassistant.helpers import service

from .const import DOMAIN, DEFAULT_AWAY_TEMP
from .coordinator import HouseTempCoordinator

PLATFORMS: list[Platform] = [Platform.SENSOR]

_LOGGER = logging.getLogger(__name__)


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the housetemp domain (services)."""
    
    # Register run_hvac_optimization service
    async def async_service_handler(call):
        """Handle run_hvac_optimization service call."""
        from homeassistant.exceptions import ServiceValidationError
        
        duration = call.data.get("duration")
        
        # Extract target config entries
        # This helper resolves entity_ids, device_ids, and area_ids to config entries
        # We assume the user targets the sensor entity provided by this integration
        target_entries = await service.async_extract_config_entry_ids(call)
        
        if not target_entries:
            raise ServiceValidationError(
                "No target selected. You must target a HouseTemp entity."
            )

        results = {}
        
        for entry_id in target_entries:
            if entry_id in hass.data.get(DOMAIN, {}):
                coord = hass.data[DOMAIN][entry_id]
                # Try to get the entry title for the result key
                entry_title = coord.config_entry.title
                try:
                    res = await coord.async_trigger_optimization(duration_hours=duration)
                    results[entry_title] = res
                except Exception as e:
                    results[entry_title] = {"error": str(e)}
            else:
                 # Should not happen if extract_config_entry_ids works correctly for loaded entries
                 pass
        
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
        
        target_entries = await service.async_extract_config_entry_ids(call)
        
        if not target_entries:
            raise ServiceValidationError(
                "No target selected. You must target a HouseTemp entity."
            )
        
        results = {}
        for entry_id in target_entries:
            if entry_id in hass.data.get(DOMAIN, {}):
                coord = hass.data[DOMAIN][entry_id]
                entry_title = coord.config_entry.title
                try:
                    opt_result = await coord.async_set_away_mode(duration, safety_temp)
                    
                    response = {"success": True}
                    
                    # check if we should return energy stats
                    # opt_result is the full structure returned by async_trigger_optimization
                    # which contains "optimization_summary"
                    if opt_result and "optimization_summary" in opt_result:
                        summary = opt_result["optimization_summary"]
                        
                        # Check if away_end is within the optimization horizon
                        forecast_duration_hours = coord.config_entry.options.get("forecast_duration", 48) # default 48
                        away_duration_hours = duration.total_seconds() / 3600.0
                        
                        if away_duration_hours <= forecast_duration_hours:
                            response["energy_used_schedule_kwh"] = summary.get("total_energy_use_kwh")
                            response["energy_used_optimized_kwh"] = summary.get("total_energy_use_optimized_kwh")

                    results[entry_title] = response
                except Exception as e:
                    _LOGGER.error("Failed to set away mode for %s: %s", entry_title, e)
                    results[entry_title] = {"success": False, "error": str(e)}
        
        if call.return_response:
            return results

    hass.services.async_register(
        DOMAIN,
        "set_away",
        async_handle_set_away,
        supports_response=SupportsResponse.OPTIONAL
    )
    
    # Register reset_stats service
    async def async_handle_reset_stats(call):
        """Handle reset_stats service call."""
        from homeassistant.exceptions import ServiceValidationError
        
        target_entries = await service.async_extract_config_entry_ids(call)
        
        if not target_entries:
            raise ServiceValidationError(
                "No target selected. You must target a HouseTemp entity."
            )
        
        results = {}
        for entry_id in target_entries:
            stats_store = hass.data.get(DOMAIN, {}).get(f"{entry_id}_stats")
            if stats_store:
                try:
                    await stats_store.async_reset()
                    results[entry_id] = {"success": True}
                except Exception as e:
                    _LOGGER.error("Failed to reset stats for %s: %s", entry_id, e)
                    results[entry_id] = {"success": False, "error": str(e)}
        
        if call.return_response:
            return results
    
    hass.services.async_register(
        DOMAIN,
        "reset_stats",
        async_handle_reset_stats,
        supports_response=SupportsResponse.OPTIONAL
    )
    
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up House Temp Prediction from a config entry."""
    from .statistics import StatsStore

    coordinator = HouseTempCoordinator(hass, entry)
    
    # Initialize StatsStore for this entry
    stats_store = StatsStore(hass, entry.entry_id)
    await stats_store.async_load()
    
    # 1. Setup reactive trackers so we catch any state changes during/after setup
    coordinator.async_setup_trackers()
    
    # Pass stats_store to coordinator for recording
    coordinator.stats_store = stats_store

    # 2. Fetch initial data
    try:
        await coordinator.async_config_entry_first_refresh()
    except Exception as e:
        # Check if it's a transient "not ready" state
        # We know if it's None in prepare_simulation_inputs it might return self.data (which is None)
        # but async_config_entry_first_refresh might raise UpdateFailed if it doesn't get data.
        # Actually coordinator.py:245 might return self.data (None if first run)
        _LOGGER.debug("First refresh failed: %s", e)
        from homeassistant.exceptions import ConfigEntryNotReady
        raise ConfigEntryNotReady(f"Input sensors not ready: {e}") from e

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = coordinator
    hass.data[DOMAIN][f"{entry.entry_id}_stats"] = stats_store

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Reload entry when options update
    entry.async_on_unload(entry.add_update_listener(async_update_options))

    return True


async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Update options."""
    coordinator = hass.data[DOMAIN].get(entry.entry_id)
    if not coordinator:
        await hass.config_entries.async_reload(entry.entry_id)
        return

    # Only reload if critical init-time settings change (like update_interval)
    # Most settings (physics, schedule, away) are read dynamically by the coordinator.
    new_interval = entry.options.get("update_interval") # We rely on default handling in coordinator if None, but here we check change
    # Note: coordinator.update_interval is a timedelta
    
    should_reload = False
    if new_interval is not None:
        if coordinator.update_interval != timedelta(minutes=new_interval):
            should_reload = True
            
    if should_reload:
        await hass.config_entries.async_reload(entry.entry_id)
    else:
        # Just refresh to apply new settings (physics, schedule, away)
        _LOGGER.debug("Options updated, skipping reload and refreshing coordinator.")
        await coordinator.async_request_refresh()


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    
    # Save stats before unloading
    stats_store = hass.data[DOMAIN].get(f"{entry.entry_id}_stats")
    if stats_store:
        stats_store.prune_old_data()
        await stats_store.async_save()
    
    # Cancel away timer if exists
    if hasattr(coordinator, "_away_timer_unsub") and coordinator._away_timer_unsub:
        coordinator._away_timer_unsub()
    
    # Unload platforms
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)
        hass.data[DOMAIN].pop(f"{entry.entry_id}_stats", None)
        
        # Unregister services if this is the last entry
        remaining = hass.config_entries.async_entries(DOMAIN)
        if not remaining:
            if hass.services.has_service(DOMAIN, "run_hvac_optimization"):
                hass.services.async_remove(DOMAIN, "run_hvac_optimization")
            if hass.services.has_service(DOMAIN, "set_away"):
                hass.services.async_remove(DOMAIN, "set_away")
            if hass.services.has_service(DOMAIN, "reset_stats"):
                hass.services.async_remove(DOMAIN, "reset_stats")
    
    return unload_ok
