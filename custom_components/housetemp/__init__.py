"""The House Temp Prediction integration."""
from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .coordinator import HouseTempCoordinator

PLATFORMS: list[Platform] = [Platform.SENSOR]

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

    # Register Service
    async def async_handle_run_optimization(call):
        """Handle the service call to run optimization."""
        # Check if specific entry targeted via entity_id etc, or just run for this entry
        # Ideally we iterate all entries if global, but here we are in setup_entry.
        # Let's simple register it globally once? No, async_setup_entry runs multiple times.
        # But services are global to the domain.
        pass

    # Actually, proper place for global service is async_setup (top level) 
    # OR we check if service exists.
    if not hass.services.has_service(DOMAIN, "run_hvac_optimization"):
        from homeassistant.core import SupportsResponse
        
        async def async_service_handler(call):
            duration = call.data.get("duration")
            # Iterate all config entries for this domain
            current_entries = hass.config_entries.async_entries(DOMAIN)
            
            # We assume single entry or run for all. 
            # If multiple, we return result from the first one? Or map of results?
            # Typically integrations are 1 per house.
            # Let's return dict keyed by entry_name or just list if response requested.
            
            results = {}
            
            for entry in current_entries:
                if entry.entry_id in hass.data[DOMAIN]:
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
        
    if not hass.services.has_service(DOMAIN, "set_away"):
        from homeassistant.helpers import config_validation as cv
        import voluptuous as vol
        
        async def async_handle_set_away(call):
            # Parse Duration
            duration_data = call.data.get("duration")
            # cv.time_period creates a timedelta from dict or string
            try:
                if isinstance(duration_data, dict):
                    duration = cv.time_period(duration_data)
                elif isinstance(duration_data, str):
                     # Handle "HH:MM:SS" or "days" format if cv supports it directly
                     # cv.time_period expects dict usually for "hours": 1
                     # Actually cv.time_period_str might be needed for string?
                     # Let's rely on UI selector passing dict {"hours": X} usually.
                     # But manual call might pass string.
                     duration = cv.time_period_str(duration_data)
                else:
                    # Fallback or error
                    duration = cv.time_period(duration_data) # Let cv handle it
            except Exception:
                # Fallback to defaults or fail
                import logging
                logging.getLogger(DOMAIN).warning("Invalid duration format: %s", duration_data)
                return

            safety_temp = call.data.get("safety_temp", 50.0)
            
            current_entries = hass.config_entries.async_entries(DOMAIN)
            
            results = {}
            for entry in current_entries:
                if entry.entry_id in hass.data[DOMAIN]:
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

async def async_update_options(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Update options."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    if unload_ok := await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        hass.data[DOMAIN].pop(entry.entry_id)

    return unload_ok
