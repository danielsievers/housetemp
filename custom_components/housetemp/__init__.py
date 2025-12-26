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
    
    # Register calibrate service
    async def async_handle_calibrate(call):
        """Handle calibrate service call.
        
        Fetches history for the target entity (indoor temp), weather, solar, 
        and provided climate entities (hvac action).
        Runs optimization to fit parameters C, UA, K, Q, H.
        """
        import pandas as pd
        import numpy as np
        from homeassistant.exceptions import ServiceValidationError
        from homeassistant.util import dt as dt_util
        
        from .housetemp.utils import fetch_history_frame
        from .housetemp.measurements import Measurements
        from .housetemp.optimize import run_optimization
        
        # 1. Parse Arguments
        start_input = call.data.get("start_time")
        end_input = call.data.get("end_time")
        hvac_entities = call.data.get("hvac_action_entities", [])
        
        override_outdoor = call.data.get("outdoor_temp_entity")
        solar_entity_cal = call.data.get("solar_power_entity")
        if not solar_entity_cal:
             raise ServiceValidationError("solar_power_entity is required.")
        
        fix_passive = call.data.get("fix_passive", False)
        
        if not hvac_entities:
            raise ServiceValidationError("hvac_action_entities is required for calibration.")
        
        if not start_input or not end_input:
             raise ServiceValidationError("Both start_time and end_time are required.")

        start_time = dt_util.parse_datetime(start_input)
        if not start_time:
             raise ServiceValidationError(f"Invalid start_time: {start_input}")
             
        end_time = dt_util.parse_datetime(end_input)
        if not end_time:
             raise ServiceValidationError(f"Invalid end_time: {end_input}")

        if start_time >= end_time:
             raise ServiceValidationError("Start time must be before end time.")
             
        _LOGGER.info("Starting Calibration: %s to %s", start_time, end_time)
        
        # 2. Identify Target & Sources
        target_entries = await service.async_extract_config_entry_ids(call)
        if not target_entries:
            raise ServiceValidationError("No target selected. Target a HouseTemp entity.")
            
        results = {}
        
        for entry_id in target_entries:
            if entry_id not in hass.data.get(DOMAIN, {}):
                continue
                
            coord = hass.data[DOMAIN][entry_id]
            entry = coord.config_entry
            
            # Determine source entities from config or override
            # Indoor is always from config (defines the zone)
            from .const import CONF_SENSOR_INDOOR_TEMP, CONF_WEATHER_ENTITY
            
            indoor_entity = entry.data.get(CONF_SENSOR_INDOOR_TEMP)
            weather_entity = override_outdoor if override_outdoor else entry.data.get(CONF_WEATHER_ENTITY)
            
            # Solar is strictly from argument
            solar_entity_use = solar_entity_cal
            
            if not indoor_entity:
                 results[entry.title] = {"error": "Missing configured indoor entity."}
                 continue
            if not weather_entity:
                 results[entry.title] = {"error": "Missing weather/outdoor entity."}
                 continue
            
            # 3. Fetch History
            # Solar is always a single string from service arg
            solar_ids = [solar_entity_use]
            
            fetch_ids = [indoor_entity] + solar_ids + hvac_entities
            
            # Weather/Outdoor
            fetch_ids.append(weather_entity)
            
            _LOGGER.debug("Fetching history for: %s", fetch_ids)
            # minimal_response=False to get attributes (setpoints)
            df = await fetch_history_frame(hass, fetch_ids, start_time, end_time, minimal_response=False)
            
            if df.empty:
                results[entry.title] = {"error": "No historical data found."}
                continue
                
            # 4. Processing Phase 1: Extract Physics Values from State/Attributes
            try:
                # The DF cells contain dicts: {'state': '...', 'attributes': {...}, ...}
                
                clean_data = [] # List of dicts
                
                # Helper to safely get float state
                def get_float(cell):
                    try: 
                        return float(cell.get('state')) 
                    except (ValueError, AttributeError, TypeError): 
                        return np.nan
    
                hvac_cols = [c for c in hvac_entities if c in df.columns]
                if not hvac_cols:
                     # Should have been caught earlier but safety checks
                     results[entry.title] = {"error": "No data for HVAC action entities."}
                     continue
    
                solar_unit_str = None

                for ts, row in df.iterrows():
                    # Indoor / Weather / Solar
                    t_in = get_float(row.get(indoor_entity, {}))
                    t_out = get_float(row.get(weather_entity, {}))
                    
                    # Solar (Single)
                    sol = 0.0
                    if solar_entity_use:
                        cell_s = row.get(solar_entity_use, {})
                        v = get_float(cell_s)
                        if not np.isnan(v): sol = v
                        
                        # Capture unit once
                        if solar_unit_str is None:
                             attrs = cell_s.get('attributes', {})
                             solar_unit_str = attrs.get('unit_of_measurement')
                    
                    # HVAC State & Setpoint
                    # Logic: OR for state. Setpoint from active entity.
                    is_heating = False
                    is_cooling = False
                    active_setpoint = np.nan
                    
                    for c in hvac_cols:
                        cell = row.get(c, {})
                        val = str(cell.get('state', '')).lower()
                        attrs = cell.get('attributes', {})
                        
                        # Check Mode
                        heating_here = val in ['heating', 'heat', 'on']
                        cooling_here = val in ['cooling', 'cool']
                        
                        if heating_here: 
                            is_heating = True
                            # Only use temperature
                            sp = attrs.get('temperature')
                            if sp is not None:
                                 try: active_setpoint = float(sp)
                                 except: pass
                                 
                        elif cooling_here: 
                            is_cooling = True
                            # Only use temperature
                            sp = attrs.get('temperature')
                            if sp is not None:
                                 try: active_setpoint = float(sp)
                                 except: pass
                    
                    hvac_st = 0.0
                    if is_heating: hvac_st = 1.0
                    elif is_cooling: hvac_st = -1.0
                    
                    clean_data.append({
                        'time': ts,
                        't_in': t_in,
                        't_out': t_out,
                        'total_solar': sol,
                        'hvac_state': hvac_st,
                        'setpoint': active_setpoint
                    })
    
                df_clean = pd.DataFrame(clean_data)
                if df_clean.empty:
                    results[entry.title] = {"error": "Insufficient data after processing."}
                    continue
                    
                df_clean.set_index('time', inplace=True)
                
                # Resample
                from .housetemp.utils import upsample_dataframe
                from .const import CONF_MODEL_TIMESTEP, DEFAULT_MODEL_TIMESTEP
                
                timestep_min = entry.options.get(CONF_MODEL_TIMESTEP, DEFAULT_MODEL_TIMESTEP)
                
                df_res = upsample_dataframe(
                    df_clean, 
                    freq=f"{timestep_min}min", 
                    cols_linear=['t_in', 't_out', 'total_solar'], # Physics vars linear
                    cols_ffill=['hvac_state', 'setpoint']         # State/Setpoint hold
                )
                
                # After ffill, we might still have NaNs at start.
                df_res['setpoint'] = df_res['setpoint'].fillna(df_res['t_in'])
                
                # Post-Process: Convert Solar Units (Energy -> Power)
                # target: kW
                if solar_unit_str:
                     u = solar_unit_str.lower()
                     factor = 1.0
                     if 'mw' in u: factor = 1000.0
                     elif 'kw' in u: factor = 1.0
                     elif 'w' in u: factor = 0.001
                     
                     vals = df_res['total_solar'] * factor
                     
                     # Check if Energy (h suffix, e.g. kWh, Wh)
                     if 'h' in u:
                          # Cumulative Energy -> Power
                          # diff() gives dE per timestep
                          # Power = dE / dt(hours)
                          dt_h = timestep_min / 60.0
                          vals = vals.diff().fillna(0.0) / dt_h
                          # Fix negative spikes from meter resets or jitter
                          vals = vals.clip(lower=0.0)
                     
                     df_res['total_solar'] = vals
                     
                # 5. Build Measurements
                # Extract numpy arrays
                # Note: columns are normalized to 't_in', 't_out', 'total_solar', 'hvac_state', 'setpoint'
                
                t_in = df_res['t_in'].values
                t_out = df_res['t_out'].values
                solar = df_res['total_solar'].values
                hvac_state = df_res['hvac_state'].values
                
                # Use extracted setpoint
                setpoint = df_res['setpoint'].values
                
                dt_hours = df_res['dt'].values
                timestamps = df_res['time'].dt.to_pydatetime()
                
                # Drop NaNs (beginning/end of interpolation)
                mask = ~np.isnan(t_in) & ~np.isnan(t_out) & ~np.isnan(dt_hours) & ~np.isnan(setpoint)
                if np.sum(mask) < 100:
                     raise ValueError(f"Insufficient valid data points ({np.sum(mask)}).")
                
                meas = Measurements(
                    timestamps=timestamps[mask],
                    t_in=t_in[mask],
                    t_out=t_out[mask],
                    solar_kw=solar[mask],
                    hvac_state=hvac_state[mask],
                    setpoint=setpoint[mask],
                    dt_hours=dt_hours[mask]
                )
    
                # 6. Run Optimization
                # Get current params as initial guess
                from .const import CONF_C_THERMAL, CONF_UA, CONF_K_SOLAR, CONF_Q_INT, CONF_H_FACTOR, CONF_EFF_DERATE
                current_params = [
                    entry.data.get(CONF_C_THERMAL, 10000),
                    entry.data.get(CONF_UA, 750),
                    entry.data.get(CONF_K_SOLAR, 400),
                    entry.data.get(CONF_Q_INT, 1500),
                    entry.data.get(CONF_H_FACTOR, 10000)
                ]
                
                fixed_passive = None
                if fix_passive:
                     fixed_passive = current_params[:4] # [C, UA, K, Q]
                     
                # Get Hardware Capability
                hw = coord.heat_pump
                if not hw:
                     await coord._setup_heat_pump()
                     hw = coord.heat_pump
                
                opt_res = await hass.async_add_executor_job(
                    run_optimization,
                    meas,
                    hw,
                    current_params if not fix_passive else None,
                    fixed_passive,
                    entry.options.get(CONF_EFF_DERATE, 0.75) # fixed efficiency
                )
                
                if opt_res.success:
                     res_params = opt_res.x
                     # Map back to names
                     # [C, UA, K, Q, H, Eff]
                     results[entry.title] = {
                         "success": True,
                         "c_thermal": round(res_params[0], 1),
                         "ua": round(res_params[1], 1),
                         "k_solar": round(res_params[2], 1),
                         "q_int": round(res_params[3], 1),
                         "h_factor": round(res_params[4], 1),
                         "efficiency_derate": round(res_params[5], 3),
                         "cost": opt_res.fun,
                         "iterations": opt_res.nit
                     }
                else:
                     results[entry.title] = {
                         "success": False, 
                         "error": str(opt_res.message)
                     }

            except Exception as e:
                _LOGGER.exception("Calibration failed for %s", entry.title)
                results[entry.title] = {"error": str(e)}

        if call.return_response:
             return results
    
    hass.services.async_register(
        DOMAIN,
        "calibrate",
        async_handle_calibrate,
        supports_response=SupportsResponse.OPTIONAL
    )
    
    return True



async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up House Temp Prediction from a config entry."""
    from .statistics import StatsStore

    coordinator = HouseTempCoordinator(hass, entry)
    
    # Initialize StatsStore for this entry
    stats_store = StatsStore(hass, entry.entry_id, unique_id=entry.unique_id)
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
