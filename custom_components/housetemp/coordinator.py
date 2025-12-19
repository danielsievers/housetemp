"""DataUpdateCoordinator for House Temp Prediction."""
from datetime import timedelta, datetime
import json
import time
import logging
import os
import re
import tempfile
from functools import partial

import numpy as np

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import (
    DataUpdateCoordinator,
    UpdateFailed,
)
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    CONF_C_THERMAL,
    CONF_UA,
    CONF_K_SOLAR,
    CONF_Q_INT,
    CONF_H_FACTOR,
    CONF_EFF_DERATE,
    CONF_SENSOR_INDOOR_TEMP,
    CONF_CENTER_PREFERENCE,
    CONF_WEATHER_ENTITY,
    CONF_SOLAR_ENTITY,
    CONF_HEAT_PUMP_CONFIG,
    CONF_SCHEDULE_CONFIG,
    CONF_SCHEDULE_ENABLED,
    CONF_FORECAST_DURATION,
    CONF_UPDATE_INTERVAL,
    DEFAULT_FORECAST_DURATION,
    DEFAULT_UPDATE_INTERVAL,
    CONF_MODEL_TIMESTEP,
    DEFAULT_MODEL_TIMESTEP,
    CONF_CONTROL_TIMESTEP,
    DEFAULT_CONTROL_TIMESTEP,
    CONF_HVAC_MODE,
    CONF_AVOID_DEFROST,
    CONF_COMFORT_MODE,
    CONF_DEADBAND_SLACK,
    DEFAULT_COMFORT_MODE,
    DEFAULT_DEADBAND_SLACK,
    DEFAULT_AWAY_TEMP,
    DEFAULT_C_THERMAL,
    DEFAULT_UA,
    DEFAULT_K_SOLAR,
    DEFAULT_Q_INT,
    DEFAULT_H_FACTOR,
    DEFAULT_EFF_DERATE,
    DEFAULT_CENTER_PREFERENCE,
    DEFAULT_SCHEDULE_CONFIG,
    DEFAULT_SCHEDULE_ENABLED,
    AWAY_WAKEUP_ADVANCE_HOURS,
    CONF_ENABLE_MULTISCALE,
    DEFAULT_ENABLE_MULTISCALE,
    DEFAULT_HEAT_PUMP_CONFIG,
    CONF_MIN_SETPOINT,
    CONF_MAX_SETPOINT,
    DEFAULT_MIN_SETPOINT,
    DEFAULT_MAX_SETPOINT,
)

# Import from the installed package
from .housetemp.run_model import run_model, HeatPump
from .housetemp.measurements import Measurements
from .housetemp.optimize import optimize_hvac_schedule
from .housetemp.schedule import process_schedule_data
from .housetemp.energy import estimate_consumption, calculate_energy_stats
from .input_handler import SimulationInputHandler

_LOGGER = logging.getLogger(DOMAIN)

class HouseTempCoordinator(DataUpdateCoordinator):
    """Class to manage fetching data and running the model."""

    def __init__(self, hass: HomeAssistant, config_entry):
        """Initialize."""
        self.config_entry = config_entry
        self.hass = hass
        
        update_interval_min = config_entry.options.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL)
        
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=update_interval_min),
        )
        self.config_entry = config_entry

        self.heat_pump = None
        # self._setup_heat_pump() # Will be called in async_update_data or first refresh
        
        # State for optimization
        # Change to dict: timestamp -> setpoint
        self.optimized_setpoints_map = {} 
        
        # New Input Handler
        self.input_handler = SimulationInputHandler(hass)
        
        # Track config state to invalidate cache on change
        self._last_config_id = self._get_config_id()
        self._optimization_status = None # Store status separately from data 
        self._unsub_state_trackers = []

    def async_setup_trackers(self):
        """Setup trackers for dependent entities."""
        # Clean up existing trackers if any
        for unsub in self._unsub_state_trackers:
            unsub()
        self._unsub_state_trackers = []

        data = self.config_entry.data
        entities = []
        if sensor_indoor := data.get(CONF_SENSOR_INDOOR_TEMP):
            entities.append(sensor_indoor)
        if weather_entity := data.get(CONF_WEATHER_ENTITY):
            entities.append(weather_entity)
        if solar_entities := data.get(CONF_SOLAR_ENTITY):
            if isinstance(solar_entities, str):
                entities.append(solar_entities)
            else:
                entities.extend(solar_entities)

        # Filter out empty/None
        entities = [e for e in entities if e]

        _LOGGER.debug("Setting up state trackers for entities: %s", entities)
        
        async def _async_state_changed(event):
            """Handle state changes of dependent entities."""
            old_state = event.data.get("old_state")
            new_state = event.data.get("new_state")
            
            if not new_state or new_state.state in ("unknown", "unavailable"):
                return

            if not old_state or old_state.state in ("unknown", "unavailable"):
                # Transition from unusable to usable -> Refresh
                _LOGGER.info("Entity %s became available. Requesting refresh.", event.data.get("entity_id"))
                self.async_set_updated_data(self.data) # Briefly update to trigger UI? Or just refresh.
                await self.async_request_refresh()

        self._unsub_state_trackers.append(
            async_track_state_change_event(self.hass, entities, _async_state_changed)
        )

    def _get_config_id(self):
        """Returns a stable tuple representing the current optimization-relevant config."""
        opts = self.config_entry.options
        data = self.config_entry.data
        
        def _get_stable_val(val):
            if isinstance(val, (dict, list)):
                 return json.dumps(val, sort_keys=True)
            return val

        return (
            _get_stable_val(opts.get(CONF_MODEL_TIMESTEP, DEFAULT_MODEL_TIMESTEP)),
            _get_stable_val(opts.get(CONF_CONTROL_TIMESTEP, DEFAULT_CONTROL_TIMESTEP)),
            _get_stable_val(opts.get(CONF_ENABLE_MULTISCALE, DEFAULT_ENABLE_MULTISCALE)),
            _get_stable_val(opts.get(CONF_COMFORT_MODE)),
            _get_stable_val(data.get(CONF_C_THERMAL)),
            _get_stable_val(data.get(CONF_UA)),
            _get_stable_val(data.get(CONF_K_SOLAR)),
            _get_stable_val(data.get(CONF_Q_INT)),
            _get_stable_val(data.get(CONF_H_FACTOR)),
            _get_stable_val(data.get(CONF_HEAT_PUMP_CONFIG)),
            _get_stable_val(data.get(CONF_SCHEDULE_CONFIG)),
        )

    async def _setup_heat_pump(self):
        """Initialize the HeatPump object from the config JSON."""
        # Heat Pump Config now lives in OPTIONS (migrated from data if needed)
        hp_config_str = self.config_entry.options.get(CONF_HEAT_PUMP_CONFIG, DEFAULT_HEAT_PUMP_CONFIG)
        if not hp_config_str:
            hp_config_str = DEFAULT_HEAT_PUMP_CONFIG

        # Run file I/O in executor
        def save_and_init():
            # Storage dir
            storage_dir = self.hass.config.path(".storage", DOMAIN)
            os.makedirs(storage_dir, exist_ok=True)
            hp_config_path = os.path.join(storage_dir, f"heat_pump_{self.config_entry.entry_id}.json")
            
            # Validate and Patch JSON
            try:
                # Strip comments for robust loading
                clean_config = re.sub(r"//.*", "", hp_config_str)
                data = json.loads(clean_config)
                
                # Check for mandatory keys required by the newer library version
                needs_patch = False
                mandatory_keys = [
                    'min_output_btu_hr', 'max_cool_btu_hr', 
                    'plf_low_load', 'plf_slope', 
                    'idle_power_kw', 'blower_active_kw'
                ]
                
                # Load defaults for patching
                default_data = json.loads(re.sub(r"//.*", "", DEFAULT_HEAT_PUMP_CONFIG))
                
                for key in mandatory_keys:
                    if key not in data:
                        _LOGGER.warning("Missing key '%s' in Heat Pump config. Patching with default. Please visit integration settings to update permanently.", key)
                        data[key] = default_data[key]
                        needs_patch = True
                
                # If we patched it, we should use the patched version
                if needs_patch:
                    final_config = json.dumps(data, indent=2)
                else:
                    final_config = hp_config_str
                
                # Basic validation 
                if "cop" not in data or "max_capacity" not in data:
                     _LOGGER.error("Heat Pump Config is missing core curves (cop, max_capacity). Sensors will be inaccurate.")
            except Exception as e:
                raise ValueError(f"Invalid Heat Pump JSON: {e}")
            
            with open(hp_config_path, "w") as f:
                f.write(final_config)
            
            return HeatPump(hp_config_path)

        try:
            self.heat_pump = await self.hass.async_add_executor_job(save_and_init)
        except Exception as e:
            _LOGGER.error("Failed to setup Heat Pump: %s", e)
            self.heat_pump = None

    def _expire_cache(self):
        """Remove stale cache entries (past timestamps only)."""
        if not self.optimized_setpoints_map:
            return
        
        now = dt_util.now()
        now_ts = int(now.timestamp())
        
        # Remove only PAST entries, keep all FUTURE entries
        old_size = len(self.optimized_setpoints_map)
        
        # Filter keys: keep if timestamp >= now
        # Note: We use a new dict comprehension for atomic replacement
        self.optimized_setpoints_map = {
            k: v for k, v in self.optimized_setpoints_map.items() 
            if k >= now_ts
        }
        
        # FIFO eviction if cache grows too large (prevent memory leak)
        # 3000 entries = ~10 days at 5 minute intervals
        if len(self.optimized_setpoints_map) > 3000:
            sorted_keys = sorted(self.optimized_setpoints_map.keys())
            keep_keys = sorted_keys[-3000:]  # Keep most recent/future 3000
            self.optimized_setpoints_map = {
                k: self.optimized_setpoints_map[k] for k in keep_keys
            }
        
        removed = old_size - len(self.optimized_setpoints_map)
        if removed > 0:
            _LOGGER.debug("Cache cleanup: removed %d past entries, %d remain", 
                         removed, len(self.optimized_setpoints_map))

    async def _async_update_data(self):
        """Fetch data and run the model."""
        # Clean up cache before processing
        self._expire_cache()
        
        # Check for Configuration changes (invalidates optimization cache)
        current_config_id = self._get_config_id()
        if current_config_id != self._last_config_id:
            _LOGGER.info("Optimization parameters changed. Clearing cache.")
            self.optimized_setpoints_map.clear()
            self._last_config_id = current_config_id

        if not self.heat_pump:
            await self._setup_heat_pump()
            if not self.heat_pump:
                raise UpdateFailed("Heat Pump not configured correctly")

        # 1. Get Inputs & Prepare Simulation Data
        try:
            res = await self._prepare_simulation_inputs()
            if res is None:
                 # Signals transient "not ready" - we don't want to raise UpdateFailed 
                 # which might trigger annoying UI error toast if it's just a startup race.
                 # But coordinator needs something. In HASS, returning None from _async_update_data 
                 # keeps the old data. If no data exists yet, it stays in 'initializing'.
                 _LOGGER.debug("Inputs not ready, skipping update.")
                 return self.data 
            measurements, params, start_time = res
        except Exception as e:
            _LOGGER.error("Error preparing simulation inputs: %s", e)
            raise UpdateFailed(f"Error preparing simulation inputs: {e}")

        # Unpack for local usage if needed, or just use measurements object
        timestamps = measurements.timestamps
        setpoint_arr = measurements.setpoint
        t_out_arr = measurements.t_out
        solar_arr = measurements.solar_kw
        
        duration_hours = self.config_entry.options.get(CONF_FORECAST_DURATION, DEFAULT_FORECAST_DURATION)

        # Reconstruct setpoints for display and simulation
        # 1. Attribute for display (fallback to 'current behavior' baseline)
        # 2. Simulation input (fallback to 'intended schedule' baseline)
        optimized_setpoint_attr = []
        sim_setpoints = []
        has_optimized_data = False
        
        found_count = 0
        for i, ts in enumerate(timestamps):
            ts_int = int(ts.timestamp())
            val = self.optimized_setpoints_map.get(ts_int)
            
            # Optimized value from cache
            if val is not None:
                optimized_setpoint_attr.append(val)
                sim_setpoints.append(val)
                found_count += 1
            else:
                # Optimized attribute strictly shows None for missing points
                optimized_setpoint_attr.append(None)
                
                # Simulation policy: Prefer Schedule -> Off (NaN)
                target = measurements.target_temp[i]
                if target is not None and not np.isnan(target):
                    sim_setpoints.append(target)
                else:
                    sim_setpoints.append(np.nan)
        
        # Explicit No-Schedule Policy: 
        # If all targets are NaN and no optimized data, force HVAC off for safety (avoid NaN propagation)
        targets = measurements.target_temp
        has_schedule = targets is not None and np.any(~np.isnan(targets))
        
        if found_count > 0:
            has_optimized_data = True
            _LOGGER.debug(f"Found {found_count} cached optimized setpoints for current window.")
        elif not has_schedule:
            _LOGGER.debug("No schedule and no optimized setpoints available. Disabling HVAC in simulation.")
            measurements.hvac_state[:] = 0
            # Explicit status for UX transparency
            self._optimization_status = {
                "success": False, 
                "message": "No schedule available; HVAC disabled for simulation",
                "code": "no_schedule",
                "scope": "simulation"
            }
        elif self._optimization_status and self._optimization_status.get("code") == "no_schedule":
            # Schedule is back! Clear the old error status
            self._optimization_status = None

        # Use best available setpoints (Optimized -> Target -> Clamped Input)
        measurements.setpoint = np.array(sim_setpoints, dtype=float)
        
        # Universal NaN-Clamping: Replace ALL NaNs in the setpoint array (partial gaps) with safe constant
        # AND force HVAC off for those specific points to avoid spurious action
        mask = np.isnan(measurements.setpoint)
        if np.any(mask):
            measurements.setpoint[mask] = float(measurements.t_in[0])
            # Defensive check: ensure hvac_state is aligned and mutable
            if isinstance(measurements.hvac_state, np.ndarray) and len(measurements.hvac_state) == len(mask):
                 measurements.hvac_state[mask] = 0


        
        sim_temps, _, hvac_delivered, hvac_produced = await self.hass.async_add_executor_job(
            run_model, params, measurements, self.heat_pump, duration_hours*60
        )
        
        if len(sim_temps) > 0:
            _LOGGER.info("Simulation complete. Predicted Final Temp: %.1f", sim_temps[-1])

        # --- Energy Calculation ---
        naive_energy_kwh = None
        optimized_energy_kwh = None

        if self.heat_pump:
            # 1. Calculate energy for the run we just did
            # Use 'hvac_produced' (Gross) for accurate billing
            current_energy_res = calculate_energy_stats(hvac_produced, measurements, self.heat_pump, params[4])
            current_energy_kwh = current_energy_res.get('total_kwh', 0.0)
            current_energy_steps = current_energy_res.get('kwh_steps')

            if has_optimized_data:
                # The run we just did was OPTIMIZED
                optimized_energy_kwh = current_energy_kwh
                
                # 2. Run Naive (Schedule) Simulation for comparison
                # Temporarily revert setpoints to the original schedule for the naive run
                optimized_setpoint_array = measurements.setpoint
                measurements.setpoint = np.array(setpoint_arr)
                
                _, _, _, hvac_produced_naive = await self.hass.async_add_executor_job(
                    run_model, params, measurements, self.heat_pump, duration_hours*60
                )
                
                naive_res = calculate_energy_stats(hvac_produced_naive, measurements, self.heat_pump, params[4])
                naive_energy_kwh = naive_res.get('total_kwh', 0.0)
                
                # Restore Optimized Array for consistency if needed downstream
                # (Though we pass setpoint_arr explicitly to build_data if we want schedule)
                measurements.setpoint = optimized_setpoint_array
                
            else:
                # Even if not optimized, this IS the naive run
                naive_energy_kwh = current_energy_kwh
                optimized_energy_kwh = None # Not available

        # 8. Return Result
        return self._build_coordinator_data(
            timestamps, sim_temps, measurements, optimized_setpoint_attr if has_optimized_data else None,
            naive_energy_kwh, optimized_energy_kwh, setpoint_arr,
            energy_kwh_steps=current_energy_steps if self.heat_pump else None
        )

    def _build_coordinator_data(self, timestamps, sim_temps, measurements, optimized_setpoints=None, naive_kwh=None, optimized_kwh=None, original_schedule=None, energy_kwh_steps=None):
        """Build the coordinator data dict from simulation results."""
        
        # Use provided original_schedule if available, otherwise fallback to measurements.setpoint 
        # (which might be optimized if not careful, but usually we pass original_schedule now)
        display_setpoints = original_schedule if original_schedule is not None else measurements.setpoint

        result = {
            "timestamps": timestamps,
            "predicted_temp": sim_temps,
            "hvac_state": measurements.hvac_state,
            "setpoint": display_setpoints, # Return original schedule for comparison
            "solar": measurements.solar_kw,
            "outdoor": measurements.t_out,
            "energy_kwh": naive_kwh,
            "optimized_energy_kwh": optimized_kwh,
            "energy_kwh_steps": energy_kwh_steps
        }
        
        # Add Away Info for Sensor
        is_away, away_end, away_temp = self._get_away_status()
        if is_away:
             result["away_info"] = {
                 "active": True,
                 "temp": float(away_temp),
                 "end": away_end.isoformat()
             }
        else:
             result["away_info"] = {"active": False}
        
        if optimized_setpoints is not None:
            result["optimized_setpoint"] = optimized_setpoints
        
        # Include current optimization status in data for sensor attributes
        if self._optimization_status:
            result["optimization_status"] = self._optimization_status
        
        return result

    async def async_trigger_optimization(self, duration_hours=None):
        """Manually trigger the HVAC optimization process."""
        _LOGGER.info("Manual optimization triggered. Duration override: %s", duration_hours)
        
        # 1. Ensure heat pump is ready
        if not self.heat_pump:
            await self._setup_heat_pump()
            
        # Check Schedule Enabled
        schedule_enabled = self.config_entry.options.get(CONF_SCHEDULE_ENABLED, DEFAULT_SCHEDULE_ENABLED)
        if not schedule_enabled:
            _LOGGER.warning("Optimization triggered but schedule is disabled. Ignoring.")
            return # Should we raise or just return empty?
        
        try:
            # Pass duration override if provided
            measurements, params, start_time = await self._prepare_simulation_inputs(duration_override=duration_hours)
        except Exception as e:
            _LOGGER.error("Could not prepare data for optimization: %s", e)
            raise e # Raise to caller for Service handling

        # Optimization Parameters
        model_timestep = self.config_entry.options.get(CONF_MODEL_TIMESTEP, DEFAULT_MODEL_TIMESTEP)
        control_timestep = self.config_entry.options.get(CONF_CONTROL_TIMESTEP, DEFAULT_CONTROL_TIMESTEP)
        
        _LOGGER.info(f"Running HVAC Optimization (Model: {model_timestep}m, Control: {control_timestep}m)...")
        opt_start_time = time.time()
        
        # Target Temps (Schedule)
        target_temps = measurements.setpoint.copy()
        
        # -- Baseline Energy Calculation (Before Optimization) --
        # We need to compute energy using the schedule (measurements.setpoint)
        # Note: estimate_consumption might mutate measurements? It shouldn't, but let's be safe.
        # It calls run_model, which doesn't mutate.
        baseline_kwh = 0.0
        try:
             baseline_res = await self.hass.async_add_executor_job(
                 estimate_consumption, measurements, params, self.heat_pump
             )
             baseline_kwh = baseline_res.get('total_kwh', 0.0)
        except Exception as e:
             _LOGGER.warning("Failed to estimate baseline energy: %s", e)
             
        # Use configured preference (Default 1.0)
        # Build comfort_config from options
        options = self.config_entry.options
        comfort_config = {
            "mode": options.get(CONF_HVAC_MODE, "heat"),
            "center_preference": float(options.get(CONF_CENTER_PREFERENCE, DEFAULT_CENTER_PREFERENCE)),
            "avoid_defrost": options.get(CONF_AVOID_DEFROST, True),
            "comfort_mode": options.get(CONF_COMFORT_MODE, DEFAULT_COMFORT_MODE),
            "deadband_slack": float(options.get(CONF_DEADBAND_SLACK, DEFAULT_DEADBAND_SLACK)),
            "min_setpoint": float(options.get(CONF_MIN_SETPOINT, DEFAULT_MIN_SETPOINT)),
            "max_setpoint": float(options.get(CONF_MAX_SETPOINT, DEFAULT_MAX_SETPOINT)),
        }
        
        try:
            optimization_result = await self.hass.async_add_executor_job(
                partial(
                    optimize_hvac_schedule,
                    measurements,
                    params,
                    self.heat_pump,
                    target_temps,
                    comfort_config,
                    block_size_minutes=control_timestep,
                    enable_multiscale=self.config_entry.options.get(CONF_ENABLE_MULTISCALE, DEFAULT_ENABLE_MULTISCALE)
                )
            )
            
            # Unpack result (setpoints, meta)
            if isinstance(optimization_result, tuple):
                optimized_setpoints, meta = optimization_result
                if self.data is None:
                    self.data = {}
                self.data["optimization_status"] = meta
                
                # Check for Failure (Strict Mode)
                if optimized_setpoints is None:
                     _LOGGER.error("Optimization Failed: %s. CLEARING CACHE.", meta.get("message"))
                     self.optimized_setpoints_map.clear()
                     return
                else:
                     if meta.get("success"):
                         _LOGGER.info("Optimization converged (Cost: %.2f)", meta.get("cost", 0.0))
                     else:
                         _LOGGER.warning("Optimization warning: %s", meta.get("message"))
            else:
                # Legacy fallback
                optimized_setpoints = optimization_result
                self.data["optimization_status"] = {"success": True, "message": "Legacy Result"}
            
            opt_duration = time.time() - opt_start_time
            _LOGGER.info("Optimization completed in %.2f seconds", opt_duration)
            
            # Map optimized setpoints to timestamps
            timestamps = measurements.timestamps
            if optimized_setpoints is not None and len(optimized_setpoints) == len(timestamps):
                 new_cache = {}
                 for i, ts in enumerate(timestamps):
                     ts_int = int(ts.timestamp())
                     new_cache[ts_int] = optimized_setpoints[i]
                 self.optimized_setpoints_map.update(new_cache)
            

            
            # Update data directly (Race-free rebuild from cache)
            sim_setpoints = []
            for ts in timestamps:
                ts_int = int(ts.timestamp())
                if ts_int in self.optimized_setpoints_map:
                    sim_setpoints.append(self.optimized_setpoints_map[ts_int])
                else:
                    # Missing (e.g. failure cleared it) -> None
                    sim_setpoints.append(None)
            
            measurements.setpoint = np.array(sim_setpoints)

            # Ensure duration is valid for simulation
            sim_duration_hours = duration_hours
            if sim_duration_hours is None:
                 sim_duration_hours = self.config_entry.options.get(CONF_FORECAST_DURATION, DEFAULT_FORECAST_DURATION)
                 
            # Run Model for Temp Curve
            _LOGGER.info("Running simulation for service response (duration: %.1f h)...", sim_duration_hours)
            sim_temps, _, hvac_delivered, hvac_produced = await self.hass.async_add_executor_job(
                run_model, params, measurements, self.heat_pump, sim_duration_hours*60
            )
            
            # -- Optimized Energy Calculation --
            # Now measurements.setpoint is OPTIMIZED.
            optimized_kwh = 0.0
            optimized_steps = None
            try:
                 # Reuse HVAC PRODUCED outputs from the simulation we just ran!
                 opt_res = calculate_energy_stats(hvac_produced, measurements, self.heat_pump, params[4])
                 optimized_kwh = opt_res.get('total_kwh', 0.0)
                 optimized_steps = opt_res.get('kwh_steps')
            except Exception as e:
                 _LOGGER.warning("Failed to estimate optimized energy: %s", e)

            # --- Update Coordinator Data Immediately ---
            # Update data directly to avoid timestamp mismatches from a refresh request
            result_data = self._build_coordinator_data(
                timestamps, sim_temps, measurements, optimized_setpoints,
                naive_kwh=baseline_kwh, optimized_kwh=optimized_kwh,
                original_schedule=target_temps,
                energy_kwh_steps=optimized_steps
            )
            
            # Preserve optimization status
            if self.data and "optimization_status" in self.data:
                result_data["optimization_status"] = self.data["optimization_status"]

            self.async_set_updated_data(result_data)
            
            # Return Forecast Structure (similar to sensor)
            forecast_data = []
            for i, ts in enumerate(timestamps):
                # Prepare item dict
                item = {
                    "datetime": ts.isoformat(),
                    "target_temp": float(target_temps[i]), # Original Schedule
                    "outdoor_temp": float(measurements.t_out[i]),
                    "solar_kw": float(measurements.solar_kw[i]),
                    "ideal_setpoint": float(optimized_setpoints[i]) if i < len(optimized_setpoints) else None,
                    "predicted_temp": float(sim_temps[i]) if i < len(sim_temps) else None,
                    "energy_kwh": float(optimized_steps[i]) if optimized_steps is not None and i < len(optimized_steps) else None
                }
                # hvac_action from schedule
                state_val = measurements.hvac_state[i]
                if state_val > 0:
                    item["hvac_action"] = "heating"
                elif state_val < 0:
                     item["hvac_action"] = "cooling"
                else:
                     item["hvac_action"] = "off"
                     
                forecast_data.append(item)
                
            return {
                "forecast": forecast_data,
                "optimization_summary": {
                    "duration_seconds": opt_duration,
                    "points": len(timestamps),
                    "start_time": start_time.isoformat(),
                    "total_energy_use_kwh": float(baseline_kwh),
                    "total_energy_use_optimized_kwh": float(optimized_kwh)
                }
            }
            
        except Exception as e:
            _LOGGER.error("Optimization failed: %s", e)
            import traceback
            _LOGGER.error(traceback.format_exc())
            raise e

    async def _prepare_simulation_inputs(self, duration_override=None):
        """Helper to fetch data and prepare measurements (shared logic)."""
        
        # 1. Get Inputs
        # Fixed Identity (DATA)
        data = self.config_entry.data
        sensor_indoor = data.get(CONF_SENSOR_INDOOR_TEMP)
        weather_entity = data.get(CONF_WEATHER_ENTITY)
        solar_entity = data.get(CONF_SOLAR_ENTITY)
        
        # Modifiable Settings (OPTIONS)
        options = self.config_entry.options
        if duration_override is not None:
             try:
                 duration_hours = float(duration_override)
             except (ValueError, TypeError):
                 _LOGGER.warning("Invalid duration override '%s', using config", duration_override)
                 duration_hours = options.get(CONF_FORECAST_DURATION, DEFAULT_FORECAST_DURATION)
        else:
             duration_hours = options.get(CONF_FORECAST_DURATION, DEFAULT_FORECAST_DURATION)
        

        _LOGGER.debug("DEBUG_OPTIONS: %s", options)
        
        # Parameters (Physics) - From Options
        params = [
            options.get(CONF_C_THERMAL, DEFAULT_C_THERMAL),
            options.get(CONF_UA, DEFAULT_UA),
            options.get(CONF_K_SOLAR, DEFAULT_K_SOLAR),
            options.get(CONF_Q_INT, DEFAULT_Q_INT),
            options.get(CONF_H_FACTOR, DEFAULT_H_FACTOR),
            options.get(CONF_EFF_DERATE, DEFAULT_EFF_DERATE),  # Duct delivery efficiency
        ]
        _LOGGER.debug("DEBUG_PARAMS: %s", params)

        
        _LOGGER.debug("Preparing simulation inputs with params: %s", params)

        # 2. Get Current State
        indoor_state = self.hass.states.get(sensor_indoor)
        if not indoor_state:
             raise UpdateFailed(f"Indoor sensor {sensor_indoor} not found in state machine")
             
        if indoor_state.state in ("unknown", "unavailable"):
             # This is a transient error during startup/update
             _LOGGER.debug("Indoor sensor %s is currently unavailable", sensor_indoor)
             return None # Signal 'not ready' clearly if we want, or raise specific error
        
        try:
            current_temp = float(indoor_state.state)
        except ValueError:
            raise UpdateFailed(f"Invalid indoor temp (not a number): {indoor_state.state}")

        # 3. Get Weather Forecast
        weather_state = self.hass.states.get(weather_entity)
        if not weather_state:
            raise UpdateFailed(f"Weather entity {weather_entity} not found")
        
        if weather_state.state in ("unknown", "unavailable"):
             _LOGGER.debug("Weather entity %s is currently unavailable", weather_entity)
             return None # Signal 'not ready'
        
        # Try Attribute first (Legacy)
        forecast = weather_state.attributes.get("forecast")
        
        # Try modern service call if attribute missing
        if forecast is None:
             try:
                 response = await self.hass.services.async_call(
                     "weather", 
                     "get_forecasts", 
                     {"type": "hourly", "entity_id": weather_entity}, 
                     blocking=True, 
                     return_response=True
                 )
                 if response and weather_entity in response:
                     forecast = response[weather_entity].get("forecast")
             except Exception as e:
                 _LOGGER.debug("Failed to get hourly forecast via service: %s", e)
                 
             # Fallback to daily if hourly failed or returned nothing
             if forecast is None:
                 try:
                     response = await self.hass.services.async_call(
                         "weather", 
                         "get_forecasts", 
                         {"type": "daily", "entity_id": weather_entity}, 
                         blocking=True, 
                         return_response=True
                     )
                     if response and weather_entity in response:
                         forecast = response[weather_entity].get("forecast")
                 except Exception as e:
                     _LOGGER.debug("Failed to get daily forecast via service: %s", e)

        if not forecast:
            raise UpdateFailed(
                f"No forecast data available from {weather_entity}. "
                "Check weather integration and entity configuration."
            )

        # 4. Get Solar Forecast
        solar_forecast_data = []
        solar_entities = solar_entity
        if solar_entities:
            if isinstance(solar_entities, str):
                solar_entities = [solar_entities]
            
            for entity_id in solar_entities:
                s_state = self.hass.states.get(entity_id)
                if s_state:
                     partial_forecast = s_state.attributes.get("forecast")
                     if not partial_forecast:
                         partial_forecast = s_state.attributes.get("detailedForecast", [])
                     
                     if partial_forecast:
                         solar_forecast_data.extend(partial_forecast)
                else:
                    _LOGGER.warning("Solar entity %s not found", entity_id)

        # 5. Prepare Simulation Data using Shared Handler
        now = dt_util.now()
        start_time = now
        model_timestep = self.config_entry.options.get(CONF_MODEL_TIMESTEP, DEFAULT_MODEL_TIMESTEP)
        
        timestamps, t_out_arr, solar_arr, dt_values = await self.input_handler.prepare_simulation_data(
            forecast if forecast else [],
            solar_forecast_data,
            start_time,
            duration_hours,
            model_timestep
        )
        
        if not timestamps:
            raise UpdateFailed("No forecast data available for simulation period")
        
        schedule_json = self.config_entry.options.get(CONF_SCHEDULE_CONFIG, DEFAULT_SCHEDULE_CONFIG)
        schedule_enabled = self.config_entry.options.get(CONF_SCHEDULE_ENABLED, DEFAULT_SCHEDULE_ENABLED)
        
        # Prepare arrays
        if not schedule_enabled:
             # Disabled: HVAC OFF (0), Setpoint None/0, Fixed False
             steps = len(timestamps)
             hvac_state_arr = np.zeros(steps)
             setpoint_arr = np.full(steps, None) # Or 0? Coordinator usually expects float. 
             # Let's use 0.0 and handle display in frontend, or None if typed array supports it.
             # numpy float array doesn't support None (NaN). 
             # For now use 0.0 and we'll see. Actually measurements.setpoint is used in run_model.
             # run_model needs valid float. If disabled, we probably want 0.0 or something that means "off semantics"
             # But if hvac_state is 0, setpoint is ignored for energy/action, but might be used for error calc?
             # Actually run_model uses setpoint just for control logic if it's deciding.
             # If hvac_state is provided (from measurements), run_model might just output physics.
             # Let's check run_model usage. It calculates heat/cool required based on setpoint if not provided?
             # Actually run_model takes hvac_state from measurements if we are just simulating what happened?
             # No, run_model is usually strictly thermodynamic response given inputs.
             # If we pass measurements, it might use them.
             
             # Safest: 0.0 or a really low/high number that won't trigger?
             # If HVAC state is 0, setpoint doesn't matter for energy calc.
             setpoint_arr = np.full(steps, np.nan) 
             fixed_mask_arr = np.zeros(steps, dtype=bool)
             
             # Log once per update?
             # _LOGGER.debug("Schedule disabled, skipping processing.")
             
        else:
             try:
                 schedule_data = json.loads(schedule_json)
             except Exception as e:
                 _LOGGER.error("Invalid Schedule JSON: %s", e)
                 raise ValueError(f"Invalid Schedule JSON: {e}")
     
             is_away, away_end, away_temp = self._get_away_status()
             
             # Get HVAC Mode from Options (Tunable setting)
             configured_mode = self.config_entry.options.get(CONF_HVAC_MODE, "heat")
             
             # Run process_schedule_data in executor to avoid blocking event loop (pytz I/O)
             from functools import partial
             hvac_state_arr, setpoint_arr, fixed_mask_arr = await self.hass.async_add_executor_job(
                 partial(
                     process_schedule_data,
                     timestamps, 
                     schedule_data, 
                     away_status=(is_away, away_end, away_temp),
                     timezone=self.hass.config.time_zone,
                     default_mode=configured_mode
                 )
             )

        steps = len(timestamps)
        t_in_arr = np.zeros(steps)
        t_in_arr[0] = current_temp

        measurements = Measurements(
            timestamps=np.array(timestamps),
            t_in=t_in_arr.astype(float),
            t_out=t_out_arr.astype(float),
            solar_kw=solar_arr.astype(float),
            hvac_state=np.array(hvac_state_arr, dtype=int),
            setpoint=np.array(setpoint_arr, dtype=float),
            dt_hours=np.array(dt_values, dtype=float),
            is_setpoint_fixed=np.array(fixed_mask_arr, dtype=bool),
            target_temp=np.array(setpoint_arr, dtype=float).copy()
        )
        
        return measurements, params, start_time


    async def async_set_away_mode(self, duration_delta: timedelta, safety_temp: float):
        """Set the away mode with a duration and safety temperature."""
        _LOGGER.info("Setting away mode. Duration: %s, Safety Temp: %s", duration_delta, safety_temp)
        
        # 1. Calculate End Time
        now = dt_util.now()
        away_end = now + duration_delta
        
        # 2. Persist to Config Entry (Options)
        # We need to update options to persist across restarts
        new_options = self.config_entry.options.copy()
        new_options["away_end"] = away_end.isoformat()
        new_options["away_temp"] = float(safety_temp)
        
        self.hass.config_entries.async_update_entry(self.config_entry, options=new_options)
        
        # 3. Schedule Smart Wake-Up (12 hours before return)
        # Cancel any existing timer
        if hasattr(self, "_away_timer_unsub") and self._away_timer_unsub:
            self._away_timer_unsub()
            self._away_timer_unsub = None
            
        wakeup_time = away_end - timedelta(hours=AWAY_WAKEUP_ADVANCE_HOURS)
        if wakeup_time > dt_util.now():
            from homeassistant.helpers.event import async_track_point_in_time
            
            async def _wake_up_callback(now):
                _LOGGER.info("Smart Wake-Up Triggered! Re-optimizing for return...")
                try:
                    await self.async_trigger_optimization()
                except Exception as e:
                    _LOGGER.error("Smart Wake-Up Optimization failed: %s", e)
                    
            _LOGGER.info("Scheduling Smart Wake-Up optimization for %s", wakeup_time)
            self._away_timer_unsub = async_track_point_in_time(self.hass, _wake_up_callback, wakeup_time)
        
        # 4. Trigger Immediate Optimization
        # This will use the new away settings (via _process_schedule checking config options)
        return await self.async_trigger_optimization()

    def _get_away_status(self):
        """Get current away status from config."""
        options = self.config_entry.options
        away_end_str = options.get("away_end")
        away_temp = options.get("away_temp", DEFAULT_AWAY_TEMP)
        
        if not away_end_str:
            return False, None, None
            
        try:
            away_end = dt_util.parse_datetime(away_end_str)
            if away_end.tzinfo is None: # handle legacy/missing TZ
                 away_end = away_end.replace(tzinfo=dt_util.get_time_zone(self.hass.config.time_zone))
                 
            if dt_util.now() < away_end:
                 return True, away_end, away_temp
            
            # Away expired - clean up stale entries
            new_options = {k: v for k, v in options.items() 
                          if k not in ("away_end", "away_temp")}
            self.hass.config_entries.async_update_entry(self.config_entry, options=new_options)
        except Exception as e:
            _LOGGER.warning("Error parsing away_end: %s", e)
            
        return False, None, None
