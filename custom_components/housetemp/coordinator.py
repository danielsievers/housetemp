"""DataUpdateCoordinator for House Temp Prediction."""
from datetime import timedelta, datetime
import json
import time
import logging
import os
import tempfile

import numpy as np

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import (
    DataUpdateCoordinator,
    UpdateFailed,
)
from homeassistant.util import dt as dt_util

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
    DEFAULT_UPDATE_INTERVAL,
    CONF_MODEL_TIMESTEP,
    DEFAULT_MODEL_TIMESTEP,
    CONF_CONTROL_TIMESTEP,
    DEFAULT_CONTROL_TIMESTEP,
    DEFAULT_AWAY_TEMP,

    AWAY_WAKEUP_ADVANCE_HOURS,
)

# Import from the installed package
from .housetemp.run_model import run_model, HeatPump
from .housetemp.measurements import Measurements
from .housetemp.optimize import optimize_hvac_schedule
from .housetemp.optimize import optimize_hvac_schedule
from .housetemp.energy import estimate_consumption
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

    async def _setup_heat_pump(self):
        """Initialize the HeatPump object from the config JSON."""
        # Heat Pump Config is FIXED (Data)
        hp_config_str = self.config_entry.data.get(CONF_HEAT_PUMP_CONFIG)
        if not hp_config_str:
            return

        # Run file I/O in executor
        def save_and_init():
            # Storage dir
            storage_dir = self.hass.config.path(".storage", DOMAIN)
            os.makedirs(storage_dir, exist_ok=True)
            hp_config_path = os.path.join(storage_dir, f"heat_pump_{self.config_entry.entry_id}.json")
            
            # Validate JSON first
            try:
                data = json.loads(hp_config_str)
                # Basic validation - check for expected keys from heat_pump.json format
                if "cop" not in data or "max_capacity" not in data:
                     _LOGGER.warning("Heat Pump Config might be missing required keys (cop, max_capacity)")
            except Exception as e:
                raise ValueError(f"Invalid Heat Pump JSON: {e}")
            
            with open(hp_config_path, "w") as f:
                f.write(hp_config_str)
            
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

        if not self.heat_pump:
            await self._setup_heat_pump()
            if not self.heat_pump:
                raise UpdateFailed("Heat Pump not configured correctly")

        # 1. Get Inputs & Prepare Simulation Data
        try:
            measurements, params, start_time = await self._prepare_simulation_inputs()
        except Exception as e:
            _LOGGER.error("Error preparing simulation inputs: %s", e)
            raise UpdateFailed(f"Error preparing simulation inputs: {e}")

        # Unpack for local usage if needed, or just use measurements object
        timestamps = measurements.timestamps
        setpoint_arr = measurements.setpoint
        t_out_arr = measurements.t_out
        solar_arr = measurements.solar_kw
        
        duration_hours = self.config_entry.options.get(CONF_FORECAST_DURATION, DEFAULT_FORECAST_DURATION)

        # 6. Apply Cached Optimization Results (if available)
        # We do NOT run optimization here. We strictly lookup the cache.
        
        # optimized_setpoints_map is {timestamp (float): setpoint (float)}
        # We need to build an array aligned with 'timestamps'
        optimized_setpoint_arr = []
        has_optimized_data = False
        
        if self.optimized_setpoints_map:
            # We must be careful with floating point timestamps.
            # Let's assume exact match since they come from the same generation process,
            # BUT if the main loop has slightly different timestamps (due to weather update time shifting),
            # we need to likely use nearest neighbour or just map by ISO string if possible.
            # However, usually simulation starts at 'now' upsampled.
            
            # Robust approach: Find value for each timestamp if it exists in map
            # We can use a small tolerance or just round to minute.
            
            # Better: When we run optimization, we store result keyed by int(timestamp) (seconds)
            # Here we lookup by int(timestamp).
            
            # Let's try to populate
            found_count = 0
            for ts in timestamps:
                ts_int = int(ts.timestamp())
                # Try exact match first
                val = self.optimized_setpoints_map.get(ts_int)
                
                # If not found, maybe try +- 30 seconds? 
                # (Upsampling is usually aligned to model_timestep minutes)
                if val is None:
                     # fallback logic if needed, but for now simple lookup
                     pass
                
                if val is not None:
                    optimized_setpoint_arr.append(val)
                    found_count += 1
                else:
                    optimized_setpoint_arr.append(None)
            
            if found_count > 0:
                has_optimized_data = True
                _LOGGER.debug(f"Found {found_count} cached optimized setpoints for current window.")

        # 7. Run Model (in executor to avoid blocking)
        # Use the optimized setpoints for simulation IF we have them, otherwise use schedule
        # The 'measurements' object currently has 'setpoint' from schedule.
        # If we have optimized setpoints, we should use them for the simulation to show the prediction *if* the plan were followed.
        
        if has_optimized_data:
            # fill Nones with schedule as fallback for simulation
            sim_setpoints = []
            for i, opt_val in enumerate(optimized_setpoint_arr):
                if opt_val is not None:
                    sim_setpoints.append(opt_val)
                else:
                    sim_setpoints.append(setpoint_arr[i])
            measurements.setpoint = np.array(sim_setpoints)
        
        sim_temps, _, _ = await self.hass.async_add_executor_job(
            run_model, params, measurements, self.heat_pump, duration_hours*60
        )
        
        if len(sim_temps) > 0:
            _LOGGER.info("Simulation complete. Predicted Final Temp: %.1f", sim_temps[-1])

        # 8. Return Result
        result = {
            "timestamps": timestamps,
            "predicted_temp": sim_temps,
            "hvac_state": measurements.hvac_state,
            "setpoint": setpoint_arr, # Return original schedule for comparison
            "solar": solar_arr,
            "outdoor": t_out_arr
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
        
        if has_optimized_data:
            result["optimized_setpoint"] = optimized_setpoint_arr
        
        return result

    async def async_trigger_optimization(self, duration_hours=None):
        """Manually trigger the HVAC optimization process."""
        _LOGGER.info("Manual optimization triggered. Duration override: %s", duration_hours)
        
        # 1. Ensure heat pump is ready
        if not self.heat_pump:
            await self._setup_heat_pump()
        
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
        user_pref = self.config_entry.options.get("center_preference", 1.0)
        comfort_config = {"mode": "heat", "center_preference": float(user_pref)}
        
        try:
            optimized_setpoints = await self.hass.async_add_executor_job(
                optimize_hvac_schedule,
                measurements,
                params,
                self.heat_pump,
                target_temps,
                comfort_config,
                control_timestep
            )
            
            opt_duration = time.time() - opt_start_time
            _LOGGER.info("Optimization completed in %.2f seconds", opt_duration)
            
            # Map optimized setpoints to timestamps
            timestamps = measurements.timestamps
            new_cache = {}
            if len(optimized_setpoints) == len(timestamps):
                 for i, ts in enumerate(timestamps):
                     ts_int = int(ts.timestamp())
                     new_cache[ts_int] = optimized_setpoints[i]
            
            self.optimized_setpoints_map.update(new_cache)
            
            await self.async_request_refresh()
            
            # --- Run Simulation for Forecast (Predicted Temp) ---
            # We want to return the predicted temperature path based on the NEW optimized setpoints.
            # We reuse the `measurements` object but override the setpoints.
            
            # Create a clean array for simulation inputs
            sim_setpoints = []
            for i, ts in enumerate(timestamps):
                # Use optimized value if available (it should be for this window)
                if i < len(optimized_setpoints):
                     sim_setpoints.append(optimized_setpoints[i])
                else:
                     sim_setpoints.append(measurements.setpoint[i])
            
            measurements.setpoint = np.array(sim_setpoints)
            
            # Ensure duration is valid for simulation
            sim_duration_hours = duration_hours
            if sim_duration_hours is None:
                 sim_duration_hours = self.config_entry.options.get(CONF_FORECAST_DURATION, DEFAULT_FORECAST_DURATION)
                 
            # Run Model for Temp Curve
            _LOGGER.info("Running simulation for service response (duration: %.1f h)...", sim_duration_hours)
            sim_temps, _, _ = await self.hass.async_add_executor_job(
                run_model, params, measurements, self.heat_pump, sim_duration_hours*60
            )
            
            # -- Optimized Energy Calculation --
            # Now measurements.setpoint is OPTIMIZED.
            optimized_kwh = 0.0
            try:
                 # Note: estimate_consumption will run simulation again internally.
                 opt_res = await self.hass.async_add_executor_job(
                     estimate_consumption, measurements, params, self.heat_pump
                 )
                 optimized_kwh = opt_res.get('total_kwh', 0.0)
            except Exception as e:
                 _LOGGER.warning("Failed to estimate optimized energy: %s", e)
            
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
                    "predicted_temp": float(sim_temps[i]) if i < len(sim_temps) else None
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
        
        # Parameters (Physics) - From Options
        params = [
            options.get(CONF_C_THERMAL, 10000.0),
            options.get(CONF_UA, 750.0),
            options.get(CONF_K_SOLAR, 3000.0),
            options.get(CONF_Q_INT, 2000.0),
            options.get(CONF_H_FACTOR, 5000.0),
        ]
        
        _LOGGER.debug("Preparing simulation inputs with params: %s", params)

        # 2. Get Current State
        indoor_state = self.hass.states.get(sensor_indoor)
        if not indoor_state or indoor_state.state in ("unknown", "unavailable"):
            raise UpdateFailed(f"Indoor sensor {sensor_indoor} unavailable")
        
        try:
            current_temp = float(indoor_state.state)
        except ValueError:
            raise UpdateFailed(f"Invalid indoor temp: {indoor_state.state}")

        # 3. Get Weather Forecast
        weather_state = self.hass.states.get(weather_entity)
        if not weather_state:
            raise UpdateFailed(f"Weather entity {weather_entity} not found")
        
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
        
        steps = len(timestamps)
        
        schedule_json = self.config_entry.options.get(CONF_SCHEDULE_CONFIG, "[]")
        hvac_state_arr, setpoint_arr = self._process_schedule(timestamps, schedule_json)

        t_in_arr = np.zeros(steps)
        t_in_arr[0] = current_temp

        measurements = Measurements(
            timestamps=np.array(timestamps),
            t_in=t_in_arr,
            t_out=t_out_arr,
            solar_kw=solar_arr,
            hvac_state=np.array(hvac_state_arr),
            setpoint=np.array(setpoint_arr),
             dt_hours=dt_values
        )
        
        return measurements, params, start_time

    def _process_schedule(self, timestamps, schedule_json):
        """Generate HVAC state and setpoint arrays from schedule."""
        _LOGGER.debug("Processing schedule for %d timestamps", len(timestamps))
        from datetime import time as datetime_time
        from datetime import datetime
        
        # Check for Away Mode
        is_away, away_end, away_temp = self._get_away_status()
        
        try:
            schedule_data = json.loads(schedule_json)
        except Exception as e:
            _LOGGER.error("Invalid Schedule JSON: %s", e)
            raise ValueError(f"Invalid Schedule JSON: {e}")

        if not isinstance(schedule_data, dict) or "schedule" not in schedule_data:
            _LOGGER.error("Schedule must be a dictionary with 'schedule' key. Got: %s", schedule_json)
            raise ValueError("Schedule must be a dictionary with 'schedule' key")

        # Helper to find daily schedule for a given date
        def get_daily_schedule(date_obj):
            day_name = date_obj.strftime("%A").lower()
            for rule in schedule_data.get("schedule", []):
                if day_name in [d.lower() for d in rule.get("weekdays", [])]:
                    return rule.get("daily_schedule", [])
            return []

        hvac_arr = []
        setpoint_arr = []

        for ts in timestamps:
            daily_items = get_daily_schedule(ts)
            
            if not daily_items:
                raise ValueError(f"No schedule coverage for timestamp {ts}")
            
            # Use local time for schedule lookup
            ts_local = dt_util.as_local(ts)
            current_time = ts_local.time()
            
            # Sort items by time (defensive, though validation enforces it)
            # Use datetime.time for robust comparison
            try:
                # Convert string times to datetime.time objects once if optimization needed,
                # but for simplicity we do it here. validation ensures "HH:MM" format.
                sorted_items = sorted(
                    daily_items, 
                    key=lambda x: datetime.strptime(x['time'], "%H:%M").time() if isinstance(x['time'], str) else x['time']
                )
            except ValueError:
                 _LOGGER.error("Invalid time format in schedule for timestamp %s", ts)
                 raise ValueError(f"Invalid time format in schedule for timestamp {ts}")

            # Find active rule: last time less than or equal to current time
            # Validation ensures daily_schedule is not empty.
            active_item = sorted_items[-1] # Default to last item (wrap around previous day effectively)
            
            for item in sorted_items:
                # Parse item time
                t_str = item['time']
                t_obj = datetime.strptime(t_str, "%H:%M").time()
                
                if t_obj <= current_time:
                    active_item = item
                else:
                    break
            
            # Extract setpoint and mode
            val = active_item.get("temp")
            if val is None:
                 raise ValueError(f"Missing 'temp' in schedule item: {active_item}")
            setpoint = float(val)
            
            global_mode = schedule_data.get("mode", "heat").lower()
            
            # Map logic to integers
            state_val = 0
            if global_mode == 'heat':
                state_val = 1
            elif global_mode == 'cool':
                state_val = -1
            
            hvac_arr.append(state_val)
            
            # Apply Away Override
            final_setpoint = setpoint
            if is_away and ts < away_end:
                 final_setpoint = away_temp
            
            setpoint_arr.append(final_setpoint)

        return np.array(hvac_arr), np.array(setpoint_arr)

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
        await self.async_trigger_optimization()

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
        except Exception as e:
            _LOGGER.warning("Error parsing away_end: %s", e)
            
        return False, None, None
