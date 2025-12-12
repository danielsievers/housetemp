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
    DEFAULT_FALLBACK_SETPOINT,
    DEFAULT_OUTDOOR_TEMP_FALLBACK,
    AWAY_WAKEUP_ADVANCE_HOURS,
)

# Import from the installed package
from .housetemp.run_model import run_model, HeatPump
from .housetemp.measurements import Measurements
from .housetemp.optimize import optimize_hvac_schedule
from .housetemp.energy import estimate_consumption

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

    async def _async_update_data(self):
        """Fetch data and run the model."""
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
        # This is a refactored extraction of lines 116-364 from original _async_update_data
        # For the sake of this tool use, I will assume I need to implement this helper 
        # and replace the logic in _async_update_data to use it too.
        
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

        # 5. Prepare Simulation Data using Shared Upsampling Logic
        import pandas as pd
        from .housetemp.utils import upsample_dataframe
        
        def parse_points(forecast_list, dt_key_opts, val_key_opts):
            pts = []
            for item in forecast_list:
                dt_val = next((item.get(k) for k in dt_key_opts if item.get(k)), None)
                key_found = next((k for k in val_key_opts if item.get(k) is not None), None)
                val = item.get(key_found) if key_found else None
                
                if dt_val and val is not None:
                    if isinstance(dt_val, str):
                        dt = dt_util.parse_datetime(dt_val)
                    else:
                        dt = dt_val
                    
                    if dt:
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=dt_util.get_time_zone(self.hass.config.time_zone))
                        
                        val_float = float(val)
                        if key_found in ['watts', 'wh_hours']: 
                             val_float /= 1000.0
                        elif key_found == 'pv_estimate':
                             if val_float > 50: 
                                 val_float /= 1000.0
                        
                        pts.append({'time': dt, 'value': val_float})
            return pts

        weather_pts = parse_points(forecast, ['datetime'], ['temperature'])
        if not weather_pts:
             raise UpdateFailed(f"No forecast data available from {weather_entity}") 

        solar_pts = parse_points(solar_forecast_data, 
                                 ['datetime', 'period_end', 'period_start'], 
                                 ['pv_estimate', 'watts', 'wh_hours', 'value'])
        if not solar_pts:
             now = dt_util.now()
             if now.tzinfo is None:
                 now = now.replace(tzinfo=dt_util.get_time_zone(self.hass.config.time_zone))
             solar_pts = [{'time': now, 'value': 0.0}]
             
        now = dt_util.now()
        start_time = now
        end_time = start_time + timedelta(hours=duration_hours)
        model_timestep = self.config_entry.options.get(CONF_MODEL_TIMESTEP, DEFAULT_MODEL_TIMESTEP)
        
        def prepare_simulation_data():
            df_w = pd.DataFrame(weather_pts).set_index('time').rename(columns={'value': 'outdoor_temp'})
            df_s = pd.DataFrame(solar_pts).set_index('time').rename(columns={'value': 'solar_kw'})
            
            df_raw = df_w.join(df_s, how='outer').sort_index()
            
            if df_raw.index.min() > start_time:
                row = pd.DataFrame({'outdoor_temp': [df_raw['outdoor_temp'].iloc[0]], 'solar_kw': [0]}, index=[start_time])
                df_raw = pd.concat([row, df_raw])
                 
            if df_raw.index.max() < end_time:
                row = pd.DataFrame({'outdoor_temp': [df_raw['outdoor_temp'].iloc[-1]], 'solar_kw': [0]}, index=[end_time])
                df_raw = pd.concat([df_raw, row])
                 
            df_raw = df_raw.reset_index().rename(columns={'index': 'time'})
            
            freq_str = f"{model_timestep}min"
            df_sim = upsample_dataframe(
                df_raw, 
                freq=freq_str, 
                cols_linear=['outdoor_temp', 'solar_kw'],
                cols_ffill=[] 
            )
            
            df_sim = df_sim[(df_sim['time'] >= start_time) & (df_sim['time'] <= end_time)]
            
            timestamps = df_sim['time'].tolist()
            t_out_arr = df_sim['outdoor_temp'].ffill().fillna(50).values
            solar_arr = df_sim['solar_kw'].fillna(0).values
            dt_values = df_sim['dt'].values
            
            return timestamps, t_out_arr, solar_arr, dt_values
        
        timestamps, t_out_arr, solar_arr, dt_values = await self.hass.async_add_executor_job(
            prepare_simulation_data
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

    # ... (Scheduling helper methods) ...

    def _get_interpolated_weather(self, timestamps, forecast, default_val=DEFAULT_OUTDOOR_TEMP_FALLBACK):
        """Interpolate weather forecast to match timestamps."""
        # Simple nearest neighbor or linear interpolation
        # Forecast structure: [{'datetime': '...', 'temperature': 20}, ...]
        # Map forecast to a time-value list
        points = []
        for item in forecast:
            try:
                dt_str = item.get('datetime')
                val = item.get('temperature')
                if dt_str and val is not None:
                    dt = dt_util.parse_datetime(dt_str)
                    if dt:
                        # Enforce Timezone Awareness
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=dt_util.get_time_zone(self.hass.config.time_zone))
                        points.append((dt.timestamp(), float(val)))
            except:
                continue
        
        if not points:
            return [default_val] * len(timestamps)
            
        points.sort(key=lambda x: x[0])
        xp = [p[0] for p in points]
        fp = [p[1] for p in points]
        
        target_ts = [t.timestamp() for t in timestamps]
        return np.interp(target_ts, xp, fp)

    def _get_interpolated_solar(self, timestamps, forecast):
        """Interpolate solar forecast."""
        # Forecast structure: [{'period_end': '...', 'pv_estimate': 2.0}, ...] (Forecast.Solar)
        # Or generic: {'datetime': ..., 'value': ...}
        # We need to handle generic case or specific.
        # Let's assume keys 'datetime'/'period_end' and 'value'/'pv_estimate'/'wh_watts'
        
        points = []
        for item in forecast:
            try:
                # Try common keys (including period_start for Solcast)
                dt_val = item.get('datetime') or item.get('period_end') or item.get('period_start')
                val = item.get('value') or item.get('pv_estimate')
                
                if dt_val and val is not None:
                    # Handle both string and datetime inputs (Solcast uses datetime objects)
                    if isinstance(dt_val, str):
                        dt = dt_util.parse_datetime(dt_val)
                    else:
                        # Already a datetime object
                        dt = dt_val
                    
                    if dt:
                        # Enforce Timezone Awareness
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=dt_util.get_time_zone(self.hass.config.time_zone))
                        
                        # For now, take raw value.
                        points.append((dt.timestamp(), float(val)))
            except:
                continue

        if not points:
            return [0.0] * len(timestamps)

        points.sort(key=lambda x: x[0])
        xp = [p[0] for p in points]
        fp = [p[1] for p in points]
        
        target_ts = [t.timestamp() for t in timestamps]
        return np.interp(target_ts, xp, fp)

    def _process_schedule(self, timestamps, schedule_json):
        """Generate HVAC state and setpoint arrays from schedule."""
        _LOGGER.debug("Processing schedule for %d timestamps", len(timestamps))
        
        # Check for Away Mode
        is_away, away_end, away_temp = self._get_away_status()
        
        # Schedule can be old list or new nested dict
        try:
            schedule_data = json.loads(schedule_json)
            # Basic Schema Validation
            if isinstance(schedule_data, dict):
                 if "schedule" not in schedule_data or not isinstance(schedule_data["schedule"], list):
                      _LOGGER.error("Schedule JSON missing 'schedule' list")
                      return np.zeros(len(timestamps)), np.full(len(timestamps), 70.0)
        except Exception as e:
            _LOGGER.error("Invalid Schedule JSON: %s", e)
            return np.zeros(len(timestamps)), np.full(len(timestamps), 70.0)

        if not schedule_data:
             return np.zeros(len(timestamps)), np.full(len(timestamps), 70.0)

        # Handle New format: {"schedule": [...]}
        if isinstance(schedule_data, dict) and "schedule" in schedule_data:
            # We need to flatten this based on weekdays?
            # For simplicity in this coordinator, let's assume we just want to look up the schedule for the current day.
            # But the simulation spans multiple hours (maybe crossing days).
            # We need a helper to look up properties.
            # Let's re-use the logic or adapt it.
            
            # Extract the relevant daily schedule for each timestamp
            hvac_arr = []
            setpoint_arr = []
            
            # Helper to find daily schedule for a given date
            def get_daily_schedule(date_obj):
                day_name = date_obj.strftime("%A").lower()
                for rule in schedule_data.get("schedule", []):
                    if day_name in [d.lower() for d in rule.get("weekdays", [])]:
                        return rule.get("daily_schedule", [])
                return []

            for ts in timestamps:
                daily_items = get_daily_schedule(ts)
                if not daily_items:
                    # Fallback
                    hvac_arr.append(0)
                    setpoint_arr.append(DEFAULT_FALLBACK_SETPOINT)
                    continue
                    
                # Find item in daily_items
                # They are sorted by time? They should be.
                # daily_items = [{"time": "00:00", ...}, ...]
                
                # CRITICAL: The timestamp 'ts' is likely UTC (from dt_util.now() or forecast).
                # The schedule "07:00" is meant for Local Time.
                # We must convert 'ts' to local time before comparing.
                ts_local = dt_util.as_local(ts)
                current_time_str = ts_local.strftime("%H:%M")
                
                # Sort just in case
                daily_items = sorted(daily_items, key=lambda x: x['time'])
                
                active_item = daily_items[-1] # Default to last of prev day? OR last item
                
                for item in daily_items:
                    if item['time'] <= current_time_str:
                        active_item = item
                    else:
                        break
                
                # mode logic
                if "temp" in active_item:
                     # New format uses "temp" which is setpoint
                     setpoint = float(active_item["temp"])
                     # Mode is global in new format? Or implied?
                     # The new format comfort.json has "mode": "heat" at top level.
                     # We should use that.
                     global_mode = schedule_data.get("mode", "heat").lower()
                     mode = global_mode
                else:
                     # Old format
                     mode = active_item.get('mode', 'off').lower()
                     setpoint = float(active_item.get('setpoint', DEFAULT_FALLBACK_SETPOINT))
                
                state_val = 0
                if mode == 'heat':
                    state_val = 1
                elif mode == 'cool':
                    state_val = -1
                
                hvac_arr.append(state_val)
                
                # Apply Away Override
                final_setpoint = setpoint
                if is_away and ts < away_end:
                     final_setpoint = away_temp
                     # Force mode to heat if away? 
                     # Usually away mode implies heating/cooling to safety. 
                     # But current implementation just overrides setpoint.
                     # The optimization will then see a low target and likely keep hvac off.
                
                setpoint_arr.append(final_setpoint)

            return np.array(hvac_arr), np.array(setpoint_arr)

        # Legacy List Format handling (keep for backward compat if needed, or just fail)
        if isinstance(schedule_data, list):
             schedule = schedule_data
             # ... (Original logic for list) ...
             # We can copy paste the original loop here or just deprecate list.
             # Let's keep it simple and assume list format is dead or handled by above if we enforce dict.
             pass

        # If we got here and it's a list, run original logic:
        schedule = schedule_data
        hvac_arr = []
        setpoint_arr = []
        
        for ts in timestamps:
            current_time_str = ts.strftime("%H:%M")
            schedule.sort(key=lambda x: x['time'])
            active_item = schedule[-1]
            for item in schedule:
                if item['time'] <= current_time_str:
                    active_item = item
                else:
                    break
            
            mode = active_item.get('mode', 'off').lower()
            setpoint = float(active_item.get('setpoint', 70))
            
            state_val = 0
            if mode == 'heat':
                state_val = 1
            elif mode == 'cool':
                state_val = -1
            
            hvac_arr.append(state_val)
            setpoint_arr.append(setpoint)

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
