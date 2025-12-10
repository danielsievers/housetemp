"""DataUpdateCoordinator for House Temp Prediction."""
from datetime import timedelta, datetime
import json
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
    CONF_OPTIMIZATION_ENABLED,
    CONF_OPTIMIZATION_INTERVAL,
    DEFAULT_OPTIMIZATION_INTERVAL,
    CONF_MODEL_TIMESTEP,
    DEFAULT_MODEL_TIMESTEP,
    CONF_CONTROL_TIMESTEP,
    DEFAULT_CONTROL_TIMESTEP,
)

# Import from the installed package
from housetemp.run_model import run_model, HeatPump
from housetemp.measurements import Measurements
from housetemp.optimize import optimize_hvac_schedule

_LOGGER = logging.getLogger(__name__)

class HouseTempCoordinator(DataUpdateCoordinator):
    """Class to manage fetching data and running the model."""

    def __init__(self, hass: HomeAssistant, config_entry):
        """Initialize."""
        self.config_entry = config_entry
        self.hass = hass
        
        update_interval_min = config_entry.data.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL)
        
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=update_interval_min),
        )
        self.config_entry = config_entry

        self.heat_pump = None
        self._setup_heat_pump()
        
        # State for optimization
        self.last_optimization_time = None

    def _setup_heat_pump(self):
        """Initialize the HeatPump object from the config JSON."""
        hp_config_str = self.config_entry.data.get(CONF_HEAT_PUMP_CONFIG)
        if not hp_config_str:
            return

        # The HeatPump class expects a file path.
        # We will write the config to a file in the storage directory.
        # Using a stable path so we don't create infinite temp files.
        storage_dir = self.hass.config.path(".storage", DOMAIN)
        os.makedirs(storage_dir, exist_ok=True)
        hp_config_path = os.path.join(storage_dir, f"heat_pump_{self.config_entry.entry_id}.json")
        
        try:
            # Validate JSON first
            json.loads(hp_config_str)
            
            with open(hp_config_path, "w") as f:
                f.write(hp_config_str)
            
            self.heat_pump = HeatPump(hp_config_path)
        except Exception as e:
            _LOGGER.error("Failed to setup Heat Pump: %s", e)
            self.heat_pump = None

    async def _async_update_data(self):
        """Fetch data and run the model."""
        if not self.heat_pump:
            self._setup_heat_pump()
            if not self.heat_pump:
                raise UpdateFailed("Heat Pump not configured correctly")

        # 1. Get Inputs
        data = self.config_entry.data
        sensor_indoor = data.get(CONF_SENSOR_INDOOR_TEMP)
        weather_entity = data.get(CONF_WEATHER_ENTITY)
        solar_entity = data.get(CONF_SOLAR_ENTITY)
        duration_hours = data.get(CONF_FORECAST_DURATION, DEFAULT_FORECAST_DURATION)
        
        # Parameters
        params = [
            data.get(CONF_C_THERMAL),
            data.get(CONF_UA),
            data.get(CONF_K_SOLAR),
            data.get(CONF_Q_INT),
            data.get(CONF_H_FACTOR),
        ]

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
        
        # 3. Get Weather Forecast
        weather_state = self.hass.states.get(weather_entity)
        if not weather_state:
            raise UpdateFailed(f"Weather entity {weather_entity} not found")
        
        # Try Attribute first (Legacy)
        forecast = weather_state.attributes.get("forecast")
        
        # Try modern service call if attribute missing
        if forecast is None:
             try:
                 # Support both 'hourly' and 'daily' - prefer hourly for simulation
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
             _LOGGER.warning("No forecast data found for %s (tried attribute and service calls)", weather_entity)
             forecast = []

        # 4. Get Solar Forecast
        # Expecting sensors that have a 'forecast' attribute (like Forecast.Solar or Solcast)
        # Config might be a single string or a list of strings (if multiple=True)
        solar_forecast_data = []
        
        solar_entities = solar_entity
        if solar_entities:
            # Normalize to list
            if isinstance(solar_entities, str):
                solar_entities = [solar_entities]
            
            for entity_id in solar_entities:
                s_state = self.hass.states.get(entity_id)
                if s_state:
                     # Get forecast (list of dicts)
                     # Standard is 'forecast', Solcast uses 'detailedForecast'
                     partial_forecast = s_state.attributes.get("forecast")
                     if not partial_forecast:
                         partial_forecast = s_state.attributes.get("detailedForecast", [])
                     
                     if partial_forecast:
                         solar_forecast_data.extend(partial_forecast)
                else:
                    _LOGGER.warning("Solar entity %s not found", entity_id)

        # 5. Prepare Simulation Data using Shared Upsampling Logic
        # Construct a DataFrame with the sparse forecast points
        import pandas as pd
        from housetemp.utils import upsample_dataframe
        
        # Helper to parse forecast points into a list of dicts
        def parse_points(forecast_list, dt_key_opts, val_key_opts):
            pts = []
            for item in forecast_list:
                dt_str = next((item.get(k) for k in dt_key_opts if item.get(k)), None)
                key_found = next((k for k in val_key_opts if item.get(k) is not None), None)
                val = item.get(key_found) if key_found else None
                
                if dt_str and val is not None:
                    dt = dt_util.parse_datetime(dt_str)
                    if dt:
                        val_float = float(val)
                        # Auto-detect units
                        if key_found in ['watts', 'wh_hours']: 
                             # Forecast.Solar often uses 'watts' (power) or 'wh_hours' (energy/hour -> power)
                             # Convert W to kW
                             val_float /= 1000.0
                        elif key_found == 'pv_estimate':
                             # Solcast usually returns kW, but let's sanity check
                             # If value > 100, it's probably Watts (unless user has 100kW+ array)
                             if val_float > 50: 
                                 val_float /= 1000.0
                        
                        pts.append({'time': dt, 'value': val_float})
            return pts

        # Parse Weather
        weather_pts = parse_points(forecast, ['datetime'], ['temperature'])
        if not weather_pts:
             # Fallback if no forecast: current timestamp + current outdoor
             weather_pts = [{'time': dt_util.now(), 'value': 50.0}] 

        # Parse Solar
        # Support common keys: 
        # 'pv_estimate' (Solcast, kW/W)
        # 'watts' (Forecast.Solar, W)
        # 'wh_hours' (Forecast.Solar, Wh -> W if 1h block)
        # 'period_start' (Solcast detailedForecast)
        solar_pts = parse_points(solar_forecast_data, 
                                 ['datetime', 'period_end', 'period_start'], 
                                 ['pv_estimate', 'watts', 'wh_hours', 'value'])
        if not solar_pts:
             solar_pts = [{'time': dt_util.now(), 'value': 0.0}]
             
        # Create DataFrames
        df_w = pd.DataFrame(weather_pts).set_index('time').rename(columns={'value': 'outdoor_temp'})
        df_s = pd.DataFrame(solar_pts).set_index('time').rename(columns={'value': 'solar_kw'})
        
        # Merge (Outer Join to keep all points)
        df_raw = df_w.join(df_s, how='outer').sort_index()
        
        # Ensure we cover the full duration requsted
        now = dt_util.now()
        start_time = now
        end_time = start_time + timedelta(hours=duration_hours)
        
        # Clip/Extend DataFrame index to cover start_time to end_time
        # We add dummy rows at start and end if missing, so interpolation covers the range
        if df_raw.index.min() > start_time:
             # Prepend start row (use existing first values or defaults)
             # Better: concatenation
             row = pd.DataFrame({'outdoor_temp': [df_raw['outdoor_temp'].iloc[0]], 'solar_kw': [0]}, index=[start_time])
             df_raw = pd.concat([row, df_raw])
             
        if df_raw.index.max() < end_time:
             row = pd.DataFrame({'outdoor_temp': [df_raw['outdoor_temp'].iloc[-1]], 'solar_kw': [0]}, index=[end_time])
             df_raw = pd.concat([df_raw, row])
             
        # Reset index for upsampling utility
        df_raw = df_raw.reset_index().rename(columns={'index': 'time'})
        
        # Get Configured Timesteps
        model_timestep = self.config_entry.options.get(CONF_MODEL_TIMESTEP, DEFAULT_MODEL_TIMESTEP)
        control_timestep = self.config_entry.options.get(CONF_CONTROL_TIMESTEP, DEFAULT_CONTROL_TIMESTEP)
        
        # Upsample to configured resolution
        freq_str = f"{model_timestep}min"
        
        df_sim = upsample_dataframe(
            df_raw, 
            freq=freq_str, 
            cols_linear=['outdoor_temp', 'solar_kw'],
            cols_ffill=[] 
        )
        
        # Filter to exactly the duration requested (from start_time)
        df_sim = df_sim[(df_sim['time'] >= start_time) & (df_sim['time'] <= end_time)]
        
        # If empty after slice (shouldn't happen with logic above), fail safe
        if df_sim.empty:
            _LOGGER.warning("Simulation DataFrame empty after slice! Using fallback.")
            # ... fallback logic logic or minimal DF

        # Extract Arrays
        timestamps = df_sim['time'].tolist() # pydatetime objects
        t_out_arr = df_sim['outdoor_temp'].fillna(method='ffill').fillna(50).values
        solar_arr = df_sim['solar_kw'].fillna(0).values
        dt_values = df_sim['dt'].values # Should be ~0.25 (15 min) or whatever configured
        
        # Create Result Arrays
        steps = len(timestamps)
        
        # Schedule Processing
        schedule_json = data.get(CONF_SCHEDULE_CONFIG)
        hvac_state_arr, setpoint_arr = self._process_schedule(timestamps, schedule_json)

        # Indoor Temp Array
        t_in_arr = np.zeros(steps)
        t_in_arr[0] = current_temp # Start temp

        measurements = Measurements(
            timestamps=np.array(timestamps),
            t_in=t_in_arr,
            t_out=t_out_arr,
            solar_kw=solar_arr,
            hvac_state=np.array(hvac_state_arr),
            setpoint=np.array(setpoint_arr),
            dt_hours=dt_values
        )

        # 6. Run Optimization (Optional & Throttled)
        opt_enabled = self.config_entry.options.get(CONF_OPTIMIZATION_ENABLED, False)
        opt_interval = self.config_entry.options.get(CONF_OPTIMIZATION_INTERVAL, DEFAULT_OPTIMIZATION_INTERVAL)
        
        run_opt = False
        if opt_enabled:
            now_ts = now.timestamp()
            if self.last_optimization_time is None:
                run_opt = True
            else:
                elapsed_min = (now_ts - self.last_optimization_time) / 60.0
                if elapsed_min >= opt_interval:
                    run_opt = True
                    
        if run_opt:
            _LOGGER.info(f"Running HVAC Optimization (Model: {model_timestep}m, Control: {control_timestep}m)...")
            target_temps = setpoint_arr.copy()
            comfort_config = {"mode": "heat", "center_preference": 0.5}
            
            try:
                optimized_setpoints = await self.hass.async_add_executor_job(
                    optimize_hvac_schedule,
                    measurements,
                    params,
                    self.heat_pump,
                    target_temps,
                    comfort_config,
                    control_timestep # block_size_minutes
                )
                
                measurements.setpoint = optimized_setpoints
                setpoint_arr = optimized_setpoints
                # hvac_state_arr = measurements.hvac_state # Optimization updates this too
                
                self.last_optimization_time = now.timestamp()
                
            except Exception as e:
                _LOGGER.error("Optimization failed: %s", e)
                import traceback
                _LOGGER.error(traceback.format_exc())

        # 7. Run Model
        sim_temps, _, _ = run_model(params, measurements, self.heat_pump, duration_minutes=duration_hours*60)

        # 7. Return Result
        return {
            "timestamps": timestamps,
            "predicted_temp": sim_temps,
            "hvac_state": measurements.hvac_state, # Use measuring array as it might change
            "setpoint": setpoint_arr,
            "solar": solar_arr,
            "outdoor": t_out_arr
        }

    def _get_interpolated_weather(self, timestamps, forecast, default_val=50.0):
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
                # Try common keys
                dt_str = item.get('datetime') or item.get('period_end')
                val = item.get('value') or item.get('pv_estimate')
                
                if dt_str and val is not None:
                    dt = dt_util.parse_datetime(dt_str)
                    if dt:
                        # Convert Watts to kW if needed? 
                        # Assuming value is kW if it says 'solar_kw' in our model.
                        # Forecast.Solar uses Watts usually? 'wh_watts'?
                        # Let's assume the user provides a sensor that gives kW or we might be off by 1000x.
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
        # Schedule can be old list or new nested dict
        try:
            schedule_data = json.loads(schedule_json)
        except:
            _LOGGER.error("Invalid Schedule JSON")
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
                    setpoint_arr.append(70.0)
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
                     setpoint = float(active_item.get('setpoint', 70))
                
                state_val = 0
                if mode == 'heat':
                    state_val = 1
                elif mode == 'cool':
                    state_val = -1
                
                hvac_arr.append(state_val)
                setpoint_arr.append(setpoint)

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
