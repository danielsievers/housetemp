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
    DEFAULT_FORECAST_DURATION,
    DEFAULT_UPDATE_INTERVAL,
    CONF_OPTIMIZATION_ENABLED,
    CONF_OPTIMIZATION_INTERVAL,
    DEFAULT_OPTIMIZATION_INTERVAL,
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
        
        forecast = weather_state.attributes.get("forecast")
        if not forecast:
             # Try to get forecast from the new service response approach if attribute is missing?
             # For now, assume attribute exists (legacy/standard way for many).
             # If using the new `weather.get_forecast` service, we'd need to call it.
             # But `weather` entities usually still have attributes or we can't easily await service calls here without more complex logic.
             # Let's assume attribute for now or `forecast` property.
             pass

        if not forecast:
             # Fallback or error
             # Some integrations don't have forecast attribute anymore.
             # We might need to handle this. But let's proceed assuming it exists or user selected a compatible one.
             _LOGGER.warning("No forecast attribute found on %s", weather_entity)
             # Create dummy forecast for testing?
             forecast = []

        # 4. Get Solar Forecast
        # Expecting a sensor that has a 'forecast' attribute (like Forecast.Solar)
        solar_forecast_data = []
        if solar_entity:
            solar_state = self.hass.states.get(solar_entity)
            if solar_state:
                solar_forecast_data = solar_state.attributes.get("forecast", [])

        # 5. Prepare Simulation Data
        # We need to generate arrays for the duration
        # Time step = 30 mins (0.5 hours)
        dt_hours = 0.5
        steps = int(duration_hours / dt_hours)
        
        now = dt_util.now()
        timestamps = [now + timedelta(minutes=30 * i) for i in range(steps)]
        
        # Interpolate Outdoor Temp
        t_out_arr = self._get_interpolated_weather(timestamps, forecast, current_temp) # Use current temp? No, current outdoor.
        # We need current outdoor temp for the first step?
        # The weather entity state is usually current condition.
        try:
            current_outdoor = float(weather_state.state)
        except ValueError:
            current_outdoor = t_out_arr[0] if len(t_out_arr) > 0 else 50.0 # Fallback

        # Fix: t_out_arr should start with current outdoor?
        # The model simulation loop:
        # q_leak = UA * (data.t_out[i] - current_temp)
        # So t_out[i] is the outdoor temp at step i.
        
        # Interpolate Solar
        solar_arr = self._get_interpolated_solar(timestamps, solar_forecast_data)

        # Schedule Processing
        schedule_json = data.get(CONF_SCHEDULE_CONFIG)
        hvac_state_arr, setpoint_arr = self._process_schedule(timestamps, schedule_json)

        # Indoor Temp Array (Initial state is known, rest is 0/placeholder)
        t_in_arr = np.zeros(steps)
        t_in_arr[0] = current_temp

        # Construct Measurements
        # Note: timestamps in Measurements are expected to be numpy array?
        # The original code uses `len(data)` so list is fine, but type hint says np.array.
        
        measurements = Measurements(
            timestamps=np.array(timestamps),
            t_in=t_in_arr,
            t_out=np.array(t_out_arr),
            solar_kw=np.array(solar_arr),
            hvac_state=np.array(hvac_state_arr),
            setpoint=np.array(setpoint_arr),
            dt_hours=np.full(steps, dt_hours)
        )

        # 6. Run Optimization (Optional & Throttled)
        # Check options first, then fallback to config? Options are best for runtime.
        opt_enabled = self.config_entry.options.get(CONF_OPTIMIZATION_ENABLED, False)
        opt_interval = self.config_entry.options.get(CONF_OPTIMIZATION_INTERVAL, DEFAULT_OPTIMIZATION_INTERVAL)
        
        run_opt = False
        if opt_enabled:
            # Check throttling
            now_ts = now.timestamp()
            if self.last_optimization_time is None:
                run_opt = True
            else:
                elapsed_min = (now_ts - self.last_optimization_time) / 60.0
                if elapsed_min >= opt_interval:
                    run_opt = True
                    
        if run_opt:
            _LOGGER.info("Running HVAC Optimization...")
            # We need target temps. In fixed mode, 'setpoint' IS the target temp.
            target_temps = setpoint_arr.copy()
            
            # Simple Comfort Config for now
            # TODO: Expose these as options?
            comfort_config = {
                "mode": "heat", # Assume heat for winter
                "center_preference": 0.5
            }
            
            # Run Optimization (Blocking call - might need executor if slow, but L-BFGS-B is fastish)
            # Home Assistant warns on blocking I/O. If this takes < 0.1s it's fine.
            # If complex, run in executor:
            try:
                optimized_setpoints = await self.hass.async_add_executor_job(
                    optimize_hvac_schedule,
                    measurements,
                    params,
                    self.heat_pump,
                    target_temps,
                    comfort_config
                )
                
                # Apply optimization results
                measurements.setpoint = optimized_setpoints
                
                # We also need to update hvac_state in measurements?
                # optimize_hvac_schedule modifies hvac_state in place inside!
                # But let's be sure.
                # Actually optimize_hvac_schedule sets data.hvac_state[:] = hvac_mode_val
                # So measurements.hvac_state is already updated to the correct mode (1 or -1).
                
                # Result arrays for sensor need to be updated too
                setpoint_arr = optimized_setpoints.tolist() if hasattr(optimized_setpoints, "tolist") else optimized_setpoints
                hvac_state_arr = measurements.hvac_state.tolist() if hasattr(measurements.hvac_state, "tolist") else measurements.hvac_state
                
                self.last_optimization_time = now.timestamp()
                
            except Exception as e:
                _LOGGER.error("Optimization failed: %s", e)
                # Fallback to fixed schedule (do nothing, continue with original measurements)

        # 7. Run Model
        # run_model returns (sim_temps, rmse, hvac_outputs)
        sim_temps, _, _ = run_model(params, measurements, self.heat_pump, duration_minutes=duration_hours*60)

        # 7. Return Result
        return {
            "timestamps": timestamps,
            "predicted_temp": sim_temps,
            "hvac_state": hvac_state_arr,
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
