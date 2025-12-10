"""Dry Run script for Home Assistant component."""
import asyncio
import json
import logging
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Add root and homeassistant to python path
import pathlib
script_dir = pathlib.Path(__file__).parent.absolute()
root_dir = script_dir.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / 'homeassistant'))

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

# Now we can import from custom_components.housetemp
from custom_components.housetemp.coordinator import HouseTempCoordinator
from custom_components.housetemp.const import *

# Data provided by user
INDOOR_TEMP = 67

# User-Provided Data (Solcast)
SOLCAST_TODAY = [
    {'period_start': '2025-12-10T00:00:00-08:00', 'pv_estimate': 0},
    {'period_start': '2025-12-10T01:00:00-08:00', 'pv_estimate': 0},
    {'period_start': '2025-12-10T02:00:00-08:00', 'pv_estimate': 0},
    {'period_start': '2025-12-10T03:00:00-08:00', 'pv_estimate': 0},
    {'period_start': '2025-12-10T04:00:00-08:00', 'pv_estimate': 0},
    {'period_start': '2025-12-10T05:00:00-08:00', 'pv_estimate': 0},
    {'period_start': '2025-12-10T06:00:00-08:00', 'pv_estimate': 0},
    {'period_start': '2025-12-10T07:00:00-08:00', 'pv_estimate': 0.0667},
    {'period_start': '2025-12-10T07:30:00-08:00', 'pv_estimate': 1.0239},
    {'period_start': '2025-12-10T08:00:00-08:00', 'pv_estimate': 1.4821},
    {'period_start': '2025-12-10T08:30:00-08:00', 'pv_estimate': 2.1149},
    {'period_start': '2025-12-10T09:00:00-08:00', 'pv_estimate': 2.6393},
    {'period_start': '2025-12-10T09:30:00-08:00', 'pv_estimate': 2.7362},
    {'period_start': '2025-12-10T10:00:00-08:00', 'pv_estimate': 2.6279},
    {'period_start': '2025-12-10T10:30:00-08:00', 'pv_estimate': 3.0628},
    {'period_start': '2025-12-10T11:00:00-08:00', 'pv_estimate': 2.9983},
    {'period_start': '2025-12-10T11:30:00-08:00', 'pv_estimate': 2.827},
    {'period_start': '2025-12-10T12:00:00-08:00', 'pv_estimate': 2.6133},
    {'period_start': '2025-12-10T12:30:00-08:00', 'pv_estimate': 2.2645},
    {'period_start': '2025-12-10T13:00:00-08:00', 'pv_estimate': 1.7979},
    {'period_start': '2025-12-10T13:30:00-08:00', 'pv_estimate': 1.3416},
    {'period_start': '2025-12-10T14:00:00-08:00', 'pv_estimate': 0.7981},
    {'period_start': '2025-12-10T14:30:00-08:00', 'pv_estimate': 0.3005},
    {'period_start': '2025-12-10T15:00:00-08:00', 'pv_estimate': 0.1081},
    {'period_start': '2025-12-10T15:30:00-08:00', 'pv_estimate': 0.0975},
    {'period_start': '2025-12-10T16:00:00-08:00', 'pv_estimate': 0.0752},
    {'period_start': '2025-12-10T16:30:00-08:00', 'pv_estimate': 0.0214},
    {'period_start': '2025-12-10T17:00:00-08:00', 'pv_estimate': 0},
]
SOLCAST_TOMORROW = [
    {'period_start': '2025-12-11T07:30:00-08:00', 'pv_estimate': 1.0637},
    {'period_start': '2025-12-11T08:00:00-08:00', 'pv_estimate': 1.8742},
    {'period_start': '2025-12-11T08:30:00-08:00', 'pv_estimate': 2.3208},
    {'period_start': '2025-12-11T09:00:00-08:00', 'pv_estimate': 2.6829},
    {'period_start': '2025-12-11T09:30:00-08:00', 'pv_estimate': 2.9219},
    {'period_start': '2025-12-11T10:00:00-08:00', 'pv_estimate': 3.0597},
    {'period_start': '2025-12-11T12:00:00-08:00', 'pv_estimate': 2.6265},
    {'period_start': '2025-12-11T14:00:00-08:00', 'pv_estimate': 0.7877},
]

# Synthetic Hourly Weather (matching ~58F range)
# Current time 2025-12-10, so forecast should cover 12-10 and 12-11
WEATHER_FORECAST_SERVICE_RESPONSE = []
for hour in range(48):
    # Determine basic day cycle
    import datetime
    import numpy as np
    # Make it timezone-aware (matching user's -08:00)
    # Using fixed offset for simplicity in dry run to match Solcast data properties
    tz = datetime.timezone(datetime.timedelta(hours=-8)) 
    base = datetime.datetime(2025, 12, 10, 0, 0, 0, tzinfo=tz) + datetime.timedelta(hours=hour)
    
    # Simple curve: 50F low + 10 * sin(...)
    temp = 55.0 + 5.0 * np.sin((hour - 9) * 2 * np.pi / 24)
    WEATHER_FORECAST_SERVICE_RESPONSE.append({
        'datetime': base.isoformat(),
        'temperature': round(temp, 1)
    })

# Real Configs loaded from files
with open("data/heat_pump.json") as f:
    HEAT_PUMP_CONFIG = f.read()

with open("data/comfort.json") as f:
    COMFORT_CONFIG = f.read()

with open("data/occupied.json") as f:
    params = json.load(f)
    
CONFIG_DATA = {
    CONF_SENSOR_INDOOR_TEMP: "sensor.indoor",
    CONF_WEATHER_ENTITY: "weather.tomorrow_io_home_daily",
    CONF_SOLAR_ENTITY: ["sensor.solcast_pv_forecast_forecast_today", "sensor.solcast_pv_forecast_forecast_tomorrow"],
    CONF_C_THERMAL: params["C_thermal"],
    CONF_UA: params["UA_overall"],
    CONF_K_SOLAR: params["K_solar"],
    CONF_Q_INT: params["Q_int"],
    CONF_H_FACTOR: params["H_factor"],
    CONF_SCHEDULE_CONFIG: COMFORT_CONFIG,
    CONF_HEAT_PUMP_CONFIG: HEAT_PUMP_CONFIG,
    CONF_FORECAST_DURATION: 24, # hours
    CONF_UPDATE_INTERVAL: 60,
}

OPTIONS_DATA = {
    CONF_OPTIMIZATION_ENABLED: True,
    CONF_OPTIMIZATION_INTERVAL: 60
}

async def main():
    print("--- Starting Dry Run ---")
    
    # 1. Mock Home Assistant
    hass = MagicMock(spec=HomeAssistant)
    hass.config = MagicMock() # Needs explicit creation if spec=HomeAssistant doesn't include it (it should, but safety first)
    hass.services = MagicMock()
    hass.services.async_call = MagicMock(side_effect=lambda *args, **kwargs: asyncio.Future())
    hass.services.async_call.return_value.set_result({}) # Return empty dict by default
    
    hass.config.path = lambda *x: os.path.join(os.getcwd(), *x)
    
    # Configure Timezone for dt_util
    # This is critical for dt_util.as_local() to work correctly in the script
    import homeassistant.util.dt as dt_util
    
    # Attempt to use system local timezone
    try:
        import tzlocal
        local_tz = tzlocal.get_localzone()
    except ImportError:
        # Fallback to America/Los_Angeles given user metadata, or UTC
        print("Warning: tzlocal not found, defaulting to America/Los_Angeles for dry run.")
        import zoneinfo
        local_tz = zoneinfo.ZoneInfo("America/Los_Angeles")

    dt_util.set_default_time_zone(local_tz)
    hass.config.time_zone = str(local_tz)
    
    # Mock States
    hass.states = MagicMock()
    hass.states.get = MagicMock()
    
    def get_state(entity_id):
        m = MagicMock()
        if entity_id == "sensor.indoor":
            m.state = str(INDOOR_TEMP)
        elif entity_id == "weather.tomorrow_io_home_daily":
            m.state = "58.0" # Current outdoor
            # No 'forecast' attribute implies it needs service call
            m.attributes = {} 
        elif entity_id == "sensor.solcast_pv_forecast_forecast_today":
            m.state = "15.4"
            m.attributes = {"detailedForecast": SOLCAST_TODAY}
        elif entity_id == "sensor.solcast_pv_forecast_forecast_tomorrow":
            m.state = "16.1"
            m.attributes = {"detailedForecast": SOLCAST_TOMORROW}
        else:
            return None
        return m
    
    hass.states.get.side_effect = get_state
    
    # Mock Service Call Response for Weather
    async def mock_service_call(domain, service, data, blocking, return_response):
        if domain == "weather" and service == "get_forecasts":
             # Return valid forecast for our entity
             return {"weather.tomorrow_io_home_daily": {"forecast": WEATHER_FORECAST_SERVICE_RESPONSE}}
        return {}

    hass.services.async_call = MagicMock(side_effect=mock_service_call)
    
    # Mock Executor for Optimization (so it actually runs synchronously for this script)
    async def async_add_executor_job(func, *args):
        print("  -> Executing job (optimization)...")
        return func(*args)
    
    hass.async_add_executor_job = async_add_executor_job

    # 2. Setup Coordinator
    config_entry = MagicMock()
    config_entry.data = CONFIG_DATA
    config_entry.options = { 
        CONF_OPTIMIZATION_ENABLED: True,
        CONF_OPTIMIZATION_INTERVAL: 0, # Force run immediately
        # Use defaults for new settings
        CONF_MODEL_TIMESTEP: DEFAULT_MODEL_TIMESTEP,
        CONF_CONTROL_TIMESTEP: DEFAULT_CONTROL_TIMESTEP,
    }
    config_entry.entry_id = "test_entry_123"
    
    print("Initializing Coordinator...")
    coordinator = HouseTempCoordinator(hass, config_entry)
    
    # 3. Run Update
    print("Running Update (Prediction & Optimization)...")
    try:
        result = await coordinator._async_update_data()
        print("Update Successful!")
        
        # In a real coordinator, this is set automatically by the update wrapper, 
        # but calling the internal method directly returns the data.
        # data = coordinator.data # This is None because we bypassed the wrapper
        if result:
            timestamps = result.get("timestamps")
            temps = result.get("predicted_temp")
            setpoints = result.get("setpoint") # This is the optimized setpoint used
            hvac_state = result.get("hvac_state")
            outdoor_temps = result.get("outdoor")
            solar_power = result.get("solar")
            
            # Parse Schedule for Comparison
            try:
                sched_data = json.loads(COMFORT_CONFIG)
                # Helper to look up scheduled temp for a given LOCAL time
                def get_sched_temp(dt_local):
                    # Finds the rule active for this time
                    t_str = dt_local.strftime("%H:%M")
                    # Assuming today is weekday, find relevant daily schedule
                    # ... simplified lookup for dry run ...
                    # Just flatten the first available schedule for now
                    daily = sched_data['schedule'][0]['daily_schedule']
                    daily = sorted(daily, key=lambda x: x['time'])
                    
                    val = 70 # Default
                    # Last rule of previous day
                    val = daily[-1]['temp']
                    
                    for item in daily:
                        if item['time'] <= t_str:
                            val = item['temp']
                        else:
                            break
                    return float(val)

            except Exception as e:
                print(f"Error parsing comparisons: {e}")
                get_sched_temp = lambda x: 0.0

            logging.info(f"\nResults (Full Duration: {len(timestamps)} steps):")
            logging.info(f"{'Time':<20} | {'Pred T':<8} | {'Opt Set':<8} | {'Sched T':<8} | {'HVAC':<5} | {'Outdoor':<8} | {'Solar kW':<8}")
            logging.info("-" * 90)
            
            # Re-parse scheduled setpoints to show comparison (approximate, since we don't have the schedule logic exposed here easily unless we dup code)
            # Actually, let's just show what we have. 
            # The 'setpoint' in result IS the one used by the model. If optimization ran, it's the optimized one.
            
            # Let's verify against original schedule if we want, but user just wants table.
            # The 'setpoints' array returned by coordinator IS the final target temp array.

            for i in range(len(timestamps)):
                dt_str = timestamps[i].strftime("%m-%d %H:%M")
                t_pred = temps[i]
                t_set = setpoints[i]
                h_state = hvac_state[i]
                t_out = outdoor_temps[i] if outdoor_temps is not None and i < len(outdoor_temps) else 0.0
                sol = solar_power[i] if solar_power is not None and i < len(solar_power) else 0.0
                
                # Check if setpoint differs from standard schedule? 
                # We don't have standard schedule handy here easily. 
                # Just mark if HVAC is ON?
                
                # Highlight if HVAC is active
                hvac_str = f"{int(h_state)}"
                
                # Check for optimized vs base? We don't have base here.
                # Just print.
                
                # Convert to Local Time for display
                # Note: ts[i] is likely UTC. dt_util.as_local() handles conversion to system local time.
                ts_local = dt_util.as_local(timestamps[i])
                
                sched_val = get_sched_temp(ts_local)
                
                # Mark changes in Schedule
                sched_marker = f"{sched_val:.1f}"
                
                logging.info(f"{dt_str:<20} | {t_pred:<8.1f} | {t_set:<8.1f} | {sched_marker:<8} | {hvac_str:<5} | {t_out:<8.1f} | {sol:<8.2f}")
                
            logging.info(f"\nTotal Steps: {len(timestamps)}")
        else:
             logging.error("Coordinator Data is None!")
            
    except Exception as e:
        print(f"Update Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure logs go to stdout
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
