"""Dry Run script for Home Assistant component."""
import asyncio
import json
import logging
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Add root to python path to allow imports
sys.path.append(os.getcwd())
# Also add homeassistant/custom_components to path to find the module directly if needed,
# or better yet, import as if we are inside homeassistant.
# The issue is 'custom_components' is not a package in root.
# It is inside 'homeassistant/'.
sys.path.append(os.path.join(os.getcwd(), 'homeassistant'))

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

# Now we can import from custom_components.housetemp
from custom_components.housetemp.coordinator import HouseTempCoordinator
from custom_components.housetemp.const import *

# Data provided by user
INDOOR_TEMP = 67

# Real Weather Forecast (Tomorrow)
WEATHER_FORECAST = [
    {"datetime": "2025-12-10T20:00:00+00:00", "temperature": 55},
    {"datetime": "2025-12-10T21:00:00+00:00", "temperature": 58},
    {"datetime": "2025-12-10T22:00:00+00:00", "temperature": 61},
    {"datetime": "2025-12-10T23:00:00+00:00", "temperature": 61},
    {"datetime": "2025-12-11T00:00:00+00:00", "temperature": 59},
    {"datetime": "2025-12-11T01:00:00+00:00", "temperature": 54},
    {"datetime": "2025-12-11T02:00:00+00:00", "temperature": 53},
    {"datetime": "2025-12-11T03:00:00+00:00", "temperature": 52},
    {"datetime": "2025-12-11T04:00:00+00:00", "temperature": 52},
    {"datetime": "2025-12-11T05:00:00+00:00", "temperature": 52},
    {"datetime": "2025-12-11T06:00:00+00:00", "temperature": 52},
    {"datetime": "2025-12-11T07:00:00+00:00", "temperature": 51},
    {"datetime": "2025-12-11T08:00:00+00:00", "temperature": 51},
    {"datetime": "2025-12-11T09:00:00+00:00", "temperature": 50},
    {"datetime": "2025-12-11T10:00:00+00:00", "temperature": 49},
    {"datetime": "2025-12-11T11:00:00+00:00", "temperature": 48},
    {"datetime": "2025-12-11T12:00:00+00:00", "temperature": 48},
    {"datetime": "2025-12-11T13:00:00+00:00", "temperature": 47},
    {"datetime": "2025-12-11T14:00:00+00:00", "temperature": 46},
    {"datetime": "2025-12-11T15:00:00+00:00", "temperature": 46},
    {"datetime": "2025-12-11T16:00:00+00:00", "temperature": 46},
    {"datetime": "2025-12-11T17:00:00+00:00", "temperature": 52},
    {"datetime": "2025-12-11T18:00:00+00:00", "temperature": 55},
    {"datetime": "2025-12-11T19:00:00+00:00", "temperature": 59}
]

# Real Solar Forecast (Solcast)
# Note: Format is slightly different in coordinator (it looks for 'datetime' or 'period_end' and 'value')
# Coordinator logic: dt_str = item.get('datetime') or item.get('period_end')
# val = item.get('value') or item.get('pv_estimate')
SOLAR_FORECAST = [
    {"period_end": "2025-12-10T00:00:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T00:30:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T01:00:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T01:30:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T02:00:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T02:30:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T03:00:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T03:30:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T04:00:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T04:30:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T05:00:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T05:30:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T06:00:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T06:30:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T07:00:00-08:00", "pv_estimate": 0.0667},
    {"period_end": "2025-12-10T07:30:00-08:00", "pv_estimate": 1.0239},
    {"period_end": "2025-12-10T08:00:00-08:00", "pv_estimate": 1.4821},
    {"period_end": "2025-12-10T08:30:00-08:00", "pv_estimate": 2.1149},
    {"period_end": "2025-12-10T09:00:00-08:00", "pv_estimate": 2.6393},
    {"period_end": "2025-12-10T09:30:00-08:00", "pv_estimate": 2.7362},
    {"period_end": "2025-12-10T10:00:00-08:00", "pv_estimate": 2.6279},
    {"period_end": "2025-12-10T10:30:00-08:00", "pv_estimate": 3.0628},
    {"period_end": "2025-12-10T11:00:00-08:00", "pv_estimate": 2.9983},
    {"period_end": "2025-12-10T11:30:00-08:00", "pv_estimate": 2.827},
    {"period_end": "2025-12-10T12:00:00-08:00", "pv_estimate": 2.6133},
    {"period_end": "2025-12-10T12:30:00-08:00", "pv_estimate": 2.2645},
    {"period_end": "2025-12-10T13:00:00-08:00", "pv_estimate": 1.7979},
    {"period_end": "2025-12-10T13:30:00-08:00", "pv_estimate": 1.3416},
    {"period_end": "2025-12-10T14:00:00-08:00", "pv_estimate": 0.7981},
    {"period_end": "2025-12-10T14:30:00-08:00", "pv_estimate": 0.3005},
    {"period_end": "2025-12-10T15:00:00-08:00", "pv_estimate": 0.1088},
    {"period_end": "2025-12-10T15:30:00-08:00", "pv_estimate": 0.0975},
    {"period_end": "2025-12-10T16:00:00-08:00", "pv_estimate": 0.0752},
    {"period_end": "2025-12-10T16:30:00-08:00", "pv_estimate": 0.0214},
    {"period_end": "2025-12-10T17:00:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T17:30:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T18:00:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T18:30:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T19:00:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T19:30:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T20:00:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T20:30:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T21:00:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T21:30:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T22:00:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T22:30:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T23:00:00-08:00", "pv_estimate": 0},
    {"period_end": "2025-12-10T23:30:00-08:00", "pv_estimate": 0}
]

# Real Configs loaded from files
with open("data/heat_pump.json") as f:
    HEAT_PUMP_CONFIG = f.read()

with open("data/comfort.json") as f:
    COMFORT_CONFIG = f.read()

with open("data/occupied.json") as f:
    params = json.load(f)
    
CONFIG_DATA = {
    CONF_SENSOR_INDOOR_TEMP: "sensor.indoor",
    CONF_WEATHER_ENTITY: "weather.home",
    CONF_SOLAR_ENTITY: "sensor.solar", # If used
    CONF_C_THERMAL: params["C_thermal"],
    CONF_UA: params["UA_overall"],
    CONF_K_SOLAR: params["K_solar"],
    CONF_Q_INT: params["Q_int"],
    CONF_H_FACTOR: params["H_factor"],
    CONF_HEAT_PUMP_CONFIG: HEAT_PUMP_CONFIG,
    CONF_SCHEDULE_CONFIG: COMFORT_CONFIG, # Note: Currently coordinator expects this raw
    CONF_FORECAST_DURATION: 24,
    CONF_UPDATE_INTERVAL: 15,
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
    hass.config.path = MagicMock(side_effect=lambda *args: os.path.join(os.getcwd(), *args))
    
    # Mock States
    hass.states = MagicMock()
    hass.states.get = MagicMock()
    
    def get_state(entity_id):
        m = MagicMock()
        if entity_id == "sensor.indoor":
            m.state = str(INDOOR_TEMP)
        elif entity_id == "weather.home":
            m.state = "50.0" # Current outdoor
            m.attributes = {"forecast": WEATHER_FORECAST}
        elif entity_id == "sensor.solar":
            m.state = "0.0"
            m.attributes = {"forecast": SOLAR_FORECAST}
        else:
            return None
        return m
    
    hass.states.get.side_effect = get_state
    
    # Mock Executor for Optimization (so it actually runs synchronously for this script)
    async def async_add_executor_job(func, *args):
        print("  -> Executing job (optimization)...")
        return func(*args)
    
    hass.async_add_executor_job = async_add_executor_job

    # 2. Setup Coordinator
    config_entry = MagicMock()
    config_entry.data = CONFIG_DATA
    config_entry.options = OPTIONS_DATA
    config_entry.entry_id = "test_entry"
    
    print("Initializing Coordinator...")
    coordinator = HouseTempCoordinator(hass, config_entry)
    
    # 3. Run Update
    print("Running Update (Prediction & Optimization)...")
    try:
        data = await coordinator._async_update_data()
        print("Update Successful!")
        
        # In a real coordinator, this is set automatically by the update wrapper, 
        # but calling the internal method directly returns the data.
        # data = coordinator.data # This is None because we bypassed the wrapper
        if data:
            ts = data['timestamps']
            pred = data['predicted_temp']
            setp = data['setpoint']
            hvac = data['hvac_state']
            out = data['outdoor']
            
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

            logging.info(f"\nResults (Full Duration: {len(ts)} steps):")
            logging.info(f"{'Time':<20} | {'Pred T':<8} | {'Opt Set':<8} | {'Sched T':<8} | {'HVAC':<5} | {'Outdoor':<8}")
            logging.info("-" * 80)
            
            prev_sched = None

            for i in range(len(ts)):
                # Convert to Local Time for display
                # Note: ts[i] is likely UTC. dt_util.as_local() handles conversion to system local time.
                ts_local = dt_util.as_local(ts[i])
                t_str = ts_local.strftime("%m-%d %H:%M")
                
                sched_val = get_sched_temp(ts_local)
                
                # Mark changes in Schedule
                sched_marker = f"{sched_val:.1f}"
                if prev_sched is not None and sched_val != prev_sched:
                    sched_marker += " *" # Highlight change
                
                logging.info(f"{t_str:<20} | {pred[i]:<8.1f} | {setp[i]:<8.1f} | {sched_marker:<8} | {int(hvac[i]):<5} | {out[i]:<8.1f}")
                
                prev_sched = sched_val
                
            logging.info(f"\nTotal Steps: {len(ts)}")
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
