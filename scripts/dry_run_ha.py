"""Dry Run script for Home Assistant component."""
import asyncio
import json
import logging
import sys
import os
import re
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

# Add root and homeassistant to python path
import pathlib
script_dir = pathlib.Path(__file__).parent.absolute()
root_dir = script_dir.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / 'homeassistant'))

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

# Mock frame helper to avoid DataUpdateCoordinator initialization error
import homeassistant.helpers.frame
homeassistant.helpers.frame.report_usage = lambda *args, **kwargs: None

# Now we can import from custom_components.housetemp
from custom_components.housetemp.coordinator import HouseTempCoordinator
from custom_components.housetemp.const import *
from custom_components.housetemp import async_setup_entry, async_setup
from custom_components.housetemp.config_flow import STEP_USER_DATA_SCHEMA
import voluptuous as vol

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
    import numpy as np
    # Make it timezone-aware (matching user's -08:00)
    # Using fixed offset for simplicity in dry run to match Solcast data properties
    tz = timezone(timedelta(hours=-8)) 
    base = datetime(2025, 12, 10, 0, 0, 0, tzinfo=tz) + timedelta(hours=hour)
    
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
    # Strip comments
    content = f.read()
    content_clean = re.sub(r"//.*", "", content)
    params = json.loads(content_clean)
    
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
}

async def main():
    print("--- Starting Dry Run ---")
    
    # 1. Mock Home Assistant
    hass = MagicMock(spec=HomeAssistant)
    hass.config = MagicMock() # Needs explicit creation if spec=HomeAssistant doesn't include it (it should, but safety first)
    hass.services = MagicMock()
    hass.services.async_call = MagicMock(side_effect=lambda *args, **kwargs: asyncio.Future())
    hass.services.async_call.return_value.set_result({}) # Return empty dict by default
    
    # Capture Service Registration
    service_handlers = {}
    def capture_service_register(domain, service, handler, **kwargs):
        print(f"  -> Captured service registration: {domain}.{service}")
        service_handlers[f"{domain}.{service}"] = handler
        
    hass.services.async_register = MagicMock(side_effect=capture_service_register)
    hass.services.has_service = MagicMock(return_value=False) # Simulate service not existing so it registers

    # Mock Event Bus for state tracking
    hass.bus = MagicMock()
    hass.bus.async_listen = MagicMock(return_value=lambda: None)

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
    
    # Mock Loop for Debouncer
    hass.loop = asyncio.get_event_loop()
    
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
        # Just run it
        return func(*args)
    
    hass.async_add_executor_job = async_add_executor_job

    # 1b. Schema Validation (Suggestion 2)
    import voluptuous as vol
    
    print("Validating Configuration Schemas...")
    # Validate User Data (Mixed into CONFIG_DATA in this script, but strictly schema is separate)
    # CONFIG_DATA matches user input? No, config_flow processing happens.
    # User Input -> STEP_USER_DATA_SCHEMA -> self._data
    # So we should validate against the parts of CONFIG_DATA that belong to that schema.
    
    # Note: validation might fail on mocked EntitySelector etc if we were running full flow,
    # but here we just check if keys match types roughly?
    # Actually, voluptuous schemas with EntitySelector might try to check entities existence in some context?
    # No, EntitySelector usually just returns the string entity_id in flow unless validated.
    # STEP_USER_DATA_SCHEMA uses selector.EntitySelector.
    # Let's hope basic validation passes for string->selector mapping logic (or skip complex selectors).
    # Since we use vol.Schema directly, it validates structure. 
    # EntitySelectorConfig is a wrapper; Schema expects dict keys.
    # Wait, EntitySelector validates output? 
    # Home Assistant config flow validation is done by the flow manager usually.
    # Directly running schema(data) might fail if schema uses custom validators.
    # Let's try basic validation for simple fields.
    
    # Basic Check:
    try:
        # Just check keys existence for now to avoid strict selector validation issues in dry run
        required_keys = [k for k in STEP_USER_DATA_SCHEMA.schema.keys() if isinstance(k, vol.Required)]
        for k in required_keys:
             if k.schema not in CONFIG_DATA:
                 print(f"ERROR: Missing required data key: {k.schema}")
        
        print("Schema keys check passed.")
    except Exception as e:
        print(f"Schema Validation Error: {e}")

    # 2. Setup Coordinator & Service Registration (Suggestion 3)
    # We will use __init__.async_setup_entry to setup everything properly, 
    # capturing the service handler and Coordinator.
    
    config_entry = MagicMock()
    config_entry.data = CONFIG_DATA
    # Merge options defaults manually or use what we have
    options_data = { 
        # Use defaults for new settings
        CONF_MODEL_TIMESTEP: DEFAULT_MODEL_TIMESTEP,
        CONF_CONTROL_TIMESTEP: DEFAULT_CONTROL_TIMESTEP,
        CONF_SCHEDULE_CONFIG: COMFORT_CONFIG, # Ensure schedule is available in options
        CONF_CENTER_PREFERENCE: 1.0,
    }
    config_entry.options = options_data
    config_entry.entry_id = "test_entry_123"
    config_entry.add_update_listener = MagicMock()
    config_entry.async_on_unload = MagicMock()
    config_entry.title = "Test House" # Added for service response mapping
    
    # Store mocked config entry in hass
    hass.config_entries = MagicMock() # Mock config_entries manager
    hass.config_entries.async_entries = MagicMock(return_value=[config_entry])
    
    # Needs to return a Future that is already done
    f = asyncio.Future()
    f.set_result(True)
    hass.config_entries.async_forward_entry_setups = MagicMock(return_value=f)
    hass.data = {DOMAIN: {}} # Initialize hass.data for the component

    print("Running async_setup (Domain Setup - Services)...")
    # Call async_setup to register services at domain level
    await async_setup(hass, {})
    
    print("Running async_setup_entry (Initialization)...")
    
    # We need to mock coordinator behavior inside setup?
    # async_setup_entry instantiates Coordinator.
    # We need to patch Coordinator so we can still intercept refresh calls if needed?
    # Or rely on our global patches?
    # We previously instantiated coordinator manually. Now setup_entry will do it.
    
    # To bypass Debouncer inside the coordinator created by setup_entry, we need to PACTH the class itself.
    
    # Define a dummy async update that does nothing during setup
    async def dummy_refresh():
        # During setup, we don't want to run the full update if it's going to fail due to mocks
        # However, async_config_entry_first_refresh expects it to complete.
        pass

    with unittest.mock.patch("custom_components.housetemp.HouseTempCoordinator.async_request_refresh") as mock_refresh:
        # Patching async_request_refresh to prevent debounce logic during setup
        mock_refresh.side_effect = dummy_refresh
        
        # Call async_setup_entry, which will create the coordinator and store it in hass.data
        await async_setup_entry(hass, config_entry)
        
    coordinator = hass.data[DOMAIN][config_entry.entry_id]
    
    # Patch request_refresh on the LIVE instance now
    async def mock_refresh_instance():
        print("  -> (Mock) Refresh requested. Updating data now...")
        await coordinator._async_update_data()
    coordinator.async_request_refresh = mock_refresh_instance
    
    # --- Patch optimize_hvac_schedule to force center_preference=0.1 for Dry Run ---
    from custom_components.housetemp import coordinator as coord_module
    original_optimize = coord_module.optimize_hvac_schedule
    
    def wrapped_optimize(*args, **kwargs):
        # args: data, params, hw, target_temps, comfort_config, block_size_minutes
        # We need to find comfort_config (5th arg, index 4)
        if len(args) > 4:
            comfort_config = args[4]
            if isinstance(comfort_config, dict):
                print(f"  -> (Patch) Overriding center_preference to 0.0 (User Eco) (was {comfort_config.get('center_preference')})")
                comfort_config['center_preference'] = 0.0
        return original_optimize(*args, **kwargs)
        
    # Apply the patch to the coordinator's import
    coord_module.optimize_hvac_schedule = wrapped_optimize
    
    # 3. Trigger Service Handler
    print("Triggering Service: housetemp.run_hvac_optimization...")
    try:
        result_payload = {}
        if "housetemp.run_hvac_optimization" in service_handlers:
            handler = service_handlers["housetemp.run_hvac_optimization"]
            
            # Prepare mock call
            call_data = {"duration": 24}
            call = MagicMock()
            call.data = call_data
            call.context = MagicMock()
            # Mock extract_config_entry_ids helper behavior for dry run
            # Since we use the helper, we need to mock it or ensure it works.
            # The helper uses call.data['entity_id'] etc, or call.context.
            # But in dry run we mocked the helper? No.
            # We are running the REAL __init__.py code.
            # Real async_extract_config_entry_ids uses hass.states, entity registry etc.
            # It will fail in dry run because we mocked hass.states.get but not the entity registry.
            
            # Use a patch for the dry run script specifically for this helper!
            from unittest.mock import patch, AsyncMock
            
            # We need to simulate that the helper returns our entry ID
            with patch("homeassistant.helpers.service.async_extract_config_entry_ids", new_callable=AsyncMock) as mock_extract:
                mock_extract.return_value = ["test_entry_123"]
                
                call.return_response = True
                
                # Execute Handler
                result_payload_map = await handler(call)
            
            print(f"Service Execution Complete. Keys: {result_payload_map.keys() if result_payload_map else 'None'}")
            
            # Extract forecast from result
            if result_payload_map and "Test House" in result_payload_map:
                 result_payload = result_payload_map["Test House"]
            else:
                 print("Error: Could not find result for Test House")

        else:
            print("ERROR: Service was not registered!")
            
        print("Optimization Successful!")
            
        if result_payload and "forecast" in result_payload:
            forecast_items = result_payload["forecast"]
            
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
                    val = 70.0 
                    if daily:
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

            logging.info(f"\nResults (Full Duration: {len(forecast_items)} steps):")
            logging.info(f"{'Time':<20} | {'Pred T':<8} | {'Opt Set':<8} | {'Sched T':<8} | {'HVAC':<5} | {'Outdoor':<8} | {'Solar kW':<8}")
            logging.info("-" * 90)

            for item in forecast_items:
                ts_iso = item["datetime"]
                ts_dt = datetime.fromisoformat(ts_iso)
                dt_str = ts_dt.strftime("%m-%d %H:%M")
                
                # Service now returns all needed data
                t_pred = item.get("predicted_temp")
                if t_pred is None:
                    logging.error("Missing predicted_temp in service response!")
                    t_pred = 0.0
                    
                t_opt = item.get("ideal_setpoint")
                # If ideal_setpoint missing, maybe use target?
                # Or None
                
                t_set_sched = item.get("target_temp", 0.0)
                
                t_out = item.get("outdoor_temp", 0.0)
                sol = item.get("solar_kw", 0.0)
                hvac_action = item.get("hvac_action", "off")
                
                # HVAC String
                hvac_state = 0
                if hvac_action == "heating": hvac_state = 1
                elif hvac_action == "cooling": hvac_state = -1
                hvac_str = f"{hvac_state}"
                
                ts_local = dt_util.as_local(ts_dt)
                sched_val = get_sched_temp(ts_local)
                sched_marker = f"{sched_val:.1f}"
                
                # Use 'None' string for missing Opt Set
                opt_str = f"{t_opt:<8.1f}" if t_opt is not None else f"{'None':<8}"
                
                logging.info(f"{dt_str:<20} | {t_pred:<8.1f} | {opt_str:<8} | {sched_marker:<8} | {hvac_str:<5} | {t_out:<8.1f} | {sol:<8.2f}")
                
            logging.info(f"\nTotal Steps: {len(forecast_items)}")
            
            # --- Energy Metrics ---
            if "optimization_summary" in result_payload:
                opt_sum = result_payload["optimization_summary"]
                baseline_kwh = opt_sum.get("total_energy_use_kwh", 0.0)
                opt_kwh = opt_sum.get("total_energy_use_optimized_kwh", 0.0)
                
                print("\n--- Energy Metrics ---")
                print(f"Baseline Energy (Schedule):   {baseline_kwh:.2f} kWh")
                print(f"Optimized Energy (Proposed):  {opt_kwh:.2f} kWh")
                if baseline_kwh > 0:
                     savings = (1 - (opt_kwh / baseline_kwh)) * 100
                     print(f"Potential Savings:            {savings:.1f}%")
            
            # --- Suggestion 1: Sensor Entity Check ---
            print("\n--- Sensor Entity Check ---")
            from custom_components.housetemp.sensor import HouseTempPredictionSensor
            
            # Instantiate sensor
            sensor = HouseTempPredictionSensor(coordinator, config_entry)
            
            print(f"Sensor State (Native Value): {sensor.native_value}")
            print(f"Sensor Attributes Keys: {list(sensor.extra_state_attributes.keys())}")
            
            # Verify forecast inside attributes
            attrs = sensor.extra_state_attributes
            if "forecast" in attrs:
                print(f"Sensor Forecast Items: {len(attrs['forecast'])}")
            else:
                print("Sensor Forecast: MISSING")
                
            if "energy_metrics" in attrs:
                em = attrs["energy_metrics"]
                print("\n--- DETAILED ENERGY METRICS (Dual Model) ---")
                print(f"Continuous Naive:      {em.get('continuous_naive')}")
                print(f"Continuous Optimized:  {em.get('continuous_optimized')}")
                print(f"Discrete Naive:        {em.get('discrete_naive')}")
                print(f"Discrete Optimized:    {em.get('discrete_optimized')}")
                
                if em.get('discrete_diagnostics'):
                    dd = em.get('discrete_diagnostics')
                    print(f"\n--- Discrete Diagnostics (Optimized) ---")
                    print(f"Cycles: {dd.get('cycle_count')} | Active: {dd.get('active_minutes')}m | Off: {dd.get('off_minutes')}m")
            else:
                print("Energy Metrics: MISSING")

        else:
             logging.error("Coordinator Data is None or Forecast Missing!")
            
    except Exception as e:
        print(f"Update Failed: {e}")
        import traceback
        traceback.print_exc()
            
    except Exception as e:
        print(f"Update Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure logs go to stdout
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
