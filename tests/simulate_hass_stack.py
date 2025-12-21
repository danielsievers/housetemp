
import sys
import os
import asyncio
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../custom_components")))


# Mock homeassistant.core.SupportsResponse before importing housetemp
import homeassistant.core
if not hasattr(homeassistant.core, "SupportsResponse"):
    class SupportsResponse:
        ONLY = "only"
        OPTIONAL = "optional"
    homeassistant.core.SupportsResponse = SupportsResponse

from housetemp.input_handler import SimulationInputHandler

from housetemp.housetemp.schedule import process_schedule_data

from housetemp.housetemp.run_model import run_model, HeatPump
from housetemp.housetemp.energy import calculate_energy_stats

# Configure logging
logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger("simulate_hass")

# --- MOCKS ---
class MockHass:
    def __init__(self):
        self.config = MockConfig()
        
    async def async_add_executor_job(self, func, *args, **kwargs):
        # Run synchronously for simulation script
        return func(*args, **kwargs)

class MockConfig:
    def __init__(self):
        self.time_zone = "UTC" # Input CSV is +00:00

# --- MAIN ---
async def main():
    hass = MockHass()
    handler = SimulationInputHandler(hass)
    
    # 1. Load CSV Data
    csv_path = "data/8hr.csv"
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Convert CSV to "Forecast" format expected by handler
    # data/8hr.csv columns: time,outdoor_temp,solar_kw,indoor_temp (optional)
    weather_forecast = []
    solar_forecast = []
    
    for _, row in df.iterrows():
        ts_str = row['time']
        # Ensure ISO format
        
        weather_forecast.append({
            "datetime": ts_str,
            "temperature": row['outdoor_temp']
        })
        
        solar_forecast.append({
            "datetime": ts_str,
            "period_end": ts_str, # Handler uses period_end/start or datetime
            "watts": float(row['solar_kw']) * 1000 # Handler divides by 1000 for 'watts'
        })
        
    start_time = datetime.fromisoformat(df['time'].iloc[0])
    
    # Simulation Params
    duration_hours = 8
    model_timestep = 5 # minutes
    
    print("\n[HASS Stack] Preparing Simulation Data...")
    timestamps, t_out_arr, solar_arr, dt_values = await handler.prepare_simulation_data(
        weather_forecast,
        solar_forecast,
        start_time,
        duration_hours,
        model_timestep
    )
    
    # 2. Process Schedule
    print("\n[HASS Stack] Processing Schedule (data/cooling.json)...")
    with open("data/cooling.json") as f:
        schedule_data = json.load(f)
        
    # Mock Away status
    away_status = (False, None, None) 
    
    # Run process_schedule_data
    hvac_state_arr, setpoint_arr, fixed_mask_arr, rate_arr = process_schedule_data(
        timestamps, 
        schedule_data, 
        away_status, 
        timezone="UTC"
    )
    
    # 3. Setup Heat Pump
    print("\n[HASS Stack] Loading Heat Pump (data/water_heater.json)...")
    # Note: simulate_hass uses raw file, HASS handler copies to .storage. 
    # Logic is same once object is created.
    hp = HeatPump("data/water_heater.json")
    
    # 4. Setup Measurements Object
    from housetemp.housetemp.measurements import Measurements
    
    steps = len(timestamps)
    t_in_arr = np.zeros(steps)
    t_in_arr[0] = 59.5 # User start temp
    
    measurements = Measurements(
        timestamps=np.array(timestamps),
        t_in=t_in_arr,
        t_out=t_out_arr,
        solar_kw=solar_arr,
        hvac_state=np.array(hvac_state_arr),
        setpoint=np.array(setpoint_arr),
        dt_hours=dt_values,
        is_setpoint_fixed=np.array(fixed_mask_arr)
    )
    
    # 5. Load Model Params
    print("\n[HASS Stack] Loading Model (data/garage.json)...")
    with open("data/garage.json") as f:
        mdata = json.load(f)
        params = [
            mdata['C_thermal'],
            mdata['UA_overall'],
            mdata['K_solar'],
            mdata['Q_int'],
            mdata['H_factor']
        ]
        
    # 6. Run Simulation (Naive / Schedule Based)
    # HASS coordinator runs `run_model` with `duration_hours*60`
    
    print(f"\n[HASS Stack] Running Simulation ({len(timestamps)} steps)...")
    print(f"\n[HASS Stack] Running Simulation ({len(timestamps)} steps)...")
    
    # Manually unpack for continuous model
    max_caps_list = hp.get_max_capacity(measurements.t_out).tolist()
    
    # Import
    from housetemp.housetemp.run_model import run_model_continuous
    
    sim_temps, hvac_outputs, _ = run_model_continuous(
        params, 
        t_out_list=measurements.t_out.tolist(),
        solar_kw_list=measurements.solar_kw.tolist(),
        dt_hours_list=measurements.dt_hours.tolist(),
        setpoint_list=measurements.setpoint.tolist(),
        hvac_state_list=measurements.hvac_state.tolist(),
        max_caps_list=max_caps_list,
        min_output=hp.min_output_btu_hr,
        max_cool=hp.max_cool_btu_hr,
        eff_derate=1.0,
        start_temp=measurements.t_in[0]
    )
    
    # Mock RMSE (meaningless here as t_in is dummy)
    rmse = 0.0
    
    # 7. Energy Stats
    # Assuming params[4] matches H_factor from file (it does)
    energy_res = calculate_energy_stats(hvac_outputs, measurements, hp, params[4])
    
    print("\n" + "="*40)
    print("HASS SIMULATION RESULTS")
    print("="*40)
    print(f"Final Temp:       {sim_temps[-1]:.2f} F")
    print(f"Total Energy:     {energy_res.get('total_kwh', 0):.2f} kWh")
    print(f"RMSE (vs self?):  {rmse:.4f}") # RMSE against initialized zero-array is meaningless here but printed by function
    print(f"First 5 Setpoints: {setpoint_arr[:5]}")
    print(f"First 5 HVAC Modes: {hvac_state_arr[:5]}")
    
    # Check if HVAC ever turned on
    on_mask = hvac_state_arr != 0
    if np.any(on_mask):
        print(f"HVAC Active Steps: {np.sum(on_mask)} / {steps}")
    else:
        print("HVAC was OFF for entire duration.")

if __name__ == "__main__":
    asyncio.run(main())
