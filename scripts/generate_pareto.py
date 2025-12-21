
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import requests

# Add project root to path
sys.path.insert(0, os.getcwd())

from custom_components.housetemp.housetemp.optimize import optimize_hvac_schedule
from custom_components.housetemp.housetemp.run_model import HeatPump
from custom_components.housetemp.housetemp.measurements import Measurements
from custom_components.housetemp.housetemp.energy import estimate_consumption

DATA_DIR = 'data/scenarios'

def get_weather_data(name, date_str, lat=None, lon=None):
    # Default San Jose if not specified
    if lat is None: lat = 37.3382
    if lon is None: lon = -121.8863
    
    # Differentiate cache by location if not default
    suffix = ""
    if abs(lat - 37.3382) > 0.001:
        suffix = f"_{lat:.2f}_{lon:.2f}"
        
    filename = f"{name.split(' (')[0].lower().replace(' ', '_')}{suffix}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"Loading cached weather from {filepath}...")
        df = pd.read_csv(filepath, index_col=0, parse_dates=[0])
        # Ensure 5min freq
        series_5min = df['temperature']
        timestamps = series_5min.index
        t_out = series_5min.values
        return timestamps, t_out
        
    print(f"Fetching OpenMeteo data for {date_str} at {lat}, {lon}...")
    
    start_date = date_str
    # OpenMeteo daily API needs end_date too
    end_date = (pd.to_datetime(start_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "temperature_unit": "fahrenheit",
        "timezone": "America/Los_Angeles"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    # Process Hourly Data
    hourly = data['hourly']
    time_idx = pd.to_datetime(hourly['time'])
    temps = np.array(hourly['temperature_2m'])
    
    series = pd.Series(temps, index=time_idx)
    
    # Resample to 5min
    series_5min = series.resample('5min').interpolate(method='linear')
    # Limit to 24h exactly (288 steps)
    series_5min = series_5min.iloc[:288] 
    
    # Save to CSV
    os.makedirs(DATA_DIR, exist_ok=True)
    series_5min.name = 'temperature'
    series_5min.index.name = 'time'
    series_5min.to_csv(filepath, index=True, header=True)
    print(f"Saved weather data to {filepath}")
    
    timestamps = series_5min.index
    t_out = series_5min.values
    return timestamps, t_out

def get_kwh(meas, params, hw, setpoints, mode_val):
    res = estimate_consumption(meas, params, hw, setpoints=setpoints, hvac_mode_val=mode_val, min_setpoint=60, max_setpoint=90)
    return res['total_kwh']

def run_sanity_check(hw, params):
    print("--- Physics Sanity Check (kWh) ---")
    n = 288
    timestamps = pd.date_range("2024-01-01", periods=n, freq="5min")
    t_out = np.full(n, 45.0)
    
    meas = Measurements(
        timestamps.to_pydatetime(), t_out, np.zeros(n), 
        np.full(n, 69.0), np.full(n, 69.0), np.ones(n), np.full(n, 1/12)
    )
    
    e_69 = get_kwh(meas, params, hw, np.full(n, 69.0), 1)
    e_66 = get_kwh(meas, params, hw, np.full(n, 66.0), 1)
    
    print(f"Energy(69F): {e_69:.2f} kWh")
    print(f"Energy(66F): {e_66:.2f} kWh")
    
    if e_66 >= e_69:
        print("WARNING: Physics model shows NO savings for setbacks!")
    else:
        diff = e_69 - e_66
        pct = (diff / e_69) * 100
        print(f"Physics Model OK: {pct:.1f}% savings potential confirmed.")
    print("----------------------------")

def main():
    hw = HeatPump('data/heat_pump.json')
    # Params: C, UA, K_solar, Q_int, H_factor, Eff_Derate
    params = [5000, 300, 1000, 2000, 10000, 0.8] 
    hw.max_cool_btu_hr = 36000 
    
    run_sanity_check(hw, params)
    
    scenarios = [
        ("Cold Winter (Jan 13)", "2024-01-13", "heat"),
        ("Shoulder Season (Nov 12)", "2023-11-12", "heat"),
        ("Mild Summer (Jun 15)", "2024-06-15", "cool"),
        ("Hot Summer (Jul 05)",  "2024-07-05", "cool"),
        ("Mild Summer (Eco Sched)", "2024-06-15", "coolval_eco"),
        ("Hot Summer (Eco Sched)",  "2024-07-05", "coolval_eco"),
    ]
    
    plt.figure(figsize=(10, 6))
    
    for item in scenarios:
        name, date_str, mode = item[:3]
        lat, lon = item[3] if len(item) > 3 else (None, None)
        
        print(f"Running Scenario: {name}...")
        try:
            timestamps, t_out = get_weather_data(name, date_str, lat, lon)
            n = len(t_out)
            
            # Target Schedule
            target_temps = np.zeros(n)
            if mode == 'heat':
                target_temps[:] = 63.0
                target_temps[int(6*12):int(22*12)] = 69.0  # 6am-10pm (index * 12 for 5min steps)
                mode_val = 1
            elif mode == 'coolval_eco':
                # Deep Setback "Eco" Cooling Schedule (Mirroring eco.json aggression)
                # Sleep/Away: 85F
                # Home: 78F
                target_temps[:] = 85.0
                target_temps[int(6*12):int(22*12)] = 78.0
                mode_val = -1
                mode = 'cool' # specific key for config
            else:
                target_temps[:] = 75.0 
                target_temps[int(6*12):int(22*12)] = 72.0
                mode_val = -1
            
            # Setup Measurements
            meas = Measurements(
                timestamps.to_pydatetime(), t_out, np.zeros(n),
                np.full(n, target_temps[0]), # t_in start
                target_temps, 
                np.ones(n) if mode=='heat' else np.full(n, -1),
                np.full(n, 1/12)
            )
            
            # Calculate Baseline kWh
            baseline_kwh = get_kwh(meas, params, hw, target_temps, mode_val)
            print(f"  Baseline kWh: {baseline_kwh:.2f}")

            cp_values = np.linspace(0.0, 1.0, 11)
            results = []
            
            for u in cp_values:
                # 4.0F Slack to allow variation
                config = {'mode': mode, 'center_preference': float(u), 'comfort_mode': 'deadband', 
                          'deadband_slack': 4.0, 'min_setpoint': 60, 'max_setpoint': 90}
                
                # Use fine block size (15m)
                opt, meta = optimize_hvac_schedule(meas, params, hw, target_temps, config, block_size_minutes=15, enable_multiscale=True)
                
                if opt is not None:
                    opt_kwh = get_kwh(meas, params, hw, opt, mode_val)
                    if baseline_kwh > 0:
                        savings = (1 - opt_kwh / baseline_kwh) * 100
                    else:
                        savings = 0.0
                    
                    results.append({'u': u, 'savings': savings})
                    print(f"  u={u:.1f}: Savings={savings:.1f}% (kWh: {opt_kwh:.2f})")
            
            df = pd.DataFrame(results)
            if not df.empty:
                styles = [
                    {'color': 'blue', 'marker': 'o', 'linestyle': '-', 'label': 'Cold Winter', 'linewidth': 2},
                    {'color': 'cyan', 'marker': 's', 'linestyle': '--', 'label': 'Shoulder Season', 'linewidth': 2},
                    {'color': 'green', 'marker': '^', 'linestyle': '-.', 'label': 'Mild Summer', 'linewidth': 4, 'alpha': 0.6},
                    {'color': 'red', 'marker': 'x', 'linestyle': ':', 'label': 'Hot Summer', 'linewidth': 1.5},
                    {'color': 'lime', 'marker': '*', 'linestyle': '-', 'label': 'Mild Summer (Eco)', 'linewidth': 2},
                    {'color': 'orange', 'marker': 'D', 'linestyle': '--', 'label': 'Hot Summer (Eco)', 'linewidth': 2},
                ]
                
                # Map name to style
                style = next((s for s in styles if s['label'] in name), None)
                if style:
                    plt.plot(df['u'], df['savings'], **style)
                else:
                    plt.plot(df['u'], df['savings'], marker='o', label=name)
                
        except Exception as e:
            print(f"Failed {name}: {e}")
            import traceback
            traceback.print_exc()

    plt.title('Energy Savings vs CP (Dual-Weight, Real Data, kWh)')
    plt.xlabel('Comfort Preference (u) [0=Eco, 1=Comfort]')
    plt.ylabel('Estimated Savings (%) vs Baseline Schedule')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('cp_pareto.png')
    print("Plot saved to cp_pareto.png")

if __name__ == "__main__":
    main()
