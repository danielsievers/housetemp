#!/usr/bin/python3

import pandas as pd
import numpy as np

# --- CONFIGURATION ---
# Create 48 hours of data (30 min intervals)
timestamps = pd.date_range(start="2025-01-15 00:00", periods=96, freq="30min")

# 1. Outdoor Temp: Sinusoidal wave between 40F (night) and 60F (day)
# Shifted so min is at 6am, max at 3pm
hours = timestamps.hour + (timestamps.minute / 60)
t_out = 50 - 10 * np.cos((hours - 3) * np.pi / 12)

# 2. Solar KW: Bell curve active between 7am and 5pm
solar = np.zeros(len(timestamps))
daylight_mask = (hours > 7) & (hours < 17)
# Simple peak of 4.0 kW at noon
solar[daylight_mask] = 4.0 * np.sin((hours[daylight_mask] - 7) * np.pi / 10)
solar = np.maximum(0, solar) # Clip negative values

# 3. Target Temp (Thermostat Schedule)
# 62F at night, 68F during day (7am - 9pm)
targets = np.where((hours >= 7) & (hours < 21), 68.0, 62.0)

# 4. Simulate Indoor Temp & HVAC State (The Physics)
# We simulate "Reality" so the data makes sense
t_in = np.zeros(len(timestamps))
hvac_mode = np.zeros(len(timestamps))
current_temp = 66.0 # Start temp

for i in range(len(timestamps)):
    t_in[i] = current_temp
    
    # Logic: If temp is below target - 1, turn HEAT ON
    if current_temp < targets[i] - 0.5:
        mode = 1 # Heat
    # If temp is above target + 0.5, turn OFF
    elif current_temp > targets[i] + 0.5:
        mode = 0 # Off
    else:
        mode = 0 # Deadband
        
    hvac_mode[i] = mode
    
    # Apply simplified physics to get next temp
    # Loss
    loss = 310 * (t_out[i] - current_temp)
    # Solar Gain
    sun = solar[i] * 17000 # Assume K=17000
    # Heater Gain (if On)
    heat = mode * 45000 # Assume 45k BTU output
    
    # Mass (C=8000)
    delta_T = (loss + sun + heat) * 0.5 / 8000 # 0.5 hours step
    current_temp += delta_T

# --- EXPORT TO CSV ---
df = pd.DataFrame({
    'time': timestamps,
    'indoor_temp': np.round(t_in, 2),
    'outdoor_temp': np.round(t_out, 2),
    'solar_kw': np.round(solar, 3),
    'hvac_mode': hvac_mode.astype(int),
    'target_temp': targets
})

filename = "test_data.csv"
df.to_csv(filename, index=False)
print(f"Successfully generated {filename} with {len(df)} rows.")
print("You can now run: python main.py test_data.csv -r")
