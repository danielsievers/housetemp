import pandas as pd
import numpy as np
import json
from custom_components.housetemp.housetemp.run_model import run_model
from custom_components.housetemp.housetemp.measurements import Measurements

# Load Data
df = pd.read_csv("data/8hr.csv", parse_dates=["time"])
# Basic Upsample logic manual
timestamps = pd.date_range(df["time"].iloc[0], df["time"].iloc[-1], freq="15min")
df_resampled = df.set_index("time").reindex(timestamps).interpolate(method="time").reset_index()
df = df_resampled.rename(columns={"index": "time"})

# Prep Measurements
timestamps_dt = df["time"].dt.to_pydatetime()
t_in = np.full(len(df), 59.4) # Start temp
t_out = df["outdoor_temp"].values
solar_kw = df["solar_kw"].values
setpoint = np.full(len(df), 95.0)
hvac_state = np.full(len(df), -1) # Cooling Mode
dt_hours = np.full(len(df), 0.25) # 15 min

m = Measurements(timestamps_dt, t_in, t_out, solar_kw, hvac_state, setpoint, dt_hours)

# Load Mixed Params
with open("data/mixed_config.json") as f:
    d = json.load(f)
params = [d["C_thermal"], d["UA_overall"], d["K_solar"], d["Q_int"], d["H_factor"]]

# Run
sim_temps, _, _ = run_model(params, m)

print(f"Start Temp: {sim_temps[0]:.2f}")
print(f"Final Temp: {sim_temps[-1]:.2f}")
print(f"Peak Temp:  {np.max(sim_temps):.2f}")
