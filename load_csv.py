import pandas as pd
import numpy as np
from measurements import Measurements

def load_csv(filepath: str) -> Measurements:
    print(f"Loading data from {filepath}...")
    
    # Expects CSV columns: time, indoor_temp, outdoor_temp, solar_kw, hvac_mode, target_temp
    # Note: Adjust column names below to match your specific CSV export if needed
    df = pd.read_csv(filepath)
    
    # Convert time
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # Calculate time deltas in hours (for physics integration)
    # We shift to align dt with the interval duration
    time_diffs = df['time'].diff().dt.total_seconds() / 3600
    df['dt'] = time_diffs.fillna(method='bfill') # Fill first row

    # Clean data (drop NaNs)
    # TODO
    #df = df.dropna()
    
    print(f"Successfully loaded {len(df)} rows.")

    return Measurements(
        timestamps=df['time'].values,
        t_in=df['indoor_temp'].values,
        t_out=df['outdoor_temp'].values,
        solar_kw=df['solar_kw'].values,
        hvac_state=df['hvac_mode'].values, # Ensure input is 1, 0, or -1
        setpoint=df['target_temp'].values,
        dt_hours=df['dt'].values
    )
