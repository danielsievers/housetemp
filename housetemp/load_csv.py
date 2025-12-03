import pandas as pd
import numpy as np
from .measurements import Measurements

def load_csv(filepath: str, override_start_temp: float = None) -> Measurements:
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
    df['dt'] = time_diffs.bfill() # Fill first row

    # Clean data (drop NaNs)
    # TODO
    #df = df.dropna()
    
    print(f"Successfully loaded {len(df)} rows.")

    # Check for required columns
    required_cols = ['time', 'outdoor_temp', 'solar_kw']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Handle optional columns (for forecast data)
    if 'indoor_temp' not in df.columns:
        if override_start_temp is not None:
            print(f"Warning: 'indoor_temp' missing. Using provided start temp: {override_start_temp} F.")
            df['indoor_temp'] = override_start_temp
        else:
            raise ValueError("Missing 'indoor_temp' column. You must provide --start-temp for forecast data.")
        
    if 'hvac_mode' not in df.columns:
        print("Warning: 'hvac_mode' missing. Assuming 0 (OFF).")
        df['hvac_mode'] = 0
        
    if 'target_temp' not in df.columns:
        print("Warning: 'target_temp' missing. Assuming 70.0 F.")
        df['target_temp'] = 70.0

    return Measurements(
        timestamps=df['time'].values,
        t_in=df['indoor_temp'].values,
        t_out=df['outdoor_temp'].values,
        solar_kw=df['solar_kw'].values,
        hvac_state=df['hvac_mode'].fillna(0).values, # Ensure input is 1, 0, or -1
        setpoint=df['target_temp'].values,
        dt_hours=df['dt'].values
    )
