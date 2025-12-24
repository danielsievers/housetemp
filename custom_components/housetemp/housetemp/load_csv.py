import pandas as pd
import numpy as np
from . import utils
from .measurements import Measurements

def load_csv(filepath: str, override_start_temp: float = None, upsample_freq: str = None) -> Measurements:
    print(f"Loading data from {filepath}...")
    
    # Expects CSV columns: time, indoor_temp, outdoor_temp, solar_kw, hvac_mode, target_temp
    # Note: Adjust column names below to match your specific CSV export if needed
    df = pd.read_csv(filepath)
    # Convert time immediately
    # PRIORITIZE 'time_local' if available (corrected for local time)
    if 'time_local' in df.columns:
        print("Using 'time_local' column for timestamps.")
        df['time'] = pd.to_datetime(df['time_local'])
    else:
        df['time'] = pd.to_datetime(df['time'])

    # Ensure measurements use NAIVE LOCAL TIME (Project Standard)
    # If aware, satisfy Measurements contract (which uses numpy array -> naive)
    # by converting to Local first, then stripping TZ.
    if df['time'].dt.tz is not None:
        # Default to US/Pacific if not specified (since project context implies it)
        # Verify if it's already in a TZ or if it's UTC
        # If the CSV has UTC, we must convert.
        # Check an item
        sample_tz = df['time'].iloc[0].tz
        if str(sample_tz) == 'UTC':
             # Convert to System Local Time
             local_tz = utils.get_system_timezone()
             print(f"Converting UTC timestamps to System Local Time ({local_tz})...")
             df['time'] = df['time'].dt.tz_convert(local_tz)
        
        # Finally strip timezone to make it naive (so numpy doesn't revert to UTC)
        df['time'] = df['time'].dt.tz_localize(None)

    # Calculate median time diff for warning/check
    if len(df) > 1:
        median_diff_min = df['time'].diff().dt.total_seconds().median() / 60.0
        
        # Warning for coarse data
        if upsample_freq is None and median_diff_min > 10:
            print(f"Warning: Data resolution is coarse (~{median_diff_min:.1f} min). "
                  f"Physics simulation may be unstable. Consider using --upsample.")

    # Upsampling Logic
    if upsample_freq:
        print(f"Upsampling data to {upsample_freq} resolution...")
        from .utils import upsample_dataframe
        
        cols_linear = ['outdoor_temp', 'solar_kw', 'indoor_temp']
        cols_ffill = ['hvac_mode', 'target_temp']
        
        df = upsample_dataframe(df, upsample_freq, cols_linear, cols_ffill)
        
        print(f"Upsampled to {len(df)} rows.")
    else:
        # Calculate dt if not upsampled (upsample_dataframe handles it otherwise)
        time_diffs = df['time'].diff().dt.total_seconds() / 3600
        df['dt'] = time_diffs.bfill()

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
    # Check if column is missing OR if the first value is NaN (indicating empty placeholder)
    if 'indoor_temp' not in df.columns or pd.isna(df['indoor_temp'].iloc[0]):
        if override_start_temp is not None:
            print(f"Warning: 'indoor_temp' missing/NaN. Using provided start temp: {override_start_temp} F.")
            df['indoor_temp'] = override_start_temp
        else:
            raise ValueError("Indoor temperature data is missing/NaN and no --start-temp was provided.")
        
    if 'hvac_mode' not in df.columns:
        print("Warning: 'hvac_mode' missing. Assuming 0 (OFF).")
        df['hvac_mode'] = 0
        
    if 'target_temp' not in df.columns:
        print("Warning: 'target_temp' missing. Assuming 70.0 F.")
        df['target_temp'] = 70.0
    
    # Fill remaining NaNs in 'linear' columns that might have slipped through (e.g. solar at night)
    df['solar_kw'] = df['solar_kw'].fillna(0)
    
    return Measurements(
        timestamps=df['time'].values,
        t_in=df['indoor_temp'].values,
        t_out=df['outdoor_temp'].values,
        solar_kw=df['solar_kw'].values,
        hvac_state=df['hvac_mode'].fillna(0).values, # Ensure input is 1, 0, or -1
        setpoint=df['target_temp'].values,
        dt_hours=df['dt'].values
    )
