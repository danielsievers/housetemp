import pandas as pd
import datetime

def get_system_timezone():
    """Returns the system local timezone."""
    return datetime.datetime.now().astimezone().tzinfo

def upsample_dataframe(df: pd.DataFrame, freq: str, cols_linear: list = None, cols_ffill: list = None) -> pd.DataFrame:
    """
    Upsamples a dataframe to a higher frequency using appropriate interpolation methods.
    
    Args:
        df: DataFrame with a 'time' column (or datetime index).
        freq: Target frequency string (e.g. '10min', '15T').
        cols_linear: List of column names to interpolate linearly (continuous physics vars).
        cols_ffill: List of column names to forward fill (discrete states/setpoints).
    
    Returns:
        Upsampled DataFrame with 'dt' column recalculated.
    """
    # Ensure time index
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], utc=True) # Ensure UTC/tz-aware consistency if possible, or just to_datetime
        df = df.set_index('time')
    
    # Verify we have a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
         # Try to cast index
         try:
            df.index = pd.to_datetime(df.index)
         except:
            raise ValueError("upsample_dataframe requires a DatetimeIndex or a 'time' column convertible to datetime")
    
    if cols_linear is None: cols_linear = []
    if cols_ffill is None: cols_ffill = []
    
    # Filter columns that actually exist in df
    cols_linear = [c for c in cols_linear if c in df.columns]
    cols_ffill = [c for c in cols_ffill if c in df.columns]
    
    # 1. Create target grid and union index
    # We must use union to keep original points for accurate interpolation
    start = df.index.min().floor(freq)
    end = df.index.max().ceil(freq)
    grid = pd.date_range(start, end, freq=freq)
    
    # Union of original points and target grid
    union_idx = df.index.union(grid).sort_values().unique()
    df_union = df.reindex(union_idx)
    
    # 2. Interpolate Continuous variables
    if cols_linear:
        # method='time' requires a DatetimeIndex
        df_union[cols_linear] = df_union[cols_linear].interpolate(method='time')
        
    # 3. Fill Discrete variables
    if cols_ffill:
        df_union[cols_ffill] = df_union[cols_ffill].ffill()
    
    # 4. Filter back to just the target grid
    df_resampled = df_union.reindex(grid)
    
    # Guarantee index name so reset_index produces 'time' column
    df_resampled.index.name = 'time'
    df_out = df_resampled.reset_index()
    
    # 5. Calculate time deltas (dt) for physics
    
    # 4. Calculate time deltas (dt) for physics
    # Shift to align dt with interval (bfill first row)
    time_diffs = df_out['time'].diff().dt.total_seconds() / 3600.0
    df_out['dt'] = time_diffs.bfill()
    
    return df_out


async def fetch_history_frame(hass, entity_ids, start_time, end_time, minimal_response=True):
    """
    Fetch history for multiple entities and flatten into a single DataFrame.
    If minimal_response=False, the cells will contain the full state dict (including attributes).
    """
    from homeassistant.components.recorder import history
    from homeassistant.util import dt as dt_util

    # Run in executor (DB ops)
    def _fetch():
        return history.get_significant_states(
            hass,
            start_time,
            end_time,
            entity_ids=entity_ids,
            significant_changes_only=False,
            formatted=True, 
            minimal_response=minimal_response
        )
    
    history_data = await hass.async_add_executor_job(_fetch)
    
    # Process into dict of Series
    series_dict = {}
    
    for eid in entity_ids:
        states = history_data.get(eid, [])
        if not states:
            continue
            
        times = []
        values = []
        
        for s in states:
            try:
                ts = dt_util.parse_datetime(s['last_updated'])
                
                if minimal_response:
                    # Parse simplified state
                    state = s['state']
                    try:
                        val = float(state)
                    except ValueError:
                        val = state
                    values.append(val)
                else:
                    # Return full dict (caller processes state/attributes)
                    values.append(s)
                    
                times.append(ts)
            except (KeyError, ValueError):
                continue
                
        if times:
            series_dict[eid] = pd.Series(values, index=pd.to_datetime(times, utc=True))
    
    if not series_dict:
        return pd.DataFrame()
        
    # Concat (Outer Join)
    df = pd.DataFrame(series_dict)
    
    # Sort index
    df = df.sort_index()
    
    # No filling here
    df.index.name = 'time'
    
    return df

import numpy as np
import math


def quantize_setpoint(raw: float) -> int:
    """
    Canonical quantization: Half-up rounding (0.5 -> 1).
    All code paths MUST use this for consistent thermostat commands.
    """
    return int(math.floor(raw + 0.5))


def is_off_intent(sp: int, min_sp: int, max_sp: int, 
                  hvac_mode: int, eps: int = 0) -> bool:
    """
    Canonical OFF detection on quantized setpoints.
    
    Args:
        sp: Quantized integer setpoint.
        min_sp: Minimum setpoint boundary.
        max_sp: Maximum setpoint boundary.
        hvac_mode: +1 heating, -1 cooling, 0 off.
        eps: Integer tolerance (default 0 for 1°F steps).
    
    Returns:
        True if setpoint indicates "True Off" intent.
    """
    if hvac_mode > 0:  # Heating
        return sp <= (min_sp + eps)
    elif hvac_mode < 0:  # Cooling
        return sp >= (max_sp - eps)
    return True  # Mode is off

def get_effective_hvac_state(hvac_state_arr, setpoint_arr, min_setpoint, max_setpoint, off_eps):
    """
    Applies 'True Off' logic: Force HVAC State to 0 if setpoints are pinned to boundaries.
    
    Args:
        hvac_state_arr: Array of intended states (-1, 0, 1) or scalar.
        setpoint_arr: Array of setpoints or scalar.
        min_setpoint: User config min (float).
        max_setpoint: User config max (float).
        off_eps: Tolerance epsilon.
    
    Returns:
        Effective HVAC state array (0 where pinned).
    """
    # Ensure numpy arrays for vectorized logic
    states = np.array(hvac_state_arr, dtype=int, copy=True)
    setpoints = np.array(setpoint_arr, dtype=float)
    
    # Logic
    is_heating = states > 0
    is_cooling = states < 0
    
    # Heat: setpoint <= min + eps
    off_heat = is_heating & (setpoints <= (min_setpoint + off_eps))
    
    # Cool: setpoint >= max - eps
    off_cool = is_cooling & (setpoints >= (max_setpoint - off_eps))
    
    # Apply
    states[off_heat | off_cool] = 0
    
    return states
