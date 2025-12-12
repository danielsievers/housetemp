import pandas as pd

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
