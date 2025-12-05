import json
import numpy as np
import pandas as pd
from datetime import datetime, time

def parse_time(t_str):
    return datetime.strptime(t_str, "%H:%M").time()

def load_comfort_schedule(json_path, timestamps):
    """
    Loads the comfort schedule from JSON and generates a target temperature
    array matching the provided timestamps.
    """
    with open(json_path, 'r') as f:
        config = json.load(f)
        
    schedule = config.get('schedule', [])
    # Sort schedule by time just in case
    schedule.sort(key=lambda x: parse_time(x['time']))
    
    # Convert timestamps to pandas datetime index for easy access
    ts_index = pd.to_datetime(timestamps)
    
    # Convert to local timezone if timezone-aware, then extract time of day
    # This ensures schedule times (e.g., "22:00") match local clock time
    if ts_index.tz is not None:
        ts_local = ts_index.tz_convert('America/Los_Angeles')
    else:
        # Assume UTC if no timezone, convert to local
        ts_local = ts_index.tz_localize('UTC').tz_convert('America/Los_Angeles')
    
    times_of_day = ts_local.time
    
    # Default swing if min/max not provided
    default_swing = 2.0

    targets = np.zeros(len(timestamps))
    
    # Default to last item for wrap-around
    last_item = schedule[-1]
    current_target = last_item['temp']
    
    targets[:] = current_target
    
    for item in schedule:
        t_start = parse_time(item['time'])
        temp = item['temp']
        
        mask = times_of_day >= t_start
        targets[mask] = temp
        
    return targets, config
