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
    
    # Create target array
    targets = np.zeros(len(timestamps))
    
    # For each timestamp, find the active schedule item
    # Since schedule is daily, we just look at the time of day
    
    # Optimization: Create a lookup for the day
    # But simpler: Iterate through the schedule intervals
    
    # Default to first item if only one, or last item of previous day
    # Let's assume the schedule covers the full 24h cycle.
    # The last item wraps around until the first item of the next day.
    
    # We can assign the target for the whole array based on time of day
    times_of_day = ts_index.time
    
    # Default swing if min/max not provided
    default_swing = 2.0

    targets = np.zeros(len(timestamps))
    min_bounds = np.zeros(len(timestamps))
    max_bounds = np.zeros(len(timestamps))
    
    # Default to last item for wrap-around
    last_item = schedule[-1]
    current_target = last_item['temp']
    current_min = last_item.get('min', current_target - default_swing)
    current_max = last_item.get('max', current_target + default_swing)
    
    targets[:] = current_target
    min_bounds[:] = current_min
    max_bounds[:] = current_max
    
    for item in schedule:
        t_start = parse_time(item['time'])
        temp = item['temp']
        t_min = item.get('min', temp - default_swing)
        t_max = item.get('max', temp + default_swing)
        
        mask = times_of_day >= t_start
        targets[mask] = temp
        min_bounds[mask] = t_min
        max_bounds[mask] = t_max
        
    return targets, min_bounds, max_bounds, config
