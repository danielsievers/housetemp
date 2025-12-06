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
    if not schedule:
        # Empty schedule, return existing targets (zeros) or handle gracefully
        # For now, let's assume valid schedule or return zeros
        return np.zeros(len(timestamps)), config

    # Convert timestamps to pandas datetime index for easy access
    ts_index = pd.to_datetime(timestamps)
    
    # Convert to local timezone if timezone-aware, then extract time of day
    if ts_index.tz is not None:
        ts_local = ts_index.tz_convert('America/Los_Angeles')
    else:
        # Assume UTC if no timezone, convert to local
        ts_local = ts_index.tz_localize('UTC').tz_convert('America/Los_Angeles')

    targets = np.zeros(len(timestamps))
    
    # Enforce Nested Schema: Items must have 'weekdays' and 'daily_schedule'
    
    weekdays_map = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }
    
    daily_schedules = {} # Map weekday_index -> list of {time, temp}
    seen_days = set()
    
    for i, group in enumerate(schedule):
        days = group.get('weekdays')
        if not days:
            raise ValueError(f"Schedule group at index {i} is missing 'weekdays' list. Legacy flat format is no longer supported.")
            
        day_schedule = group.get('daily_schedule')
        if not day_schedule:
             raise ValueError(f"Schedule group at index {i} is missing 'daily_schedule'.")

        # Sort the inner schedule
        day_schedule.sort(key=lambda x: parse_time(x['time']))
        
        for d in days:
            d_lower = d.lower()
            if d_lower not in weekdays_map:
                    raise ValueError(f"Invalid weekday: {d}")
            wd_idx = weekdays_map[d_lower]
            
            if wd_idx in seen_days:
                    raise ValueError(f"Weekday {d} is defined in multiple schedule groups.")
            seen_days.add(wd_idx)
            
            daily_schedules[wd_idx] = day_schedule
    
    # Check if we have schedules for all days present in the data
    present_weekdays = np.unique(ts_local.dayofweek)
    for wd_idx in present_weekdays:
        if wd_idx not in daily_schedules:
            raise ValueError(f"No schedule defined for weekday index {wd_idx} (0=Mon, 6=Sun) which is present in the data.")
    
    # Apply schedules
    times_of_day = ts_local.time
    day_of_week = ts_local.dayofweek
    
    for wd_idx in present_weekdays:
        day_schedule = daily_schedules[wd_idx]
        day_mask = (day_of_week == wd_idx)
        
        day_times = times_of_day[day_mask]
        
        # Apply wrap-around from the LAST item of the SAME day's schedule
        last_item = day_schedule[-1]
        day_targets = np.full(np.sum(day_mask), last_item['temp'])
        
        for item in day_schedule:
            t_start = parse_time(item['time'])
            temp = item['temp']
            mask_ge_start = day_times >= t_start
            day_targets[mask_ge_start] = temp
            
        targets[day_mask] = day_targets
    
    return targets, config
