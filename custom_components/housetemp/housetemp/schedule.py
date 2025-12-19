import json
import numpy as np
import pandas as pd
from datetime import datetime, time

# --- DEFAULT OVERRIDES (Fallbacks) ---
# --- DEFAULT OVERRIDES (Fallbacks) ---
DEFAULT_WEEKDAYS_MAP = {
    'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
    'friday': 4, 'saturday': 5, 'sunday': 6
}

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
    # Convert to local timezone if timezone-aware, then extract time of day
    if ts_index.tz is not None:
        # Use existing timezone
        ts_local = ts_index
    else:
        # Assume UTC if no timezone, but prefer caller to handle this
        ts_local = ts_index.tz_localize('UTC')

    targets = np.zeros(len(timestamps))
    
    # Enforce Nested Schema: Items must have 'weekdays' and 'daily_schedule'
    
    weekdays_map = DEFAULT_WEEKDAYS_MAP
    
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

def process_schedule_data(timestamps, schedule_data, away_status=None, timezone=None, default_mode=None):
    """
    Process schedule data (dict) into setpoints and hvac state arrays.
    
    Args:
        timestamps: List/Array of datetime objects.
        schedule_data: Dictionary containing the schedule config.
        away_status: Optional tuple (is_away, away_end, away_temp).
        timezone: Optional timezone string or object. If provided, timestamps will be 
                 converted to this timezone before extracting time-of-day.
        default_mode: Optional string ('heat', 'cool') to use if schedule_data['mode'] is missing.
                      Must be provided if 'mode' is strictly required (it is).
        
    Returns:
        hvac_state (np.array), setpoint (np.array)
    """
    if "schedule" not in schedule_data or not isinstance(schedule_data["schedule"], list):
        raise ValueError("Schedule must be a dictionary with 'schedule' key.")

    # Use pandas for efficient processing
    ts_index = pd.to_datetime(timestamps)
    
    # Convert to local timezone if specified
    if timezone:
        if ts_index.tz is None:
            ts_index = ts_index.tz_localize('UTC').tz_convert(timezone)
        else:
            ts_index = ts_index.tz_convert(timezone)

    
    # Validation logic reused from load_comfort_schedule but applied here
    # To keep it DRY, we could extract the mapping logic, but for now copying the
    # vectorized filling logic is safest.
    
    weekdays_map = DEFAULT_WEEKDAYS_MAP
    
    daily_schedules = {}
    seen_days = set()
    
    for i, group in enumerate(schedule_data["schedule"]):
        days = group.get('weekdays', [])
        day_schedule = group.get('daily_schedule', [])
        
        # Sort schedule
        day_schedule.sort(key=lambda x: parse_time(x['time']))
        
        for d in days:
            d_lower = d.lower()
            if d_lower not in weekdays_map:
                raise ValueError(f"Invalid weekday: {d}")
                
            wd_idx = weekdays_map[d_lower]
            if wd_idx in seen_days:
                 # Validation upstream should catch duplicates, or we overwrite
                 pass 
            seen_days.add(wd_idx)
            daily_schedules[wd_idx] = day_schedule

    targets = np.zeros(len(timestamps))
    fixed_mask = np.zeros(len(timestamps), dtype=bool)
    
    # Process Timestamps
    times_of_day = ts_index.time
    day_of_week = ts_index.dayofweek
    
    # Default to first available schedule if gaps? 
    # Or assume coverage. Coordinator validates coverage.
    
    present_weekdays = np.unique(day_of_week)
    for wd_idx in present_weekdays:
        if wd_idx in daily_schedules:
            day_schedule = daily_schedules[wd_idx]
            day_mask = (day_of_week == wd_idx)
            day_times = times_of_day[day_mask]
            
            if not day_schedule:
                continue
                
            last_item = day_schedule[-1]
            day_targets = np.full(np.sum(day_mask), float(last_item['temp']))
            # Default last item logic applies for 'fixed' too?
            # If last item is fixed, it wraps around until the first item of same day.
            day_fixed = np.full(np.sum(day_mask), bool(last_item.get('fixed', False)), dtype=bool)
            
            for item in day_schedule:
                t_start = parse_time(item['time'])
                temp = float(item['temp'])
                is_fixed = bool(item.get('fixed', False))
                
                mask_ge_start = day_times >= t_start
                day_targets[mask_ge_start] = temp
                day_fixed[mask_ge_start] = is_fixed
                
            targets[day_mask] = day_targets
            fixed_mask[day_mask] = day_fixed
            
    # Apply HVAC Mode
    # Priority: 
    # 1. schedule_data['mode'] (Explicit JSON override)
    # 2. default_mode (Argument fallback)
    # Fail if neither
    
    global_mode = schedule_data.get("mode") # Explicit config wins
    if not global_mode:
        global_mode = default_mode # Fallback
        
    if not global_mode:
        raise ValueError("HVAC Mode ('heat' or 'cool') must be specified in schedule JSON or provided as default.")
        
    global_mode = global_mode.lower()
    hvac_state_val = 1 if global_mode == 'heat' else (-1 if global_mode == 'cool' else 0)
    hvac_state = np.full(len(timestamps), hvac_state_val)
    
    # Apply Away Override
    if away_status:
        is_away, away_end, away_temp = away_status
        if is_away and away_temp is not None:
             # Create mask for away time
             # timestamps < away_end
             # Ensure comparison is straight forward (tz-aware vs tz-aware)
             
             # Convert away_end to same tz as timestamps[0] if needed?
             # Assuming inputs are compatible.
             
             # Using list comp for robustness with mixed types if any, 
             # but ts_index is DatetimeIndex
             if away_end.tzinfo and ts_index.tz is None:
                  # localize index? or convert scalar?
                  pass
             
             # Handle Timezone matching for robust comparison
             # away_end (datetime) vs ts_index (DatetimeIndex)
             start_cmp = away_end
             if ts_index.tz is not None:
                 if start_cmp.tzinfo is None:
                     # If index is aware but away_end is naive, force aware (assume user local/config tz)
                     # But better to assume they match context if possible.
                     # Here we simply localize to match index if naive.
                     start_cmp = start_cmp.replace(tzinfo=ts_index.tz)
                 else:
                     # Both aware: Convert away_end to index's timezone
                     start_cmp = start_cmp.astimezone(ts_index.tz)
             else:
                 # Index is naive. If away_end is aware, make it naive
                 if start_cmp.tzinfo is not None:
                     start_cmp = start_cmp.replace(tzinfo=None)
             
             away_mask = ts_index < start_cmp
             targets[away_mask] = float(away_temp)
             # During Away, setpoint is strictly ENFORCED? 
             # For now, let's treat away mode as fixed/enforced too to prevent optimization overriding it
             fixed_mask[away_mask] = True
             
    return hvac_state, targets, fixed_mask
