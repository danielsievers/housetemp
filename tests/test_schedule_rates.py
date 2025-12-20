"""Tests for TOU rate parsing in schedule."""
import pytest
import numpy as np
from datetime import datetime, timedelta, timezone
from custom_components.housetemp.housetemp.schedule import process_schedule_data

def test_schedule_rate_defaults():
    """Verify default rate is 1.0 if not specified."""
    timestamps = [
        datetime(2023, 1, 1, 8, 0, tzinfo=timezone.utc),
        datetime(2023, 1, 1, 9, 0, tzinfo=timezone.utc),
    ]
    schedule = {
        "mode": "heat",
        "schedule": [
            {
                "weekdays": ["sunday"],
                "daily_schedule": [{"time": "00:00", "temp": 70}]
            }
        ]
    }
    
    _, _, _, rates = process_schedule_data(timestamps, schedule)
    assert np.all(rates == 1.0) # Should be 1.0 everywhere

def test_schedule_rate_carry_forward():
    """Verify rates carry forward and update independently of temp."""
    # 30-min steps for 24 hours
    start = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc) # Sunday
    timestamps = [start + timedelta(minutes=30*i) for i in range(48)] 
    
    schedule = {
        "mode": "heat",
        "schedule": [
            {
                "weekdays": ["sunday"],
                "daily_schedule": [
                    {"time": "00:00", "temp": 60, "rate": 0.5}, # Low rate
                    {"time": "08:00", "temp": 70},              # Temp change, Rate should persist (0.5)
                    {"time": "16:00", "temp": 70, "rate": 2.0}, # Rate change (Peak), Temp persists
                    {"time": "21:00", "temp": 60, "rate": 0.5}, # Both change
                ]
            }
        ]
    }
    
    _, _, _, rates = process_schedule_data(timestamps, schedule)
    
    # 00:00 - 08:00 -> Rate 0.5
    assert np.all(rates[0:16] == 0.5)
    
    # 08:00 - 16:00 -> Rate 0.5 (Carried forward)
    assert np.all(rates[16:32] == 0.5)
    
    # 16:00 - 21:00 -> Rate 2.0
    assert np.all(rates[32:42] == 2.0)
    
    # 21:00 - 00:00 -> Rate 0.5
    assert np.all(rates[42:48] == 0.5)

def test_schedule_inline_rate_no_carry():
    """Test that rate resets if not carried forward by new day logic?
       Actually valid logic is last_item carries forward to start of next day logic
       only if we implement it. schedule.py logic currently wraps around for temp.
       Let's check if my implementation supports wrap around for rates.
    """
    pass # Implementation details: I implemented simple "last known" carry forward within the loop.
         # But I initialized day_rates with `last_item['rate']`.
         # So yes, it wraps around (last entry of day applies to start of day until first entry).

def test_schedule_rate_wrap_around():
    """Verify rate from end of schedule applies to start of day."""
    start = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [start + timedelta(minutes=30*i) for i in range(10)] # 00:00 to 05:00
    
    schedule = {
        "mode": "heat",
        "schedule": [
            {
                "weekdays": ["sunday"],
                "daily_schedule": [
                    {"time": "08:00", "temp": 70, "rate": 2.0}, # Starts at 8am
                    {"time": "20:00", "temp": 60, "rate": 0.5}  # Ends at 20pm
                ]
            }
        ]
    }
    # 00:00 -> 05:00 is BEFORE first entry (08:00).
    # Should use LAST entry (20:00 -> Rate 0.5).
    
    _, _, _, rates = process_schedule_data(timestamps, schedule)
    
    assert np.all(rates == 0.5)
