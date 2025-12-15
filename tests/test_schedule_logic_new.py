"""Tests for schedule logic."""
import pytest
import numpy as np
from datetime import datetime, timedelta, timezone
from custom_components.housetemp.housetemp.schedule import process_schedule_data

def test_process_schedule_basic():
    """Test basic schedule processing."""
    timestamps = [
        datetime(2023, 1, 1, 8, 0, tzinfo=timezone.utc), # Sunday
        datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc),
        datetime(2023, 1, 2, 8, 0, tzinfo=timezone.utc), # Monday
    ]
    
    schedule = {
        "mode": "heat",
        "schedule": [
            {
                "weekdays": ["sunday"],
                "daily_schedule": [{"time": "00:00", "temp": 70}, {"time": "10:00", "temp": 72}]
            },
            {
                "weekdays": ["monday"],
                "daily_schedule": [{"time": "00:00", "temp": 68}]
            }
        ]
    }
    
    hvac, setpoints, _ = process_schedule_data(timestamps, schedule)
    
    # Sun 8:00 -> 70 (before 10)
    assert setpoints[0] == 70.0
    # Sun 12:00 -> 72 (after 10)
    assert setpoints[1] == 72.0
    # Mon 8:00 -> 68
    assert setpoints[2] == 68.0
    
    assert np.all(hvac == 1)

def test_process_schedule_away():
    """Test away mode override."""
    timestamps = [
        datetime(2023, 1, 1, 8, 0, tzinfo=timezone.utc), 
        datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc),
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
    
    # Away until 10:00
    away_end = datetime(2023, 1, 1, 10, 0, tzinfo=timezone.utc)
    away_status = (True, away_end, 50.0)
    
    hvac, setpoints, _ = process_schedule_data(timestamps, schedule, away_status)
    
    # 8:00 is away -> 50
    assert setpoints[0] == 50.0
    # 12:00 is back -> 70
    assert setpoints[1] == 70.0
