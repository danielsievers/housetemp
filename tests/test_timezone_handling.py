"""Tests for timezone handling and generalization."""
import pytest
import pandas as pd
import datetime
from unittest.mock import patch, MagicMock
from custom_components.housetemp.housetemp import utils, load_csv, schedule

def test_get_system_timezone():
    """Test that get_system_timezone returns a valid tzinfo object."""
    tz = utils.get_system_timezone()
    assert isinstance(tz, datetime.tzinfo)
    # Sanity check it matches local
    local_now = datetime.datetime.now().astimezone()
    assert tz == local_now.tzinfo

@patch('custom_components.housetemp.housetemp.utils.get_system_timezone')
@patch('pandas.read_csv')
def test_load_csv_converts_utc_to_system_local(mock_read_csv, mock_get_system_timezone):
    """Test that load_csv converts UTC timestamps to the system local timezone."""
    # Mock system timezone as 'America/New_York' (EST/EDT)
    # We'll use a timezone that has a clear offset difference from UTC
    import pytz
    mock_tz = pytz.timezone('America/New_York')
    mock_get_system_timezone.return_value = mock_tz

    # Create dummy data with UTC timestamps
    # 2023-01-01 12:00 UTC -> 07:00 EST
    utc_time = pd.Timestamp('2023-01-01 12:00:00', tz='UTC')
    data = {
        'time': [utc_time],
        'indoor_temp': [70],
        'outdoor_temp': [50],
        'solar_kw': [0],
        'hvac_mode': ['heat'],
        'target_temp': [70]
    }
    df = pd.DataFrame(data)
    mock_read_csv.return_value = df

    # Run load_csv
    measurements = load_csv.load_csv("dummy.csv")

    # Verify timestamps are now naive local time (EST)
    # 12:00 UTC is 07:00 EST
    expected_local = pd.Timestamp('2023-01-01 07:00:00')
    actual_ts = measurements.timestamps[0]
    
    # Measurements timestamps are numpy datetimes, convert back to check
    actual_pd = pd.Timestamp(actual_ts)
    
    assert actual_pd == expected_local
    assert actual_pd.tz is None  # Must be naive

@patch('custom_components.housetemp.housetemp.utils.get_system_timezone')
def test_schedule_fallback_to_system_local(mock_get_system_timezone):
    """Test that schedule.py falls back to system timezone if input is aware and no tz arg."""
    import pytz
    import numpy as np
    
    # Mock system timezone as 'America/Chicago'
    mock_tz = pytz.timezone('America/Chicago')
    mock_get_system_timezone.return_value = mock_tz
    
    # Create aware timestamps in UTC
    # 12:00 UTC -> 06:00 CST
    timestamps_utc = [pd.Timestamp('2023-01-01 12:00:00', tz='UTC')]
    
    # Simple schedule (Always 70F)
    schedule_data = {
        "mode": "heat",
        "schedule": [
           {"weekdays": ["sunday"], "daily_schedule": [{"time": "00:00", "temp": 70}]}
        ]
    }
    
    # Call process_schedule_data without explicit timezone arg
    # This triggers the fallback logic in the `elif` block
    hvac, setpoints, _, _ = schedule.process_schedule_data(timestamps_utc, schedule_data)
    
    # The function should convert to Chicago time internally for matching
    # Sunday 06:00 is definitely covered by the schedule
    assert setpoints[0] == 70.0
    
    # To truly verify conversion happened, let's make a schedule that changes at 07:00 CST
    # If it didn't convert, 12:00 UTC would hit the post-7am rule.
    # If it did convert, 06:00 CST would hit the pre-7am rule.
    schedule_data_split = {
        "mode": "heat",
        "schedule": [
           {"weekdays": ["sunday"], "daily_schedule": [
               {"time": "00:00", "temp": 60}, 
               {"time": "07:00", "temp": 80}
            ]}
        ]
    }
    
    hvac, setpoints, _, _ = schedule.process_schedule_data(timestamps_utc, schedule_data_split)
    
    # 12:00 UTC = 06:00 CST -> Should match 00:00-07:00 rule (60F)
    # If it remained UTC (12:00), it would match >07:00 rule (80F)
    assert setpoints[0] == 60.0

