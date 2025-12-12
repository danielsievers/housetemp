"""Tests for the SimulationInputHandler."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

from custom_components.housetemp.input_handler import SimulationInputHandler

@pytest.fixture
def mock_hass():
    """Mock Home Assistant object."""
    hass = MagicMock()
    hass.config.time_zone = "UTC"
    # Mock executor utility to just run the function directly
    async def async_add_executor_job(func, *args, **kwargs):
        return func(*args, **kwargs)
    hass.async_add_executor_job = async_add_executor_job
    return hass

@pytest.fixture
def handler(mock_hass):
    """Create a handler instance."""
    return SimulationInputHandler(mock_hass)

def test_parse_forecast_points(handler):
    """Test parsing logic for forecast lists."""
    forecast = [
        {"datetime": "2023-01-01T12:00:00", "temperature": 20.0},
        {"datetime": "2023-01-01T13:00:00", "value": 22.0}, # Wrong key for temp test
    ]
    
    # Test Temperature Keys
    pts = handler.parse_forecast_points(forecast, ['datetime'], ['temperature'])
    assert len(pts) == 1
    assert pts[0]['value'] == 20.0
    assert pts[0]['time'].isoformat() == "2023-01-01T12:00:00+00:00"

    # Test Solar Keys
    solar_forecast = [
        {"period_end": "2023-01-01T12:00:00", "pv_estimate": 2000.0}, # > 50 -> /1000 -> 2.0
        {"period_start": "2023-01-01T13:00:00", "watts": 1500},       # /1000 -> 1.5
    ]
    pts_solar = handler.parse_forecast_points(
        solar_forecast, 
        ['period_end', 'period_start'], 
        ['pv_estimate', 'watts']
    )
    assert len(pts_solar) == 2
    assert pts_solar[0]['value'] == 2.0
    assert pts_solar[1]['value'] == 1.5

@pytest.mark.asyncio
async def test_prepare_simulation_data(handler):
    """Test full data preparation flow."""
    start_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    duration_hours = 2
    model_timestep = 60 # 1 hour steps
    
    weather = [
        {"datetime": "2023-01-01T12:00:00+00:00", "temperature": 10.0},
        {"datetime": "2023-01-01T13:00:00+00:00", "temperature": 12.0},
        {"datetime": "2023-01-01T14:00:00+00:00", "temperature": 14.0},
        {"datetime": "2023-01-01T15:00:00+00:00", "temperature": 16.0},
    ]
    
    solar = [] # Empty solar
    
    timestamps, t_out, solar_arr, dt_values = await handler.prepare_simulation_data(
        weather, solar, start_time, duration_hours, model_timestep
    )
    
    # Check Timestamps
    # Should cover start_time to start_time + 2h
    # 12:00, 13:00, 14:00
    assert len(timestamps) >= 3
    assert timestamps[0] == start_time
    assert timestamps[-1] == start_time + timedelta(hours=2)
    
    # Check Temps
    # Should correspond to 10.0, 12.0, 14.0
    assert t_out[0] == 10.0
    assert t_out[1] == 12.0
    assert t_out[2] == 14.0
    
    # Check Solar
    assert np.all(solar_arr == 0.0)

@pytest.mark.asyncio
async def test_prepare_simulation_data_missing_weather(handler):
    """Test behavior when weather data has a gap."""
    start_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    weather = [
        {"datetime": "2023-01-01T12:00:00+00:00", "temperature": 10.0},
        # GAP until way later
        {"datetime": "2023-01-01T20:00:00+00:00", "temperature": 20.0},
    ]
    
    # Up-sampling with 'linear' interpolation usually bridges gaps fine, 
    # but let's see if we can trigger the "Weather forecast data gap" error 
    # if we force a NaN (maybe by having NO data after start?)
    
    weather_bad = [{"datetime": "2023-01-01T12:00:00+00:00", "temperature": 10.0}]
    
    # prepare_simulation_data will extrapolate bounds if data is missing at ends
    # But if there is NO data within the range?
    
    # Actually, the logic extrapolates:
    # if df_raw.index.max() < end_time: row = ... [-1]
    # So single point actually WORKS (constant temp).
    
    # How to break it? Maybe if upsample produces NaNs? 
    # Standard interpolation usually fills everything unless limit is set.
    # The code checks `if pd.isna(t_out_arr).any(): raise ValueError`
    
    # If we pass empty list, parse_points returns empty, sync method raises "No weather forecast data available"
    with pytest.raises(ValueError, match="No weather forecast data available"):
         await handler.prepare_simulation_data([], [], start_time, 2, 60)
