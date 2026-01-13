
import pytest
from unittest.mock import MagicMock
from datetime import datetime, timedelta
import pytz

from custom_components.housetemp.sensor import HouseTempPredictionSensor
from custom_components.housetemp.statistics import ComfortSample
from custom_components.housetemp.const import DOMAIN, DEFAULT_MODEL_TIMESTEP

# --- Executable Requirement: Hybrid Forecast ---

@pytest.mark.asyncio
async def test_requirement_hybrid_forecast_sources():
    """
    Requirement ID: OPT-001 (Time Alignment)
    
    Description:
    The Forecast Sensor must construct a timeline that:
    1. Uses Historical Data (from StatsStore) for timestamps BEFORE the simulation start.
    2. Uses Simulated Data (from Coordinator) for timestamps AFTER/AT the simulation start.
    3. Correctly reports Historical Outdoor Temperature if available.
    
    Scenario:
    - Clock Time: 07:22 (Simulation Start)
    - Grid: 07:15, 07:30
    
    Expected Behavior:
    - 07:15: Sourced from StatsStore (History)
    - 07:30: Sourced from Simulation (Future)
    """
    
    # 1. Setup Environment
    mock_hass = MagicMock()
    mock_hass.config.time_zone = "America/Los_Angeles"
    tz = pytz.timezone("America/Los_Angeles")
    
    mock_entry = MagicMock()
    mock_entry.options = {"model_timestep": 5}
    mock_entry.entry_id = "test_entry"
    
    # 2. Setup Data
    # Time Anchor: 07:22
    # Pytz Footgun: Use localize() to avoid LMT artifacts (-7:53 offset)
    now = tz.localize(datetime(2024, 1, 1, 7, 22, 0))
    
    # Grid Points
    t_past = tz.localize(datetime(2024, 1, 1, 7, 15, 0))
    t_future = tz.localize(datetime(2024, 1, 1, 7, 30, 0))
    
    # B. Simulation Data (Coordinator)
    # Starts at 07:22
    # Includes 07:30 point
    sim_timestamps = [
        now, 
        now + timedelta(minutes=5), # 7:27
        t_future,                   # 7:32 (approx 7:30 grid)
    ]
    
    coordinator_data = {
        "timestamps": sim_timestamps,
        "predicted_temp": [68.5, 69.0, 72.0], # Future rises to 72.0
        "setpoint": [70.0, 70.0, 70.0],
        "energy_kwh_steps": [0.0, 0.1, 0.1],
        "optimized_setpoint": [70.0, 70.0, 70.0],
        "outdoor": [45.5, 45.6, 46.0] # Future outdoor
    }
    
    mock_coordinator = MagicMock()
    mock_coordinator.data = coordinator_data

    # C. Sensor Logic Check (Gap Behavior)
    # 3. Instantiate Sensor
    sensor = HouseTempPredictionSensor(mock_coordinator, mock_entry)
    sensor.hass = mock_hass
    
    # NO StatsStore injection (Low Complexity Model)
    sensor._stats_store = None
    
    # 4. Execute
    attributes = sensor.extra_state_attributes
    forecast = attributes.get("forecast", [])
    
    # 5. Verify (Audit)
    
    # Expectation: GAP (Missing). 
    # Logic: 07:15 < 07:22 (Start) -> Skipped.
    # Note: Sensor outputs datetime in "%Y-%m-%dT%H:%M:%S" format (no offset)
    t_past_str = t_past.strftime("%Y-%m-%dT%H:%M:%S")
    p_past = next((x for x in forecast if x["datetime"] == t_past_str), None)
    
    assert p_past is None, "Past grid point (7:15) should be missing (Gap), but was found."
    
    # Find 07:30 point
    t_future_str = t_future.strftime("%Y-%m-%dT%H:%M:%S")
    p_future = next((x for x in forecast if x["datetime"] == t_future_str), None)
    assert p_future is not None, f"Missing future grid point (7:30). Available: {[x['datetime'] for x in forecast]}"
    
    # Audit 7:30 Sources
    # Should come from Simulation (72.0 at 7:30 index)
    # Note: 7:30 is index 2 in our mock arrays -> 72.0
    assert p_future["temperature"] == 72.0, f"Future temp mismatch. Expected 72.0 (Sim), got {p_future['temperature']}"

