import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta, timezone

from custom_components.housetemp.coordinator import HouseTempCoordinator
from custom_components.housetemp.const import DOMAIN, DEFAULT_UPDATE_INTERVAL

# Mock Constants
MOCK_ENTRY_ID = "test_entry"

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {DOMAIN: {}}
    return hass

@pytest.fixture
def mock_config_entry():
    entry = MagicMock()
    entry.entry_id = MOCK_ENTRY_ID
    entry.options = {"update_interval": 15} # 15 min
    entry.data = {}
    return entry

async def test_schedule_alignment(mock_hass, mock_config_entry):
    """Test that the scheduler aligns to the next grid point."""
    
    # 1. Setup Coordinator
    with patch("custom_components.housetemp.coordinator.HouseTempCoordinator._schedule_next_refresh") as mock_init_sched:
        coord = HouseTempCoordinator(mock_hass, mock_config_entry)
    
    # Verify default interval is None (Custom Scheduling)
    assert coord.update_interval is None
    assert coord._target_update_interval == timedelta(minutes=15)

    # 2. Test Case A: Mid-Block (10:03)
    now_a = datetime(2023, 1, 1, 10, 3, 0, tzinfo=timezone.utc)
    expected_a = datetime(2023, 1, 1, 10, 15, 0, tzinfo=timezone.utc)
    
    with patch("homeassistant.util.dt.now", return_value=now_a), \
         patch("homeassistant.helpers.event.async_track_point_in_time") as mock_track:
        
        coord._schedule_next_refresh()
        
        # Check call
        args, _ = mock_track.call_args
        scheduled_time = args[2]
        assert scheduled_time == expected_a
        assert coord._unsub_refresh is not None

    # 3. Test Case B: Exact Grid Point (10:15:00) -> Should schedule next (10:30)
    # Rationale: We add 1.0s buffer, so 10:15:00 becomes 10:15:01 -> Ceil to 10:30
    now_b = datetime(2023, 1, 1, 10, 15, 0, tzinfo=timezone.utc)
    expected_b = datetime(2023, 1, 1, 10, 30, 0, tzinfo=timezone.utc)
    
    with patch("homeassistant.util.dt.now", return_value=now_b), \
         patch("homeassistant.helpers.event.async_track_point_in_time") as mock_track:
        
        coord._schedule_next_refresh()
        
        args, _ = mock_track.call_args
        scheduled_time = args[2]
        assert scheduled_time == expected_b

    # 4. Test Case C: Just Before (10:14:59) -> Should schedule 10:15
    now_c = datetime(2023, 1, 1, 10, 14, 59, tzinfo=timezone.utc)
    expected_c = datetime(2023, 1, 1, 10, 15, 0, tzinfo=timezone.utc)

    with patch("homeassistant.util.dt.now", return_value=now_c), \
         patch("homeassistant.helpers.event.async_track_point_in_time") as mock_track:
        
        coord._schedule_next_refresh()
        
        args, _ = mock_track.call_args
        scheduled_time = args[2]
        assert scheduled_time == expected_c
