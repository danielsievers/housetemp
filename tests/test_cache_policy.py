import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import time
from homeassistant.util import dt as dt_util

from custom_components.housetemp.coordinator import HouseTempCoordinator
from custom_components.housetemp.const import DOMAIN, CONF_CONTROL_TIMESTEP, DEFAULT_CONTROL_TIMESTEP

@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.config.path = MagicMock(return_value="/tmp")
    hass.data = {DOMAIN: {}}
    return hass

@pytest.fixture
def mock_config_entry():
    entry = MagicMock()
    entry.options = {}
    entry.data = {}
    entry.entry_id = "test_entry"
    return entry

@pytest.fixture
def coordinator(mock_hass, mock_config_entry):
    return HouseTempCoordinator(mock_hass, mock_config_entry)

def test_expire_cache_removes_past_entries(coordinator):
    """Test that past timestamps are removed but future/present ones remain."""
    
    # Mock current time
    now = datetime(2023, 1, 1, 12, 0, 0)
    now_ts = int(now.timestamp())
    
    with patch("homeassistant.util.dt.now", return_value=now):
        # Populate cache
        # 1. Past (1 hour ago)
        past_ts = int(now.timestamp()) - 3600
        # 2. Present (Now)
        present_ts = now_ts
        # 3. Future (1 hour ahead)
        future_ts = int(now.timestamp()) + 3600
        
        coordinator.optimized_setpoints_map = {
            past_ts: 70.0,
            present_ts: 71.0,
            future_ts: 72.0
        }
        
        coordinator._expire_cache()
        
        assert past_ts not in coordinator.optimized_setpoints_map
        assert present_ts in coordinator.optimized_setpoints_map
        assert future_ts in coordinator.optimized_setpoints_map
        assert len(coordinator.optimized_setpoints_map) == 2

def test_fifo_eviction_limit(coordinator):
    """Test that cache size is capped at 3000 entries using FIFO."""
    
    # Mock current time very far in past so nothing expires by time
    now = datetime(2020, 1, 1, 0, 0, 0) 
    
    with patch("homeassistant.util.dt.now", return_value=now):
        # Fill cache with 3500 entries (future relative to 2020)
        base_ts = int(now.timestamp()) + 10000 
        
        # Create 3500 entries
        cache_data = {
            base_ts + i: 70.0 for i in range(3500)
        }
        coordinator.optimized_setpoints_map = cache_data
        
        coordinator._expire_cache()
        
        # Should be capped at 3000
        assert len(coordinator.optimized_setpoints_map) == 3000
        
        # Check that we kept the *latest* ones (FIFO means drop oldest keys)
        # Verify oldest keys are gone
        assert base_ts not in coordinator.optimized_setpoints_map # First one
        assert (base_ts + 3499) in coordinator.optimized_setpoints_map # Last one

def test_cache_cleared_on_reinstantiation(mock_hass, mock_config_entry):
    """Verify that creating a new coordinator instance starts with empty cache."""
    # Instance 1
    c1 = HouseTempCoordinator(mock_hass, mock_config_entry)
    c1.optimized_setpoints_map = {1234567890: 70.0}
    
    # Simulating reload: Instance 2
    c2 = HouseTempCoordinator(mock_hass, mock_config_entry)
    assert len(c2.optimized_setpoints_map) == 0

def test_cache_expiration_honors_control_alignment(coordinator):
    """
    Regression Test:
    When start_time is floored to control_timestep (e.g., 30 min), 
    we must NOT expire cache entries that fall within the current block 
    but are technically in the past relative to 'now'.
    
    Scenario:
    - Control Step: 30 min
    - Now: 10:14
    - Start Time (Floored): 10:00
    - Cache contains: 10:00, 10:05, 10:10, 10:15
    - Expected: 10:00, 10:05, 10:10 MUST remain in cache because 
      simulation will request them (start_time=10:00).
    """
    # 2023-01-01 10:14:00
    now = datetime(2023, 1, 1, 10, 14, 0)
    
    # Configure coordinator with 30-min timestep for this test
    coordinator.config_entry.options = {CONF_CONTROL_TIMESTEP: 30}
    
    # Timestamps in the block
    ts_10_00 = int(datetime(2023, 1, 1, 10, 0, 0).timestamp())
    ts_10_05 = int(datetime(2023, 1, 1, 10, 5, 0).timestamp())
    ts_10_10 = int(datetime(2023, 1, 1, 10, 10, 0).timestamp())
    ts_10_15 = int(datetime(2023, 1, 1, 10, 15, 0).timestamp())
    
    coordinator.optimized_setpoints_map = {
        ts_10_00: 70.0,
        ts_10_05: 70.0,
        ts_10_10: 70.0,
        ts_10_15: 71.0, 
    }
    
    with patch("homeassistant.util.dt.now", return_value=now):
        coordinator._expire_cache()
        
        # Verify
        # 10:15 is future -> SHOULD exist
        assert ts_10_15 in coordinator.optimized_setpoints_map, "Future entry missing"
        
        # 10:00, 10:05, 10:10 are technically past (10:14)
        # BUT they are needed for the simulation block starting at 10:00.
        # They MUST remain.
        assert ts_10_00 in coordinator.optimized_setpoints_map
        assert ts_10_05 in coordinator.optimized_setpoints_map
        assert ts_10_10 in coordinator.optimized_setpoints_map
