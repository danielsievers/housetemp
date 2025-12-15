"""Test the House Temp Prediction services."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import timedelta

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.setup import async_setup_component
from homeassistant.const import CONF_PLATFORM, CONF_SCAN_INTERVAL

from custom_components.housetemp.const import DOMAIN
from custom_components.housetemp import async_setup_entry

# Mock config entry
from pytest_homeassistant_custom_component.common import MockConfigEntry

@pytest.fixture
def mock_coord_1():
    coord = MagicMock()
    coord.config_entry = MagicMock()
    coord.config_entry.title = "Entry 1"
    coord.config_entry.options = {"forecast_duration": 48}
    coord.async_trigger_optimization = AsyncMock(return_value={"status": "optimized"})
    coord.async_set_away_mode = AsyncMock(return_value={"success": True, "optimization_summary": {}})
    return coord

@pytest.fixture
def mock_coord_2():
    coord = MagicMock()
    coord.config_entry = MagicMock()
    coord.config_entry.title = "Entry 2"
    coord.config_entry.options = {"forecast_duration": 48}
    coord.async_trigger_optimization = AsyncMock(return_value={"status": "optimized"})
    coord.async_set_away_mode = AsyncMock(return_value={"success": True, "optimization_summary": {}})
    return coord

async def test_run_hvac_optimization_targeted(hass: HomeAssistant, mock_coord_1, mock_coord_2):
    """Test run_hvac_optimization service targeting specific entity."""
    
    # Setup 2 entries
    entry1 = MockConfigEntry(domain=DOMAIN, title="Entry 1", entry_id="entry1")
    entry1.add_to_hass(hass)
    entry2 = MockConfigEntry(domain=DOMAIN, title="Entry 2", entry_id="entry2")
    entry2.add_to_hass(hass)
    
    # Mock data
    hass.data[DOMAIN] = {
        "entry1": mock_coord_1,
        "entry2": mock_coord_2
    }
    
    # Register services
    assert await async_setup_component(hass, DOMAIN, {})
    
    # 1. Target Entry 1
    # We must mock service.async_extract_config_entry_ids because it relies on entity registry.
    # It is an async function, so we must mock it to return an awaitable.
    
    with patch("homeassistant.helpers.service.async_extract_config_entry_ids", new_callable=AsyncMock) as mock_extract:
        mock_extract.return_value = ["entry1"]
        await hass.services.async_call(
            DOMAIN, "run_hvac_optimization", {"duration": 24}, target={"entity_id": "sensor.entry1_prediction"}, blocking=True
        )
        
        mock_coord_1.async_trigger_optimization.assert_awaited_once_with(duration_hours=24)
        mock_coord_2.async_trigger_optimization.assert_not_awaited()

    # 2. Target Entry 2
    mock_coord_1.reset_mock()
    mock_coord_2.reset_mock()
    
    with patch("homeassistant.helpers.service.async_extract_config_entry_ids", new_callable=AsyncMock) as mock_extract:
         mock_extract.return_value = ["entry2"]
         await hass.services.async_call(
            DOMAIN, "run_hvac_optimization", {"duration": 24}, target={"entity_id": "sensor.entry2_prediction"}, blocking=True
        )
         mock_coord_2.async_trigger_optimization.assert_awaited_once_with(duration_hours=24)
         mock_coord_1.async_trigger_optimization.assert_not_awaited()

async def test_set_away_targeted(hass: HomeAssistant, mock_coord_1, mock_coord_2):
    """Test set_away service targeting specific entity."""
    
    # Setup entries
    entry1 = MockConfigEntry(domain=DOMAIN, title="Entry 1", entry_id="entry1")
    entry1.add_to_hass(hass)
    entry2 = MockConfigEntry(domain=DOMAIN, title="Entry 2", entry_id="entry2")
    entry2.add_to_hass(hass)
    
    hass.data[DOMAIN] = {
        "entry1": mock_coord_1,
        "entry2": mock_coord_2
    }
    
    assert await async_setup_component(hass, DOMAIN, {})

    # Target Entry 1
    with patch("homeassistant.helpers.service.async_extract_config_entry_ids", new_callable=AsyncMock) as mock_extract:
        mock_extract.return_value = ["entry1"]
        await hass.services.async_call(
            DOMAIN, "set_away", {"duration": "01:00:00"}, target={"entity_id": "sensor.entry1_prediction"}, blocking=True
        )
        # Check call arguments
        # Duration string might be converted to timedelta by cv
        # The mock expects whatever async_set_away_mode receives
        assert mock_coord_1.async_set_away_mode.called
        assert not mock_coord_2.async_set_away_mode.called


async def test_service_missing_target_error(hass: HomeAssistant):
    """Test functionality when target is missing (should error)."""
    assert await async_setup_component(hass, DOMAIN, {})
    
    with patch("homeassistant.helpers.service.async_extract_config_entry_ids", new_callable=AsyncMock) as mock_extract:
        mock_extract.return_value = []
        with pytest.raises(ServiceValidationError):
            await hass.services.async_call(
                DOMAIN, "run_hvac_optimization", {"duration": 24}, blocking=True
            )
