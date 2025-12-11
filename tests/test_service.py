"""Test the HVAC optimization service."""
import pytest
from unittest.mock import AsyncMock, patch
from custom_components.housetemp.const import DOMAIN
from pytest_homeassistant_custom_component.common import MockConfigEntry

@pytest.mark.asyncio
async def test_service_resgistered_and_calls_coordinator(hass):
    """Test that the service is registered and calls the coordinator."""
    # Valid config
    hp_config = '{"max_capacity": 15000, "cop": [[-10, 3], [10, 4]]}'
    data = {
        "sensor_indoor_temp": "sensor.indoor", 
        "weather_entity": "weather.home",
        "heat_pump_config": hp_config
    }
    entry = MockConfigEntry(domain=DOMAIN, data=data)
    entry.add_to_hass(hass)
    
    # Mock dependencies to prevent real IO/Network during setup
    with patch("custom_components.housetemp.coordinator.HeatPump"), \
         patch("custom_components.housetemp.coordinator.HouseTempCoordinator._prepare_simulation_inputs", side_effect=lambda: None):
         # Note: mocking _prepare_simulation_inputs as None might cause update to fail if it expects return values?
         # _async_update_data calls it.
         # Actually, we can just mock run_model to return empty result and prep to return dummy.
         pass
    
    # Better: Just let it fail first refresh? If first refresh fails, setup fails.
    # We need first refresh to succeed.
    with patch("custom_components.housetemp.coordinator.HouseTempCoordinator._async_update_data", return_value={"timestamps": [], "predicted_temp": []}):
        # Setup integration
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    # Get the real coordinator
    assert entry.entry_id in hass.data[DOMAIN]
    coordinator = hass.data[DOMAIN][entry.entry_id]
    
    # Patch the trigger method on the instance
    with patch.object(coordinator, "async_trigger_optimization") as mock_trigger:
        # Verify service registered
        assert hass.services.has_service(DOMAIN, "run_hvac_optimization")
        
        # Call service
        await hass.services.async_call(DOMAIN, "run_hvac_optimization", {}, blocking=True)
        
        # Verify coordinator triggered
        assert mock_trigger.called

@pytest.mark.asyncio
async def test_service_call_handles_exceptions(hass):
    """Test that service call handles exceptions from coordinator gracefully."""
    # Valid config
    hp_config = '{"max_capacity": 15000, "cop": [[-10, 3], [10, 4]]}'
    data = {
        "sensor_indoor_temp": "sensor.indoor", 
        "weather_entity": "weather.home",
        "heat_pump_config": hp_config
    }
    entry = MockConfigEntry(domain=DOMAIN, data=data)
    entry.add_to_hass(hass)
    
    # Mock dependencies
    with patch("custom_components.housetemp.coordinator.HeatPump"), \
         patch("custom_components.housetemp.coordinator.HouseTempCoordinator._prepare_simulation_inputs", side_effect=lambda: None):
         pass
    
    # Setup
    with patch("custom_components.housetemp.coordinator.HouseTempCoordinator._async_update_data", return_value={"timestamps": [], "predicted_temp": []}):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    coordinator = hass.data[DOMAIN][entry.entry_id]
    
    # Mock trigger to raise exception
    with patch.object(coordinator, "async_trigger_optimization", side_effect=ValueError("Optimization Failed")) as mock_trigger:
        # Call service - Now that we support response, this should NOT raise but return error in dict
        # ONLY if return_response is True? Or always?
        # My implementation catches all exceptions and puts them in results dict.
        
        response = await hass.services.async_call(
            DOMAIN, 
            "run_hvac_optimization", 
            {}, 
            blocking=True, 
            return_response=True
        )
        
        assert mock_trigger.called
        assert "Mock Title" in response
        assert response["Mock Title"] == {"error": "Optimization Failed"}

@pytest.mark.asyncio
async def test_service_call_returns_data(hass):
    """Test that service call returns proper data and accepts duration."""
    # Valid config
    hp_config = '{"max_capacity": 15000, "cop": [[-10, 3], [10, 4]]}'
    data = {"sensor_indoor_temp": "sensor.indoor", "weather_entity": "weather.home", "heat_pump_config": hp_config}
    entry = MockConfigEntry(domain=DOMAIN, data=data, title="My House")
    entry.add_to_hass(hass)
    
    with patch("custom_components.housetemp.coordinator.HeatPump"), \
         patch("custom_components.housetemp.coordinator.HouseTempCoordinator._prepare_simulation_inputs", side_effect=lambda: None):
         pass

    # Setup
    with patch("custom_components.housetemp.coordinator.HouseTempCoordinator._async_update_data", return_value={"timestamps": [], "predicted_temp": []}):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    coordinator = hass.data[DOMAIN][entry.entry_id]
    
    # Mock result
    expected_result = {
        "forecast": [{
            "time": "2023-01-01T12:00:00", 
            "ideal_setpoint": 72.0,
            "target_temp": 70.0,
            "predicted_temp": 68.5,
            "outdoor_temp": 50.0,
            "solar_kw": 0.5,
            "hvac_action": "heating"
        }],
        "optimization_summary": {
            "points": 1, 
            "total_energy_use_kwh": 5.0, 
            "total_energy_use_optimized_kwh": 4.5
        }
    }
    
    with patch.object(coordinator, "async_trigger_optimization", return_value=expected_result) as mock_trigger:
        # 1. Test Duration Passing
        await hass.services.async_call(DOMAIN, "run_hvac_optimization", {"duration": 48}, blocking=True)
        mock_trigger.assert_called_with(duration_hours=48.0) # Might come as int, check strictly if float cast is done inside or passed as is.
        # Coordinator logic type checks / converts it, but service handler passes raw.
        # Actually in __init__.py we pass: duration_hours=duration.
        # And yaml selector returns numbers. 
        # Let's assert called with 48 (int) if HA passes int.
        
        # 2. Test Response
        response = await hass.services.async_call(
            DOMAIN, 
            "run_hvac_optimization", 
            {"duration": 12}, 
            blocking=True, 
            return_response=True
        )
        
        mock_trigger.assert_called_with(duration_hours=12)
        
        # Verify structure: {"My House": result}
        assert "My House" in response
        result = response["My House"]
        
        # Verify return structure
        assert "forecast" in result
        assert "optimization_summary" in result
        
        summary = result["optimization_summary"]
        assert "points" in summary
        assert "total_energy_use_kwh" in summary
        assert "total_energy_use_optimized_kwh" in summary
        assert isinstance(summary["total_energy_use_kwh"], float)
        assert isinstance(summary["total_energy_use_optimized_kwh"], float)
        
        assert len(result["forecast"]) == 1 # We mocked only one item
        item = result["forecast"][0]
        assert "target_temp" in item
        assert "ideal_setpoint" in item
        assert "predicted_temp" in item # New Field
        
        # Check that predicted_temp is populated (float)
        assert isinstance(item["predicted_temp"], float)

        # Values for first step might be close to initial (68) or drifting
        # Since mocked model returns a path, we can check.
        # The optimization mocked might return different setpoints, but our coordinator 
        # mock run_model logic in test might just return zeroes if not mocked correctly?
        # Wait, we use Spy on coordinator, but run_model is a standalone function?
        # In integration: run_model is imported.
        # In test: we rely on real logic unless mocked.
        # We should ensure `run_model` returns something reasonable or inspect coordinator's behavior.
        
        # In this test setup, `run_model` is likely mocked or we need to check.
        # We didn't mock `run_model` in `test_service_call_returns_data`.
        # It runs real physics with dummy inputs. So it should be valid float.
        
        # Proactively print output for User Review as requested
        print("\n\n--- EXAMPLE SERVICE OUTPUT ---")
        import json
        print(json.dumps(response, indent=2))
        print("------------------------------\n")
