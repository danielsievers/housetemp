
"""Test coordinator fixed schedule logic."""
from unittest.mock import patch, MagicMock, ANY
import pytest
import json
import numpy as np
from datetime import datetime

from homeassistant.helpers.update_coordinator import UpdateFailed
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.housetemp.const import (
    DOMAIN,
    CONF_SENSOR_INDOOR_TEMP,
    CONF_WEATHER_ENTITY,
    CONF_SOLAR_ENTITY,
    CONF_C_THERMAL,
    CONF_UA,
    CONF_K_SOLAR,
    CONF_Q_INT,
    CONF_H_FACTOR,
    CONF_HEAT_PUMP_CONFIG,
    CONF_SCHEDULE_CONFIG,
    CONF_FORECAST_DURATION,
    CONF_UPDATE_INTERVAL,
)
from custom_components.housetemp.coordinator import HouseTempCoordinator

# Config with a FIXED schedule block
FIXED_SCHEDULE = {
    "schedule": [
        {
            "weekdays": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
            "daily_schedule": [
                {"time": "00:00", "temp": 60, "fixed": False},
                {"time": "12:00", "temp": 75, "fixed": True}, # FIXED block
                {"time": "18:00", "temp": 60, "fixed": False}
            ]
        }
    ]
}

CONFIG_DATA = {
    CONF_SENSOR_INDOOR_TEMP: "sensor.indoor",
    CONF_WEATHER_ENTITY: "weather.home",
    CONF_SOLAR_ENTITY: "sensor.solar",
    CONF_C_THERMAL: 5000.0,
    CONF_UA: 200.0,
    CONF_K_SOLAR: 0.0,
    CONF_Q_INT: 0.0,
    CONF_H_FACTOR: 5000.0,
    CONF_HEAT_PUMP_CONFIG: '{"max_capacity": 20000, "cop_curve": [[0, 3], [100, 3]]}',
    CONF_SCHEDULE_CONFIG: json.dumps(FIXED_SCHEDULE),
    CONF_FORECAST_DURATION: 24,
    CONF_UPDATE_INTERVAL: 60,
}

@pytest.fixture
def mock_coordinator(hass):
    """Create a coordinator instance with mocked heat pump."""
    entry = MockConfigEntry(domain=DOMAIN, data=CONFIG_DATA)
    entry.add_to_hass(hass)
    
    with patch("custom_components.housetemp.coordinator.HeatPump") as mock_hp:
        mock_hp.return_value = MagicMock()
        # Ensure it has basic methods
        mock_hp.return_value.get_max_capacity.return_value = np.zeros(24) 
        mock_hp.return_value.get_cop.return_value = np.zeros(24)
        
        coord = HouseTempCoordinator(hass, entry)
        coord.heat_pump = mock_hp.return_value
        return coord

@pytest.mark.asyncio
async def test_coordinator_passes_fixed_flag(hass, mock_coordinator):
    """
    Integration Test:
    Verify that async_trigger_optimization calls optimize_hvac_schedule
    with a measurements object containing the correct 'is_setpoint_fixed' mask.
    """
    coordinator = mock_coordinator
    
    # 1. Setup Mock State (needed for update)
    hass.config.time_zone = "UTC"
    hass.states.async_set("sensor.indoor", "70.0")
    
    # Needs valid forecast for coordinator to run
    now = datetime.now()
    # Create unique timestamps to avoid duplicate index error
    from datetime import timedelta
    forecast = [
        {"datetime": (now + timedelta(hours=i)).isoformat(), "temperature": 50.0} 
        for i in range(48)
    ]
    hass.states.async_set("weather.home", "50.0", {"forecast": forecast})
    
    # Needs non-empty forecast to get timestamps? 
    # Coordinator creates timestamps from now() + duration.
    # It fetches weather forecast. If empty, it might default or error?
    # It tries to get forecast. If empty, it uses current temperature.
    # Let's verify coordinator behavior. It seems robust enough.
    
    # 2. Mock time to ensure we hit the 12:00-18:00 window properly
    # Must be timezone aware (UTC) to match dt_util.now()
    from datetime import timezone
    fixed_now = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc) # Sunday, UTC
    
    with patch("custom_components.housetemp.coordinator.dt_util.now", return_value=fixed_now), \
         patch("custom_components.housetemp.coordinator.optimize_hvac_schedule") as mock_optimize, \
         patch("custom_components.housetemp.coordinator.run_model_continuous") as mock_continuous, \
         patch("custom_components.housetemp.coordinator.estimate_consumption") as mock_estimate, \
         patch("custom_components.housetemp.coordinator.process_schedule_data") as mock_process:
         
        # Mock process_schedule_data to return Explicit Fixed Mask DYNAMICALLY
        def mock_process_side_effect(timestamps, *args, **kwargs):
            steps = len(timestamps)
            fixed_mask_arr = np.zeros(steps, dtype=bool)
            # Mark middle 10 steps as fixed
            start = max(0, int(steps/2) - 5)
            end = min(steps, start + 10)
            fixed_mask_arr[start:end] = True
            return (np.zeros(steps), np.full(steps, 70.0), fixed_mask_arr)
            
        mock_process.side_effect = mock_process_side_effect
        
        # 3. Setup Weather with matching timestamps
        from datetime import timedelta
        # 48 hours for safety, though only 24 used? Config says forecast_duration=24
        forecast = [
            {"datetime": (fixed_now + timedelta(hours=i)).isoformat(), "temperature": 50.0} 
            for i in range(48)
        ]
        hass.states.async_set("weather.home", "50.0", {"forecast": forecast})

        # Update HeatPump mock to be dynamic
        coordinator.heat_pump.get_max_capacity.side_effect = lambda t: np.zeros(len(t))
        coordinator.heat_pump.get_cop.side_effect = lambda t: np.ones(len(t)) * 3.0

        # Mock continuous model: dynamic length based on t_out_list (2nd arg)
        def mock_run_dynamic(*args, **kwargs):
             # args[1] is t_out_list
             n = len(args[1])
             return (np.zeros(n), np.zeros(n), np.zeros(n))
        mock_continuous.side_effect = mock_run_dynamic

        # Mock estimate_consumption
        mock_estimate.return_value = {'total_kwh': 50.0}

        # Mock optimize: dynamic setpoints
        def mock_opt_dynamic(data, *args, **kwargs):
             n = len(data.timestamps)
             return (np.zeros(n), {"success": True, "fun": 0.0})
        mock_optimize.side_effect = mock_opt_dynamic # 24 steps for 24h? Depends on dt.
        # dt depends on forecast interval? 
        # Actually coordinator interpolates to control_timestep (30m).
        # 24h / 30m = 48 steps.
        mock_optimize.return_value = (np.zeros(48), {"success": True, "fun": 0.0})
        
        # 3. Trigger generic update (populates self.data)
        # async_refresh calls _async_update_data
        await coordinator.async_refresh()
        
        # Trigger explicit optimization
        await coordinator.async_trigger_optimization()
        
        # 4. Assertions
        assert mock_optimize.called
        
        # Get args passed to optimize_hvac_schedule
        # args[0] is 'data' (Measurements object)
        call_args = mock_optimize.call_args
        measurements = call_args[0][0]
        
        fixed_mask = measurements.is_setpoint_fixed
        assert fixed_mask is not None, "Measurements.is_setpoint_fixed should not be None"
        
        # Debug info
        count_fixed = np.sum(fixed_mask)
        print(f"Fixed Mask: {fixed_mask.astype(int)}")
        print(f"Fixed Steps Found: {count_fixed}")
        print(f"Start Time: {measurements.timestamps[0]}")
        print(f"End Time: {measurements.timestamps[-1]}")
        print(f"Timestamps: {measurements.timestamps}")
        
        assert np.any(fixed_mask), "Fixed mask should contain True values for the 12:00-18:00 block"
        assert count_fixed > 0
