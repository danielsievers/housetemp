"""Sensor platform for House Temp Prediction."""
from __future__ import annotations

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTemperature, UnitOfEnergy
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import HouseTempCoordinator

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the sensor platform."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([HouseTempPredictionSensor(coordinator, entry)])

class HouseTempPredictionSensor(CoordinatorEntity, SensorEntity):
    """Representation of a House Temp Prediction Sensor."""

    _attr_has_entity_name = True
    _attr_name = "Indoor Temperature Forecast"
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_native_unit_of_measurement = UnitOfTemperature.FAHRENHEIT
    _attr_state_class = SensorStateClass.MEASUREMENT

    def __init__(self, coordinator: HouseTempCoordinator, entry: ConfigEntry):
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._entry = entry
        self._attr_unique_id = f"{entry.entry_id}_prediction"

    @property
    def native_value(self):
        """Return the state of the sensor."""
        # State is the current target temp (optimized or schedule)
        # We use index 0 which corresponds to the current time block
        if not self.coordinator.data:
            return None
        
        # Check Away Status for Fallback
        away_info = self.coordinator.data.get("away_info", {})
        
        optimized_setpoints = self.coordinator.data.get("optimized_setpoint")
        if optimized_setpoints is not None and len(optimized_setpoints) > 0:
            val = optimized_setpoints[0]
            if val is not None:
                return round(float(val), 1)
        
        # Fallback: If optimization missing but Away is active, show Away Temp
        if away_info.get("active") and away_info.get("temp") is not None:
             return round(float(away_info["temp"]), 1)
            
        setpoints = self.coordinator.data.get("setpoint")
        if setpoints is not None and len(setpoints) > 0:
            return round(float(setpoints[0]), 1)
            
        return None

    @property
    def extra_state_attributes(self):
        """Return the state attributes."""
        if not self.coordinator.data:
            return {}

        data = self.coordinator.data
        timestamps = data.get("timestamps", [])
        temps = data.get("predicted_temp", [])
        setpoints = data.get("setpoint", [])  # Schedule setpoint (target_temp)
        optimized_setpoints = data.get("optimized_setpoint", [])  # From HVAC optimization

        if timestamps is None or len(timestamps) == 0:
            return {}

        from homeassistant.util import dt as dt_util
        from datetime import timedelta

        # Resample to 15-minute intervals
        # Find start time rounded to nearest 15 min
        start_dt = timestamps[0]
        start_minute = (start_dt.minute // 15) * 15
        current_dt = start_dt.replace(minute=start_minute, second=0, microsecond=0)
        if current_dt < start_dt:
            current_dt += timedelta(minutes=15)

        end_dt = timestamps[-1]
        
        # Determine extra away info from data
        away_info = data.get("away_info", {})
        
        forecast = []
        while current_dt <= end_dt:
            # Find nearest data point (or interpolate)
            # Simple nearest-neighbor for temperature, last-value for setpoints
            best_idx = 0
            min_diff = abs((timestamps[0] - current_dt).total_seconds())
            for i, ts in enumerate(timestamps):
                diff = abs((ts - current_dt).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    best_idx = i
            
            local_dt = dt_util.as_local(current_dt)
            item = {
                "datetime": local_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "temperature": float(round(temps[best_idx], 1)) if best_idx < len(temps) else None,
                "target_temp": float(setpoints[best_idx]) if best_idx < len(setpoints) else None,
            }
            
            # ideal_setpoint only if optimization was run AND covers this time slot
            # Fallback to Away Temp if active and missing optimization
            
            if len(optimized_setpoints) > 0 and best_idx < len(optimized_setpoints):
                val = optimized_setpoints[best_idx]
                if val is not None:
                    item["ideal_setpoint"] = float(val)
                elif away_info.get("active") and away_info.get("temp") is not None:
                     # Gap in optimization but away is active
                     item["ideal_setpoint"] = float(away_info["temp"])
            elif away_info.get("active") and away_info.get("temp") is not None:
                 # No optimization data at all, but away is active
                 item["ideal_setpoint"] = float(away_info["temp"])
            
            forecast.append(item)
            current_dt += timedelta(minutes=15)

        to_return = {
            "forecast": forecast,
            "forecast_points": len(timestamps),
        }
        
        if away_info.get("active"):
            to_return["away"] = True
            end_s = away_info.get("end")
            if end_s:
                try:
                    # Parse as UTC/Aware
                    dt_end = dt_util.parse_datetime(end_s)
                    # Convert to local
                    if dt_end:
                         if dt_end.tzinfo is None:
                             dt_end = dt_end.replace(tzinfo=dt_util.get_time_zone(self.hass.config.time_zone))
                         to_return["away_end"] = dt_util.as_local(dt_end).isoformat()
                except Exception:
                    pass
        else:
             to_return["away"] = False

        return to_return
