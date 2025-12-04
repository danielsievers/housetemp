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
        # State is the predicted temp at the end of the simulation
        if not self.coordinator.data:
            return None
        
        predicted_temps = self.coordinator.data.get("predicted_temp")
        if predicted_temps is not None and len(predicted_temps) > 0:
            return round(predicted_temps[-1], 1)
        return None

    @property
    def extra_state_attributes(self):
        """Return the state attributes."""
        if not self.coordinator.data:
            return {}

        data = self.coordinator.data
        timestamps = data.get("timestamps", [])
        temps = data.get("predicted_temp", [])
        hvac = data.get("hvac_state", [])
        setpoints = data.get("setpoint", [])
        solar = data.get("solar", [])
        outdoor = data.get("outdoor", [])

        forecast = []
        total_energy_kwh = 0.0
        
        # Calculate energy for each step
        # We need to know the power consumption logic.
        # It's inside run_model but not returned directly as array.
        # We can estimate it or modify run_model to return it.
        # For now, we don't have energy array.
        # We can just output the other fields.
        
        for i in range(len(timestamps)):
            item = {
                "datetime": timestamps[i].isoformat(),
                "temperature": round(temps[i], 1) if i < len(temps) else None,
                "hvac_state": int(hvac[i]) if i < len(hvac) else None,
                "setpoint": float(setpoints[i]) if i < len(setpoints) else None,
                "solar_kw": round(solar[i], 2) if i < len(solar) else None,
                "outdoor_temp": round(outdoor[i], 1) if i < len(outdoor) else None,
            }
            forecast.append(item)

        return {
            "forecast": forecast,
            # "total_energy_kwh": ... # Need to extract from model or re-calculate
        }
