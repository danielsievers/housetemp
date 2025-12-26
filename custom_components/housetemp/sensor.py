"""Sensor platform for House Temp Prediction."""
from __future__ import annotations
import math
from datetime import datetime, timedelta
import numpy as np
from homeassistant.util import dt as dt_util

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfTemperature, UnitOfEnergy, PERCENTAGE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import HouseTempCoordinator
from .statistics import StatsStore, StatsCalculator

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the sensor platform."""
    coordinator = hass.data[DOMAIN][entry.entry_id]
    stats_store = hass.data[DOMAIN].get(f"{entry.entry_id}_stats")
    
    entities = [HouseTempPredictionSensor(coordinator, entry)]
    
    # Add statistics sensors if stats store is available
    if stats_store:
        entities.extend([
            HouseTempAccuracySensor(coordinator, entry, stats_store),
            HouseTempComfortSensor(coordinator, entry, stats_store),
            HouseTempEnergySensor(coordinator, entry, stats_store),
        ])
    
    async_add_entities(entities)

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
                f_val = float(val)
                if math.isfinite(f_val):
                    return int(round(f_val))
        
        # Fallback: If optimization missing but Away is active, show Away Temp
        if away_info.get("active") and away_info.get("temp") is not None:
             f_val = float(away_info["temp"])
             if math.isfinite(f_val):
                 return int(round(f_val))
            
        setpoints = self.coordinator.data.get("setpoint")
        if setpoints is not None and len(setpoints) > 0:
            f_val = float(setpoints[0])
            if math.isfinite(f_val):
                return int(round(f_val))
            
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
        energy_steps = data.get("energy_kwh_steps", [])

        if timestamps is None or len(timestamps) == 0:
            return {}

        # Calculate Hourly Energy
        energy_per_hour = []
        if energy_steps is not None and len(energy_steps) == len(timestamps):
             hourly_map = {}
             for ts, kwh in zip(timestamps, energy_steps):
                 if kwh is None or not math.isfinite(kwh):
                     continue
                 
                 # Round down to hour
                 # ts is a datetime object (likely offset-aware from coordinator)
                 hour_key = ts.replace(minute=0, second=0, microsecond=0)
                 hourly_map[hour_key] = hourly_map.get(hour_key, 0.0) + kwh
             
             # Convert to list
             for t in sorted(hourly_map.keys()):
                 energy_per_hour.append({
                     "datetime": t.isoformat(),
                     "kwh": round(hourly_map[t], 3)
                 })


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
        
        import bisect
        
        # Determine extra away info from data
        away_info = data.get("away_info", {})
        away_end_dt = None
        if away_info.get("active") and away_info.get("end"):
            try:
                # Parse available away_end. It usually comes as ISO string
                dt_end = dt_util.parse_datetime(away_info["end"])
                if dt_end:
                     if dt_end.tzinfo is None:
                         dt_end = dt_end.replace(tzinfo=dt_util.get_time_zone(self.hass.config.time_zone))
                     away_end_dt = dt_end
            except Exception:
                pass
        
        forecast = []
        n_timestamps = len(timestamps)
        
        while current_dt <= end_dt:
            # Find nearest data point using binary search
            idx = bisect.bisect_left(timestamps, current_dt)
            
            best_idx = 0
            if idx == 0:
                best_idx = 0
            elif idx >= n_timestamps:
                best_idx = n_timestamps - 1
            else:
                # Check idx-1 and idx to see which is closer
                dist_left = abs((timestamps[idx-1] - current_dt).total_seconds())
                dist_right = abs((timestamps[idx] - current_dt).total_seconds())
                if dist_left < dist_right:
                    best_idx = idx - 1
                else:
                    best_idx = idx
            
            local_dt = dt_util.as_local(current_dt)
            # Get energy for this point
            energy_steps = data.get("energy_kwh_steps")
            energy_val = float(energy_steps[best_idx]) if energy_steps is not None and best_idx < len(energy_steps) else None

            item = {
                "datetime": local_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "temperature": float(round(temps[best_idx], 1)) if best_idx < len(temps) else None,
                "target_temp": float(setpoints[best_idx]) if best_idx < len(setpoints) else None,
                "energy_kwh": round(energy_val, 3) if energy_val is not None else None,
            }
            
            # Check if this specific point is within the generic "Away" window
            is_point_away = False
            if away_end_dt:
                if current_dt < away_end_dt:
                    is_point_away = True
            elif away_info.get("active"):
                # Fallback if no end time (shouldn't happen usually)
                is_point_away = True

            # ideal_setpoint only if optimization was run AND covers this time slot
            # Fallback to Away Temp if active and missing optimization
            
            if len(optimized_setpoints) > 0 and best_idx < len(optimized_setpoints):
                val = optimized_setpoints[best_idx]
                if val is not None and math.isfinite(val):
                    item["ideal_setpoint"] = float(val)
                elif is_point_away and away_info.get("temp") is not None:
                     # Gap in optimization but away is active
                     val = float(away_info["temp"])
                     if math.isfinite(val):
                         item["ideal_setpoint"] = val
            elif is_point_away and away_info.get("temp") is not None:
                 # No optimization data at all, but away is active
                 val = float(away_info["temp"])
                 if math.isfinite(val):
                     item["ideal_setpoint"] = val
            
            # Sanitization for temperature and target_temp
            if item["temperature"] is not None and not math.isfinite(item["temperature"]):
                item["temperature"] = None
            if item["target_temp"] is not None and not math.isfinite(item["target_temp"]):
                item["target_temp"] = None

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
             
        # Add Energy Stats
        kwh = data.get("energy_kwh")
        opt_kwh = data.get("optimized_energy_kwh")
        
        # Add HVAC Mode
        from .const import CONF_HVAC_MODE
        to_return["hvac_mode"] = self.coordinator.config_entry.options.get(CONF_HVAC_MODE, "heat")
        
        if kwh is not None:
            f_kwh = float(kwh)
            if math.isfinite(f_kwh):
                to_return["energy_kwh"] = round(f_kwh, 2)
            
        if opt_kwh is not None:
            f_opt = float(opt_kwh)
            if math.isfinite(f_opt):
                to_return["optimized_energy_kwh"] = round(f_opt, 2)
            
        if kwh is not None and opt_kwh is not None:
            # Re-fetch floats in case checked above
            f_kwh = float(kwh)
            f_opt = float(opt_kwh)
            if math.isfinite(f_kwh) and math.isfinite(f_opt):
                savings = f_kwh - f_opt
                to_return["savings_kwh"] = round(float(savings), 2)
                
        # Optimization Status
        opt_status = data.get("optimization_status")
        if opt_status:
            to_return["optimization_status"] = opt_status.get("message", "Unknown")
            to_return["optimization_cost"] = opt_status.get("cost", 0.0)
            to_return["optimization_converged"] = opt_status.get("success", False)
            
            # True Off Recommendation (current timestep)
            off_rec = opt_status.get("off_recommended", [])
            if off_rec and len(off_rec) > 0:
                to_return["off_recommended"] = bool(off_rec[0])

        if energy_per_hour:
            to_return["energy_per_hour"] = energy_per_hour

        # Expose Detailed Energy Metrics (Continuous vs Discrete)
        if "energy_metrics" in data:
            to_return["energy_metrics"] = data["energy_metrics"]

        return to_return


class HouseTempAccuracySensor(CoordinatorEntity, SensorEntity):
    """Sensor for prediction accuracy statistics (7-day rolling)."""

    _attr_has_entity_name = True
    _attr_name = "Prediction Accuracy"
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_native_unit_of_measurement = UnitOfTemperature.FAHRENHEIT
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:target"

    def __init__(
        self, 
        coordinator: HouseTempCoordinator, 
        entry: ConfigEntry,
        stats_store: StatsStore,
    ):
        """Initialize the accuracy sensor."""
        super().__init__(coordinator)
        self._entry = entry
        self._stats_store = stats_store
        self._attr_unique_id = f"{entry.entry_id}_accuracy"

    @property
    def native_value(self):
        """Return 6-hour ahead MAE as the main state."""
        stats = StatsCalculator.compute_accuracy_7d(
            self._stats_store.predictions,
            horizon_hours=6,
        )
        return stats.get("mae")

    @property
    def extra_state_attributes(self):
        """Return detailed accuracy statistics."""
        # 6-hour ahead stats
        stats_6h = StatsCalculator.compute_accuracy_7d(
            self._stats_store.predictions,
            horizon_hours=6,
        )
        
        # Full forecast stats (all horizons)
        stats_all = StatsCalculator.compute_accuracy_7d(
            self._stats_store.predictions,
            horizon_hours=None,
        )
        
        attrs = {
            "window_days": 7,
            "mae_6h": stats_6h.get("mae"),
            "bias_6h": stats_6h.get("bias"),
            "samples_6h": stats_6h.get("count", 0),
            "mae_full_forecast": stats_all.get("mae"),
            "bias_full_forecast": stats_all.get("bias"),
            "samples_full": stats_all.get("count", 0),
        }
        
        if self._stats_store.epoch_start:
            attrs["epoch_start"] = self._stats_store.epoch_start.isoformat()
        
        return attrs


class HouseTempComfortSensor(CoordinatorEntity, SensorEntity):
    """Sensor for comfort statistics (24h rolling + lifetime)."""

    _attr_has_entity_name = True
    _attr_name = "Comfort Score"
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:home-thermometer"

    def __init__(
        self, 
        coordinator: HouseTempCoordinator, 
        entry: ConfigEntry,
        stats_store: StatsStore,
    ):
        """Initialize the comfort sensor."""
        super().__init__(coordinator)
        self._entry = entry
        self._stats_store = stats_store
        self._attr_unique_id = f"{entry.entry_id}_comfort"

    @property
    def native_value(self):
        """Return time-in-range (vs schedule target) as the main state."""
        stats_24h = StatsCalculator.compute_comfort_24h(
            self._stats_store.comfort_samples,
        )
        val = stats_24h.get("time_in_range")
        return round(val, 1) if val is not None else None

    @property
    def extra_state_attributes(self):
        """Return detailed comfort statistics."""
        stats_24h = StatsCalculator.compute_comfort_24h(
            self._stats_store.comfort_samples,
        )
        stats_lifetime = StatsCalculator.compute_comfort_lifetime(
            self._stats_store.lifetime_comfort,
        )
        
        attrs = {
            # 24h rolling - deviation from schedule target
            "time_in_range_24h": stats_24h.get("time_in_range"),
            "mean_deviation_24h": stats_24h.get("mean_deviation"),
            "max_deviation_24h": stats_24h.get("max_deviation"),
            
            # Lifetime - deviation from schedule target
            "time_in_range_lifetime": stats_lifetime.get("time_in_range"),
            "mean_deviation_lifetime": stats_lifetime.get("mean_deviation"),
            "max_deviation_lifetime": stats_lifetime.get("max_deviation"),
            
            "total_samples_lifetime": self._stats_store.lifetime_comfort.total_samples,
        }
        
        if self._stats_store.epoch_start:
            attrs["epoch_start"] = self._stats_store.epoch_start.isoformat()
        
        return attrs


class HouseTempEnergySensor(CoordinatorEntity, SensorEntity):
    """Sensor for energy statistics (24h rolling + lifetime)."""

    _attr_has_entity_name = True
    _attr_name = "Energy Savings"
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.TOTAL
    _attr_icon = "mdi:lightning-bolt"

    def __init__(
        self, 
        coordinator: HouseTempCoordinator, 
        entry: ConfigEntry,
        stats_store: StatsStore,
    ):
        """Initialize the energy sensor."""
        super().__init__(coordinator)
        self._entry = entry
        self._stats_store = stats_store
        self._attr_unique_id = f"{entry.entry_id}_energy_stats"

    @property
    def native_value(self):
        """Return lifetime energy saved as the main state."""
        stats = StatsCalculator.compute_energy_lifetime(
            self._stats_store.lifetime_energy,
        )
        return stats.get("saved_kwh")

    @property
    def extra_state_attributes(self):
        """Return detailed energy statistics."""
        stats_24h = StatsCalculator.compute_energy_24h(
            self._stats_store.energy_samples,
        )
        stats_lifetime = StatsCalculator.compute_energy_lifetime(
            self._stats_store.lifetime_energy,
        )
        
        attrs = {
            # 24h rolling
            "used_kwh_24h": stats_24h.get("used_kwh"),
            "saved_kwh_24h": stats_24h.get("saved_kwh"),
            
            # Lifetime
            "used_kwh_lifetime": stats_lifetime.get("used_kwh"),
            "saved_kwh_lifetime": stats_lifetime.get("saved_kwh"),
        }
        
        if self._stats_store.epoch_start:
            attrs["epoch_start"] = self._stats_store.epoch_start.isoformat()
        
        return attrs
