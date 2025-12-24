"""Statistics tracking for HouseTemp integration.

Tracks accuracy (prediction vs actual), comfort (deviation from target),
and energy (usage and savings) over rolling and lifetime windows.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from .const import DOMAIN

_LOGGER = logging.getLogger(DOMAIN)

# Storage version for migrations
STORAGE_VERSION = 1
STORAGE_KEY = f"{DOMAIN}_stats"

# Rolling window durations
ACCURACY_WINDOW_DAYS = 7
COMFORT_WINDOW_HOURS = 24

# Default comfort tolerance
DEFAULT_SCHEDULE_TOLERANCE = 2.0  # ±°F - fallback if swing_temp not available

# Prune predictions older than this
PREDICTION_RETENTION_DAYS = 8  # Slightly longer than accuracy window


@dataclass
class PredictionRecord:
    """A prediction to be compared against actual temperature later."""
    
    timestamp: str  # ISO format, when comparison should happen
    predicted_temp: float  # What we predicted for this timestamp
    actual_temp: float | None = None  # Filled when timestamp arrives
    horizon_hours: int = 6  # How far ahead this prediction was made
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "PredictionRecord":
        return cls(**data)


@dataclass 
class ComfortSample:
    """A single comfort measurement."""
    
    timestamp: str  # ISO format
    actual_temp: float
    schedule_target: float  # User's schedule setpoint
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ComfortSample":
        # Handle legacy data that had optimized_target field
        data = {k: v for k, v in data.items() if k in ('timestamp', 'actual_temp', 'schedule_target')}
        return cls(**data)


@dataclass
class EnergySample:
    """A single energy measurement."""
    
    timestamp: str  # ISO format
    used_kwh: float  # Actual/predicted usage
    baseline_kwh: float  # What would have been used without optimization
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "EnergySample":
        return cls(**data)


@dataclass
class LifetimeComfortStats:
    """Accumulated lifetime comfort statistics."""
    
    total_samples: int = 0
    
    # Schedule target deviation
    schedule_in_range_count: int = 0
    schedule_deviation_sum: float = 0.0
    schedule_max_deviation: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "LifetimeComfortStats":
        # Handle legacy data that had optimized fields
        valid_keys = ('total_samples', 'schedule_in_range_count', 'schedule_deviation_sum', 'schedule_max_deviation')
        data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**data)


@dataclass
class LifetimeEnergyStats:
    """Accumulated lifetime energy statistics."""
    
    used_kwh: float = 0.0
    saved_kwh: float = 0.0  # baseline - used
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "LifetimeEnergyStats":
        return cls(**data)


class StatsStore:
    """Persistent storage for statistics using HA Store helper."""
    
    def __init__(self, hass: HomeAssistant, entry_id: str):
        self.hass = hass
        self.entry_id = entry_id
        self._store = Store(hass, STORAGE_VERSION, f"{STORAGE_KEY}_{entry_id}")
        
        # In-memory state
        self.predictions: list[PredictionRecord] = []
        self.comfort_samples: list[ComfortSample] = []
        self.energy_samples: list[EnergySample] = []
        self.lifetime_comfort = LifetimeComfortStats()
        self.lifetime_energy = LifetimeEnergyStats()
        self.epoch_start: datetime | None = None
        
        self._loaded = False
    
    async def async_load(self) -> None:
        """Load stored data from disk."""
        if self._loaded:
            return
            
        data = await self._store.async_load()
        if data is None:
            # First run, initialize epoch
            self.epoch_start = dt_util.now()
            self._loaded = True
            return
        
        try:
            self.predictions = [
                PredictionRecord.from_dict(p) 
                for p in data.get("predictions", [])
            ]
            self.comfort_samples = [
                ComfortSample.from_dict(s) 
                for s in data.get("comfort_samples", [])
            ]
            self.energy_samples = [
                EnergySample.from_dict(s) 
                for s in data.get("energy_samples", [])
            ]
            
            if "lifetime_comfort" in data:
                self.lifetime_comfort = LifetimeComfortStats.from_dict(
                    data["lifetime_comfort"]
                )
            if "lifetime_energy" in data:
                self.lifetime_energy = LifetimeEnergyStats.from_dict(
                    data["lifetime_energy"]
                )
            
            epoch_str = data.get("epoch_start")
            if epoch_str:
                self.epoch_start = dt_util.parse_datetime(epoch_str)
            else:
                self.epoch_start = dt_util.now()
                
        except Exception as e:
            _LOGGER.error("Failed to load stats data, starting fresh: %s", e)
            self.epoch_start = dt_util.now()
        
        self._loaded = True
    
    async def async_save(self) -> None:
        """Save current state to disk."""
        data = {
            "predictions": [p.to_dict() for p in self.predictions],
            "comfort_samples": [s.to_dict() for s in self.comfort_samples],
            "energy_samples": [s.to_dict() for s in self.energy_samples],
            "lifetime_comfort": self.lifetime_comfort.to_dict(),
            "lifetime_energy": self.lifetime_energy.to_dict(),
            "epoch_start": self.epoch_start.isoformat() if self.epoch_start else None,
        }
        await self._store.async_save(data)
    
    async def async_reset(self) -> None:
        """Reset all statistics."""
        self.predictions = []
        self.comfort_samples = []
        self.energy_samples = []
        self.lifetime_comfort = LifetimeComfortStats()
        self.lifetime_energy = LifetimeEnergyStats()
        self.epoch_start = dt_util.now()
        await self.async_save()
        _LOGGER.info("Statistics reset, new epoch started at %s", self.epoch_start)
    
    def discard_pending_predictions(self) -> int:
        """Discard all unresolved predictions (invalidated by mode/config change).
        
        Returns: Number of predictions discarded.
        """
        before = len(self.predictions)
        self.predictions = [p for p in self.predictions if p.actual_temp is not None]
        discarded = before - len(self.predictions)
        if discarded > 0:
            _LOGGER.info("Discarded %d pending predictions due to mode change", discarded)
        return discarded
    
    def prune_old_data(self) -> None:
        """Remove data older than retention windows."""
        now = dt_util.now()
        
        # Prune predictions older than retention period
        cutoff_predictions = now - timedelta(days=PREDICTION_RETENTION_DAYS)
        self.predictions = [
            p for p in self.predictions 
            if dt_util.parse_datetime(p.timestamp) > cutoff_predictions
        ]
        
        # Prune comfort samples older than 24h (we use lifetime for longer)
        cutoff_comfort = now - timedelta(hours=COMFORT_WINDOW_HOURS + 1)
        self.comfort_samples = [
            s for s in self.comfort_samples
            if dt_util.parse_datetime(s.timestamp) > cutoff_comfort
        ]
        
        # Prune energy samples older than 24h
        self.energy_samples = [
            s for s in self.energy_samples
            if dt_util.parse_datetime(s.timestamp) > cutoff_comfort
        ]
    
    def _validate_float(self, val: Any) -> float | None:
        """Helper: Ensure value is a valid float."""
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def record_prediction(
        self, 
        target_timestamp: datetime, 
        predicted_temp: float,
        horizon_hours: int = 6
    ) -> None:
        """Record a prediction for later comparison."""
        f_pred = self._validate_float(predicted_temp)
        if f_pred is None:
            return

        # Rate Limit: Don't record if we already have a prediction for close to this time
        # This prevents spamming the DB if optimization runs frequently.
        target_dt = dt_util.parse_datetime(target_timestamp.isoformat())
        if self.predictions:
            last_dt = dt_util.parse_datetime(self.predictions[-1].timestamp)
            if last_dt and abs((target_dt - last_dt).total_seconds()) < 1800:  # 30 minutes
                # Update existing record with latest prediction
                self.predictions[-1].predicted_temp = f_pred
                self.predictions[-1].horizon_hours = horizon_hours
                # Note: We keep original timestamp to maintain 30m grid, 
                # or update it? Updating it shifts the grid. 
                # Better to just update the value.
                return

        record = PredictionRecord(
            timestamp=target_timestamp.isoformat(),
            predicted_temp=f_pred,
            horizon_hours=horizon_hours,
        )
        self.predictions.append(record)
    
    def resolve_predictions(self, current_time: datetime, actual_temp: float) -> int:
        """Fill in actual temperatures for predictions whose timestamps have passed.
        
        Returns: Number of predictions resolved.
        """
        f_actual = self._validate_float(actual_temp)
        if f_actual is None:
            return 0
            
        resolved = 0
        for pred in self.predictions:
            if pred.actual_temp is not None:
                continue
            
            pred_time = dt_util.parse_datetime(pred.timestamp)
            if pred_time and pred_time <= current_time:
                pred.actual_temp = f_actual
                resolved += 1
        
        return resolved
    
    def record_comfort_sample(
        self,
        actual_temp: float,
        schedule_target: float,
        schedule_tolerance: float = DEFAULT_SCHEDULE_TOLERANCE,
    ) -> None:
        """Record a comfort measurement and update lifetime stats."""
        # Validate inputs to avoid math errors
        f_actual = self._validate_float(actual_temp)
        f_sched = self._validate_float(schedule_target)
        
        if f_actual is None or f_sched is None:
            return
            
        now = dt_util.now()
        
        sample = ComfortSample(
            timestamp=now.isoformat(),
            actual_temp=f_actual,
            schedule_target=f_sched,
        )
        self.comfort_samples.append(sample)
        
        # Update lifetime stats
        self.lifetime_comfort.total_samples += 1
        
        # Schedule target deviation
        schedule_dev = abs(f_actual - f_sched)
        self.lifetime_comfort.schedule_deviation_sum += schedule_dev
        self.lifetime_comfort.schedule_max_deviation = max(
            self.lifetime_comfort.schedule_max_deviation, schedule_dev
        )
        if schedule_dev <= schedule_tolerance:
            self.lifetime_comfort.schedule_in_range_count += 1
    
    def record_energy_sample(
        self,
        used_kwh: float,
        baseline_kwh: float,
    ) -> None:
        """Record an energy measurement and update lifetime stats."""
        f_used = self._validate_float(used_kwh)
        f_base = self._validate_float(baseline_kwh)
        
        if f_used is None or f_base is None:
            return

        now = dt_util.now()
        
        sample = EnergySample(
            timestamp=now.isoformat(),
            used_kwh=f_used,
            baseline_kwh=f_base,
        )
        self.energy_samples.append(sample)
        
        # Update lifetime stats
        self.lifetime_energy.used_kwh += f_used
        self.lifetime_energy.saved_kwh += (f_base - f_used)


class StatsCalculator:
    """Computes rolling statistics from stored data."""
    
    @staticmethod
    def compute_accuracy_7d(
        predictions: list[PredictionRecord],
        horizon_hours: int | None = None,
    ) -> dict[str, float | None]:
        """Compute accuracy stats for predictions in the last 7 days.
        
        Args:
            predictions: List of prediction records
            horizon_hours: If set, only consider predictions with this horizon
            
        Returns:
            Dict with mae, bias, and sample count. Values are None if no data.
        """
        now = dt_util.now()
        cutoff = now - timedelta(days=ACCURACY_WINDOW_DAYS)
        
        errors = []
        for pred in predictions:
            if pred.actual_temp is None:
                continue
            
            pred_time = dt_util.parse_datetime(pred.timestamp)
            if not pred_time or pred_time < cutoff:
                continue
            
            if horizon_hours is not None and pred.horizon_hours != horizon_hours:
                continue
            
            error = pred.predicted_temp - pred.actual_temp
            errors.append(error)
        
        if not errors:
            return {"mae": None, "bias": None, "count": 0}
        
        mae = sum(abs(e) for e in errors) / len(errors)
        bias = sum(errors) / len(errors)
        
        return {
            "mae": round(mae, 2),
            "bias": round(bias, 2),
            "count": len(errors),
        }
    
    @staticmethod
    def compute_comfort_24h(
        samples: list[ComfortSample],
        schedule_tolerance: float = DEFAULT_SCHEDULE_TOLERANCE,
    ) -> dict[str, Any]:
        """Compute comfort stats for the last 24 hours."""
        now = dt_util.now()
        cutoff = now - timedelta(hours=COMFORT_WINDOW_HOURS)
        
        recent = [
            s for s in samples
            if dt_util.parse_datetime(s.timestamp) > cutoff
        ]
        
        if not recent:
            return {
                "time_in_range": None,
                "mean_deviation": None,
                "max_deviation": None,
                "count": 0,
            }
        
        # Schedule target stats
        schedule_devs = [abs(s.actual_temp - s.schedule_target) for s in recent]
        schedule_in_range = sum(1 for d in schedule_devs if d <= schedule_tolerance)
        
        return {
            "time_in_range": round(100 * schedule_in_range / len(schedule_devs), 1),
            "mean_deviation": round(sum(schedule_devs) / len(schedule_devs), 2),
            "max_deviation": round(max(schedule_devs), 2),
            "count": len(schedule_devs),
        }
    
    @staticmethod
    def compute_comfort_lifetime(
        stats: LifetimeComfortStats,
    ) -> dict[str, Any]:
        """Format lifetime comfort stats for display."""
        if stats.total_samples == 0:
            return {
                "time_in_range": None,
                "mean_deviation": None,
                "max_deviation": None,
            }
        
        return {
            "time_in_range": round(100 * stats.schedule_in_range_count / stats.total_samples, 1),
            "mean_deviation": round(stats.schedule_deviation_sum / stats.total_samples, 2),
            "max_deviation": round(stats.schedule_max_deviation, 2),
        }
    
    @staticmethod
    def compute_energy_24h(samples: list[EnergySample]) -> dict[str, float | None]:
        """Compute energy stats for the last 24 hours."""
        now = dt_util.now()
        cutoff = now - timedelta(hours=COMFORT_WINDOW_HOURS)  # Same 24h window
        
        recent = [
            s for s in samples
            if dt_util.parse_datetime(s.timestamp) > cutoff
        ]
        
        if not recent:
            return {"used_kwh": None, "saved_kwh": None}
        
        used = sum(s.used_kwh for s in recent)
        saved = sum(s.baseline_kwh - s.used_kwh for s in recent)
        
        return {
            "used_kwh": round(used, 2),
            "saved_kwh": round(saved, 2),
        }
    
    @staticmethod
    def compute_energy_lifetime(stats: LifetimeEnergyStats) -> dict[str, float]:
        """Format lifetime energy stats for display."""
        return {
            "used_kwh": round(stats.used_kwh, 2),
            "saved_kwh": round(stats.saved_kwh, 2),
        }
