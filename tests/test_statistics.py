"""Tests for the statistics module."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from homeassistant.util import dt as dt_util

from custom_components.housetemp.statistics import (
    PredictionRecord,
    ComfortSample,
    EnergySample,
    LifetimeComfortStats,
    LifetimeEnergyStats,
    StatsStore,
    StatsCalculator,
    DEFAULT_SCHEDULE_TOLERANCE,
    DEFAULT_OPTIMIZED_TOLERANCE,
    ACCURACY_WINDOW_DAYS,
)


class TestPredictionRecord:
    """Tests for PredictionRecord dataclass."""
    
    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        now = datetime.now()
        record = PredictionRecord(
            timestamp=now.isoformat(),
            predicted_temp=70.5,
            actual_temp=69.8,
            horizon_hours=6,
        )
        
        data = record.to_dict()
        restored = PredictionRecord.from_dict(data)
        
        assert restored.timestamp == record.timestamp
        assert restored.predicted_temp == record.predicted_temp
        assert restored.actual_temp == record.actual_temp
        assert restored.horizon_hours == record.horizon_hours


class TestStatsCalculator:
    """Tests for StatsCalculator static methods."""
    
    def test_accuracy_7d_empty_list(self):
        """MAE with no data returns None."""
        result = StatsCalculator.compute_accuracy_7d([])
        
        assert result["mae"] is None
        assert result["bias"] is None
        assert result["count"] == 0
    
    def test_accuracy_7d_no_resolved_predictions(self):
        """Predictions without actual temps are ignored."""
        now = dt_util.now()
        predictions = [
            PredictionRecord(
                timestamp=(now - timedelta(hours=1)).isoformat(),
                predicted_temp=70.0,
                actual_temp=None,  # Not resolved
                horizon_hours=6,
            )
        ]
        
        result = StatsCalculator.compute_accuracy_7d(predictions)
        
        assert result["mae"] is None
        assert result["count"] == 0
    
    def test_accuracy_7d_basic_mae(self):
        """Basic MAE calculation."""
        now = dt_util.now()
        predictions = [
            PredictionRecord(
                timestamp=(now - timedelta(hours=1)).isoformat(),
                predicted_temp=70.0,
                actual_temp=69.0,  # Error = +1.0
                horizon_hours=6,
            ),
            PredictionRecord(
                timestamp=(now - timedelta(hours=2)).isoformat(),
                predicted_temp=68.0,
                actual_temp=70.0,  # Error = -2.0
                horizon_hours=6,
            ),
        ]
        
        result = StatsCalculator.compute_accuracy_7d(predictions, horizon_hours=6)
        
        # MAE = (1 + 2) / 2 = 1.5
        assert result["mae"] == 1.5
        # Bias = (1 + (-2)) / 2 = -0.5
        assert result["bias"] == -0.5
        assert result["count"] == 2
    
    def test_accuracy_7d_filters_by_horizon(self):
        """Only predictions with matching horizon are included."""
        now = dt_util.now()
        predictions = [
            PredictionRecord(
                timestamp=(now - timedelta(hours=1)).isoformat(),
                predicted_temp=70.0,
                actual_temp=69.0,
                horizon_hours=6,  # Match
            ),
            PredictionRecord(
                timestamp=(now - timedelta(hours=2)).isoformat(),
                predicted_temp=68.0,
                actual_temp=70.0,
                horizon_hours=12,  # No match
            ),
        ]
        
        result = StatsCalculator.compute_accuracy_7d(predictions, horizon_hours=6)
        
        assert result["count"] == 1
        assert result["mae"] == 1.0
    
    def test_accuracy_7d_excludes_old_predictions(self):
        """Predictions older than 7 days are excluded."""
        now = dt_util.now()
        predictions = [
            PredictionRecord(
                timestamp=(now - timedelta(days=1)).isoformat(),
                predicted_temp=70.0,
                actual_temp=69.0,
                horizon_hours=6,
            ),
            PredictionRecord(
                timestamp=(now - timedelta(days=10)).isoformat(),  # Too old
                predicted_temp=68.0,
                actual_temp=70.0,
                horizon_hours=6,
            ),
        ]
        
        result = StatsCalculator.compute_accuracy_7d(predictions, horizon_hours=6)
        
        assert result["count"] == 1  # Only recent one
    
    def test_accuracy_7d_bias_positive_means_hot(self):
        """Positive bias means model predicts higher than actual (runs hot)."""
        now = dt_util.now()
        predictions = [
            PredictionRecord(
                timestamp=(now - timedelta(hours=1)).isoformat(),
                predicted_temp=72.0,
                actual_temp=70.0,  # Predicted 2° too high
                horizon_hours=6,
            ),
        ]
        
        result = StatsCalculator.compute_accuracy_7d(predictions, horizon_hours=6)
        
        assert result["bias"] == 2.0  # Positive = runs hot
    
    def test_comfort_24h_empty_samples(self):
        """Empty samples return None values."""
        result = StatsCalculator.compute_comfort_24h([])
        
        assert result["schedule"]["time_in_range"] is None
        assert result["schedule"]["count"] == 0
        assert result["optimized"]["time_in_range"] is None
    
    def test_comfort_24h_schedule_deviation(self):
        """Comfort stats correctly calculate schedule deviation."""
        now = dt_util.now()
        samples = [
            ComfortSample(
                timestamp=(now - timedelta(hours=1)).isoformat(),
                actual_temp=70.0,
                schedule_target=70.0,  # Perfect
                optimized_target=None,
            ),
            ComfortSample(
                timestamp=(now - timedelta(hours=2)).isoformat(),
                actual_temp=68.0,
                schedule_target=70.0,  # Deviation = 2.0
                optimized_target=None,
            ),
        ]
        
        result = StatsCalculator.compute_comfort_24h(
            samples, schedule_tolerance=1.0, optimized_tolerance=0.5
        )
        
        # 1 of 2 within 1.0°F tolerance
        assert result["schedule"]["time_in_range"] == 50.0
        # Mean deviation = (0 + 2) / 2 = 1.0
        assert result["schedule"]["mean_deviation"] == 1.0
        assert result["schedule"]["max_deviation"] == 2.0
        assert result["schedule"]["count"] == 2
    
    def test_comfort_24h_optimized_deviation(self):
        """Optimized target deviation uses tighter tolerance."""
        now = dt_util.now()
        samples = [
            ComfortSample(
                timestamp=(now - timedelta(hours=1)).isoformat(),
                actual_temp=70.0,
                schedule_target=70.0,
                optimized_target=70.3,  # Deviation = 0.3 (within 0.5 tolerance)
            ),
            ComfortSample(
                timestamp=(now - timedelta(hours=2)).isoformat(),
                actual_temp=70.0,
                schedule_target=70.0,
                optimized_target=71.0,  # Deviation = 1.0 (outside 0.5 tolerance)
            ),
        ]
        
        result = StatsCalculator.compute_comfort_24h(
            samples, schedule_tolerance=2.0, optimized_tolerance=0.5
        )
        
        # Optimized: 1 of 2 within 0.5°F tolerance
        assert result["optimized"]["time_in_range"] == 50.0
        assert result["optimized"]["count"] == 2
        
        # Schedule: both within 2.0°F tolerance
        assert result["schedule"]["time_in_range"] == 100.0
    
    def test_comfort_24h_excludes_old_samples(self):
        """Only samples from last 24h are included."""
        now = dt_util.now()
        samples = [
            ComfortSample(
                timestamp=(now - timedelta(hours=1)).isoformat(),
                actual_temp=70.0,
                schedule_target=70.0,
                optimized_target=None,
            ),
            ComfortSample(
                timestamp=(now - timedelta(hours=30)).isoformat(),  # Too old
                actual_temp=60.0,
                schedule_target=70.0,
                optimized_target=None,
            ),
        ]
        
        result = StatsCalculator.compute_comfort_24h(samples)
        
        assert result["schedule"]["count"] == 1
    
    def test_comfort_lifetime_empty(self):
        """Empty lifetime stats return None."""
        stats = LifetimeComfortStats()
        
        result = StatsCalculator.compute_comfort_lifetime(stats)
        
        assert result["schedule"]["time_in_range"] is None
    
    def test_comfort_lifetime_computation(self):
        """Lifetime comfort stats are correctly formatted."""
        stats = LifetimeComfortStats(
            total_samples=100,
            schedule_in_range_count=80,
            schedule_deviation_sum=50.0,
            schedule_max_deviation=3.0,
            optimized_sample_count=50,
            optimized_in_range_count=45,
            optimized_deviation_sum=10.0,
            optimized_max_deviation=1.5,
        )
        
        result = StatsCalculator.compute_comfort_lifetime(stats)
        
        assert result["schedule"]["time_in_range"] == 80.0  # 80%
        assert result["schedule"]["mean_deviation"] == 0.5  # 50/100
        assert result["schedule"]["max_deviation"] == 3.0
        
        assert result["optimized"]["time_in_range"] == 90.0  # 45/50 = 90%
        assert result["optimized"]["mean_deviation"] == 0.2  # 10/50
    
    def test_energy_24h_empty(self):
        """Empty energy samples return None."""
        result = StatsCalculator.compute_energy_24h([])
        
        assert result["used_kwh"] is None
        assert result["saved_kwh"] is None
    
    def test_energy_24h_summation(self):
        """Energy stats sum correctly."""
        now = dt_util.now()
        samples = [
            EnergySample(
                timestamp=(now - timedelta(hours=1)).isoformat(),
                used_kwh=1.5,
                baseline_kwh=2.0,
            ),
            EnergySample(
                timestamp=(now - timedelta(hours=2)).isoformat(),
                used_kwh=2.0,
                baseline_kwh=2.5,
            ),
        ]
        
        result = StatsCalculator.compute_energy_24h(samples)
        
        assert result["used_kwh"] == 3.5  # 1.5 + 2.0
        assert result["saved_kwh"] == 1.0  # (2.0-1.5) + (2.5-2.0)
    
    def test_energy_lifetime_computation(self):
        """Lifetime energy stats are formatted correctly."""
        stats = LifetimeEnergyStats(used_kwh=100.123, saved_kwh=25.456)
        
        result = StatsCalculator.compute_energy_lifetime(stats)
        
        assert result["used_kwh"] == 100.12
        assert result["saved_kwh"] == 25.46


class TestStatsStore:
    """Tests for StatsStore persistence and recording."""
    
    @pytest.fixture
    def mock_hass(self):
        """Create a mock Home Assistant instance."""
        hass = MagicMock()
        return hass
    
    @pytest.fixture
    def stats_store(self, mock_hass):
        """Create a StatsStore instance with mocked storage."""
        with patch("custom_components.housetemp.statistics.Store") as MockStore:
            mock_store_instance = MagicMock()
            mock_store_instance.async_load = AsyncMock(return_value=None)
            mock_store_instance.async_save = AsyncMock()
            MockStore.return_value = mock_store_instance
            
            store = StatsStore(mock_hass, "test_entry_id")
            store._store = mock_store_instance
            return store
    
    @pytest.mark.asyncio
    async def test_async_load_first_run(self, stats_store):
        """First run initializes epoch."""
        await stats_store.async_load()
        
        assert stats_store.epoch_start is not None
        assert stats_store._loaded is True
    
    def test_record_prediction(self, stats_store):
        """Predictions are recorded correctly."""
        target_time = dt_util.now() + timedelta(hours=6)
        
        stats_store.record_prediction(
            target_timestamp=target_time,
            predicted_temp=70.5,
            horizon_hours=6,
        )
        
        assert len(stats_store.predictions) == 1
        assert stats_store.predictions[0].predicted_temp == 70.5
        assert stats_store.predictions[0].horizon_hours == 6
        assert stats_store.predictions[0].actual_temp is None
    
    def test_resolve_predictions(self, stats_store):
        """Predictions are resolved when timestamp passes."""
        past_time = dt_util.now() - timedelta(hours=1)
        future_time = dt_util.now() + timedelta(hours=1)
        
        stats_store.predictions = [
            PredictionRecord(
                timestamp=past_time.isoformat(),
                predicted_temp=70.0,
                actual_temp=None,
            ),
            PredictionRecord(
                timestamp=future_time.isoformat(),
                predicted_temp=72.0,
                actual_temp=None,
            ),
        ]
        
        resolved = stats_store.resolve_predictions(dt_util.now(), actual_temp=69.5)
        
        assert resolved == 1
        assert stats_store.predictions[0].actual_temp == 69.5
        assert stats_store.predictions[1].actual_temp is None  # Still future
    
    def test_record_comfort_sample(self, stats_store):
        """Comfort samples update lifetime stats."""
        stats_store.record_comfort_sample(
            actual_temp=70.0,
            schedule_target=70.0,
            optimized_target=70.2,
            schedule_tolerance=1.0,
            optimized_tolerance=0.5,
        )
        
        assert len(stats_store.comfort_samples) == 1
        assert stats_store.lifetime_comfort.total_samples == 1
        assert stats_store.lifetime_comfort.schedule_in_range_count == 1  # Within 1.0
        assert stats_store.lifetime_comfort.optimized_in_range_count == 1  # Within 0.5
    
    def test_record_comfort_out_of_range(self, stats_store):
        """Out of range samples don't increment in_range counts."""
        stats_store.record_comfort_sample(
            actual_temp=72.5,
            schedule_target=70.0,  # Deviation = 2.5, outside 1.0 tolerance
            optimized_target=70.0,  # Deviation = 2.5, outside 0.5 tolerance
            schedule_tolerance=1.0,
            optimized_tolerance=0.5,
        )
        
        assert stats_store.lifetime_comfort.total_samples == 1
        assert stats_store.lifetime_comfort.schedule_in_range_count == 0
        assert stats_store.lifetime_comfort.optimized_in_range_count == 0
        assert stats_store.lifetime_comfort.schedule_max_deviation == 2.5
    
    def test_record_energy_sample(self, stats_store):
        """Energy samples update lifetime stats."""
        stats_store.record_energy_sample(used_kwh=5.0, baseline_kwh=7.0)
        
        assert len(stats_store.energy_samples) == 1
        assert stats_store.lifetime_energy.used_kwh == 5.0
        assert stats_store.lifetime_energy.saved_kwh == 2.0  # 7 - 5
    
    @pytest.mark.asyncio
    async def test_async_reset(self, stats_store):
        """Reset clears all data and starts new epoch."""
        # Add some data
        stats_store.predictions.append(
            PredictionRecord(timestamp=dt_util.now().isoformat(), predicted_temp=70.0)
        )
        stats_store.lifetime_comfort.total_samples = 100
        old_epoch = stats_store.epoch_start
        
        await stats_store.async_reset()
        
        assert len(stats_store.predictions) == 0
        assert stats_store.lifetime_comfort.total_samples == 0
        assert stats_store.epoch_start != old_epoch
    
    def test_prune_old_data(self, stats_store):
        """Old data is pruned correctly."""
        now = dt_util.now()
        
        stats_store.predictions = [
            PredictionRecord(
                timestamp=(now - timedelta(days=10)).isoformat(),  # Too old
                predicted_temp=70.0,
            ),
            PredictionRecord(
                timestamp=(now - timedelta(days=1)).isoformat(),  # Recent
                predicted_temp=72.0,
            ),
        ]
        
        stats_store.comfort_samples = [
            ComfortSample(
                timestamp=(now - timedelta(hours=30)).isoformat(),  # Too old
                actual_temp=70.0,
                schedule_target=70.0,
                optimized_target=None,
            ),
            ComfortSample(
                timestamp=(now - timedelta(hours=1)).isoformat(),  # Recent
                actual_temp=72.0,
                schedule_target=70.0,
                optimized_target=None,
            ),
        ]
        
        stats_store.prune_old_data()
        
        assert len(stats_store.predictions) == 1
        assert stats_store.predictions[0].predicted_temp == 72.0
        assert len(stats_store.comfort_samples) == 1
    
    def test_discard_pending_predictions(self, stats_store):
        """Pending predictions are discarded, resolved ones are kept."""
        now = dt_util.now()
        
        stats_store.predictions = [
            PredictionRecord(
                timestamp=(now - timedelta(hours=1)).isoformat(),
                predicted_temp=70.0,
                actual_temp=69.5,  # Resolved
            ),
            PredictionRecord(
                timestamp=(now + timedelta(hours=5)).isoformat(),
                predicted_temp=72.0,
                actual_temp=None,  # Pending
            ),
            PredictionRecord(
                timestamp=(now + timedelta(hours=6)).isoformat(),
                predicted_temp=73.0,
                actual_temp=None,  # Pending
            ),
        ]
        
        discarded = stats_store.discard_pending_predictions()
        
        assert discarded == 2
        assert len(stats_store.predictions) == 1
        assert stats_store.predictions[0].actual_temp == 69.5  # Only resolved remains

