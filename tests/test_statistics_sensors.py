"""Tests for the statistics sensors (HouseTempAccuracy/Comfort/EnergySensor)."""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.housetemp.const import DOMAIN, CONF_SENSOR_INDOOR_TEMP
from custom_components.housetemp.sensor import (
    HouseTempAccuracySensor,
    HouseTempComfortSensor,
    HouseTempEnergySensor,
)
from custom_components.housetemp.statistics import (
    StatsStore,
    PredictionRecord,
    ComfortSample,
    EnergySample,
    LifetimeComfortStats,
    LifetimeEnergyStats,
)


class TestHouseTempAccuracySensor:
    """Tests for the accuracy sensor."""

    @pytest.fixture
    def mock_setup(self, hass: HomeAssistant):
        """Create mock coordinator, entry, and stats_store."""
        entry = MockConfigEntry(
            domain=DOMAIN,
            data={CONF_SENSOR_INDOOR_TEMP: "sensor.indoor"},
            entry_id="test_entry"
        )
        entry.add_to_hass(hass)
        
        coord = MagicMock()
        coord.data = {"timestamps": [], "predicted_temp": []}
        
        stats_store = MagicMock(spec=StatsStore)
        stats_store.predictions = []
        stats_store.epoch_start = dt_util.now()
        
        return coord, entry, stats_store

    def test_native_value_returns_mae(self, hass: HomeAssistant, mock_setup):
        """Test that native_value returns MAE from StatsCalculator."""
        coord, entry, stats_store = mock_setup
        
        # Add resolved predictions
        now = dt_util.now()
        stats_store.predictions = [
            PredictionRecord(
                timestamp=(now - timedelta(hours=1)).isoformat(),
                predicted_temp=70.0,
                actual_temp=69.0,  # Error = 1.0
                horizon_hours=6,
            ),
            PredictionRecord(
                timestamp=(now - timedelta(hours=2)).isoformat(),
                predicted_temp=72.0,
                actual_temp=70.0,  # Error = 2.0
                horizon_hours=6,
            ),
        ]
        
        sensor = HouseTempAccuracySensor(coord, entry, stats_store)
        
        # MAE = (1 + 2) / 2 = 1.5
        assert sensor.native_value == 1.5

    def test_native_value_returns_none_when_no_data(self, hass: HomeAssistant, mock_setup):
        """Test that native_value returns None when no predictions exist."""
        coord, entry, stats_store = mock_setup
        stats_store.predictions = []
        
        sensor = HouseTempAccuracySensor(coord, entry, stats_store)
        assert sensor.native_value is None

    def test_extra_state_attributes(self, hass: HomeAssistant, mock_setup):
        """Test that extra_state_attributes contains expected keys."""
        coord, entry, stats_store = mock_setup
        
        now = dt_util.now()
        stats_store.predictions = [
            PredictionRecord(
                timestamp=(now - timedelta(hours=1)).isoformat(),
                predicted_temp=71.0,
                actual_temp=70.0,
                horizon_hours=6,
            ),
        ]
        
        sensor = HouseTempAccuracySensor(coord, entry, stats_store)
        attrs = sensor.extra_state_attributes
        
        assert "window_days" in attrs
        assert attrs["window_days"] == 7
        assert "mae_6h" in attrs
        assert "bias_6h" in attrs
        assert "samples_6h" in attrs
        assert "mae_full_forecast" in attrs
        assert "epoch_start" in attrs


class TestHouseTempComfortSensor:
    """Tests for the comfort sensor."""

    @pytest.fixture
    def mock_setup(self, hass: HomeAssistant):
        """Create mock coordinator, entry, and stats_store."""
        entry = MockConfigEntry(
            domain=DOMAIN,
            data={CONF_SENSOR_INDOOR_TEMP: "sensor.indoor"},
            entry_id="test_entry"
        )
        entry.add_to_hass(hass)
        
        coord = MagicMock()
        coord.data = {}
        
        stats_store = MagicMock(spec=StatsStore)
        stats_store.comfort_samples = []
        stats_store.lifetime_comfort = LifetimeComfortStats()
        stats_store.epoch_start = dt_util.now()
        
        return coord, entry, stats_store

    def test_native_value_returns_time_in_range(self, hass: HomeAssistant, mock_setup):
        """Test that native_value returns time_in_range percentage."""
        coord, entry, stats_store = mock_setup
        
        now = dt_util.now()
        # All samples within tolerance (100% in range)
        stats_store.comfort_samples = [
            ComfortSample(
                timestamp=(now - timedelta(hours=1)).isoformat(),
                actual_temp=70.0,
                schedule_target=70.0,
            ),
            ComfortSample(
                timestamp=(now - timedelta(hours=2)).isoformat(),
                actual_temp=70.5,
                schedule_target=70.0,
            ),
        ]
        
        sensor = HouseTempComfortSensor(coord, entry, stats_store)
        
        # Both samples within default tolerance (1.0°F)
        assert sensor.native_value == 100.0

    def test_native_value_returns_none_when_no_data(self, hass: HomeAssistant, mock_setup):
        """Test that native_value returns None when no samples exist."""
        coord, entry, stats_store = mock_setup
        stats_store.comfort_samples = []
        
        sensor = HouseTempComfortSensor(coord, entry, stats_store)
        assert sensor.native_value is None

    def test_extra_state_attributes(self, hass: HomeAssistant, mock_setup):
        """Test that extra_state_attributes contains expected keys."""
        coord, entry, stats_store = mock_setup
        
        sensor = HouseTempComfortSensor(coord, entry, stats_store)
        attrs = sensor.extra_state_attributes
        
        # 24h keys
        assert "time_in_range_24h" in attrs
        assert "mean_deviation_24h" in attrs
        assert "max_deviation_24h" in attrs
        
        # Lifetime keys
        assert "time_in_range_lifetime" in attrs
        assert "total_samples_lifetime" in attrs
        assert "epoch_start" in attrs


class TestHouseTempEnergySensor:
    """Tests for the energy sensor."""

    @pytest.fixture
    def mock_setup(self, hass: HomeAssistant):
        """Create mock coordinator, entry, and stats_store."""
        entry = MockConfigEntry(
            domain=DOMAIN,
            data={CONF_SENSOR_INDOOR_TEMP: "sensor.indoor"},
            entry_id="test_entry"
        )
        entry.add_to_hass(hass)
        
        coord = MagicMock()
        coord.data = {}
        
        stats_store = MagicMock(spec=StatsStore)
        stats_store.energy_samples = []
        stats_store.lifetime_energy = LifetimeEnergyStats(used_kwh=50.0, saved_kwh=10.0)
        stats_store.epoch_start = dt_util.now()
        
        return coord, entry, stats_store

    def test_native_value_returns_saved_kwh(self, hass: HomeAssistant, mock_setup):
        """Test that native_value returns lifetime saved_kwh."""
        coord, entry, stats_store = mock_setup
        
        sensor = HouseTempEnergySensor(coord, entry, stats_store)
        assert sensor.native_value == 10.0

    def test_native_value_returns_zero_when_no_savings(self, hass: HomeAssistant, mock_setup):
        """Test that native_value returns 0 when no savings."""
        coord, entry, stats_store = mock_setup
        stats_store.lifetime_energy = LifetimeEnergyStats(used_kwh=0.0, saved_kwh=0.0)
        
        sensor = HouseTempEnergySensor(coord, entry, stats_store)
        assert sensor.native_value == 0.0

    def test_extra_state_attributes(self, hass: HomeAssistant, mock_setup):
        """Test that extra_state_attributes contains expected keys."""
        coord, entry, stats_store = mock_setup
        
        now = dt_util.now()
        stats_store.energy_samples = [
            EnergySample(
                timestamp=(now - timedelta(hours=1)).isoformat(),
                used_kwh=2.0,
                baseline_kwh=2.5,
            ),
        ]
        
        sensor = HouseTempEnergySensor(coord, entry, stats_store)
        attrs = sensor.extra_state_attributes
        
        # 24h keys
        assert "used_kwh_24h" in attrs
        assert "saved_kwh_24h" in attrs
        
        # Lifetime keys
        assert "used_kwh_lifetime" in attrs
        assert "saved_kwh_lifetime" in attrs
        assert "epoch_start" in attrs


class TestSensorSetupCreatesStatsSensors:
    """Test that async_setup_entry creates all 4 sensors when stats_store exists."""

    @pytest.mark.asyncio
    async def test_sensor_setup_creates_all_entities(self, hass: HomeAssistant):
        """Test that setup creates main sensor + 3 stats sensors."""
        from custom_components.housetemp.sensor import async_setup_entry
        
        entry = MockConfigEntry(
            domain=DOMAIN,
            data={CONF_SENSOR_INDOOR_TEMP: "sensor.indoor"},
            entry_id="test_entry"
        )
        entry.add_to_hass(hass)
        
        # Mock coordinator
        coord = MagicMock()
        coord.data = {"timestamps": [], "predicted_temp": []}
        
        # Mock stats_store
        stats_store = MagicMock(spec=StatsStore)
        stats_store.predictions = []
        stats_store.comfort_samples = []
        stats_store.energy_samples = []
        stats_store.lifetime_comfort = LifetimeComfortStats()
        stats_store.lifetime_energy = LifetimeEnergyStats()
        stats_store.epoch_start = dt_util.now()
        
        # Setup domain data
        hass.data[DOMAIN] = {
            "test_entry": coord,
            "test_entry_stats": stats_store,
        }
        
        # Capture added entities
        added_entities = []
        
        def mock_add_entities(entities):
            added_entities.extend(entities)
        
        await async_setup_entry(hass, entry, mock_add_entities)
        
        # Should have 4 entities: main sensor + 3 stats sensors
        assert len(added_entities) == 4
        
        # Verify types
        entity_types = [type(e).__name__ for e in added_entities]
        assert "HouseTempPredictionSensor" in entity_types
        assert "HouseTempAccuracySensor" in entity_types
        assert "HouseTempComfortSensor" in entity_types
        assert "HouseTempEnergySensor" in entity_types

    @pytest.mark.asyncio
    async def test_sensor_setup_without_stats_store(self, hass: HomeAssistant):
        """Test that setup only creates main sensor when stats_store is missing."""
        from custom_components.housetemp.sensor import async_setup_entry
        
        entry = MockConfigEntry(
            domain=DOMAIN,
            data={CONF_SENSOR_INDOOR_TEMP: "sensor.indoor"},
            entry_id="test_entry"
        )
        entry.add_to_hass(hass)
        
        coord = MagicMock()
        coord.data = {"timestamps": [], "predicted_temp": []}
        
        # No stats_store
        hass.data[DOMAIN] = {
            "test_entry": coord,
        }
        
        added_entities = []
        
        def mock_add_entities(entities):
            added_entities.extend(entities)
        
        await async_setup_entry(hass, entry, mock_add_entities)
        
        # Should only have 1 entity (main sensor)
        assert len(added_entities) == 1
        assert type(added_entities[0]).__name__ == "HouseTempPredictionSensor"
