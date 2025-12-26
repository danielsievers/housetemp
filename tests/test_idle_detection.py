"""Tests for detect_idle_blocks helper function."""
import numpy as np
import pytest

from custom_components.housetemp.housetemp.utils import detect_idle_blocks


class TestDetectIdleBlocks:
    """Tests for simulation-based idle detection."""

    def test_all_active_returns_no_idle(self):
        """When thermostat is always active, no blocks are idle."""
        # 12 steps, 6 per block = 2 blocks
        actual_state = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        sim_temps = np.full(12, 70.0)
        intent = np.ones(12, dtype=int)
        setpoints = np.full(12, 68.0)
        
        result = detect_idle_blocks(
            actual_state=actual_state,
            sim_temps=sim_temps,
            intent=intent,
            setpoints=setpoints,
            swing=1.0,
            steps_per_block=6,
            margin=0.5
        )
        
        assert not np.any(result), "No blocks should be idle when always active"

    def test_idle_block_with_margin_detected(self):
        """Idle block with temp well above ON threshold is detected."""
        # Block 0: all zeros, intent=1, temp=70 (setpoint=68, swing=1 -> ON at 67)
        # Block 1: active
        actual_state = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        sim_temps = np.full(12, 70.0)  # 3°F above ON threshold (67)
        intent = np.ones(12, dtype=int)
        setpoints = np.full(12, 68.0)
        
        result = detect_idle_blocks(
            actual_state=actual_state,
            sim_temps=sim_temps,
            intent=intent,
            setpoints=setpoints,
            swing=1.0,
            steps_per_block=6,
            margin=0.5
        )
        
        # Block 0 should be idle (first 6 elements)
        assert np.all(result[:6]), "Block 0 should be detected as idle"
        assert not np.any(result[6:]), "Block 1 should not be idle"

    def test_idle_near_threshold_not_detected(self):
        """Idle block with temp close to ON threshold is NOT detected (unsafe)."""
        # Temp = 67.2°F, ON threshold = 67°F, safe_floor = 67.5°F
        # 67.2 < 67.5 -> fails margin check
        actual_state = np.zeros(6, dtype=int)
        sim_temps = np.full(6, 67.2)
        intent = np.ones(6, dtype=int)
        setpoints = np.full(6, 68.0)
        
        result = detect_idle_blocks(
            actual_state=actual_state,
            sim_temps=sim_temps,
            intent=intent,
            setpoints=setpoints,
            swing=1.0,
            steps_per_block=6,
            margin=0.5
        )
        
        assert not np.any(result), "Block near ON threshold should not be flagged as idle"

    def test_true_off_intent_not_flagged(self):
        """When intent is already 0 (True-Off), block should NOT be flagged."""
        # This avoids redundant signals - True-Off is already handled elsewhere
        actual_state = np.zeros(6, dtype=int)
        sim_temps = np.full(6, 70.0)
        intent = np.zeros(6, dtype=int)  # Already off by intent
        setpoints = np.full(6, 68.0)
        
        result = detect_idle_blocks(
            actual_state=actual_state,
            sim_temps=sim_temps,
            intent=intent,
            setpoints=setpoints,
            swing=1.0,
            steps_per_block=6,
            margin=0.5
        )
        
        assert not np.any(result), "True-Off intent should not be re-flagged as idle"

    def test_cooling_mode_idle_detection(self):
        """Idle detection works correctly in cooling mode."""
        # Cooling: ON when T > (setpoint + swing)
        # Setpoint=75, swing=1 -> ON at 76, safe_ceiling=75.5
        actual_state = np.zeros(6, dtype=int)
        sim_temps = np.full(6, 73.0)  # Well below ceiling
        intent = np.full(6, -1, dtype=int)  # Cooling mode
        setpoints = np.full(6, 75.0)
        
        result = detect_idle_blocks(
            actual_state=actual_state,
            sim_temps=sim_temps,
            intent=intent,
            setpoints=setpoints,
            swing=1.0,
            steps_per_block=6,
            margin=0.5
        )
        
        assert np.all(result), "Cooling idle block should be detected"

    def test_cooling_near_threshold_not_detected(self):
        """Cooling idle near ON threshold is NOT detected."""
        # Temp = 75.8°F, ceiling = 76-0.5 = 75.5
        # 75.8 > 75.5 -> fails margin check
        actual_state = np.zeros(6, dtype=int)
        sim_temps = np.full(6, 75.8)
        intent = np.full(6, -1, dtype=int)
        setpoints = np.full(6, 75.0)
        
        result = detect_idle_blocks(
            actual_state=actual_state,
            sim_temps=sim_temps,
            intent=intent,
            setpoints=setpoints,
            swing=1.0,
            steps_per_block=6,
            margin=0.5
        )
        
        assert not np.any(result), "Cooling block near ON threshold should not be flagged"

    def test_partial_block_at_end(self):
        """Partial blocks at end of array are handled correctly."""
        # 10 steps, 6 per block = 1 full block + 1 partial (4 steps)
        actual_state = np.zeros(10, dtype=int)
        sim_temps = np.full(10, 70.0)
        intent = np.ones(10, dtype=int)
        setpoints = np.full(10, 68.0)
        
        result = detect_idle_blocks(
            actual_state=actual_state,
            sim_temps=sim_temps,
            intent=intent,
            setpoints=setpoints,
            swing=1.0,
            steps_per_block=6,
            margin=0.5
        )
        
        # Both blocks should be idle
        assert np.all(result), "Partial block should be handled correctly"

    def test_mixed_sign_intent_not_flagged(self):
        """Block with mixed heat/cool intent should NOT be flagged (conservative)."""
        actual_state = np.zeros(6, dtype=int)
        sim_temps = np.full(6, 70.0)
        intent = np.array([1, 1, 1, -1, -1, -1], dtype=int)  # Mixed signs
        setpoints = np.full(6, 68.0)
        
        result = detect_idle_blocks(
            actual_state=actual_state,
            sim_temps=sim_temps,
            intent=intent,
            setpoints=setpoints,
            swing=1.0,
            steps_per_block=6,
            margin=0.5
        )
        
        assert not np.any(result), "Mixed-sign blocks should not be flagged"

    def test_accepts_list_inputs(self):
        """Function should accept Python lists (type safety)."""
        # Pass lists instead of numpy arrays
        actual_state = [0, 0, 0, 0, 0, 0]
        sim_temps = [70.0, 70.0, 70.0, 70.0, 70.0, 70.0]
        intent = [1, 1, 1, 1, 1, 1]
        setpoints = [68.0, 68.0, 68.0, 68.0, 68.0, 68.0]
        
        # Should not raise, should return numpy array
        result = detect_idle_blocks(
            actual_state=actual_state,
            sim_temps=sim_temps,
            intent=intent,
            setpoints=setpoints,
            swing=1.0,
            steps_per_block=6,
            margin=0.5
        )
        
        assert isinstance(result, np.ndarray), "Should return numpy array"
        assert np.all(result), "Should detect idle with list inputs"
