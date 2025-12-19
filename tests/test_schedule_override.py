import pytest
import pandas as pd
import numpy as np
import json
from datetime import datetime
from custom_components.housetemp.housetemp.schedule import process_schedule_data

class TestScheduleOverride:
    def setup_method(self):
        self.timestamps = pd.date_range("2023-01-01 00:00", periods=24, freq="h")
        self.base_schedule = {
            "schedule": [
                {
                    "weekdays": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                    "daily_schedule": [{"time": "00:00", "temp": 70}]
                }
            ]
        }

    def test_missing_mode_raises_error(self):
        """Standard behavior: No mode in JSON, no default arg -> Error"""
        with pytest.raises(ValueError, match="HVAC Mode"):
            process_schedule_data(self.timestamps, self.base_schedule)

    def test_default_argument_used(self):
        """Function argument default is used if JSON missing"""
        # Default says COOL
        hvac_state, _, _ = process_schedule_data(self.timestamps, self.base_schedule, default_mode="cool")
        
        # Should be COOL (-1)
        assert np.all(hvac_state == -1)

    def test_json_override_precedence(self):
        """JSON config overrides EVERYTHING (Argument default)"""
        json_sched = self.base_schedule.copy()
        json_sched["mode"] = "cool"
        
        # Even if we pass 'heat' as default, JSON 'cool' wins (explicit strict override)
        hvac_state, _, _ = process_schedule_data(self.timestamps, json_sched, default_mode="heat")
        
        assert np.all(hvac_state == -1) # COOL wins

