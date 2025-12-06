import unittest
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz
from housetemp.schedule import load_comfort_schedule
import os

class TestScheduleSchema(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_comfort_schema.json"
        self.tz_la = tz.gettz("America/Los_Angeles")

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def create_comfort_file(self, schedule_data):
        data = {
            "center_preference": 0.5,
            "mode": "heat",
            "schedule": schedule_data
        }
        with open(self.test_file, "w") as f:
            json.dump(data, f)

    def test_legacy_format_raises_error(self):
        # Legacy format (no weekdays keys) should now fail
        schedule = [
            {"time": "08:00", "temp": 70},
            {"time": "22:00", "temp": 60}
        ]
        self.create_comfort_file(schedule)
        
        timestamps = [datetime(2023, 1, 2, tzinfo=self.tz_la)]
        with self.assertRaisesRegex(ValueError, "Legacy flat format is no longer supported"):
            load_comfort_schedule(self.test_file, timestamps)

    def test_nested_format_success(self):
        # Correct nested format
        schedule = [
            {
                "weekdays": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                "daily_schedule": [
                     {"time": "00:00", "temp": 70}
                ]
            }
        ]
        self.create_comfort_file(schedule)
        
        # Test 24 hours
        start = datetime(2023, 1, 2, 0, 0, tzinfo=self.tz_la)
        timestamps = [start + timedelta(hours=i) for i in range(24)]
        
        targets, _ = load_comfort_schedule(self.test_file, timestamps)
        self.assertTrue(np.all(targets == 70))

    def test_missing_daily_schedule_error(self):
        schedule = [
            {"weekdays": ["monday"]}
            # Missing daily_schedule
        ]
        self.create_comfort_file(schedule)
        
        timestamps = [datetime(2023, 1, 2, tzinfo=self.tz_la)]
        with self.assertRaisesRegex(ValueError, "missing 'daily_schedule'"):
            load_comfort_schedule(self.test_file, timestamps)

    def test_validation_overlap_error(self):
        # Monday defined in two groups
        schedule = [
            {
                "weekdays": ["monday"],
                "daily_schedule": [{"time": "08:00", "temp": 70}]
            },
            {
                "weekdays": ["monday"],
                "daily_schedule": [{"time": "22:00", "temp": 60}]
            }
        ]
        self.create_comfort_file(schedule)
        
        timestamps = [datetime(2023, 1, 2, tzinfo=self.tz_la)]
        with self.assertRaisesRegex(ValueError, "defined in multiple schedule groups"):
            load_comfort_schedule(self.test_file, timestamps)

    def test_missing_day_runtime_error(self):
        # Only Monday defined
        schedule = [
            {
                "weekdays": ["monday"],
                "daily_schedule": [{"time": "00:00", "temp": 70}]
            }
        ]
        self.create_comfort_file(schedule)
        
        # Try to load for Tuesday
        timestamps = [datetime(2023, 1, 3, tzinfo=self.tz_la)] # Tuesday
        
        with self.assertRaisesRegex(ValueError, "No schedule defined for weekday index"):
            load_comfort_schedule(self.test_file, timestamps)

    def test_case_insensitive_weekday(self):
        schedule = [
            {
                "weekdays": ["MONDAY"],
                "daily_schedule": [{"time": "00:00", "temp": 70}]
            }
        ]
        self.create_comfort_file(schedule)
        
        timestamps = [datetime(2023, 1, 2, tzinfo=self.tz_la)] # Monday
        targets, _ = load_comfort_schedule(self.test_file, timestamps)
        self.assertEqual(targets[0], 70)

if __name__ == '__main__':
    unittest.main()
