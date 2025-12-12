import unittest
import voluptuous as vol
from custom_components.housetemp.config_flow import (
    SCHEDULE_SCHEMA, 
    validate_schedule_timeline
)
import pytest

class TestConfigSchema(unittest.TestCase):
    
    def test_schema_valid_structure(self):
        """Test that a fully valid schema passes structral validation."""
        valid_data = {
            "mode": "heat",
            "schedule": [
                {
                    "weekdays": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                    "daily_schedule": [{"time": "08:00", "temp": 70.0}]
                },
                {
                    "weekdays": ["saturday", "sunday"],
                    "daily_schedule": [{"time": "09:00", "temp": 72.0}]
                }
            ]
        }
        # Voluptuous returns the data if valid
        out = SCHEDULE_SCHEMA(valid_data)
        self.assertEqual(out["mode"], "heat")
        self.assertTrue(validate_schedule_timeline(out))

    def test_schema_missing_required(self):
        """Test missing required fields in schema."""
        invalid_data = {
            "schedule": [] # Missing mode
        }
        with self.assertRaises(vol.Invalid):
            SCHEDULE_SCHEMA(invalid_data)

    def test_schema_daily_schedule_structure(self):
        """Test invalid types in daily schedule."""
        invalid_data = {
            "mode": "heat",
            "schedule": [{
                "weekdays": ["monday"],
                "daily_schedule": [{"time": "bad_time", "temp": 70}] # bad regex
            }]
        }
        with self.assertRaises(vol.Invalid):
            SCHEDULE_SCHEMA(invalid_data)

    def test_timeline_missing_day(self):
        """Test validator catches missing days."""
        # Only Monday covered
        data = {
            "schedule": [{
                "weekdays": ["monday"], 
                "daily_schedule": [{"time": "08:00", "temp": 70}]
            }]
        }
        with self.assertRaisesRegex(ValueError, "Schedule must cover all 7 days"):
            validate_schedule_timeline(data)

    def test_timeline_duplicate_day(self):
        """Test validator catches duplicate days."""
        data = {
            "schedule": [
                {
                    "weekdays": ["monday"], 
                    "daily_schedule": [{"time": "08:00", "temp": 70}]
                },
                {
                    "weekdays": ["monday"],  # Monday repeated
                    "daily_schedule": [{"time": "09:00", "temp": 72}]
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "defined in multiple schedule blocks"):
            validate_schedule_timeline(data)

    def test_timeline_empty_daily(self):
        """Test validator catches empty daily schedule."""
        data = {
            "schedule": [{
                "weekdays": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                "daily_schedule": []
            }]
        }
        with self.assertRaisesRegex(ValueError, "daily_schedule cannot be empty"):
            validate_schedule_timeline(data)

    def test_timeline_unsorted_times(self):
        """Test validator catches unsorted times."""
        data = {
            "schedule": [{
                "weekdays": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                "daily_schedule": [
                    {"time": "12:00", "temp": 70},
                    {"time": "08:00", "temp": 70} # Not sorted
                ]
            }]
        }
        with self.assertRaisesRegex(ValueError, "sorted chronologically"):
            validate_schedule_timeline(data)
