
import unittest
import numpy as np
import pandas as pd
from housetemp.schedule import process_schedule_data

class TestScheduleProcessing(unittest.TestCase):
    def test_rate_carry_forward(self):
        """
        Verify that 'rate' entries in the schedule carry forward to subsequent items
        that do not specify a rate, and wrap around correctly.
        """
        timestamps = pd.date_range("2024-01-01 00:00", "2024-01-01 23:55", freq="5min")
        
        # Schedule with mixed rate specifications
        schedule_data = {
            "mode": "heat",
            "schedule": [
                {
                    "weekdays": ["monday"],
                    "daily_schedule": [
                        {"time": "06:00", "temp": 68.0, "rate": 2.0},  # Peak start
                        {"time": "09:00", "temp": 65.0},              # Inherits 2.0 (Peak implies setback?)
                        {"time": "17:00", "temp": 68.0},              # Inherits 2.0
                        {"time": "21:00", "temp": 62.0, "rate": 1.0}   # Off-Peak
                    ]
                }
            ]
        }
        
        # Force Monday (2024-01-01 is a Monday)
        _, targets, _, rates = process_schedule_data(timestamps, schedule_data, default_mode="heat")
        
        # 1. Check Wrap-Around (00:00 - 06:00)
        # Should inherit from last item (21:00, rate 1.0)
        idx_0000 = 0
        idx_0555 = (6 * 12) - 1
        self.assertEqual(rates[idx_0000], 1.0, "00:00 should inherit wrap-around rate 1.0")
        self.assertEqual(rates[idx_0555], 1.0, "05:55 should inherit wrap-around rate 1.0")
        
        # 2. Check Explicit Set (06:00 - 09:00)
        idx_0600 = 6 * 12
        self.assertEqual(rates[idx_0600], 2.0, "06:00 should be set to 2.0")
        
        # 3. Check Carry Forward (09:00 - 21:00)
        # 09:00 entry has NO rate, should keep 2.0 from 06:00
        idx_0900 = 9 * 12
        self.assertEqual(rates[idx_0900], 2.0, "09:00 should carry forward rate 2.0")
        
        idx_1700 = 17 * 12
        self.assertEqual(rates[idx_1700], 2.0, "17:00 should carry forward rate 2.0")
        
        # 4. Check Reset (21:00 - End)
        idx_2100 = 21 * 12
        self.assertEqual(rates[idx_2100], 1.0, "21:00 should reset to rate 1.0")
        
    def test_default_rate_is_one(self):
        timestamps = pd.date_range("2024-01-01 00:00", "2024-01-01 23:55", freq="5min")
        schedule_data = {
            "mode": "heat",
            "schedule": [
                {
                    "weekdays": ["monday"],
                    "daily_schedule": [
                        {"time": "08:00", "temp": 70.0},
                        {"time": "20:00", "temp": 60.0}
                    ]
                }
            ]
        }
        
        _, _, _, rates = process_schedule_data(timestamps, schedule_data, default_mode="heat")
        self.assertTrue(np.all(rates == 1.0), "Default rate should be 1.0 for all steps")

if __name__ == "__main__":
    unittest.main()
