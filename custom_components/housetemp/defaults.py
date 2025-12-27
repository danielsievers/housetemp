"""
Default JSON Configurations for HouseTemp.
Moved here to keep const.py clean.
"""

DEFAULT_SCHEDULE_CONFIG = """{
  "schedule": [
    {
      "weekdays": [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday"
      ],
      "daily_schedule": [
        {
          "time": "00:00",
          "temp": 63.0
        },
        {
          "time": "07:00",
          "temp": 69.0
        },
        {
          "time": "21:00",
          "temp": 63.0
        }
      ]
    }
  ]
}"""

# Heat Pump Specification Defaults (Mitsubishi MXZ-SM60NAM)
DEFAULT_HEAT_PUMP_CONFIG = """{
  "description": "Mitsubishi MXZ-SM60NAM (5-Ton Hyper-Heat)",
  "min_output_btu_hr": 12000,
  "max_cool_btu_hr": 54000,
  "plf_low_load": 1.4,
  "plf_slope": 0.4,
  "idle_power_kw": 0.25,
  "blower_active_kw": 0.0,
  "max_capacity": {
    "x_outdoor_f": [-13, -5, 5, 17, 47, 65],
    "y_btu_hr": [38000, 45000, 60000, 54000, 66000, 72000]
  },
  "cop": {
    "x_outdoor_f": [5, 17, 47, 65],
    "y_cop": [1.75, 2.10, 3.40, 4.20]
  },
  "defrost": {
    "description": "Defrost cycle parameters for cold/humid conditions (Mitsubishi MXZ-SM60NAM)",
    "trigger_temp_f": 32,
    "risk_zone_f": [28, 42],
    "cycle_duration_minutes": 10,
    "cycle_interval_minutes": 60,
    "power_kw": 4.5
  }
}"""
