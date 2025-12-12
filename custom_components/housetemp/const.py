"""Constants for the House Temp Prediction integration."""

DOMAIN = "housetemp"

CONF_C_THERMAL = "c_thermal"
CONF_UA = "ua"
CONF_K_SOLAR = "k_solar"
CONF_Q_INT = "q_int"
CONF_H_FACTOR = "h_factor"
CONF_CENTER_PREFERENCE = "center_preference"

CONF_SENSOR_INDOOR_TEMP = "sensor_indoor_temp"
CONF_WEATHER_ENTITY = "weather_entity"
CONF_SOLAR_ENTITY = "solar_entity"

CONF_HEAT_PUMP_CONFIG = "heat_pump_config"
CONF_SCHEDULE_CONFIG = "schedule_config"
CONF_FORECAST_DURATION = "forecast_duration"
CONF_UPDATE_INTERVAL = "update_interval"

DEFAULT_FORECAST_DURATION = 8
DEFAULT_UPDATE_INTERVAL = 15

# Optimization


# Advanced Physics/Control Settings
CONF_MODEL_TIMESTEP = "model_timestep"       # Physics resolution (minutes)
DEFAULT_MODEL_TIMESTEP = 5
MIN_MODEL_TIMESTEP = 1

CONF_CONTROL_TIMESTEP = "control_timestep"   # Control block size (minutes)
DEFAULT_CONTROL_TIMESTEP = 30
MIN_CONTROL_TIMESTEP = 30

# Default Temperatures (Fahrenheit)
DEFAULT_AWAY_TEMP = 50.0


# Away Mode
AWAY_WAKEUP_ADVANCE_HOURS = 12

CONF_HVAC_MODE = "hvac_mode"
CONF_AVOID_DEFROST = "avoid_defrost"

# Physics Defaults (US single-family home)
DEFAULT_C_THERMAL = 10000.0
DEFAULT_UA = 750.0
DEFAULT_K_SOLAR = 3000.0
DEFAULT_Q_INT = 2000.0
DEFAULT_H_FACTOR = 5000.0
DEFAULT_CENTER_PREFERENCE = 1.0
DEFAULT_SCHEDULE_CONFIG = "[]"
