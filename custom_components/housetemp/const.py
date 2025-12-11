"""Constants for the House Temp Prediction integration."""

DOMAIN = "housetemp"

CONF_C_THERMAL = "c_thermal"
CONF_UA = "ua"
CONF_K_SOLAR = "k_solar"
CONF_Q_INT = "q_int"
CONF_H_FACTOR = "h_factor"

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
