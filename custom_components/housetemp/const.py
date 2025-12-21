"""Constants for the House Temp Prediction integration."""

DOMAIN = "housetemp"

CONF_C_THERMAL = "c_thermal"
CONF_UA = "ua"
CONF_MAX_COOL = "max_cool_btu_hr"
CONF_SOLAR_GAIN = "solar_gain"
CONF_TIME_CONSTANT = "time_constant"
CONF_EFFICIENCY_DERATE = "efficiency_derate"
CONF_SWING_TEMP = "swing_temp"
CONF_MIN_CYCLE_DURATION = "min_cycle_duration_minutes"
CONF_K_SOLAR = "k_solar"
CONF_Q_INT = "q_int"
CONF_H_FACTOR = "h_factor"
CONF_EFF_DERATE = "eff_derate"
# --- Defaults for Config Flow ---
CONF_CENTER_PREFERENCE = "center_preference"

CONF_SENSOR_INDOOR_TEMP = "sensor_indoor_temp"
CONF_WEATHER_ENTITY = "weather_entity"
CONF_SOLAR_ENTITY = "solar_entity"

CONF_HEAT_PUMP_CONFIG = "heat_pump_config"
CONF_SCHEDULE_CONFIG = "schedule_config"
CONF_SCHEDULE_ENABLED = "schedule_enabled"
CONF_FORECAST_DURATION = "forecast_duration"
CONF_UPDATE_INTERVAL = "update_interval"

DEFAULT_FORECAST_DURATION = 8
DEFAULT_UPDATE_INTERVAL = 15
DEFAULT_SCHEDULE_ENABLED = True

# Optimization


# Advanced Physics/Control Settings
CONF_MODEL_TIMESTEP = "model_timestep"       # Physics resolution (minutes)
DEFAULT_MODEL_TIMESTEP = 5
MIN_MODEL_TIMESTEP = 1

CONF_CONTROL_TIMESTEP = "control_timestep"   # Control block size (minutes)
DEFAULT_CONTROL_TIMESTEP = 30
MIN_CONTROL_TIMESTEP = 30

CONF_ENABLE_MULTISCALE = "enable_multiscale"
DEFAULT_ENABLE_MULTISCALE = True

# Default Temperatures (Fahrenheit)
DEFAULT_AWAY_TEMP = 50.0


# Away Mode
AWAY_WAKEUP_ADVANCE_HOURS = 12

# Optimizer Setpoint Bounds (Fahrenheit)
CONF_MIN_SETPOINT = "min_setpoint"
CONF_MAX_SETPOINT = "max_setpoint"
DEFAULT_MIN_SETPOINT = 60.0
DEFAULT_MAX_SETPOINT = 75.0
SNAP_REG_WEIGHT = 0.001  # Tie-breaker weight for snap-to-boundary

CONF_HVAC_MODE = "hvac_mode"
CONF_AVOID_DEFROST = "avoid_defrost"
CONF_COMFORT_MODE = "comfort_mode"
CONF_DEADBAND_SLACK = "deadband_slack"

DEFAULT_COMFORT_MODE = "quadratic"
DEFAULT_DEADBAND_SLACK = 1.5
DEFAULT_SWING_TEMP = 1.0

# Physics Defaults (Imported from Core Lib)
from .housetemp.constants import (
    DEFAULT_C_THERMAL,
    DEFAULT_UA,
    DEFAULT_K_SOLAR,
    DEFAULT_Q_INT,
    DEFAULT_H_FACTOR,
    DEFAULT_SOLAR_GAIN,
    DEFAULT_TIME_CONSTANT,
    DEFAULT_EFFICIENCY_DERATE,
    DEFAULT_EFF_DERATE,
    DEFAULT_SWING_TEMP,
    DEFAULT_MIN_CYCLE_MINUTES,
    DEFAULT_OFF_INTENT_EPS,
    TOLERANCE_BTU_ACTIVE,
    TOLERANCE_BTU_FRACTION,
    KW_TO_WATTS  # If needed here
)

# --- Configuration Keys ---
DEFAULT_CENTER_PREFERENCE = 1.0
from .defaults import (
    DEFAULT_SCHEDULE_CONFIG,
    DEFAULT_HEAT_PUMP_CONFIG
)

# Heat Pump Specification Defaults (Mitsubishi MXZ-SM60NAM) 
# Note: These scalar defaults are strictly just "backup" checks or UI hints, 
# the real logic parsing happens in run_model.HeatPump init from the JSON above.
DEFAULT_HP_MIN_OUTPUT = 12000
DEFAULT_HP_MAX_COOL = 54000
DEFAULT_HP_PLF_LOW = 1.4
DEFAULT_HP_PLF_SLOPE = 0.4
DEFAULT_HP_IDLE_KW = 0.25
DEFAULT_HP_BLOWER_KW = 0.0
