"""
Core Physics and Mathematical Constants.
These are independent of Home Assistant and define the simulation defaults.
"""

# Physics Defaults (US single-family home)
DEFAULT_C_THERMAL = 10000.0
DEFAULT_UA = 750.0
DEFAULT_K_SOLAR = 3000.0
DEFAULT_Q_INT = 2000.0
DEFAULT_H_FACTOR = 20000.0
DEFAULT_SOLAR_GAIN = 2.0
DEFAULT_TIME_CONSTANT = 30.0
DEFAULT_EFFICIENCY_DERATE = 0.75
DEFAULT_EFF_DERATE = 0.9

# Simulation / Control Defaults
DEFAULT_SWING_TEMP = 1.0
DEFAULT_MIN_CYCLE_MINUTES = 15.0
DEFAULT_OFF_INTENT_EPS = 0.1  # Setpoint tolerance for "True Off" in get_effective_hvac_state (continuous)
DEFAULT_MIN_SETPOINT = 60.0   # Lower bound for setpoint optimization
DEFAULT_MAX_SETPOINT = 75.0   # Upper bound for setpoint optimization
DEFAULT_IDLE_MARGIN = 0.5     # Margin above ON-threshold for reliable idle detection (°F)

# Optimizer Snapping (True Off Redesign)
W_BOUNDARY_PULL_HEAT = 0.05    # Tie-break weight for heating
W_BOUNDARY_PULL_COOL = 0.0     # Tie-break weight for cooling (avoids rebound)
OFF_BLOCK_MINUTES = 60        # Aggregation window for OFF recommendation
S_GAP_SMOOTH = 0.2            # Sigmoid width for w_idle gating (°F)

# Energy/Physics Tolerances
TOLERANCE_BTU_ACTIVE = 1.0    # Minimum output (BTU/hr) to consider "active" (absolute floor)
TOLERANCE_BTU_FRACTION = 0.05  # Fraction of min_output to consider "active" (relative threshold)

# Costs
DEFAULT_COST_PER_KWH = 0.45

# Unit Conversions
KW_TO_WATTS = 1000.0
BTU_TO_WATTS = 0.293071 # 1 / 3.412
BTU_TO_KWH = 0.000293071
