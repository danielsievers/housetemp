import json
import numpy as np
import logging
from .measurements import Measurements

try:
    from .constants import TOLERANCE_BTU_ACTIVE, TOLERANCE_BTU_FRACTION, DEFAULT_EFFICIENCY_DERATE
except (ImportError, ValueError):
    # Fallback for standalone library usage
    TOLERANCE_BTU_ACTIVE = 1.0
    TOLERANCE_BTU_FRACTION = 0.05
    DEFAULT_EFFICIENCY_DERATE = 1.0  # No derate fallback

_LOGGER = logging.getLogger(__name__)

# --- TRUE CONSTANTS (Physical/Mathematical) ---
# Soft Start / Thermal Inertia Logic
SOFT_START_RAMP_MINUTES = 5.0  # Time to reach full capacity
SOFT_START_MIN_FACTOR = 0.01   # Minimum ramp factor (avoid zero)

# Defrost Defaults
DEFAULT_DEFROST_TRIGGER = 32
DEFAULT_DEFROST_RISK_ZONE = [28, 42]
DEFAULT_DEFROST_DURATION = 10
DEFAULT_DEFROST_INTERVAL = 60
DEFAULT_DEFROST_POWER = 4.5
DEFAULT_PLF_MIN = 0.5


class HeatPump:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            # Support // comments
            content = f.read()
            import re
            content = re.sub(r'//.*', '', content)
            data = json.loads(content)
        
        # Mandatory Physics/Config Parameters
        self.cap_x = data['max_capacity']['x_outdoor_f']
        self.cap_y = data['max_capacity']['y_btu_hr']
        self.cop_x = data['cop']['x_outdoor_f']
        self.cop_y = data['cop']['y_cop']
        
        # Performance/Load parameters (Must be in JSON)
        self.min_output_btu_hr = data['min_output_btu_hr']
        self.max_cool_btu_hr = data['max_cool_btu_hr']
        self.plf_low_load = data['plf_low_load']
        self.plf_slope = data['plf_slope']
        self.plf_min = data.get('plf_min', DEFAULT_PLF_MIN)
        self.idle_power_kw = data['idle_power_kw']
        self.blower_active_kw = data['blower_active_kw']
        
        # Optional Cooling COP (Fallback to Heat COP if missing)
        if 'cooling_cop' in data:
            self.cool_cop_x = data['cooling_cop']['x_outdoor_f']
            self.cool_cop_y = data['cooling_cop']['y_cop']
        else:
            _LOGGER.info("Note: No 'cooling_cop' defined in heat_pump.json. Using Heating COP curve for cooling.")
            self.cool_cop_x = self.cop_x
            self.cool_cop_y = self.cop_y
        
        # Defrost parameters (optional - None if not specified)
        if 'defrost' in data:
            defrost = data['defrost']
            self.defrost_trigger_temp = defrost.get('trigger_temp_f', DEFAULT_DEFROST_TRIGGER)
            self.defrost_risk_zone = defrost.get('risk_zone_f', DEFAULT_DEFROST_RISK_ZONE)
            self.defrost_duration_min = defrost.get('cycle_duration_minutes', DEFAULT_DEFROST_DURATION)
            self.defrost_interval_min = defrost.get('cycle_interval_minutes', DEFAULT_DEFROST_INTERVAL)
            self.defrost_power_kw = defrost.get('power_kw', DEFAULT_DEFROST_POWER)
        else:
            self.defrost_risk_zone = None  # Signals no defrost modeling

    def _check_bounds(self, t_out_array, x_axis, curve_name):
        """Helper to warn about extrapolation"""
        min_x = x_axis[0]
        max_x = x_axis[-1]
        
        low_viol = np.sum(t_out_array < min_x)
        high_viol = np.sum(t_out_array > max_x)
        
        if low_viol > 0:
            _LOGGER.debug(f"{curve_name} Extrapolation detected (Low Temp): {low_viol} steps < {min_x}F. Clamping to floor.")
        if high_viol > 0:
            _LOGGER.debug(f"{curve_name} Extrapolation detected (High Temp): {high_viol} steps > {max_x}F. Holding boundary value.")

    def get_max_capacity(self, t_out_array):
        return np.interp(t_out_array, self.cap_x, self.cap_y)

    def get_cop(self, t_out_array):
        # Base interpolation
        cop = np.interp(t_out_array, self.cop_x, self.cop_y)
        
        # Safety: Check bounds
        self._check_bounds(t_out_array, self.cop_x, "Heating COP")
        
        # Safety: Low Temp clamp (Resistance Floor)
        min_defined_temp = self.cop_x[0]
        cop = np.where(t_out_array < min_defined_temp, 1.0, cop)
        
        return cop

    def get_cooling_cop(self, t_out_array):
        # Base interpolation using cooling curve
        cop = np.interp(t_out_array, self.cool_cop_x, self.cool_cop_y)
        
        # Safety: Check bounds
        self._check_bounds(t_out_array, self.cool_cop_x, "Cooling COP")
        
        # Note: High temp extrapolation for cooling is handled by np.interp holding the last value.
        # This is optimistic (efficiency should drop), but we logged the warning above.
        return cop

try:
    from .constants import (
        TOLERANCE_BTU_ACTIVE, 
        TOLERANCE_BTU_FRACTION,
        DEFAULT_SWING_TEMP,
        DEFAULT_MIN_CYCLE_MINUTES,
        DEFAULT_OFF_INTENT_EPS
    )
except ImportError:
    # This might happen if running script directly without package context?
    # But usually .constants should work if in same package.
    # Fallback just in case, or assume package structure.
    # Actually, if we are inside the package, simple relative import works.
    from .constants import *

# (Better: just explicit imports)
from .constants import (
    TOLERANCE_BTU_ACTIVE,
    TOLERANCE_BTU_FRACTION,
    DEFAULT_SWING_TEMP,
    DEFAULT_MIN_CYCLE_MINUTES,
    DEFAULT_OFF_INTENT_EPS
)

def run_model_continuous(params, *, t_out_list, solar_kw_list, dt_hours_list, setpoint_list, hvac_state_list, max_caps_list, min_output, max_cool, eff_derate, start_temp):
    """
    Continuous Physics Model.
    Calculates thermodynamic response in a single pass.
    """
    # Unpack params
    C_thermal = params[0]
    UA = params[1]
    K_solar = params[2]
    Q_int = params[3]
    H_factor = params[4]
    
    total_steps = len(t_out_list)
    sim_temps_list = [0.0] * total_steps
    hvac_outputs_list = [0.0] * total_steps
    hvac_produced_list = [0.0] * total_steps
    
    current_temp = float(start_temp)
    
    # Soft Start State
    elapsed_active_minutes = 0.0
    
    for i in range(total_steps):
        sim_temps_list[i] = current_temp
        
        q_leak = UA * (t_out_list[i] - current_temp)
        q_solar = K_solar * solar_kw_list[i]
        
        # HVAC Logic (Continuous Proportional)
        setpoint = setpoint_list[i]
        requested_mode = hvac_state_list[i] # 1=Heat, -1=Cool, 0=Off
        
        q_hvac = 0.0
        
        if requested_mode != 0:
            if requested_mode > 0: # HEATING
                # Proportional Control (Inverter Logic)
                gap = setpoint - current_temp
                
                # Continuous Logic: Pure Proportional (Duty Cycle Approximation)
                # Correction: Ensure NO output if gap <= 0 (True Off capability)
                if gap <= 0:
                    request = 0.0
                else:
                    # If gap > 0, request scales with gap.
                    # Since H_factor is large (10k), small gap yields large request.
                    request = H_factor * gap
                
                # Cap
                cap = max_caps_list[i]
                if request > cap:
                    request = cap
                
                q_hvac = request

            elif requested_mode < 0: # COOLING
                gap = current_temp - setpoint
                
                if gap <= 0:
                    request = 0.0
                else:
                    request = H_factor * gap
                
                if request > max_cool:
                    request = max_cool
                    
                q_hvac = -request
            
            # Store produced BEFORE derate/ramp (gross output for energy billing)
            hvac_produced_list[i] = q_hvac
            
            # --- Soft-Start Ramp (Gated on Actual Output) ---
            # Only advance ramp timer if compressor is actually producing output.
            # This prevents the optimizer from exploiting "free" ramp-ups after
            # periods where demand was satisfied (gap <= 0) but mode was enabled.
            # Use min_output-relative threshold to avoid spurious resets on small-but-real outputs.
            active_threshold = max(TOLERANCE_BTU_ACTIVE, TOLERANCE_BTU_FRACTION * min_output)
            if abs(q_hvac) > active_threshold:
                elapsed_active_minutes += (dt_hours_list[i] * 60.0)
            else:
                # No meaningful output -> reset ramp state
                elapsed_active_minutes = 0.0
            
            # Apply ramp factor
            ramp_factor = 1.0
            if elapsed_active_minutes < SOFT_START_RAMP_MINUTES:
                ramp_factor = elapsed_active_minutes / SOFT_START_RAMP_MINUTES
                if ramp_factor < SOFT_START_MIN_FACTOR: 
                    ramp_factor = SOFT_START_MIN_FACTOR
            
            if eff_derate != 1.0:
                q_hvac *= eff_derate
            
            q_hvac *= ramp_factor
            
        else:
            elapsed_active_minutes = 0.0
            hvac_produced_list[i] = 0.0
        
        hvac_outputs_list[i] = q_hvac

        # C. Integration
        q_total = q_leak + q_solar + Q_int + q_hvac
        delta_T = (q_total * dt_hours_list[i]) / C_thermal
        current_temp += delta_T

    return sim_temps_list, hvac_outputs_list, hvac_produced_list

def run_model_discrete(params, *, t_out_list, solar_kw_list, dt_hours_list, setpoint_list, hvac_state_list, max_caps_list, min_output, max_cool, eff_derate, start_temp, swing_temp, min_cycle_minutes):
    """
    Discrete Verification Model.
    Implements Hysteresis (Swing) + Minimum On/Off Timers.
    Returns actual_hvac_state + Diagnostics.
    """
    # Unpack params
    C_thermal = params[0]
    UA = params[1]
    K_solar = params[2]
    Q_int = params[3]
    H_factor = params[4]
    
    total_steps = len(t_out_list)
    sim_temps_list = [0.0] * total_steps
    hvac_outputs_list = [0.0] * total_steps
    hvac_produced_list = [0.0] * total_steps
    actual_hvac_state_list = [0] * total_steps 
    
    current_temp = float(start_temp)
    elapsed_active_minutes = 0.0
    
    # State Machine
    current_state = 0 # 0=Off, 1=Heat, -1=Cool
    min_on_timer = 0.0
    min_off_timer = 0.0
    
    # swing_temp is half-band: ±swing from setpoint
    
    # Diagnostics
    diag_cycles = 0
    diag_on_min = 0.0
    diag_off_min = 0.0
    diag_active_min = 0.0 # Based on produced > TOLERANCE
    TOLERANCE_BTU = 1.0

    for i in range(total_steps):
        sim_temps_list[i] = current_temp
        
        q_leak = UA * (t_out_list[i] - current_temp)
        q_solar = solar_kw_list[i] * K_solar
        dt_min = dt_hours_list[i] * 60.0
        
        # 1. Decrement Timers
        if min_on_timer > 0: min_on_timer -= dt_min
        if min_off_timer > 0: min_off_timer -= dt_min
        
        # 2. Determine Intent
        # req_intent is purely from schedule (+1/-1/0)
        req_intent = hvac_state_list[i]
        setpoint = setpoint_list[i]
        
        # 3. State Transition Logic
        next_state = current_state
        
        if current_state == 0:
            # OFF -> Need min_off_timer <= 0 to switch ON
            if min_off_timer <= 0 and req_intent != 0:
                # Check thresholds
                if req_intent > 0: # Heat
                    if current_temp < (setpoint - swing_temp):
                        next_state = 1
                        min_on_timer = min_cycle_minutes
                        diag_cycles += 1
                elif req_intent < 0: # Cool
                    if current_temp > (setpoint + swing_temp):
                        next_state = -1
                        min_on_timer = min_cycle_minutes
                        diag_cycles += 1
        else:
            # ON (1 or -1) -> Need min_on_timer <= 0 to switch OFF
            # Also forced OFF if disabled by schedule or Mode Swap
            
            # Check Force OFF conditions
            force_off = False
            if req_intent == 0: force_off = True # Schedule disabled
            elif current_state == 1 and req_intent == -1: force_off = True # Mode Swap Heat->Cool
            elif current_state == -1 and req_intent == 1: force_off = True # Mode Swap Cool->Heat
            
            if force_off:
                 # If timer locked, we might violate min_on? 
                 # Usually safety implies satisfying min-on first?
                 # OR safety implies shutting down if user disabled?
                 # Let's say user override respects timers? Or safety (Force Off) overrides timers?
                 # Assuming "Schedule Disable" respects timers? 
                 # For robust parity, let's say we hold ON until timer expires, UNLESS safety/emergency.
                 # Simplifying: Wait for timer.
                 if min_on_timer <= 0:
                     next_state = 0
                     min_off_timer = min_cycle_minutes
            else:
                # Normal Hysteresis Check
                if min_on_timer <= 0:
                    if current_state == 1: # Heat
                        if current_temp > (setpoint + swing_temp):
                            next_state = 0
                            min_off_timer = min_cycle_minutes
                    elif current_state == -1: # Cool
                        if current_temp < (setpoint - swing_temp):
                            next_state = 0
                            min_off_timer = min_cycle_minutes
        
        current_state = next_state
        actual_hvac_state_list[i] = current_state
        
        # 4. HVAC Physics (Discrete)
        q_hvac = 0.0
        if current_state != 0:
            elapsed_active_minutes += dt_min
            diag_on_min += dt_min
            
            # Ramp
            ramp_factor = 1.0
            if elapsed_active_minutes < SOFT_START_RAMP_MINUTES:
                ramp_factor = elapsed_active_minutes / SOFT_START_RAMP_MINUTES
                if ramp_factor < SOFT_START_MIN_FACTOR: ramp_factor = SOFT_START_MIN_FACTOR
            
            if current_state == 1: # Heat
                # Demand calculation: similar to continuous but gated by state
                gap = setpoint - current_temp
                # Usually we run harder in recovery? Using same P-logic for now.
                request = min_output + (H_factor * max(0, gap))
                if request < min_output: request = min_output # CLAMP to min if ON
                
                cap = max_caps_list[i]
                if request > cap: request = cap
                q_hvac = request
                
            elif current_state == -1: # Cool
                gap = current_temp - setpoint
                request = min_output + (H_factor * max(0, gap))
                if request < min_output: request = min_output
                
                if request > max_cool: request = max_cool
                q_hvac = -request
            
            hvac_produced_list[i] = q_hvac
            if abs(q_hvac) > TOLERANCE_BTU:
                diag_active_min += dt_min

            if eff_derate != 1.0: q_hvac *= eff_derate
            q_hvac *= ramp_factor
            
        else:
            elapsed_active_minutes = 0.0
            hvac_produced_list[i] = 0.0
            diag_off_min += dt_min
        
        hvac_outputs_list[i] = q_hvac
        
        q_total = q_leak + q_solar + Q_int + q_hvac
        delta_T = (q_total * dt_hours_list[i]) / C_thermal
        current_temp += delta_T

    sim_temps_list = sim_temps_list
    diagnostics = {
        "cycles": diag_cycles,
        "on_minutes": diag_on_min,
        "off_minutes": diag_off_min,
        "active_minutes": diag_active_min
    }
    
    return sim_temps_list, hvac_outputs_list, hvac_produced_list, actual_hvac_state_list, diagnostics



