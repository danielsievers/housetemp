import json
import numpy as np
import logging
from .measurements import Measurements

_LOGGER = logging.getLogger(__name__)

# --- DEFAULT OVERRIDES (Fallbacks) ---
DEFAULT_EFFICIENCY_DERATE = 1.0

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

def run_model_fast(params, t_out_list, solar_kw_list, dt_hours_list, setpoint_list, hvac_state_list, max_caps_list, min_output, max_cool, eff_derate, start_temp):
    """
    Fastest possible simulation loop for HA Green (ARM).
    Accepts PURE PYTHON LISTS (not numpy arrays) for maximum scalar iteration speed.
    
    Returns:
        sim_temps_list: Indoor temperature history.
        hvac_outputs_list: Delivered Heat (Net) - After Derate & Soft Start. (Physics)
        hvac_produced_list: Produced Heat (Gross) - Before Derate & Ramp. (Energy Bill)
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
    hvac_produced_list = [0.0] * total_steps # Unramped for Energy Calc
    
    current_temp = float(start_temp)
    
    # Pre-resolve len to avoid lookup
    # Actually range(total_steps) is efficient enough
    elapsed_active_minutes = 0.0
    prev_mode = 0
    
    for i in range(total_steps):
        sim_temps_list[i] = current_temp
        
        # A. Passive Physics
        q_leak = UA * (t_out_list[i] - current_temp)
        q_solar = solar_kw_list[i] * K_solar
        
        # B. Active HVAC Physics
        q_hvac = 0.0
        mode = hvac_state_list[i]
        
        if mode != 0:
            # Soft Start / Thermal Inertia Logic
            # High static ducts take time to pressurize/heat. 
            # Prevents optimizer from "bursting" heat for 1-2 mins.
            if mode != prev_mode:
                # Just started (or switched mode)
                elapsed_active_minutes = 0.0
            
            # Increment elapsed time
            dt_min = dt_hours_list[i] * 60.0
            elapsed_active_minutes += dt_min
            
            # 5-minute ramp up to full capacity
            ramp_factor = 1.0
            if elapsed_active_minutes < SOFT_START_RAMP_MINUTES:
                ramp_factor = elapsed_active_minutes / SOFT_START_RAMP_MINUTES
                if ramp_factor < SOFT_START_MIN_FACTOR: ramp_factor = SOFT_START_MIN_FACTOR # Avoid true zero if dt is tiny? No, 0 is fine.

            
            if mode > 0: # HEATING
                gap = setpoint_list[i] - current_temp
                if gap > 0:
                    request = min_output + (H_factor * gap)
                    cap = max_caps_list[i]
                    if request > cap:
                        q_hvac = cap
                    else:
                        q_hvac = request
                else:
                    q_hvac = 0.0
                
            elif mode < 0: # COOLING
                gap = current_temp - setpoint_list[i]
                if gap > 0:
                    request = min_output + (H_factor * gap)
                    if request > max_cool:
                        q_hvac = -max_cool
                    else:
                        q_hvac = -request
                else:
                    q_hvac = 0.0
            
            # Capture Gross "Produced" Energy (Before Derate, Before Ramp)
            hvac_produced_list[i] = q_hvac
            
            if eff_derate != 1.0:
                q_hvac *= eff_derate
                
            # Apply Soft Start Ramp
            q_hvac *= ramp_factor
        
        prev_mode = mode # Update state tracker
        
        hvac_outputs_list[i] = q_hvac
        # If mode==0, produced is 0.0 (default)

        # C. Total Energy / Integration
        q_total = q_leak + q_solar + Q_int + q_hvac
        delta_T = (q_total * dt_hours_list[i]) / C_thermal
        current_temp += delta_T

    return sim_temps_list, hvac_outputs_list, hvac_produced_list

def run_model(params, data: Measurements, hw: HeatPump = None, duration_minutes: int = 0):
    # --- 1. Unpack Parameters ---
    # Optional efficiency
    eff_derate = DEFAULT_EFFICIENCY_DERATE
    if len(params) > 5:
        eff_derate = params[5]

    # --- 2. Determines Limits ---
    # Convert to list for speed
    t_out_list = data.t_out.tolist()
    solar_kw_list = data.solar_kw.tolist()
    dt_hours_list = data.dt_hours.tolist()
    
    if hw:
        max_caps_list = hw.get_max_capacity(data.t_out).tolist()
        min_output = hw.min_output_btu_hr
        max_cool = hw.max_cool_btu_hr
        setpoint_list = data.setpoint.tolist()
        hvac_state_list = data.hvac_state.tolist()
    else:
        max_caps_list = [0.0] * len(data.t_out)
        min_output = 0
        max_cool = 0
        setpoint_list = [0.0] * len(data.t_out)
        hvac_state_list = [0] * len(data.t_out)
    
    # --- 3. Determine Simulation Steps ---
    total_steps = len(data.t_out) # Default to full
    
    # Handle duration_minutes by truncation of input lists
    if duration_minutes > 0:
        avg_dt_minutes = np.mean(data.dt_hours) * 60
        if avg_dt_minutes > 0:
            steps_needed = int(duration_minutes / avg_dt_minutes)
            total_steps = min(total_steps, steps_needed)
            
            # Truncate lists
            t_out_list = t_out_list[:total_steps]
            solar_kw_list = solar_kw_list[:total_steps]
            dt_hours_list = dt_hours_list[:total_steps]
            max_caps_list = max_caps_list[:total_steps]
            setpoint_list = setpoint_list[:total_steps]
            hvac_state_list = hvac_state_list[:total_steps]

    # --- 4. Call Kernel ---
    sim_temps_list, hvac_delivered_list, hvac_produced_list = run_model_fast(
        params, t_out_list, solar_kw_list, dt_hours_list, setpoint_list, hvac_state_list, 
        max_caps_list, min_output, max_cool, eff_derate, data.t_in[0]
    )

    # --- 5. Calculate Error & Return ---
    sim_temps = np.array(sim_temps_list)
    hvac_delivered = np.array(hvac_delivered_list)
    hvac_produced = np.array(hvac_produced_list)
    
    actual_temps = data.t_in[:len(sim_temps)]
    
    mse = np.mean((sim_temps - actual_temps)**2)
    rmse = np.sqrt(mse)

    # Return delivered (for physics plotting) AND produced (for energy calc)
    return sim_temps, rmse, hvac_delivered, hvac_produced
