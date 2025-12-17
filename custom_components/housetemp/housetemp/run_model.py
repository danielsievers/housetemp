import json
import numpy as np
from .measurements import Measurements

class HeatPump:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.cap_x = data['max_capacity']['x_outdoor_f']
        self.cap_y = data['max_capacity']['y_btu_hr']
        self.cop_x = data['cop']['x_outdoor_f']
        self.cop_y = data['cop']['y_cop']
        
        # Min/max operating parameters (with sensible defaults)
        self.min_output_btu_hr = data.get('min_output_btu_hr', 3000)
        self.max_cool_btu_hr = data.get('max_cool_btu_hr', 54000)
        
        # Part-Load Factor correction curve: PLF = plf_low_load - (plf_slope * load_ratio)
        # Mitsubishi inverters: 1.4 at low load, 1.0 at full load (slope = 0.4)
        self.plf_low_load = data.get('plf_low_load', 1.4)
        self.plf_slope = data.get('plf_slope', 0.4)
        
        # Defrost parameters (optional - None if not specified)
        if 'defrost' in data:
            defrost = data['defrost']
            self.defrost_trigger_temp = defrost.get('trigger_temp_f', 32)
            self.defrost_risk_zone = defrost.get('risk_zone_f', [28, 42])
            self.defrost_duration_min = defrost.get('cycle_duration_minutes', 10)
            self.defrost_interval_min = defrost.get('cycle_interval_minutes', 60)
            self.defrost_power_kw = defrost.get('power_kw', 4.5)
        else:
            self.defrost_risk_zone = None  # Signals no defrost modeling

    def get_max_capacity(self, t_out_array):
        return np.interp(t_out_array, self.cap_x, self.cap_y)

    def get_cop(self, t_out_array):
        return np.interp(t_out_array, self.cop_x, self.cop_y)

def run_model_fast(params, t_out_list, solar_kw_list, dt_hours_list, setpoint_list, hvac_state_list, max_caps_list, min_output, max_cool, eff_derate, start_temp):
    """
    Fastest possible simulation loop for HA Green (ARM).
    Accepts PURE PYTHON LISTS (not numpy arrays) for maximum scalar iteration speed.
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
    
    current_temp = float(start_temp)
    
    # Pre-resolve len to avoid lookup
    # Actually range(total_steps) is efficient enough
    
    for i in range(total_steps):
        sim_temps_list[i] = current_temp
        
        # A. Passive Physics
        q_leak = UA * (t_out_list[i] - current_temp)
        q_solar = solar_kw_list[i] * K_solar
        
        # B. Active HVAC Physics
        q_hvac = 0.0
        mode = hvac_state_list[i]
        
        if mode != 0:
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
            
            if eff_derate != 1.0:
                q_hvac *= eff_derate
        
        hvac_outputs_list[i] = q_hvac

        # C. Total Energy / Integration
        q_total = q_leak + q_solar + Q_int + q_hvac
        delta_T = (q_total * dt_hours_list[i]) / C_thermal
        current_temp += delta_T

    return sim_temps_list, hvac_outputs_list

def run_model(params, data: Measurements, hw: HeatPump = None, duration_minutes: int = 0):
    # --- 1. Unpack Parameters ---
    # Optional efficiency
    eff_derate = 1.0
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
    sim_temps_list, hvac_outputs_list = run_model_fast(
        params, t_out_list, solar_kw_list, dt_hours_list, setpoint_list, hvac_state_list, 
        max_caps_list, min_output, max_cool, eff_derate, data.t_in[0]
    )

    # --- 5. Calculate Error & Return ---
    sim_temps = np.array(sim_temps_list)
    hvac_outputs = np.array(hvac_outputs_list)
    actual_temps = data.t_in[:len(sim_temps)]
    
    mse = np.mean((sim_temps - actual_temps)**2)
    rmse = np.sqrt(mse)

    return sim_temps, rmse, hvac_outputs
