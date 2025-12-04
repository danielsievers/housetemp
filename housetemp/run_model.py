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

def run_model(params, data: Measurements, hw: HeatPump = None, duration_minutes: int = 0):
    # --- 1. Unpack Parameters (The things we are optimizing) ---
    C_thermal = params[0]   # Thermal Mass (BTU/F)
    UA = params[1]          # Insulation Leakage (BTU/hr/F)
    K_solar = params[2]     # Solar Gain Factor (BTU/hr per kW)
    Q_int = params[3]       # Internal Heat (BTU/hr)
    H_factor = params[4]    # Inverter Aggressiveness (BTU per degree gap)

    # --- 2. Pre-calculate Limits ---
    # We know the max capacity for every hour based on weather
    if hw:
        max_caps = hw.get_max_capacity(data.t_out)
    else:
        # If no hardware model, we assume no active HVAC capability
        # or infinite? For now, let's assume 0 capacity if no hardware defined.
        # This effectively forces Passive Mode.
        max_caps = np.zeros(len(data))
    
    # --- 3. Determine Simulation Steps ---
    total_steps = len(data)
    if duration_minutes > 0:
        # Calculate steps based on actual dt
        # dt_hours is in hours. duration_minutes is in minutes.
        avg_dt_minutes = np.mean(data.dt_hours) * 60
        if avg_dt_minutes > 0:
            steps_needed = int(duration_minutes / avg_dt_minutes)
            total_steps = min(total_steps, steps_needed)

    # --- 4. Simulation Loop ---
    sim_temps = np.zeros(total_steps)
    hvac_outputs = np.zeros(total_steps) # Track Q_hvac for energy calc
    current_temp = data.t_in[0] # Start at actual temp
    
    for i in range(total_steps):
        sim_temps[i] = current_temp
        
        # A. Passive Physics
        # Heat flowing IN/OUT through walls
        q_leak = UA * (data.t_out[i] - current_temp)
        
        # Heat from Sun
        q_solar = data.solar_kw[i] * K_solar
        
        # B. Active HVAC Physics (Inverter Logic)
        q_hvac = 0
        
        if hw and data.hvac_state[i] != 0: # If HVAC is enabled AND we have hardware
            # Calculate the "Gap" (Error)
            gap = data.setpoint[i] - current_temp
            
            mode = data.hvac_state[i]
            
            # AUTO MODE (2) - Decide based on gap
            if mode == 2:
                if gap > 0: mode = 1  # Need Heat
                elif gap < 0: mode = -1 # Need Cool
                else: mode = 0 # Satisfied
            
            if mode > 0: # HEATING
                gap = data.setpoint[i] - current_temp
                if gap > 0:
                    # Base load (3000) + Turbo Ramp
                    request = 3000 + (H_factor * gap)
                    # Clamp to hardware limits
                    q_hvac = min(request, max_caps[i])
                else:
                    q_hvac = 0
                
            elif mode < 0: # COOLING
                gap = current_temp - data.setpoint[i]
                if gap > 0:
                    request = 3000 + (H_factor * gap)
                    # Cap cooling ~54k
                    q_hvac = -min(request, 54000)
                else:
                    q_hvac = 0
        
        hvac_outputs[i] = q_hvac

        # C. Total Energy Balance
        q_total = q_leak + q_solar + Q_int + q_hvac
        
        # D. Temperature Change (Integration)
        # delta_T = (Net Heat / Mass) * Time Step
        delta_T = (q_total * data.dt_hours[i]) / C_thermal
        
        current_temp += delta_T

    # --- 5. Calculate Error (RMSE) ---
    # Only calculate error for the steps we simulated
    # And only if we have actual data (indoor_temp might be NaN or forecast)
    # For now, assuming data.t_in is populated (even if dummy for forecast)
    # If it's forecast data, t_in might be zeros or start temp, so error is meaningless?
    # User asked to calculate error "from the data".
    
    # We need to slice the actual data to match the simulation length
    actual_temps = data.t_in[:total_steps]
    
    # Check if actual_temps has meaningful data (not just start temp repeated or zeros)
    # But for optimization, we rely on this.
    mse = np.mean((sim_temps - actual_temps)**2)
    rmse = np.sqrt(mse)

    return sim_temps, rmse, hvac_outputs
