import numpy as np
from .measurements import Measurements
from .heat_pump import MitsubishiHeatPump

def run_model(params, data: Measurements, hw: MitsubishiHeatPump):
    # --- 1. Unpack Parameters (The things we are optimizing) ---
    C_thermal = params[0]   # Thermal Mass (BTU/F)
    UA = params[1]          # Insulation Leakage (BTU/hr/F)
    K_solar = params[2]     # Solar Gain Factor (BTU/hr per kW)
    Q_int = params[3]       # Internal Heat (BTU/hr)
    H_factor = params[4]    # Inverter Aggressiveness (BTU per degree gap)

    # --- 2. Pre-calculate Limits ---
    # We know the max capacity for every hour based on weather
    max_caps = hw.get_max_capacity(data.t_out)
    
    # --- 3. Simulation Loop ---
    sim_temps = np.zeros(len(data))
    current_temp = data.t_in[0] # Start at actual temp
    
    for i in range(len(data)):
        sim_temps[i] = current_temp
        
        # A. Passive Physics
        # Heat flowing IN/OUT through walls
        q_leak = UA * (data.t_out[i] - current_temp)
        
        # Heat from Sun
        q_solar = data.solar_kw[i] * K_solar
        
        # B. Active HVAC Physics (Inverter Logic)
        q_hvac = 0
        
        if data.hvac_state[i] != 0: # If HVAC is enabled
            # Calculate the "Gap" (Error)
            gap = data.setpoint[i] - current_temp
            
            if data.hvac_state[i] > 0: # HEATING
                gap = max(0, gap) # Only care if temp is too low
                # Base load (12k) + Turbo Ramp
                request = 3000 + (H_factor * gap)
                # Clamp to hardware limits
                q_hvac = min(request, max_caps[i])
                
            elif data.hvac_state[i] < 0: # COOLING
                gap = max(0, -gap) # Only care if temp is too high
                request = 3000 + (H_factor * gap)
                # Cap cooling ~54k
                q_hvac = -min(request, 54000)

        # C. Total Energy Balance
        q_total = q_leak + q_solar + Q_int + q_hvac
        
        # D. Temperature Change (Integration)
        # delta_T = (Net Heat / Mass) * Time Step
        delta_T = (q_total * data.dt_hours[i]) / C_thermal
        
        current_temp += delta_T

    return sim_temps
