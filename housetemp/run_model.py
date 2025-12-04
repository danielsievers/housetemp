import numpy as np
from .measurements import Measurements
from .heat_pump import MitsubishiHeatPump

def run_model(params, data: Measurements, hw: MitsubishiHeatPump, duration_minutes: int = 0):
    # --- 1. Unpack Parameters (The things we are optimizing) ---
    C_thermal = params[0]   # Thermal Mass (BTU/F)
    UA = params[1]          # Insulation Leakage (BTU/hr/F)
    K_solar = params[2]     # Solar Gain Factor (BTU/hr per kW)
    Q_int = params[3]       # Internal Heat (BTU/hr)
    H_factor = params[4]    # Inverter Aggressiveness (BTU per degree gap)

    # --- 2. Pre-calculate Limits ---
    # We know the max capacity for every hour based on weather
    max_caps = hw.get_max_capacity(data.t_out)
    
    # --- 3. Determine Simulation Steps ---
    total_steps = len(data)
    if duration_minutes > 0:
        # Assuming 30-minute intervals (0.5 hours)
        # TODO: Make this dynamic based on data.dt_hours if needed, but for now assuming uniform
        steps_needed = int(duration_minutes / 30)
        total_steps = min(total_steps, steps_needed)

    # --- 4. Simulation Loop ---
    sim_temps = np.zeros(total_steps)
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

    return sim_temps, rmse
