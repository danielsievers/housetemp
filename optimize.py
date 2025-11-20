import numpy as np
from scipy.optimize import minimize
import run_model
from heat_pump import MitsubishiHeatPump

def loss_function(params, data, hw):
    # 1. Run Simulation with current parameter guess
    predicted_temps = run_model.run_model(params, data, hw)
    
    # 2. Compare to Reality (Root Mean Square Error)
    error = np.sqrt(np.mean((predicted_temps - data.t_in)**2))
    
    # 3. Penalize Physics Violations (Soft Constraints)
    if params[0] < 1000: return 1e6 # Mass too low
    if params[1] < 50: return 1e6   # UA too low
    
    return error

def run_optimization(data, initial_guess=None):
    hw = MitsubishiHeatPump()
    
    # [C, UA, K_solar, Q_int, H_factor]
    # If no linear fit provided, fall back to hardcoded defaults
    if initial_guess is None:
        initial_guess = [8000, 310, 17000, 1900, 15000]
    
    # Bounds for the solver
    # (min, max)
    bounds = [
        (2000, 15000),  # C (Mass)
        (100, 1000),    # UA (Leakage)
        (5000, 40000),  # K_solar (Window Factor)
        (0, 5000),      # Q_int (Internal Heat)
        (5000, 30000)   # H_factor (Inverter Ramp)
    ]
    
    print("Starting Optimization (this may take a few seconds)...")
    result = minimize(
        loss_function, 
        initial_guess, 
        args=(data, hw), 
        bounds=bounds, 
        method='L-BFGS-B'
    )
    
    return result
