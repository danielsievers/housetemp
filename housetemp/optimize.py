import numpy as np
from scipy.optimize import minimize
from . import run_model
from . import run_model

def loss_function(params, data, hw):
    # 1. Run Simulation with current parameter guess
    predicted_temps, sim_error, _ = run_model.run_model(params, data, hw)
    
    # 2. Compare to Reality (Root Mean Square Error)
    # We use the error returned by run_model directly, but let's verify:
    manual_error = np.sqrt(np.mean((predicted_temps - data.t_in)**2))
    
    # Print both (as requested) - Note: This will spam the console during optimization
    # print(f"DEBUG: Sim Error={sim_error:.6f}, Manual Error={manual_error:.6f}")
    
    # Assert that they match (sanity check)
    if not np.isclose(sim_error, manual_error, atol=1e-5):
        raise ValueError(f"Error mismatch! Sim: {sim_error}, Manual: {manual_error}")
    
    error = sim_error
    
    # 3. Penalize Physics Violations (Soft Constraints)
    if params[0] < 1000: return 1e6 # Mass too low
    if params[1] < 50: return 1e6   # UA too low
    
    return error

def run_optimization(data, hw, initial_guess=None):
    # hw is now passed in
    
    # [C, UA, K_solar, Q_int, H_factor]
    # If no linear fit provided, fall back to hardcoded defaults
    if initial_guess is None:
        initial_guess = [4732, 213, 836, 600, 10000]
    
    # Bounds for the solver
    # (min, max)
    bounds = [
# vacation run:
#        (4000, 20000),  # C (Mass) - Widen this. Let it go higher if it wants.
#        (150, 800),     # UA (Leakage) - TIGHTEN THIS. Cap at 500 (Reasonable Max).
#        (700, 2000),  # K_solar (Window Factor) - Lower cap (Low-E glass confirmed).
#        (200, 2500),    # Q_int (Internal Heat) - TIGHTEN THIS. Cap at 2500 (730 Watts).
#        (1000, 30000)   # H_factor (Inverter Ramp)

        (4000, 20000),  # C (Mass) - Widen this. Let it go higher if it wants.
        (200, 1500),     # UA (Leakage) - TIGHTEN THIS. Cap at 500 (Reasonable Max).
        (700, 2000),  # K_solar (Window Factor) - Lower cap (Low-E glass confirmed).
        (200, 10000),    # Q_int (Internal Heat) - TIGHTEN THIS. Cap at 2500 (730 Watts).
        (5000, 20000)   # H_factor (Inverter Ramp)
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
