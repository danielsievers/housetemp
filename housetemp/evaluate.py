import numpy as np
from . import run_model
from . import run_model
from .measurements import Measurements

def run_rolling_evaluation(data: Measurements, params, hw):
    """
    Runs a rolling window evaluation.
    - Starts at every hour (index 0, 2, 4...)
    - Runs a 12-hour prediction
    - Calculates RMSE for 6h and 12h horizons
    - Prints average RMSE
    """
    # hw is passed in
    
    # Assuming 30-minute intervals
    # TODO: Make dynamic based on dt_hours if possible, but 30min is standard here
    steps_per_hour = 2
    steps_6h = 6 * steps_per_hour
    steps_12h = 12 * steps_per_hour
    
    errors_6h = []
    errors_12h = []
    
    total_steps = len(data)
    
    # Iterate through data, starting at every hour
    # Stop when we don't have enough data for a full 12h forecast
    for start_idx in range(0, total_steps - steps_12h, steps_per_hour):
        
        # 1. Prepare Data Slice for this window
        # We need to create a new Measurements object or slice the existing one
        # run_model takes 'data' and uses it from index 0.
        # So we need to slice the arrays.
        
        # Slice length: 12 hours
        end_idx = start_idx + steps_12h
        
        # Create a sliced Measurements object
        # Note: We need to be careful about t_in. 
        # run_model uses t_in[0] as start temp.
        # So t_in[start_idx] is the correct start temp.
        
        sliced_data = Measurements(
            timestamps=data.timestamps[start_idx:end_idx],
            t_in=data.t_in[start_idx:end_idx],
            t_out=data.t_out[start_idx:end_idx],
            solar_kw=data.solar_kw[start_idx:end_idx],
            hvac_state=data.hvac_state[start_idx:end_idx],
            setpoint=data.setpoint[start_idx:end_idx],
            dt_hours=data.dt_hours[start_idx:end_idx]
        )
        
        # 2. Run Simulation (12 hours)
        # We don't need to pass duration_minutes because the slice is exactly 12h
        sim_temps, _ = run_model.run_model(params, sliced_data, hw)
        
        # 3. Calculate Errors
        actual_temps = sliced_data.t_in
        
        # 6-Hour Point Error
        # We want the error specifically at the 6h mark (index steps_6h - 1)
        idx_6h = steps_6h - 1
        error_6h = abs(sim_temps[idx_6h] - actual_temps[idx_6h])
        errors_6h.append(error_6h)
        
        # 12-Hour Point Error
        idx_12h = steps_12h - 1
        error_12h = abs(sim_temps[idx_12h] - actual_temps[idx_12h])
        errors_12h.append(error_12h)
        
    # --- REPORT ---
    avg_rmse_6h = np.mean(errors_6h)
    avg_rmse_12h = np.mean(errors_12h)
    
    print("\n" + "="*40)
    print("ROLLING EVALUATION RESULTS")
    print("="*40)
    print(f"Data Points Evaluated: {len(errors_12h)} windows")
    print(f"6-Hour Forecast Error (Avg):  {avg_rmse_6h:.3f} F")
    print(f"12-Hour Forecast Error (Avg): {avg_rmse_12h:.3f} F")
    print("="*40)
