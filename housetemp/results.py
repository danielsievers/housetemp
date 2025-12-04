import matplotlib.pyplot as plt
import numpy as np
from . import run_model
from . import run_model

def plot_results(data, optimized_params, hw, title_suffix="", duration_minutes=0):
    # hw is passed in
    
    # Run final simulation with best params
    simulated_t_in, rmse, _ = run_model.run_model(optimized_params, data, hw, duration_minutes=duration_minutes)
    
    # Calculate Error (Manual Verification)
    # We need to slice data.t_in to match simulation length if duration was limited
    sim_len = len(simulated_t_in)
    manual_rmse = np.sqrt(np.mean((simulated_t_in - data.t_in[:sim_len])**2))
    
    # --- PRINT REPORT ---
    print("\n" + "="*40)
    print(f"OPTIMIZATION RESULTS")
    print(f"Simulation RMSE: {rmse:.3f} F")
    print(f"Calculated RMSE: {manual_rmse:.3f} F")
    print("="*40)
    print(f"Thermal Mass (C):      {optimized_params[0]:.0f} BTU/F")
    print(f"Insulation (UA):       {optimized_params[1]:.0f} BTU/hr/F")
    print(f"Solar Factor (K):      {optimized_params[2]:.0f}")
    print(f"Internal Heat (Q_int): {optimized_params[3]:.0f} BTU/hr")
    print(f"Inverter Gain (H_fac): {optimized_params[4]:.0f} BTU/deg")
    print("="*40)

    # --- PLOT ---
    plt.figure(figsize=(14, 8))
    
    # Slice data to match simulation length
    sim_len = len(simulated_t_in)
    timestamps = data.timestamps[:sim_len]
    actual_t_in = data.t_in[:sim_len]
    outdoor_t = data.t_out[:sim_len]
    
    # Subplot 1: Temperature
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, actual_t_in, label='Actual Indoor', color='grey', linewidth=2)
    plt.plot(timestamps, simulated_t_in, label='Model Prediction', color='orange', linestyle='--', linewidth=2)
    plt.plot(timestamps, outdoor_t, label='Outdoor', color='blue', alpha=0.3)
    plt.ylabel("Temperature (F)")
    
    title = "Model Simulation"
    if title_suffix:
        title += f" - {title_suffix}"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: HVAC Activity (Visualization of what the model thinks happened)
    plt.subplot(2, 1, 2)
    # Recalculate heat flow for plotting logic (simplified)
    # We need to slice inputs for calculation too
    t_out_sliced = data.t_out[:sim_len]
    setpoint_sliced = data.setpoint[:sim_len]
    hvac_state_sliced = data.hvac_state[:sim_len]
    solar_kw_sliced = data.solar_kw[:sim_len]

    max_caps = hw.get_max_capacity(t_out_sliced)
    gap = np.maximum(0, setpoint_sliced - simulated_t_in)
    est_btu = 12000 + (optimized_params[4] * gap)
    # Apply hvac state mask (1, 0, -1) and limits
    est_btu = np.where(hvac_state_sliced > 0, np.minimum(est_btu, max_caps), 
                       np.where(hvac_state_sliced < 0, -np.minimum(est_btu, 54000), 0))
    
    plt.plot(timestamps, est_btu, label='Modeled HVAC Output (BTU)', color='red', alpha=0.6)
    plt.plot(timestamps, solar_kw_sliced * optimized_params[2], label='Solar Gain (BTU)', color='gold', alpha=0.6)
    plt.ylabel("Heat Flow (BTU/hr)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
