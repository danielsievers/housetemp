import matplotlib.pyplot as plt
import numpy as np
from . import run_model
from .heat_pump import MitsubishiHeatPump

def plot_results(data, optimized_params, title_suffix=""):
    hw = MitsubishiHeatPump()
    
    # Run final simulation with best params
    simulated_t_in = run_model.run_model(optimized_params, data, hw)
    
    # Calculate Error
    rmse = np.sqrt(np.mean((simulated_t_in - data.t_in)**2))
    
    # --- PRINT REPORT ---
    print("\n" + "="*40)
    print(f"OPTIMIZATION RESULTS (Error: {rmse:.3f} F)")
    print("="*40)
    print(f"Thermal Mass (C):      {optimized_params[0]:.0f} BTU/F")
    print(f"Insulation (UA):       {optimized_params[1]:.0f} BTU/hr/F")
    print(f"Solar Factor (K):      {optimized_params[2]:.0f}")
    print(f"Internal Heat (Q_int): {optimized_params[3]:.0f} BTU/hr")
    print(f"Inverter Gain (H_fac): {optimized_params[4]:.0f} BTU/deg")
    print("="*40)

    # --- PLOT ---
    plt.figure(figsize=(14, 8))
    
    # Subplot 1: Temperature
    plt.subplot(2, 1, 1)
    plt.plot(data.timestamps, data.t_in, label='Actual Indoor', color='grey', linewidth=2)
    plt.plot(data.timestamps, simulated_t_in, label='Model Prediction', color='orange', linestyle='--', linewidth=2)
    plt.plot(data.timestamps, data.t_out, label='Outdoor', color='blue', alpha=0.3)
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
    max_caps = hw.get_max_capacity(data.t_out)
    gap = np.maximum(0, data.setpoint - simulated_t_in)
    est_btu = 12000 + (optimized_params[4] * gap)
    # Apply hvac state mask (1, 0, -1) and limits
    est_btu = np.where(data.hvac_state > 0, np.minimum(est_btu, max_caps), 
                       np.where(data.hvac_state < 0, -np.minimum(est_btu, 54000), 0))
    
    plt.plot(data.timestamps, est_btu, label='Modeled HVAC Output (BTU)', color='red', alpha=0.6)
    plt.plot(data.timestamps, data.solar_kw * optimized_params[2], label='Solar Gain (BTU)', color='gold', alpha=0.6)
    plt.ylabel("Heat Flow (BTU/hr)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
