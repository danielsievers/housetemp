import matplotlib.pyplot as plt
import numpy as np
import sys
from . import run_model
from . import run_model

def plot_results(data, optimized_params, hw, title_suffix="", duration_minutes=0, marker_interval_minutes=None, target_temps=None, energy_stats=None):
    # hw is passed in
    
    # Run final simulation with best params
    simulated_t_in, rmse, hvac_outputs, _ = run_model.run_model(optimized_params, data, hw, duration_minutes=duration_minutes)
    
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
    if len(optimized_params) > 5:
        print(f"Efficiency Derate:     {optimized_params[5]*100:.1f}%")
    
    if energy_stats:
        print("-" * 40)
        print(f"Est. Energy:           {energy_stats['total_kwh']:.2f} kWh")
        print(f"Est. Cost:             ${energy_stats['total_cost']:.2f}")
    print("="*40)
    sys.stdout.flush()

    # --- PLOT ---
    plt.figure(figsize=(14, 8))
    
    # Slice data to match simulation length
    sim_len = len(simulated_t_in)
    timestamps_raw = data.timestamps[:sim_len]
    
    # Convert to local timezone for display (naive datetime for matplotlib)
    import pandas as pd
    ts_idx = pd.to_datetime(timestamps_raw)
    if ts_idx.tz is not None:
        timestamps = ts_idx.tz_convert('America/Los_Angeles').tz_localize(None)
    else:
        timestamps = ts_idx.tz_localize('UTC').tz_convert('America/Los_Angeles').tz_localize(None)
    actual_t_in = data.t_in[:sim_len]
    outdoor_t = data.t_out[:sim_len]
    
    # Subplot 1: Temperature
    plt.subplot(2, 1, 1)
    

    # Plot Ideal Target (if provided)
    if target_temps is not None:
        targets = target_temps[:sim_len]
        plt.plot(timestamps, targets, label='Ideal Target', color='green', linestyle=':', linewidth=1.5, alpha=0.7)

    plt.plot(timestamps, actual_t_in, label='Actual Indoor', color='grey', linewidth=2)
    plt.plot(timestamps, simulated_t_in, label='Model Prediction', color='orange', linestyle='--', linewidth=2)
    # Filter target temp markers if interval is specified
    target_timestamps = timestamps
    target_values = data.setpoint[:sim_len]
    
    if marker_interval_minutes:
        # Simpler: Use numpy searchsorted
        # Convert to numpy array of datetime64 if not already
        ts_np = np.array(timestamps, dtype='datetime64[ns]')
        
        # Align start to marker interval
        import pandas as pd
        start_ts = pd.Timestamp(ts_np[0])
        end_ts = pd.Timestamp(ts_np[-1])
        
        freq = f"{marker_interval_minutes}min"
        aligned_start = start_ts.floor(freq)
        aligned_end = end_ts.ceil(freq)
        
        marker_times = pd.date_range(start=aligned_start, end=aligned_end, freq=freq).values
        
        # Find closest indices
        indices = np.searchsorted(ts_np, marker_times)
        # Clip to valid range
        indices = indices[indices < len(ts_np)]
        
        target_timestamps = timestamps[indices]
        target_values = data.setpoint[:sim_len][indices]

    plt.plot(target_timestamps, target_values, label='Optimized Setpoint', color='green', marker='x', linestyle='None', markersize=6, markeredgewidth=1.5, alpha=0.8)
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
    # hvac_state_sliced = data.hvac_state[:sim_len] # No longer needed for plotting output
    solar_kw_sliced = data.solar_kw[:sim_len]

    # Unpack params for plotting
    UA = optimized_params[1]
    K_solar = optimized_params[2]
    Q_int = optimized_params[3]
    # H_factor = optimized_params[4]

    # Calculate Flows
    # max_caps = hw.get_max_capacity(t_out_sliced)
    # gap = np.maximum(0, setpoint_sliced - simulated_t_in)
    # est_btu = 12000 + (H_factor * gap)
    # Apply hvac state mask (1, 0, -1) and limits
    # est_btu = np.where(hvac_state_sliced > 0, np.minimum(est_btu, max_caps), 
    #                    np.where(hvac_state_sliced < 0, -np.minimum(est_btu, 54000), 0))
    
    # Use the actual output from the simulation
    est_btu = hvac_outputs[:sim_len]
    
    q_leak = UA * (t_out_sliced - simulated_t_in)
    q_solar = solar_kw_sliced * K_solar
    q_total = est_btu + q_solar + q_leak + Q_int
    
    plt.plot(timestamps, est_btu, label='Modeled HVAC', color='red', alpha=0.6)
    plt.plot(timestamps, q_solar, label='Solar Gain', color='gold', alpha=0.6)
    plt.plot(timestamps, q_leak, label='Leakage (UA)', color='cyan', alpha=0.6, linestyle=':')
    plt.axhline(y=Q_int, label='Internal Heat', color='green', alpha=0.5, linestyle='--')
    plt.plot(timestamps, q_total, label='Net Gain', color='black', linewidth=1.5, linestyle='-')
    
    plt.ylabel("Heat Flow (BTU/hr)")
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    
    # Format x-axis dates
    import matplotlib.dates as mdates
    myFmt = mdates.DateFormatter('%m-%d %H:%M')
    
    # Apply to both subplots
    plt.subplot(2, 1, 1)
    plt.gca().xaxis.set_major_formatter(myFmt)
    
    plt.subplot(2, 1, 2)
    plt.gca().xaxis.set_major_formatter(myFmt)
    
    plt.tight_layout()
    plt.show()
