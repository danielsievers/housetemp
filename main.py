#!/usr/bin/python3

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
from housetemp import load_csv
from housetemp import optimize
from housetemp import results
from housetemp import linear_fit
from housetemp import energy
from housetemp import evaluate
from housetemp import run_model

from housetemp import schedule
from housetemp.run_model import run_model_continuous
import numpy as np

def run_simulation_local(params, data, hw, duration_minutes=0):
    """Local helper to unpack data for run_model_continuous."""
    # Slicing Logic
    t_out = data.t_out.tolist()
    solar_kw = data.solar_kw.tolist()
    dt_hours = data.dt_hours.tolist()
    setpoint = data.setpoint.tolist()
    hvac_state = data.hvac_state.tolist()
    
    if duration_minutes > 0:
        avg_dt = np.mean(data.dt_hours) * 60
        if avg_dt > 0:
            steps = int(duration_minutes / avg_dt)
            steps = min(steps, len(t_out))
            t_out = t_out[:steps]
            solar_kw = solar_kw[:steps]
            dt_hours = dt_hours[:steps]
            setpoint = setpoint[:steps]
            hvac_state = hvac_state[:steps]
    
    # Caps
    t_out_np = np.array(t_out)
    max_caps = hw.get_max_capacity(t_out_np).tolist() if hw else [0.0]*len(t_out_np)
    
    start_temp = float(data.t_in[0])
    eff_derate = params[5] if len(params) > 5 else 1.0

    sim_temps, delivered, produced = run_model_continuous(
        params,
        t_out_list=t_out,
        solar_kw_list=solar_kw,
        dt_hours_list=dt_hours,
        setpoint_list=setpoint,
        hvac_state_list=hvac_state,
        max_caps_list=max_caps,
        min_output=hw.min_output_btu_hr if hw else 0,
        max_cool=hw.max_cool_btu_hr if hw else 0,
        eff_derate=eff_derate,
        start_temp=start_temp
    )
    
    # Calculate RMSE
    sim_np = np.array(sim_temps)
    # Use original t_in sliced safely
    actual = data.t_in[:len(sim_np)]
    rmse = np.sqrt(np.mean((sim_np - actual)**2))
    
    return sim_temps, rmse, delivered, produced, np.array(hvac_state)

def save_model(params, filename):
    data = {
        "description": "Home Thermal Model Parameters",
        "C_thermal": params[0],
        "UA_overall": params[1],
        "K_solar": params[2],
        "Q_int": params[3],
        "H_factor": params[4]
    }
    # Add optional efficiency if present
    if len(params) > 5:
        data["efficiency_derate"] = params[5]
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\nModel parameters saved to: {filename}")

def load_model(filename):
    if not os.path.exists(filename):
        print(f"Error: Model file '{filename}' not found.")
        sys.exit(1)
        
    with open(filename, 'r') as f:
        # Support C-style // comments to allow user annotations
        content = f.read()
        import re
        content = re.sub(r'//.*', '', content)
        data = json.loads(content)
    
    # Reconstruct params list from named keys
    # Order: [C, UA, K, Q, H, [efficiency_derate]]
    try:
        params = [
            data['C_thermal'],
            data['UA_overall'],
            data['K_solar'],
            data['Q_int'],
            data['H_factor']
        ]
        
        if 'efficiency_derate' in data:
            params.append(data['efficiency_derate'])
        else:
            # Default to 1.0 if not present, but only if we need it? 
            # Actually, run_model logic handles if it's missing (len 5 vs 6).
            # But let's append it if we want to be explicit, or just leave it as 5 items.
            # existing run_model handles len > 5 check.
            pass
            
        print(f"Loaded model from {filename}")
        print(f" -> C: {data['C_thermal']:.0f}, UA: {data['UA_overall']:.0f}")
        return params
        
    except KeyError as e:
        print(f"Error: Missing required parameter in {filename}: {e}")
        sys.exit(1)

def export_debug_output(filename, mode, params, measurements, sim_temps, hvac_outputs, 
                        energy_kwh=None, rmse=None, comfort_config=None, 
                        target_temps=None):
    """Export debug results to JSON for agent/automation use."""
    import datetime
    
    # Convert numpy arrays to lists for JSON serialization
    def to_list(arr):
        if arr is None:
            return None
        return [float(x) if not isinstance(x, (str, np.datetime64)) else str(x) for x in arr]
    
    debug_data = {
        "generated_at": datetime.datetime.now().isoformat(),
        "mode": mode,
        "model_params": {
            "C_thermal": float(params[0]),
            "UA_overall": float(params[1]),
            "K_solar": float(params[2]),
            "Q_int": float(params[3]),
            "H_factor": float(params[4])
        },
        "summary": {
            "rmse": float(rmse) if rmse else None,
            "energy_kwh": float(energy_kwh) if energy_kwh else None,
            "num_steps": len(sim_temps)
        },
        "comfort_config": comfort_config,
        "timeseries": {
            "timestamps": to_list(measurements.timestamps),
            "actual_temp": to_list(measurements.t_in[:len(sim_temps)]),
            "sim_temp": to_list(sim_temps),
            "setpoint": to_list(measurements.setpoint[:len(sim_temps)]),
            "hvac_output_btu": to_list(hvac_outputs),
            "outdoor_temp": to_list(measurements.t_out[:len(sim_temps)]),
            "target_temp": to_list(target_temps[:len(sim_temps)] if target_temps is not None else None)
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(debug_data, f, indent=2)
    print(f"Debug output saved to: {filename}")

def run_main(args_list=None):
    parser = argparse.ArgumentParser(
        description="Home Thermal Model Optimizer & Estimator",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Input Data (Made optional here so we can print help if missing)
    parser.add_argument("csv_file", nargs='?', help="Path to input CSV data (Home Assistant History)")
    
    # Optional Start Temp (for forecast data)
    parser.add_argument("--start-temp", type=float, help="Override starting indoor temperature (F). Useful for forecast data.")

    # Upsampling
    parser.add_argument("--upsample", nargs='?', const='1min', 
                        help="Upsample data to higher resolution (e.g. '1min', '5min').\n"
                             "Defaults to '1min' if flag is present without value.\n"
                             "Recommended for forecast data to ensure physics stability.")

    # Mode Flags
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-r", "--use-regression", action="store_true", 
                        help="Initialize with Linear Regression (Passive fit) before optimization.\n"
                             "If not specified, uses hardcoded defaults.")
    group.add_argument("-p", "--predict", metavar="MODEL_JSON", 
                        help="Load an existing model JSON file and run Prediction/Estimation.\n"
                             "Skips optimization. Applies saved physics to new data.")
    group.add_argument("-e", "--eval", metavar="MODEL_JSON",
                        help="Run Rolling Evaluation (6h/12h RMSE) over the dataset.\n"
                             "Requires a model JSON file.")
    
    # HVAC Optimization Flags
    parser.add_argument("--optimize-hvac", action="store_true", help="Optimize HVAC schedule to minimize energy (requires -p)")
    parser.add_argument("--comfort", metavar="JSON_FILE", help="Path to comfort schedule JSON (required for --optimize-hvac)")
    parser.add_argument("--control-timestep", type=int, default=30, help="Optimization setpoint block size in minutes (default: 30)")
    
    
    # Output Options
    parser.add_argument("--duration", type=int, default=0, help="Limit simulation duration in minutes (default: 0 = full duration)")
    parser.add_argument("--heat-pump", default="data/heat_pump.json", help="Path to Heat Pump JSON config (default: data/heat_pump.json)")
    parser.add_argument("-o", "--output", default="model.json", 
                        help="Filename to save optimized model parameters (default: model.json)")
    parser.add_argument("--debug-output", metavar="JSON_FILE",
                        help="Export debug results to JSON file (for agent/automation use)")
    
    parser.add_argument("--fix-passive", metavar="VACATION_MODEL_JSON",
                        help="Fix passive parameters (C, UA, K, Q) from this file and ONLY optimize active parameters.")
                        
    # --- CUSTOM HELP DISPLAY ---
    # If no args provided (and not called programmatically with empty list), show help
    if args_list is None and len(sys.argv) == 1:
        parser.print_help()
        print("\nUsage Examples:")
        print("  1. Train Model (Hardcoded Defaults):")
        print("     python main.py january_data.csv -o my_house.json")
        print("\n  2. Train Model (Initialize with Regression):")
        print("     python main.py january_data.csv -r -o my_house.json")
        print("\n  3. Predict/Estimate (Using saved model on new data):")
        print("     python main.py forecast_data.csv -p my_house.json")
        print("\n  4. Predict with Custom Start Temp:")
        print("     python main.py forecast_data.csv -p my_house.json --start-temp 68.0")
        print("\n  5. Optimize HVAC Schedule:")
        print("     python main.py forecast.csv -p my_house.json --optimize-hvac --comfort comfort.json")
        print("\n  6. Active-Only Optimization (Fix Passive Params):")
        print("     python main.py occupied.csv --fix-passive vacation.json -o active.json")
        return 1

    args = parser.parse_args(args_list)

    if not args.csv_file:
        print("Error: You must provide a CSV file.")
        return 1
        
    # Validation
    if args.fix_passive and args.use_regression:
        print("Error: You cannot use --use-regression with --fix-passive (params are fixed).")
        return 1

    if args.optimize_hvac and not args.predict:
        print("Error: --optimize-hvac currently requires --predict <MODEL_JSON> to be specified.")
        print("       (Optimization of schedule requires a pre-loaded model)")
        return 1

    # 1. Load Data
    measurements = load_csv.load_csv(args.csv_file, override_start_temp=args.start_temp, upsample_freq=args.upsample)
    
    # 2. Load Heat Pump Config (Required for Optimization, Prediction, Evaluation)
    # Always load it since all modes now require it (Optimization is the default fall-through).
    
    hw = None
    if not os.path.exists(args.heat_pump):
        print(f"Error: Heat Pump config file '{args.heat_pump}' not found.")
        print("You must provide a valid JSON config via --heat-pump.")
        return 1
    try:
        hw = run_model.HeatPump(args.heat_pump)
    except Exception as e:
        print(f"Error loading heat pump config: {e}")
        return 1
    
    # --- MODE 1: PREDICTION / ESTIMATION (Existing Model) ---
    if args.predict:
        print("\n--- RUNNING IN PREDICTION MODE ---")
        params = load_model(args.predict)
        
        if args.optimize_hvac:
            if not args.comfort:
                print("Error: --optimize-hvac requires --comfort <json_file>")
                return 1
            
            print("\n--- OPTIMIZING HVAC SCHEDULE ---")
            
            # Slice data if duration is set
            if args.duration > 0:
                # Find index where accumulated time exceeds duration
                sim_start = measurements.timestamps[0]
                sim_end = sim_start + pd.Timedelta(minutes=args.duration)
                end_time_np = np.datetime64(sim_end)
                limit_idx = np.searchsorted(measurements.timestamps, end_time_np)
                
                if limit_idx < len(measurements):
                    print(f"Limiting optimization to first {args.duration} minutes ({limit_idx} steps).")
                    measurements = measurements.slice(0, limit_idx)
            
            # Load Comfort Schedule
            target_temps, comfort_config = schedule.load_comfort_schedule(args.comfort, measurements.timestamps)
            
            # Run Optimization
            block_size = args.control_timestep
            # Default to Legacy (Single-Scale) on Desktop unless flag added later
            opt_result = optimize.optimize_hvac_schedule(measurements, params, hw, target_temps, comfort_config, block_size_minutes=block_size, enable_multiscale=False)
            
            # Unpack
            if isinstance(opt_result, tuple):
                 optimized_setpoints, meta = opt_result
            else:
                 # Should not happen
                 optimized_setpoints = opt_result
                 meta = {'success': True}
                 
            if optimized_setpoints is None:
                 print(f"Error: Optimization Failed: {meta.get('message')}")
                 sys.exit(1)

            # Update Measurements with Optimized Schedule
            measurements.setpoint[:] = optimized_setpoints
            
            # Set HVAC Mode based on config
            mode_str = comfort_config.get('mode', '').lower()
            if mode_str == 'heat':
                measurements.hvac_state[:] = 1
            elif mode_str == 'cool':
                measurements.hvac_state[:] = -1
            else:
                raise ValueError("Comfort config must specify 'mode': 'heat' or 'cool'. Auto mode is no longer supported.")
            
            title_suffix = "Optimized Schedule"
            marker_interval = block_size
        else:
            title_suffix = "Prediction Mode"
            marker_interval = None
            target_temps = None

        if args.optimize_hvac:
             setpoint_label = "Optimized Setpoint"
        else:
             setpoint_label = "Historical Setpoint"

        energy_result = energy.estimate_consumption(measurements, params, hw, cost_per_kwh=0.45)
        if not args.debug_output:
            results.plot_results(measurements, params, hw, title_suffix=title_suffix, duration_minutes=args.duration, marker_interval_minutes=marker_interval, target_temps=target_temps, energy_stats=energy_result, setpoint_label=setpoint_label)

        
        # Debug output (optional)
        if args.debug_output:
            sim_temps, rmse, hvac_delivered, hvac_produced, _ = run_simulation_local(params, measurements, hw, duration_minutes=args.duration)
            export_debug_output(
                args.debug_output,
                mode="optimize-hvac" if args.optimize_hvac else "predict",
                params=params,
                measurements=measurements,
                sim_temps=sim_temps,
                hvac_outputs=hvac_delivered,
                energy_kwh=energy_result.get('total_kwh') if energy_result else None,
                rmse=rmse,
                comfort_config=comfort_config if args.optimize_hvac else None,
                target_temps=target_temps
            )
        return 0

    # --- MODE 1.5: ROLLING EVALUATION ---
    if args.eval:
        print("\n--- RUNNING ROLLING EVALUATION ---")
        params = load_model(args.eval)
        evaluate.run_rolling_evaluation(measurements, params, hw)
        return 0

    # --- MODE 2: INITIALIZATION (Regression or Defaults) ---
    initial_params = None
    fixed_passive_params = None
    
    if args.fix_passive:
        # Load Passive params from file
        full_source_params = load_model(args.fix_passive)
        # Extract [C, UA, K, Q]
        fixed_passive_params = full_source_params[:4]
        print("Passive Parameters FIXED for optimization.")
    
    elif args.use_regression:
        print("\n--- RUNNING LINEAR REGRESSION (Initialization) ---")
        initial_params = linear_fit.linear_fit(measurements)
        
        print("\n" + "="*40)
        print("LINEAR REGRESSION RESULTS")
        print("="*40)
        print(f"Thermal Mass (C):      {initial_params[0]:.0f}")
        print(f"Insulation (UA):       {initial_params[1]:.0f}")
        print(f"Solar Factor (K):      {initial_params[2]:.0f}")
        print(f"Internal Heat (Q_int): {initial_params[3]:.0f} (Fixed)")
        
    # --- MODE 3: FULL OPTIMIZATION (Train Model) ---
    fixed_eff_derate = None
    if args.fix_passive and fixed_passive_params:
         # Try to extract efficiency from the loaded model (Index 5)
         full_source_params = load_model(args.fix_passive)
         if len(full_source_params) > 5:
             fixed_eff_derate = full_source_params[5]
             print(f"Using fixed efficiency_derate from file: {fixed_eff_derate}")
         else:
             fixed_eff_derate = optimize.DEFAULT_EFFICIENCY_DERATE
             print(f"File missing efficiency_derate. Using default: {fixed_eff_derate}")

    optimization_result = optimize.run_optimization(
        measurements, 
        hw, 
        initial_guess=initial_params, 
        fixed_passive_params=fixed_passive_params,
        fixed_efficiency_derate=fixed_eff_derate
    )
    
    if optimization_result.success:
        print("\nOptimization converged successfully!")
        best_params = optimization_result.x
        save_model(best_params, args.output)
        energy_result = energy.estimate_consumption(measurements, best_params, hw, cost_per_kwh=0.45)
        results.plot_results(measurements, best_params, hw, title_suffix="Optimization Result", energy_stats=energy_result, setpoint_label="Historical Setpoint")
        return 0
    else:
        print("Optimization failed:", optimization_result.message)
        return 1

if __name__ == "__main__":
    sys.exit(run_main())
