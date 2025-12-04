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

def save_model(params, filename):
    data = {
        "description": "Home Thermal Model Parameters",
        "C_thermal": params[0],
        "UA_overall": params[1],
        "K_solar": params[2],
        "Q_int": params[3],
        "H_factor": params[4],
        "raw_params": list(params)
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\nModel parameters saved to: {filename}")

def load_model(filename):
    if not os.path.exists(filename):
        print(f"Error: Model file '{filename}' not found.")
        sys.exit(1)
        
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded model from {filename}")
    print(f" -> C: {data['C_thermal']:.0f}, UA: {data['UA_overall']:.0f}")
    return data['raw_params']

def run_main(args_list=None):
    parser = argparse.ArgumentParser(
        description="Home Thermal Model Optimizer & Estimator",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Input Data (Made optional here so we can print help if missing)
    parser.add_argument("csv_file", nargs='?', help="Path to input CSV data (Home Assistant History)")
    
    # Optional Start Temp (for forecast data)
    parser.add_argument("--start-temp", type=float, help="Override starting indoor temperature (F). Useful for forecast data.")

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
    
    # Output Options
    parser.add_argument("--duration", type=int, default=0, help="Limit simulation duration in minutes (default: 0 = full duration)")
    parser.add_argument("--heat-pump", default="data/heat_pump.json", help="Path to Heat Pump JSON config (default: data/heat_pump.json)")
    parser.add_argument("-o", "--output", default="model.json", 
                        help="Filename to save optimized model parameters (default: model.json)")

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
        return 1

    args = parser.parse_args(args_list)

    if not args.csv_file:
        print("Error: You must provide a CSV file.")
        return 1

    # 1. Load Data
    measurements = load_csv.load_csv(args.csv_file, override_start_temp=args.start_temp)
    
    # 2. Load Heat Pump Config (Required for Optimization, Prediction, Evaluation)
    # It is NOT strictly required for Regression, but we might as well load it if present.
    # If missing, we error out unless we are ONLY doing regression (which doesn't use it).
    
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
                # Assuming dt_hours is in hours, duration is in minutes
                # Simpler: just use timestamps
                start_time = measurements.timestamps[0]
                end_time = start_time + pd.Timedelta(minutes=args.duration)
                
                # Find index
                # timestamps is numpy array of datetime64
                # Convert end_time to datetime64
                end_time_np = np.datetime64(end_time)
                
                # Searchsorted finds the index
                limit_idx = np.searchsorted(measurements.timestamps, end_time_np)
                
                if limit_idx < len(measurements):
                    print(f"Limiting optimization to first {args.duration} minutes ({limit_idx} steps).")
                    measurements = measurements.slice(0, limit_idx)
            
            # Load Comfort Schedule
            target_temps, min_bounds, max_bounds, comfort_config = schedule.load_comfort_schedule(args.comfort, measurements.timestamps)
            
            # Run Optimization
            block_size = 30
            optimized_setpoints = optimize.optimize_hvac_schedule(measurements, params, hw, target_temps, min_bounds, max_bounds, comfort_config, block_size_minutes=block_size)
            
            # Update Measurements with Optimized Schedule
            measurements.setpoint[:] = optimized_setpoints
            measurements.hvac_state[:] = 2 # Force Auto Mode
            
            title_suffix = "Optimized Schedule"
            marker_interval = block_size
        else:
            title_suffix = "Prediction Mode"
            marker_interval = None
            # In prediction mode, we don't have these loaded yet unless we load them
            # But plot_results handles None gracefully
            target_temps = None
            min_bounds = None
            max_bounds = None

        results.plot_results(measurements, params, hw, title_suffix=title_suffix, duration_minutes=args.duration, marker_interval_minutes=marker_interval, target_temps=target_temps, min_bounds=min_bounds, max_bounds=max_bounds)
        energy.estimate_consumption(measurements, params, hw, cost_per_kwh=0.45)
        return 0

    # --- MODE 1.5: ROLLING EVALUATION ---
    if args.eval:
        print("\n--- RUNNING ROLLING EVALUATION ---")
        params = load_model(args.eval)
        evaluate.run_rolling_evaluation(measurements, params, hw)
        return 0

    # --- MODE 2: INITIALIZATION (Regression or Defaults) ---
    initial_params = None
    
    if args.use_regression:
        print("\n--- RUNNING LINEAR REGRESSION (Initialization) ---")
        initial_params = linear_fit.linear_fit(measurements)
        
        print("\n" + "="*40)
        print("LINEAR REGRESSION RESULTS")
        print("="*40)
        print(f"Thermal Mass (C):      {initial_params[0]:.0f}")
        print(f"Insulation (UA):       {initial_params[1]:.0f}")
        print(f"Solar Factor (K):      {initial_params[2]:.0f}")
        print(f"Internal Heat (Q_int): {initial_params[3]:.0f} (Fixed)")
        
        # We don't exit here anymore, we proceed to optimization

    # --- MODE 3: FULL OPTIMIZATION (Train Model) ---
    optimization_result = optimize.run_optimization(measurements, hw, initial_guess=initial_params)
    
    if optimization_result.success:
        print("\nOptimization converged successfully!")
        best_params = optimization_result.x
        save_model(best_params, args.output)
        results.plot_results(measurements, best_params, hw, title_suffix="Optimization Result")
        energy.estimate_consumption(measurements, best_params, hw, cost_per_kwh=0.45)
        return 0
    else:
        print("Optimization failed:", optimization_result.message)
        return 1

if __name__ == "__main__":
    sys.exit(run_main())
