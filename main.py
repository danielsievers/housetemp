#!/usr/bin/python3

import argparse
import json
import os
import sys
import load_csv
import optimize
import results
import linear_fit
import energy

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

def main():
    parser = argparse.ArgumentParser(
        description="Home Thermal Model Optimizer & Estimator",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Input Data (Made optional here so we can print help if missing)
    parser.add_argument("csv_file", nargs='?', help="Path to input CSV data (Home Assistant History)")
    
    # Mode Flags
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-r", "--reg-only", action="store_true", 
                        help="Run Linear Regression only (Passive fit) and exit.\n"
                             "Useful for checking C and UA without waiting for full optimization.")
    group.add_argument("-e", "--estimate", metavar="MODEL_JSON", 
                        help="Load an existing model JSON file and run Energy Estimation only.\n"
                             "Skips optimization. Applies saved physics to new data.")
    
    # Output Options
    parser.add_argument("-o", "--output", default="model.json", 
                        help="Filename to save optimized model parameters (default: model.json)")

    # --- CUSTOM HELP DISPLAY ---
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nUsage Examples:")
        print("  1. Train Model (Full Optimization):")
        print("     python main.py january_data.csv -o my_house.json")
        print("\n  2. Quick Physics Check (Linear Fit Only):")
        print("     python main.py january_data.csv -r")
        print("\n  3. Estimate Energy Cost (Using saved model on new data):")
        print("     python main.py february_data.csv -e my_house.json")
        sys.exit(1)

    args = parser.parse_args()

    if not args.csv_file:
        print("Error: You must provide a CSV file.")
        sys.exit(1)

    # 1. Load Data
    measurements = load_csv.load_csv(args.csv_file)
    
    # --- MODE 1: ENERGY ESTIMATION (Existing Model) ---
    if args.estimate:
        print("\n--- RUNNING IN ESTIMATION MODE ---")
        params = load_model(args.estimate)
        results.plot_results(measurements, params)
        energy.estimate_consumption(measurements, params, cost_per_kwh=0.45)
        return

    # --- MODE 2: LINEAR CHECK (Passive Fit) ---
    initial_params = linear_fit.linear_fit(measurements)
    
    if args.reg_only:
        print("\n" + "="*40)
        print("LINEAR REGRESSION RESULTS (Passive Only)")
        print("="*40)
        print(f"Thermal Mass (C):      {initial_params[0]:.0f}")
        print(f"Insulation (UA):       {initial_params[1]:.0f}")
        print(f"Solar Factor (K):      {initial_params[2]:.0f}")
        print(f"Internal Heat (Q_int): {initial_params[3]:.0f} (Fixed)")
        
        results.plot_results(measurements, initial_params)
        return

    # --- MODE 3: FULL OPTIMIZATION (Train Model) ---
    optimization_result = optimize.run_optimization(measurements, initial_guess=initial_params)
    
    if optimization_result.success:
        print("\nOptimization converged successfully!")
        best_params = optimization_result.x
        save_model(best_params, args.output)
        results.plot_results(measurements, best_params)
        energy.estimate_consumption(measurements, best_params, cost_per_kwh=0.45)
    else:
        print("Optimization failed:", optimization_result.message)

if __name__ == "__main__":
    main()
