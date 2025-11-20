#!/usr/bin/python3

import argparse
import load_csv
import optimize
import results
import linear_fit 

def main():
    parser = argparse.ArgumentParser(description="Home Thermal Model Optimizer")
    parser.add_argument("csv_file", help="Path to input CSV data")
    
    # NEW ARGUMENT
    parser.add_argument("-r", "--reg-only", action="store_true", 
                        help="Run Linear Regression only (Passive fit) and exit.")
    
    args = parser.parse_args()

    # 1. Load Data
    measurements = load_csv.load_csv(args.csv_file)
    
    # 2. Run Linear Regression (First Pass)
    # This looks at passive cooldowns to find C and UA mathematically
    initial_params = linear_fit.linear_fit(measurements)
    
    # If user only wanted the regression check:
    if args.reg_only:
        print("\n" + "="*40)
        print("LINEAR REGRESSION RESULTS (Passive Only)")
        print("="*40)
        print(f"Thermal Mass (C):      {initial_params[0]:.0f}")
        print(f"Insulation (UA):       {initial_params[1]:.0f}")
        print(f"Solar Factor (K):      {initial_params[2]:.0f}")
        print(f"Internal Heat (Q_int): {initial_params[3]:.0f} (Fixed Assumption)")
        print(f"HVAC params not fitted in regression mode.")
        
        # Plot using the linear guesses to see how they look
        results.plot_results(measurements, initial_params)
        return

    # 3. Run Full Optimization (Non-Linear)
    # We pass the Linear results as the "Seed" for the optimizer.
    # This drastically improves convergence speed and accuracy.
    optimization_result = optimize.run_optimization(measurements, initial_guess=initial_params)
    
    if optimization_result.success:
        print("\nOptimization converged successfully!")
        best_params = optimization_result.x
        
        # 4. Show Results
        results.plot_results(measurements, best_params)
    else:
        print("Optimization failed:", optimization_result.message)

if __name__ == "__main__":
    main()
