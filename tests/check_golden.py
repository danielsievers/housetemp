#!/usr/bin/python3
import subprocess
import json
import os
import sys
import tempfile
import shutil

GOLDEN_DIR = "golden"
TEMP_DIR = "tests/temp_regression"

TESTS = [
    {
        "name": "Prediction (Occupied)",
        "golden": "occupied_pred.json",
        "cmd": [
            "python3", "main.py", "data/occupied.csv", 
            "-p", "data/occupied.json", 
            "--debug-output", "{temp_file}"
        ]
    },
    {
        "name": "Optimization (Occupied)",
        "golden": "occupied_hvac.json",
        "cmd": [
            "python3", "main.py", "data/occupied.csv", 
            "-p", "data/occupied.json", 
            "--optimize-hvac", "--comfort", "data/comfort.json",
            "--debug-output", "{temp_file}"
        ]
    }
]

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    # Remove dynamic fields for comparison
    if 'generated_at' in data:
        del data['generated_at']
    return data

def compare_data(golden, current, path=""):
    """Recursively compare JSON data with tolerance for floats."""
    if isinstance(golden, dict):
        if not isinstance(current, dict):
            return f"{path}: Expected dict, got {type(current)}"
        
        all_keys = set(golden.keys()) | set(current.keys())
        for k in all_keys:
            if k not in golden:
                return f"{path}: Unexpected key '{k}' in current"
            if k not in current:
                return f"{path}: Missing key '{k}' in current"
            
            err = compare_data(golden[k], current[k], path=f"{path}.{k}" if path else k)
            if err: return err
            
    elif isinstance(golden, list):
        if not isinstance(current, list):
            return f"{path}: Expected list, got {type(current)}"
        if len(golden) != len(current):
            return f"{path}: Length mismatch (Golden: {len(golden)}, Current: {len(current)})"
        
        for i, (g, c) in enumerate(zip(golden, current)):
            err = compare_data(g, c, path=f"{path}[{i}]")
            if err: return err
            
    elif isinstance(golden, (int, float)):
        if not isinstance(current, (int, float)):
             return f"{path}: Expected number, got {type(current)}"
        
        # Tolerance check
        if abs(golden - current) > 1e-4:
            return f"{path}: Value mismatch (Golden: {golden}, Current: {current})"
            
    else:
        if golden != current:
            return f"{path}: Mismatch (Golden: {golden}, Current: {current})"
            
    return None

def run_tests():
    # Setup temp dir
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    overall_success = True
    
    print(f"Running {len(TESTS)} Regression Tests...")
    print("="*60)
    
    for test in TESTS:
        name = test['name']
        print(f"Test: {name}")
        
        golden_path = os.path.join(GOLDEN_DIR, test['golden'])
        if not os.path.exists(golden_path):
            print(f"  [SKIPPED] Golden file missing: {golden_path}")
            overall_success = False
            continue
            
        temp_out = os.path.join(TEMP_DIR, test['golden'])
        cmd = [arg.format(temp_file=temp_out) for arg in test['cmd']]
        
        # Run Command
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"  [FAILED] Command crashed.")
            print(e.stderr.decode())
            overall_success = False
            continue
            
        # Compare
        try:
            golden_data = load_json(golden_path)
            current_data = load_json(temp_out)
            
            error = compare_data(golden_data, current_data)
            
            if error:
                print(f"  [FAILED] {error}")
                overall_success = False
            else:
                print(f"  [PASSED]")
                
        except Exception as e:
            print(f"  [ERROR] Comparison failed: {e}")
            overall_success = False
            
    print("="*60)
    
    # Cleanup
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        
    if overall_success:
        print("ALL REGRESSION TESTS PASSED.")
        sys.exit(0)
    else:
        print("REGRESSION TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
