#!/usr/bin/env python3
"""
Verification suite for True Off snapping redesign (commit 0a9853b).

Runs optimization on scenario weather data, extracts per-step diagnostics
from post-solve verification outputs, and computes diagnostic rates.

Usage:
    .venv/bin/python scripts/verify_snapping.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.getcwd())

from custom_components.housetemp.housetemp.optimize import optimize_hvac_schedule
from custom_components.housetemp.housetemp.run_model import HeatPump, run_model_continuous, run_model_discrete
from custom_components.housetemp.housetemp.measurements import Measurements
from custom_components.housetemp.housetemp.constants import S_GAP_SMOOTH
from custom_components.housetemp.housetemp.energy import calculate_energy_vectorized
from custom_components.housetemp.housetemp.utils import get_effective_hvac_state

# === DIAGNOSTIC CONSTANTS ===
T_IDLE_BTU = 5.0        # BTU/hr threshold for "true idle"
SAFE_MARGIN_F = 0.2     # °F margin for comfort-safe classification
W_IDLE_ON = 0.9         # Threshold for idle_gate = "on"

def get_best_fit_case(rates):
    """Classifies the result into one of the diagnostic cases from implementation plan."""
    # Identify primary failure candidate
    fails = {
        'D (Weak Pull)': rates['weak_pull_rate'],
        'E (Gate Miss)': rates['miss_gate_rate'],
        'F (Bias Leak)': rates['bias_leak_rate']
    }
    best_fail = max(fails, key=fails.get)
    
    # If the worst failure is significant, return it
    if fails[best_fail] > 0.05:
        return best_fail
        
    # Otherwise, it's structurally the "Goal" (pinned correctly)
    return "Goal"

W_IDLE_OFF = 0.1        # Threshold for idle_gate = "off"

# Ablation settings
DEFAULT_BOUNDARY_PULL_WEIGHT = 0.05  # Match production default

DATA_DIR = 'data/scenarios'
OUTPUT_DIR = 'data/scenarios'


def load_scenario_weather(name: str) -> tuple:
    """Load weather data from cached scenario CSV."""
    filepath = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Scenario file not found: {filepath}")
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=[0])
    timestamps = df.index
    t_out = df['temperature'].values
    return timestamps, t_out


def compute_w_idle(gap: np.ndarray) -> np.ndarray:
    """Compute w_idle sigmoid exactly as in optimizer (with clipping)."""
    z = np.clip(gap / S_GAP_SMOOTH, -50, 50)
    sigmoid = 1 / (1 + np.exp(-z))
    return 1 - sigmoid  # w_idle ≈ 1 when gap <= 0


def compute_w_idle_produced(Q_verify: np.ndarray, threshold: float = T_IDLE_BTU) -> np.ndarray:
    """Ablation: produced-based idle gate (diagnostic only)."""
    # w_idle_Q ≈ 1 when produced is near zero
    return np.exp(-np.abs(Q_verify) / threshold)


def classify_idle_gate(w_idle: np.ndarray) -> np.ndarray:
    """3-state classification: 'on', 'off', 'ambiguous'."""
    result = np.full(len(w_idle), 'ambiguous', dtype=object)
    result[w_idle >= W_IDLE_ON] = 'on'
    result[w_idle <= W_IDLE_OFF] = 'off'
    return result


def run_scenario(name: str, mode: str, target_home: float, target_away: float,
                 variant: str, params: list, hw: HeatPump, 
                 comfort_config: dict) -> dict:
    """Run optimization on a scenario and extract diagnostics.
    
    Args:
        variant: 'idle_opp', 'recovery', or 'at_target'
    """
    
    print(f"\n{'='*60}")
    print(f"Scenario: {name} ({mode}) [{variant}]")
    print(f"{'='*60}")
    
    # Load weather
    timestamps, t_out = load_scenario_weather(name)
    n = len(t_out)
    
    # Derive dt_hours from timestamp cadence (not hardcoded)
    ts_series = pd.Series(timestamps)
    dt_seconds = ts_series.diff().dt.total_seconds().median()
    dt_hours_val = dt_seconds / 3600.0
    print(f"  Timestamp cadence: {dt_seconds:.0f}s ({dt_hours_val*60:.1f} min)")
    
    # Build target schedule (6am-10pm = home, else away)
    # Compute step indices based on actual cadence
    steps_per_hour = int(round(1.0 / dt_hours_val))
    target_temps = np.full(n, target_away)
    home_start = 6 * steps_per_hour
    home_end = 22 * steps_per_hour
    target_temps[home_start:home_end] = target_home
    
    mode_val = 1 if mode == 'heat' else -1
    
    deadband_slack = comfort_config.get('deadband_slack', 2.0)
    
    # Initial condition with offset
    # idle_opp: start above target (for heat) or below target (for cool) → idle opportunity
    # recovery: start outside comfort boundary → test bias leak during recovery
    # at_target: start exactly at the target schedule value
    if mode == 'heat':
        floor = target_temps[0] - deadband_slack
        if variant == 'at_target':
            start_temp = target_temps[0]
        elif variant == 'idle_opp':
            start_temp = target_temps[0] + 0.5
        else: # recovery
            start_temp = floor - 0.5
    else: # cool
        ceiling = target_temps[0] + deadband_slack
        if variant == 'at_target':
            start_temp = target_temps[0]
        elif variant == 'idle_opp':
            start_temp = target_temps[0] - 0.5
        else: # recovery
            start_temp = ceiling + 0.5
    
    # Construct Measurements (order: timestamps, t_in, t_out, solar_kw, hvac_state, setpoint, dt_hours)
    meas = Measurements(
        timestamps.to_pydatetime(),
        np.full(n, start_temp),  # t_in starts at offset position
        t_out,                    # t_out (outdoor temp)
        np.zeros(n),              # solar_kw = 0
        np.full(n, mode_val),     # hvac_state
        target_temps,             # setpoint
        np.full(n, dt_hours_val)  # dt_hours derived from timestamps
    )
    
    # Update comfort config for this scenario
    config = comfort_config.copy()
    config['mode'] = mode
    
    min_sp = config.get('min_setpoint', 60)
    max_sp = config.get('max_setpoint', 90)
    deadband_slack = config.get('deadband_slack', 2.0)
    
    # Run optimization
    opt_setpoints, meta = optimize_hvac_schedule(
        meas, params, hw, target_temps, config,
        block_size_minutes=30,
        enable_multiscale=False
    )
    
    if opt_setpoints is None:
        print(f"  ERROR: Optimization failed: {meta.get('message')}")
        return None
    
    print(f"  Optimization: {meta.get('message', 'OK')}")
    optimized_kwh = meta.get('verify_energy_kwh', 0)
    print(f"  Verified kWh: {optimized_kwh:.2f}")
    
    # === NAIVE SIMULATION (target temps as setpoints) ===
    eff_derate = params[5] if len(params) > 5 else 1.0
    naive_setpoints = np.round(target_temps[:n]).astype(int)
    
    # Mode-aware COP and HVAC state
    if mode == 'heat':
        base_cops = hw.get_cop(t_out)
        naive_hvac_state = get_effective_hvac_state(
            np.ones(n), naive_setpoints.astype(float),
            min_sp, max_sp, 0.1
        )
    else:
        base_cops = hw.get_cooling_cop(t_out)
        naive_hvac_state = get_effective_hvac_state(
            np.full(n, -1), naive_setpoints.astype(float),
            min_sp, max_sp, 0.1
        )
        
    max_caps = hw.get_max_capacity(t_out)
    
    naive_temps, _, naive_produced = run_model_continuous(
        params,
        t_out_list=t_out.tolist(),
        solar_kw_list=np.zeros(n).tolist(),
        dt_hours_list=np.full(n, dt_hours_val).tolist(),
        setpoint_list=naive_setpoints.astype(float).tolist(),
        hvac_state_list=naive_hvac_state.tolist(),
        max_caps_list=max_caps.tolist(),
        min_output=hw.min_output_btu_hr,
        max_cool=hw.max_cool_btu_hr,
        eff_derate=eff_derate,
        start_temp=float(start_temp)
    )
    
    naive_energy = calculate_energy_vectorized(
        np.array(naive_produced),
        np.full(n, dt_hours_val),
        max_caps,
        base_cops,
        hw,
        eff_derate=1.0, # Gross capacity doesn't need second derate
        hvac_states=naive_hvac_state
    )
    naive_kwh = naive_energy['kwh']
    savings_pct = (1 - optimized_kwh / naive_kwh) * 100 if naive_kwh > 0 else 0
    print(f"  Naive kWh:    {naive_kwh:.2f} (savings: {savings_pct:.1f}%)")
    
    # === DISCRETE SIMULATION (realistic thermostat behavior) ===
    swing_temp = config.get('swing_temp', 1.0)
    min_cycle_minutes = 5.0  # Standard HVAC min cycle time
    
    # Discrete naive (baseline)
    # Returns: (temps, outputs, produced, hvac_state, diagnostics)
    naive_disc_temps, naive_disc_outputs, naive_disc_produced, naive_disc_hvac, _ = run_model_discrete(
        params,
        t_out_list=t_out.tolist(),
        solar_kw_list=np.zeros(n).tolist(),
        dt_hours_list=np.full(n, dt_hours_val).tolist(),
        setpoint_list=naive_setpoints.astype(float).tolist(),
        hvac_state_list=naive_hvac_state.tolist(),
        max_caps_list=max_caps.tolist(),
        min_output=hw.min_output_btu_hr,
        max_cool=hw.max_cool_btu_hr,
        eff_derate=eff_derate,
        start_temp=float(start_temp),
        swing_temp=swing_temp,
        min_cycle_minutes=min_cycle_minutes
    )
    naive_discrete_energy = calculate_energy_vectorized(
        np.array(naive_disc_produced),
        np.full(n, dt_hours_val),
        max_caps,
        base_cops,
        hw,
        eff_derate=1.0,
        hvac_states=np.array(naive_disc_hvac).astype(float)
    )
    naive_discrete_kwh = naive_discrete_energy['kwh']
    
    # Discrete optimized
    opt_disc_temps, opt_disc_outputs, opt_disc_produced, opt_disc_hvac, _ = run_model_discrete(
        params,
        t_out_list=t_out.tolist(),
        solar_kw_list=np.zeros(n).tolist(),
        dt_hours_list=np.full(n, dt_hours_val).tolist(),
        setpoint_list=opt_setpoints.astype(float).tolist(),
        hvac_state_list=get_effective_hvac_state(
            naive_hvac_state, opt_setpoints.astype(float), min_sp, max_sp, 0.1
        ).tolist(),
        max_caps_list=max_caps.tolist(),
        min_output=hw.min_output_btu_hr,
        max_cool=hw.max_cool_btu_hr,
        eff_derate=eff_derate,
        start_temp=float(start_temp),
        swing_temp=swing_temp,
        min_cycle_minutes=min_cycle_minutes
    )
    opt_discrete_energy = calculate_energy_vectorized(
        np.array(opt_disc_produced),
        np.full(n, dt_hours_val),
        max_caps,
        base_cops,
        hw,
        eff_derate=1.0,
        hvac_states=np.array(opt_disc_hvac).astype(float)
    )
    opt_discrete_kwh = opt_discrete_energy['kwh']
    discrete_savings_pct = (1 - opt_discrete_kwh / naive_discrete_kwh) * 100 if naive_discrete_kwh > 0 else 0

    
    # === EXTRACT FROM POST-SOLVE VERIFICATION (canonical source) ===
    sp_cmd = opt_setpoints  # Already quantized integers
    T_verify = np.array(meta.get('verify_temps', []))
    Q_verify = np.array(meta.get('verify_produced', []))
    
    # Align to N steps (T_verify has N+1 entries typically)
    n_steps = min(len(sp_cmd), len(T_verify) - 1, len(Q_verify))
    sp_cmd = sp_cmd[:n_steps]
    Q_verify = Q_verify[:n_steps]
    target_temps = target_temps[:n_steps]
    
    # Exact indexing: start-of-step and end-of-step temps
    T_start = T_verify[:n_steps]
    T_end = T_verify[1:n_steps+1]
    
    # === COMPUTE GAP (exactly as optimizer) ===
    # Derive slack from mode
    swing_temp = config.get('swing_temp', 1.0)
    if config.get('comfort_mode') == 'deadband':
        slack = float(deadband_slack)
    else:
        slack = 0.5 * float(swing_temp)
    
    # Match optimizer's safe parameters
    s_margin = 0.5 * float(swing_temp)
    
    if mode == 'heat':
        gap = sp_cmd.astype(float) - T_start
        floor = target_temps - slack
        # Conservative safe: both start AND end above floor + margin
        safe = np.minimum(T_start, T_end) >= (floor + s_margin)
        pinned = (sp_cmd <= min_sp + 0.1)
    else:
        gap = T_start - sp_cmd.astype(float)
        ceiling = target_temps + slack
        # Conservative safe: both start AND end below ceiling - margin
        safe = np.maximum(T_start, T_end) <= (ceiling - s_margin)
        pinned = (sp_cmd >= max_sp - 0.1)
    # === COMPUTE w_idle (gap-based, exactly as optimizer) ===
    w_idle = compute_w_idle(gap)
    idle_gate = classify_idle_gate(w_idle)
    
    # === ABLATION: produced-based w_idle ===
    w_idle_Q = compute_w_idle_produced(Q_verify)
    idle_gate_Q = classify_idle_gate(w_idle_Q)
    
    # === idle_true from Q_verify ===
    idle_true = np.abs(Q_verify) < T_IDLE_BTU

    # --- SANITY CHECKS (User Requested) ---
    actual_min_sp = np.min(sp_cmd)
    actual_max_sp = np.max(sp_cmd)
    top_clamp_rate = np.mean(sp_cmd >= actual_max_sp - 0.01)
    bottom_clamp_rate = np.mean(sp_cmd <= actual_min_sp + 0.01)
    
    # Redefine boundary_pinned based on mode to ensure it uses correct script bounds
    if mode == 'heat':
        boundary_val = min_sp
        boundary_pinned = (sp_cmd <= min_sp + 0.01)
    else:
        boundary_val = max_sp
        boundary_pinned = (sp_cmd >= max_sp - 0.01)

    print(f"  Boundary Sanity Check:")
    print(f"    Mode: {mode.upper()} (Target Boundary: {boundary_val})")
    print(f"    sp_cmd range: [{actual_min_sp}, {actual_max_sp}]")
    print(f"    Script Bounds: min={min_sp}, max={max_sp}")
    print(f"    Steps at Actual Max/Min: {top_clamp_rate*100:.1f}% / {bottom_clamp_rate*100:.1f}%")
    print(f"    Steps at Script Boundary: {np.mean(boundary_pinned)*100:.1f}%")

    
    # === COMPUTE DIAGNOSTIC RATES ===
    gate_on = idle_gate == 'on'
    gate_off = idle_gate == 'off'
    gate_on_Q = idle_gate_Q == 'on'
    
    # Counts
    n_safe = int(np.sum(safe))
    n_idle_true = int(np.sum(idle_true))
    n_idle_safe = int(np.sum(idle_true & safe))
    
    # Matrix cases (gap-based gate)
    case_a = int(np.sum(idle_true & gate_on & boundary_pinned))
    case_b = int(np.sum(~idle_true & gate_off & ~boundary_pinned))
    case_c = int(np.sum(idle_gate == 'ambiguous'))
    case_d = int(np.sum(idle_true & safe & gate_on & ~boundary_pinned))
    case_e = int(np.sum(idle_true & safe & gate_off))
    case_f = int(np.sum(~idle_true & gate_on & boundary_pinned))
    
    # Ablation: what if we used produced-based gate?
    case_e_Q = int(np.sum(idle_true & safe & (idle_gate_Q == 'off')))
    
    # Rates
    def safe_rate(num, denom):
        return float(num / denom) if denom > 0 else 0.0
    
    n_idle_safe_gate_on = int(np.sum(idle_true & safe & gate_on))
    print(f"    Weak Pull Denominator: {n_idle_safe_gate_on}")
    
    n_gate_on_pinned = int(np.sum(gate_on & boundary_pinned))
    
    miss_gate_rate = safe_rate(case_e, n_idle_safe)
    weak_pull_rate = safe_rate(case_d, n_idle_safe_gate_on) if n_idle_safe_gate_on > 0 else 0.0
    bias_leak_rate = safe_rate(case_f, n_gate_on_pinned) if n_gate_on_pinned > 0 else 0.0
    pinned_given_idle_safe = safe_rate(np.sum(idle_true & safe & boundary_pinned), n_idle_safe)
    
    # Ablation rate
    miss_gate_rate_Q = safe_rate(case_e_Q, n_idle_safe)
    
    current_weight = comfort_config.get('boundary_pull_weight', DEFAULT_BOUNDARY_PULL_WEIGHT)
    
    result = {
        'scenario': name,
        'mode': mode,
        'variant': variant,
        'start_temp': float(start_temp),
        'n_steps': n_steps,
        'n_safe': n_safe,
        'n_idle_true': n_idle_true,
        'n_idle_safe': n_idle_safe,
        'cases': {
            'A_working': case_a,
            'B_active': case_b,
            'C_ambiguous': case_c,
            'D_weak_pull': case_d,
            'E_gate_miss': case_e,
            'F_bias_leak': case_f,
        },
        'ablation': {
            'E_gate_miss_produced': case_e_Q,
            'miss_gate_rate_produced': miss_gate_rate_Q,
        },
        'rates': {
            'miss_gate_rate': miss_gate_rate,
            'weak_pull_rate': weak_pull_rate,
            'bias_leak_rate': bias_leak_rate,
            'pinned_given_idle_safe': pinned_given_idle_safe,
            'boundary_pull_weight': current_weight,
        },
        'energy_kwh': optimized_kwh,           # Continuous optimized
        'naive_kwh': naive_kwh,                 # Continuous naive
        'savings_pct': savings_pct,             # Continuous savings
        'discrete_kwh': opt_discrete_kwh,       # Discrete optimized
        'naive_discrete_kwh': naive_discrete_kwh,  # Discrete naive
        'discrete_savings_pct': discrete_savings_pct,  # Discrete savings
        'diagnosis': get_best_fit_case({
            'pinned_given_idle_safe': pinned_given_idle_safe,
            'weak_pull_rate': weak_pull_rate,
            'miss_gate_rate': miss_gate_rate,
            'bias_leak_rate': bias_leak_rate
        }),
        # Plot data (use string timestamps for JSON serialization)
        'plot_data': {
            'timestamps': [str(t) for t in timestamps[:n_steps]],
            'sp_cmd': sp_cmd.tolist(),
            'target_temps': target_temps.tolist(),
            'T_verify': T_start.tolist(),
            't_out': t_out[:n_steps].tolist(),
        },
        'comfort_config': config.copy(),  # For plotting bounds
    }
    
    # Print summary
    print(f"\n  Diagnostic Matrix (n={n_steps}, safe={n_safe}, idle_true={n_idle_true}):")
    print(f"    Case A (working):    {case_a:4d}")
    print(f"    Case B (active):     {case_b:4d}")
    print(f"    Case C (ambiguous):  {case_c:4d}")
    print(f"    Case D (weak pull):  {case_d:4d}")
    print(f"    Case E (gate miss):  {case_e:4d} (produced-based: {case_e_Q})")
    print(f"    Case F (bias leak):  {case_f:4d}")
    print(f"\n  Diagnostic Rates (W={current_weight}):")
    print(f"    miss_gate_rate:           {miss_gate_rate:.3f} (produced-based gate miss: {miss_gate_rate_Q:.3f})")
    print(f"    weak_pull_rate:           {weak_pull_rate:.3f}")
    print(f"    bias_leak_rate:           {bias_leak_rate:.3f}")
    print(f"    pinned_given_idle_safe:   {pinned_given_idle_safe:.3f} (KPI ≈ 1.0)")
    
    if miss_gate_rate > miss_gate_rate_Q + 0.05:
        print(f"    [!] ALERT: Gap gate is over-triggering (miss {miss_gate_rate:.2f} > prod-miss {miss_gate_rate_Q:.2f})")
    
    return result


def main():
    print("=" * 60)
    print("True Off Snapping Verification Suite")
    print("=" * 60)
    
    # Load heat pump config
    hw = HeatPump('data/heat_pump.json')
    hw.max_cool_btu_hr = 36000  # Enable cooling
    
    # Load model params from calibrated house data
    with open('data/occupied.json', 'r') as f:
        import re
        content = re.sub(r'//.*', '', f.read())  # Strip JS-style comments
        house_params = json.loads(content)
    
    # Model params: C, UA, K_solar, Q_int, H_factor, Eff_Derate
    params = [
        house_params['C_thermal'],
        house_params['UA_overall'],
        house_params['K_solar'],
        house_params['Q_int'],
        house_params['H_factor'],
        house_params['efficiency_derate']
    ]
    print(f"Loaded params: C={params[0]:.0f}, UA={params[1]:.0f}, Q_int={params[3]:.0f}")
    
    # Standard comfort config (internal, not HASS-exposed values)
    comfort_config = {
        'center_preference': 0.2,      # Matches data/*.comfort.json
        'comfort_mode': 'deadband',
        'deadband_slack': 2.0,
        'swing_temp': 1.0,            # HASS swing default
        'min_setpoint': 60,    # Loose bounds for snapping test
        'max_setpoint': 82,    # Loose bounds for snapping test
        'boundary_pull_weight': DEFAULT_BOUNDARY_PULL_WEIGHT,  # Ablation toggle
    }
    
    # Define scenarios with gaps between target and boundary
    # Heating: using heating_comfort.json schedule (63-70°F, target 69°F home, 63°F away)
    # Cooling: using cooling_comfort.json schedule (69-78°F, target 70°F home, 68°F away)
    scenarios = [
        # (csv_name, mode, target_home, target_away)
        # Heating: target_away=63 gives gap to min_setpoint
        ('cold_winter', 'heat', 69.0, 63.0),
        ('shoulder_season', 'heat', 69.0, 63.0),
        # Cooling: target_home=70, target_away=68 (user's schedule)
        ('mild_summer', 'cool', 70.0, 68.0),
        ('hot_summer', 'cool', 70.0, 68.0),
    ]
    
    # Initial condition variants
    variants = ["at_target", "idle_opp", "recovery"]
    
    results = []
    # Ablation Pass: compare different pull weights
    grouped = {}  # Initialize to store results by scenario for comparison
    
    for csv_name, mode, target_home, target_away in scenarios:
        # Ablation weights: heating uses 0 vs 0.05, cooling uses 0 vs tiny sweep
        if mode == 'heat':
            weights = [0.0, 0.05]
        else:
            weights = [0.0, 0.001, 0.005, 0.01]  # Extended cooling sweep with tiny weight
        
        for variant in variants:
            for w in weights:
                try:
                    conf = comfort_config.copy()
                    conf['boundary_pull_weight'] = w
                    
                    # Use mode-specific bounds from user's comfort configs
                    if mode == 'heat':
                        conf['min_setpoint'] = 63  # from heating_comfort.json
                        conf['max_setpoint'] = 70  # from heating_comfort.json
                    else:
                        conf['min_setpoint'] = 69  # from cooling_comfort.json
                        conf['max_setpoint'] = 78  # from cooling_comfort.json
                        
                    res = run_scenario(csv_name, mode, target_home, target_away,
                                      variant, params, hw, conf)
                    if res:
                        # --- SECOND GATE: ENERGY REGRESSION CHECK ---
                        # baseline is always the first weight (0.0)
                        baseline = grouped.get((csv_name, variant, mode), {}).get('W=0')
                        if baseline and w > 0:
                            save_baseline = baseline.get('savings_pct', 0)
                            save_test = res.get('savings_pct', 0)
                            # FAIL if test is worse by > 2.0% (stricter margin as requested)
                            if save_test < (save_baseline - 2.0):
                                res['diagnosis'] = "FAIL (Regr)"
                                print(f"  [!] FAIL: Energy regression detected ({save_test:.1f}% vs baseline {save_baseline:.1f}%)")

                        results.append(res)
                        
                        # Store in grouped for summary
                        key = (csv_name, variant, mode)
                        if key not in grouped: grouped[key] = {}
                        weight_key = 'W=0' if w == 0 else 'W=def'
                        grouped[key][weight_key] = res

                except FileNotFoundError as e:
                    print(f"  SKIP: {e}")
                except Exception as e:
                    print(f"  ERROR: {e}")
    
    # Save results to JSON
    output_path = os.path.join(OUTPUT_DIR, 'snapping_diagnostics.json')
    with open(output_path, 'w') as f:
        # Filter out plot_data for smaller JSON if needed, or keep for audit
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {output_path}")
    # Comparative Summary Table
    print("\n" + "=" * 130)
    print("COMPARATIVE SUMMARY (Ablation: W=0 vs W=Default)")
    print("=" * 130)
    header = f"{'Scenario':<18} {'Var':<10} {'Mode':<5} | {'pinned%':<15} | {'Save%':<15} | {'Diagnosis':<15}"
    print(header)
    print(f"{'':<35} | {'W=0':>6} {'W=def':>7} | {'W=0':>6} {'W=def':>7} | {'W=def'}")
    print("-" * 130)

    # Results already grouped in main loop

    for (scenario, variant, mode), data in grouped.items():
        r_w0 = data.get('W=0')
        r_wdef = data.get('W=def')

        pinned_w0 = f"{r_w0['rates']['pinned_given_idle_safe']*100:>6.1f}" if r_w0 else "N/A"
        pinned_wdef = f"{r_wdef['rates']['pinned_given_idle_safe']*100:>7.1f}" if r_wdef else "N/A"
        
        save_w0 = f"{r_w0.get('savings_pct', 0):>5.1f}%" if r_w0 else "N/A"
        save_wdef = f"{r_wdef.get('savings_pct', 0):>6.1f}%" if r_wdef else "N/A"
        
        diagnosis_wdef = r_wdef.get('diagnosis', 'N/A') if r_wdef else "N/A"

        print(f"{scenario:<18} {variant:<10} {mode:<5} | {pinned_w0} {pinned_wdef} | {save_w0} {save_wdef} | {diagnosis_wdef}")
    
    # Generate plots for all scenarios
    import time
    ts = int(time.time())
    print("\n\nGenerating plots...")
    for r in results:
        if 'plot_data' not in r:
            continue
            
        fig, ax = plt.subplots(figsize=(14, 6))
        
        pd_data = r['plot_data']
        timestamps = pd.to_datetime(pd_data['timestamps'])
        hours = (timestamps - timestamps[0]).total_seconds() / 3600
        
        # Plot setpoints and targets
        ax.plot(hours, pd_data['sp_cmd'], 'b-', linewidth=2, label='Optimized Setpoint')
        ax.plot(hours, pd_data['target_temps'], 'g--', linewidth=1.5, label='Target Schedule')
        ax.plot(hours, pd_data['T_verify'], 'r-', linewidth=1, alpha=0.7, label='Indoor Temp (Verify)')
        ax.plot(hours, pd_data['t_out'], 'c-', linewidth=1, alpha=0.5, label='Outdoor Temp')
        
        # Mark boundaries (show both for all modes)
        min_sp = r['comfort_config']['min_setpoint']
        max_sp = r['comfort_config']['max_setpoint']
        ax.axhline(y=min_sp, color='purple', linestyle=':', alpha=0.7, label=f'min_setpoint ({min_sp}°F)')
        ax.axhline(y=max_sp, color='orange', linestyle=':', alpha=0.7, label=f'max_setpoint ({max_sp}°F)')
        
        ax.set_xlabel('Hours')
        ax.set_ylabel('Temperature (°F)')
        mode_label = 'Heating' if r['mode'] == 'heat' else 'Cooling'
        ax.set_title(f"{r['scenario']} ({r['variant']}) - {mode_label}\n"
                    f"pinned={r['rates']['pinned_given_idle_safe']*100:.0f}%")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Save plot
        current_weight = r['rates']['boundary_pull_weight']
        w_label = f"W{current_weight:.2f}".replace('.', '')
        plot_path = os.path.join(OUTPUT_DIR, f"setpoints_{r['scenario']}_{r['variant']}_{w_label}_{ts}.png")
        fig.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {plot_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
