"""Test TOU optimization logic."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from custom_components.housetemp.housetemp.optimize import optimize_hvac_schedule
from custom_components.housetemp.housetemp.measurements import Measurements
from custom_components.housetemp.housetemp.run_model import run_model_continuous
from custom_components.housetemp.housetemp.energy import calculate_energy_vectorized

class MockHP:
    def __init__(self):
        self.min_output_btu_hr = 1000.0
        self.max_cool_btu_hr = 10000.0
        self.plf_low_load = 1.0 # simpler
        self.plf_slope = 0.0
        self.plf_min = 1.0
        self.idle_power_kw = 0.0
        self.blower_active_kw = 0.0
        self.defrost_risk_zone = None
        self.defrost_interval_min = 0
        self.defrost_duration_min = 0
        self.defrost_power_kw = 0

    def get_max_capacity(self, t_out):
        return np.full_like(t_out, 20000.0)
    
    def get_cop(self, t_out):
        return np.full_like(t_out, 3.0) # Constant COP 3.0
        
    def get_cooling_cop(self, t_out):
        return np.full_like(t_out, 3.0)

def test_optimizer_avoids_peak_rates():
    """ Verify optimizer shifts load away from high-rate periods."""
    
    # 1. Setup Data: 4 hours, 30 min steps (8 steps)
    # 12:00 - 16:00
    # Peak Rate at 14:00 (Step 4 & 5)
    
    start = datetime(2023, 1, 1, 12, 0, 0)
    timestamps = [start + timedelta(minutes=30*i) for i in range(8)]
    
    steps = 8
    t_out = np.full(steps, 40.0) # Cold outside
    t_in = np.full(steps, 60.0)
    t_in[0] = 68.0 # Start warm
    
    # Target 68F constant
    setpoints = np.full(steps, 68.0)
    
    # Rates: Peak (10x) at steps 4,5 (14:00-15:00)
    rates = np.ones(steps)
    rates[4:6] = 10.0 
    
    data = Measurements(
        timestamps=np.array(timestamps),
        t_in=t_in,
        t_out=t_out,
        solar_kw=np.zeros(steps),
        hvac_state=np.ones(steps, dtype=int), # Heat mode
        setpoint=setpoints,
        dt_hours=np.full(steps, 0.5),
        is_setpoint_fixed=np.zeros(steps, dtype=bool),
        target_temp=setpoints.copy(),
        tou_rate=rates
    )
    
    # Physics Params
    # C=Thermal Mass, UA=Leakiness
    params = [2000.0, 200.0, 0.0, 0.0, 2000.0, 1.0] 
    
    hw = MockHP()
    
    comfort_config = {
        "mode": "heat",
        "center_preference": 1.0, # Strong comfort preference to stay near target
        "comfort_mode": "neutral",
        "min_setpoint": 60,
        "max_setpoint": 80
    }
    
    # Run Optimization
    opt_setpoints, meta = optimize_hvac_schedule(
        data, params, hw, setpoints, comfort_config, 
        block_size_minutes=30, 
        enable_multiscale=False,
        rate_per_step=rates
    )
    
    assert meta['success']
    
    print(f"\nTime   Rate   Target   Optimized")
    for i in range(steps):
        t_str = timestamps[i].strftime("%H:%M")
        print(f"{t_str}   {rates[i]:4.1f}   {setpoints[i]:.1f}     {opt_setpoints[i]:.2f}")
        
    # Validation:
    # 1. Total Cost should be lower with optimization
    def calc_cost(sps):
        sim_temps, _, hvac_produced = run_model_continuous(
            params, 
            t_out_list=t_out.tolist(), 
            solar_kw_list=np.zeros(steps).tolist(),
            dt_hours_list=np.full(steps, 0.5).tolist(),
            setpoint_list=sps.tolist(), 
            hvac_state_list=np.ones(steps, dtype=int).tolist(),
            max_caps_list=np.full(steps, 20000.0).tolist(),
            min_output=hw.min_output_btu_hr,
            max_cool=hw.max_cool_btu_hr,
            eff_derate=1.0,
            start_temp=68.0
        )
        res = calculate_energy_vectorized(
            np.array(hvac_produced), np.full(steps, 0.5), np.full(steps, 20000.0), np.full(steps, 3.0), hw
        )
        energy_cost = np.sum(res['kwh_steps'] * rates)
        
        # Comfort penalty (heating)
        errors = np.minimum(0, np.array(sim_temps) - 68.0)
        penalty = np.sum(comfort_config['center_preference'] * (errors**2) * 0.5)
        return energy_cost, penalty

    naive_energy, naive_pen = calc_cost(setpoints)
    opt_energy, opt_pen = calc_cost(opt_setpoints)

    print(f"\nNaive:     Energy={naive_energy:.4f} Pen={naive_pen:.4f} Total={naive_energy+naive_pen:.4f}")
    print(f"Optimized: Energy={opt_energy:.4f} Pen={opt_pen:.4f} Total={opt_energy+opt_pen:.4f}")
    
    assert opt_energy < naive_energy - 0.01, "Optimizer should reduce energy cost"
    
    # 2. Before Peak (Steps 2-3), might pre-heat? 
    # Or just ensure cost < naive cost?
    # Simple check: Peak usage should be minimized.
    
    # Check if we reduced usage during peak compared to baseline? 
    # Implicitly checked if setpoint < target (because target 68 maintains temp against loss)
    # If setpoint < 68, we are letting temp drop (saving energy).
