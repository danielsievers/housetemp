# linear_fit.py
import numpy as np
from .measurements import Measurements

def linear_fit(data: Measurements, assumed_q_int=1900, default_hvac_cap=40000):
    """
    Performs Linear Regression (OLS) on passive data periods (HVAC=0)
    to derive calibrated starting points for C, UA, and K_solar.
    
    Logic:
    dT/dt = (UA/C)*(Tout - Tin) + (K/C)*Solar + (Qint/C)
    Y     = B1 * X1             + B2 * X2     + B3
    """
    print("Running Linear Regression on Passive periods...")

    # 1. Filter Data: Only use times when HVAC is OFF
    # We also need the NEXT timestamp's temp to calculate delta, so we shift indices
    mask = (data.hvac_state[:-1] == 0) & (data.dt_hours[:-1] > 0)
    
    if np.sum(mask) < 10:
        print("Warning: Not enough passive data points for linear fit. Using defaults.")
        return [8000, 310, 17000, assumed_q_int, default_hvac_cap]

    # 2. Prepare Y (Rate of Change)
    # Delta T / Delta Time
    delta_T = data.t_in[1:] - data.t_in[:-1]
    Y = delta_T[mask] / data.dt_hours[:-1][mask]

    # 3. Prepare X Matrix (Features)
    # X1: Temp Diff (Driving force for leakage)
    X1 = data.t_out[:-1][mask] - data.t_in[:-1][mask]
    
    # X2: Solar Gain
    X2 = data.solar_kw[:-1][mask]
    
    # X3: Intercept (Represents Internal Heat)
    X3 = np.ones(len(X1))
    
    # Stack into matrix [Rows, 3]
    X = np.column_stack((X1, X2, X3))

    # 4. Run Least Squares Solver
    # Solves Y = X * Beta
    # Beta = [UA/C, K/C, Qint/C]
    beta, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    
    b_leakage = beta[0] # UA / C
    b_solar   = beta[1] # K / C
    b_const   = beta[2] # Qint / C

    # 5. Unpack Physical Parameters
    # Since we have 3 coefficients but 4 unknowns, we must fix ONE to solve the rest.
    # Q_int is usually the most stable/guessable variable (Fridge + Humans).
    # We verify b_const is positive (physics check), otherwise linear fit failed.
    
    if b_const <= 0:
        print("Warning: Linear fit yielded negative internal heat. Defaulting.")
        return [8000, 310, 17000, assumed_q_int, default_hvac_cap]

    C_derived = assumed_q_int / b_const
    UA_derived = b_leakage * C_derived
    K_derived = b_solar * C_derived
    
    # Sanity Bounds (prevent physics explosions if data is noisy)
    C_derived = np.clip(C_derived, 2000, 20000)
    UA_derived = np.clip(UA_derived, 50, 1000)
    K_derived = np.clip(K_derived, 1000, 50000)
    
    # Guess H_factor (Linear fit can't see HVAC, so we estimate standard Inverter gain)
    # A 5-ton unit usually ramps 15k-20k BTU per degree of gap.
    H_factor_guess = 15000 

    print(f"  -> Linear Fit Found: C={C_derived:.0f}, UA={UA_derived:.0f}, K={K_derived:.0f}")
    
    return [C_derived, UA_derived, K_derived, assumed_q_int, H_factor_guess]
