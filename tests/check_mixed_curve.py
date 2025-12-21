import pandas as pd
import numpy as np
import json
from custom_components.housetemp.housetemp.run_model import run_model_continuous

# Run
max_caps = [0.0] * len(m)
sim_temps, _, _ = run_model_continuous(
    params, t_out_list=m.t_out.tolist(),
    solar_kw_list=m.solar_kw.tolist(),
    dt_hours_list=m.dt_hours.tolist(),
    setpoint_list=m.setpoint.tolist(),
    hvac_state_list=m.hvac_state.tolist(),
    max_caps_list=max_caps,
    min_output=0, max_cool=0, eff_derate=1.0,
    start_temp=m.t_in[0]
)

print(f"Start Temp: {sim_temps[0]:.2f}")
print(f"Final Temp: {sim_temps[-1]:.2f}")
print(f"Peak Temp:  {np.max(sim_temps):.2f}")
