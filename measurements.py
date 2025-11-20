from dataclasses import dataclass
import numpy as np

@dataclass
class Measurements:
    """
    Container for synchronized time-series data required for the thermal model.
    All arrays must be the same length.
    """
    timestamps: np.array  # Datetime objects
    t_in: np.array        # Indoor Temp (F)
    t_out: np.array       # Outdoor Temp (F)
    solar_kw: np.array    # Solar Panel Production (kW)
    hvac_state: np.array  # 1 (Heat), 0 (Off), -1 (Cool)
    setpoint: np.array    # Thermostat Target (F)
    dt_hours: np.array    # Time step size in hours (e.g., 0.5 for 30 mins)

    def __len__(self):
        return len(self.timestamps)
