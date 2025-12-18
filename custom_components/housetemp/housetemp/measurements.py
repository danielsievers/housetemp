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
    is_setpoint_fixed: np.array = None # Boolean mask: True if setpoint is fixed/mandatory
    target_temp: np.ndarray = None       # Original schedule targets (F)

    def __len__(self):
        return len(self.timestamps)

    def slice(self, start_idx, end_idx):
        """Returns a new Measurements object sliced from start_idx to end_idx."""
        return Measurements(
            timestamps=self.timestamps[start_idx:end_idx],
            t_in=self.t_in[start_idx:end_idx],
            t_out=self.t_out[start_idx:end_idx],
            solar_kw=self.solar_kw[start_idx:end_idx],
            hvac_state=self.hvac_state[start_idx:end_idx],
            setpoint=self.setpoint[start_idx:end_idx],
            dt_hours=self.dt_hours[start_idx:end_idx],
            is_setpoint_fixed=self.is_setpoint_fixed[start_idx:end_idx] if self.is_setpoint_fixed is not None else None,
            target_temp=self.target_temp[start_idx:end_idx] if self.target_temp is not None else None
        )
