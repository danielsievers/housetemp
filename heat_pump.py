import numpy as np

class MitsubishiHeatPump:
    """
    Defines the physical limits and efficiency curves of the
    Mitsubishi MXZ-SM60NAM (5-Ton Hyper-Heat).
    """
    
    def get_max_capacity(self, t_out_array):
        """
        Returns Max BTU/hr output based on outdoor temperature.
        Data points from Engineering Manual.
        """
        # X: Outdoor Temps (F)
        x_points = [-5, 5, 17, 47, 65]
        # Y: Max Capacity (BTU/hr)
        # Note plateau between 17F and 47F (Hyper-Heat behavior)
        y_points = [40000, 57000, 66000, 66000, 72000]
        
        return np.interp(t_out_array, x_points, y_points)

    def get_cop(self, t_out_array):
        """
        Returns base Coefficient of Performance at full load.
        """
        x_points = [5, 17, 47, 60]
        y_points = [1.9, 1.9, 3.7, 4.6]
        
        return np.interp(t_out_array, x_points, y_points)
