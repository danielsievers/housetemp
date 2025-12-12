"""Helper class to handle data preparation for the simulation."""
import logging
from datetime import timedelta
import pandas as pd
import numpy as np

from homeassistant.util import dt as dt_util
from .const import CONF_MODEL_TIMESTEP, DEFAULT_MODEL_TIMESTEP
from .housetemp.utils import upsample_dataframe
from .housetemp.measurements import Measurements

_LOGGER = logging.getLogger(__name__)

class SimulationInputHandler:
    """Handles fetching and preparing data for the model simulation."""

    def __init__(self, hass):
        """Initialize the handler."""
        self.hass = hass

    def parse_forecast_points(self, forecast_list, dt_key_opts, val_key_opts):
        """Parse forecast list into standardized points.
        
        Args:
            forecast_list: List of dictionaries containing forecast data
            dt_key_opts: List of keys to look for datetime
            val_key_opts: List of keys to look for value
            
        Returns:
            List of dicts with 'time' and 'value' keys
        """
        pts = []
        for item in forecast_list:
            dt_val = next((item.get(k) for k in dt_key_opts if item.get(k)), None)
            key_found = next((k for k in val_key_opts if item.get(k) is not None), None)
            val = item.get(key_found) if key_found else None
            
            if dt_val and val is not None:
                if isinstance(dt_val, str):
                    dt = dt_util.parse_datetime(dt_val)
                else:
                    dt = dt_val
                
                if dt:
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=dt_util.get_time_zone(self.hass.config.time_zone))
                    
                    val_float = float(val)
                    if key_found in ['watts', 'wh_hours']: 
                            val_float /= 1000.0
                    elif key_found == 'pv_estimate':
                            if val_float > 50: 
                                val_float /= 1000.0
                    
                    pts.append({'time': dt, 'value': val_float})
        return pts

    async def prepare_simulation_data(self, weather_forecast, solar_forecast, start_time, duration_hours, model_timestep):
        """Process forecasts into a synchronized dataframe and upsample.
        
        Args:
            weather_forecast: List of weather forecast items
            solar_forecast: List of solar forecast items
            start_time: Datetime to start simulation from
            duration_hours: Duration in hours
            model_timestep: Timestep in minutes
            
        Returns:
            Tuple of (timestamps, t_out_arr, solar_arr, dt_values)
        """
        # Run in executor because pandas/numpy can be heavy
        return await self.hass.async_add_executor_job(
            self._prepare_simulation_data_sync,
            weather_forecast,
            solar_forecast,
            start_time,
            duration_hours,
            model_timestep
        )

    def _prepare_simulation_data_sync(self, weather_forecast, solar_forecast, start_time, duration_hours, model_timestep):
        """Synchronous part of data preparation."""
        weather_pts = self.parse_forecast_points(weather_forecast, ['datetime'], ['temperature'])
        if not weather_pts:
            raise ValueError("No weather forecast data available")

        solar_pts = self.parse_forecast_points(
            solar_forecast, 
            ['datetime', 'period_end', 'period_start'], 
            ['pv_estimate', 'watts', 'wh_hours', 'value']
        )
        
        if not solar_pts:
            now = dt_util.now()
            if now.tzinfo is None:
                now = now.replace(tzinfo=dt_util.get_time_zone(self.hass.config.time_zone))
            solar_pts = [{'time': now, 'value': 0.0}]

        end_time = start_time + timedelta(hours=duration_hours)
        
        df_w = pd.DataFrame(weather_pts).set_index('time').rename(columns={'value': 'outdoor_temp'})
        df_s = pd.DataFrame(solar_pts).set_index('time').rename(columns={'value': 'solar_kw'})
        
        # Combine
        df_raw = df_w.join(df_s, how='outer').sort_index()
        
        # Extrapolate boundaries if needed
        if df_raw.index.min() > start_time:
            row = pd.DataFrame({'outdoor_temp': [df_raw['outdoor_temp'].iloc[0]], 'solar_kw': [0]}, index=[start_time])
            df_raw = pd.concat([row, df_raw])
                
        if df_raw.index.max() < end_time:
            row = pd.DataFrame({'outdoor_temp': [df_raw['outdoor_temp'].iloc[-1]], 'solar_kw': [0]}, index=[end_time])
            df_raw = pd.concat([df_raw, row])
                
        df_raw = df_raw.reset_index().rename(columns={'index': 'time'})
        
        # Upsample
        freq_str = f"{model_timestep}min"
        df_sim = upsample_dataframe(
            df_raw, 
            freq=freq_str, 
            cols_linear=['outdoor_temp', 'solar_kw'],
            cols_ffill=[] 
        )
        
        df_sim = df_sim[(df_sim['time'] >= start_time) & (df_sim['time'] <= end_time)]
        
        timestamps = df_sim['time'].tolist()
        t_out_arr = df_sim['outdoor_temp'].ffill().values
        
        # Check for missing data (NaNs)
        if pd.isna(t_out_arr).any():
                raise ValueError("Weather forecast data gap: unable to interpolate outdoor temperature for full duration.")
                
        solar_arr = df_sim['solar_kw'].fillna(0).values
        dt_values = df_sim['dt'].values
        
        return timestamps, t_out_arr, solar_arr, dt_values
