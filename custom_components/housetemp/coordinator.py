"""DataUpdateCoordinator for House Temp Prediction."""
from datetime import timedelta, datetime
import json
import time
import logging
import os
import re
import tempfile
from functools import partial
from typing import Optional

import numpy as np

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import (
    DataUpdateCoordinator,
    UpdateFailed,
)
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    CONF_C_THERMAL,
    CONF_UA,
    CONF_K_SOLAR,
    CONF_Q_INT,
    CONF_H_FACTOR,
    CONF_EFF_DERATE,
    CONF_SENSOR_INDOOR_TEMP,
    CONF_CENTER_PREFERENCE,
    CONF_WEATHER_ENTITY,
    CONF_SOLAR_ENTITY,
    CONF_HEAT_PUMP_CONFIG,
    CONF_SCHEDULE_CONFIG,
    CONF_SCHEDULE_ENABLED,
    CONF_FORECAST_DURATION,
    CONF_UPDATE_INTERVAL,
    DEFAULT_FORECAST_DURATION,
    DEFAULT_UPDATE_INTERVAL,
    CONF_MODEL_TIMESTEP,
    DEFAULT_MODEL_TIMESTEP,
    CONF_CONTROL_TIMESTEP,
    DEFAULT_CONTROL_TIMESTEP,
    CONF_HVAC_MODE,
    CONF_AVOID_DEFROST,
    CONF_COMFORT_MODE,
    CONF_DEADBAND_SLACK,
    DEFAULT_COMFORT_MODE,
    DEFAULT_DEADBAND_SLACK,
    DEFAULT_AWAY_TEMP,
    DEFAULT_C_THERMAL,
    DEFAULT_UA,
    DEFAULT_K_SOLAR,
    DEFAULT_Q_INT,
    DEFAULT_H_FACTOR,
    DEFAULT_EFF_DERATE,
    DEFAULT_CENTER_PREFERENCE,
    DEFAULT_SCHEDULE_CONFIG,
    DEFAULT_SCHEDULE_ENABLED,
    AWAY_WAKEUP_ADVANCE_HOURS,
    CONF_ENABLE_MULTISCALE,
    DEFAULT_ENABLE_MULTISCALE,
    DEFAULT_HEAT_PUMP_CONFIG,
    CONF_MIN_SETPOINT,
    CONF_MAX_SETPOINT,
    DEFAULT_MIN_SETPOINT,
    DEFAULT_MAX_SETPOINT,
    DEFAULT_MAX_SETPOINT,
    CONF_SWING_TEMP,
    DEFAULT_SWING_TEMP,
    CONF_MIN_CYCLE_DURATION,
    DEFAULT_MIN_CYCLE_MINUTES,
    DEFAULT_OFF_INTENT_EPS,
    DEFAULT_IDLE_MARGIN
)

# Import from the installed package
from .housetemp.run_model import run_model_continuous, run_model_discrete, HeatPump

from .housetemp.measurements import Measurements
from .housetemp.optimize import optimize_hvac_schedule
from .housetemp.schedule import process_schedule_data
from .housetemp.energy import estimate_consumption, calculate_energy_stats, calculate_energy_vectorized
from .housetemp.utils import get_effective_hvac_state, detect_idle_blocks
from .input_handler import SimulationInputHandler

_LOGGER = logging.getLogger(DOMAIN)

class HouseTempCoordinator(DataUpdateCoordinator):
    """Class to manage fetching data and running the model."""

    def __init__(self, hass: HomeAssistant, config_entry):
        """Initialize."""
        self.config_entry = config_entry
        self.hass = hass
        
        update_interval_min = config_entry.options.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL)
        
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=update_interval_min),
        )
        self.config_entry = config_entry

        self.heat_pump = None
        # self._setup_heat_pump() # Will be called in async_update_data or first refresh
        
        # State for optimization
        # Change to dict: timestamp -> setpoint
        self.optimized_setpoints_map = {} 
        
        # New Input Handler
        self.input_handler = SimulationInputHandler(hass)
        
        # Track config state to invalidate cache on change
        self._last_config_id = self._get_config_id()
        self._optimization_status = None # Store status separately from data 
        self._unsub_state_trackers = []

    def async_setup_trackers(self):
        """Setup trackers for dependent entities."""
        # Clean up existing trackers if any
        for unsub in self._unsub_state_trackers:
            unsub()
        self._unsub_state_trackers = []

        data = self.config_entry.data
        entities = []
        if sensor_indoor := data.get(CONF_SENSOR_INDOOR_TEMP):
            entities.append(sensor_indoor)
        if weather_entity := data.get(CONF_WEATHER_ENTITY):
            entities.append(weather_entity)
        if solar_entities := data.get(CONF_SOLAR_ENTITY):
            if isinstance(solar_entities, str):
                entities.append(solar_entities)
            else:
                entities.extend(solar_entities)

        # Filter out empty/None
        entities = [e for e in entities if e]

        _LOGGER.debug("Setting up state trackers for entities: %s", entities)
        
        async def _async_state_changed(event):
            """Handle state changes of dependent entities."""
            old_state = event.data.get("old_state")
            new_state = event.data.get("new_state")
            
            if not new_state or new_state.state in ("unknown", "unavailable"):
                return

            if not old_state or old_state.state in ("unknown", "unavailable"):
                # Transition from unusable to usable -> Refresh
                _LOGGER.info("Entity %s became available. Requesting refresh.", event.data.get("entity_id"))
                self.async_set_updated_data(self.data) # Briefly update to trigger UI? Or just refresh.
                await self.async_request_refresh()

        self._unsub_state_trackers.append(
            async_track_state_change_event(self.hass, entities, _async_state_changed)
        )

    def _get_config_id(self):
        """Returns a stable tuple representing the current optimization-relevant config."""
        opts = self.config_entry.options
        data = self.config_entry.data
        
        def _get_stable_val(val):
            if isinstance(val, (dict, list)):
                 return json.dumps(val, sort_keys=True)
            return val

        return (
            _get_stable_val(opts.get(CONF_MODEL_TIMESTEP, DEFAULT_MODEL_TIMESTEP)),
            _get_stable_val(opts.get(CONF_CONTROL_TIMESTEP, DEFAULT_CONTROL_TIMESTEP)),
            _get_stable_val(opts.get(CONF_ENABLE_MULTISCALE, DEFAULT_ENABLE_MULTISCALE)),
            _get_stable_val(opts.get(CONF_COMFORT_MODE)),
            _get_stable_val(data.get(CONF_C_THERMAL)),
            _get_stable_val(data.get(CONF_UA)),
            _get_stable_val(data.get(CONF_K_SOLAR)),
            _get_stable_val(data.get(CONF_Q_INT)),
            _get_stable_val(data.get(CONF_H_FACTOR)),
            _get_stable_val(data.get(CONF_HEAT_PUMP_CONFIG)),
            _get_stable_val(data.get(CONF_SCHEDULE_CONFIG)),
        )

    async def _setup_heat_pump(self):
        """Initialize the HeatPump object from the config JSON."""
        # Heat Pump Config now lives in OPTIONS (migrated from data if needed)
        hp_config_str = self.config_entry.options.get(CONF_HEAT_PUMP_CONFIG, DEFAULT_HEAT_PUMP_CONFIG)
        if not hp_config_str:
            hp_config_str = DEFAULT_HEAT_PUMP_CONFIG

        # Run file I/O in executor
        def save_and_init():
            # Storage dir
            storage_dir = self.hass.config.path(".storage", DOMAIN)
            os.makedirs(storage_dir, exist_ok=True)
            hp_config_path = os.path.join(storage_dir, f"heat_pump_{self.config_entry.entry_id}.json")
            
            # Validate and Patch JSON
            try:
                # Strip comments for robust loading
                clean_config = re.sub(r"//.*", "", hp_config_str)
                data = json.loads(clean_config)
                
                # Check for mandatory keys required by the newer library version
                needs_patch = False
                mandatory_keys = [
                    'min_output_btu_hr', 'max_cool_btu_hr', 
                    'plf_low_load', 'plf_slope', 
                    'idle_power_kw', 'blower_active_kw'
                ]
                
                # Load defaults for patching
                default_data = json.loads(re.sub(r"//.*", "", DEFAULT_HEAT_PUMP_CONFIG))
                
                for key in mandatory_keys:
                    if key not in data:
                        _LOGGER.warning("Missing key '%s' in Heat Pump config. Patching with default. Please visit integration settings to update permanently.", key)
                        data[key] = default_data[key]
                        needs_patch = True
                
                # If we patched it, we should use the patched version
                if needs_patch:
                    final_config = json.dumps(data, indent=2)
                else:
                    final_config = hp_config_str
                
                # Basic validation 
                if "cop" not in data or "max_capacity" not in data:
                     _LOGGER.error("Heat Pump Config is missing core curves (cop, max_capacity). Sensors will be inaccurate.")
            except Exception as e:
                raise ValueError(f"Invalid Heat Pump JSON: {e}")
            
            with open(hp_config_path, "w") as f:
                f.write(final_config)
            
            return HeatPump(hp_config_path)

        try:
            self.heat_pump = await self.hass.async_add_executor_job(save_and_init)
        except Exception as e:
            _LOGGER.error("Failed to setup Heat Pump: %s", e)
            self.heat_pump = None

    def _expire_cache(self):
        """Remove stale cache entries (past timestamps only)."""
        if not self.optimized_setpoints_map:
            return
        
        now = dt_util.now()
        now_ts = int(now.timestamp())
        
        # Remove only PAST entries, keep all FUTURE entries
        old_size = len(self.optimized_setpoints_map)
        
        # Filter keys: keep if timestamp >= now
        # Note: We use a new dict comprehension for atomic replacement
        self.optimized_setpoints_map = {
            k: v for k, v in self.optimized_setpoints_map.items() 
            if k >= now_ts
        }
        
        # FIFO eviction if cache grows too large (prevent memory leak)
        # 3000 entries = ~10 days at 5 minute intervals
        if len(self.optimized_setpoints_map) > 3000:
            sorted_keys = sorted(self.optimized_setpoints_map.keys())
            keep_keys = sorted_keys[-3000:]  # Keep most recent/future 3000
            self.optimized_setpoints_map = {
                k: self.optimized_setpoints_map[k] for k in keep_keys
            }
        
        removed = old_size - len(self.optimized_setpoints_map)
        if removed > 0:
            _LOGGER.debug("Cache cleanup: removed %d past entries, %d remain", 
                         removed, len(self.optimized_setpoints_map))

    async def _async_update_data(self):
        """Fetch data and run the model."""
        # Clean up cache before processing
        self._expire_cache()
        
        # Check for Configuration changes (invalidates optimization cache)
        current_config_id = self._get_config_id()
        if current_config_id != self._last_config_id:
            _LOGGER.info("Optimization parameters changed. Clearing cache.")
            self.optimized_setpoints_map.clear()
            self._last_config_id = current_config_id

        if not self.heat_pump:
            await self._setup_heat_pump()
            if not self.heat_pump:
                raise UpdateFailed("Heat Pump not configured correctly")

        # 1. Get Inputs & Prepare Simulation Data
        try:
            res = await self._prepare_simulation_inputs()
            if res is None:
                 # Signals transient "not ready" - we don't want to raise UpdateFailed 
                 # which might trigger annoying UI error toast if it's just a startup race.
                 # But coordinator needs something. In HASS, returning None from _async_update_data 
                 # keeps the old data. If no data exists yet, it stays in 'initializing'.
                 _LOGGER.debug("Inputs not ready, skipping update.")
                 return self.data 
            measurements, params, start_time = res
        except Exception as e:
            _LOGGER.error("Error preparing simulation inputs: %s", e)
            raise UpdateFailed(f"Error preparing simulation inputs: {e}")

        # Unpack for local usage if needed, or just use measurements object
        timestamps = measurements.timestamps
        setpoint_arr = measurements.setpoint
        t_out_arr = measurements.t_out
        solar_arr = measurements.solar_kw
        
        duration_hours = self.config_entry.options.get(CONF_FORECAST_DURATION, DEFAULT_FORECAST_DURATION)

        # Reconstruct setpoints for display and simulation
        # 1. Attribute for display (fallback to 'current behavior' baseline)
        # 2. Simulation input (fallback to 'intended schedule' baseline)
        optimized_setpoint_attr = []
        sim_setpoints = []
        has_optimized_data = False
        
        found_count = 0
        for i, ts in enumerate(timestamps):
            ts_int = int(ts.timestamp())
            val = self.optimized_setpoints_map.get(ts_int)
            
            # Optimized value from cache
            if val is not None:
                optimized_setpoint_attr.append(val)
                sim_setpoints.append(val)
                found_count += 1
            else:
                # Optimized attribute strictly shows None for missing points
                optimized_setpoint_attr.append(None)
                
                # Simulation policy: Prefer Schedule -> Off (NaN)
                target = measurements.target_temp[i]
                if target is not None and not np.isnan(target):
                    sim_setpoints.append(target)
                else:
                    sim_setpoints.append(np.nan)
        
        # Explicit No-Schedule Policy: 
        # If all targets are NaN and no optimized data, force HVAC off for safety (avoid NaN propagation)
        targets = measurements.target_temp
        has_schedule = targets is not None and np.any(~np.isnan(targets))
        
        if found_count > 0:
            has_optimized_data = True
            _LOGGER.debug(f"Found {found_count} cached optimized setpoints for current window.")
        elif not has_schedule:
            _LOGGER.debug("No schedule and no optimized setpoints available. Disabling HVAC in simulation.")
            measurements.hvac_state[:] = 0
            # Explicit status for UX transparency
            self._optimization_status = {
                "success": False, 
                "message": "No schedule available; HVAC disabled for simulation",
                "code": "no_schedule",
                "scope": "simulation"
            }
        elif self._optimization_status and self._optimization_status.get("code") == "no_schedule":
            # Schedule is back! Clear the old error status
            self._optimization_status = None

        # Use best available setpoints (Optimized -> Target -> Clamped Input)
        measurements.setpoint = np.array(sim_setpoints, dtype=float)
        
        # Universal NaN-Clamping: Replace ALL NaNs in the setpoint array (partial gaps) with safe constant
        # AND force HVAC off for those specific points to avoid spurious action
        mask = np.isnan(measurements.setpoint)
        if np.any(mask):
            measurements.setpoint[mask] = float(measurements.t_in[0])
            # Defensive check: ensure hvac_state is aligned and mutable
            if isinstance(measurements.hvac_state, np.ndarray) and len(measurements.hvac_state) == len(mask):
                 measurements.hvac_state[mask] = 0


        
        # Recompute effective HVAC state for True-Off gating (optimized intent)
        # This ensures proper idle/blower accounting
        min_sp = self.config_entry.options.get(CONF_MIN_SETPOINT, DEFAULT_MIN_SETPOINT)
        max_sp = self.config_entry.options.get(CONF_MAX_SETPOINT, DEFAULT_MAX_SETPOINT)
        effective_hvac_state = get_effective_hvac_state(
            measurements.hvac_state,
            measurements.setpoint,
            min_sp,
            max_sp,
            DEFAULT_OFF_INTENT_EPS
        )
        
        # 1. Run Continuous Model (for Chart/Display/Optimizer)
        # --------------------------------------------------------------------------------
        sim_temps_continuous, _, hvac_produced_continuous = await self.hass.async_add_executor_job(
            partial(run_model_continuous, params, 
                t_out_list=measurements.t_out.tolist(), 
                solar_kw_list=measurements.solar_kw.tolist(),
                dt_hours_list=measurements.dt_hours.tolist(), 
                setpoint_list=measurements.setpoint.tolist(), 
                hvac_state_list=effective_hvac_state.tolist(),  # Off-gated
                max_caps_list=self.heat_pump.get_max_capacity(measurements.t_out).tolist(), 
                min_output=self.heat_pump.min_output_btu_hr, 
                max_cool=self.heat_pump.max_cool_btu_hr, 
                eff_derate=params[5], 
                start_temp=float(measurements.t_in[0])
            )
        )
        sim_temps = sim_temps_continuous # Primary Display

        if len(sim_temps) > 0:
            _LOGGER.info("Simulation complete. Predicted Final Temp: %.1f", sim_temps[-1])

        # --- Energy Calculation ---
        # We now compute 2x2 = 4 metrics:
        # 1. Continuous Naive (Baseline)
        # 2. Continuous Optimized
        # 3. Discrete Naive (Baseline Verification)
        # 4. Discrete Optimized (Verification)
        
        # Helper for efficient energy calc
        def calc_energy(produced, hvac_states=None):
            # "True Off" is already enforced in hvac_states upstream!
             return calculate_energy_vectorized(
                 produced, measurements.dt_hours, 
                 self.heat_pump.get_max_capacity(measurements.t_out),
                 self.heat_pump.get_cop(measurements.t_out), 
                 self.heat_pump,
                 # ACTUALLY:
                 # If we run at 50% capacity:
                 # We consume 50% compressor power.
                 # We deliver 50% * Derate heat.
                 # So if produced is "Compressor Capacity Used", then Watts = Produced / Base COP.
                 # So yes, eff_derate cancels out in the Watts calc if produced is capacity-side.
                 # So eff_derate=1.0 is correct if produced is "gross capacity".
                 hvac_states=hvac_states,
                 t_out=measurements.t_out,
                 include_defrost=True
             )


        # Config Mode
        hvac_mode = self.config_entry.options.get(CONF_HVAC_MODE, "heat")
        hvac_mode_val = 1 if hvac_mode == "heat" else -1
        

        # A. Continuous Optimized (The run we just did)
        eng_continuous_opt_res = calc_energy(
            np.array(hvac_produced_continuous),
            hvac_states=effective_hvac_state  # Off-gated
        )
        energy_kwh_continuous_optimized = eng_continuous_opt_res['kwh']
        # energy_kwh_steps is now fetched from Discrete below

        # B. Discrete Optimized (Verification and Reporting)
        # --------------------------------------------------------------------------------
        swing = self.config_entry.options.get(CONF_SWING_TEMP, DEFAULT_SWING_TEMP)
        min_cycle = self.config_entry.options.get(CONF_MIN_CYCLE_DURATION, DEFAULT_MIN_CYCLE_MINUTES)
        
        sim_temps_discrete, _, hvac_produced_discrete, actual_state_discrete, diag_discrete = await self.hass.async_add_executor_job(
            partial(run_model_discrete, params, 
                t_out_list=measurements.t_out.tolist(), 
                solar_kw_list=measurements.solar_kw.tolist(),
                dt_hours_list=measurements.dt_hours.tolist(), 
                setpoint_list=measurements.setpoint.tolist(), 
                hvac_state_list=effective_hvac_state.tolist(),  # Off-gated
                max_caps_list=self.heat_pump.get_max_capacity(measurements.t_out).tolist(), 
                min_output=self.heat_pump.min_output_btu_hr, 
                max_cool=self.heat_pump.max_cool_btu_hr, 
                eff_derate=params[5], 
                start_temp=float(measurements.t_in[0]),
                swing_temp=float(swing), 
                min_cycle_minutes=float(min_cycle)
            )
        )
        
        eng_discrete_opt_res = calc_energy(
            np.array(hvac_produced_discrete),
            hvac_states=effective_hvac_state  # Off-gated
        )
        energy_kwh_discrete_optimized = eng_discrete_opt_res['kwh']
        # Use Discrete Steps for Hourly Graph (Real-World)
        energy_kwh_steps = eng_discrete_opt_res.get('kwh_steps')
        
        # Diagnostics from Discrete Optimized
        diagnostics = diag_discrete
        
        # C. Discrete Naive Baseline
        # --------------------------------------------------------------------------------
        # Use original schedule setpoints for naive simulation (without mutating measurements.setpoint)
        original_schedule_setpoints = np.array(setpoint_arr, dtype=float)
        energy_kwh_discrete_naive = None
        energy_kwh_naive_steps = None
        
        if has_optimized_data:
            # Simulate with original schedule setpoints (naive baseline)
            _, _, hvac_produced_naive_disc, _, _ = await self.hass.async_add_executor_job(
                partial(run_model_discrete, params, 
                    t_out_list=measurements.t_out.tolist(), 
                    solar_kw_list=measurements.solar_kw.tolist(),
                    dt_hours_list=measurements.dt_hours.tolist(), 
                    setpoint_list=original_schedule_setpoints.tolist(),  # Use original schedule
                    hvac_state_list=measurements.hvac_state.tolist(),
                    max_caps_list=self.heat_pump.get_max_capacity(measurements.t_out).tolist(), 
                    min_output=self.heat_pump.min_output_btu_hr, 
                    max_cool=self.heat_pump.max_cool_btu_hr, 
                    eff_derate=params[5], 
                    start_temp=float(measurements.t_in[0]),
                    swing_temp=float(swing), 
                    min_cycle_minutes=float(min_cycle)
                )
            )
            eng_discrete_naive_res = calc_energy(
                np.array(hvac_produced_naive_disc),
                hvac_states=measurements.hvac_state
            )
            energy_kwh_discrete_naive = eng_discrete_naive_res['kwh']
            energy_kwh_naive_steps = eng_discrete_naive_res.get('kwh_steps')
        else:
            # No optimization data: Optimized = Naive (savings = 0)
            energy_kwh_discrete_naive = energy_kwh_discrete_optimized
            energy_kwh_naive_steps = energy_kwh_steps

        # --- Statistics Recording ---
        # Use explicit None check for numpy arrays to avoid truth value ambiguity
        used_step = energy_kwh_steps[0] if energy_kwh_steps is not None and len(energy_kwh_steps) > 0 else 0.0
        base_step = energy_kwh_naive_steps[0] if energy_kwh_naive_steps is not None and len(energy_kwh_naive_steps) > 0 else 0.0

        await self._record_stats(
            actual_temp=float(measurements.t_in[0]),
            schedule_target=float(setpoint_arr[0]) if len(setpoint_arr) > 0 else None,
            optimized_target=float(optimized_setpoint_attr[0]) if optimized_setpoint_attr and optimized_setpoint_attr[0] is not None else None,
            used_kwh=used_step,
            baseline_kwh=base_step,
        )

        # D. Off Recommendation: Combine boundary-based True-Off with simulation-based idle detection
        # --------------------------------------------------------------------------------
        model_timestep = max(1, self.config_entry.options.get(CONF_MODEL_TIMESTEP, DEFAULT_MODEL_TIMESTEP))
        control_timestep = self.config_entry.options.get(CONF_CONTROL_TIMESTEP, DEFAULT_CONTROL_TIMESTEP)
        steps_per_block = max(1, int(round(control_timestep / model_timestep)))
        
        simulated_idle = detect_idle_blocks(
            actual_state=np.array(actual_state_discrete),
            sim_temps=np.array(sim_temps_discrete),
            intent=effective_hvac_state,
            setpoints=measurements.setpoint,
            swing=float(swing),
            steps_per_block=steps_per_block,
            margin=DEFAULT_IDLE_MARGIN
        )
        
        # Combined signal: True-Off (boundary) OR simulated idle
        # Ensure effective_hvac_state is numpy array for proper boolean operations
        effective_hvac_state = np.asarray(effective_hvac_state)
        off_recommended = (effective_hvac_state == 0) | simulated_idle

        # 8. Return Result
        return self._build_coordinator_data(
            timestamps, sim_temps, measurements, optimized_setpoint_attr if has_optimized_data else None,
            metrics={
                "continuous_optimized": energy_kwh_continuous_optimized,
                "discrete_naive": energy_kwh_discrete_naive,
                "discrete_optimized": energy_kwh_discrete_optimized,
                "discrete_diagnostics": diagnostics
            },
            original_schedule=setpoint_arr,
            energy_kwh_steps=energy_kwh_steps,
            off_recommended_list=off_recommended.tolist()
        )

    def _build_coordinator_data(self, timestamps, sim_temps, measurements, optimized_setpoints=None, metrics=None, original_schedule=None, energy_kwh_steps=None, off_recommended_list=None):
        """Build the coordinator data dict from simulation results."""
        if metrics is None: metrics = {}
        
        # Use provided original_schedule if available, otherwise fallback to measurements.setpoint 
        display_setpoints = original_schedule if original_schedule is not None else measurements.setpoint

        result = {
            "timestamps": timestamps,
            "predicted_temp": sim_temps,
            "hvac_state": measurements.hvac_state,
            "setpoint": display_setpoints, # Return original schedule for comparison
            "solar": measurements.solar_kw,
            "outdoor": measurements.t_out,
            # Legacy fields (mapped directly to Discrete for Sensor Reporting)
            "energy_kwh": metrics.get("discrete_naive"),
            "optimized_energy_kwh": metrics.get("discrete_optimized"),
            "energy_kwh_steps": energy_kwh_steps,  # Per-step energy for sensor hourly aggregation
            # Detailed Metrics for debugging/transparency
            "energy_metrics": {
                "continuous_optimized": metrics.get("continuous_optimized"),
                "discrete_naive": metrics.get("discrete_naive"),
                "discrete_optimized": metrics.get("discrete_optimized"),
                "discrete_diagnostics": metrics.get("discrete_diagnostics")
            }
        }
        
        # Add Away Info for Sensor
        is_away, away_end, away_temp = self._get_away_status()
        if is_away:
             result["away_info"] = {
                 "active": True,
                 "temp": float(away_temp),
                 "end": away_end.isoformat()
             }
        else:
             result["away_info"] = {"active": False}
        
        if optimized_setpoints is not None:
            result["optimized_setpoint"] = optimized_setpoints
        
        # Include current optimization status in data for sensor attributes
        if self._optimization_status:
            # Inject dynamic off_recommended list into status
            opt_status = self._optimization_status.copy()
            if off_recommended_list is not None:
                opt_status["off_recommended"] = off_recommended_list
            result["optimization_status"] = opt_status
        
        return result




    async def _record_stats(
        self,
        actual_temp: float,
        schedule_target: Optional[float],
        optimized_target: Optional[float],
        used_kwh: Optional[float],
        baseline_kwh: Optional[float],
    ) -> None:
        """Record statistics samples for accuracy, comfort, and energy tracking."""
        # Check if stats_store is available (attached by __init__.py)
        stats_store = getattr(self, "stats_store", None)
        if stats_store is None:
            return
        
        try:
            # Get tolerance from config - use swing_temp for schedule target
            swing_temp = self.config_entry.options.get(CONF_SWING_TEMP, DEFAULT_SWING_TEMP)
            
            # 1. Resolve pending predictions (compare past predictions with actual)
            now = dt_util.now()
            resolved = stats_store.resolve_predictions(now, actual_temp)
            if resolved > 0:
                _LOGGER.debug("Resolved %d predictions with actual temp %.1f", resolved, actual_temp)
            
            # 2. Record comfort sample
            if schedule_target is not None:
                stats_store.record_comfort_sample(
                    actual_temp=actual_temp,
                    schedule_target=schedule_target,
                    schedule_tolerance=swing_temp,  # Use swing for "in range"
                )
            
            # 3. Record energy sample
            if used_kwh is not None and baseline_kwh is not None:
                stats_store.record_energy_sample(
                    used_kwh=used_kwh,
                    baseline_kwh=baseline_kwh
                )
            
            # 4. Prune old data periodically
            stats_store.prune_old_data()
            
            # 5. Save periodically (every ~10 updates to limit I/O)
            if not hasattr(self, "_stats_save_counter"):
                self._stats_save_counter = 0
            self._stats_save_counter += 1
            if self._stats_save_counter >= 10:
                await stats_store.async_save()
                self._stats_save_counter = 0
                
        except Exception as e:
            _LOGGER.warning("Failed to record stats: %s", e)

    async def async_trigger_optimization(self, duration_hours=None):
        """Manually trigger the HVAC optimization process."""
        _LOGGER.info("Manual optimization triggered. Duration override: %s", duration_hours)
        
        # 1. Ensure heat pump is ready
        if not self.heat_pump:
            await self._setup_heat_pump()
            
        # Check Schedule Enabled
        schedule_enabled = self.config_entry.options.get(CONF_SCHEDULE_ENABLED, DEFAULT_SCHEDULE_ENABLED)
        if not schedule_enabled:
            _LOGGER.warning("Optimization triggered but schedule is disabled. Ignoring.")
            return # Should we raise or just return empty?
        
        try:
            # Pass duration override if provided
            measurements, params, start_time = await self._prepare_simulation_inputs(duration_override=duration_hours)
        except Exception as e:
            _LOGGER.error("Could not prepare data for optimization: %s", e)
            raise e # Raise to caller for Service handling

        # Optimization Parameters
        model_timestep = self.config_entry.options.get(CONF_MODEL_TIMESTEP, DEFAULT_MODEL_TIMESTEP)
        control_timestep = self.config_entry.options.get(CONF_CONTROL_TIMESTEP, DEFAULT_CONTROL_TIMESTEP)
        
        _LOGGER.info(f"Running HVAC Optimization (Model: {model_timestep}m, Control: {control_timestep}m)...")
        opt_start_time = time.time()
        
        # Target Temps (Schedule)
        target_temps = measurements.setpoint.copy()
        
        # Get True-Off config for consistent energy accounting
        options = self.config_entry.options
        hvac_mode = options.get(CONF_HVAC_MODE, "heat")
        hvac_mode_val_baseline = 1 if hvac_mode == "heat" else -1
        min_setpoint_val = float(options.get(CONF_MIN_SETPOINT, DEFAULT_MIN_SETPOINT))
        max_setpoint_val = float(options.get(CONF_MAX_SETPOINT, DEFAULT_MAX_SETPOINT))
              
        # Build comfort_config from options
        options = self.config_entry.options
        comfort_config = {
            "mode": options.get(CONF_HVAC_MODE, "heat"),
            "center_preference": float(options.get(CONF_CENTER_PREFERENCE, DEFAULT_CENTER_PREFERENCE)),
            "avoid_defrost": options.get(CONF_AVOID_DEFROST, True),
            "min_setpoint": float(options.get(CONF_MIN_SETPOINT, DEFAULT_MIN_SETPOINT)),
            "max_setpoint": float(options.get(CONF_MAX_SETPOINT, DEFAULT_MAX_SETPOINT)),
        }
        
        try:
            optimization_result = await self.hass.async_add_executor_job(
                partial(
                    optimize_hvac_schedule,
                    measurements,
                    params,
                    self.heat_pump,
                    target_temps,
                    comfort_config,
                    block_size_minutes=control_timestep,
                    enable_multiscale=self.config_entry.options.get(CONF_ENABLE_MULTISCALE, DEFAULT_ENABLE_MULTISCALE),
                    rate_per_step=measurements.tou_rate  # TOU rate weighting
                )
            )
            
            # Unpack result (setpoints, meta)
            if isinstance(optimization_result, tuple):
                optimized_setpoints, meta = optimization_result
                if self.data is None:
                    self.data = {}
                
                # --- Immediate Derivation of "True Off" Signal ---
                # We must derive this NOW so the sensor updates immediately with the correct signal.
                # Reusing logic from _async_update_data / utils.py
                min_sp = comfort_config.get("min_setpoint", DEFAULT_MIN_SETPOINT)
                max_sp = comfort_config.get("max_setpoint", DEFAULT_MAX_SETPOINT)
                
                derived_effective_hvac = get_effective_hvac_state(
                    measurements.hvac_state, 
                    optimized_setpoints,     
                    min_sp,
                    max_sp,
                    DEFAULT_OFF_INTENT_EPS
                )
                off_recommended_list = (derived_effective_hvac == 0).tolist()
                
                meta_for_display = meta.copy()
                meta_for_display["off_recommended"] = off_recommended_list
                self.data["optimization_status"] = meta_for_display
                
                # Persistence: Store clean status (without large list)
                # Derived again on next update.
                self._optimization_status = meta.copy() # Meta itself is clean now (removed from optimize.py)
                
                # Check for Failure (Strict Mode)
                if optimized_setpoints is None:
                     _LOGGER.error("Optimization Failed: %s. CLEARING CACHE.", meta.get("message"))
                     self.optimized_setpoints_map.clear()
                     return
                else:
                     if meta.get("success"):
                         _LOGGER.info("Optimization converged (Cost: %.2f)", meta.get("cost", 0.0))
                     else:
                         _LOGGER.warning("Optimization warning: %s", meta.get("message"))
            else:
                # Legacy fallback
                optimized_setpoints = optimization_result
                self.data["optimization_status"] = {"success": True, "message": "Legacy Result"}
            
            opt_duration = time.time() - opt_start_time
            _LOGGER.info("Optimization completed in %.2f seconds", opt_duration)
            
            # Map optimized setpoints to timestamps
            timestamps = measurements.timestamps
            if optimized_setpoints is not None and len(optimized_setpoints) == len(timestamps):
                 new_cache = {}
                 for i, ts in enumerate(timestamps):
                     ts_int = int(ts.timestamp())
                     new_cache[ts_int] = optimized_setpoints[i]
                 self.optimized_setpoints_map.update(new_cache)
            

            
            # Update data directly (Race-free rebuild from cache)
            sim_setpoints = []
            for ts in timestamps:
                ts_int = int(ts.timestamp())
                if ts_int in self.optimized_setpoints_map:
                    sim_setpoints.append(self.optimized_setpoints_map[ts_int])
                else:
                    # Missing (e.g. failure cleared it) -> None
                    sim_setpoints.append(None)
            
            measurements.setpoint = np.array(sim_setpoints)
            

            # Recompute effective HVAC state for True-Off gating
            min_sp = comfort_config.get("min_setpoint", DEFAULT_MIN_SETPOINT)
            max_sp = comfort_config.get("max_setpoint", DEFAULT_MAX_SETPOINT)
            effective_hvac_state = get_effective_hvac_state(
                measurements.hvac_state,
                measurements.setpoint,
                min_sp,
                max_sp,
                DEFAULT_OFF_INTENT_EPS
            )

            # -- Optimized Energy Calculation (Discrete) --
            
            # Helper for calc
            def calc_energy_svc(produced, hvac_state_arr):
                 # hvac_produced from run_model_* is GROSS (pre-derate).
                 # eff_derate is already applied in thermal physics (run_model).
                 # Pass eff_derate=1.0 to avoid double-derate in energy calc.
                 return calculate_energy_vectorized(
                     produced, measurements.dt_hours, 
                     self.heat_pump.get_max_capacity(measurements.t_out),
                     self.heat_pump.get_cop(measurements.t_out), 
                     self.heat_pump,
                     eff_derate=1.0,  # Produced is Gross (Pre-Derate)
                     hvac_states=hvac_state_arr,
                     t_out=measurements.t_out,
                     include_defrost=True
                 )
            
            hvac_mode_val_svc = 1 if self.config_entry.options.get(CONF_HVAC_MODE, "heat") == "heat" else -1

            # Discrete Optimized (Verification Run)
            swing = self.config_entry.options.get(CONF_SWING_TEMP, DEFAULT_SWING_TEMP)
            min_cycle = self.config_entry.options.get(CONF_MIN_CYCLE_DURATION, DEFAULT_MIN_CYCLE_MINUTES)
            
            sim_temps_disc_svc, _, hvac_produced_disc_svc, actual_state_disc_svc, diag_disc_svc = await self.hass.async_add_executor_job(
                partial(run_model_discrete, params, 
                    t_out_list=measurements.t_out.tolist(), 
                    solar_kw_list=measurements.solar_kw.tolist(),
                    dt_hours_list=measurements.dt_hours.tolist(), 
                    setpoint_list=measurements.setpoint.tolist(), 
                    hvac_state_list=effective_hvac_state.tolist(),  # Off-gated
                    max_caps_list=self.heat_pump.get_max_capacity(measurements.t_out).tolist(), 
                    min_output=self.heat_pump.min_output_btu_hr, 
                    max_cool=self.heat_pump.max_cool_btu_hr, 
                    eff_derate=params[5], 
                    start_temp=float(measurements.t_in[0]),
                    swing_temp=float(swing), 
                    min_cycle_minutes=float(min_cycle)
                )
            )
            
            # Use 'effective_hvac_state' (Intent) for energy calc, NOT 'actual_state_discrete' (Thermostat State).
            # This ensures we charge idle power when the system is enabled (Standby) but temperature is satisfied (Off).
            disc_opt_res = calc_energy_svc(np.array(hvac_produced_disc_svc), effective_hvac_state)
            optimized_kwh_discrete = disc_opt_res['kwh']
            optimized_steps_discrete = disc_opt_res['kwh_steps']

            # --- Record predictions for accuracy tracking (from Discrete Verification) ---
            stats_store = getattr(self, "stats_store", None)
            if stats_store and len(sim_temps_disc_svc) > 0:
                try:
                    now = dt_util.now()
                    # Find index closest to 6 hours out
                    target_time_6h = now + timedelta(hours=6)
                    for i, ts in enumerate(timestamps):
                        if ts >= target_time_6h:
                            # Record this prediction
                            stats_store.record_prediction(
                                target_timestamp=ts,
                                predicted_temp=float(sim_temps_disc_svc[i]),
                                horizon_hours=6,
                            )
                            _LOGGER.debug("Recorded 6h prediction (Discrete): %.1f°F at %s", sim_temps_disc_svc[i], ts)
                            break
                except Exception as e:
                    _LOGGER.warning("Failed to record prediction: %s", e)

            
            # 3. Naive Metrics (Discrete & Continuous)
            # Calculate Discrete Naive (Baseline)
            # Baseline uses the original schedule setpoints and Raw HVAC State (1s).
            # It should NOT assume "True Off" at min setpoint; it should try to maintain 63F.
            _, _, hvac_produced_naive_disc, actual_state_naive_disc, _ = await self.hass.async_add_executor_job(
                partial(run_model_discrete, params, 
                    t_out_list=measurements.t_out.tolist(), 
                    solar_kw_list=measurements.solar_kw.tolist(),
                    dt_hours_list=measurements.dt_hours.tolist(), 
                    setpoint_list=target_temps.tolist(), # Original Schedule
                    hvac_state_list=measurements.hvac_state.tolist(), # Raw Schedule Intent (1s, not gated)
                    max_caps_list=self.heat_pump.get_max_capacity(measurements.t_out).tolist(), 
                    min_output=self.heat_pump.min_output_btu_hr, 
                    max_cool=self.heat_pump.max_cool_btu_hr, 
                    eff_derate=params[5], 
                    start_temp=float(measurements.t_in[0]),
                    swing_temp=float(swing), 
                    min_cycle_minutes=float(min_cycle)
                )
            )
            
            # Use 'measurements.hvac_state' (Raw Intent) for energy calc.
            # This ensures idle power is charged if system is enabled (1) but satisfied.
            disc_naive_res = calc_energy_svc(np.array(hvac_produced_naive_disc), measurements.hvac_state)
            discrete_naive_kwh = disc_naive_res['kwh']
            
            metrics = {
                "continuous_optimized": None,  # Not computed in service path
                "discrete_naive": discrete_naive_kwh,
                "discrete_optimized": optimized_kwh_discrete,
                "discrete_diagnostics": diag_disc_svc
            }

            # --- Update Coordinator Data Immediately ---
            result_data = self._build_coordinator_data(
                timestamps, sim_temps_disc_svc, measurements, optimized_setpoints,
                metrics=metrics,
                original_schedule=target_temps,
                energy_kwh_steps=optimized_steps_discrete  # Per-step energy for hourly aggregation (Discrete)
            )
            
            # Preserve optimization status
            if self.data and "optimization_status" in self.data:
                result_data["optimization_status"] = self.data["optimization_status"]

            self.async_set_updated_data(result_data)
            
            # Return Forecast Structure (similar to sensor)
            forecast_data = []
            for i, ts in enumerate(timestamps):
                # Prepare item dict
                item = {
                    "datetime": ts.isoformat(),
                    "target_temp": float(target_temps[i]), # Original Schedule
                    "outdoor_temp": float(measurements.t_out[i]),
                    "solar_kw": float(measurements.solar_kw[i]),
                    "ideal_setpoint": float(optimized_setpoints[i]) if i < len(optimized_setpoints) else None,
                    "predicted_temp": float(sim_temps_disc_svc[i]) if i < len(sim_temps_disc_svc) else None,
                    "energy_kwh": float(optimized_steps_discrete[i]) if optimized_steps_discrete is not None and i < len(optimized_steps_discrete) else None
                }
                # hvac_action from schedule
                state_val = measurements.hvac_state[i]
                if state_val > 0:
                    item["hvac_action"] = "heating"
                elif state_val < 0:
                     item["hvac_action"] = "cooling"
                else:
                     item["hvac_action"] = "off"
                     
                forecast_data.append(item)
                
            return {
                "forecast": forecast_data,
                "optimization_summary": {
                    "duration_seconds": opt_duration,
                    "points": len(timestamps),
                    "start_time": start_time.isoformat(),
                    "total_energy_use_kwh": float(discrete_naive_kwh),
                    "total_energy_use_optimized_kwh": float(optimized_kwh_discrete)
                }
            }
            
        except Exception as e:
            _LOGGER.error("Optimization failed: %s", e)
            import traceback
            _LOGGER.error(traceback.format_exc())
            raise e

    async def _prepare_simulation_inputs(self, duration_override=None):
        """Helper to fetch data and prepare measurements (shared logic)."""
        
        # 1. Get Inputs
        # Fixed Identity (DATA)
        data = self.config_entry.data
        sensor_indoor = data.get(CONF_SENSOR_INDOOR_TEMP)
        weather_entity = data.get(CONF_WEATHER_ENTITY)
        solar_entity = data.get(CONF_SOLAR_ENTITY)
        
        # Modifiable Settings (OPTIONS)
        options = self.config_entry.options
        if duration_override is not None:
             try:
                 duration_hours = float(duration_override)
             except (ValueError, TypeError):
                 _LOGGER.warning("Invalid duration override '%s', using config", duration_override)
                 duration_hours = options.get(CONF_FORECAST_DURATION, DEFAULT_FORECAST_DURATION)
        else:
             duration_hours = options.get(CONF_FORECAST_DURATION, DEFAULT_FORECAST_DURATION)
        

        _LOGGER.debug("DEBUG_OPTIONS: %s", options)
        
        # Parameters (Physics) - From Options
        params = [
            options.get(CONF_C_THERMAL, DEFAULT_C_THERMAL),
            options.get(CONF_UA, DEFAULT_UA),
            options.get(CONF_K_SOLAR, DEFAULT_K_SOLAR),
            options.get(CONF_Q_INT, DEFAULT_Q_INT),
            options.get(CONF_H_FACTOR, DEFAULT_H_FACTOR),
            options.get(CONF_EFF_DERATE, DEFAULT_EFF_DERATE),  # Duct delivery efficiency
        ]
        _LOGGER.debug("DEBUG_PARAMS: %s", params)

        
        _LOGGER.debug("Preparing simulation inputs with params: %s", params)

        # 2. Get Current State
        indoor_state = self.hass.states.get(sensor_indoor)
        if not indoor_state:
             raise UpdateFailed(f"Indoor sensor {sensor_indoor} not found in state machine")
             
        if indoor_state.state in ("unknown", "unavailable"):
             # This is a transient error during startup/update
             _LOGGER.debug("Indoor sensor %s is currently unavailable", sensor_indoor)
             return None # Signal 'not ready' clearly if we want, or raise specific error
        
        try:
            current_temp = float(indoor_state.state)
        except ValueError:
            raise UpdateFailed(f"Invalid indoor temp (not a number): {indoor_state.state}")

        # 3. Get Weather Forecast
        weather_state = self.hass.states.get(weather_entity)
        if not weather_state:
            raise UpdateFailed(f"Weather entity {weather_entity} not found")
        
        if weather_state.state in ("unknown", "unavailable"):
             _LOGGER.debug("Weather entity %s is currently unavailable", weather_entity)
             return None # Signal 'not ready'
        
        # Try Attribute first (Legacy)
        forecast = weather_state.attributes.get("forecast")
        
        # Try modern service call if attribute missing
        if forecast is None:
             try:
                 response = await self.hass.services.async_call(
                     "weather", 
                     "get_forecasts", 
                     {"type": "hourly", "entity_id": weather_entity}, 
                     blocking=True, 
                     return_response=True
                 )
                 if response and weather_entity in response:
                     forecast = response[weather_entity].get("forecast")
             except Exception as e:
                 _LOGGER.debug("Failed to get hourly forecast via service: %s", e)
                 
             # Fallback to daily if hourly failed or returned nothing
             if forecast is None:
                 try:
                     response = await self.hass.services.async_call(
                         "weather", 
                         "get_forecasts", 
                         {"type": "daily", "entity_id": weather_entity}, 
                         blocking=True, 
                         return_response=True
                     )
                     if response and weather_entity in response:
                         forecast = response[weather_entity].get("forecast")
                 except Exception as e:
                     _LOGGER.debug("Failed to get daily forecast via service: %s", e)

        if not forecast:
            raise UpdateFailed(
                f"No forecast data available from {weather_entity}. "
                "Check weather integration and entity configuration."
            )

        # 4. Get Solar Forecast
        solar_forecast_data = []
        solar_entities = solar_entity
        if solar_entities:
            if isinstance(solar_entities, str):
                solar_entities = [solar_entities]
            
            for entity_id in solar_entities:
                s_state = self.hass.states.get(entity_id)
                if s_state:
                     partial_forecast = s_state.attributes.get("forecast")
                     if not partial_forecast:
                         partial_forecast = s_state.attributes.get("detailedForecast", [])
                     
                     if partial_forecast:
                         solar_forecast_data.extend(partial_forecast)
                else:
                    _LOGGER.warning("Solar entity %s not found", entity_id)

        # 5. Prepare Simulation Data using Shared Handler
        now = dt_util.now()
        start_time = now
        model_timestep = self.config_entry.options.get(CONF_MODEL_TIMESTEP, DEFAULT_MODEL_TIMESTEP)
        
        timestamps, t_out_arr, solar_arr, dt_values = await self.input_handler.prepare_simulation_data(
            forecast if forecast else [],
            solar_forecast_data,
            start_time,
            duration_hours,
            model_timestep
        )
        
        if not timestamps:
            raise UpdateFailed("No forecast data available for simulation period")
        
        schedule_json = self.config_entry.options.get(CONF_SCHEDULE_CONFIG, DEFAULT_SCHEDULE_CONFIG)
        schedule_enabled = self.config_entry.options.get(CONF_SCHEDULE_ENABLED, DEFAULT_SCHEDULE_ENABLED)
        
        # Prepare arrays
        if not schedule_enabled:
             # Disabled: HVAC OFF (0), Setpoint None/0, Fixed False
             steps = len(timestamps)
             hvac_state_arr = np.zeros(steps)
             setpoint_arr = np.full(steps, None) # Or 0? Coordinator usually expects float. 
             # Let's use 0.0 and handle display in frontend, or None if typed array supports it.
             # numpy float array doesn't support None (NaN). 
             # For now use 0.0 and we'll see. Actually measurements.setpoint is used in run_model.
             # run_model needs valid float. If disabled, we probably want 0.0 or something that means "off semantics"
             # But if hvac_state is 0, setpoint is ignored for energy/action, but might be used for error calc?
             # Actually run_model uses setpoint just for control logic if it's deciding.
             # If hvac_state is provided (from measurements), run_model might just output physics.
             # Let's check run_model usage. It calculates heat/cool required based on setpoint if not provided?
             # Actually run_model takes hvac_state from measurements if we are just simulating what happened?
             # No, run_model is usually strictly thermodynamic response given inputs.
             # If we pass measurements, it might use them.
             
             # Safest: 0.0 or a really low/high number that won't trigger?
             # If HVAC state is 0, setpoint doesn't matter for energy calc.
             setpoint_arr = np.full(steps, np.nan) 
             fixed_mask_arr = np.zeros(steps, dtype=bool)
             rate_arr = np.ones(steps)  # Default rate = 1.0 (no TOU weighting)
             
             # Log once per update?
             # _LOGGER.debug("Schedule disabled, skipping processing.")
             
        else:
             try:
                 schedule_data = json.loads(schedule_json)
             except Exception as e:
                 _LOGGER.error("Invalid Schedule JSON: %s", e)
                 raise ValueError(f"Invalid Schedule JSON: {e}")
     
             is_away, away_end, away_temp = self._get_away_status()
             
             # Get HVAC Mode from Options (Tunable setting)
             configured_mode = self.config_entry.options.get(CONF_HVAC_MODE, "heat")
             
             # Run process_schedule_data in executor to avoid blocking event loop (pytz I/O)
             from functools import partial
             hvac_state_arr, setpoint_arr, fixed_mask_arr, rate_arr = await self.hass.async_add_executor_job(
                 partial(
                     process_schedule_data,
                     timestamps, 
                     schedule_data, 
                     away_status=(is_away, away_end, away_temp),
                     timezone=self.hass.config.time_zone,
                     default_mode=configured_mode
                 )
             )
            
             # --- Upstream "True Off" Enforcement ---
             min_setpoint = self.config_entry.options.get(CONF_MIN_SETPOINT, DEFAULT_MIN_SETPOINT)
             max_setpoint = self.config_entry.options.get(CONF_MAX_SETPOINT, DEFAULT_MAX_SETPOINT)

        steps = len(timestamps)
        t_in_arr = np.zeros(steps)
        t_in_arr[0] = current_temp

        measurements = Measurements(
            timestamps=np.array(timestamps),
            t_in=t_in_arr.astype(float),
            t_out=t_out_arr.astype(float),
            solar_kw=solar_arr.astype(float),
            hvac_state=np.array(hvac_state_arr, dtype=int),
            setpoint=np.array(setpoint_arr, dtype=float),
            dt_hours=np.array(dt_values, dtype=float),
            is_setpoint_fixed=np.array(fixed_mask_arr, dtype=bool),
            target_temp=np.array(setpoint_arr, dtype=float).copy(),
            tou_rate=np.array(rate_arr, dtype=float)
        )
        
        return measurements, params, start_time


    async def async_set_away_mode(self, duration_delta: timedelta, safety_temp: float):
        """Set the away mode with a duration and safety temperature."""
        _LOGGER.info("Setting away mode. Duration: %s, Safety Temp: %s", duration_delta, safety_temp)
        
        # 1. Calculate End Time
        now = dt_util.now()
        away_end = now + duration_delta
        
        # 2. Persist to Config Entry (Options)
        # We need to update options to persist across restarts
        new_options = self.config_entry.options.copy()
        new_options["away_end"] = away_end.isoformat()
        new_options["away_temp"] = float(safety_temp)
        
        self.hass.config_entries.async_update_entry(self.config_entry, options=new_options)
        
        # Discard pending predictions - they were made for normal schedule, now invalid
        stats_store = getattr(self, "stats_store", None)
        if stats_store:
            stats_store.discard_pending_predictions()
        
        # 3. Schedule Smart Wake-Up (12 hours before return)
        # Cancel any existing timer
        if hasattr(self, "_away_timer_unsub") and self._away_timer_unsub:
            self._away_timer_unsub()
            self._away_timer_unsub = None
            
        wakeup_time = away_end - timedelta(hours=AWAY_WAKEUP_ADVANCE_HOURS)
        if wakeup_time > dt_util.now():
            from homeassistant.helpers.event import async_track_point_in_time
            
            async def _wake_up_callback(now):
                _LOGGER.info("Smart Wake-Up Triggered! Re-optimizing for return...")
                try:
                    await self.async_trigger_optimization()
                except Exception as e:
                    _LOGGER.error("Smart Wake-Up Optimization failed: %s", e)
                    
            _LOGGER.info("Scheduling Smart Wake-Up optimization for %s", wakeup_time)
            self._away_timer_unsub = async_track_point_in_time(self.hass, _wake_up_callback, wakeup_time)
        
        # 4. Trigger Immediate Optimization
        # This will use the new away settings (via _process_schedule checking config options)
        return await self.async_trigger_optimization()

    def _get_away_status(self):
        """Get current away status from config."""
        options = self.config_entry.options
        away_end_str = options.get("away_end")
        away_temp = options.get("away_temp", DEFAULT_AWAY_TEMP)
        
        if not away_end_str:
            return False, None, None
            
        try:
            away_end = dt_util.parse_datetime(away_end_str)
            if away_end.tzinfo is None: # handle legacy/missing TZ
                 away_end = away_end.replace(tzinfo=dt_util.get_time_zone(self.hass.config.time_zone))
                 
            if dt_util.now() < away_end:
                 return True, away_end, away_temp
            
            # Away expired - clean up stale entries
            new_options = {k: v for k, v in options.items() 
                          if k not in ("away_end", "away_temp")}
            self.hass.config_entries.async_update_entry(self.config_entry, options=new_options)
            
            # Discard pending predictions - they were made for away mode, now invalid
            stats_store = getattr(self, "stats_store", None)
            if stats_store:
                stats_store.discard_pending_predictions()
        except Exception as e:
            _LOGGER.warning("Error parsing away_end: %s", e)
            
        return False, None, None
