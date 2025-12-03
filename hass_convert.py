#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ================== CONFIG ==================

# Core columns
TIMESTAMP_COL = "last_changed"   # change if your time column is named differently
ENTITY_COL    = "entity_id"
STATE_COL     = "state"

# Indoor temp averaging
RESAMPLE_FREQ       = "60s"   # grid for time-weighted indoor average
INDOOR_AVG_COL_NAME = "indoor_temp"

# Lifetime power entity & output column
LIFETIME_POWER_ENTITY = "sensor.lifetime_power"
LIFETIME_POWER_KW_COL = "lifetime_power_kw"

# Climate processing
CLIMATE_PREFIX = "climate."

# Column holding hvac_action for climate entities
CLIMATE_HVAC_ACTION_COL = "hvac_action"

# Column holding target temperature for climate entities
CLIMATE_TARGET_TEMP_COL = "temperature"

# Column holding current temperature for climate entities
CLIMATE_CURRENT_TEMP_COL = "current_temperature"

# Outdoor temp output column
OUTDOOR_TEMP_COL = "outdoor_temp"

# ============================================

def build_indoor_average(df: pd.DataFrame) -> pd.Series:
    """
    Time-weighted indoor average temperature using:
      - sensor.* temperature-like entities (if any, and not outside/outdoor)
      - current_temperature from all climate.* entities

    Implementation:
      - For each entity, build its own time series of temps.
      - Resample each series on RESAMPLE_FREQ with ffill (step function).
      - Align all series on a common index and average row-wise.

    Returns a Series indexed by the resampled timestamps with the
    averaged indoor temp.
    """

    ent = df[ENTITY_COL].astype(str)
    series_list = []
    used_entities = []

    # -------------------------
    # 1) Indoor sensor temperatures (if you ever add sensor.*)
    # -------------------------
    is_sensor = ent.str.startswith("sensor.")
    is_tempish = (
    ent.str.contains("temperature", case=False, na=False)
    | ent.str.contains("temp", case=False, na=False)
    )
    has_outside = ent.str.contains("outside", case=False, na=False)
    has_outdoor = ent.str.contains("outdoor", case=False, na=False)
    has_jbw = ent.str.contains("jbw_station", case=False, na=False)

    mask_indoor_sensor = is_sensor & is_tempish & ~(has_outside | has_outdoor | has_jbw)
    sensor_df = df[mask_indoor_sensor].copy()

    if not sensor_df.empty:
        sensor_df[TIMESTAMP_COL] = pd.to_datetime(sensor_df[TIMESTAMP_COL])
        sensor_df = sensor_df.sort_values(TIMESTAMP_COL)

    for entity_id, sub in sensor_df.groupby(ENTITY_COL):
        vals = pd.to_numeric(sub[STATE_COL], errors="coerce")
        vals = vals.dropna()
        if vals.empty:
            continue
        ts = pd.to_datetime(sub.loc[vals.index, TIMESTAMP_COL])
        ser = pd.Series(vals.values, index=ts).sort_index()
        ser = ser.resample(RESAMPLE_FREQ).ffill()
        series_list.append(ser.rename(entity_id))
        used_entities.append(entity_id)

    # -------------------------
    # 2) Climate current_temperature
    # -------------------------
    mask_climate = ent.str.startswith(CLIMATE_PREFIX)
    climate_df = df[mask_climate].copy()

    if CLIMATE_CURRENT_TEMP_COL in df.columns and not climate_df.empty:
        climate_df[TIMESTAMP_COL] = pd.to_datetime(climate_df[TIMESTAMP_COL])
        climate_df = climate_df.sort_values(TIMESTAMP_COL)

        for entity_id, sub in climate_df.groupby(ENTITY_COL):
            vals = pd.to_numeric(sub[CLIMATE_CURRENT_TEMP_COL], errors="coerce")
            vals = vals.dropna()
            if vals.empty:
               continue
            ts = pd.to_datetime(sub.loc[vals.index, TIMESTAMP_COL])
            ser = pd.Series(vals.values, index=ts).sort_index()
            ser = ser.resample(RESAMPLE_FREQ).ffill()
            series_list.append(ser.rename(entity_id))
            used_entities.append(entity_id)

    # -------------------------
    # 3) Combine & average
    # -------------------------
    if not series_list:
        print("No indoor temperature inputs (sensor or climate) found.")
        return pd.Series(dtype=float)

    print("Indoor temperature inputs used for averaging:")
    for e in sorted(set(used_entities)):
        print(f"  - {e}")

    # build common index
    common_index = series_list[0].index
    for ser in series_list[1:]:
        common_index = common_index.union(ser.index)
    common_index = common_index.sort_values()

    # align all series on common index
    wide = pd.DataFrame(index=common_index)
    for ser in series_list:
        wide[ser.name] = ser.reindex(common_index)

    avg_series = wide.mean(axis=1).dropna()

    return avg_series


def interpolate_outdoor(df: pd.DataFrame, primary_index: pd.DatetimeIndex) -> pd.Series:
    """
    Build a step-function outdoor temp from outdoor sensors and
    evaluate it at primary_index timestamps.
    """
    ent = df[ENTITY_COL].astype(str)
    is_sensor = ent.str.startswith("sensor.")
    is_temperature = ent.str.contains("_temp")
    has_outside = ent.str.contains("outside", case=False, na=False)
    has_outdoor = ent.str.contains("outdoor", case=False, na=False)
    has_jbw = ent.str.contains("jbw_station", case=False, na=False)

    mask_outdoor = is_sensor & is_temperature & (has_outside | has_outdoor | has_jbw)
    out_df = df[mask_outdoor].copy()

    if out_df.empty:
        print("No outdoor temperature entities found; outdoor_temp will be NaN.")
        return pd.Series(index=primary_index, dtype=float)

    out_df[TIMESTAMP_COL] = pd.to_datetime(out_df[TIMESTAMP_COL])
    out_df = out_df.sort_values(TIMESTAMP_COL)
    out_df["value_num"] = pd.to_numeric(out_df[STATE_COL], errors="coerce")
    out_df = out_df.dropna(subset=["value_num"])

    if out_df.empty:
        print("Outdoor temp rows exist, but all states are non-numeric.")
        return pd.Series(index=primary_index, dtype=float)

    wide = out_df.pivot_table(
        index=TIMESTAMP_COL,
        columns=ENTITY_COL,
        values="value_num",
        aggfunc="last",
    ).sort_index()

    combined_index = primary_index.union(wide.index)
    wide_on_union = wide.reindex(combined_index).ffill()
    wide_at_primary = wide_on_union.reindex(primary_index)

    outdoor_series = wide_at_primary.mean(axis=1)
    return outdoor_series


def build_lifetime_power_kw(df: pd.DataFrame, primary_index: pd.DatetimeIndex) -> pd.Series:
    """
    Convert cumulative kWh in sensor.lifetime_power into instantaneous
    kW (average over intervals), then interpolate to primary_index.
    """
    lp_df = df[df[ENTITY_COL] == LIFETIME_POWER_ENTITY].copy()
    if lp_df.empty:
        print(f"No rows for {LIFETIME_POWER_ENTITY}; lifetime_power_kw will be NaN.")
        return pd.Series(index=primary_index, dtype=float)

    lp_df[TIMESTAMP_COL] = pd.to_datetime(lp_df[TIMESTAMP_COL])
    lp_df = lp_df.sort_values(TIMESTAMP_COL)
    lp_df["kwh"] = pd.to_numeric(lp_df[STATE_COL], errors="coerce")
    lp_df = lp_df.dropna(subset=["kwh"])

    if lp_df.empty:
        print("Lifetime power rows exist, but all states are non-numeric.")
        return pd.Series(index=primary_index, dtype=float)

    lp_df = lp_df.drop_duplicates(subset=[TIMESTAMP_COL], keep="last")

    kwh = lp_df["kwh"]
    ts = lp_df[TIMESTAMP_COL]

    delta_kwh = kwh.diff()
    delta_sec = ts.diff().dt.total_seconds()

    with np.errstate(divide="ignore", invalid="ignore"):
        kw = (delta_kwh / (delta_sec / 3600.0))

    kw_series = pd.Series(kw.values, index=ts)
    kw_series = kw_series.dropna()

    if kw_series.empty:
        print("No valid kW values could be computed from lifetime_power.")
        return pd.Series(index=primary_index, dtype=float)

    combined_index = primary_index.union(kw_series.index)
    kw_on_union = kw_series.reindex(combined_index).sort_index()
    kw_interp = kw_on_union.interpolate(method="time")
    kw_at_primary = kw_interp.reindex(primary_index)

    return kw_at_primary


def build_hvac_and_target(df: pd.DataFrame, primary_index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """
    From climate.* entities, build:
      - hvac_mode: integer sum across entities (+1 heating, -1 cooling, 0 otherwise)
      - target_temp: average target temperature across entities

    Both are evaluated at primary_index using step-function (ffill) behavior.
    """
    ent = df[ENTITY_COL].astype(str)
    mask_climate = ent.str.startswith(CLIMATE_PREFIX)
    climate_df = df[mask_climate].copy()

    if climate_df.empty:
        print("No climate.* entities; hvac_mode=0 and target_temp=NaN.")
        hvac_mode = pd.Series(0, index=primary_index, dtype=int)
        target_temp = pd.Series(index=primary_index, dtype=float)
        return hvac_mode, target_temp

    climate_df[TIMESTAMP_COL] = pd.to_datetime(climate_df[TIMESTAMP_COL])
    climate_df = climate_df.sort_values(TIMESTAMP_COL)

    # ---- HVAC MODE ----
    if CLIMATE_HVAC_ACTION_COL in climate_df.columns:
        hvac_map = {"heating": 1, "cooling": -1}
        hvac_wide = []

        for entity_id, sub in climate_df.groupby(ENTITY_COL):
            hvac_series = sub.set_index(TIMESTAMP_COL)[CLIMATE_HVAC_ACTION_COL].astype(str)
            numeric = hvac_series.map(lambda s: hvac_map.get(s, 0))

            combined_index = primary_index.union(numeric.index)
            numeric_union = numeric.reindex(combined_index).sort_index().ffill()
            numeric_primary = numeric_union.reindex(primary_index)

            hvac_wide.append(numeric_primary.rename(entity_id))

        if hvac_wide:
            hvac_df = pd.concat(hvac_wide, axis=1)
            hvac_mode = hvac_df.sum(axis=1).astype(int)
        else:
            hvac_mode = pd.Series(0, index=primary_index, dtype=int)
    else:
        print(f"Column '{CLIMATE_HVAC_ACTION_COL}' not found; hvac_mode will be 0.")
        hvac_mode = pd.Series(0, index=primary_index, dtype=int)

    # ---- TARGET TEMP ----
    if CLIMATE_TARGET_TEMP_COL in climate_df.columns:
        tgt_wide = []
        for entity_id, sub in climate_df.groupby(ENTITY_COL):
            tgt_series = pd.to_numeric(
            sub.set_index(TIMESTAMP_COL)[CLIMATE_TARGET_TEMP_COL],
            errors="coerce",
            ).dropna()
            if tgt_series.empty:
                continue

            combined_index = primary_index.union(tgt_series.index)
            tgt_union = tgt_series.reindex(combined_index).sort_index().ffill()
            tgt_primary = tgt_union.reindex(primary_index)

            tgt_wide.append(tgt_primary.rename(entity_id))

        if tgt_wide:
            tgt_df = pd.concat(tgt_wide, axis=1)
            target_temp = tgt_df.mean(axis=1)
        else:
            target_temp = pd.Series(index=primary_index, dtype=float)
    else:
        print(f"Column '{CLIMATE_TARGET_TEMP_COL}' not found; target_temp will be NaN.")
        target_temp = pd.Series(index=primary_index, dtype=float)

    return hvac_mode, target_temp


def main():
    parser = argparse.ArgumentParser(
    description="Build feature CSV with indoor avg, outdoor temp, lifetime power kW, and HVAC state."
    )
    parser.add_argument("input_csv", type=Path, help="Input CSV file")
    parser.add_argument("output_csv", type=Path, help="Output CSV file")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    for col in (TIMESTAMP_COL, ENTITY_COL, STATE_COL):
        if col not in df.columns:
            raise SystemExit(f"ERROR: '{col}' not found in CSV columns: {df.columns.tolist()}")

    # --- 1) Indoor average temp -> defines primary time index ---
    indoor_avg = build_indoor_average(df)
    if indoor_avg.empty:
        raise SystemExit("Could not compute indoor average temperature (no inputs); aborting.")

    primary_index = indoor_avg.index

    # --- 2) Outdoor temp at primary timestamps ---
    outdoor_temp = interpolate_outdoor(df, primary_index)

    # --- 3) Lifetime power -> kW -> interpolate to primary timestamps ---
    lifetime_kw = build_lifetime_power_kw(df, primary_index)

    # --- 4) Climate -> hvac_mode + target_temp ---
    hvac_mode, target_temp = build_hvac_and_target(df, primary_index)

    # --- Assemble final output ---
    out = pd.DataFrame(index=primary_index)
    out.index.name = TIMESTAMP_COL

    out[INDOOR_AVG_COL_NAME] = indoor_avg
    out[OUTDOOR_TEMP_COL] = outdoor_temp
    out[LIFETIME_POWER_KW_COL] = lifetime_kw
    out["hvac_mode"] = hvac_mode
    out["target_temp"] = target_temp

    # Rename solar column (this works on columns)
    out = out.rename(columns={LIFETIME_POWER_KW_COL: "solar_kw"})

    # Drop rows without solar data
    out = out[out["solar_kw"].notna()]

    # Turn index into a column, then rename that column to "time"
    out = out.reset_index()           # index -> column named TIMESTAMP_COL
    out = out.rename(columns={TIMESTAMP_COL: "time"})

    # Add local-time column (America/Los_Angeles)
    out["time_local"] = (
        out["time"]
        .dt.tz_convert("America/Los_Angeles")
    )

    # Write final CSV
    out.to_csv(args.output_csv, index=False)
    print(f"Wrote feature CSV with primary key timestamps (indoor avg) to {args.output_csv}")

if __name__ == "__main__":
    main()

