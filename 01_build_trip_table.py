#!/usr/bin/env python3
"""
Step 1 of the Driving Behaviour & Fuel Consumption project (VED dataset).

What this script does
  1. Loads the static vehicle table and keeps only conventional ICE vehicles.
  2. Loads N weekly dynamic CSVs (start small, scale up later).
  3. Writes a data-quality report (SRS FR-02).
  4. Cleans the data (invalid values -> NaN, duplicates dropped) (FR-03).
  5. Builds a TRIP-LEVEL table: duration, distance, fuel, L/100 km.
  6. Saves the cleaned per-sample data of the kept trips for feature engineering.

Usage (from ~/ved):
  python3 01_build_trip_table.py --raw-dir data/raw \
      --static "data/static/VED_Static_Data_ICE&HEV.xlsx" --weeks 4

Outputs:
  reports/data_quality.txt
  data/processed/trips.csv
  data/processed/samples_clean.csv.gz
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
SEED = 42  # fixed seed for reproducibility (NFR-04), used in later steps
MAX_GAP_S = 5.0          # sample gaps longer than this are not integrated
SPEED_MAX_KMH = 250.0    # above this -> invalid
RPM_MAX = 8000.0         # above this -> invalid
FUEL_MAX_LPH = 100.0     # above this -> invalid
MAF_MAX_GPS = 600.0      # above this -> invalid
# Fallback fuel estimate from MAF (only used when Fuel Rate is missing):
# fuel [g/s] = MAF / AFR, assuming stoichiometric gasoline (AFR ~14.7)
# and gasoline density ~745 g/L. This is an APPROXIMATION - document it.
AFR = 14.7
GASOLINE_G_PER_L = 745.0

# Column name prefixes -> clean names (robust to small header differences)
COLUMN_PREFIXES = {
    "DayNum": "day_num",
    "VehId": "veh_id",
    "Trip": "trip",
    "Timestamp": "t_ms",
    "Vehicle Speed": "speed_kmh",
    "MAF": "maf_gps",
    "Engine RPM": "rpm",
    "Absolute Load": "load_pct",
    "OAT": "oat_c",
    "Outside Air Temperature": "oat_c",
    "Fuel Rate": "fuel_lph",
}


def log(msg, report=None):
    print(msg)
    if report is not None:
        report.append(msg)


def rename_columns(columns):
    """Map raw header names to clean names using prefix matching."""
    mapping = {}
    for col in columns:
        for prefix, clean in COLUMN_PREFIXES.items():
            if col.startswith(prefix) and clean not in mapping.values():
                mapping[col] = clean
                break
    return mapping


def load_static(path, engine_type, report):
    static = pd.read_excel(path)
    static.columns = [str(c).strip() for c in static.columns]
    log(f"Static table columns: {list(static.columns)}", report)

    # The type column is called 'Vehicle Type' in the real file
    type_col = next((c for c in ("Vehicle Type", "EngineType") if c in static.columns), None)
    if type_col is None or "VehId" not in static.columns:
        sys.exit("ERROR: static file needs 'VehId' and 'Vehicle Type' columns. "
                 "Check the column list printed above.")

    counts = static[type_col].astype(str).value_counts().to_dict()
    log(f"'{type_col}' values in static file: {counts}", report)
    keep = static[type_col].astype(str).str.strip().str.upper() == engine_type.upper()
    ice = static[keep].copy()
    log(f"Vehicles kept ({type_col} == {engine_type}): {len(ice)} of {len(static)}", report)
    if ice.empty:
        sys.exit(f"ERROR: no vehicles with {type_col} == {engine_type}. "
                 "Use --engine-type with one of the values listed above.")
    return ice


def load_dynamic(raw_dir, n_weeks, veh_ids, report):
    files = sorted(glob.glob(os.path.join(raw_dir, "**", "VED_*_week.csv"), recursive=True))
    if not files:
        sys.exit(f"ERROR: no VED_*_week.csv files found under {raw_dir}")
    if n_weeks:
        files = files[:n_weeks]
    log(f"Loading {len(files)} weekly file(s): {[os.path.basename(f) for f in files]}", report)

    frames = []
    for f in files:
        header = pd.read_csv(f, nrows=0).columns
        mapping = rename_columns(header)
        wanted = [c for c in mapping if mapping[c] in
                  ("day_num", "veh_id", "trip", "t_ms", "speed_kmh", "maf_gps",
                   "rpm", "load_pct", "oat_c", "fuel_lph")]
        df = pd.read_csv(f, usecols=wanted).rename(columns=mapping)
        df = df[df["veh_id"].isin(veh_ids)]
        df["source_file"] = os.path.basename(f)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    for col in ("maf_gps", "rpm", "load_pct", "oat_c", "fuel_lph"):
        if col not in data.columns:
            data[col] = np.nan
    return data


def quality_report(df, report):
    log("\n=== DATA QUALITY REPORT (before cleaning) ===", report)
    log(f"Rows: {len(df):,} | vehicles: {df['veh_id'].nunique()} | "
        f"trips: {df.groupby(['veh_id', 'trip']).ngroups:,}", report)
    miss = (df.isna().mean() * 100).round(1)
    log("Missing values (% of rows):", report)
    for col, pct in miss.items():
        log(f"  {col:<12} {pct:5.1f}%", report)
    dups = df.duplicated(subset=["veh_id", "trip", "t_ms"]).sum()
    log(f"Duplicate (veh, trip, timestamp) rows: {dups:,}", report)
    checks = {
        "speed < 0 or > %g km/h" % SPEED_MAX_KMH:
            ((df["speed_kmh"] < 0) | (df["speed_kmh"] > SPEED_MAX_KMH)).sum(),
        "rpm < 0 or > %g" % RPM_MAX:
            ((df["rpm"] < 0) | (df["rpm"] > RPM_MAX)).sum(),
        "fuel rate < 0 or > %g L/h" % FUEL_MAX_LPH:
            ((df["fuel_lph"] < 0) | (df["fuel_lph"] > FUEL_MAX_LPH)).sum(),
        "MAF < 0 or > %g g/s" % MAF_MAX_GPS:
            ((df["maf_gps"] < 0) | (df["maf_gps"] > MAF_MAX_GPS)).sum(),
    }
    for name, n in checks.items():
        log(f"Invalid values - {name}: {int(n):,}", report)
    has_fuel = df["fuel_lph"].notna().groupby(df["veh_id"]).mean()
    has_maf = df["maf_gps"].notna().groupby(df["veh_id"]).mean()
    log(f"Vehicles with any Fuel Rate data: {(has_fuel > 0).sum()} of {len(has_fuel)}", report)
    log(f"Vehicles with any MAF data (fallback): {(has_maf > 0).sum()} of {len(has_maf)}", report)


def clean(df, report):
    log("\n=== CLEANING (FR-03) ===", report)
    n0 = len(df)
    df = df.drop_duplicates(subset=["veh_id", "trip", "t_ms"]).copy()
    log(f"Dropped duplicate rows: {n0 - len(df):,}", report)

    # Invalid values become NaN (the row is kept; only that signal is ignored)
    df.loc[(df["speed_kmh"] < 0) | (df["speed_kmh"] > SPEED_MAX_KMH), "speed_kmh"] = np.nan
    df.loc[(df["rpm"] < 0) | (df["rpm"] > RPM_MAX), "rpm"] = np.nan
    df.loc[(df["fuel_lph"] < 0) | (df["fuel_lph"] > FUEL_MAX_LPH), "fuel_lph"] = np.nan
    df.loc[(df["maf_gps"] < 0) | (df["maf_gps"] > MAF_MAX_GPS), "maf_gps"] = np.nan

    df = df.sort_values(["veh_id", "trip", "t_ms"]).reset_index(drop=True)

    # Fuel rate: use measured/derived Fuel Rate; fall back to MAF estimate.
    maf_est = df["maf_gps"] / AFR / GASOLINE_G_PER_L * 3600.0   # L/h
    df["fuel_source"] = np.where(df["fuel_lph"].notna(), "fuel_rate",
                                 np.where(maf_est.notna(), "maf_estimate", "none"))
    df["fuel_lph_used"] = df["fuel_lph"].fillna(maf_est)

    # Time step to the NEXT sample inside each trip (left Riemann sum).
    grp = df.groupby(["veh_id", "trip"], sort=False)["t_ms"]
    df["dt_s"] = (grp.shift(-1) - df["t_ms"]) / 1000.0
    ok = df["dt_s"].between(0, MAX_GAP_S, inclusive="right")
    df["dt_valid_s"] = np.where(ok, df["dt_s"], 0.0)
    return df


def build_trips(df, static, min_km, min_min, min_cov, report):
    log("\n=== TRIP TABLE ===", report)
    df = df.copy()
    both = df["speed_kmh"].notna() & df["fuel_lph_used"].notna()
    df["dt_both_s"] = np.where(both, df["dt_valid_s"], 0.0)
    df["dist_km"] = np.where(both, df["speed_kmh"] * df["dt_valid_s"] / 3600.0, 0.0)
    df["fuel_l"] = np.where(both, df["fuel_lph_used"] * df["dt_valid_s"] / 3600.0, 0.0)
    df["maf_time_s"] = np.where(both & (df["fuel_source"] == "maf_estimate"),
                                df["dt_valid_s"], 0.0)

    g = df.groupby(["veh_id", "trip"], sort=False)
    trips = g.agg(
        n_samples=("t_ms", "size"),
        day_num=("day_num", "min"),
        dur_total_s=("dt_valid_s", "sum"),
        dur_both_s=("dt_both_s", "sum"),
        distance_km=("dist_km", "sum"),
        fuel_l=("fuel_l", "sum"),
        maf_time_s=("maf_time_s", "sum"),
    ).reset_index()
    trips["duration_min"] = trips["dur_total_s"] / 60.0
    trips["fuel_coverage"] = trips["dur_both_s"] / trips["dur_total_s"].replace(0, np.nan)
    trips["pct_fuel_from_maf"] = 100 * trips["maf_time_s"] / trips["dur_both_s"].replace(0, np.nan)
    trips["avg_speed_kmh"] = trips["distance_km"] / (trips["dur_both_s"] / 3600.0).replace(0, np.nan)
    trips["l_per_100km"] = 100.0 * trips["fuel_l"] / trips["distance_km"].replace(0, np.nan)

    n_all = len(trips)
    log(f"Trips before filtering: {n_all:,}", report)
    steps = [
        (f"duration >= {min_min:g} min", trips["duration_min"] >= min_min),
        (f"distance >= {min_km:g} km", trips["distance_km"] >= min_km),
        (f"fuel coverage >= {min_cov:g}", trips["fuel_coverage"] >= min_cov),
        ("L/100km between 1 and 40 (plausibility)", trips["l_per_100km"].between(1, 40)),
    ]
    mask = pd.Series(True, index=trips.index)
    for name, cond in steps:
        before = int(mask.sum())
        mask &= cond.fillna(False)
        log(f"  after {name}: {int(mask.sum()):,} (dropped {before - int(mask.sum()):,})", report)
    kept = trips[mask].copy()

    # Attach static vehicle info (class, displacement, weight, ...)
    kept = kept.merge(static.rename(columns={"VehId": "veh_id"}), on="veh_id", how="left")
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--static", required=True)
    ap.add_argument("--weeks", type=int, default=4, help="number of weekly files to load (0 = all)")
    ap.add_argument("--engine-type", default="ICE")
    ap.add_argument("--min-km", type=float, default=2.0)
    ap.add_argument("--min-minutes", type=float, default=5.0)
    ap.add_argument("--min-coverage", type=float, default=0.9)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--report-dir", default="reports")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    report = []

    static = load_static(args.static, args.engine_type, report)
    raw = load_dynamic(args.raw_dir, args.weeks, set(static["VehId"]), report)
    if raw.empty:
        sys.exit("ERROR: no dynamic rows for the selected vehicles in the loaded weeks. "
                 "Try more weeks (--weeks 8).")

    quality_report(raw, report)
    df = clean(raw, report)
    trips = build_trips(df, static, args.min_km, args.min_minutes, args.min_coverage, report)

    log(f"\nTrips kept: {len(trips):,} from {trips['veh_id'].nunique()} vehicle(s)", report)
    if not trips.empty:
        log("L/100km summary of kept trips:", report)
        log(trips["l_per_100km"].describe().round(2).to_string(), report)
        log(f"Trips relying on MAF-estimated fuel for >50% of their time: "
            f"{int((trips['pct_fuel_from_maf'] > 50).sum()):,}", report)

    trips.to_csv(os.path.join(args.out_dir, "trips.csv"), index=False)
    keep_keys = trips[["veh_id", "trip"]]
    samples = df.merge(keep_keys, on=["veh_id", "trip"], how="inner")
    samples.to_csv(os.path.join(args.out_dir, "samples_clean.csv.gz"), index=False)
    with open(os.path.join(args.report_dir, "data_quality.txt"), "w") as fh:
        fh.write("\n".join(report) + "\n")
    print(f"\nSaved: {args.out_dir}/trips.csv, {args.out_dir}/samples_clean.csv.gz, "
          f"{args.report_dir}/data_quality.txt")


if __name__ == "__main__":
    main()
