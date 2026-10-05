#!/usr/bin/env python3
"""
Step 2 – Feature Engineering  (SRS FR-04)

Reads:
  data/processed/trips.csv
  data/processed/samples_clean.csv.gz

Writes:
  data/processed/features.csv   – one row per trip, all behavioural + fuel features

Features produced
─────────────────
Speed          : avg, std, max, pct_time_over_80, pct_time_over_110
Acceleration   : mean, std, max_acc, max_dec,
                 hard_acc_count  (>= +0.3 g), hard_brk_count (<= -0.3 g),
                 pct_accel_time, pct_decel_time, pct_cruise_time
RPM            : mean, std, max, pct_time_high_rpm (>= 3000)
Engine Load    : mean, std  (may be NaN-heavy)
Idling         : idle_frac  (speed < 2 km/h, engine on i.e. rpm > 400)
Stop-and-go    : stop_go_events  (transitions from idle back to moving)
Fuel target    : l_per_100km  (carried from trips.csv)
"""

import argparse
import numpy as np
import pandas as pd

SEED = 42
G_MS2 = 9.81          # 1 g in m/s²
HARD_ACC_G  =  0.3    # threshold for "hard acceleration" (g)
HARD_BRK_G  = -0.3    # threshold for "hard braking"     (g)
IDLE_KMH    =  2.0    # speed below this = idling
IDLE_RPM    =  400    # rpm above this = engine running
HIGH_RPM    = 3000    # rpm above this = "high RPM" flag
SPD_80      = 80.0    # km/h – urban/suburban boundary
SPD_110     = 110.0   # km/h – highway boundary


def compute_accel(grp: pd.DataFrame) -> pd.Series:
    """
    Compute instantaneous acceleration in m/s² using central differences
    where possible, forward/backward at the edges.
    Returns a Series aligned to grp's index.
    """
    v = grp["speed_kmh"].values / 3.6          # → m/s
    t = grp["t_ms"].values / 1000.0            # → s

    dt = np.diff(t, append=np.nan)
    dt_prev = np.diff(t, prepend=np.nan)

    # central difference (most samples)
    dv = np.diff(v, append=np.nan)
    dv_prev = np.diff(v, prepend=np.nan)

    acc = np.where(
        np.isfinite(dt) & np.isfinite(dt_prev),
        (dv / np.where(dt > 0, dt, np.nan) +
         dv_prev / np.where(dt_prev > 0, dt_prev, np.nan)) / 2.0,
        np.where(
            np.isfinite(dt),
            dv / np.where(dt > 0, dt, np.nan),
            dv_prev / np.where(dt_prev > 0, dt_prev, np.nan),
        ),
    )
    return pd.Series(acc, index=grp.index)


def feature_row(grp: pd.DataFrame) -> dict:
    """Compute all behavioural features for one trip group."""
    acc = compute_accel(grp) / G_MS2          # in g-units
    spd = grp["speed_kmh"]
    rpm = grp["rpm"]
    load = grp["load_pct"]
    dt = grp["dt_valid_s"].fillna(0)           # seconds per sample interval
    total_s = dt.sum()
    if total_s == 0:
        return {}

    # ── speed features ──────────────────────────────────────────────────────
    spd_valid = spd.dropna()
    f = {
        "speed_mean_kmh"     : spd_valid.mean(),
        "speed_std_kmh"      : spd_valid.std(),
        "speed_max_kmh"      : spd_valid.max(),
        "pct_time_over_80"   : (dt[spd > SPD_80]).sum() / total_s * 100,
        "pct_time_over_110"  : (dt[spd > SPD_110]).sum() / total_s * 100,
    }

    # ── acceleration features ────────────────────────────────────────────────
    acc_valid = acc[spd.notna()]
    hard_acc = (acc_valid >= HARD_ACC_G).sum()
    hard_brk = (acc_valid <= HARD_BRK_G).sum()
    accel_time  = dt[(acc >= HARD_ACC_G * 0.5) & spd.notna()].sum()   # gentle threshold for % time
    decel_time  = dt[(acc <= HARD_BRK_G * 0.5) & spd.notna()].sum()
    cruise_time = dt[acc_valid.reindex(dt.index).between(-0.05, 0.05)].sum()

    f.update({
        "acc_mean_g"         : acc_valid.mean(),
        "acc_std_g"          : acc_valid.std(),
        "acc_max_g"          : acc_valid.max(),
        "dec_max_g"          : acc_valid.min(),
        "hard_acc_count"     : int(hard_acc),
        "hard_brk_count"     : int(hard_brk),
        "pct_accel_time"     : accel_time  / total_s * 100,
        "pct_decel_time"     : decel_time  / total_s * 100,
        "pct_cruise_time"    : cruise_time / total_s * 100,
    })

    # ── RPM features ─────────────────────────────────────────────────────────
    rpm_valid = rpm.dropna()
    f.update({
        "rpm_mean"           : rpm_valid.mean(),
        "rpm_std"            : rpm_valid.std(),
        "rpm_max"            : rpm_valid.max(),
        "pct_time_high_rpm"  : (dt[rpm >= HIGH_RPM]).sum() / total_s * 100,
    })

    # ── engine load ──────────────────────────────────────────────────────────
    f.update({
        "load_mean_pct"      : load.mean(),     # likely NaN for most vehicles
        "load_std_pct"       : load.std(),
    })

    # ── idling & stop-and-go ─────────────────────────────────────────────────
    idle_mask = (spd < IDLE_KMH) & (rpm > IDLE_RPM)
    idle_frac = dt[idle_mask].sum() / total_s

    # count transitions: not-idle → idle (each counts as one stop event)
    was_idle = idle_mask.shift(1, fill_value=False)
    stop_go_events = int((idle_mask & ~was_idle).sum())

    f.update({
        "idle_frac"          : idle_frac,
        "stop_go_events"     : stop_go_events,
    })

    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-dir", default="data/processed")
    ap.add_argument("--out-dir",  default="data/processed")
    args = ap.parse_args()

    trips   = pd.read_csv(f"{args.proc_dir}/trips.csv")
    samples = pd.read_csv(f"{args.proc_dir}/samples_clean.csv.gz")

    print(f"Loaded {len(trips):,} trips and {len(samples):,} sample rows.")

    # Compute features trip by trip
    rows = []
    grouped = samples.groupby(["veh_id", "trip"], sort=False)
    n = len(grouped)
    for i, ((veh, trip), grp) in enumerate(grouped):
        if i % 200 == 0:
            print(f"  {i}/{n} trips processed…", end="\r")
        grp = grp.sort_values("t_ms").reset_index(drop=True)
        f = feature_row(grp)
        if f:
            f["veh_id"] = veh
            f["trip"]   = trip
            rows.append(f)

    feats = pd.DataFrame(rows)
    print(f"\nFeature rows computed: {len(feats):,}")

    # Merge with trip-level fuel / distance / static vehicle info
    trip_cols = ["veh_id", "trip", "l_per_100km", "distance_km",
                 "duration_min", "avg_speed_kmh", "pct_fuel_from_maf",
                 "Vehicle Class", "Engine Configuration & Displacement",
                 "Transmission", "Drive Wheels", "Generalized_Weight"]
    trip_cols = [c for c in trip_cols if c in trips.columns]
    feats = feats.merge(trips[trip_cols], on=["veh_id", "trip"], how="left")

    # Basic sanity report
    print("\n── Feature completeness (% non-NaN) ──")
    pct = feats.notna().mean().mul(100).round(1).sort_values(ascending=False)
    print(pct.to_string())

    out_path = f"{args.out_dir}/features.csv"
    feats.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(feats):,} rows × {len(feats.columns)} cols)")


if __name__ == "__main__":
    main()