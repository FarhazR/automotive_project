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
HARD_ACC_G  =  0.2    # threshold for "hard acceleration" (g)
MAX_ABS_ACCEL_G = 1.0  # Provisional data-quality threshold
HARD_BRK_G  = -0.2    # threshold for "hard braking"     (g)
IDLE_KMH    =  2.0    # speed below this = idling
IDLE_RPM    =  400    # rpm above this = engine running
HIGH_RPM    = 3000    # rpm above this = "high RPM" flag
SPD_80      = 80.0    # km/h – urban/suburban boundary
SPD_110     = 110.0   # km/h – highway boundary



MAX_ACCEL_GAP_S = 5.0  # Ignore acceleration intervals spanning >5 seconds

SMOOTH_WINDOW_S = 3   # centred moving average (seconds) on the 1 Hz grid

def compute_accel(grp: pd.DataFrame) -> pd.Series:
    """
    Acceleration in m/s^2 from a speed signal that updates ~1 Hz but is
    sampled at irregular, sometimes 0.1 s, intervals.

    Steps: resample speed to a 1 Hz grid -> blank grid points inside long
    gaps -> smooth with a 3 s centred mean -> central difference over the
    grid -> map back onto the original sample timestamps.
    """
    n = len(grp)
    out = np.full(n, np.nan)
    if n < 3:
        return pd.Series(out, index=grp.index, dtype=float)

    t = grp["t_ms"].to_numpy(dtype=float) / 1000.0
    v = grp["speed_kmh"].to_numpy(dtype=float) / 3.6

    ok = np.isfinite(t) & np.isfinite(v)
    if ok.sum() < 3:
        return pd.Series(out, index=grp.index, dtype=float)
    t_ok, v_ok = t[ok], v[ok]

    t_grid = np.arange(np.ceil(t_ok[0]), np.floor(t_ok[-1]) + 1.0, 1.0)
    if len(t_grid) < SMOOTH_WINDOW_S + 2:
        return pd.Series(out, index=grp.index, dtype=float)

    v_grid = np.interp(t_grid, t_ok, v_ok)

    # Grid points that fall inside a gap longer than MAX_ACCEL_GAP_S are invalid
    nxt = np.clip(np.searchsorted(t_ok, t_grid, side="right"), 1, len(t_ok) - 1)
    v_grid[(t_ok[nxt] - t_ok[nxt - 1]) > MAX_ACCEL_GAP_S] = np.nan

    vs = pd.Series(v_grid).rolling(
        SMOOTH_WINDOW_S, center=True, min_periods=SMOOTH_WINDOW_S
    ).mean()
    a = ((vs.shift(-1) - vs.shift(1)) / 2.0).to_numpy()   # grid step = 1 s

    acc = np.interp(t, t_grid, a)           # NaN propagates across invalid points
    acc[(t < t_grid[0]) | (t > t_grid[-1])] = np.nan
    acc[~np.isfinite(v)] = np.nan
    return pd.Series(acc, index=grp.index, dtype=float)




def feature_row(grp: pd.DataFrame) -> dict:
    """Compute behavioural features for one trip group."""
    acc_raw = compute_accel(grp) / G_MS2

    # Preserve the raw signal for diagnostics, but exclude extreme
    # estimates from behavioural features.
    acc = acc_raw.where(acc_raw.abs() <= MAX_ABS_ACCEL_G)
    spd = grp["speed_kmh"]
    rpm = grp["rpm"]
    load = grp["load_pct"]

    dt = pd.to_numeric(grp["dt_valid_s"], errors="coerce").fillna(0)
    dt = dt.clip(lower=0)

    total_s = dt.sum()
    if total_s <= 0:
        return {}

    # Denominators use intervals where the relevant signal is available.
    speed_valid = spd.notna()
    rpm_valid = rpm.notna()
    accel_valid = speed_valid & acc.notna()

    speed_time = dt[speed_valid].sum()
    rpm_time = dt[rpm_valid].sum()
    accel_time_valid = dt[accel_valid].sum()

    def percentage(numerator_s, denominator_s):
        if denominator_s <= 0:
            return np.nan
        return 100.0 * numerator_s / denominator_s

    # ── Speed features ────────────────────────────────────────────────
    spd_valid = spd[speed_valid]

    f = {
        "speed_mean_kmh": spd_valid.mean(),
        "speed_std_kmh": spd_valid.std(),
        "speed_max_kmh": spd_valid.max(),
        "pct_time_over_80": percentage(
            dt[speed_valid & (spd > SPD_80)].sum(), speed_time
        ),
        "pct_time_over_110": percentage(
            dt[speed_valid & (spd > SPD_110)].sum(), speed_time
        ),
    }

    # ── Acceleration and braking ──────────────────────────────────────
        # Valid acceleration samples after applying the data-quality filter
    accel_valid = speed_valid & acc.notna()
    acc_valid = acc[accel_valid]

    # Valid time denominator for acceleration-related percentages
    accel_time_valid = dt[accel_valid].sum()

    def percentage(numerator_s, denominator_s):
        if denominator_s <= 0:
            return np.nan
        return 100.0 * numerator_s / denominator_s

    # Event flags: filtered acceleration only
    hard_acc_flag = (acc >= HARD_ACC_G).fillna(False)
    hard_brk_flag = (acc <= HARD_BRK_G).fillna(False)

    hard_acc_starts = (
        hard_acc_flag
        & ~hard_acc_flag.shift(1, fill_value=False)
    )
    hard_brk_starts = (
        hard_brk_flag
        & ~hard_brk_flag.shift(1, fill_value=False)
    )

    # Time-weighted behaviour categories
    accel_mask = accel_valid & (acc >= HARD_ACC_G * 0.5)
    decel_mask = accel_valid & (acc <= HARD_BRK_G * 0.5)
    cruise_mask = accel_valid & acc.between(-0.05, 0.05)

    f.update({
        "acc_mean_g": acc_valid.mean(),
        "acc_std_g": acc_valid.std(),
        "acc_p95_g": acc_valid[acc_valid > 0].quantile(0.95),
        "acc_p05_g": acc_valid[acc_valid < 0].quantile(0.05),
        "hard_acc_count": int(hard_acc_starts.sum()),
        "hard_brk_count": int(hard_brk_starts.sum()),
        "pct_accel_time": percentage(
            dt[accel_mask].sum(), accel_time_valid
        ),
        "pct_decel_time": percentage(
            dt[decel_mask].sum(), accel_time_valid
        ),
        "pct_cruise_time": percentage(
            dt[cruise_mask].sum(), accel_time_valid
        ),
    })

    # Moderate acceleration/deceleration and near-constant-speed time.
    accel_mask = accel_valid & (acc >= HARD_ACC_G * 0.5)
    decel_mask = accel_valid & (acc <= HARD_BRK_G * 0.5)
    cruise_mask = accel_valid & acc.between(-0.05, 0.05)

    f.update({
        "acc_mean_g": acc_valid.mean(),
        "acc_std_g": acc_valid.std(),
        "acc_max_g": acc_valid.max(),
        "dec_max_g": acc_valid.min(),
        "hard_acc_count": int(hard_acc_starts.sum()),
        "hard_brk_count": int(hard_brk_starts.sum()),
        "pct_accel_time": percentage(
            dt[accel_mask].sum(), accel_time_valid
        ),
        "pct_decel_time": percentage(
            dt[decel_mask].sum(), accel_time_valid
        ),
        "pct_cruise_time": percentage(
            dt[cruise_mask].sum(), accel_time_valid
        ),
    })

    # ── RPM features ──────────────────────────────────────────────────
    rpm_values = rpm[rpm_valid]

    f.update({
        "rpm_mean": rpm_values.mean(),
        "rpm_std": rpm_values.std(),
        "rpm_max": rpm_values.max(),
        "pct_time_high_rpm": percentage(
            dt[rpm_valid & (rpm >= HIGH_RPM)].sum(), rpm_time
        ),
    })

    # ── Engine load ───────────────────────────────────────────────────
    f.update({
        "load_mean_pct": load.mean(),
        "load_std_pct": load.std(),
    })

    # ── Idling and stop-and-go ────────────────────────────────────────
    idle_valid = speed_valid & rpm_valid
    idle_mask = (
        idle_valid
        & (spd < IDLE_KMH)
        & (rpm > IDLE_RPM)
    )

    idle_observed_time = dt[idle_valid].sum()
    idle_time = dt[idle_mask].sum()

    f["idle_frac"] = (
        idle_time / idle_observed_time
        if idle_observed_time > 0
        else np.nan
    )

    # Count transitions from an observed idle sample to an
    # observed moving sample with the engine running.
    moving_mask = (
        idle_valid
        & (spd >= IDLE_KMH)
        & (rpm > IDLE_RPM)
    )

    was_idle = idle_mask.shift(1, fill_value=False)

    # dt_valid_s describes the current -> next interval.
    # Shift it to validate the previous -> current transition.
    transition_dt = dt.shift(1)

    valid_transition = (
        transition_dt.gt(0)
        & transition_dt.le(MAX_ACCEL_GAP_S)
    )

    stop_go_events = (
        was_idle
        & moving_mask
        & valid_transition
    )

    f["stop_go_events"] = int(stop_go_events.sum())

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
    feats["hard_acc_per_100km"] = feats["hard_acc_count"] / feats["distance_km"] * 100
    feats["hard_brk_per_100km"] = feats["hard_brk_count"] / feats["distance_km"] * 100
    feats["stop_go_per_km"]     = feats["stop_go_events"] / feats["distance_km"]
    
    MIN_TRIPS_PER_VEHICLE = 5
    feats["veh_n_trips"] = feats.groupby("veh_id")["trip"].transform("size")
    veh_median = feats.groupby("veh_id")["l_per_100km"].transform("median")
    feats["fuel_rel_vehicle"] = np.where(
        feats["veh_n_trips"] >= MIN_TRIPS_PER_VEHICLE,
        feats["l_per_100km"] / veh_median,
        np.nan,
    )

    # Basic sanity report
    print("\n── Feature completeness (% non-NaN) ──")
    pct = feats.notna().mean().mul(100).round(1).sort_values(ascending=False)
    print(pct.to_string())

    out_path = f"{args.out_dir}/features.csv"
    feats.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(feats):,} rows × {len(feats.columns)} cols)")


if __name__ == "__main__":
    main()