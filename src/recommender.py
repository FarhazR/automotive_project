"""
Recommendation engine (SRS FR-08). Imported by 05_profile_and_recommender.py
and 06_dashboard.py so the logic exists in exactly one place.

A feature is only recommended when the within-vehicle model (built in
05_profile_and_recommender.py) found a clearly positive association with
fuel use. Recommendations are ranked by the model's estimated fuel impact.
"""
import textwrap

import numpy as np
import pandas as pd

# Flag a feature when the trip is at least this many within-vehicle standard
# deviations worse than the efficient reference.
FLAG_THRESHOLD_SD = 0.5

FEAT_LABELS = {
    "stop_go_per_km":     "Stop-and-Go Events (per km)",
    "idle_frac":          "Idle Fraction",
    "hard_brk_per_100km": "Hard Braking (per 100 km)",
    "hard_acc_per_100km": "Hard Acceleration (per 100 km)",
    "acc_std_g":          "Acceleration Variability (g std)",
}

MESSAGES = {
    "stop_go_per_km": (
        "This trip had {val:.2f} stop-and-go events per km vs {ref:.2f} in the most "
        "fuel-efficient trips. In this dataset, trips with more stop-and-go than usual "
        "used more fuel in the same vehicle. Where traffic allows, anticipating signals "
        "and keeping a steady roll can avoid some full stops."
    ),
    "idle_frac": (
        "The engine idled for {pct:.0f}% of the trip vs {ref_pct:.0f}% in the most "
        "fuel-efficient trips. Idling uses fuel without covering distance. Route or "
        "departure-time choices that avoid long waits help, and where safe and "
        "permitted, switching off during long stationary waits is worth considering."
    ),
    "hard_brk_per_100km": (
        "This trip had {val:.0f} hard braking events per 100 km vs {ref:.0f} in the "
        "most fuel-efficient trips. Hard braking discards speed that fuel was spent "
        "building. Looking further ahead allows gentler, earlier deceleration."
    ),
    "hard_acc_per_100km": (
        "This trip had {val:.0f} hard acceleration events per 100 km vs {ref:.0f} in "
        "the most fuel-efficient trips. Gradual, progressive acceleration is "
        "associated with lower fuel use."
    ),
    "acc_std_g": (
        "Acceleration variability (std {val:.3f} g vs {ref:.3f} g) is higher than in "
        "the most fuel-efficient trips. Smoother, more even speed changes are "
        "associated with lower fuel use."
    ),
}


def generate_recommendations(trip: pd.Series, profile: dict,
                             flag_threshold: float = FLAG_THRESHOLD_SD) -> list:
    """Compare one trip with the efficient reference profile.

    Returns recommendation dicts sorted by estimated fuel impact (largest first).
    """
    recs = []
    for feat, info in profile.items():
        if feat.startswith("_") or not info.get("supported", False):
            continue
        if feat not in trip.index or feat not in MESSAGES:
            continue
        val = trip[feat]
        if pd.isna(val):
            continue

        dev_sd = (val - info["mean"]) / info["within_sd"]   # > 0 means worse
        if dev_sd < flag_threshold:
            continue

        impact_pct = (np.exp(info["coef_log_per_sd"] * dev_sd) - 1) * 100
        msg = MESSAGES[feat].format(
            val=val, ref=info["mean"], pct=val * 100, ref_pct=info["mean"] * 100)
        recs.append({
            "feature": feat,
            "label": FEAT_LABELS.get(feat, feat),
            "val": round(float(val), 3),
            "ref_mean": round(float(info["mean"]), 3),
            "deviation_sd": round(float(dev_sd), 2),
            "est_extra_fuel_pct": round(float(impact_pct), 1),
            "message": msg,
        })
    recs.sort(key=lambda r: r["est_extra_fuel_pct"], reverse=True)
    return recs


def _fmt(trip, key, spec):
    v = trip.get(key, np.nan)
    return format(v, spec) if pd.notna(v) else "n/a"


def format_report(trip: pd.Series, recs: list, profile: dict,
                  pattern_label: str) -> str:
    meta = profile["_meta"]
    lines = [
        "=" * 65,
        "  DRIVING EFFICIENCY REPORT",
        "=" * 65,
        "  Trip summary",
        f"    Distance      : {_fmt(trip, 'distance_km', '.1f')} km",
        f"    Duration      : {_fmt(trip, 'duration_min', '.1f')} min",
        f"    Avg speed     : {_fmt(trip, 'avg_speed_kmh', '.1f')} km/h",
        f"    Fuel use      : {_fmt(trip, 'l_per_100km', '.2f')} L/100 km (estimated)",
        f"    vs own median : {_fmt(trip, 'fuel_rel_vehicle', '.2f')}x "
        "(1.00 = this vehicle's typical trip)",
        f"    Trip pattern  : {pattern_label}",
        "-" * 65,
    ]
    if not recs:
        lines.append("  No clear inefficiencies detected against the reference.")
    else:
        lines.append(f"  {len(recs)} suggestion(s), ranked by estimated fuel impact:\n")
        for i, r in enumerate(recs, 1):
            lines.append(f"  [{i}] {r['label']}  "
                         f"(est. +{r['est_extra_fuel_pct']:.1f}% fuel vs reference)")
            for ln in textwrap.wrap(r["message"], width=60):
                lines.append(f"      {ln}")
            lines.append("")
    lines.append("=" * 65)
    lines.append(
        "  Note: estimates come from a within-vehicle model on the VED\n"
        "  dataset (associations, not proof of cause). Fuel is estimated\n"
        "  from MAF; road gradient, traffic, weather and load are not\n"
        "  modelled. Stop-go and idling are partly set by traffic."
    )
    lines.append("=" * 65)
    return "\n".join(lines)