#!/usr/bin/env python3
"""
Step 5 & 6 – Fuel-Efficient Driving Profile + Recommendation Engine
(SRS FR-07, FR-08)

Reads:
  data/processed/features_clustered.csv

Writes:
  data/processed/efficient_profile.json   – the reference profile
  reports/figures/16_efficient_profile.png
  reports/figures/17_recommendation_demo.png
  reports/efficient_profile.txt
  src/recommender.py                       – importable recommendation module
"""

import argparse
import json
import os
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)

SEED = 42
EFFICIENT_LABEL = "Efficient"

# Features used in the profile and recommendations
PROFILE_FEATS = [
    "speed_mean_kmh",
    "speed_std_kmh",
    "acc_std_g",
    "hard_acc_count",
    "hard_brk_count",
    "rpm_mean",
    "idle_frac",
    "stop_go_events",
    "acc_max_g",
]

# Human-readable labels for display
FEAT_LABELS = {
    "speed_mean_kmh"  : "Avg Speed (km/h)",
    "speed_std_kmh"   : "Speed Variability (km/h std)",
    "acc_std_g"       : "Accel. Variability (g std)",
    "hard_acc_count"  : "Hard Acceleration Events",
    "hard_brk_count"  : "Hard Braking Events",
    "rpm_mean"        : "Mean RPM",
    "idle_frac"       : "Idle Fraction",
    "stop_go_events"  : "Stop-and-Go Events",
    "acc_max_g"       : "Peak Acceleration (g)",
}

# Direction: 'lower' means lower values are more efficient
DIRECTION = {
    "speed_mean_kmh"  : "higher",   # higher steady speed → more efficient (highway effect)
    "speed_std_kmh"   : "lower",
    "acc_std_g"       : "lower",
    "hard_acc_count"  : "lower",
    "hard_brk_count"  : "lower",
    "rpm_mean"        : "higher",   # higher RPM here reflects highway (efficient cluster)
    "idle_frac"       : "lower",
    "stop_go_events"  : "lower",
    "acc_max_g"       : "lower",
}

# Thresholds: how far from the profile before flagging (in normalised units)
# 0.5 = half a population std-dev away from profile mean
FLAG_THRESHOLD = 0.5


# ── build profile ────────────────────────────────────────────────────────────
def build_profile(df: pd.DataFrame) -> dict:
    """
    Profile = mean ± std of each feature across the top-25% most
    fuel-efficient trips (lowest L/100km), regardless of cluster label.
    Using the top quartile rather than just the Efficient cluster makes
    the profile data-driven and robust to cluster boundary effects.
    """
    cutoff = df["l_per_100km"].quantile(0.25)
    efficient = df[df["l_per_100km"] <= cutoff].copy()
    print(f"  Profile built from {len(efficient):,} trips "
          f"(L/100km ≤ {cutoff:.2f}, i.e. bottom 25%)")

    profile = {}
    for feat in PROFILE_FEATS:
        if feat not in df.columns:
            continue
        vals = efficient[feat].dropna()
        profile[feat] = {
            "mean"      : float(vals.mean()),
            "std"       : float(vals.std()),
            "p25"       : float(vals.quantile(0.25)),
            "p75"       : float(vals.quantile(0.75)),
            "direction" : DIRECTION.get(feat, "lower"),
        }

    # Population std (for normalising deviations in the recommender)
    for feat in PROFILE_FEATS:
        if feat in profile:
            pop_std = df[feat].dropna().std()
            profile[feat]["pop_std"] = float(pop_std) if pop_std > 0 else 1.0

    profile["_meta"] = {
        "n_trips"          : int(len(efficient)),
        "l100km_threshold" : float(cutoff),
        "l100km_mean"      : float(efficient["l_per_100km"].mean()),
        "l100km_std"       : float(efficient["l_per_100km"].std()),
    }
    return profile


# ── plot profile ──────────────────────────────────────────────────────────────
def plot_profile_comparison(df: pd.DataFrame, profile: dict, figdir: str):
    feats = [f for f in PROFILE_FEATS if f in df.columns and f in profile]
    styles = ["Efficient", "Moderate", "Aggressive"]

    means = {s: df[df["driving_style"] == s][feats].mean() for s in styles
             if s in df["driving_style"].values}
    prof_means = pd.Series({f: profile[f]["mean"] for f in feats})

    x = np.arange(len(feats))
    width = 0.2
    colors = {"Efficient": "#6ACC65", "Moderate": "#4878CF",
              "Aggressive": "#D65F5F", "Profile (top 25%)": "#FF8C00"}

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (label, series) in enumerate(
        list(means.items()) + [("Profile (top 25%)", prof_means)]
    ):
        # Normalise by population std for comparable display
        norm = pd.Series({f: series[f] / profile[f]["pop_std"] for f in feats})
        ax.bar(x + i * width, norm, width, label=label,
               color=colors.get(label, "#aaa"), alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([FEAT_LABELS.get(f, f) for f in feats],
                       rotation=30, ha="right", fontsize=9)
    ax.set(ylabel="Normalised value (÷ population std)",
           title="Driving Style Profiles vs Fuel-Efficient Reference Profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "16_efficient_profile.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {os.path.join(figdir, '16_efficient_profile.png')}")


# ── recommendation engine ─────────────────────────────────────────────────────
def generate_recommendations(trip: pd.Series, profile: dict,
                              flag_threshold: float = FLAG_THRESHOLD) -> list[dict]:
    """
    Compare one trip's features against the efficient profile.
    Returns a list of recommendation dicts, sorted by severity.
    Each dict has: feature, deviation, severity (0–1), message.
    """
    recs = []

    MESSAGES = {
        "stop_go_events": (
            "Your trip had {val:.0f} stop-and-go events vs {ref:.0f} in efficient trips. "
            "Frequent stopping and restarting burns significantly more fuel. "
            "Where conditions allow, maintain a steady roll rather than coming to a complete stop."
        ),
        "idle_frac": (
            "Your engine idled for {pct:.0f}% of trip time vs {ref_pct:.0f}% in efficient trips. "
            "Extended idling wastes fuel with zero distance gain. "
            "If stationary for more than ~60 seconds, switching off is worth considering."
        ),
        "hard_acc_count": (
            "You recorded {val:.0f} hard acceleration events vs {ref:.0f} in efficient trips. "
            "Rapid acceleration sharply increases fuel demand. "
            "Gradual, progressive acceleration reduces fuel consumption and mechanical wear."
        ),
        "hard_brk_count": (
            "You recorded {val:.0f} hard braking events vs {ref:.0f} in efficient trips. "
            "Hard braking dissipates kinetic energy that cost fuel to build. "
            "Anticipating traffic flow further ahead allows gentler deceleration."
        ),
        "acc_std_g": (
            "Acceleration variability (std {val:.3f}g vs {ref:.3f}g) is elevated. "
            "Smoother, more consistent throttle inputs are associated with lower fuel use. "
            "Try to maintain even pressure on the accelerator and avoid surging."
        ),
        "speed_std_kmh": (
            "Speed variability is higher than the efficient profile ({val:.1f} vs {ref:.1f} km/h std). "
            "Maintaining a more consistent speed reduces the repeated acceleration cost. "
            "On open roads, cruise control or conscious speed-steadying helps."
        ),
        "speed_mean_kmh": (
            "Average trip speed ({val:.1f} km/h) is below the efficient profile ({ref:.1f} km/h). "
            "This suggests mostly low-speed urban operation with high idle/stop overhead. "
            "Where route choice is possible, roads with fewer traffic signals can improve efficiency."
        ),
        "rpm_mean": (
            "Mean RPM ({val:.0f}) is lower than in efficient trips ({ref:.0f}). "
            "In combination with low speed, this may indicate short urban trips where the engine "
            "rarely reaches its efficient operating range. "
            "Consolidating short trips where possible reduces per-km fuel cost."
        ),
        "acc_max_g": (
            "Peak acceleration ({val:.2f}g vs {ref:.2f}g in efficient trips) is high. "
            "Occasional aggressive inputs significantly raise instantaneous fuel demand. "
            "A lighter right foot during the first few seconds of acceleration makes the largest difference."
        ),
    }

    for feat, info in profile.items():
        if feat.startswith("_") or feat not in trip.index:
            continue
        val = trip[feat]
        if pd.isna(val):
            continue

        ref_mean  = info["mean"]
        pop_std   = info["pop_std"]
        direction = info["direction"]

        # Signed deviation in population-std units
        # Positive deviation = trip is WORSE than profile in that feature's direction
        if direction == "lower":
            deviation = (val - ref_mean) / pop_std   # positive = worse (too high)
        else:
            deviation = (ref_mean - val) / pop_std   # positive = worse (too low)

        if deviation < flag_threshold:
            continue   # within acceptable range

        severity = min(1.0, deviation / 2.0)   # cap at 1

        if feat not in MESSAGES:
            continue

        # Format message with actual values
        fmt_args = {"val": val, "ref": ref_mean,
                    "pct": val * 100, "ref_pct": ref_mean * 100}
        msg = MESSAGES[feat].format(**fmt_args)
        recs.append({
            "feature"  : feat,
            "label"    : FEAT_LABELS.get(feat, feat),
            "val"      : round(float(val), 3),
            "ref_mean" : round(float(ref_mean), 3),
            "deviation": round(float(deviation), 3),
            "severity" : round(float(severity), 3),
            "message"  : msg,
        })

    recs.sort(key=lambda r: r["severity"], reverse=True)
    return recs


def format_report(trip: pd.Series, recs: list[dict],
                  profile: dict, cluster_label: str) -> str:
    lines = [
        "=" * 65,
        "  DRIVING EFFICIENCY REPORT",
        "=" * 65,
        f"  Trip summary",
        f"    Distance      : {trip.get('distance_km', '?'):.1f} km",
        f"    Duration      : {trip.get('duration_min', '?'):.1f} min",
        f"    Avg speed     : {trip.get('avg_speed_kmh', '?'):.1f} km/h",
        f"    Fuel use      : {trip.get('l_per_100km', '?'):.2f} L/100 km",
        f"    Driving style : {cluster_label}",
        f"    Efficient ref : {profile['_meta']['l100km_mean']:.2f} L/100 km "
        f"(top-25% trips)",
        "-" * 65,
    ]

    if not recs:
        lines.append("  ✓ No significant inefficiencies detected.")
        lines.append("    This trip closely matches the efficient driving profile.")
    else:
        lines.append(f"  {len(recs)} recommendation(s), ranked by impact:\n")
        for i, r in enumerate(recs, 1):
            lines.append(f"  [{i}] {r['label']}  "
                         f"(severity {r['severity']:.2f})")
            for ln in textwrap.wrap(r["message"], width=60):
                lines.append(f"      {ln}")
            lines.append("")

    lines.append("=" * 65)
    lines.append(
        "  Note: Recommendations are based on patterns observed in the\n"
        "  VED dataset. Fuel consumption is also affected by vehicle\n"
        "  type, road gradient, traffic, weather, and tyre condition.\n"
        "  Results use MAF-estimated fuel (direct OBD fuel rate\n"
        "  unavailable in this dataset)."
    )
    lines.append("=" * 65)
    return "\n".join(lines)


# ── demo: run recommender on sample trips ────────────────────────────────────
def demo_recommendations(df: pd.DataFrame, profile: dict,
                         figdir: str, report_dir: str):
    # Pick one trip from each style for the demo
    demo_trips = {}
    for style in ["Aggressive", "Moderate", "Efficient"]:
        subset = df[df["driving_style"] == style]
        if subset.empty:
            continue
        # pick the median trip for that cluster (most representative)
        median_idx = (subset["l_per_100km"] - subset["l_per_100km"].median()).abs().idxmin()
        demo_trips[style] = subset.loc[median_idx]

    all_reports = []
    for style, trip in demo_trips.items():
        recs = generate_recommendations(trip, profile)
        report_text = format_report(trip, recs, profile, style)
        all_reports.append(report_text)
        print(f"\n{'─'*65}")
        print(report_text)

    # Save demo reports
    with open(os.path.join(report_dir, "demo_recommendations.txt"), "w") as fh:
        fh.write("\n\n".join(all_reports))
    print(f"\n  Saved: {report_dir}/demo_recommendations.txt")

    # ── bar chart: severity of each recommendation for the Aggressive demo ──
    agg_trip = demo_trips.get("Aggressive")
    if agg_trip is not None:
        recs = generate_recommendations(agg_trip, profile)
        if recs:
            labels  = [r["label"] for r in recs]
            sevs    = [r["severity"] for r in recs]
            colors  = ["#D65F5F" if s > 0.6 else "#FFA500" if s > 0.35
                       else "#6ACC65" for s in sevs]

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(labels[::-1], sevs[::-1], color=colors[::-1])
            ax.axvline(0.5, color="gray", linestyle="--", linewidth=1,
                       label="Moderate threshold")
            ax.set(xlabel="Severity (0 = fine, 1 = high impact)",
                   title="Recommendation Severity – Sample Aggressive Trip")
            ax.set_xlim(0, 1.05)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(figdir, "17_recommendation_demo.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  saved → {os.path.join(figdir, '17_recommendation_demo.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-file",  default="data/processed/features_clustered.csv")
    ap.add_argument("--proc-dir",   default="data/processed")
    ap.add_argument("--report-dir", default="reports")
    args = ap.parse_args()

    figdir = os.path.join(args.report_dir, "figures")
    os.makedirs(figdir, exist_ok=True)

    df = pd.read_csv(args.feat_file)
    print(f"Loaded {len(df):,} clustered trips.")

    # ── build profile ─────────────────────────────────────────────────────────
    profile = build_profile(df)
    print("\n── Efficient Driving Profile ──")
    for feat, vals in profile.items():
        if feat.startswith("_"):
            continue
        print(f"  {feat:<22} mean={vals['mean']:.3f}  std={vals['std']:.3f}  "
              f"direction={vals['direction']}")

    # ── save profile ──────────────────────────────────────────────────────────
    prof_path = os.path.join(args.proc_dir, "efficient_profile.json")
    with open(prof_path, "w") as fh:
        json.dump(profile, fh, indent=2)
    print(f"\n  Saved profile → {prof_path}")

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_profile_comparison(df, profile, figdir)

    # ── demo recommendations ──────────────────────────────────────────────────
    demo_recommendations(df, profile, figdir, args.report_dir)

    # ── text summary ─────────────────────────────────────────────────────────
    with open(os.path.join(args.report_dir, "efficient_profile.txt"), "w") as fh:
        fh.write("=== FUEL-EFFICIENT DRIVING PROFILE ===\n\n")
        fh.write(f"Built from {profile['_meta']['n_trips']} trips "
                 f"(L/100km ≤ {profile['_meta']['l100km_threshold']:.2f})\n")
        fh.write(f"Mean fuel consumption: {profile['_meta']['l100km_mean']:.2f} "
                 f"± {profile['_meta']['l100km_std']:.2f} L/100km\n\n")
        for feat, vals in profile.items():
            if feat.startswith("_"):
                continue
            fh.write(f"{feat:<25} mean={vals['mean']:.3f}  "
                     f"p25={vals['p25']:.3f}  p75={vals['p75']:.3f}  "
                     f"direction={vals['direction']}\n")
    print(f"  Saved: {args.report_dir}/efficient_profile.txt")


if __name__ == "__main__":
    main()