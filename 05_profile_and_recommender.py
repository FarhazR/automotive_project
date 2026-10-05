#!/usr/bin/env python3
"""
Step 5 & 6 - Fuel-efficient reference profile + recommendation engine
(SRS FR-07, FR-08)

Reads:
  data/processed/features_clustered.csv

Writes:
  data/processed/efficient_profile.json
  reports/figures/16_efficient_profile.png
  reports/figures/17_recommendation_demo.png
  reports/efficient_profile.txt
  reports/demo_recommendations.txt

Method
  1. Fit the within-vehicle model (log fuel relative to each vehicle's own
     median) to estimate how much each behaviour feature is associated with
     fuel use, controlling for average speed. Features whose 95% CI is not
     clearly positive are NOT recommended.
  2. Reference profile = trips in the bottom 25% of fuel_rel_vehicle
     (efficient for their own vehicle), so vehicle type does not leak in.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

sys.path.insert(0, "src")
from recommender import (FEAT_LABELS, format_report,  # noqa: E402
                         generate_recommendations)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)

EFFICIENT_QUANTILE = 0.25
# Behaviour features the data may support; the model decides which are used.
CANDIDATE_FEATS = ["stop_go_per_km", "idle_frac", "hard_brk_per_100km",
                   "hard_acc_per_100km", "acc_std_g"]
# Controlled for in the model, never recommended (mostly reflects road type).
CONTROL_FEATS = ["speed_mean_kmh"]


# -- 1. effect sizes from the within-vehicle model ---------------------------
def estimate_effects(df: pd.DataFrame) -> dict:
    cols = CANDIDATE_FEATS + CONTROL_FEATS
    sub = df.dropna(subset=["fuel_rel_vehicle"] + cols).copy()
    veh = sub["veh_id"]

    y = np.log(sub["fuel_rel_vehicle"])
    y = y - y.groupby(veh).transform("mean")
    Xw = sub[cols] - sub.groupby("veh_id")[cols].transform("mean")
    within_sd = Xw.std()
    X = Xw / within_sd

    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": veh})
    ci = res.conf_int()
    print(f"  Within-vehicle model: {len(sub):,} trips, "
          f"{sub['veh_id'].nunique()} vehicles, R2={res.rsquared:.3f}")

    effects = {}
    for c in CANDIDATE_FEATS:
        effects[c] = {
            "coef_log_per_sd": float(res.params[c]),
            "pct_per_sd":      float((np.exp(res.params[c]) - 1) * 100),
            "ci_low_pct":      float((np.exp(ci.loc[c, 0]) - 1) * 100),
            "ci_high_pct":     float((np.exp(ci.loc[c, 1]) - 1) * 100),
            "p_value":         float(res.pvalues[c]),
            "within_sd":       float(within_sd[c]),
            "supported":       bool(ci.loc[c, 0] > 0),
        }
    return effects


# -- 2. reference profile ----------------------------------------------------
def build_profile(df: pd.DataFrame, effects: dict) -> dict:
    rel = df["fuel_rel_vehicle"].dropna()
    cutoff = float(rel.quantile(EFFICIENT_QUANTILE))
    efficient = df[df["fuel_rel_vehicle"] <= cutoff]
    print(f"  Reference profile: {len(efficient):,} trips from "
          f"{efficient['veh_id'].nunique()} vehicles "
          f"(relative fuel <= {cutoff:.3f}, bottom 25%)")

    profile = {}
    for feat in CANDIDATE_FEATS:
        vals = efficient[feat].dropna()
        pop_std = float(df[feat].dropna().std())
        profile[feat] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "p25": float(vals.quantile(0.25)),
            "p75": float(vals.quantile(0.75)),
            "pop_std": pop_std if pop_std > 0 else 1.0,
            "direction": "lower",
            **effects[feat],
        }
    profile["_meta"] = {
        "n_trips": int(len(efficient)),
        "n_vehicles": int(efficient["veh_id"].nunique()),
        "rel_fuel_threshold": cutoff,
        "rel_fuel_mean": float(efficient["fuel_rel_vehicle"].mean()),
        "recommended_features": [f for f in CANDIDATE_FEATS
                                 if effects[f]["supported"]],
    }
    return profile


# -- 3. plots -----------------------------------------------------------------
def pattern_order(df):
    return (df.dropna(subset=["style_rank"])
              .sort_values("style_rank")["driving_style"].unique().tolist())


def plot_profile_comparison(df, profile, figdir):
    feats = [f for f in CANDIDATE_FEATS if f in profile]
    means = {s: df[df["driving_style"] == s][feats].mean()
             for s in pattern_order(df)}
    means["Efficient reference\n(bottom 25% relative fuel)"] = pd.Series(
        {f: profile[f]["mean"] for f in feats})

    colors = ["#6ACC65", "#D65F5F", "#4878CF", "#B47CC7", "#FF8C00"]
    x = np.arange(len(feats))
    width = 0.8 / len(means)

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (label, series) in enumerate(means.items()):
        norm = pd.Series({f: series[f] / profile[f]["pop_std"] for f in feats})
        ax.bar(x + i * width, norm, width, label=label,
               color=colors[i % len(colors)], alpha=0.85)
    ax.set_xticks(x + width * (len(means) - 1) / 2)
    ax.set_xticklabels([FEAT_LABELS.get(f, f) for f in feats],
                       rotation=25, ha="right", fontsize=9)
    ax.set(ylabel="Normalised value (/ population std)",
           title="Trip Patterns vs Fuel-Efficient Reference Profile")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(figdir, "16_efficient_profile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


# -- 4. demo ------------------------------------------------------------------
def demo_recommendations(df, profile, figdir, report_dir):
    styles = pattern_order(df)
    demo = {}
    for style in styles:
        sub = df[df["driving_style"] == style].dropna(subset=["fuel_rel_vehicle"])
        if sub.empty:
            continue
        idx = (sub["fuel_rel_vehicle"]
               - sub["fuel_rel_vehicle"].median()).abs().idxmin()
        demo[style] = sub.loc[idx]

    reports = []
    for style, trip in demo.items():
        recs = generate_recommendations(trip, profile)
        text = format_report(trip, recs, profile, style)
        reports.append(text)
        print(f"\n{'-' * 65}\n{text}")

    path = os.path.join(report_dir, "demo_recommendations.txt")
    with open(path, "w") as fh:
        fh.write("\n\n".join(reports))
    print(f"\n  Saved: {path}")

    if not styles:
        return
    last = demo.get(styles[-1])
    if last is None:
        return
    recs = generate_recommendations(last, profile)
    if not recs:
        return
    labels = [r["label"] for r in recs][::-1]
    impacts = [r["est_extra_fuel_pct"] for r in recs][::-1]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(labels, impacts, color="#D65F5F")
    ax.set(xlabel="Estimated extra fuel vs efficient reference (%)",
           title=f"Estimated Impact - Sample '{styles[-1]}' Trip")
    fig.tight_layout()
    path = os.path.join(figdir, "17_recommendation_demo.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-file", default="data/processed/features_clustered.csv")
    ap.add_argument("--proc-dir", default="data/processed")
    ap.add_argument("--report-dir", default="reports")
    args = ap.parse_args()

    figdir = os.path.join(args.report_dir, "figures")
    os.makedirs(figdir, exist_ok=True)

    df = pd.read_csv(args.feat_file)
    print(f"Loaded {len(df):,} clustered trips.")

    effects = estimate_effects(df)
    profile = build_profile(df, effects)

    print("\n-- Estimated effect per +1 within-vehicle SD (95% CI) --")
    for feat, e in effects.items():
        flag = "recommended" if e["supported"] else "NOT recommended (CI includes 0)"
        print(f"  {feat:<22} {e['pct_per_sd']:+5.1f}%  "
              f"[{e['ci_low_pct']:+5.1f}, {e['ci_high_pct']:+5.1f}]  {flag}")

    prof_path = os.path.join(args.proc_dir, "efficient_profile.json")
    with open(prof_path, "w") as fh:
        json.dump(profile, fh, indent=2)
    print(f"\n  Saved profile -> {prof_path}")

    plot_profile_comparison(df, profile, figdir)
    demo_recommendations(df, profile, figdir, args.report_dir)

    meta = profile["_meta"]
    with open(os.path.join(args.report_dir, "efficient_profile.txt"), "w") as fh:
        fh.write("=== FUEL-EFFICIENT REFERENCE PROFILE ===\n\n")
        fh.write(f"Built from {meta['n_trips']} trips of {meta['n_vehicles']} vehicles "
                 f"(relative fuel <= {meta['rel_fuel_threshold']:.3f})\n")
        fh.write(f"Recommended features: {meta['recommended_features']}\n\n")
        for feat in CANDIDATE_FEATS:
            p = profile[feat]
            fh.write(f"{feat:<22} ref_mean={p['mean']:.3f}  p25={p['p25']:.3f}  "
                     f"p75={p['p75']:.3f}  effect/SD={p['pct_per_sd']:+.1f}% "
                     f"[{p['ci_low_pct']:+.1f}, {p['ci_high_pct']:+.1f}]  "
                     f"supported={p['supported']}\n")
    print(f"  Saved: {args.report_dir}/efficient_profile.txt")


if __name__ == "__main__":
    main()