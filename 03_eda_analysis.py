#!/usr/bin/env python3
"""
Step 3 – EDA, Correlation & Regression Analysis  (SRS FR-05)

Reads:
  data/processed/features.csv

Writes:
  reports/eda_summary.txt
  reports/figures/
      01_l100km_distribution.png
      02_speed_vs_fuel.png
      03_acc_std_vs_fuel.png
      04_rpm_mean_vs_fuel.png
      05_idle_frac_vs_fuel.png
      06_hard_events_vs_fuel.png
      07_correlation_heatmap.png
      08_fuel_by_vehicle_class.png
      09_pairplot_key_features.png
      10_regression_actual_vs_pred.png
  reports/regression_summary.txt
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ── feature groups ──────────────────────────────────────────────────────────
SPEED_FEATS  = ["speed_mean_kmh", "speed_std_kmh", "speed_max_kmh",
                "pct_time_over_80", "pct_time_over_110"]
ACC_FEATS    = ["acc_mean_g", "acc_std_g", "acc_max_g", "dec_max_g",
                "hard_acc_per_100km", "hard_brk_per_100km",
                "pct_accel_time", "pct_decel_time", "pct_cruise_time"]
RPM_FEATS    = ["rpm_mean", "rpm_std", "rpm_max", "pct_time_high_rpm"]
LOAD_FEATS   = ["load_mean_pct", "load_std_pct"]
BEHAV_FEATS  = ["idle_frac", "stop_go_per_km"]
TARGET       = "l_per_100km"

ALL_NUM_FEATS = SPEED_FEATS + ACC_FEATS + RPM_FEATS + LOAD_FEATS + BEHAV_FEATS

# Features to highlight in pairplot / regression (interpretable, complete)
KEY_FEATS = ["speed_mean_kmh", "speed_std_kmh", "acc_std_g",
             "hard_acc_per_100km", "hard_brk_per_100km",
             "rpm_mean", "idle_frac", "stop_go_per_km"]


def log(msg, lines):
    print(msg)
    lines.append(str(msg))


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── 1. distribution of fuel consumption ─────────────────────────────────────
def plot_fuel_dist(df, figdir, lines):
    log("\n[1] Fuel consumption distribution", lines)
    log(df[TARGET].describe().round(2).to_string(), lines)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df[TARGET], bins=40, edgecolor="white", color="#4878CF")
    axes[0].set(xlabel="L / 100 km", ylabel="Trip count",
                title="Distribution of Fuel Consumption")
    axes[0].axvline(df[TARGET].mean(), color="red", linestyle="--", label=f"Mean {df[TARGET].mean():.1f}")
    axes[0].legend()

    stats.probplot(df[TARGET], plot=axes[1])
    axes[1].set_title("Q-Q Plot (Normality Check)")
    fig.tight_layout()
    save(fig, os.path.join(figdir, "01_l100km_distribution.png"))


# ── 2. scatter plots: key individual features vs fuel ───────────────────────
def plot_scatter_pairs(df, figdir):
    pairs = [
        ("speed_mean_kmh",  "Average Speed (km/h)",   "02_speed_vs_fuel.png"),
        ("acc_std_g",       "Accel. Std Dev (g)",      "03_acc_std_vs_fuel.png"),
        ("rpm_mean",        "Mean RPM",                "04_rpm_mean_vs_fuel.png"),
        ("idle_frac",       "Idle Fraction",           "05_idle_frac_vs_fuel.png"),
    ]
    for feat, xlabel, fname in pairs:
        if feat not in df.columns:
            continue
        sub = df[[feat, TARGET]].dropna()
        r, p = stats.pearsonr(sub[feat], sub[TARGET])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(sub[feat], sub[TARGET], alpha=0.35, s=18, color="#4878CF")
        m, b = np.polyfit(sub[feat], sub[TARGET], 1)
        xs = np.linspace(sub[feat].min(), sub[feat].max(), 200)
        ax.plot(xs, m * xs + b, color="red", linewidth=1.5)
        ax.set(xlabel=xlabel, ylabel="L / 100 km",
               title=f"{xlabel} vs Fuel Consumption\n"
                     f"Pearson r = {r:.3f}  (p {'< 0.001' if p < 0.001 else f'= {p:.3f}'})")
        save(fig, os.path.join(figdir, fname))


# ── 3. hard events (count) vs fuel ──────────────────────────────────────────
def plot_hard_events(df, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, label in zip(
        axes,
        ["hard_acc_per_100km", "hard_brk_per_100km"],
        ["Hard Acceleration Events (per 100 km)", "Hard Braking Events (per 100 km)"],
    ):
        sub = df[[col, TARGET]].dropna()
        r, p = stats.pearsonr(sub[col], sub[TARGET])
        ax.scatter(sub[col], sub[TARGET], alpha=0.3, s=15, color="#6acc65")
        m, b = np.polyfit(sub[col], sub[TARGET], 1)
        xs = np.linspace(sub[col].min(), sub[col].max(), 200)
        ax.plot(xs, m * xs + b, color="red", linewidth=1.5)
        ax.set(xlabel=label, ylabel="L / 100 km",
               title=f"{label}\nr = {r:.3f}  (p {'< 0.001' if p < 0.001 else f'= {p:.3f}'})")
    fig.tight_layout()
    save(fig, os.path.join(figdir, "06_hard_events_vs_fuel.png"))


# ── 4. correlation heatmap ───────────────────────────────────────────────────
def plot_corr_heatmap(df, figdir, lines):
    log("\n[4] Pearson correlation with L/100km (top features)", lines)
    num = df[ALL_NUM_FEATS + [TARGET]].copy()
    corr = num.corr()
    target_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    log(target_corr.round(3).to_string(), lines)

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.4, ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix (including L/100km)")
    fig.tight_layout()
    save(fig, os.path.join(figdir, "07_correlation_heatmap.png"))
    return target_corr


# ── 5. fuel by vehicle class ─────────────────────────────────────────────────
def plot_by_class(df, figdir, lines):
    if "Vehicle Class" not in df.columns:
        return
    log("\n[5] L/100km by Vehicle Class", lines)
    order = df.groupby("Vehicle Class")[TARGET].median().sort_values().index
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="Vehicle Class", y=TARGET, order=order,
                palette="Set2", ax=ax)
    ax.set(xlabel="Vehicle Class", ylabel="L / 100 km",
           title="Fuel Consumption by Vehicle Class")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    save(fig, os.path.join(figdir, "08_fuel_by_vehicle_class.png"))
    log(df.groupby("Vehicle Class")[TARGET].describe().round(2).to_string(), lines)


# ── 6. pairplot of key features ──────────────────────────────────────────────
def plot_pairplot(df, figdir):
    cols = [c for c in KEY_FEATS + [TARGET] if c in df.columns]
    sub = df[cols].dropna()
    if len(sub) < 10:
        return
    g = sns.pairplot(sub, y_vars=[TARGET], x_vars=[c for c in cols if c != TARGET],
                     plot_kws={"alpha": 0.3, "s": 12}, height=3)
    g.figure.suptitle("Key Features vs Fuel Consumption", y=1.02)
    g.figure.savefig(os.path.join(figdir, "09_pairplot_key_features.png"),
                     dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved → {os.path.join(figdir, '09_pairplot_key_features.png')}")


# ── 7. Spearman correlations (non-parametric) ────────────────────────────────
def spearman_table(df, lines):
    log("\n[6] Spearman rank correlation with L/100km", lines)
    results = []
    for col in ALL_NUM_FEATS:
        sub = df[[col, TARGET]].dropna()
        if len(sub) < 30:
            continue
        r, p = stats.spearmanr(sub[col], sub[TARGET])
        results.append({"feature": col, "spearman_r": round(r, 3),
                        "p_value": round(p, 4),
                        "significant": "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))})
    tbl = pd.DataFrame(results).sort_values("spearman_r", key=abs, ascending=False)
    log(tbl.to_string(index=False), lines)
    return tbl


# ── 8. linear regression (interpretable baseline) ───────────────────────────
def linear_regression(df, lines):
    log("\n[7] Linear Regression – interpretable baseline", lines)
    feats = [c for c in KEY_FEATS if c in df.columns]
    sub = df[feats + [TARGET]].dropna()
    log(f"  Samples used: {len(sub):,}", lines)

    pipe = Pipeline([
        ("imp",   SimpleImputer(strategy="median")),
        ("scal",  StandardScaler()),
        ("reg",   LinearRegression()),
    ])

    # Grouped CV: all trips from a vehicle stay in the same fold, so the
    # model is always tested on vehicles it has never seen.
    groups = df.loc[sub.index, "veh_id"]
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    cv_r2 = cross_val_score(pipe, sub[feats], sub[TARGET], cv=cv,
                            groups=groups, scoring="r2")
    cv_mae = cross_val_score(pipe, sub[feats], sub[TARGET], cv=cv,
                             groups=groups, scoring="neg_mean_absolute_error")
    log(f"  Grouped-by-vehicle CV  R²  : {cv_r2.mean():.3f} ± {cv_r2.std():.3f}", lines)
    log(f"  Grouped-by-vehicle CV  MAE : {-cv_mae.mean():.3f} ± {cv_mae.std():.3f} L/100km", lines)

    pipe.fit(sub[feats], sub[TARGET])
    coefs = dict(zip(feats, pipe.named_steps["reg"].coef_))
    log("  Standardised coefficients:", lines)
    for k, v in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
        log(f"    {k:<25} {v:+.4f}", lines)
    return pipe, sub, feats


# ── 9. gradient boosting (feature importance) ────────────────────────────────
def gbm_importance(df, figdir, lines):
    log("\n[8] Gradient Boosting – feature importance", lines)
    feats = [c for c in ALL_NUM_FEATS if c in df.columns]

    # Only drop rows with a missing target. Missing feature values (e.g. engine
    # load, ~19% NaN) are handled by the imputer instead of silently removing trips.
    sub = df[feats + [TARGET]].dropna(subset=[TARGET])
    log(f"  Samples used: {len(sub):,}", lines)

    pipe = Pipeline([
        ("imp",  SimpleImputer(strategy="median")),
        ("gbm",  GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                           learning_rate=0.05,
                                           subsample=0.8, random_state=42)),
    ])

    # Grouped CV: the model is always tested on vehicles it has not seen.
    groups = df.loc[sub.index, "veh_id"]
    cv = GroupKFold(n_splits=min(5, groups.nunique()))
    cv_r2 = cross_val_score(pipe, sub[feats], sub[TARGET], cv=cv,
                            groups=groups, scoring="r2")
    log(f"  Grouped-by-vehicle CV  R²  : {cv_r2.mean():.3f} ± {cv_r2.std():.3f}", lines)

    pipe.fit(sub[feats], sub[TARGET])
    imp = pd.Series(pipe.named_steps["gbm"].feature_importances_, index=feats)
    imp = imp.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    imp.plot.barh(ax=ax, color="#4878CF")
    ax.set(title="Gradient Boosting – Feature Importances",
           xlabel="Relative Importance")
    fig.tight_layout()
    save(fig, os.path.join(figdir, "10b_gbm_feature_importance.png"))

    log("  Feature importances (top 10):", lines)
    log(imp.sort_values(ascending=False).head(10).round(4).to_string(), lines)
    return imp


# ── 10. regression actual vs predicted ───────────────────────────────────────
def plot_actual_vs_pred(pipe, sub, feats, figdir):
    pred = pipe.predict(sub[feats])
    r2 = 1 - np.sum((sub[TARGET] - pred) ** 2) / np.sum((sub[TARGET] - sub[TARGET].mean()) ** 2)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sub[TARGET], pred, alpha=0.35, s=15, color="#4878CF")
    lo, hi = sub[TARGET].min(), sub[TARGET].max()
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
    ax.set(xlabel="Actual L/100km", ylabel="Predicted L/100km",
           title=f"Linear Regression – Actual vs Predicted\nTrain R² = {r2:.3f}")
    ax.legend()
    fig.tight_layout()
    save(fig, os.path.join(figdir, "10_regression_actual_vs_pred.png"))

WITHIN_BEHAV = ["stop_go_per_km", "idle_frac", "hard_acc_per_100km",
                "hard_brk_per_100km", "acc_std_g"]
WITHIN_CTRL  = ["speed_mean_kmh"]


# ── 11. within-vehicle (fixed-effects) model ────────────────────────────────
def within_vehicle_model(df, lines):
    log("\n[9] Within-vehicle model: log(fuel relative to the vehicle's median)", lines)
    all_cols = WITHIN_BEHAV + WITHIN_CTRL
    sub = df.dropna(subset=["fuel_rel_vehicle"] + all_cols).copy()
    sub["log_distance_km"] = np.log(sub["distance_km"])
    log(f"  Trips: {len(sub):,} from {sub['veh_id'].nunique()} vehicles "
        f"(>= 5 trips each)", lines)

    veh = sub["veh_id"]
    y = np.log(sub["fuel_rel_vehicle"])
    y = y - y.groupby(veh).transform("mean")          # remove vehicle effect

    specs = {
        "behaviour only": WITHIN_BEHAV,
        "behaviour + average-speed control": all_cols,
        "behaviour + speed + log-distance controls": all_cols + ["log_distance_km"],
    }
    for name, cols in specs.items():
        X = sub[cols] - sub.groupby("veh_id")[cols].transform("mean")
        X = X / X.std()                               # effect per +1 within-vehicle SD
        res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": veh})
        ci = res.conf_int()
        log(f"\n  Spec: {name}   (within-vehicle R² = {res.rsquared:.3f})", lines)
        log("    % change in fuel per +1 SD (95% CI), cluster-robust p", lines)
        for c in cols:
            pct = (np.exp(res.params[c]) - 1) * 100
            lo = (np.exp(ci.loc[c, 0]) - 1) * 100
            hi = (np.exp(ci.loc[c, 1]) - 1) * 100
            log(f"    {c:<22} {pct:+6.1f}%  [{lo:+5.1f}, {hi:+5.1f}]  "
                f"p={res.pvalues[c]:.3f}", lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-file",  default="data/processed/features.csv")
    ap.add_argument("--report-dir", default="reports")
    args = ap.parse_args()

    figdir = os.path.join(args.report_dir, "figures")
    os.makedirs(figdir, exist_ok=True)

    df = pd.read_csv(args.feat_file)
    print(f"Loaded {len(df):,} trips with {len(df.columns)} columns.")

    lines = ["=== EDA & CORRELATION REPORT (Step 3) ===", f"Trips: {len(df):,}"]

    plot_fuel_dist(df, figdir, lines)
    plot_scatter_pairs(df, figdir)
    plot_hard_events(df, figdir)
    target_corr = plot_corr_heatmap(df, figdir, lines)
    plot_by_class(df, figdir, lines)
    plot_pairplot(df, figdir)
    spearman_tbl = spearman_table(df, lines)
    pipe, sub, feats = linear_regression(df, lines)
    plot_actual_vs_pred(pipe, sub, feats, figdir)
    gbm_imp = gbm_importance(df, figdir, lines)
    within_vehicle_model(df, lines)

    with open(os.path.join(args.report_dir, "eda_summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nAll figures → {figdir}/")
    print(f"Report      → {args.report_dir}/eda_summary.txt")


if __name__ == "__main__":
    main()