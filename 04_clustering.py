#!/usr/bin/env python3
"""
Step 4 – Driving Pattern Clustering  (SRS FR-06)

Reads:
  data/processed/features.csv

Writes:
  data/processed/features_clustered.csv   – features + cluster label
  reports/figures/11_elbow_silhouette.png
  reports/figures/12_cluster_radar.png
  reports/figures/13_cluster_scatter.png
  reports/figures/14_cluster_fuel_box.png
  reports/figures/15_cluster_feature_heatmap.png
  reports/cluster_profiles.txt
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import kruskal, spearmanr

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

SEED = 42

# Features selected from Step 3 correlation analysis:
# top Spearman |r| with L/100km, no severe multicollinearity,
# all 100% complete → no imputation needed for these.

# Speed-independent features only (no average speed / RPM, which mostly encode
# road type). Selected from the within-vehicle model in step 3.
CLUSTER_FEATS = [
    "stop_go_per_km",
    "idle_frac",
    "hard_acc_per_100km",
    "hard_brk_per_100km",
    "acc_std_g",
]
# Right-skewed rate features are log-transformed so a few extreme trips
# don't dominate the clusters.
LOG_FEATS = ["stop_go_per_km", "hard_acc_per_100km", "hard_brk_per_100km"]

# Names describe traffic flow, NOT fuel use.
PATTERN_NAMES = {
    2: ["Free-flowing", "Stop-and-go"],
    3: ["Free-flowing", "Mixed", "Stop-and-go"],
    4: ["Free-flowing", "Mostly flowing", "Mostly stop-and-go", "Stop-and-go"],
}

PALETTE = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── 1. choose k via elbow + silhouette ───────────────────────────────────────
def choose_k(X_scaled, figdir):
    ks = range(2, 9)
    inertias, sils = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X_scaled, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(ks, inertias, "o-", color="#4878CF")
    ax1.set(xlabel="k", ylabel="Inertia (within-cluster SSE)",
            title="Elbow Method")
    ax2.plot(ks, sils, "o-", color="#D65F5F")
    ax2.set(xlabel="k", ylabel="Silhouette Score",
            title="Silhouette Score")
    fig.tight_layout()
    save(fig, os.path.join(figdir, "11_elbow_silhouette.png"))

    best_k = int(ks[int(np.argmax(sils))])
    print(f"\n  Best silhouette score: {max(sils):.3f} at k={best_k}")
    print("  Silhouette scores by k:", {k: round(s, 3) for k, s in zip(ks, sils)})
    return best_k, sils


# ── 2. fit final model ───────────────────────────────────────────────────────
def fit_kmeans(X_scaled, k):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=50, max_iter=500)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"  Final model: k={k}, silhouette={sil:.3f}, inertia={km.inertia_:.1f}")
    return km, labels


# ── 3. Cluster Labelling ──────────────────────────────
def prepare_matrix(df, feats):
    X = df[feats].copy()
    for c in LOG_FEATS:
        if c in X.columns:
            X[c] = np.log1p(X[c])
    return X


def label_by_pattern(df, k):
    """Rank clusters by mean z-score of stop-go rate and idle fraction
    (smoothest first) and name them from that ranking, not from fuel."""
    cols = ["stop_go_per_km", "idle_frac"]
    z = (df[cols] - df[cols].mean()) / df[cols].std()
    score = z.mean(axis=1).groupby(df["cluster"]).mean().sort_values()
    names = PATTERN_NAMES.get(k, [f"Pattern {i+1}" for i in range(k)])
    cluster_map = {int(c): names[i] for i, c in enumerate(score.index)}
    rank_map = {int(c): i for i, c in enumerate(score.index)}   # 0 = smoothest
    return cluster_map, rank_map


def validate_clusters(df):
    """Do clusters differ in fuel use relative to each vehicle's own norm?"""
    lines = ["\n=== CLUSTER VALIDATION ==="]
    sub = df.dropna(subset=["fuel_rel_vehicle"])
    groups = [g["fuel_rel_vehicle"].values for _, g in sub.groupby("cluster")]
    H, p = kruskal(*groups)
    lines.append(f"Kruskal-Wallis on fuel_rel_vehicle across clusters: H={H:.1f}, p={p:.2e}")
    rho, p2 = spearmanr(sub["style_rank"], sub["fuel_rel_vehicle"])
    lines.append(f"Spearman (style_rank vs fuel_rel_vehicle): rho={rho:.3f}, p={p2:.2e}")
    big = df.groupby("veh_id").filter(lambda g: len(g) >= 5)
    share = big.groupby("veh_id")["cluster"].agg(
        lambda s: s.value_counts(normalize=True).iloc[0]).mean()
    lines.append(f"Mean share of a vehicle's trips in its most common cluster: {share:.2f} "
                 "(near 1.0 would mean clusters mostly identify vehicles)")
    # Robustness: trips from one vehicle are not independent, so also compare
    # the smoothest and the most stop-and-go pattern WITHIN each vehicle.
    from scipy.stats import wilcoxon
    top = sub["style_rank"].max()
    diffs = []
    for veh, g in sub.groupby("veh_id"):
        a = g.loc[g["style_rank"] == 0, "fuel_rel_vehicle"]
        b = g.loc[g["style_rank"] == top, "fuel_rel_vehicle"]
        if len(a) >= 2 and len(b) >= 2:
            diffs.append(b.median() - a.median())
    diffs = np.array(diffs)
    if len(diffs) >= 10:
        _, pw = wilcoxon(diffs)
        lines.append(
            f"Within-vehicle check ({len(diffs)} vehicles with >=2 trips in both "
            f"the smoothest and the most stop-and-go cluster): median difference in "
            f"relative fuel = {np.median(diffs):+.3f}; "
            f"{(diffs > 0).mean():.0%} of vehicles use more fuel in the stop-and-go "
            f"pattern; Wilcoxon p={pw:.3g}")
    else:
        lines.append(f"Within-vehicle check: only {len(diffs)} vehicles qualify, too few to test.")
    return lines


# ── 4. radar / spider chart ──────────────────────────────────────────────────
def plot_radar(profiles, cluster_map, figdir):
    feats = [c for c in CLUSTER_FEATS if c in profiles.columns]
    # Normalise each feature to [0,1] for display
    norm = (profiles[feats] - profiles[feats].min()) / \
           (profiles[feats].max() - profiles[feats].min() + 1e-9)

    angles = np.linspace(0, 2 * np.pi, len(feats), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for idx, row in norm.iterrows():
        vals = row[feats].tolist() + [row[feats[0]]]
        name = cluster_map.get(int(idx), f"Cluster {idx}")
        color = PALETTE[int(idx) % len(PALETTE)]
        ax.plot(angles, vals, "o-", linewidth=2, label=name, color=color)
        ax.fill(angles, vals, alpha=0.10, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]),
                      [f.replace("_", "\n") for f in feats], size=9)
    ax.set_title("Driving Style Cluster Profiles\n(normalised feature means)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
    fig.tight_layout()
    save(fig, os.path.join(figdir, "12_cluster_radar.png"))


# ── 5. PCA scatter ───────────────────────────────────────────────────────────
def plot_pca_scatter(X_scaled, labels, cluster_map, figdir):
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(X_scaled)
    var = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    for cid, name in cluster_map.items():
        mask = labels == cid
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   alpha=0.45, s=20, label=name,
                   color=PALETTE[cid % len(PALETTE)])
    ax.set(xlabel=f"PC1 ({var[0]:.1f}% var)",
           ylabel=f"PC2 ({var[1]:.1f}% var)",
           title="K-Means Clusters – PCA Projection")
    ax.legend()
    fig.tight_layout()
    save(fig, os.path.join(figdir, "13_cluster_scatter.png"))

# ── 6. fuel box per cluster ──────────────────────────────────────────────────
def plot_fuel_box(df, cluster_map, figdir):
    df2 = df.dropna(subset=["fuel_rel_vehicle"]).copy()
    df2["Style"] = df2["cluster"].map(cluster_map)
    order = df2.groupby("Style")["fuel_rel_vehicle"].median().sort_values().index

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {name: PALETTE[cid % len(PALETTE)] for cid, name in cluster_map.items()}
    sns.boxplot(data=df2, x="Style", y="fuel_rel_vehicle",
                order=order, palette=colors, ax=ax)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set(xlabel="Trip pattern",
           ylabel="Fuel relative to the vehicle's own median",
           title="Relative Fuel Consumption by Trip Pattern")
    fig.tight_layout()
    save(fig, os.path.join(figdir, "14_cluster_fuel_box.png"))

# ── 7. feature heatmap ───────────────────────────────────────────────────────
def plot_feature_heatmap(profiles, cluster_map, figdir):
    feats = [c for c in CLUSTER_FEATS + ["l_per_100km", "duration_min",
                                          "distance_km", "avg_speed_kmh"]
             if c in profiles.columns]
    display = profiles[feats].copy()
    display.index = [cluster_map.get(i, f"C{i}") for i in display.index]

    # Z-score across clusters for each feature (highlights relative differences)
    z = (display - display.mean()) / (display.std() + 1e-9)

    fig, ax = plt.subplots(figsize=(12, max(3, len(display) * 1.2)))
    sns.heatmap(z, annot=display.round(2), fmt=".2f", cmap="RdYlGn_r",
                center=0, linewidths=0.5, ax=ax,
                annot_kws={"size": 9})
    ax.set_title("Cluster Profile Heatmap\n(cell = mean value; colour = z-score relative to clusters)")
    ax.set_ylabel("")
    fig.tight_layout()
    save(fig, os.path.join(figdir, "15_cluster_feature_heatmap.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-file",  default="data/processed/features.csv")
    ap.add_argument("--proc-dir",   default="data/processed")
    ap.add_argument("--report-dir", default="reports")
    ap.add_argument("--k", type=int, default=0,
                    help="Force k (0 = auto-select by silhouette)")
    args = ap.parse_args()

    figdir = os.path.join(args.report_dir, "figures")
    os.makedirs(figdir, exist_ok=True)

    df = pd.read_csv(args.feat_file)
    print(f"Loaded {len(df):,} trips.")

    # ── prep ─────────────────────────────────────────────────────────────────
    feats_present = [c for c in CLUSTER_FEATS if c in df.columns]
    Xdf = prepare_matrix(df, feats_present)
    ok = Xdf.notna().all(axis=1)
    if not ok.all():
        print(f"  Dropping {int((~ok).sum())} trips with missing cluster features")
        df = df[ok].reset_index(drop=True)
        Xdf = Xdf[ok].reset_index(drop=True)
    X_scaled = StandardScaler().fit_transform(Xdf.values)

    # ── choose k ─────────────────────────────────────────────────────────────
    best_k, sil_scores = choose_k(X_scaled, figdir)
    k = args.k if args.k >= 2 else best_k
    print(f"\n  Using k={k}")

    # ── fit ───────────────────────────────────────────────────────────────────
    km, labels = fit_kmeans(X_scaled, k)
    df["cluster"] = labels

    # ── label clusters ───────────────────────────────────────────────────────
    cluster_map, rank_map = label_by_pattern(df, k)
    df["driving_style"] = df["cluster"].map(cluster_map)
    df["style_rank"] = df["cluster"].map(rank_map)   # 0 = smoothest pattern

    # ── profiles (mean of each feature per cluster) ───────────────────────────
    profile_cols = feats_present + ["l_per_100km", "fuel_rel_vehicle","duration_min",
                                    "distance_km", "avg_speed_kmh"]
    profile_cols = [c for c in profile_cols if c in df.columns]
    profiles = df.groupby("cluster")[profile_cols].mean()

    print("\n── Cluster Profiles (means) ──")
    print(profiles.round(2).to_string())

    # ── sizes ─────────────────────────────────────────────────────────────────
    sizes = df.groupby("driving_style").agg(
        trips=("l_per_100km", "count"),
        median_l100=("l_per_100km", "median"),
        median_rel_fuel=("fuel_rel_vehicle", "median"),
    ).round(3)
    print("\n── Cluster Sizes & Fuel ──")
    print(sizes.to_string())
    val_lines = validate_clusters(df)
    print("\n".join(val_lines))

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_radar(profiles, cluster_map, figdir)
    plot_pca_scatter(X_scaled, labels, cluster_map, figdir)
    plot_fuel_box(df, cluster_map, figdir)
    plot_feature_heatmap(profiles, cluster_map, figdir)

    # ── save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(args.proc_dir, "features_clustered.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    report_lines = [
        "=== CLUSTER PROFILES ===",
        f"k = {k}",
        f"Cluster → Style: {cluster_map}",
        "\nMean feature values per cluster:",
        profiles.round(3).to_string(),
        "\nCluster sizes & fuel:",
        sizes.to_string(),
    ] + val_lines
    with open(os.path.join(args.report_dir, "cluster_profiles.txt"), "w") as fh:
        fh.write("\n".join(report_lines) + "\n")
    print(f"Report: {args.report_dir}/cluster_profiles.txt")


if __name__ == "__main__":
    main()