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

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

SEED = 42

# Features selected from Step 3 correlation analysis:
# top Spearman |r| with L/100km, no severe multicollinearity,
# all 100% complete → no imputation needed for these.
CLUSTER_FEATS = [
    "speed_mean_kmh",    # strongest negative correlation
    "rpm_mean",          # second strongest negative
    "stop_go_events",    # strongest positive
    "idle_frac",         # second strongest positive
    "acc_std_g",         # acceleration aggressiveness
    "hard_acc_count",    # event-based aggressiveness
    "hard_brk_count",    # braking aggressiveness
    "acc_max_g",         # peak aggressiveness
]

CLUSTER_NAMES = {
    # filled in after profiling; placeholders overwritten by auto-labelling below
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2",
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


# ── 3. auto-label clusters by fuel consumption ──────────────────────────────
def auto_label(df, k):
    """
    Rank clusters by median L/100km.
    Lowest  → 'Efficient'
    Highest → 'Aggressive' (or 'Heavy')
    Middle  → 'Moderate'
    Returns a dict {cluster_id: label}
    """
    medians = df.groupby("cluster")["l_per_100km"].median().sort_values()
    labels_ordered = ["Efficient", "Moderate", "Aggressive"] if k == 3 else \
                     [f"Style {i+1}" for i in range(k)]
    # pad if k != 3
    while len(labels_ordered) < k:
        labels_ordered.insert(-1, f"Moderate-{len(labels_ordered)-1}")
    return {int(cid): labels_ordered[i] for i, cid in enumerate(medians.index)}


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
    df2 = df.copy()
    df2["Style"] = df2["cluster"].map(cluster_map)
    order = df2.groupby("Style")["l_per_100km"].median().sort_values().index

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {name: PALETTE[cid % len(PALETTE)] for cid, name in cluster_map.items()}
    sns.boxplot(data=df2, x="Style", y="l_per_100km",
                order=order, palette=colors, ax=ax)
    ax.set(xlabel="Driving Style", ylabel="L / 100 km",
           title="Fuel Consumption by Driving Style Cluster")
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
    X = df[feats_present].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── choose k ─────────────────────────────────────────────────────────────
    best_k, sil_scores = choose_k(X_scaled, figdir)
    k = args.k if args.k >= 2 else best_k
    print(f"\n  Using k={k}")

    # ── fit ───────────────────────────────────────────────────────────────────
    km, labels = fit_kmeans(X_scaled, k)
    df["cluster"] = labels

    # ── label clusters ───────────────────────────────────────────────────────
    cluster_map = auto_label(df, k)
    df["driving_style"] = df["cluster"].map(cluster_map)
    print("\n  Cluster → Style mapping:", cluster_map)

    # ── profiles (mean of each feature per cluster) ───────────────────────────
    profile_cols = feats_present + ["l_per_100km", "duration_min",
                                    "distance_km", "avg_speed_kmh"]
    profile_cols = [c for c in profile_cols if c in df.columns]
    profiles = df.groupby("cluster")[profile_cols].mean()

    print("\n── Cluster Profiles (means) ──")
    print(profiles.round(2).to_string())

    # ── sizes ─────────────────────────────────────────────────────────────────
    sizes = df.groupby("driving_style")["l_per_100km"].agg(
        trips="count", median_l100="median", mean_l100="mean"
    ).round(2)
    print("\n── Cluster Sizes & Fuel ──")
    print(sizes.to_string())

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
    ]
    with open(os.path.join(args.report_dir, "cluster_profiles.txt"), "w") as fh:
        fh.write("\n".join(report_lines) + "\n")
    print(f"Report: {args.report_dir}/cluster_profiles.txt")


if __name__ == "__main__":
    main()