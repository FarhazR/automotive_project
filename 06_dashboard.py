#!/usr/bin/env python3
"""
Step 7 – Analysis Dashboard  (SRS FR-09)

Reads all outputs from previous steps and builds a single
self-contained HTML file: reports/dashboard.html

Open in any browser — no server required.
"""

import argparse
import base64
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from io import BytesIO

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

PALETTE = {
    "Efficient"  : "#2E86AB",
    "Moderate"   : "#57A773",
    "Aggressive" : "#C84B31",
}

def fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── chart helpers ─────────────────────────────────────────────────────────────
def chart_fuel_dist(df):
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor="#0f1923")
    ax.set_facecolor("#0f1923")
    colors = [PALETTE.get(s, "#aaa") for s in df["driving_style"]]
    ax.hist(df["l_per_100km"], bins=40, color="#2E86AB", edgecolor="#0f1923",
            alpha=0.85)
    ax.axvline(df["l_per_100km"].mean(), color="#F5A623", linewidth=1.8,
               linestyle="--", label=f"Mean {df['l_per_100km'].mean():.1f} L/100km")
    ax.axvline(df["l_per_100km"].quantile(0.25), color="#57A773", linewidth=1.4,
               linestyle=":", label=f"Efficient threshold {df['l_per_100km'].quantile(0.25):.1f}")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3a4a")
    ax.tick_params(colors="#9ab")
    ax.xaxis.label.set_color("#9ab"); ax.yaxis.label.set_color("#9ab")
    ax.set_xlabel("L / 100 km"); ax.set_ylabel("Trips")
    ax.legend(fontsize=8, labelcolor="#cdd")
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig)


def chart_cluster_box(df):
    order = ["Efficient", "Moderate", "Aggressive"]
    order = [o for o in order if o in df["driving_style"].values]
    fig, ax = plt.subplots(figsize=(6, 3.8), facecolor="#0f1923")
    ax.set_facecolor("#0f1923")
    bp = ax.boxplot(
        [df[df["driving_style"] == s]["l_per_100km"].dropna() for s in order],
        patch_artist=True, medianprops={"color": "#fff", "linewidth": 2},
        whiskerprops={"color": "#6a8a9a"}, capprops={"color": "#6a8a9a"},
        flierprops={"markerfacecolor": "#6a8a9a", "markersize": 3, "alpha": 0.5},
    )
    for patch, style in zip(bp["boxes"], order):
        patch.set_facecolor(PALETTE[style])
        patch.set_alpha(0.8)
    ax.set_xticklabels(order, color="#cdd")
    ax.set_ylabel("L / 100 km", color="#9ab")
    ax.tick_params(colors="#9ab")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3a4a")
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig)


def chart_scatter(df, x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(6, 3.8), facecolor="#0f1923")
    ax.set_facecolor("#0f1923")
    for style, grp in df.groupby("driving_style"):
        ax.scatter(grp[x], grp[y], alpha=0.35, s=14,
                   color=PALETTE.get(style, "#888"), label=style)
    ax.set_xlabel(xlabel, color="#9ab"); ax.set_ylabel(ylabel, color="#9ab")
    ax.set_title(title, color="#cdd", fontsize=10)
    ax.tick_params(colors="#9ab")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3a4a")
    ax.legend(fontsize=8, labelcolor="#cdd",
              facecolor="#1a2a3a", edgecolor="#2a3a4a")
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig)


def chart_feature_importance(profile):
    feats = [k for k in profile if not k.startswith("_")]
    means = [profile[f]["mean"] for f in feats]
    dirs  = [profile[f]["direction"] for f in feats]
    pop_stds = [profile[f]["pop_std"] for f in feats]
    norm = [m / s for m, s in zip(means, pop_stds)]

    labels = {
        "speed_mean_kmh": "Avg Speed", "speed_std_kmh": "Speed Variability",
        "acc_std_g": "Accel. Variability", "hard_acc_count": "Hard Acc. Events",
        "hard_brk_count": "Hard Braking Events", "rpm_mean": "Mean RPM",
        "idle_frac": "Idle Fraction", "stop_go_events": "Stop-and-Go Events",
        "acc_max_g": "Peak Acceleration",
    }
    colors = ["#2E86AB" if d == "higher" else "#C84B31" for d in dirs]
    display = [labels.get(f, f) for f in feats]

    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="#0f1923")
    ax.set_facecolor("#0f1923")
    bars = ax.barh(display, norm, color=colors, alpha=0.82)
    ax.set_xlabel("Profile mean ÷ population std", color="#9ab")
    ax.tick_params(colors="#9ab", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3a4a")
    patches = [
        mpatches.Patch(color="#2E86AB", label="Higher = more efficient"),
        mpatches.Patch(color="#C84B31", label="Lower = more efficient"),
    ]
    ax.legend(handles=patches, fontsize=8, labelcolor="#cdd",
              facecolor="#1a2a3a", edgecolor="#2a3a4a")
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig)


def chart_radar(df, profile):
    feats = ["speed_mean_kmh", "acc_std_g", "hard_acc_count",
             "idle_frac", "stop_go_events", "rpm_mean"]
    feat_labels = ["Avg Speed", "Accel.\nVariability", "Hard Acc.",
                   "Idle\nFraction", "Stop-Go\nEvents", "Mean RPM"]

    styles = ["Efficient", "Moderate", "Aggressive"]
    style_means = {s: df[df["driving_style"] == s][feats].mean()
                   for s in styles if s in df["driving_style"].values}
    prof_mean = pd.Series({f: profile[f]["mean"] for f in feats})

    # Normalise 0-1 across all series for radar display
    all_vals = pd.DataFrame({**style_means, "Profile": prof_mean})
    norm = (all_vals - all_vals.min()) / (all_vals.max() - all_vals.min() + 1e-9)

    N = len(feats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True},
                           facecolor="#0f1923")
    ax.set_facecolor("#0f1923")
    ax.spines["polar"].set_color("#2a3a4a")
    ax.tick_params(colors="#7a9aaa", labelsize=8)

    style_colors = {**PALETTE, "Profile": "#F5A623"}
    for label, series in list(norm.items()):
        vals = series[feats].tolist() + [series[feats[0]]]
        color = style_colors.get(label, "#aaa")
        lw = 2.5 if label == "Profile" else 1.5
        ax.plot(angles, vals, "o-", linewidth=lw, color=color,
                label=label, markersize=4)
        ax.fill(angles, vals, alpha=0.07, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), feat_labels,
                      color="#9ab", size=8)
    ax.set_yticklabels([])
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15),
              fontsize=8, labelcolor="#cdd",
              facecolor="#1a2a3a", edgecolor="#2a3a4a")
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_corr_bar(df):
    feats = ["speed_mean_kmh", "rpm_mean", "stop_go_events", "idle_frac",
             "acc_std_g", "hard_acc_count", "hard_brk_count",
             "acc_max_g", "speed_std_kmh"]
    labels = ["Avg Speed", "Mean RPM", "Stop-Go Events", "Idle Fraction",
              "Accel. Variability", "Hard Acc.", "Hard Braking",
              "Peak Accel.", "Speed Variability"]
    corrs = [df[f].corr(df["l_per_100km"]) for f in feats]
    colors = ["#C84B31" if c > 0 else "#2E86AB" for c in corrs]

    idx = np.argsort(corrs)
    corrs = [corrs[i] for i in idx]
    labels = [labels[i] for i in idx]
    colors = [colors[i] for i in idx]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0f1923")
    ax.set_facecolor("#0f1923")
    ax.barh(labels, corrs, color=colors, alpha=0.85)
    ax.axvline(0, color="#6a8a9a", linewidth=0.8)
    ax.set_xlabel("Pearson r with L/100km", color="#9ab")
    ax.tick_params(colors="#9ab", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3a4a")
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig)


# ── HTML builder ──────────────────────────────────────────────────────────────
def build_html(df, profile, recs_by_style) -> str:
    # Pre-compute all charts
    print("  Rendering charts...")
    c_dist   = chart_fuel_dist(df)
    c_box    = chart_cluster_box(df)
    c_s_fuel = chart_scatter(df, "speed_mean_kmh", "l_per_100km",
                              "Avg Speed (km/h)", "L / 100 km",
                              "Speed vs Fuel Consumption")
    c_r_fuel = chart_scatter(df, "rpm_mean", "l_per_100km",
                              "Mean RPM", "L / 100 km",
                              "RPM vs Fuel Consumption")
    c_a_fuel = chart_scatter(df, "acc_std_g", "l_per_100km",
                              "Accel. Variability (g std)", "L / 100 km",
                              "Acceleration Variability vs Fuel")
    c_i_fuel = chart_scatter(df, "idle_frac", "l_per_100km",
                              "Idle Fraction", "L / 100 km",
                              "Idle Fraction vs Fuel")
    c_prof   = chart_feature_importance(profile)
    c_radar  = chart_radar(df, profile)
    c_corr   = chart_corr_bar(df)

    # Stats
    n_trips   = len(df)
    n_vehs    = df["veh_id"].nunique()
    mean_l100 = df["l_per_100km"].mean()
    eff_ref   = profile["_meta"]["l100km_mean"]

    style_counts = df["driving_style"].value_counts().to_dict()
    eff_n  = style_counts.get("Efficient", 0)
    mod_n  = style_counts.get("Moderate", 0)
    agg_n  = style_counts.get("Aggressive", 0)

    def rec_html(style):
        recs = recs_by_style.get(style, [])
        if not recs:
            return "<p class='no-rec'>No significant inefficiencies detected.</p>"
        items = ""
        for r in recs:
            sev_class = "sev-high" if r["severity"] > 0.6 else \
                        "sev-mid"  if r["severity"] > 0.35 else "sev-low"
            items += f"""
            <div class="rec-item {sev_class}">
              <div class="rec-label">{r['label']}</div>
              <div class="rec-bar-wrap">
                <div class="rec-bar" style="width:{r['severity']*100:.0f}%"></div>
              </div>
              <div class="rec-msg">{r['message']}</div>
            </div>"""
        return items

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Driving Behaviour & Fuel Consumption — Analysis Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@400;600;700&display=swap');

  :root {{
    --bg:       #0f1923;
    --surface:  #162030;
    --border:   #1e3040;
    --text:     #c8d8e8;
    --muted:    #6a8a9a;
    --eff:      #2E86AB;
    --mod:      #57A773;
    --agg:      #C84B31;
    --accent:   #F5A623;
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    line-height: 1.65;
  }}

  /* ── header ── */
  .header {{
    padding: 52px 48px 40px;
    border-bottom: 1px solid var(--border);
  }}
  .header-eyebrow {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-bottom: 12px;
  }}
  .header h1 {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: #e8f2ff;
    line-height: 1.2;
    max-width: 640px;
  }}
  .header-sub {{
    margin-top: 10px;
    color: var(--muted);
    font-size: 13px;
    max-width: 560px;
  }}

  /* ── stat row ── */
  .stat-row {{
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
  }}
  .stat-cell {{
    flex: 1;
    padding: 28px 32px;
    border-right: 1px solid var(--border);
  }}
  .stat-cell:last-child {{ border-right: none; }}
  .stat-val {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #e8f2ff;
  }}
  .stat-val span {{ font-size: 14px; font-weight: 400; color: var(--muted); margin-left: 4px; }}
  .stat-lbl {{ font-size: 12px; color: var(--muted); margin-top: 3px; }}

  /* ── layout ── */
  .main {{ padding: 40px 48px; max-width: 1300px; }}
  .section {{ margin-bottom: 56px; }}
  .section-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 16px;
    font-weight: 600;
    color: #e8f2ff;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }}

  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }}

  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 20px;
  }}
  .card-title {{
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 12px;
    font-weight: 500;
  }}
  .card img {{ width: 100%; height: auto; display: block; border-radius: 3px; }}

  /* ── cluster pills ── */
  .cluster-grid {{ display: flex; gap: 16px; margin-bottom: 28px; }}
  .cluster-card {{
    flex: 1;
    padding: 22px 24px;
    border-radius: 6px;
    border-left: 4px solid;
  }}
  .cluster-card.eff {{ background: #0e1e2e; border-color: var(--eff); }}
  .cluster-card.mod {{ background: #0e1e1a; border-color: var(--mod); }}
  .cluster-card.agg {{ background: #1e0e0a; border-color: var(--agg); }}
  .cluster-name {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px; font-weight: 600; color: #e8f2ff;
    margin-bottom: 6px;
  }}
  .cluster-stat {{ font-size: 22px; font-weight: 700; color: #e8f2ff; }}
  .cluster-stat span {{ font-size: 12px; color: var(--muted); font-weight: 400; margin-left: 3px; }}
  .cluster-meta {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}

  /* ── tabs ── */
  .tabs {{ display: flex; gap: 4px; margin-bottom: 20px; }}
  .tab {{
    padding: 8px 18px;
    border-radius: 4px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    font-size: 13px;
    cursor: pointer;
    font-family: 'Inter', sans-serif;
    transition: background 0.15s, color 0.15s;
  }}
  .tab.active {{ background: var(--surface); color: #e8f2ff; border-color: #3a5a6a; }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}

  /* ── recommendations ── */
  .rec-item {{
    padding: 14px 16px;
    margin-bottom: 10px;
    border-radius: 5px;
    border-left: 3px solid;
    background: #0d1824;
  }}
  .rec-item.sev-high {{ border-color: var(--agg); }}
  .rec-item.sev-mid  {{ border-color: var(--accent); }}
  .rec-item.sev-low  {{ border-color: var(--mod); }}
  .rec-label {{ font-size: 12px; font-weight: 600; color: #e8f2ff; margin-bottom: 6px; }}
  .rec-bar-wrap {{
    height: 4px; background: #1e3040; border-radius: 2px; margin-bottom: 8px;
  }}
  .rec-bar {{ height: 100%; border-radius: 2px; background: currentColor; }}
  .sev-high .rec-bar {{ background: var(--agg); }}
  .sev-mid  .rec-bar {{ background: var(--accent); }}
  .sev-low  .rec-bar {{ background: var(--mod); }}
  .rec-msg {{ font-size: 12px; color: var(--muted); line-height: 1.6; }}
  .no-rec {{ font-size: 13px; color: var(--mod); padding: 12px 0; }}

  /* ── profile table ── */
  .profile-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .profile-table th, .profile-table td {{
    padding: 9px 14px; border-bottom: 1px solid var(--border); text-align: left;
  }}
  .profile-table th {{ color: var(--muted); font-weight: 500; }}
  .profile-table tr:last-child td {{ border-bottom: none; }}
  .dir-higher {{ color: var(--eff); }}
  .dir-lower  {{ color: var(--agg); }}

  /* ── footer ── */
  .footer {{
    padding: 28px 48px;
    border-top: 1px solid var(--border);
    font-size: 11px;
    color: var(--muted);
    line-height: 1.7;
  }}

  @media (max-width: 800px) {{
    .header, .main, .footer {{ padding: 24px 20px; }}
    .grid-2, .grid-3, .cluster-grid {{ grid-template-columns: 1fr; flex-direction: column; }}
    .stat-row {{ flex-direction: column; }}
    .stat-cell {{ border-right: none; border-bottom: 1px solid var(--border); }}
  }}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="header-eyebrow">VED Dataset · ICE Vehicles · {n_vehs} vehicles · {n_trips:,} trips</div>
  <h1>Driving Behaviour &amp; Fuel Consumption</h1>
  <p class="header-sub">Analysis of how driving style influences fuel efficiency — patterns extracted from real-world OBD telemetry using unsupervised clustering and correlation analysis.</p>
</div>

<!-- STAT ROW -->
<div class="stat-row">
  <div class="stat-cell">
    <div class="stat-val">{n_trips:,}</div>
    <div class="stat-lbl">Qualifying trips analysed</div>
  </div>
  <div class="stat-cell">
    <div class="stat-val">{n_vehs}</div>
    <div class="stat-lbl">Unique ICE vehicles</div>
  </div>
  <div class="stat-cell">
    <div class="stat-val">{mean_l100:.1f}<span>L/100km</span></div>
    <div class="stat-lbl">Fleet mean fuel consumption</div>
  </div>
  <div class="stat-cell">
    <div class="stat-val">{eff_ref:.1f}<span>L/100km</span></div>
    <div class="stat-lbl">Efficient driving reference</div>
  </div>
  <div class="stat-cell">
    <div class="stat-val">{((mean_l100 - eff_ref) / mean_l100 * 100):.0f}<span>%</span></div>
    <div class="stat-lbl">Potential reduction vs fleet mean</div>
  </div>
</div>

<!-- MAIN -->
<div class="main">

  <!-- SECTION 1: Clusters -->
  <div class="section">
    <div class="section-title">Driving Style Clusters</div>
    <div class="cluster-grid">
      <div class="cluster-card eff">
        <div class="cluster-name">Efficient</div>
        <div class="cluster-stat">{df[df['driving_style']=='Efficient']['l_per_100km'].median():.1f}<span>L/100km median</span></div>
        <div class="cluster-meta">{eff_n} trips · high steady speed · minimal stop-and-go</div>
      </div>
      <div class="cluster-card mod">
        <div class="cluster-name">Moderate</div>
        <div class="cluster-stat">{df[df['driving_style']=='Moderate']['l_per_100km'].median():.1f}<span>L/100km median</span></div>
        <div class="cluster-meta">{mod_n} trips · mixed urban · moderate idle</div>
      </div>
      <div class="cluster-card agg">
        <div class="cluster-name">Aggressive</div>
        <div class="cluster-stat">{df[df['driving_style']=='Aggressive']['l_per_100km'].median():.1f}<span>L/100km median</span></div>
        <div class="cluster-meta">{agg_n} trips · frequent hard events · high idle fraction</div>
      </div>
    </div>
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Fuel consumption distribution</div>
        <img src="data:image/png;base64,{c_dist}" alt="Fuel distribution">
      </div>
      <div class="card">
        <div class="card-title">L/100km by driving style</div>
        <img src="data:image/png;base64,{c_box}" alt="Cluster boxplot">
      </div>
    </div>
  </div>

  <!-- SECTION 2: Correlations -->
  <div class="section">
    <div class="section-title">Feature Correlations with Fuel Consumption</div>
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Pearson r with L/100km — all behavioural features</div>
        <img src="data:image/png;base64,{c_corr}" alt="Correlation bar">
      </div>
      <div class="card">
        <div class="card-title">Driving style profiles vs efficient reference (radar)</div>
        <img src="data:image/png;base64,{c_radar}" alt="Radar chart">
      </div>
    </div>
    <div class="grid-2" style="margin-top:20px">
      <div class="card">
        <div class="card-title">Average speed vs fuel consumption</div>
        <img src="data:image/png;base64,{c_s_fuel}" alt="Speed scatter">
      </div>
      <div class="card">
        <div class="card-title">Mean RPM vs fuel consumption</div>
        <img src="data:image/png;base64,{c_r_fuel}" alt="RPM scatter">
      </div>
    </div>
    <div class="grid-2" style="margin-top:20px">
      <div class="card">
        <div class="card-title">Acceleration variability vs fuel consumption</div>
        <img src="data:image/png;base64,{c_a_fuel}" alt="Accel scatter">
      </div>
      <div class="card">
        <div class="card-title">Idle fraction vs fuel consumption</div>
        <img src="data:image/png;base64,{c_i_fuel}" alt="Idle scatter">
      </div>
    </div>
  </div>

  <!-- SECTION 3: Profile -->
  <div class="section">
    <div class="section-title">Fuel-Efficient Driving Profile</div>
    <p style="color:var(--muted);font-size:13px;margin-bottom:18px">
      Built from {profile['_meta']['n_trips']} trips with L/100km ≤ {profile['_meta']['l100km_threshold']:.2f}
      (bottom 25th percentile). Mean fuel use: <strong style="color:#e8f2ff">{profile['_meta']['l100km_mean']:.2f} L/100km</strong>.
    </p>
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Profile feature values (normalised by population std)</div>
        <img src="data:image/png;base64,{c_prof}" alt="Profile chart">
      </div>
      <div class="card" style="padding:0;overflow:hidden">
        <table class="profile-table">
          <thead>
            <tr>
              <th>Feature</th><th>Profile mean</th><th>Direction</th>
            </tr>
          </thead>
          <tbody>
            {"".join(
              f"<tr><td>{k.replace('_',' ')}</td>"
              f"<td style='font-family:monospace'>{v['mean']:.3f}</td>"
              f"<td class='dir-{v['direction']}'>{v['direction']}</td></tr>"
              for k, v in profile.items() if not k.startswith("_")
            )}
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- SECTION 4: Recommendations -->
  <div class="section">
    <div class="section-title">Sample Driving Recommendations</div>
    <p style="color:var(--muted);font-size:13px;margin-bottom:18px">
      Median representative trip for each driving style, compared against the efficient profile.
    </p>
    <div class="tabs">
      <button class="tab active" onclick="switchTab(this,'tab-agg')">Aggressive</button>
      <button class="tab" onclick="switchTab(this,'tab-mod')">Moderate</button>
      <button class="tab" onclick="switchTab(this,'tab-eff')">Efficient</button>
    </div>
    <div id="tab-agg" class="tab-panel active">{rec_html("Aggressive")}</div>
    <div id="tab-mod" class="tab-panel">{rec_html("Moderate")}</div>
    <div id="tab-eff" class="tab-panel">{rec_html("Efficient")}</div>
  </div>

</div><!-- /main -->

<!-- FOOTER -->
<div class="footer">
  <strong>Data:</strong> Vehicle Energy Dataset (VED), University of Michigan.
  ICE vehicles only · 4 weeks of dynamic data · fuel estimated from MAF sensor (stoichiometric approximation; direct OBD fuel rate unavailable in this subset).
  &nbsp;·&nbsp;
  <strong>Limitations:</strong> Fuel estimates carry approximation error. Vehicle class metadata largely absent.
  Clustering silhouette 0.242 indicates overlapping real-world styles.
  Recommendations reflect dataset patterns; individual results depend on vehicle, road, and conditions.
</div>

<script>
function switchTab(btn, id) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(id).classList.add('active');
}}
</script>
</body>
</html>"""
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-file",   default="data/processed/features_clustered.csv")
    ap.add_argument("--profile",     default="data/processed/efficient_profile.json")
    ap.add_argument("--report-dir",  default="reports")
    args = ap.parse_args()

    df      = pd.read_csv(args.feat_file)
    with open(args.profile) as fh:
        profile = json.load(fh)

    print(f"Loaded {len(df):,} trips.")

    # Re-run recommender for each demo style
    from importlib.util import spec_from_loader, module_from_spec

    # Inline the generate_recommendations function (avoid import path issues)
    PROFILE_FEATS = [k for k in profile if not k.startswith("_")]
    DIRECTION = {k: profile[k]["direction"] for k in PROFILE_FEATS}
    FLAG_THRESHOLD = 0.5

    MESSAGES = {
        "stop_go_events": "Your trip had {val:.0f} stop-and-go events vs {ref:.0f} in efficient trips. Frequent stopping and restarting burns significantly more fuel. Where conditions allow, maintain a steady roll rather than coming to a complete stop.",
        "idle_frac": "Your engine idled for {pct:.0f}% of trip time vs {ref_pct:.0f}% in efficient trips. Extended idling wastes fuel with zero distance gain. If stationary for more than ~60 seconds, switching off is worth considering.",
        "hard_acc_count": "You recorded {val:.0f} hard acceleration events vs {ref:.0f} in efficient trips. Rapid acceleration sharply increases fuel demand. Gradual, progressive acceleration reduces fuel consumption and mechanical wear.",
        "hard_brk_count": "You recorded {val:.0f} hard braking events vs {ref:.0f} in efficient trips. Hard braking dissipates kinetic energy that cost fuel to build. Anticipating traffic flow further ahead allows gentler deceleration.",
        "acc_std_g": "Acceleration variability (std {val:.3f}g vs {ref:.3f}g) is elevated. Smoother, more consistent throttle inputs are associated with lower fuel use. Try to maintain even pressure on the accelerator and avoid surging.",
        "speed_std_kmh": "Speed variability is higher than the efficient profile ({val:.1f} vs {ref:.1f} km/h std). Maintaining a more consistent speed reduces the repeated acceleration cost. On open roads, cruise control or conscious speed-steadying helps.",
        "speed_mean_kmh": "Average trip speed ({val:.1f} km/h) is below the efficient profile ({ref:.1f} km/h). This suggests mostly low-speed urban operation with high idle/stop overhead. Where route choice is possible, roads with fewer traffic signals can improve efficiency.",
        "rpm_mean": "Mean RPM ({val:.0f}) is lower than in efficient trips ({ref:.0f}). In combination with low speed, this may indicate short urban trips where the engine rarely reaches its efficient operating range. Consolidating short trips where possible reduces per-km fuel cost.",
        "acc_max_g": "Peak acceleration ({val:.2f}g vs {ref:.2f}g in efficient trips) is high. Occasional aggressive inputs significantly raise instantaneous fuel demand. A lighter right foot during the first few seconds of acceleration makes the largest difference.",
    }

    def gen_recs(trip):
        recs = []
        for feat in PROFILE_FEATS:
            if feat not in trip.index or feat not in profile:
                continue
            val = trip[feat]
            if pd.isna(val):
                continue
            info = profile[feat]
            ref_mean = info["mean"]
            pop_std  = info["pop_std"]
            direction = info["direction"]
            dev = (val - ref_mean) / pop_std if direction == "lower" \
                  else (ref_mean - val) / pop_std
            if dev < FLAG_THRESHOLD or feat not in MESSAGES:
                continue
            severity = min(1.0, dev / 2.0)
            fmt = {"val": val, "ref": ref_mean, "pct": val*100, "ref_pct": ref_mean*100}
            recs.append({
                "feature": feat,
                "label": feat.replace("_", " ").title(),
                "val": round(float(val), 3),
                "ref_mean": round(float(ref_mean), 3),
                "deviation": round(float(dev), 3),
                "severity": round(float(severity), 3),
                "message": MESSAGES[feat].format(**fmt),
            })
        recs.sort(key=lambda r: r["severity"], reverse=True)
        return recs

    recs_by_style = {}
    for style in ["Aggressive", "Moderate", "Efficient"]:
        subset = df[df["driving_style"] == style]
        if subset.empty:
            continue
        median_idx = (subset["l_per_100km"] - subset["l_per_100km"].median()).abs().idxmin()
        trip = subset.loc[median_idx]
        recs_by_style[style] = gen_recs(trip)

    print("  Building HTML dashboard...")
    html = build_html(df, profile, recs_by_style)
    out  = os.path.join(args.report_dir, "dashboard.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"\n  Done → {out}  ({os.path.getsize(out)//1024} KB)")
    print("  Open in any browser: file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()