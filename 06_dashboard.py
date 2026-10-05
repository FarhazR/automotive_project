#!/usr/bin/env python3
"""
Step 7 - Analysis Dashboard (SRS FR-09)

Reads the outputs of steps 1-5 and builds one self-contained HTML file:
  reports/dashboard.html

Open it in any browser; no server needed.
"""
import argparse
import base64
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from recommender import FEAT_LABELS, generate_recommendations  # noqa: E402

BG, GRID, TXT, MUTED = "#0f1923", "#2a3a4a", "#cdd", "#9ab"
PATTERN_COLORS = ["#2E86AB", "#57A773", "#F5A623", "#C84B31"]
SUPPORTED, UNSUPPORTED = "#C84B31", "#6a8a9a"


def pattern_color(rank, n):
    if n <= 1:
        return PATTERN_COLORS[0]
    return PATTERN_COLORS[int(round(rank * (len(PATTERN_COLORS) - 1) / (n - 1)))]


def pattern_order(df):
    return (df.dropna(subset=["style_rank"])
              .sort_values("style_rank")["driving_style"].unique().tolist())


def fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def style_axes(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)


# -- charts -------------------------------------------------------------------
def chart_rel_dist(df):
    rel = df["fuel_rel_vehicle"].dropna()
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=BG)
    style_axes(ax)
    ax.hist(rel, bins=40, color="#2E86AB", edgecolor=BG, alpha=0.85)
    ax.axvline(1.0, color="#F5A623", linewidth=1.8, linestyle="--",
               label="Vehicle's own median trip (1.0)")
    ax.set_xlabel("Fuel use relative to the vehicle's own median")
    ax.set_ylabel("Trips")
    ax.legend(fontsize=8, labelcolor=TXT, facecolor="#1a2a3a", edgecolor=GRID)
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig)


def chart_pattern_box(df, styles, colors):
    sub = df.dropna(subset=["fuel_rel_vehicle"])
    fig, ax = plt.subplots(figsize=(6, 3.8), facecolor=BG)
    style_axes(ax)
    bp = ax.boxplot(
        [sub[sub["driving_style"] == s]["fuel_rel_vehicle"] for s in styles],
        patch_artist=True, medianprops={"color": "#fff", "linewidth": 2},
        whiskerprops={"color": "#6a8a9a"}, capprops={"color": "#6a8a9a"},
        flierprops={"markerfacecolor": "#6a8a9a", "markersize": 3, "alpha": 0.5})
    for patch, s in zip(bp["boxes"], styles):
        patch.set_facecolor(colors[s])
        patch.set_alpha(0.8)
    ax.axhline(1.0, color="#6a8a9a", linestyle="--", linewidth=0.8)
    ax.set_xticklabels(styles, color=TXT)
    ax.set_ylabel("Fuel relative to vehicle's own median")
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig)


def chart_effects(profile):
    feats = [k for k in profile if not k.startswith("_")]
    order = sorted(feats, key=lambda f: profile[f]["pct_per_sd"])
    est = [profile[f]["pct_per_sd"] for f in order]
    lo = [profile[f]["pct_per_sd"] - profile[f]["ci_low_pct"] for f in order]
    hi = [profile[f]["ci_high_pct"] - profile[f]["pct_per_sd"] for f in order]
    cols = [SUPPORTED if profile[f]["supported"] else UNSUPPORTED for f in order]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
    style_axes(ax)
    ax.barh([FEAT_LABELS.get(f, f) for f in order], est, xerr=[lo, hi],
            color=cols, ecolor=MUTED, capsize=3, alpha=0.9)
    ax.axvline(0, color="#6a8a9a", linewidth=0.8)
    ax.set_xlabel("% change in fuel per +1 within-vehicle SD (95% CI)")
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig)


def chart_scatter_rel(df, x, xlabel, styles, colors):
    sub = df.dropna(subset=["fuel_rel_vehicle", x])
    fig, ax = plt.subplots(figsize=(6, 3.8), facecolor=BG)
    style_axes(ax)
    for s in styles:
        g = sub[sub["driving_style"] == s]
        ax.scatter(g[x], g["fuel_rel_vehicle"], alpha=0.35, s=14,
                   color=colors[s], label=s)
    ax.axhline(1.0, color="#6a8a9a", linestyle="--", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fuel relative to vehicle's own median")
    ax.legend(fontsize=8, labelcolor=TXT, facecolor="#1a2a3a", edgecolor=GRID)
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig)


# -- statistics ---------------------------------------------------------------
def pattern_gap(df):
    """Same-vehicle gap between the smoothest and most stop-and-go pattern."""
    sub = df.dropna(subset=["fuel_rel_vehicle", "style_rank"])
    top = sub["style_rank"].max()
    diffs = []
    for _, g in sub.groupby("veh_id"):
        a = g.loc[g["style_rank"] == 0, "fuel_rel_vehicle"]
        b = g.loc[g["style_rank"] == top, "fuel_rel_vehicle"]
        if len(a) >= 2 and len(b) >= 2:
            diffs.append(b.median() - a.median())
    if len(diffs) < 10:
        return None
    d = np.array(diffs)
    return {"median": float(np.median(d)), "share_pos": float((d > 0).mean()),
            "n": int(len(d))}


def cluster_silhouette(df):
    """Recompute the silhouette using the exact preprocessing of step 4."""
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        spec = spec_from_file_location("clustering04", "04_clustering.py")
        clu = module_from_spec(spec)
        spec.loader.exec_module(clu)
        feats = [c for c in clu.CLUSTER_FEATS if c in df.columns]
        X = StandardScaler().fit_transform(clu.prepare_matrix(df, feats).values)
        return float(silhouette_score(X, df["cluster"].values))
    except Exception as exc:  # the dashboard must still build without it
        print(f"  (silhouette not computed: {exc})")
        return None


# -- HTML ---------------------------------------------------------------------
CSS = """
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@400;600;700&display=swap');
  :root { --bg:#0f1923; --surface:#162030; --border:#1e3040; --text:#c8d8e8;
          --muted:#6a8a9a; --accent:#F5A623; --good:#57A773; --bad:#C84B31; }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text);
         font-family: 'Inter', sans-serif; font-size: 14px; line-height: 1.65; }
  .header { padding: 52px 48px 40px; border-bottom: 1px solid var(--border); }
  .header-eyebrow { font-family: 'Space Grotesk', sans-serif; font-size: 11px;
                    letter-spacing: 0.12em; color: var(--muted); margin-bottom: 12px; }
  .header h1 { font-family: 'Space Grotesk', sans-serif; font-size: 32px;
               font-weight: 700; color: #e8f2ff; line-height: 1.2; max-width: 700px; }
  .header-sub { margin-top: 10px; color: var(--muted); font-size: 13px; max-width: 680px; }
  .stat-row { display: flex; border-bottom: 1px solid var(--border); }
  .stat-cell { flex: 1; padding: 28px 32px; border-right: 1px solid var(--border); }
  .stat-cell:last-child { border-right: none; }
  .stat-val { font-family: 'Space Grotesk', sans-serif; font-size: 28px;
              font-weight: 700; color: #e8f2ff; }
  .stat-val span { font-size: 14px; font-weight: 400; color: var(--muted); margin-left: 4px; }
  .stat-lbl { font-size: 12px; color: var(--muted); margin-top: 3px; }
  .main { padding: 40px 48px; max-width: 1300px; }
  .section { margin-bottom: 56px; }
  .section-title { font-family: 'Space Grotesk', sans-serif; font-size: 16px;
                   font-weight: 600; color: #e8f2ff; margin-bottom: 8px;
                   padding-bottom: 10px; border-bottom: 1px solid var(--border); }
  .section-note { color: var(--muted); font-size: 13px; margin: 12px 0 18px; max-width: 900px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .card { background: var(--surface); border: 1px solid var(--border);
          border-radius: 6px; padding: 20px; }
  .card-title { font-size: 12px; color: var(--muted); margin-bottom: 12px; font-weight: 500; }
  .card img { width: 100%; height: auto; display: block; border-radius: 3px; }
  .cluster-grid { display: flex; gap: 16px; margin-bottom: 28px; }
  .cluster-card { flex: 1; padding: 22px 24px; border-radius: 6px;
                  border-left: 4px solid; background: #0e1a26; }
  .cluster-name { font-family: 'Space Grotesk', sans-serif; font-size: 15px;
                  font-weight: 600; color: #e8f2ff; margin-bottom: 6px; }
  .cluster-stat { font-size: 22px; font-weight: 700; color: #e8f2ff; }
  .cluster-stat span { font-size: 12px; color: var(--muted); font-weight: 400; margin-left: 6px; }
  .cluster-meta { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .tabs { display: flex; gap: 4px; margin-bottom: 20px; }
  .tab { padding: 8px 18px; border-radius: 4px; border: 1px solid var(--border);
         background: transparent; color: var(--muted); font-size: 13px; cursor: pointer;
         font-family: 'Inter', sans-serif; }
  .tab.active { background: var(--surface); color: #e8f2ff; border-color: #3a5a6a; }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }
  .rec-item { padding: 14px 16px; margin-bottom: 10px; border-radius: 5px;
              border-left: 3px solid; background: #0d1824; }
  .rec-item.high { border-color: var(--bad); }
  .rec-item.mid { border-color: var(--accent); }
  .rec-item.low { border-color: var(--good); }
  .rec-head { display: flex; justify-content: space-between; font-size: 12px;
              font-weight: 600; color: #e8f2ff; margin-bottom: 6px; }
  .rec-head span { color: var(--muted); font-weight: 400; }
  .rec-bar-wrap { height: 4px; background: #1e3040; border-radius: 2px; margin-bottom: 8px; }
  .rec-bar { height: 100%; border-radius: 2px; background: var(--accent); }
  .high .rec-bar { background: var(--bad); }
  .low .rec-bar { background: var(--good); }
  .rec-msg { font-size: 12px; color: var(--muted); line-height: 1.6; }
  .no-rec { font-size: 13px; color: var(--good); padding: 12px 0; }
  .profile-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .profile-table th, .profile-table td { padding: 9px 14px;
        border-bottom: 1px solid var(--border); text-align: left; }
  .profile-table th { color: var(--muted); font-weight: 500; }
  .profile-table tr:last-child td { border-bottom: none; }
  .yes { color: var(--bad); } .no { color: var(--muted); }
  .footer { padding: 28px 48px; border-top: 1px solid var(--border);
            font-size: 11px; color: var(--muted); line-height: 1.7; }
  @media (max-width: 800px) {
    .header, .main, .footer { padding: 24px 20px; }
    .grid-2 { grid-template-columns: 1fr; }
    .cluster-grid, .stat-row { flex-direction: column; }
    .stat-cell { border-right: none; border-bottom: 1px solid var(--border); }
  }
"""

SCRIPT = """
function switchTab(btn, id) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(id).classList.add('active');
}
"""


def rec_html(recs):
    if not recs:
        return "<p class='no-rec'>No clear inefficiencies detected against the reference.</p>"
    top = max(r["est_extra_fuel_pct"] for r in recs) or 1.0
    items = ""
    for r in recs:
        pct = r["est_extra_fuel_pct"]
        cls = "high" if pct >= 5 else "mid" if pct >= 2 else "low"
        items += f"""
        <div class="rec-item {cls}">
          <div class="rec-head">{r['label']}<span>est. +{pct:.1f}% fuel vs reference</span></div>
          <div class="rec-bar-wrap"><div class="rec-bar" style="width:{pct / top * 100:.0f}%"></div></div>
          <div class="rec-msg">{r['message']}</div>
        </div>"""
    return items


def build_html(df, profile, recs_by_style, silhouette, gap) -> str:
    print("  Rendering charts...")
    styles = pattern_order(df)
    colors = {s: pattern_color(i, len(styles)) for i, s in enumerate(styles)}

    c_dist = chart_rel_dist(df)
    c_box = chart_pattern_box(df, styles, colors)
    c_eff = chart_effects(profile)
    c_sg = chart_scatter_rel(df, "stop_go_per_km", "Stop-and-go events per km",
                             styles, colors)
    c_idle = chart_scatter_rel(df, "idle_frac", "Idle fraction", styles, colors)

    n_trips = len(df)
    n_vehs = df["veh_id"].nunique()
    n_model = df.dropna(subset=["fuel_rel_vehicle"])["veh_id"].nunique()
    mean_l100 = df["l_per_100km"].mean()

    if gap:
        gap_val = f"+{gap['median'] * 100:.0f}<span>%</span>"
        gap_lbl = (f"Same-car fuel gap, stop-and-go vs free-flowing trips "
                   f"({gap['n']} vehicles; {gap['share_pos'] * 100:.0f}% use more)")
    else:
        gap_val, gap_lbl = "n/a", "Same-car fuel gap (too few vehicles with both patterns)"

    cards = ""
    for s in styles:
        sub = df[df["driving_style"] == s]
        rel = sub["fuel_rel_vehicle"].median()
        cards += f"""
        <div class="cluster-card" style="border-color:{colors[s]}">
          <div class="cluster-name">{s}</div>
          <div class="cluster-stat">{rel:.2f}x<span>median fuel vs the vehicle's own median</span></div>
          <div class="cluster-meta">{len(sub)} trips &middot; {sub['stop_go_per_km'].mean():.2f} stop-go/km
            &middot; {sub['idle_frac'].mean() * 100:.0f}% idle
            &middot; {sub['l_per_100km'].median():.1f} L/100km median (raw, mixes vehicles)</div>
        </div>"""

    rows = ""
    for feat, p in profile.items():
        if feat.startswith("_"):
            continue
        used = "<td class='yes'>yes</td>" if p["supported"] else \
               "<td class='no'>no (CI includes 0)</td>"
        rows += (f"<tr><td>{FEAT_LABELS.get(feat, feat)}</td>"
                 f"<td style='font-family:monospace'>{p['mean']:.3f}</td>"
                 f"<td style='font-family:monospace'>{p['pct_per_sd']:+.1f}% "
                 f"[{p['ci_low_pct']:+.1f}, {p['ci_high_pct']:+.1f}]</td>{used}</tr>")

    tabs, panels = "", ""
    for i, s in enumerate(styles):
        active = " active" if i == len(styles) - 1 else ""
        tabs += f"<button class='tab{active}' onclick=\"switchTab(this,'tab-{i}')\">{s}</button>"
        panels += f"<div id='tab-{i}' class='tab-panel{active}'>{rec_html(recs_by_style.get(s, []))}</div>"

    meta = profile["_meta"]
    sil_txt = (f"Cluster silhouette {silhouette:.2f}: the patterns overlap, so trips form "
               f"more of a continuum than distinct groups. " if silhouette is not None else "")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Driving Behaviour &amp; Fuel Consumption - Analysis Report</title>
<style>{CSS}</style>
</head>
<body>

<div class="header">
  <div class="header-eyebrow">VED DATASET &middot; ICE VEHICLES &middot; {n_vehs} VEHICLES &middot; {n_trips:,} TRIPS</div>
  <h1>Driving Behaviour &amp; Fuel Consumption</h1>
  <p class="header-sub">How trip-level driving patterns relate to fuel use once each vehicle is compared
  with itself. Fuel is estimated from real-world OBD telemetry; results are associations,
  not proof of cause.</p>
</div>

<div class="stat-row">
  <div class="stat-cell"><div class="stat-val">{n_trips:,}</div>
    <div class="stat-lbl">Qualifying trips analysed</div></div>
  <div class="stat-cell"><div class="stat-val">{n_model}</div>
    <div class="stat-lbl">Vehicles with &ge;5 trips (used in the within-vehicle model)</div></div>
  <div class="stat-cell"><div class="stat-val">{mean_l100:.1f}<span>L/100km</span></div>
    <div class="stat-lbl">Mean estimated fuel use (mixes vehicle types)</div></div>
  <div class="stat-cell"><div class="stat-val">{gap_val}</div>
    <div class="stat-lbl">{gap_lbl}</div></div>
</div>

<div class="main">

  <div class="section">
    <div class="section-title">Trip Patterns</div>
    <p class="section-note">Trips were grouped by stop-and-go rate, idling, braking and acceleration
    variability (not by speed or fuel). Fuel is shown relative to each vehicle's own median trip, so
    differences between cars do not distort the comparison.</p>
    <div class="cluster-grid">{cards}</div>
    <div class="grid-2">
      <div class="card"><div class="card-title">Relative fuel use across all trips</div>
        <img src="data:image/png;base64,{c_dist}" alt="Relative fuel distribution"></div>
      <div class="card"><div class="card-title">Relative fuel use by trip pattern</div>
        <img src="data:image/png;base64,{c_box}" alt="Pattern boxplot"></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">What Is Associated With Fuel Use Within the Same Vehicle</div>
    <p class="section-note">Fixed-effects model on log relative fuel, controlling for average speed,
    standard errors clustered by vehicle. Bars show the estimated change in fuel for a trip that is
    one within-vehicle standard deviation higher on that feature. Grey bars have a confidence interval
    that includes zero, so they are not used in recommendations.</p>
    <div class="grid-2">
      <div class="card"><div class="card-title">Estimated effect per +1 SD (95% CI)</div>
        <img src="data:image/png;base64,{c_eff}" alt="Effect sizes"></div>
      <div class="card" style="padding:0;overflow:hidden">
        <table class="profile-table">
          <thead><tr><th>Feature</th><th>Efficient reference</th><th>Effect per +1 SD</th><th>Recommended</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </div>
    <div class="grid-2" style="margin-top:20px">
      <div class="card"><div class="card-title">Stop-and-go rate vs relative fuel use</div>
        <img src="data:image/png;base64,{c_sg}" alt="Stop-go scatter"></div>
      <div class="card"><div class="card-title">Idle fraction vs relative fuel use</div>
        <img src="data:image/png;base64,{c_idle}" alt="Idle scatter"></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Sample Recommendations</div>
    <p class="section-note">Reference profile: the {meta['n_trips']} trips ({meta['n_vehicles']} vehicles)
    in the most fuel-efficient 25% relative to their own vehicle. Each example is the median trip of its
    pattern, with suggestions ranked by estimated fuel impact. Estimates are per factor and not additive.</p>
    <div class="tabs">{tabs}</div>
    {panels}
  </div>

</div>

<div class="footer">
  <strong>Data:</strong> Vehicle Energy Dataset (VED), University of Michigan; ICE vehicles only, a subset
  of the weekly files. Fuel is estimated from the MAF sensor assuming stoichiometric petrol combustion,
  so absolute L/100km values are approximate; the analysis relies on relative comparisons within each vehicle.
  &nbsp;&middot;&nbsp;
  <strong>Limitations:</strong> {sil_txt}Stop-and-go and idling are partly imposed by traffic and signals.
  Road gradient, weather and vehicle load are not modelled, vehicle class metadata is largely missing, and
  the data come from one US region. Findings are associations from observational data.
</div>

<script>{SCRIPT}</script>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-file", default="data/processed/features_clustered.csv")
    ap.add_argument("--profile", default="data/processed/efficient_profile.json")
    ap.add_argument("--report-dir", default="reports")
    args = ap.parse_args()

    df = pd.read_csv(args.feat_file)
    with open(args.profile) as fh:
        profile = json.load(fh)
    print(f"Loaded {len(df):,} trips.")

    styles = pattern_order(df)
    recs_by_style = {}
    for s in styles:
        sub = df[df["driving_style"] == s].dropna(subset=["fuel_rel_vehicle"])
        if sub.empty:
            continue
        idx = (sub["fuel_rel_vehicle"] - sub["fuel_rel_vehicle"].median()).abs().idxmin()
        recs_by_style[s] = generate_recommendations(sub.loc[idx], profile)

    print("  Building HTML dashboard...")
    html = build_html(df, profile, recs_by_style, cluster_silhouette(df), pattern_gap(df))
    os.makedirs(args.report_dir, exist_ok=True)
    out = os.path.join(args.report_dir, "dashboard.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"\n  Done -> {out}  ({os.path.getsize(out) // 1024} KB)")
    print("  Open in any browser: file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()