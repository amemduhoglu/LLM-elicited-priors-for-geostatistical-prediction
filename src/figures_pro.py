#!/usr/bin/env python
"""figures_pro.py — publication-grade figures for the C&G submission.

Single authoritative generator for every figure used in paper/draft_v6.md:
  study_area_map · method_schematic · range_scatter · rmse_vs_density_triptych ·
  coverage90_vs_density_triptych · pit_hist · sim_misspec_curve

The generic figure writers in analyze_paper.py / report_extras.py are intentionally
disabled (they used to overwrite these filenames); this script owns all seven so the
pipeline order no longer matters. Country/state boundaries are Natural Earth.

Usage: .venv/bin/python src/figures_pro.py [map|schematic|analysis|all]
Outputs -> results/paper/figures/ and mirrored to paper/figures/.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import geopandas as gpd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import config as configmod   # noqa: E402
import data as datamod       # noqa: E402

ROOT = SRC.parent
FIGS = ROOT / "results" / "paper" / "figures"
PAPER_FIGS = ROOT / "paper" / "figures"
NE = ROOT / "data" / "raw" / "naturalearth"
FIGS.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ shared theme
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#3a3a3a", "axes.linewidth": 0.9,
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
    "legend.frameon": False, "font.family": "DejaVu Sans",
    "axes.titlepad": 8,
})
LAND = "#f4f1ea"; OCEAN = "#dce7ef"; BORDER = "#b8b2a6"

# Okabe-Ito colourblind-safe palette (used consistently across analysis figures)
C_VAGUE = "#2b2b2b"; C_COEF = "#d55e00"; C_BOTH = "#0072b2"
C_VARIO = "#009e73"; C_LOCAL = "#e69f00"

# network -> (which, target col, pretty name, covariate-strength label, cmap, unit)
NETS = [
    ("main",  "tavg", "(a) Greater Himalaya: temperature",         "strong · r = −0.92", "RdYlBu_r", "°C"),
    ("andes", "tavg", "(b) Andes cordillera: temperature",         "medium · r = −0.50", "RdYlBu_r", "°C"),
    ("prcp",  "prcp", "(c) Himalaya: annual precipitation",        "weak · r = −0.21",   "YlGnBu",   "mm"),
    ("urban", "tavg", "(d) US Northeast megalopolis: temperature", "urban, medium · r = −0.68", "RdYlBu_r", "°C"),
]


def _save(fig, name):
    for d in (FIGS, PAPER_FIGS):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name)
    plt.close(fig)
    print(f"[fig] wrote {name}")


NE_BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"


def _ensure_ne_10m(kind):
    """Fetch the high-resolution (10m) Natural Earth layer once if absent. Public domain,
    so it is reproducible; large geojson lives under the gitignored data/ tree."""
    f = NE / f"ne_10m_admin_{kind}.geojson"
    if f.exists():
        return f
    import urllib.request
    NE.mkdir(parents=True, exist_ok=True)
    try:
        print(f"[ne] downloading 10m {kind} ...")
        urllib.request.urlretrieve(f"{NE_BASE}/ne_10m_admin_{kind}.geojson", f)
        return f
    except Exception as e:
        print(f"[ne] 10m download failed ({e}); will fall back to 110m")
        return None


def _ne(kind):
    """Load Natural Earth boundaries, preferring the high-resolution 10m layer (crisp
    coastlines) and falling back to the coarse 110m layer if 10m is unavailable."""
    _ensure_ne_10m(kind)
    for res in ("10m", "110m"):
        f = NE / f"ne_{res}_admin_{kind}.geojson"
        if f.exists():
            return gpd.read_file(f)
    return None


def study_area_map():
    cfg = configmod.load("config.yaml")
    world = _ne("0_countries")
    states = _ne("1_states_provinces")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (which, tcol, title, cov, cmap, unit) in zip(axes.ravel(), NETS):
        p = cfg.dataset(which)
        bb = p["bbox"]  # lat0, lat1, lon0, lon1
        b = datamod.load_dataset(cfg, which)
        fr = b["frame"]
        lon, lat, val = fr["lon"].to_numpy(), fr["lat"].to_numpy(), fr[tcol].to_numpy()
        mlon = (bb[3] - bb[2]) * 0.06; mlat = (bb[1] - bb[0]) * 0.06
        xlim = (bb[2] - mlon, bb[3] + mlon); ylim = (bb[0] - mlat, bb[1] + mlat)

        # ocean background + land + borders (clip to a padded window for speed/detail)
        ax.set_facecolor(OCEAN)
        wx = (xlim[0] - 2, xlim[1] + 2, ylim[0] - 2, ylim[1] + 2)
        w = world.cx[wx[0]:wx[1], wx[2]:wx[3]]
        w.plot(ax=ax, color=LAND, edgecolor=BORDER, linewidth=0.45, zorder=1)
        if states is not None:
            st = states.cx[wx[0]:wx[1], wx[2]:wx[3]]
            st.boundary.plot(ax=ax, color=BORDER, linewidth=0.25, zorder=2)
        sc = ax.scatter(lon, lat, c=val, cmap=cmap, s=28, edgecolor="#222222",
                        linewidth=0.35, zorder=4, alpha=0.95)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(ylim))))  # equirectangular aspect
        ax.set_title(f"{title}\n{cov} · n = {len(val)}", fontsize=10.5, loc="left")
        ax.tick_params(labelsize=7, length=2)
        ax.grid(True, color="white", linewidth=0.4, alpha=0.6, zorder=3)
        cb = fig.colorbar(sc, ax=ax, fraction=0.040, pad=0.02)
        cb.set_label(unit, fontsize=8); cb.ax.tick_params(labelsize=7)

    fig.tight_layout()
    _save(fig, "study_area_map.png")


def method_schematic():
    """Clean left-to-right pipeline. Colour carries meaning: blue = the LLM-supplied
    prior, the only component that varies across conditions; grey = fixed across all
    conditions (data, model, sampler, evaluation)."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    # palette ---------------------------------------------------------------
    ACC_FC, ACC_EC, ACC_TC = "#dbe8f6", "#3b6ea5", "#16314d"   # manipulated (LLM/prior)
    NEU_FC, NEU_EC, NEU_TC = "#eef0f2", "#8b9098", "#2c2c2c"   # fixed
    GRD_EC = "#c07d2b"                                          # constraint / guard
    BODY, MUTE = "#363a40", "#5d6168"
    STYLE = {"acc": (ACC_FC, ACC_EC, ACC_TC), "neu": (NEU_FC, NEU_EC, NEU_TC)}

    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.axis("off"); ax.set_xlim(0, 101); ax.set_ylim(0, 100)

    YB, H, MY = 56, 30, 71                       # box bottom, height, arrow/mid line
    W, xs = 16, [1, 21.75, 42.5, 63.25, 84]

    def node(x, title, body, kind, title_fs=9.4):
        fc, ec, tc = STYLE[kind]
        ax.add_patch(FancyBboxPatch((x, YB), W, H, boxstyle="round,pad=0,rounding_size=2.4",
                                    fc=fc, ec=ec, lw=1.6, zorder=2))
        cx = x + W / 2
        nt = title.count("\n") + 1                      # multi-line titles drop the divider
        div_y = YB + H - 10.2 - (nt - 1) * 4.0
        ax.text(cx, YB + H - 6 - (nt - 1) * 0.5, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=tc, linespacing=1.25, zorder=3)
        ax.plot([x + 2.6, x + W - 2.6], [div_y] * 2, color=ec, lw=0.7, alpha=0.55, zorder=3)
        ax.text(cx, (YB + div_y) / 2 - 0.3, body, ha="center", va="center",
                fontsize=7.9, color=BODY, linespacing=1.55, zorder=3)

    def arrow(x0, x1, label=""):
        ax.annotate("", xy=(x1, MY), xytext=(x0, MY), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", lw=1.9, color="#5a5e64",
                                    shrinkA=0, shrinkB=0, mutation_scale=16))
        if label:
            ax.text((x0 + x1) / 2, MY + 3.2, label, ha="center", va="bottom",
                    fontsize=7, style="italic", color=MUTE, zorder=5)

    # pipeline nodes --------------------------------------------------------
    node(xs[0], "Task description",
         "region · target\ncovariate names · units\n(no observed values)", "neu")
    node(xs[1], "LLM elicitation",
         "18 language models\n9 local (2–35B) +\n6 open-weight + 3 frontier", "acc")
    node(xs[2], "Elicited priors",
         "strict JSON\ncoefficients + variogram\nsign · mean · sd", "acc")
    node(xs[3], "Bayesian\nregression-kriging",
         "data · model · sampler\nidentical: only the\nprior block changes", "neu")
    node(xs[4], "Spatial blocked CV",
         "RMSE · CRPS · PIT\n90% coverage\nvs. data density", "neu")

    arrow(xs[0] + W, xs[1], "text only")
    arrow(xs[1] + W, xs[2], "JSON")
    arrow(xs[2] + W, xs[3], "set prior")
    arrow(xs[3] + W, xs[4], "predict")

    # leakage guard — a constraint on what text reaches the model ------------
    gx, gw, gy, gh = 6, 27, 23, 15
    xg = gx + gw / 2
    ax.add_patch(FancyBboxPatch((gx, gy), gw, gh, boxstyle="round,pad=0,rounding_size=2.2",
                                fc="#ffffff", ec=GRD_EC, lw=1.4, ls=(0, (5, 2)), zorder=2))
    ax.text(xg, gy + gh - 4.6, "Leakage guard", ha="center", va="center",
            fontsize=9, fontweight="bold", color=GRD_EC, zorder=3)
    ax.text(xg, gy + (gh - 8) / 2, "prompt scanned for summary-stat vocabulary;\n"
            "no values or summaries reach the model", ha="center", va="center",
            fontsize=7.6, color=BODY, linespacing=1.5, zorder=3)
    ax.annotate("", xy=(xg, MY - 3.2), xytext=(xg, gy + gh), zorder=1,
                arrowprops=dict(arrowstyle="-|>", lw=1.3, color=GRD_EC,
                                ls=(0, (5, 2)), shrinkA=0, shrinkB=0, mutation_scale=12))

    # the four prior conditions — what the "prior block" varies over ---------
    cx4, cy, cw, ch = xs[3] + W / 2, 45, 34, 8
    ax.add_patch(FancyBboxPatch((cx4 - cw / 2, cy), cw, ch, boxstyle="round,pad=0,rounding_size=1.8",
                                fc="#eef4fb", ec=ACC_EC, lw=1.1, zorder=2))
    ax.annotate("", xy=(cx4, YB), xytext=(cx4, cy + ch), zorder=1,
                arrowprops=dict(arrowstyle="-", lw=1.0, color=ACC_EC, ls=(0, (4, 2))))
    ax.text(cx4, cy + ch - 2.8, "prior block: the only thing that changes", ha="center",
            va="center", fontsize=7.4, color=ACC_TC, fontweight="bold", zorder=3)
    ax.text(cx4, cy + 2.9, "vague   ·   coef-only   ·   variogram-only   ·   both",
            ha="center", va="center", fontsize=7.8, color=BODY, zorder=3)

    # title + legend --------------------------------------------------------
    ax.text(50.5, 95, "Controlled prior-only design", ha="center", va="center",
            fontsize=13.5, fontweight="bold", color="#1a1a1a")
    ax.text(50.5, 90, "the LLM supplies priors from text; the data, model and sampler "
            "are identical across all conditions", ha="center", va="center",
            fontsize=9.5, color=MUTE, style="italic")

    def swatch(x, fc, ec, label):
        ax.add_patch(Rectangle((x, 15.6), 2.4, 2.4, fc=fc, ec=ec, lw=1.3, zorder=3))
        ax.text(x + 3.3, 16.8, label, ha="left", va="center", fontsize=8, color=BODY)
    swatch(9, ACC_FC, ACC_EC, "LLM-supplied: the only component that varies")
    swatch(55, NEU_FC, NEU_EC, "fixed across all conditions: data · model · sampler · evaluation")

    _save(fig, "method_schematic.png")


# ------------------------------------------------------------------ analysis figures (pro)
# network display order + covariate-strength subtitle (urban added)
PANELS = [
    ("main",  "Greater Himalaya", "strong · r = −0.92"),
    ("andes", "Andes",            "medium · r = −0.50"),
    ("urban", "NE megalopolis (urban)", "medium · r = −0.68"),
    ("prcp",  "Himalaya precip.", "weak · r = −0.21"),
]
COND_STYLE = {
    "vague":              dict(color=C_VAGUE, marker="o", ls="-",  label="vague (control)"),
    "coef_frontier":      dict(color=C_COEF,  marker="o", ls="-",  label="coef (frontier)"),
    "both_frontier":      dict(color=C_BOTH,  marker="o", ls="-",  label="both (frontier)"),
    "variogram_frontier": dict(color=C_VARIO, marker="s", ls=":",  label="variogram (frontier)"),
    "coef_local":         dict(color=C_LOCAL, marker="^", ls="--", label="coef (local)"),
}
EXTENT_KM = {"main": 2500, "andes": 4500, "prcp": 2500, "urban": 900}


def _series(summ, ds, cond, tier=None):
    q = summ[(summ.dataset == ds) & (summ.predictor == "bayesian_kriging") & (summ.condition == cond)]
    if tier is not None:
        q = q[q.tier == tier]
    return q.groupby("density").mean(numeric_only=True).reset_index().sort_values("density")


def range_scatter_pro(pq):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (ds, name, cov) in zip(axes.ravel(), PANELS):
        d = pq[(pq.dataset == ds) & pq.range_km.notna()]
        ext = EXTENT_KM[ds]
        ax.axhspan(ext * 0.1, ext, color="#dff0d8", alpha=0.55, zorder=0)
        ax.axhline(ext, color="#5a8f3c", lw=1, ls="--", zorder=1)
        ax.text(0.97, ext * 1.15, "domain scale", color="#4a7a30", fontsize=8,
                ha="right", transform=ax.get_yaxis_transform())
        for tlab, x0, col in [("local", 0.0, "#8a8a8a"), ("frontier", 1.0, C_BOTH)]:
            s = d[d.tier == tlab]
            if not len(s):
                continue
            xs = x0 + rng.uniform(-0.12, 0.12, len(s))
            ax.scatter(xs, s.range_km, c=col, s=48, edgecolor="#2a2a2a", linewidth=0.4,
                       alpha=0.9, zorder=3)
        loc = d[d.tier == "local"]
        if len(loc):                       # annotate the largest local range
            wm = loc.loc[loc.range_km.idxmax()]
            ax.annotate(f"{wm.model.split('/')[-1]}\n{wm.range_km:,.0f} km",
                        xy=(0, wm.range_km), xytext=(0.18, wm.range_km),
                        fontsize=7.5, color="#a33", va="center",
                        arrowprops=dict(arrowstyle="-", color="#a33", lw=0.6))
        ax.set_yscale("log"); ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["local\n(2–35B)", "frontier\n(2026)"])
        ax.set_title(f"{name}  ({cov})", loc="left", fontsize=10.5)
        ax.grid(True, axis="y", color="#ececec", zorder=0)
    axes[0, 0].set_ylabel("elicited variogram range (km, log)")
    axes[1, 0].set_ylabel("elicited variogram range (km, log)")
    fig.tight_layout()
    _save(fig, "range_scatter.png")


def _density_panel(ax, summ, ds, metric, annotate_blowup=False):
    def plot(cond_key, cond, tier=None):
        s = _series(summ, ds, cond, tier)
        if not len(s):
            return
        st = COND_STYLE[cond_key]
        ycol = f"{metric}_mean"; ecol = f"{metric}_std"
        ax.plot(s.density, s[ycol], color=st["color"], marker=st["marker"], ls=st["ls"],
                ms=5, lw=1.8, label=st["label"], zorder=3)
        # band only for the control and the headline condition, to keep the panel legible
        if ecol in s and metric == "rmse" and cond_key in ("vague", "both_frontier"):
            lo = np.clip(s[ycol] - s[ecol], 0, None)
            ax.fill_between(s.density, lo, s[ycol] + s[ecol],
                            color=st["color"], alpha=0.15, lw=0, zorder=1)
    plot("vague", "vague")
    plot("coef_frontier", "llm_coef", "frontier")
    plot("both_frontier", "llm_both", "frontier")
    plot("variogram_frontier", "llm_variogram", "frontier")
    plot("coef_local", "llm_coef", "local")
    if annotate_blowup and ds == "prcp" and metric == "rmse":
        sl = _series(summ, "prcp", "llm_coef", "local")
        if len(sl):
            pk = sl.loc[sl["rmse_mean"].idxmax()]
            ax.annotate("small-model\nconfident error", xy=(pk.density, pk["rmse_mean"]),
                        xytext=(pk.density + 8, pk["rmse_mean"]), fontsize=7.5, color="#b9560f",
                        arrowprops=dict(arrowstyle="-|>", color="#b9560f", lw=1))


def density_grid(summ, metric, ylabel, fname, nominal=None):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), sharex=True)
    for ax, (ds, name, cov) in zip(axes.ravel(), PANELS):
        if nominal is not None:
            ax.axhspan(nominal - 0.02, nominal + 0.02, color="#f0f0f0", zorder=0)
            ax.axhline(nominal, color="#888", lw=1, ls="--", zorder=1)
        _density_panel(ax, summ, ds, metric, annotate_blowup=True)
        ax.set_title(f"{name}  ({cov})", loc="left", fontsize=10.5)
        ax.grid(True, color="#f2f2f2", zorder=0)
        ax.set_xticks([10, 25, 50, 100])
        if metric == "rmse":
            ax.set_ylim(bottom=0)
    axes[1, 0].set_xlabel("observation density (%)"); axes[1, 1].set_xlabel("observation density (%)")
    axes[0, 0].set_ylabel(ylabel); axes[1, 0].set_ylabel(ylabel)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, fname)


def pit_pro():
    f = ROOT / "results" / "convergence" / "pit.csv"
    if not f.exists():
        print("[pit] no pit.csv"); return
    import pandas as pd
    d = pd.read_csv(f)
    conds = [("vague", "vague prior", "#7f8c8d"), ("llm_both", "frontier  llm_both", C_BOTH)]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True)
    bins = np.linspace(0, 1, 11)
    for ax, (c, lab, col) in zip(axes, conds):
        s = d[d.condition == c]["pit"].values
        ax.hist(s, bins=bins, density=True, color=col, alpha=0.85, edgecolor="white")
        ax.axhline(1.0, color="#c0392b", ls="--", lw=1.2)
        ax.set_title(f"{lab}   (n = {len(s):,})", fontsize=10.5, loc="left")
        ax.set_xlabel("PIT value"); ax.grid(True, axis="y", color="#f0f0f0")
    axes[0].set_ylabel("density")
    fig.tight_layout()
    _save(fig, "pit_hist.png")


SIM_TRUE_RANGE_KM = 150.0  # src/sim.py TRUE_RANGE, ~ Himalaya residual field scale


def sim_curve_pro(pq):
    """Range-misspecification damage curve. Legend lives OUTSIDE the axes and the tier
    rug-marks sit in a dedicated top headroom band, so nothing overlaps the data."""
    import pandas as pd
    f = ROOT / "results" / "paper" / "tbl_sim.csv"
    if not f.exists():
        print("[sim] no tbl_sim.csv"); return
    st = pd.read_csv(f)
    ncol = {12: C_COEF, 30: C_LOCAL, 60: C_VARIO}

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    # harmless-valley shading (order-of-magnitude around ratio 1)
    ax.axvspan(1 / 3, 3, color="#eaf4ea", zorder=0)
    for n in sorted(st.n_obs.unique()):
        s = st[st.n_obs == n].sort_values("ratio")
        ax.plot(s.ratio, s.rmse, marker="o", ms=5, lw=1.9, color=ncol[n], zorder=3)
        if "rmse_sem" in s:
            ax.fill_between(s.ratio, s.rmse - s.rmse_sem, s.rmse + s.rmse_sem,
                            color=ncol[n], alpha=0.16, lw=0, zorder=1)
        ax.axhline(s["vague_rmse"].iloc[0], color=ncol[n], ls=":", lw=1, alpha=0.55, zorder=1)
    ax.set_xscale("log")
    ax.axvline(1.0, color="#999", lw=0.9, alpha=0.7, zorder=1)
    ax.set_xlabel("elicited range / true range  (misspecification ratio)")
    ax.set_ylabel("held-out RMSE")

    # add top headroom, then place the two tier rug-marks inside it (no overlap)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + (y1 - y0) * 0.24)
    y0, y1 = ax.get_ylim()
    main_pq = pq[(pq.dataset == "main") & pq.range_km.notna()]
    for tr, col, yf in [("local", "#8a8a8a", 0.94), ("frontier", C_BOTH, 0.87)]:
        rr = main_pq[main_pq.tier == tr].range_km / SIM_TRUE_RANGE_KM
        if len(rr):
            yy = y0 + (y1 - y0) * yf
            ax.scatter(rr, np.full(len(rr), yy), marker="|", s=240, color=col, lw=1.4, zorder=4)
            ax.scatter([rr.median()], [yy], marker="v", s=70, color=col,
                       edgecolor="white", lw=0.6, zorder=5)
    ax.text(0.99, 0.985, "elicited ranges placed on the curve", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="#666")

    handles = [Line2D([0], [0], color=ncol[n], marker="o", lw=1.9, label=f"n = {n}")
               for n in sorted(st.n_obs.unique())]
    handles += [
        Line2D([0], [0], color="#888", ls=":", lw=1, label="vague baseline (per n)"),
        Line2D([0], [0], color="#8a8a8a", marker="|", ls="none", ms=11, mew=1.6, label="local elicited"),
        Line2D([0], [0], color=C_BOTH, marker="|", ls="none", ms=11, mew=1.6, label="frontier elicited"),
        Line2D([0], [0], marker="s", ls="none", ms=9, mfc="#eaf4ea", mec="#cfe3cf",
               label="harmless valley"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=8.5, handlelength=1.6)
    ax.set_title("Range-misspecification damage curve", loc="left")
    fig.tight_layout()
    _save(fig, "sim_misspec_curve.png")


def analysis_figs():
    sys.path.insert(0, str(SRC))
    import pandas as pd
    import analyze_paper as ap
    summ = ap.load_summary()
    # Read the cached prior-quality table (authoritative; recomputing from the raw
    # elicitation JSON is unnecessary here and depends on the full cache being present).
    pq = pd.read_csv(ROOT / "results" / "paper" / "tbl_prior_quality.csv")
    range_scatter_pro(pq)
    density_grid(summ, "rmse", "held-out RMSE", "rmse_vs_density_triptych.png")
    density_grid(summ, "coverage90", "90% interval coverage",
                 "coverage90_vs_density_triptych.png", nominal=0.90)
    pit_pro()
    sim_curve_pro(pq)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("map", "all"):
        study_area_map()
    if what in ("schematic", "all"):
        method_schematic()
    if what in ("analysis", "all"):
        analysis_figs()
