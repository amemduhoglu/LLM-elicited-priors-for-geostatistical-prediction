#!/usr/bin/env python
"""analyze_paper.py — definitive results pack for the Q1 paper (Phase A of paper/PLAN.md).

Reads results/summary.csv, results/cells_long.csv and results/elicit/**/consensus.json and
emits paper-ready tables + figures to results/paper/. Tier-split (local vs frontier) is the
organizing axis of the new thesis: frontier coefficient priors help under weak covariates,
the variogram harm was a small-model capability artifact.

Usage: .venv/bin/python src/analyze_paper.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "paper"
FIGS = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

DATASETS = ["main", "andes", "prcp", "urban"]
COVAR = {"main": "strong (r=-0.92)", "andes": "medium (r=-0.50)", "prcp": "weak (r=-0.21)",
         "urban": "medium (r=-0.68)"}
DENS = [10, 25, 50, 100]
OUTLIER = "gemma4:e2b"  # capability-floor; log-transform exp() blowup on prcp. Reported separately.


def is_frontier(m) -> bool:
    return isinstance(m, str) and "/" in m


def tier(m) -> str:
    return "frontier" if is_frontier(m) else ("none" if m == "none" or pd.isna(m) else "local")


def load_summary(drop_outlier=True) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "results" / "summary.csv")
    d["tier"] = d["elicit_model"].map(tier)
    d["base_model"] = d["elicit_model"].astype(str).str.replace("__v3", "", regex=False)
    if drop_outlier:
        d = d[~d["base_model"].str.startswith(OUTLIER)]
    return d


def load_cells(drop_outlier=True) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "results" / "cells_long.csv")
    d["tier"] = d["elicit_model"].map(tier)
    d["base_model"] = d["elicit_model"].astype(str).str.replace("__v3", "", regex=False)
    if drop_outlier:
        d = d[~d["base_model"].str.startswith(OUTLIER)]
    return d


# ---------------------------------------------------------------- A1: prior-quality table
def tbl_prior_quality() -> pd.DataFrame:
    rows = []
    for cj in sorted((ROOT / "results" / "elicit").rglob("consensus.json")):
        try:
            c = json.loads(cj.read_text())
        except Exception:
            continue
        ds = cj.parent.parent.name
        if ds not in DATASETS:
            continue
        model = c.get("model", cj.parent.name)
        proto = c.get("prompt_version", "v3" if "__v3" in cj.parent.name else "v2")
        cons = c.get("consensus", {})
        coef = cons.get("coefficients", {})
        # first covariate coefficient
        ck = next(iter(coef), None)
        coef_mean = coef.get(ck, {}).get("mean") if ck else None
        rng = cons.get("variogram", {}).get("range", {})
        rng_km = (rng.get("mean") / 1000.0) if rng.get("mean") is not None else None
        rows.append({
            "dataset": ds, "model": model, "protocol": proto, "tier": tier(model),
            "coef": coef_mean, "range_km": rng_km,
            "range_sd_km": (rng.get("sd") / 1000.0) if rng.get("sd") is not None else None,
        })
    df = pd.DataFrame(rows).sort_values(["dataset", "tier", "model", "protocol"])
    df.to_csv(OUT / "tbl_prior_quality.csv", index=False)
    return df


# ---------------------------------------------------------------- A1: main results (tier-split)
METRICS = ["rmse_mean", "mae_mean", "crps_mean", "pit_ks_mean", "coverage90_mean", "interval_width_mean"]


def tbl_main_results(summ: pd.DataFrame) -> pd.DataFrame:
    bk = summ[summ.predictor == "bayesian_kriging"].copy()
    recs = []
    for ds in DATASETS:
        for dens in DENS:
            sub = bk[(bk.dataset == ds) & (bk.density == dens)]
            # tier-agnostic conditions (no LLM model attached): vague + the PC-prior baseline
            for cond in ["vague", "pc_range"]:
                g = sub[sub.condition == cond]
                if len(g):
                    recs.append(_agg_row(ds, dens, cond, "-", g))
            # LLM-bearing conditions, split by capability tier (hybrid = llm_coef + PC range)
            for cond in ["llm_coef", "llm_variogram", "llm_both", "hybrid"]:
                for tr in ["local", "frontier"]:
                    g = sub[(sub.condition == cond) & (sub.tier == tr)]
                    if len(g):
                        recs.append(_agg_row(ds, dens, cond, tr, g))
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "tbl_main_results.csv", index=False)
    return df


def _agg_row(ds, dens, cond, tr, g):
    row = {"dataset": ds, "density": dens, "condition": cond, "tier": tr, "n_models": g.elicit_model.nunique()}
    for m in METRICS:
        row[m.replace("_mean", "")] = g[m].mean()
    return row


# ---------------------------------------------------------------- A1: significance (Wilcoxon)
def tbl_significance(cells: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import wilcoxon
    bk = cells[(cells.predictor == "bayesian_kriging") & (cells.status == "ok")].copy()
    recs = []
    for ds in DATASETS:
        for dens in DENS:
            sub = bk[(bk.dataset == ds) & (bk.density == dens)]
            vg = (sub[sub.condition == "vague"]
                  .groupby(["seed", "fold"])[["rmse", "crps"]].mean())

            def _pair(cond, tr, g):
                gm = g.groupby(["seed", "fold"])[["rmse", "crps"]].mean()
                j = vg.join(gm, lsuffix="_v", rsuffix="_c", how="inner").dropna()
                if len(j) < 6:
                    return None
                rec = {"dataset": ds, "density": dens, "condition": cond, "tier": tr, "n_pairs": len(j)}
                for met in ["rmse", "crps"]:
                    d_med = float(np.median(j[f"{met}_c"] - j[f"{met}_v"]))
                    try:
                        p = float(wilcoxon(j[f"{met}_c"], j[f"{met}_v"]).pvalue)
                    except ValueError:
                        p = float("nan")
                    rec[f"{met}_median_delta"] = d_med
                    rec[f"{met}_p"] = p
                    rec[f"{met}_effect"] = ("better" if d_med < 0 else "worse") if p < 0.05 else "ns"
                return rec

            # tier-agnostic PC-prior baseline (no LLM model)
            g = sub[sub.condition == "pc_range"]
            if len(g):
                r = _pair("pc_range", "-", g)
                if r:
                    recs.append(r)
            # LLM-bearing conditions, tier-split (hybrid = llm_coef + PC range)
            for cond in ["llm_coef", "llm_variogram", "llm_both", "hybrid"]:
                for tr in ["local", "frontier"]:
                    g = sub[(sub.condition == cond) & (sub.tier == tr)]
                    if len(g):
                        r = _pair(cond, tr, g)
                        if r:
                            recs.append(r)
    df = pd.DataFrame(recs)
    # WP4: multiple-comparison control. Each elicited-condition-vs-vague test is one member of
    # a large family; report Benjamini-Hochberg FDR-adjusted p alongside the raw p so the
    # headline survives multiplicity. Adjust within each metric across the whole family.
    for met in ["rmse", "crps"]:
        col = f"{met}_p"
        if col in df.columns:
            df[f"{met}_p_bh"] = _bh_adjust(df[col].to_numpy())
            df[f"{met}_sig_bh"] = (df[f"{met}_p_bh"] < 0.05)
    df.to_csv(OUT / "tbl_significance.csv", index=False)
    return df


def _bh_adjust(p):
    """Benjamini-Hochberg FDR adjustment; NaN-safe, no external dependency."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    idx = np.where(~np.isnan(p))[0]
    if idx.size == 0:
        return out
    pv = p[idx]
    order = np.argsort(pv)
    ranked = pv[order]
    m = pv.size
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]   # enforce monotonicity
    adj = np.clip(adj, 0, 1)
    res = np.empty(m)
    res[order] = adj
    out[idx] = res
    return out


# ---------------------------------------------------------------- A1: classical baselines
def tbl_classical(summ: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for ds in DATASETS:
        for dens in DENS:
            sub = summ[(summ.dataset == ds) & (summ.density == dens)]
            def grab(pred, cond=None, tr=None):
                q = sub[sub.predictor == pred]
                if cond is not None:
                    q = q[q.condition == cond]
                if tr is not None:
                    q = q[q.tier == tr]
                return q.rmse_mean.mean() if len(q) else np.nan
            recs.append({
                "dataset": ds, "density": dens,
                "OK": grab("ordinary_kriging"),
                "RK": grab("regression_kriging"),
                "RF_RK": grab("rf_residual_kriging"),
                "bayes_vague": grab("bayesian_kriging", "vague"),
                "bayes_pc_range": grab("bayesian_kriging", "pc_range"),
                "bayes_front_coef": grab("bayesian_kriging", "llm_coef", "frontier"),
                "bayes_front_both": grab("bayesian_kriging", "llm_both", "frontier"),
                "bayes_front_hybrid": grab("bayesian_kriging", "hybrid", "frontier"),
            })
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "tbl_classical.csv", index=False)
    return df


# ---------------------------------------------------------------- A2: figures
def fig_metric_vs_density(main_tbl: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = [  # (condition, tier, label, style)
        ("vague", "-", "vague", dict(color="black", lw=2)),
        ("pc_range", "-", "PC range", dict(color="tab:brown", ls="--", alpha=.8)),
        ("llm_coef", "local", "coef (local)", dict(color="tab:orange", ls="--", alpha=.7)),
        ("llm_coef", "frontier", "coef (frontier)", dict(color="tab:red", lw=2)),
        ("llm_both", "frontier", "both (frontier)", dict(color="tab:blue", lw=2)),
        ("hybrid", "frontier", "hybrid (frontier)", dict(color="tab:purple", lw=2, ls="-.")),
        ("llm_variogram", "frontier", "variogram (frontier)", dict(color="tab:green", ls=":")),
    ]
    for metric, ylab in [("rmse", "RMSE"), ("crps", "CRPS"), ("coverage90", "90% coverage")]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
        for ax, ds in zip(axes, DATASETS):
            for cond, tr, lbl, st in series:
                q = main_tbl[(main_tbl.dataset == ds) & (main_tbl.condition == cond)
                             & (main_tbl.tier == tr)].sort_values("density")
                if len(q):
                    ax.plot(q.density, q[metric], marker="o", label=lbl, **st)
            if metric == "coverage90":
                ax.axhline(0.90, color="gray", ls="-", lw=.8, alpha=.5)
            ax.set_title(f"{ds} — {COVAR[ds]}")
            ax.set_xlabel("density (%)")
        axes[0].set_ylabel(ylab)
        axes[0].legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(FIGS / f"{metric}_vs_density_triptych.png", dpi=130)
        plt.close(fig)


def fig_range_scatter(pq: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        sub = pq[(pq.dataset == ds) & pq.range_km.notna()]
        for i, tr in enumerate(["local", "frontier"]):
            g = sub[sub.tier == tr]
            x = np.full(len(g), i) + np.random.uniform(-.12, .12, len(g))
            ax.scatter(x, g.range_km, alpha=.7,
                       color="tab:gray" if tr == "local" else "tab:red", s=40)
        ax.set_yscale("log")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["local", "frontier"])
        ax.set_title(f"{ds} — {COVAR[ds]}")
    axes[0].set_ylabel("elicited variogram range (km, log)")
    fig.suptitle("Variogram-range scatter collapses for frontier models")
    fig.tight_layout()
    fig.savefig(FIGS / "range_scatter.png", dpi=130)
    plt.close(fig)


SIM_TRUE_RANGE_KM = 150.0  # src/sim.py TRUE_RANGE, chosen ~ Himalaya residual field scale


def tbl_sim() -> pd.DataFrame:
    import glob
    rows = [json.loads(Path(f).read_text()) for f in glob.glob(str(ROOT / "results/sim/cells/*.json"))]
    d = pd.DataFrame([r for r in rows if r.get("status") == "ok"])
    g = d[d.condition == "ratio"].groupby(["n_obs", "ratio"])
    rat = g[["rmse", "crps", "coverage90"]].mean().reset_index()
    # across-replicate spread for the shaded band (SEM = std / sqrt(n_reps))
    sem = (g["rmse"].std() / np.sqrt(g["rmse"].count())).reset_index(name="rmse_sem")
    rat = rat.merge(sem, on=["n_obs", "ratio"])
    vague = d[d.condition == "vague"].groupby("n_obs")[["rmse", "crps", "coverage90"]].mean()
    vague.columns = [f"vague_{c}" for c in vague.columns]
    out = rat.merge(vague, on="n_obs")
    out.to_csv(OUT / "tbl_sim.csv", index=False)
    return out, d


def fig_sim_curve(sim_tbl: pd.DataFrame, pq: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {12: "tab:red", 30: "tab:orange", 60: "tab:green"}
    for n in sorted(sim_tbl.n_obs.unique()):
        s = sim_tbl[sim_tbl.n_obs == n].sort_values("ratio")
        ax.plot(s.ratio, s.rmse, marker="o", color=colors[n], label=f"n={n} (prior-centred range)")
        if "rmse_sem" in s:
            ax.fill_between(s.ratio, s.rmse - s.rmse_sem, s.rmse + s.rmse_sem,
                            color=colors[n], alpha=0.18, lw=0)
        vb = s[f"vague_rmse"].iloc[0]
        ax.axhline(vb, color=colors[n], ls=":", lw=1, alpha=.6)
    ax.set_xscale("log")
    ax.set_xlabel("elicited range / true range  (misspecification ratio)")
    ax.set_ylabel("held-out RMSE")
    ax.axvline(1.0, color="gray", lw=.8, alpha=.5)
    # place tier median elicited ratios (main network) on the damage curve
    main_pq = pq[(pq.dataset == "main") & pq.range_km.notna()]
    ymax = ax.get_ylim()[1]
    for tr, col, yfrac in [("local", "dimgray", 0.96), ("frontier", "tab:blue", 0.90)]:
        rr = (main_pq[main_pq.tier == tr].range_km / SIM_TRUE_RANGE_KM)
        if len(rr):
            ax.scatter(rr, np.full(len(rr), ymax * yfrac), marker="|", s=300,
                       color=col, label=f"{tr} elicited ranges")
            ax.scatter([rr.median()], [ymax * yfrac], marker="v", s=70, color=col, zorder=5)
    ax.legend(fontsize=8, loc="upper center", ncol=2)
    ax.set_title("Range-misspecification damage curve; local elicited ranges scatter onto the\n"
                 "harmful tails, frontier ranges cluster near the harmless valley (ratio ~1)")
    fig.tight_layout()
    fig.savefig(FIGS / "sim_misspec_curve.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------ WP1: variogram elicitation vs PC-prior baseline
def tbl_variogram_vs_pc(summ: pd.DataFrame) -> pd.DataFrame:
    """The WP1 headline: is frontier variogram elicitation any better than a principled,
    LLM-free PC-prior range? Columns place vague, pc_range, frontier llm_variogram, frontier
    hybrid and frontier llm_both side by side per dataset x density."""
    bk = summ[summ.predictor == "bayesian_kriging"].copy()
    recs = []
    for ds in DATASETS:
        for dens in DENS:
            sub = bk[(bk.dataset == ds) & (bk.density == dens)]
            def grab(cond, tr=None, met="rmse_mean"):
                q = sub[sub.condition == cond]
                if tr is not None:
                    q = q[q.tier == tr]
                return float(q[met].mean()) if len(q) else np.nan
            for met, lab in [("rmse_mean", "rmse"), ("crps_mean", "crps"), ("coverage90_mean", "cov90")]:
                recs.append({
                    "dataset": ds, "density": dens, "metric": lab,
                    "vague": grab("vague", None, met),
                    "pc_range": grab("pc_range", None, met),
                    "front_variogram": grab("llm_variogram", "frontier", met),
                    "front_hybrid": grab("hybrid", "frontier", met),
                    "front_both": grab("llm_both", "frontier", met),
                })
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "tbl_variogram_vs_pc.csv", index=False)
    return df


# ------------------------------------------------------- WP5: coverage with bootstrap 95% CI
def tbl_calibration_ci(cells: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    """90% interval coverage with a fold-bootstrap 95% CI, per dataset x density for the
    decision-relevant conditions. Gives the calibration claims an uncertainty band (WP5)."""
    rng = np.random.default_rng(0)
    bk = cells[(cells.predictor == "bayesian_kriging") & (cells.status == "ok")].copy()
    recs = []
    conds = [("vague", None), ("pc_range", None), ("llm_both", "frontier"), ("hybrid", "frontier")]
    for ds in DATASETS:
        for dens in DENS:
            sub = bk[(bk.dataset == ds) & (bk.density == dens)]
            for cond, tr in conds:
                g = sub[sub.condition == cond]
                if tr is not None:
                    g = g[g.tier == tr]
                vals = g["coverage90"].dropna().to_numpy()
                if vals.size < 6:
                    continue
                boot = rng.choice(vals, size=(n_boot, vals.size), replace=True).mean(axis=1)
                recs.append({
                    "dataset": ds, "density": dens, "condition": cond, "tier": tr or "-",
                    "n_cells": int(vals.size), "coverage90": float(vals.mean()),
                    "ci_lo": float(np.quantile(boot, 0.025)),
                    "ci_hi": float(np.quantile(boot, 0.975)),
                })
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "tbl_calibration_ci.csv", index=False)
    return df


def main():
    summ = load_summary()
    cells = load_cells()
    pq = tbl_prior_quality()
    mt = tbl_main_results(summ)
    sig = tbl_significance(cells)
    cl = tbl_classical(summ)
    vpc = tbl_variogram_vs_pc(summ)
    cal = tbl_calibration_ci(cells)
    sim_tbl, _ = tbl_sim()
    # NOTE: the paper figures (range_scatter, *_vs_density_triptych, sim_misspec_curve)
    # are produced by src/figures_pro.py, the single authoritative generator. The generic
    # writers below are kept for ad-hoc inspection but disabled here to avoid overwriting
    # the publication-grade versions. Run `python src/figures_pro.py all` to (re)build them.
    # fig_metric_vs_density(mt); fig_range_scatter(pq); fig_sim_curve(sim_tbl, pq)
    print("== wrote results/paper/ ==")
    for f in sorted(OUT.glob("*.csv")):
        print("  table:", f.name)
    for f in sorted(FIGS.glob("*.png")):
        print("  figure:", f.name)
    # headline significance peek
    print("\n== frontier vs vague, RMSE @ d10 (Wilcoxon) ==")
    h = sig[(sig.density == 10) & (sig.tier == "frontier")][
        ["dataset", "condition", "n_pairs", "rmse_median_delta", "rmse_p", "rmse_effect"]]
    print(h.to_string(index=False))


if __name__ == "__main__":
    np.random.seed(0)
    main()
