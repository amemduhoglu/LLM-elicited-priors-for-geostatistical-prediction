#!/usr/bin/env python
"""report_extras.py — paper-ready artifacts for the C&G revision (review #3/#4 + urban).

Consumes the convergence subset and (when present) the urban bayes cells, emitting:
  results/paper/tbl_convergence.csv   MCMC R-hat / ESS / divergence summary  (#3)
  results/paper/figures/pit_hist.png  pooled PIT histogram, frontier vs vague (#4)
  results/paper/tbl_urban.csv         urban headline RMSE/CRPS/coverage table  (#1)
and prints a markdown digest for pasting into the manuscript.

Usage: .venv/bin/python src/report_extras.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "paper"
FIGS = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


def convergence_table():
    f = ROOT / "results" / "convergence" / "summary.csv"
    if not f.exists():
        print("[conv] no summary yet"); return None
    d = pd.read_csv(f)
    ok = d[d["status"] == "ok"] if "status" in d else d
    print(f"\n=== MCMC convergence ({len(ok)} re-fit cells) ===")
    g = ok.groupby(["dataset", "condition"]).agg(
        n=("max_rhat", "size"),
        rhat_median=("max_rhat", "median"), rhat_max=("max_rhat", "max"),
        ess_bulk_min=("min_ess_bulk", "min"), ess_bulk_median=("min_ess_bulk", "median"),
        div_total=("n_divergent", "sum")).reset_index()
    g.to_csv(OUT / "tbl_convergence.csv", index=False)
    print(g.round(3).to_string(index=False))
    print(f"\nOverall: max R-hat={ok['max_rhat'].max():.3f}, "
          f"min ESS_bulk={ok['min_ess_bulk'].min():.0f}, "
          f"divergent draws={int(ok['n_divergent'].sum())}/{len(ok)} fits "
          f"({100*(ok['n_divergent']>0).mean():.1f}% of fits had any).")
    return g


def pit_histogram():
    f = ROOT / "results" / "convergence" / "pit.csv"
    if not f.exists():
        print("[pit] no pit.csv yet"); return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_csv(f)
    conds = [("vague", "vague prior"), ("llm_both", "frontier llm_both")]
    fig, axes = plt.subplots(1, len(conds), figsize=(9, 3.4), sharey=True)
    bins = np.linspace(0, 1, 11)
    for ax, (c, lab) in zip(np.atleast_1d(axes), conds):
        s = d[d["condition"] == c]["pit"].values
        ax.hist(s, bins=bins, density=True, color="tab:blue", alpha=0.75,
                edgecolor="white")
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.set_title(f"{lab}  (n={len(s)})", fontsize=10)
        ax.set_xlabel("PIT value")
    np.atleast_1d(axes)[0].set_ylabel("density")
    fig.suptitle("PIT histograms (pooled held-out points; flat = calibrated)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "pit_hist.png", dpi=130)
    print(f"[pit] wrote pit_hist.png from {len(d)} pooled points")


def urban_table():
    f = ROOT / "results" / "summary.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    if "dataset" not in d or "urban" not in set(d["dataset"]):
        print("[urban] no urban bayes cells yet — run run_urban.sh"); return
    u = d[(d["dataset"] == "urban")].copy()
    u["tier"] = u["elicit_model"].map(
        lambda m: "frontier" if isinstance(m, str) and "/" in m else
        ("none" if m == "none" or pd.isna(m) else "local"))
    # headline: 10% density, vague vs frontier conditions
    sub = u[(u["density"] == 10) & (u["predictor"] == "bayesian_kriging")]
    fr = sub[(sub["tier"] == "frontier") | (sub["condition"] == "vague")]
    tab = (fr.groupby("condition")[["rmse_mean", "crps_mean", "coverage90_mean"]]
           .mean().round(3))
    tab.to_csv(OUT / "tbl_urban.csv")
    print("\n=== URBAN (NE megalopolis) bayes @10% density ===")
    print(tab.to_string())


if __name__ == "__main__":
    convergence_table()
    # pit_hist.png is produced by src/figures_pro.py (authoritative). Disabled here to
    # avoid overwriting the publication-grade version with this generic one.
    # pit_histogram()
    urban_table()
