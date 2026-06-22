#!/usr/bin/env python
"""revision_tables.py — SELF-CONTAINED results pack for the reviewer-revision additions.

Reads results/cells_long.csv (raw per-(seed,fold) cells, the ground truth) and
results/elicit/**/consensus.json directly. Does NOT touch analyze_paper.py / summary.csv,
so it cannot contaminate the existing local/frontier tables. Emits four new tables and a
human-readable morning report to results/paper/revision/.

Why self-contained (see paper/INTEGRATION_PLAN.md FAZ 4):
  - open-weight tags also contain '/', so the old tier() would mislabel them as frontier;
    here tier() is 3-way (frontier vendors / openweight / local).
  - summary.csv does not separate width_scale, so width=2,3 would pollute frontier means;
    here we read cells_long.csv and filter width_scale explicitly.

Significance methodology matches analyze_paper.tbl_significance: average the constituent
models within a tier per (seed,fold), paired Wilcoxon vs vague, Benjamini-Hochberg FDR per
metric across the family. Only width_scale==1.0 cells enter the predictive tables.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CELLS = ROOT / "results" / "cells_long.csv"
ELIC = ROOT / "results" / "elicit"
OUT = ROOT / "results" / "paper" / "revision"
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ["main", "andes", "prcp", "urban"]
DENS = [10, 25, 50, 100]
OUTLIER = "gemma4:e2b"

FRONTIER_PREFIXES = ("anthropic/", "openai/", "google/gemini")
OW_FAMILIES = ("deepseek/", "nvidia/", "z-ai/")  # the 6-model open-weight tier


def tier(m) -> str:
    if not isinstance(m, str) or m == "none" or pd.isna(m):
        return "none"
    bm = m.replace("__v3", "")
    if bm.startswith(FRONTIER_PREFIXES):
        return "frontier"
    if bm.startswith(OW_FAMILIES):
        return "openweight"
    if "/" in bm:
        return "other_api"          # precision_control twins etc. (no bayes cells expected)
    return "local"


def _bh(p):
    p = np.asarray(p, float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    idx = np.where(ok)[0]
    ps = p[idx]
    order = np.argsort(ps)
    n = len(ps)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, ps[i] * n / (rank + 1))
        adj[i] = prev
    out[idx] = adj
    return out


def load_cells():
    d = pd.read_csv(CELLS)
    d["tier"] = d["elicit_model"].map(tier)
    d["base_model"] = d["elicit_model"].astype(str).str.replace("__v3", "", regex=False)
    d = d[(d.predictor == "bayesian_kriging") & (d.status == "ok")]
    d = d[~d["base_model"].str.startswith(OUTLIER)]
    if "width_scale" not in d.columns:
        d["width_scale"] = 1.0
    d["width_scale"] = d["width_scale"].fillna(1.0)
    return d


# ---------------------------------------------------------------- 1. open-weight predictive + Wilcoxon
def tbl_openweight(cells):
    bk = cells[cells.width_scale == 1.0]
    from scipy.stats import wilcoxon
    rows = []
    for ds in DATASETS:
        for dens in DENS:
            sub = bk[(bk.dataset == ds) & (bk.density == dens)]
            vg = sub[sub.condition == "vague"].groupby(["seed", "fold"])[["rmse", "crps", "coverage90"]].mean()
            if not len(vg):
                continue
            for trname in ["openweight", "frontier"]:
                for cond in ["llm_coef", "llm_variogram", "llm_both"]:
                    g = sub[(sub.condition == cond) & (sub.tier == trname)]
                    if not len(g):
                        continue
                    gm = g.groupby(["seed", "fold"])[["rmse", "crps", "coverage90"]].mean()
                    j = vg.join(gm, lsuffix="_v", rsuffix="_c", how="inner").dropna()
                    if len(j) < 6:
                        continue
                    rec = {"dataset": ds, "density": dens, "tier": trname, "condition": cond,
                           "n_models": g.elicit_model.nunique(), "n_pairs": len(j),
                           "vague_rmse": round(j.rmse_v.mean(), 3), "cond_rmse": round(j.rmse_c.mean(), 3),
                           "vague_crps": round(j.crps_v.mean(), 3), "cond_crps": round(j.crps_c.mean(), 3),
                           "vague_cov90": round(j.coverage90_v.mean(), 3), "cond_cov90": round(j.coverage90_c.mean(), 3)}
                    for met in ["rmse", "crps"]:
                        dmed = float(np.median(j[f"{met}_c"] - j[f"{met}_v"]))
                        try:
                            p = float(wilcoxon(j[f"{met}_c"], j[f"{met}_v"]).pvalue)
                        except ValueError:
                            p = float("nan")
                        rec[f"{met}_dmed"] = round(dmed, 4)
                        rec[f"{met}_p"] = p
                    rows.append(rec)
    df = pd.DataFrame(rows)
    for met in ["rmse", "crps"]:
        if f"{met}_p" in df.columns:
            df[f"{met}_p_bh"] = _bh(df[f"{met}_p"].to_numpy())
            df[f"{met}_sig"] = df[f"{met}_p_bh"] < 0.05
    df.to_csv(OUT / "tbl_openweight.csv", index=False)
    return df


# ---------------------------------------------------------------- 2. quantization control (elicited ranges)
def tbl_quant_control():
    rows = []

    def rng(dirname, label, vs):
        f = ELIC / "main" / dirname / "consensus.json"
        if not f.exists():
            rows.append({"group": label, "model_dir": dirname, "range_km": None, "coef": None, "vs_local_km": vs})
            return
        c = json.load(open(f)).get("consensus")
        if not c:
            rows.append({"group": label, "model_dir": dirname, "range_km": None, "coef": None, "vs_local_km": vs})
            return
        rows.append({"group": label, "model_dir": dirname,
                     "range_km": round(c["variogram"]["range"]["mean"] / 1000, 1),
                     "coef": round(c["coefficients"].get("elev", {}).get("mean", float("nan")), 5),
                     "vs_local_km": vs})
    # local quant ladder (same qwen3.5:9b weights)
    rng("qwen3.5_9b", "local_quant_ladder Q4_K_M", "-")
    rng("qwen3.5_9b-q8_0", "local_quant_ladder Q8_0", "-")
    rng("qwen3.5_9b-bf16", "local_quant_ladder bf16", "-")
    # OpenRouter bf16 twins of disciplined locals
    rng("qwen_qwen3.5-9b", "bf16_twin (vs Q4 7km)", 7)
    rng("qwen_qwen3.5-27b", "bf16_twin (vs Q4 140km)", 140)
    rng("google_gemma-4-31b-it", "bf16_twin (vs Q4 400km)", 400)
    rng("google_gemma-4-26b-a4b-it", "bf16_twin (vs Q4 100km)", 100)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tbl_quant_control.csv", index=False)
    return df


# ---------------------------------------------------------------- 3. width / overconfidence sweep
def tbl_width_sweep(cells):
    fr = cells[cells.tier == "frontier"]
    rows = []
    for ds in ["andes", "prcp", "urban"]:
        for dens in DENS:
            for cond in ["llm_coef", "llm_variogram", "llm_both"]:
                for w in [1.0, 2.0, 3.0]:
                    g = fr[(fr.dataset == ds) & (fr.density == dens) &
                           (fr.condition == cond) & (fr.width_scale == w)]
                    if not len(g):
                        continue
                    rows.append({"dataset": ds, "density": dens, "condition": cond, "width": w,
                                 "rmse": round(g.rmse.mean(), 3), "crps": round(g.crps.mean(), 3),
                                 "cov90": round(g.coverage90.mean(), 3),
                                 "width_mean": round(g.interval_width.mean(), 3),
                                 "n": len(g)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tbl_width_sweep.csv", index=False)
    return df


# ---------------------------------------------------------------- 4. v3 vs v2 cross-network
def tbl_v3(cells):
    fr = cells[(cells.tier == "frontier") & (cells.width_scale == 1.0)]
    fr = fr.copy()
    fr["proto"] = np.where(fr.elicit_model.astype(str).str.contains("__v3"), "v3", "v2")
    rows = []
    for ds in DATASETS:
        for cond in ["llm_variogram", "llm_both"]:
            for proto in ["v2", "v3"]:
                g = fr[(fr.dataset == ds) & (fr.condition == cond) &
                       (fr.proto == proto) & (fr.density == 10)]
                if not len(g):
                    continue
                rows.append({"dataset": ds, "condition": cond, "protocol": proto, "density": 10,
                             "rmse": round(g.rmse.mean(), 3), "crps": round(g.crps.mean(), 3), "n": len(g)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tbl_v3_allnetworks.csv", index=False)
    return df


def main():
    cells = load_cells()
    ow = tbl_openweight(cells)
    qc = tbl_quant_control()
    ws = tbl_width_sweep(cells)
    v3 = tbl_v3(cells)
    lines = []
    lines.append("REVISION RESULTS PACK  (generated by revision_tables.py)")
    lines.append(f"cells_long rows used: {len(cells)}\n")
    lines.append("=" * 70)
    lines.append("1. OPEN-WEIGHT PREDICTIVE @ all densities (sig = BH-FDR<0.05 vs vague)")
    lines.append("=" * 70)
    if len(ow):
        show = ow[ow.density == 10][["dataset", "tier", "condition", "vague_rmse", "cond_rmse",
                                     "rmse_dmed", "rmse_p_bh", "rmse_sig", "cond_cov90"]]
        lines.append(show.to_string(index=False))
    lines.append("\n" + "=" * 70)
    lines.append("2. QUANTIZATION CONTROL (elicited range; confound dismissed if ladder all small)")
    lines.append("=" * 70)
    lines.append(qc.to_string(index=False))
    lines.append("\n" + "=" * 70)
    lines.append("3. WIDTH / OVERCONFIDENCE SWEEP (frontier; width 1 vs 2 vs 3)")
    lines.append("=" * 70)
    if len(ws):
        lines.append(ws[ws.density == 10].to_string(index=False))
    lines.append("\n" + "=" * 70)
    lines.append("4. PROTOCOL v3 vs v2 (frontier variogram, all networks, 10%)")
    lines.append("=" * 70)
    if len(v3):
        lines.append(v3.to_string(index=False))
    report = "\n".join(lines)
    (OUT / "REVISION_RESULTS.txt").write_text(report)
    print(report)
    print(f"\n[revision_tables] wrote tables + REVISION_RESULTS.txt to {OUT}")


if __name__ == "__main__":
    main()
