#!/usr/bin/env python
"""convergence_check.py — MCMC convergence + calibration evidence for the paper (review #3/#4).

The production run sets compute_convergence_checks=False (speed) and stores only metrics, not
traces, so R-hat / ESS were never recorded and raw PIT values were not saved. This script
re-fits a STRATIFIED SUBSET of the bayes cells with convergence checks ON, capturing per-fit
R-hat, bulk/tail ESS and divergence counts, and dumps per-point PIT values for a PIT histogram.

It re-uses the exact production model (models.predict_bayesian_kriging) and the exact
density-subsample + blocked-CV fold logic, so the subset is representative, not a re-derivation.

Outputs (checkpointed; restart-safe):
  results/convergence/summary.csv   one row per (dataset,density,condition,seed,fold)
  results/convergence/pit.csv       one row per held-out point (pooled PIT values)

Usage: .venv/bin/python src/convergence_check.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import config as configmod          # noqa: E402
import data as datamod              # noqa: E402
import cv as cvmod                  # noqa: E402
import models as modelsmod          # noqa: E402
from metrics import pit_values      # noqa: E402
from run import _load_consensus     # noqa: E402

# Stratified design: all real networks x all densities x {vague, frontier llm_both},
# two seeds, all 5 folds. ~240 fits ~ 75 min on the 6-core CPU. Frontier model fixed to
# Claude Opus 4.8 as the representative capable elicitor (the headline tier).
DATASETS = ["main", "andes", "prcp"]
CONDITIONS = [("vague", "none"), ("llm_both", "anthropic/claude-opus-4.8")]
SEEDS = [11, 22]
SUMMARY_VARS = ["beta0", "betas", "ls", "eta", "sigma"]


def _iter_folds(cfg, coords, density, seed):
    n = len(coords)
    keep = cvmod.density_subsample(n, density, seed)
    sub = coords[keep]
    k = int(cfg.cv.get("k", 5))
    for fi, (tr_s, te_s) in enumerate(cvmod.blocked_kfold(sub, k, seed)):
        yield fi, keep[tr_s], keep[te_s]


def main():
    cfg = configmod.load("config.yaml")
    outdir = cfg.output_dir / "convergence"
    outdir.mkdir(parents=True, exist_ok=True)
    sum_path, pit_path = outdir / "summary.csv", outdir / "pit.csv"

    done = set()
    if sum_path.exists():
        prev = pd.read_csv(sum_path)
        done = {tuple(r) for r in prev[["dataset", "density", "condition", "seed", "fold"]].values}
        print(f"[conv] resuming: {len(done)} fits already recorded", flush=True)

    sum_rows, pit_rows = [], []

    def flush():
        if sum_rows:
            pd.DataFrame(sum_rows).to_csv(
                sum_path, mode="a", header=not sum_path.exists(), index=False)
            sum_rows.clear()
        if pit_rows:
            pd.DataFrame(pit_rows).to_csv(
                pit_path, mode="a", header=not pit_path.exists(), index=False)
            pit_rows.clear()

    for which in DATASETS:
        bundle = datamod.load_dataset(cfg, which)
        coords, X, y = bundle["coords"], bundle["X"], bundle["y"]
        for cond, model_tag in CONDITIONS:
            spec = None if cond == "vague" else _load_consensus(cfg, model_tag, which, "v2")
            if cond != "vague" and spec is None:
                print(f"[conv] no consensus for {model_tag}/{which}; skipping {cond}", flush=True)
                continue
            for density in cfg.densities:
                for seed in SEEDS:
                    for fold, tr, te in _iter_folds(cfg, coords, density, seed):
                        ident = (which, density, cond, seed, fold)
                        if ident in done:
                            continue
                        try:
                            mu, sd, idata = modelsmod.predict_bayesian_kriging(
                                coords[tr], X[tr], y[tr], coords[te], X[te], y[te],
                                condition=cond, spec=spec, covariates=bundle["covariates"],
                                mcmc=cfg.mcmc, x_sd=bundle["X_sd"], seed=seed,
                                return_idata=True)
                            s = az.summary(idata, var_names=SUMMARY_VARS, kind="diagnostics")
                            n_div = int(idata.sample_stats["diverging"].values.sum())
                            row = dict(
                                dataset=which, density=density, condition=cond, seed=seed,
                                fold=fold, n_train=int(len(tr)), n_test=int(len(te)),
                                max_rhat=float(s["r_hat"].max()),
                                min_ess_bulk=float(s["ess_bulk"].min()),
                                min_ess_tail=float(s["ess_tail"].min()),
                                n_divergent=n_div, status="ok")
                            for p in pit_values(y[te], mu, sd):
                                pit_rows.append(dict(dataset=which, density=density,
                                                     condition=cond, pit=float(p)))
                        except Exception as e:
                            row = dict(dataset=which, density=density, condition=cond,
                                       seed=seed, fold=fold, status=f"error:{e}")
                        sum_rows.append(row)
                        print(f"[conv] {which} {cond} d{density} s{seed} f{fold} "
                              f"rhat={row.get('max_rhat','-')} div={row.get('n_divergent','-')}",
                              flush=True)
                        flush()
    flush()
    print("[conv] done.", flush=True)


if __name__ == "__main__":
    main()
