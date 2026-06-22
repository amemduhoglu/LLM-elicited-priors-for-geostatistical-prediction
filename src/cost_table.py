#!/usr/bin/env python
"""cost_table.py — WP6 cost/latency table for the practical-guidance section.

Two cost components:
  1. Elicitation — the only place tiers differ in price. Frontier: paid OpenRouter calls
     (documented design: 3 models x 3 datasets x 2 protocols x 3 phrasings ~= 54 calls ~= $4
     per full pass, ~$0.07/call); local: free (Ollama on the workstation) but slow on CPU.
  2. MCMC fit — identical model across conditions, so per-fit wall-time is condition- and
     tier-independent; we report it from the checkpointed cells to make that explicit (the
     LLM choice changes the prior, not the inference cost).

Writes results/paper/tbl_cost.csv.
"""
from __future__ import annotations
import json
import glob
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

FRONTIER = ("anthropic", "openai", "google", "claude", "gpt", "gemini")

# Documented elicitation economics (config.yaml frontier_openrouter design note).
ELICIT_COST = {
    "frontier": {"usd_per_call": 0.07, "usd_per_full_pass": 4.0, "paid": True,
                 "note": "OpenRouter API; cached per phrasing so paid calls never repeat on restart"},
    "local":    {"usd_per_call": 0.0,  "usd_per_full_pass": 0.0, "paid": False,
                 "note": "Ollama on a 6-core CPU / 6 GB GPU; free but minutes per model"},
}


def _tier(model: str) -> str:
    m = (model or "").lower()
    return "frontier" if any(k in m for k in FRONTIER) else "local"


def mcmc_latency() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(ROOT / "results/cells/*.json")):
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        if d.get("stage") != "bayes" or d.get("status") != "ok" or "seconds" not in d:
            continue
        rows.append({"condition": d.get("condition"),
                     "tier": _tier(d.get("elicit_model", "")),
                     "seconds": float(d["seconds"])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    g = (df.groupby(["condition", "tier"])["seconds"]
           .agg(n="count", median="median", mean="mean").reset_index())
    return g


def main():
    lat = mcmc_latency()
    lat.to_csv(OUT / "tbl_mcmc_latency.csv", index=False)

    recs = []
    for tier, c in ELICIT_COST.items():
        sub = lat[lat.tier == tier] if not lat.empty else lat
        recs.append({
            "tier": tier,
            "elicit_usd_per_call": c["usd_per_call"],
            "elicit_usd_per_full_pass": c["usd_per_full_pass"],
            "paid": c["paid"],
            "mcmc_fit_seconds_median": (float(sub["median"].median()) if len(sub) else np.nan),
            "note": c["note"],
        })
    cost = pd.DataFrame(recs)
    cost.to_csv(OUT / "tbl_cost.csv", index=False)
    overall = lat["median"].median() if not lat.empty else float("nan")
    print("== WP6 cost/latency ==")
    print(cost.to_string(index=False))
    print(f"\nMCMC per-fit median across all conditions/tiers: {overall:.1f}s "
          f"(condition-independent: the prior changes, not the inference cost)")


if __name__ == "__main__":
    main()
