# LLM-elicited priors for geostatistical prediction

Analysis code for the study on whether priors elicited from large language models
(LLMs) improve Bayesian geostatistical prediction in data-sparse regions, compared
with the same model under vague priors and with classical kriging.

The LLM supplies, from a task description alone (no observed data), prior distributions
for (i) covariate coefficients and (ii) variogram / Gaussian-process hyperparameters
(range, sill, nugget). The Bayesian model is held identical across prior conditions;
only the priors change, which isolates the contribution of the elicited priors.

## Repository layout

```
src/
  config.py            config.yaml loader
  data.py              dataset loading + density subsampling
  elicit.py            LLM -> priors (strict JSON), with the leakage guard
  priors.py            JSON priors -> PyMC distributions
  models.py            ordinary / regression / Bayesian kriging + random forest
  cv.py                spatial blocked cross-validation
  metrics.py           RMSE / MAE / CRPS / PIT / interval coverage
  run.py               experiment driver (reads config.yaml)
  sim.py               known-truth range-misspecification simulation
  analyze_paper.py     result tables
  figures_pro.py       publication figure generator
  convergence_check.py / cost_table.py / report_extras.py / revision_tables.py
                       auxiliary result/diagnostic tables
prompts/               versioned elicitation templates (elicit_v1..v3)
config.yaml            single source of truth: datasets, density levels, seeds,
                       prior conditions, models, run order
requirements.txt       Python dependencies
```

## Reproducibility

`config.yaml` is the single source of truth for datasets, density levels, seeds, prior
conditions, and the model tiers. All seeds are fixed there; every run is reproducible
from the configuration file.

**Leakage guard.** Elicitation prompts are built only from the configuration's text
fields (variable description, units, region, covariate names) plus a fixed template,
never from data arrays. As defence-in-depth, `elicit.py` rejects the vocabulary of data
summary statistics in the config text. Observed values and their summaries never enter
the prompt.

## Running the pipeline

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1. classical + ML baselines
python src/run.py    --config config.yaml --stage baselines

# 2. elicit priors from the configured model tier (writes priors JSON + raw log)
python src/elicit.py --config config.yaml --tier <tier>

# 3. fit the Bayesian model under each prior condition (model held identical)
python src/run.py    --config config.yaml --stage bayes --priors vague,llm_both

# 4. evaluate: metrics vs. density for all conditions
python src/run.py    --config config.yaml --stage eval

# tables and figures
python src/analyze_paper.py
python src/figures_pro.py all
```

Stages are independent and checkpointed per cell, so an interrupted run resumes where it
stopped. Available dataset names, model tiers, prior conditions, and density levels are
all defined in `config.yaml`.

## License

Released under the MIT License (see `LICENSE`).
