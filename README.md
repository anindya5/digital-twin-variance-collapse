# Precision Without Recall: Variance Collapse in LLM Digital Twins

Replication codebase for the experiments in *"Precision Without Recall:
Diagnosing and Mitigating Variance Collapse in LLM-Based Survey Simulation."*
Everything needed to reproduce the analysis from scratch with your own
Anthropic API key is in this repository. The public
[Twin-2K-500](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500)
dataset (Toubia et al., 2025 — 2,058 real US survey respondents) is downloaded
automatically.

## The hypothesis being tested

> Digital twins built from **identical profile attributes** produce
> **identical or near-identical** answers, even though real humans sharing
> that same coarse profile genuinely disagree with each other. LLM twins are
> **high precision** (they capture the modal answer) but **low recall** (they
> collapse the spread of real answers). Prompting the LLM to answer **for
> multiple people in one call** (multi-respondent prompting), rather than once
> per twin, is the proposed mitigation — and **persisting** those personas
> across the whole survey (persistent batched personas) additionally restores
> inter-question consistency.

## Experimental design

| Concept | Operationalization |
|---|---|
| "Identical profile attributes" | Real Twin-2K-500 participants grouped into **50 archetypes**: exact matches on 4 coarse demographic questions (age bucket, sex, US region, education — see `config.yaml`). |
| Ground-truth response variance | Each archetype's members' **actual wave-4 answers** — real, distinct humans, same coarse profile. |
| Independent prompting (common practice) | One Claude call per simulated twin per question, using only the archetype's shared coarse profile. 10 twins per archetype. |
| Multi-respondent prompting (the mitigation) | A **single** Claude call per (archetype, question) asking for 10 independent answers "for 10 different people" with that profile. |
| Persistent batched personas (the hybrid) | A **single** Claude call per archetype instantiating 10 distinct individuals who each answer the **entire 15-question survey**, all personas held in context simultaneously — batching + persistence combined. |
| Rich-persona external baseline | The dataset's own published GPT-4.1-mini simulation (full wave 1–3 persona per participant) — reused for free, no API calls. |
| Questions | 15 single-select multiple-choice items from wave-4 behavioral-economics replications (anchoring, framing, Allais paradox, WTA/WTP, ...) — subjective judgments where real people who look alike on paper are known to diverge. |

Each simulated condition is scored against the real human answers with:

- **RADIUS alignment suite** — ranking alignment (Top Rank Match with
  bootstrap tie-grouping, normalized Spearman Rank Correlation) and
  distribution alignment (Total Variation Distance, chi-square Distribution
  Homogeneity with a permutation fallback for small samples), at two
  granularities (per archetype–question pair, and pooled per question).
- **Forward KL divergence** D(P_human ‖ P_LLM) per archetype–question pair,
  with Jeffreys smoothing on the simulated side, plus the share of pairs whose
  unsmoothed KL is infinite (the simulator puts zero mass on a human-chosen
  option — the direct signature of variance collapse).
- **Self-Correlation Distance** — comparison of the inter-question Spearman
  correlation matrices (105 question pairs) between humans and each condition,
  with a Mantel permutation test, measuring whether simulated respondents have
  internally consistent "belief systems" across questions.
- **Variance diagnostics** — variance ratio, normalized-entropy ratio, and
  mode match versus the real humans in the same archetype.
- Paired t-tests and Wilcoxon signed-rank tests between conditions throughout.

All randomness in archetype construction and question selection is seeded
(`config.yaml`: `archetype.random_seed`, `questions.random_seed`), so those
steps are deterministic. LLM outputs are sampled at temperature 1.0 and cached
to disk, so a finished run is exactly re-analyzable; a fresh simulation rerun
will produce statistically equivalent (not bit-identical) LLM answers.

## Repository layout

```
run_experiment.py          # CLI entry point
src/
  cli.py                   # argument parsing
  pipeline/
    orchestrator.py        # ExperimentOrchestrator — coordinates all steps
    fetch.py               # download Twin-2K-500 subset
    plan.py                # build / load plan.json
    simulation_run.py      # run one LLM condition
    dataframes.py          # long-format tables for analysis
    analysis.py            # metrics, RADIUS, SCD, summary.md, plots
    plots.py               # diagnostic figures
    self_test.py           # synthetic end-to-end smoke test
  metrics.py               # variance / entropy / mode-match diagnostics
  radius.py                # RADIUS + KL divergence
  self_correlation.py      # Self-Correlation Distance
  simulate.py              # concurrent LLM API calls
tests/                     # unit tests for all evaluation modules
```

You can also drive the pipeline from Python:

```python
from src.pipeline import ExperimentOrchestrator

orch = ExperimentOrchestrator()
orch.fetch_data()
orch.build_plan()
orch.simulate("multi_respondent", skip_confirm=True)
orch.analyze()
```

## Requirements

- Python 3.10+
- An Anthropic API key (<https://console.anthropic.com/settings/keys>)
- ~50 MB disk for the dataset subset; no GPU needed

## Step-by-step replication

### 1. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your API key

```bash
cp .env.example .env
# edit .env and set: ANTHROPIC_API_KEY=sk-ant-...
```

The model is set in `config.yaml` (`simulation.model`, default
`claude-haiku-4-5-20251001`). You can override it with an `ANTHROPIC_MODEL`
environment variable without editing the file.

### 3. Sanity-check the pipeline (free, no API key needed)

```bash
python run_experiment.py self-test   # end-to-end smoke test on synthetic data
pytest                               # unit tests for all evaluation metrics
```

`self-test` exercises the orchestrator wiring. `pytest` runs focused unit tests
on every evaluation module — variance diagnostics, RADIUS (TRM / RC / TVD / DH /
KL), Self-Correlation Distance, and the pipeline itself. All tests use small
synthetic fixtures; no dataset download or API key required.

### 4. Download the dataset subset

```bash
python run_experiment.py fetch-data
```

Downloads the ~7 Twin-2K-500 files the experiment needs from Hugging Face
into `data/raw/` (cached; safe to re-run).

### 5. Build the experiment plan

```bash
python run_experiment.py build-plan
```

Groups participants into 50 archetypes and selects the 15 target questions,
writing `results/plan.json`. Deterministic given the seeds in `config.yaml`.

### 6. Run the simulations (costs API calls)

```bash
# Multi-respondent condition: 1 call per archetype x question = 750 calls
python run_experiment.py simulate --condition multi_respondent

# Independent condition: 1 call per twin x question = 7,500 calls
python run_experiment.py simulate --condition independent_coarse

# Persistent batched personas: 1 call per archetype = 50 calls
python run_experiment.py simulate --condition persistent_batched
```

With the default Haiku-class model this is inexpensive (on the order of a few
US dollars total), but check current pricing. Calls run concurrently
(`simulation.max_concurrency`) with retry/backoff, and every response is
cached on disk under `cache/` keyed by prompt/model/temperature — interrupted
runs resume for free, and re-running a finished step costs nothing.

### 7. Analyze

```bash
python run_experiment.py analyze
```

No API calls. Scores every condition you have simulated (plus the dataset's
free precomputed GPT-4.1-mini baseline, always included) against the real
human answers and writes everything to `results/`. Conditions you skipped are
simply omitted.

## What to expect in `results/`

| File | Contents |
|---|---|
| `plan.json` | The archetype groups and questions used (inspect to see exactly who is in each "identical profile" group). |
| `simulated_*.csv` | Raw simulated answers per condition. |
| `comparison_table.csv` | Per (archetype, question, source): variance, entropy, mode stats, and ratios vs. real humans. |
| `radius_per_archetype.csv`, `radius_pooled.csv` | RADIUS scores (TRM, RC, TVD, DH) + KL divergence per unit of analysis. |
| `self_correlation_summary.csv`, `self_correlation_pairs.csv` | Inter-question structure comparison per condition. |
| `summary.md` | Headline numbers and all paired significance tests, human-readable. |
| `variance_comparison.png`, `self_correlation_scatter.png` | Diagnostic plots. |

Headline results this pipeline should approximately reproduce (per
archetype–question granularity):

| Metric | Independent | Multi-respondent | Persistent batched |
|---|---|---|---|
| Distribution Homogeneity (share indistinguishable from humans) ↑ | ≈ 0.39 | ≈ 0.94 | ≈ 0.86 |
| Total Variation Distance ↓ | ≈ 0.53 | ≈ 0.33 | ≈ 0.36 |
| Smoothed forward KL (nats) ↓ | ≈ 0.91 | ≈ 0.41 | ≈ 0.51 |
| Share of pairs with unsmoothed KL = ∞ ↓ | ≈ 0.84 | ≈ 0.25 | ≈ 0.29 |
| Top Rank Match ↑ | ≈ 0.86 | ≈ 0.80 | ≈ 0.77 |
| Median variance ratio vs. humans (1.0 = perfect) | ≈ 0.00 | ≈ 1.07 | ≈ 0.99 |
| Self-correlation structure recovery r ↑ | ≈ 0 or negative | ≈ 0.2 | ≈ 0.46 |

Exact values will differ slightly because LLM sampling is stochastic, but all
cross-condition orderings and significance results should hold. The
self-correlation analysis should show that neither single-question Claude
condition reproduces human inter-question correlation structure, while the
persistent conditions do — persistent batched personas substantially
(r ≈ 0.46, though with correlations *stronger* than the human ones) and the
rich-persona baseline most closely (r ≈ 0.7). Marginal and structural
alignment are separate failure modes; persistent batched personas is the only
coarse-profile condition that addresses both at once.

## Configuration

All experiment parameters live in `config.yaml`: archetype attributes and
count, questions per run, simulated twins per archetype, model, temperature,
concurrency, and retry behavior. Scale `archetype.n_groups` /
`questions.n_questions` down for a cheaper pilot or up for a bigger run; the
pipeline recomputes everything downstream automatically.

## Notes and caveats

- The precomputed GPT-4.1-mini baseline differs from the Claude conditions
  in **both** model and persona richness, so treat its comparison as
  indicative, not controlled. The controlled comparisons in this design are
  among the three Claude conditions (same model, same coarse profile):
  `independent_coarse` vs. `multi_respondent` isolates batching, and
  `multi_respondent` vs. `persistent_batched` isolates persona persistence
  across the questionnaire.
- Twin-2K-500 wave 4 repeats measures from waves 1–3 (test–retest design).
  Twin prompts are built only from demographic profile data and scoring is
  only against wave-4 ground truth, so the predicted answers never leak into
  the prompts.
- Wave-4 experiments include between-subject variants, so the number of real
  humans answering a given question within an archetype can be smaller than
  the archetype size; the per-archetype statistics use permutation/bootstrap
  methods where small samples make asymptotic tests unreliable.

## Tests

| File | What it covers |
|---|---|
| `tests/test_metrics.py` | Distribution stats, variance/entropy ratios, paired Wilcoxon tests |
| `tests/test_radius.py` | TVD, KL divergence, TRM, RC, DH, `compute_radius_tables`, paired RADIUS tests |
| `tests/test_self_correlation.py` | Correlation matrices, SCD, structure recovery, undefined-pair detection |
| `tests/test_pipeline.py` | Orchestrator, CLI commands, `self-test` smoke path |

Run the full suite:

```bash
pytest -v
```

These tests verify that each metric behaves correctly on controlled synthetic
inputs (e.g. collapsed vs. spread distributions, matching vs. destroyed
inter-question structure). They do **not** re-run LLM simulations — use
`simulate` + `analyze` for that.
