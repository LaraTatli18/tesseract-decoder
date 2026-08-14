# Benchmarking Framework

This directory contains the benchmarking, parameter-tuning, and analysis framework used to evaluate the Tesseract decoder.

The framework is organised around the following workflow:

```
run_tesseract.py
        ↓
experiments/runs/<run_name>/
    ├── manifest.json
    ├── results.csv
    └── samples/              # optional CRN/per-shot arrays
        ↓
plot_*.py
        ↓
experiments/plots/
```

Each benchmark run is self-contained. Its run directory records the decoder configuration, benchmark metadata, and results needed to reproduce and analyse the experiment. Plotting scripts then compare one or more run directories and write generated figures to `experiments/plots/`.

---

## Benchmarking files

### `run_tesseract.py`

Main benchmarking driver.

Responsibilities:

- discovers `.stim` benchmark circuits
- supports surface-code and bivariate-bicycle-code benchmark filenames
- configures the Tesseract decoder
- runs single-shot or batch decoding
- supports multiprocessing across circuits and multithreading within the decoder
- records decoder accuracy, logical error rates, runtime, throughput, confidence intervals, and other benchmark statistics
- creates reproducible run directories and manifests
- supports common-random-number (CRN) sampling for statistically paired decoder comparisons
- optionally saves sampled detector events, logical observables, and per-shot decoder correctness as NumPy arrays for paired/per-shot analysis
- optionally enables Tesseract visualization output for dedicated diagnostic runs
- exposes decoder parameters used by the parameter studies, including beam, pqlimit, beam climbing, merge-errors, and sparsification settings

#### Common-random-number validation

`run_tesseract.py` supports opt-in common-random-number sampling using `--crn --crn-seed <seed>`. Runs using the same Stim circuit and CRN seed receive the same sampled detector and observable data, allowing decoder configurations to be compared on identical physical error instances rather than independent Monte Carlo samples.

Passing `--save-samples` additionally stores the sampled arrays in the run directory:

```
samples/
_detections.npy
_observables.npy
_shot_correct.npy
```

The `shot_correct` array records whether each individual shot was decoded correctly. This enables paired comparisons between configurations and identification of discordant shots for which one configuration succeeds and another fails.

Independent statistical repetitions should use different CRN seeds, while all configurations compared within a repetition should use the same seed. For example, seeds `1001`, `1002`, and `1003` can define three independent repetitions, with each seed shared across the configurations being compared.

#### Diagnostic visualization

Passing `--create-visualization` enables Tesseract's decoder visualization output. Visualization runs are intended for targeted diagnostic/mechanistic analysis and should not be treated as runtime replicates because visualization may introduce additional overhead. A typical final study therefore uses ordinary CRN runs for statistical validation and a separate visualization-enabled diagnostic run for selected configurations or error instances.

---

### `benchmark_utils.py`

Shared utilities for benchmarking.

Contains:

- benchmark result dataclasses
- CSV writing
- confidence intervals
- histogram utilities
- benchmark selection
- circuit filename parsing and metadata extraction

---

### `run_manifest.py`

Creates reproducible benchmark runs.

Responsible for:

- run directory naming
- manifest generation
- Git metadata
- command recording
- experiment metadata
- optional circuit-family-specific metadata
- CRN seed, sample-saving, and visualization settings for validation/diagnostic runs

---

### `optuna_tune.py`

Runs Optuna-based parameter tuning for the Tesseract decoder.

The tuner launches Tesseract benchmarks for each trial, evaluates decoder runtime and logical-error-rate objectives, and records trial results and study metadata. It supports multi-objective studies used to explore runtime/quality trade-offs and Pareto-optimal parameter settings.

---

## Plotting

Shared plotting utilities live in `plot_utils.py`. This module provides common run loading, manifest filtering and consistency checks, formatting helpers, plotting style, figure saving, and the half-shot floor used for logarithmic logical-error-rate plots.

### `plot_surface_codes.py`

Plots logical error rate versus physical error rate for a single benchmark run, with one curve per code distance.

---

### `plot_thread_sweep.py`

Compares decoder scaling across different thread counts.

Typical outputs:

- decode time versus threads
- throughput versus threads

---

### `plot_parameter_sweep.py`

Generic one- or two-parameter sweep plotter.

It groups benchmark runs by an arbitrary manifest parameter, optionally splits curves by a secondary parameter, and compares:

- decode time
- throughput
- logical error rate per round, including confidence intervals

This is the general-purpose plotting script for decoder parameters that do not require a specialised analysis.

---

### `plot_beam_sweep.py`

Plots one-dimensional beam-parameter sweeps across physical error rates.

It produces one logical-error-rate-per-round figure per code distance, with separate curves for the values of the swept beam parameter.

---

### `plot_beam2d.py`

Analyses the two-dimensional detector-beam / beam-climbing parameter study.

It produces per-distance and multi-distance comparisons of:

- logical error rate per round
- decode time
- throughput

It also writes a flattened summary CSV, records plot metadata, and reports the detector-beam setting giving the lowest logical error rate for each benchmark configuration.

---

### `plot_pqlimit_sweep.py`

Analyses priority-queue-limit sweeps across detector beam, distance, and physical error rate.

Typical outputs include:

- logical-error-rate cross sections versus pqlimit
- absolute logical-error-rate heatmaps
- heatmaps relative to the best logical error rate

---

### `plot_sparsification_sweep.py`

Compares sparsification enabled and disabled across code distances for fixed beam, pqlimit, and physical-error-rate settings.

It plots:

- logical error rate per round with confidence intervals
- decode time
- throughput

---

### `plot_optuna_sparsify.py`

Analyses the focused Optuna sparsification study.

It is used to visualise the runtime/quality trade-off and the structure of the Pareto front, together with the effects of sparsification parameters such as base degree, maximum degree, and reactivation limit.

Typical outputs include:

- runtime versus logical error rate
- Pareto-front-only plots
- parameter-usage summaries
- parameter-effect plots
- runtime and logical-error-rate heatmaps
- reactivation-limit sweeps

---

## Directories

### `jobs/`

SLURM scripts used to launch benchmark experiments on available compute resources.

Examples include:

- beam sweeps
- pqlimit sweeps
- sparsification studies
- thread sweeps
- Optuna studies
- large-shot benchmark runs

---

### `runs/`

Automatically generated benchmark runs.

Each run contains at least:

```
manifest.json
results.csv
```

Runs created with `--save-samples` additionally contain a `samples/` directory with detector, observable, and per-shot correctness arrays. These arrays are not intended to be tracked by Git.

This directory is not tracked by Git.

---

### `plots/`

Generated analysis figures and plot outputs.

This includes comparison plots, parameter-study figures, and publication-quality figures. Generated plotting outputs are not intended to be tracked as source code.

---

### `archive/`

Prototype scripts and previous versions kept for reference.

These are not part of the active benchmarking pipeline.

---

## Typical workflow

1. Run a benchmark locally or submit a SLURM job from `jobs/`:

```
python run_tesseract.py ...
```

2. The benchmark creates a reproducible run directory:

```
experiments/runs/<run_name>/
    ├── manifest.json
    ├── results.csv
    └── samples/              # present when –save-samples is enabled
```

3. Use the appropriate plotting script for the analysis. For example:

```
plot_surface_codes.py
plot_thread_sweep.py
plot_parameter_sweep.py
plot_beam_sweep.py
plot_beam2d.py
plot_pqlimit_sweep.py
plot_sparsification_sweep.py
plot_optuna_sparsify.py
```

4. For automated tuning studies, use `optuna_tune.py` to explore decoder parameter settings and runtime/quality trade-offs.

5. For final paired validation, enable CRN and sample saving, using a shared seed across the configurations being compared:

```
python run_tesseract.py … –crn –crn-seed 1001 –save-samples
```

Use a different CRN seed for each independent repetition. Use `--create-visualization` only for separate diagnostic runs.

6. Analyse the resulting figures and tables to identify parameter sensitivity, runtime/accuracy trade-offs, and tuning rules that can be tested across code families and benchmark regimes.
