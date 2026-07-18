# Benchmarking Framework

This directory contains the benchmarking and analysis framework used to evaluate the Tesseract decoder.

The framework is organised around the following workflow:

```
run_tesseract.py
        ↓
experiments/runs/<run_name>/
        ├── manifest.json
        ├── results.csv
        └── (optional plots)
        ↓
plot_*.py
        ↓
experiments/plots/
```

Each benchmark run is self-contained. The run directory stores the decoder configuration, benchmark results, and any plots generated from that run.

---

## Files

### `run_tesseract.py`

Main benchmarking driver.

Responsibilities:

- discovers `.stim` benchmark circuits
- configures the decoder
- runs single-shot or batch decoding
- supports multiprocessing and multithreading
- records benchmark statistics
- creates run directories and manifests

---

### `benchmark_utils.py`

Shared utilities for benchmarking.

Contains:

- benchmark result dataclasses
- CSV writing
- confidence intervals
- histogram utilities
- benchmark selection
- filename parsing

---

### `run_manifest.py`

Creates reproducible benchmark runs.

Responsible for:

- run directory naming
- manifest generation
- Git metadata
- command recording
- experiment metadata

---

## Plotting

### `plot_surface_codes.py`

Plots logical error rate for a single benchmark run.

Input:

```
experiments/runs/<run>/results.csv
```

Output:

```
logical_error_rate.png
```

---

### `plot_thread_sweep.py`

Compares decoder scaling across different thread counts.

Typical outputs:

- decode time vs threads
- throughput vs threads

---

### `plot_parameter_sweep.py`

Compares runtime performance across different decoder parameters.

Typical outputs:

- decode time
- throughput

---

### `plot_beam_sweep.py`

Plots decoder accuracy across beam sizes.

Produces one logical error rate plot for each code distance, allowing beam sizes to be compared without averaging over different benchmark configurations.

---

## Directories

### `jobs/`

SLURM scripts used to launch benchmark experiments on La Chouffe.

Examples:

- beam sweep
- thread sweep
- large-scale benchmark runs

---

### `runs/`

Automatically generated benchmark runs.

Each run contains:

```
manifest.json
results.csv
(optional plots)
```

This directory is not tracked by Git.

---

### `plots/`

Generated figures.

Contains comparison plots and publication-quality figures.

---

### `archive/`

Prototype scripts and previous versions kept for reference.

These are not part of the active benchmarking pipeline.

---

## Typical workflow

1. Run a benchmark

```
python run_tesseract.py ...
```

or submit a SLURM job from `jobs/`.

2. Results are written to

```
experiments/runs/<run_name>/
```

3. Generate plots

```
plot_surface_codes.py
plot_thread_sweep.py
plot_parameter_sweep.py
plot_beam_sweep.py
```

4. Analyse the results and compare benchmark configurations.
