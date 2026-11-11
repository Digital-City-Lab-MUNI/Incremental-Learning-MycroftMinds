# Paper experiment reruns

Run the notebook evaluation protocol and resource benchmarks with Python scripts.

## Run on the computation server

Prerequisites: project dependencies installed and processed train/valid/test `.npz`
files available through `data/` (see [data setup](../data/README.md)).
Run from this repository's root after transferring the updated code to the server:

```bash
make exp-all
```

This runs six hourly datasets, including Electricity and Friedman, with the eight
paper models plus xFastODT: 54 validation/test evaluations and 54 resource benchmarks.
It writes `results/rerun/raw/<dataset>/` and `results/rerun/tables/summary.md`,
`all_metrics.csv`, and `all_benchmark.csv`. Full-server runtime has not been measured.

Each resource benchmark uses a fresh Python process. All evaluations finish before
resource benchmarking starts; resource benchmarks run sequentially with one compute
thread and Random Forest `n_jobs=1` to avoid competition between benchmark workers.

## Resume and configure

```bash
make exp-all SKIP_EXISTING=1
make exp-all DATA_ROOT=/path/to/processed-data RESULTS=results/paper
make exp-all JOBS=8
```

`JOBS` parallelizes evaluation models only. `DATASET_JOBS` parallelizes evaluation
datasets. Benchmarks stay serial. Avoid other server workloads while collecting
timings. Resume checks code/config/data-file fingerprints; legacy or changed
outputs are recomputed. Data fingerprints use paths, sizes and modification times.

`PAPER_DATASETS` selects dataset keys; `PAPER_MODELS` selects model names.
Defaults are the six hourly datasets and
`BL_Mean GBDT RF HoeffdingTree ARF_Hoeffding FastODT ARF_FastODT IncubationBoost xFastODT`.
The `matched_models` configuration copies FastODT's parameters into xFastODT for
the implementation comparison; xFastODT is not separately tuned in this run.

## Outputs

- `*_valid.pkl`, `*_test.pkl`: predictions and metrics from the evaluation protocol.
- `*_meta.json`: evaluated parameters, protocol, seed, revision and environment.
- `*_benchmark.json`: full-test accuracy together with training time, prediction
  latency/throughput, fresh-process peak RSS and final device-model size.
- `tables/all_benchmark.csv`: accuracy–resource comparison, including provenance
  and whether the associated evaluation has the same git revision.

MAPE is a fraction in CSV/JSON: multiply by 100 for percent. Peak process RSS and
model-object size measure different quantities. See [measurement definitions](BENCHMARKS.md).

## Other commands

- `make exp-run DATASET=od_friedman_1h`: evaluation for one dataset.
- `make exp-benchmark DATASET=od_friedman_1h`: resources for one dataset.
- `make exp-prepare`: rebuild tables; excludes legacy benchmark schemas.
- `make exp-matrix`: all nine configured dataset variants, evaluation and resources.
- [WES job](../jobs/fastodt_rerun.wes): existing SLURM launcher.

The scripts evaluate all available devices and report their mean metrics. They do
not infer the single device/site selected in the paper's original tables.
