# Resource measurement definitions

`make exp-all` runs `experiments.run_evaluation` for all selected datasets first,
then `experiments.run_benchmark`, then `experiments.prepare_results`.
The benchmark coordinator starts a fresh interpreter for each dataset/model pair;
no process is reused between model measurements, even when evaluation uses `JOBS>1`.

## Accuracy and training

Both runners call `test_set_seq_fit_predict`: train on the last training and
validation folds, predict the complete test stream with delayed incremental updates
or periodic expanding-window batch refits, reconstruct residual predictions, and
use `all_time_aggregation_metrics`. Accuracy is the mean of per-device means at the
configured aggregation level. There is no 20,000-row training cap or device cap.

`total_train_time_s` sums only calls to `fit` and incremental updates, including
initial fitting and every periodic refit. It excludes prediction and metric
calculation. `pipeline_time_s` measures the instrumented test pipeline, including
metrics, inference probes and model-size measurements. Validation timing is
reported separately in evaluation metadata, not added to test training time.

Both runners use `protocol.seed` (default 42) for Python/NumPy global randomness
and explicitly pass it to constructor `random_state`/`seed` parameters. This common
seed overrides historical per-model seed values, including `None`; it also seeds
FastODT/boosting's independent NumPy generators and the ARF variants. Each newly
constructed device/fold model starts from that estimator seed. Benchmark workers
also start with `PYTHONHASHSEED` set to the common seed. Benchmark Random Forest uses `n_jobs=1`;
OMP/MKL/OpenBLAS/NumExpr thread environment variables are set to 1 before imports.
The actual parameters and environment are saved, including git revision and host.

Repeatability requires the same data, row order, code and library versions.
Equal seeds do not imply equal random draws or predictions across different
algorithms. Training times and memory can still vary between executions.

## Inference

For each device, retain up to 2,000 filtered feature rows from test predictions.
After fitting that device's final model, warm up one prediction, measure individual
prediction calls (mean/p50/p99 microseconds), and measure one batch over those rows.
Calls use the evaluation adapter, including array-to-dictionary conversion for
River models. Probes do not update the model. `pred_batch_samples_per_s` divides
the summed row count by the summed batch time. Accuracy still uses every test row.

## Memory

`peak_rss_mb` uses `resource.getrusage(RUSAGE_SELF).ru_maxrss`, converted to MiB
(Linux reports KiB, macOS bytes). It is the high-water mark for the fresh worker
process, including imports, loaded train/valid/test splits, predictions/metrics,
training, inference probes and Pympler measurement. It is not model-only memory
and excludes separate child processes. Benchmarks keep fitting single-process.

`baseline_peak_rss_mb` records the high-water mark after dataset loading. It is
context, not an amount subtracted from peak RSS to claim model memory usage.

`model_size_mb` is Pympler's model-object size: the maximum final device-model size
across devices. It is neither the sum of all device models nor peak fitting memory.
Device models are measured/released sequentially. Process RSS may remain above
live-object size because Python/native allocators retain freed storage.

## Resuming and interpreting comparisons

Schema-v2 benchmarks have a fingerprint covering source files, configuration,
dataset paths/sizes/mtime, probe count, seed and execution settings. `SKIP_EXISTING=1`
reuses only matching outputs. Table preparation excludes schema-v1 benchmarks;
benchmark rows expose their own revision, timestamp, settings and memory scope.
`same_evaluation_revision` compares revisions only, not complete fingerprints.

FastODT and xFastODT receive copied, matching configured parameters through
`matched_models` in `configs/experiments.yaml`. The measured results determine
whether binary-mask indexing improves time or memory; the runner assumes no win.
This benchmark does not perform hyperparameter search, repeated timing trials,
stream-length/feature-count scaling curves, or automatic manuscript updates.

## ARF ensemble-size/depth ablation

Run `make exp-arf-10x10` on the computation server to evaluate
`ARF_FastODT_10x10` and `ARF_Hoeffding_10x10` on all six hourly paper datasets.
Both use `n_models=10` and `max_depth=10`. Each variant inherits its own baseline's
remaining per-dataset parameters through `model_variants` in the configuration.
The original baseline parameters and `make exp-all` selection are preserved.
Outputs go to `results/rerun/arf_10x10/raw/` and `results/rerun/arf_10x10/tables/`;
override `ARF_RESULTS` to select another output root.

This isolates the change in configured ensemble size/depth relative to each
baseline. It does not equalize grace periods, split tests, leaf predictors, memory
budgets, feature sampling, actual node counts, or background-tree counts between
implementations. In particular, ARF(FastODT)'s existing `max_features`/`max_feature`
distinction is unchanged; this experiment alone does not fix that wiring issue.
