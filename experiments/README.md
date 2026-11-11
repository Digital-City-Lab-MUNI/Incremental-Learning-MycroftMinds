# Experiment scripts (notebook-equivalent reruns + WES job)

Script-based replacement for the evaluation notebooks (`notebooks/*batch/*_evaluation.ipynb`,
`notebooks/other_data/*_evaluation.ipynb`), plus a [WES](../../wes) job to run them on the
HPC. Created for TODO.md Phase 5: rerunning the experiments affected by the behavioral
fixes in `src/` and measuring the computational benchmark required by `../FastODT-paper/paper.todo.md`.

## Layout

```
├── configs/experiments.yaml     # dataset x model x hyperparameter matrix (transcribed from the notebooks)
├── experiments/
│   ├── run_evaluation.py        # valid-splits walk-forward + test evaluation -> raw results + provenance
│   ├── run_benchmark.py         # train time (incl. batch refits), single/batch prediction latency, peak RSS, model size, accuracy
│   ├── run_matrix.py            # full matrix in parallel without SLURM (one forked worker per dataset)
│   ├── prepare_results.py       # raw -> tables (CSV + summary.md), optional --baseline deltas
│   └── bootstrap.py / registry.py
└── jobs/
    ├── pre.sh                   # WES pre-script: link datasets into the cloned repo's data/
    ├── job.sh                   # SLURM array job: one task per dataset, then prepare tables
    └── fastodt_rerun.wes        # WES task definition (ssh: hpc)
```

Run from the **repo root**. The scripts auto-detect the repo (for the `src` package) and
default the data root to `./data`; raw results go to `results/rerun/raw`, tables to
`results/rerun/tables` (both gitignored via `results/*`).

## Datasets

9 dataset entries in `configs/experiments.yaml`. The paper uses 6 datasets; all 6 are now
available locally, processed splits in `../datasets`:

| config key       | paper dataset          | data dir   | prefix                  | on disk at `../datasets` |
|------------------|------------------------|------------|-------------------------|--------------------------|
| fb_1h, fb_15m    | Industrial MM-1st (Industrial(1)) | first_batch | dev_4_1h_wfv / dev_4_15m_new_wfv | yes |
| sb_com_1h/15m    | Commercial MM-2nd      | second_batch/commercial | sb_com_*           | **no**                   |
| sb_ind_1h/15m    | Industrial MM-2nd (Industrial(2)) | second_batch/industrial | sb_ind_*     | **no**                   |
| od_aq_1h         | Air Quality            | other_data | od_AQ_hour_wfv          | yes (note: on-disk case) |
| od_electricity_1h| Household electricity  | other_data | od_electricity_hour_wfv | yes                      |
| od_friedman_1h   | Friedman synthetic     | other_data | od_friedman_hour_wfv    | yes                      |

On disk the dirs are nested as `../datasets/<name>(current)/<name>/`, so create a staging
data root with properly named subdirs:

```bash
# actually the repo's data/ dir already has these symlinks (see data/README.md),
# so the default --data-root ./data works; a separate staging dir is only needed
# on the HPC:
mkdir -p data_staging
for n in first_batch other_data second_batch; do
  ln -s "$(pwd)/../datasets/${n}(current)/${n}" "data_staging/${n}"
done
```

## Run locally

```bash
pip install -r requirements.txt        # or: make install
python -m experiments.run_evaluation --dataset od_friedman_1h --data-root data_staging
python -m experiments.run_benchmark --dataset od_friedman_1h --data-root data_staging \
    --models FastODT xFastODT IncubationBoost HoeffdingTree ARF_Hoeffding GBDT RF
python -m experiments.prepare_results   # results/rerun/raw -> results/rerun/tables
# or via Makefile: make exp-run DATASET=od_friedman_1h DATA_ROOT=data_staging
```

Useful flags: `--models` to subset models, `--max-length N` to cap samples per split,
`--start-split/--end-split` to subset validation folds (with `--start-split K > 0` the new
results are merged into an existing `<Model>_valid.pkl`, so chunked runs equal a full run),
`--jobs N` to run models of one dataset in parallel worker processes, `--skip-existing` to skip
models whose results already exist (safe to re-run interrupted batches).
`prepare_results --baseline <prev raw dir>` adds `*_delta` columns to compare against a previous run.

## Run the full matrix in parallel (no SLURM)

```bash
python -m experiments.run_matrix --jobs 4 --model-jobs 2 --skip-existing
# or via Makefile: make exp-matrix JOBS=4 MODEL_JOBS=2 SKIP_EXISTING=1
# subset: --datasets sb_ind_1h od_aq_1h --models FastODT xFastODT
```

`run_matrix` runs one forked worker per dataset (each worker runs `run_evaluation` then
`run_benchmark` for its dataset, with `--model-jobs` as their `--jobs`), then rebuilds
`results/rerun/tables`. It is the plain-server equivalent of the SLURM array in `jobs/job.sh`;
total concurrent processes are roughly `jobs x model_jobs` and each dataset worker loads its
own data, so keep `--jobs` modest on memory-limited machines.

## Run on the HPC via WES

```bash
# push this branch first (must contain experiments/ + the src/ fixes)
# wes CLI: the current ../../wes checkout registers `dispatch` (not `launch`);
# (cd ../../wes && uv sync) regenerates the `wes` console script first:
../../wes/.venv/bin/wes dispatch jobs/fastodt_rerun.wes   # from the repo root; submits the SLURM array job
../../wes/.venv/bin/wes sync --ssh hpc                    # syncs wes.status + results/rerun into results/
../../wes/.venv/bin/wes check --ssh hpc                   # one-shot status
```

`jobs/fastodt_rerun.wes` clones this repo on `ssh: hpc`, runs `jobs/pre.sh` (links
`FASTODT_DATA_SHARED` subdirs into the clone's `data/`), and submits `jobs/job.sh`
(`#SBATCH --array=0-8%9`, one task per dataset; each runs evaluation + benchmark for its
dataset and refreshes `results/rerun/tables`). Set `partition` in the `.wes` after
`ssh hpc sinfo`. Override per launch with env: `MODELS`, `BENCH_MODELS`, `MAX_LENGTH`,
`FASTODT_DATA_ROOT`.

## Results format

`results/rerun/raw/<dataset>/`:

* `<Model>_valid.pkl`, `<Model>_test.pkl` — predictions, per-period metrics, aggregated
  metrics (same structure the notebooks keep in variables);
* `<Model>_meta.json` — params, update_freq, wall times, peak RSS, environment (host,
  python, library versions, thread env) and this repo's git revision (with `-dirty` flag);
* `<Model>_benchmark.json` — total training time (batch models include all periodic
  refits), single-sample prediction latency (mean/p50/p99 us), batched prediction time,
  peak RSS MB (`ru_maxrss`), model size MB (pympler), test rmse/mae/mape.

`results/rerun/tables/`: `<dataset>_metrics.csv`, `<dataset>_benchmark.csv`, `all_*.csv`,
`summary.md` (markdown tables + provenance) — ready to paste into the paper.
