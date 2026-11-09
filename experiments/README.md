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

9 dataset entries in `configs/experiments.yaml`. The paper uses 6 datasets; their processed
splits live in `../datasets` — 4 of 6 available locally:

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
mkdir -p data_staging
ln -s "$(pwd)/../datasets/first_batch(current)/first_batch" data_staging/first_batch
ln -s "$(pwd)/../datasets/other_data(current)/other_data"   data_staging/other_data
# add second_batch here once available
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
`--start-split/--end-split` to subset validation folds. `prepare_results --baseline <prev raw dir>`
adds `*_delta` columns to compare against a previous run.

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
