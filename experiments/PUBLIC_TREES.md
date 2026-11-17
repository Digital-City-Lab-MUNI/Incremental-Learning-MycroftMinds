# Rerun the root-only public-data configurations

The saved Air Quality, Friedman and Electricity FastODT models had one node and
zero splits. `make exp-fix-unsplit` selects a growing-tree configuration for routing
benchmarks and records any validation-accuracy cost relative to the unrestricted
winner. It does not change the Hoeffding split rule or manufacture split candidates.

## Run on the computation server

After syncing the updated source, from the repository root:

```bash
make exp-fix-unsplit
```

Prerequisites: the existing Python environment, a Linux C11 compiler and the same
processed train/valid/test files under `data/`. Production runtime is not yet measured.
The command builds the C learner, tunes the public datasets, then evaluates and
benchmarks five models per dataset (15 validation/test runs and 15 resource runs):

- `FastODT_Original`: original parameters as the labelled control.
- `BL_Prev`: persistence (zero predicted residual).
- `FastODT`, `xFastODT`, `xFastODT_C`: identical validation-selected parameters/seed.

## Selection rule

Only `tie_threshold` changes. All other per-dataset FastODT parameters are retained.
Candidates are the original threshold plus `0.01, 0.02, 0.05, 0.1`; `PUBLIC_TAUS`
overrides the grid. This expands the earlier parameter grid and must be reported
as a new experiment rather than an exact reproduction of the original tuning.

Each candidate uses every walk-forward validation fold. Select minimum hourly
validation RMSE (mean of device means) among candidates with a split in **every
validation fit**. This is an explicit structural constraint for the routing
comparison, not an assertion that a split tree is always more accurate. The report
also retains the unconstrained winner and the additional validation RMSE incurred.
No test target values are loaded during candidate selection.

If no candidate satisfies the constraint, selection fails with saved root
diagnostics; it does not silently increase thresholds or bypass split tests. After
the rerun, every selected family model must have nonzero final routing depth or
the command returns failure. The original control is allowed to remain unsplit.

## Outputs

Default output root: `results/rerun/public_trees/` (override `PUBLIC_TREE_RESULTS`).

- `tuning/<dataset>/tuning.json`: every candidate's validation errors, split-fit
  counts, root observations, last-attempt counts, candidate gains, confidence
  radius and histogram representative counts. Diagnostics are final snapshots,
  not the gain values at every historical split attempt.
- `selected_config.yaml`: full rerun configuration, including original controls.
- `raw/`, `tables/`: the same accuracy/resource artifacts as the main matrix.

The input config and existing results remain unchanged. Resume with
`make exp-fix-unsplit SKIP_EXISTING=1`. For a later full matrix using the selected
public configurations, run:

```bash
make exp-all CONFIG=results/rerun/public_trees/selected_config.yaml
```

The Industrial(2) chronology issue and the Air Quality date-range discrepancy
are separate investigations; this command does not resolve either one.
