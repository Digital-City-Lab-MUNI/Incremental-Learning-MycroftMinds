# FastODT review TODO

Review of the current implementation against `message.txt` and the manuscript in
`../FastODT-paper/templateArxiv.tex`. Findings were checked with focused Python
behavioral probes; implementation fixes have not yet been applied.

## Execution plan

Work through these phases in order. The priority checklists below provide the
finding details and source references.

### Phase 1: Establish intended behavior and experiment provenance

- [x] Trace evaluation notebooks and saved configurations to identify the tree
  implementation used for each published result; record unresolved provenance.
- [x] Record decisions for mature-tree prediction updates, maximum-depth
  convention, grace-period boundaries, and pruning loss/window initialization.
  Use the handoff as the proposed specification: MSB-first indexing, fresh child
  statistics, and greedy prefix pruning. Resolve the tension between continued
  leaf updates and the ensemble's assumption of stable mature learners.
- [x] Identify a small, data-independent test command for focused regressions;
  existing `tests/test_models.py` loads external datasets at import time.

### Phase 2: Correct boosting targets and pruning scores

- [x] Add focused tests demonstrating the growing-first-tree target bug and the
  stale incubation target after rejection, then implement the two corrections.
- [x] Cover accepted and rejected prefixes, no pruning, an unfilled first tree,
  and non-unit learning rates. Verify targets use pre-training predictions.
- [x] Implement the selected nonnegative rolling loss and initialization; verify
  that overprediction is penalized and opposite-sign errors cannot cancel.
- [x] Run the focused boosting tests. Completion criterion: all target and score
  assertions pass while preserving greedy pruning and incubation/hatching.

### Phase 3: Align tree learning and routing

- [x] Implement and test consistent threshold-equality routing first.
- [x] Implement the chosen maximum-depth behavior and fresh lazy-node statistics,
  preserving inherited predictions and MSB-first ancestor/child indexing.
- [x] Implement grace-period gating and reset after each attempted split in both
  tree variants, including unsuccessful attempts and depth-limited behavior.
- [x] Verify boundary cases with small deterministic streams: threshold equality,
  first observation in a lazy leaf, repeated failed splits, and depth saturation.
- [x] Run focused tree and boosting regressions together. Completion criterion:
  routing, statistics, split-attempt counts, and maturity behavior match the
  decisions recorded in Phase 1.

### Phase 4: Synchronize documentation and manuscript

- [x] Update code documentation and manuscript pseudocode to match the verified
  residual targets, greedy pruning, loss/window, depth limits, and grace periods.
- [x] Explicitly map `FastODT` and `xFastODT` to the algorithms and experiments.
- [ ] Review affected manuscript sections for contradictory learning-halt or
  efficiency claims, and run the paper repository's available build checks.

### Phase 5: Reproduce results and measure computational cost

Rerun tooling lives in this repo: `experiments/` (run_evaluation / run_benchmark /
prepare_results, config in `configs/experiments.yaml`) + `jobs/fastodt_rerun.wes`
for the WES scheduler (`../../wes`, `ssh: hpc`); results sync back into
`results/rerun/` and are aggregated into `results/rerun/tables/`.

- [ ] Save corrected configurations, seeds, code revisions, and environment
  details; run a small end-to-end smoke experiment before full reruns.
- [ ] Rerun experiments affected by the corrections using the established data
  splits and evaluation protocol; compare corrected results with prior results.
- [ ] Complete the linked benchmark checklist with training time, prediction
  performance, peak memory, accuracy, and consistent baseline accounting.
- [ ] Update result tables and conclusions with measured findings and record any
  remaining reproduction gaps. Completion criterion: each reported result has
  traceable code/configuration provenance and supporting measurements.

## Priority 1: IncubationBoost correctness

- [x] Fix growing-tree training targets (`src/boosting.py:173–188`).
  Capture `training_target = y - preceding_prefix_prediction` before adding the
  current tree's contribution. The first tree must learn `y`, rather than
  `y - T1(x)`. With two constant targets of 10, the current first tree predicts 5
  after its second update instead of 10.
- [x] Fix the incubation target after pruning (`src/boosting.py:177–193`).
  Use the pre-training prediction of the chain actually retained. Do not include
  rejected trees or recompute predictions after training on the current sample.
  Reproduction: target 10, retained prefix prediction 6, rejected contribution
  -2; incubation currently receives 6 instead of 4. Cover no-pruning cases too.
- [x] Replace signed pruning scores with a chosen nonnegative loss
  (`src/trees/fastodt.py:143–160`). The current score averages signed residuals
  over a zero-initialized 20-entry window, rewarding overprediction and allowing
  errors to cancel. Choose MAE or squared error and specify window initialization.
  Reproduction: target 0 and prediction 100 produce an initial score of -5.
- [x] Add focused behavioral regression tests for both residual corrections and
  pruning-loss behavior, including predictions captured before current-sample
  training and learning-rate handling.

## Priority 2: Tree update semantics

- [x] Decide whether maximum-depth/mature trees continue updating predictions,
  and align both implementations and the manuscript.
  - `src/trees/fastodt.py:215–226` returns before updating maximum-depth leaves.
    A depth-1 leaf inheriting prediction 5 remains at 5 with zero observations
    after training on target 100.
  - `src/trees/xfastodt.py:266–292` stops updating existing nodes after
    `mature_tree` is set.
  - The handoff's revised Algorithm 1 updates leaf statistics before the depth
    guard; that guard prevents structural growth, not prediction updates.
- [x] Implement the intended interval between split attempts in `FastODT`
  (`src/trees/node_hf.py:330–332`, `src/trees/fastodt.py:227–228`). The grace-period
  offset never resets, so attempts occur on every sample after the initial gate.
  Reproduction: grace period 2 and six constant samples produce four attempts.
- [x] Enforce grace-period checks in the active `xFastODT` update path
  (`src/trees/xfastodt.py:275–277`). It currently attempts a split after the first
  sample even with grace period 1000. Align depth-limited grace behavior too.
- [x] Make threshold-equality routing consistent between training and prediction.
  `src/trees/fastodt.py:232` trains left on `<=`, whereas the inherited predictor
  in `src/trees/tree.py:108` routes left only on `<`. A threshold-equal sample
  currently updates the left leaf but predicts from the right leaf.
- [x] Correct lazy-node initialization in `xFastODT`
  (`src/trees/xfastodt.py:320–338`, `src/trees/xnode_hf.py:156–186`). Lazy nodes
  currently copy ancestor observation counts and target sums. Match the handoff's
  intended inherited prediction with zero-initialized observation statistics and
  split-attempt counters. Reproduction: ancestor count 1/sum 10 plus a lazy
  child's first target 20 yields child count 2/sum 30 instead of count 1/sum 20.
- [x] Add focused behavioral tests for the chosen depth, grace-period,
  threshold-equality, and lazy-initialization semantics.

## Priority 3: Manuscript alignment and experiments

- [x] Establish which implementation produced each reported result.
  Algorithm 1's binary-mask routing is implemented by `xFastODT`, while
  `IncubationBoost` uses recursive `FastODT`. Identify the paper's intended
  implementation and make names, descriptions, and experiment provenance clear.
- [x] Describe pruning as greedy prefix pruning in the manuscript
  (`../FastODT-paper/templateArxiv.tex:305` and associated pseudocode).
  The code stops at the first non-improving mature prefix; it does not evaluate
  all prefixes and select a global minimum. Preserve the agreed heuristic.
- [x] Document the actual chosen pruning loss, rolling-window length, and
  initialization; correct the existing claim that signed residuals are absolute
  errors.
- [x] Align Algorithm 1 and surrounding text with the chosen maximum-depth,
  grace-period, inherited-prediction, and lazy-statistics behavior.
  The reviewed `xFastODT` bit order, child indices (`2i`, `2i+1`), and ancestor
  lookup (`index // 2`) already agree with the handoff's MSB-first convention.
- [ ] Complete the computational benchmark checklist in
  `../FastODT-paper/paper.todo.md`: total training time including periodic batch
  refits, single-sample and batched prediction performance, consistent peak-memory
  measurements, predictive accuracy, and reproducibility details (hardware, OS,
  library versions, thread counts, and refitting frequency).
- [ ] Align efficiency claims with measured evidence. Both conventional and
  oblivious routing require O(d) comparison work per sample; practical latency
  benefits require benchmarking the actual implementations.
- [ ] Rerun affected predictive experiments after behavior changes and update
  reported results and computational benchmarks.
