# Native xFastODT learner

Purpose: compare pointer-based FastODT with Python/C Boolean-vector routing while
keeping the learning algorithm unchanged, including online training state.

## Build and run on the computation server

Prerequisites: Linux, a C11 compiler (`cc`, GCC or Clang), and the existing Python
project environment. No Python/NumPy development headers or new packages are needed.
From the repository root with processed datasets linked under `data/`:

```bash
make exp-native
```

This builds `src/trees/_native/libxfastodt.so` locally for the server architecture,
then runs the Air Quality validation/test protocol and serial resource benchmarks
for `FastODT`, `xFastODT` and `xFastODT_C`. Results go to `results/rerun/native/raw/od_aq_1h/`
and `results/rerun/native/tables/{all_metrics.csv,all_benchmark.csv,summary.md}`.
Production runtime has not been measured. Transfer source, not a locally compiled
`.so`, to a server with a different architecture or operating system.

Override `DATA_ROOT`, `NATIVE_RESULTS`, and `NATIVE_DATASETS` as needed, for example
`make exp-native NATIVE_DATASETS="od_aq_1h od_friedman_1h"`.
Resume with `SKIP_EXISTING=1`; seed, source, config, data identities and shared-library
bytes participate in fingerprints. `make exp-all` builds the C library and includes
`xFastODT_C` alongside the other nine models on all six hourly datasets, writing to
the main `results/rerun/` directory. The all-configured-model `make exp-matrix` also
builds the C library.

## Python interface

```python
import numpy as np
from src.trees.native_xfastodt import NativeXFastODT

features = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
targets = np.array([0.0, 2.0, 3.0])
model = NativeXFastODT(depth=10, random_state=42)
model.partial_fit(features, targets)
print(model.predict(features))
print(model.native_memory_bytes, model.native_peak_memory_bytes)
model.close()
```

`partial_fit` updates online; `fit` resets and updates; `predict` returns a NumPy
vector. `learn_one`/`predict_one` accept vectors or insertion-ordered dictionaries.
Input is converted to contiguous float64; nonfinite inputs and inconsistent feature
counts are rejected. `close` releases native allocations; destruction also releases
them. Do not access the same instance concurrently from multiple threads.

## What runs in C

- Sparse nodes keyed by depth and a 128-bit MSB-first routing index.
- Read-only ancestor fallback for prediction; nodes are materialized during training.
- Node target statistics, adaptive-bin insertion/merging, running-mean histograms.
- Absolute/relative variance reduction, Hoeffding tests and grace-period resets.
- Shared depth splits and batch/single-row prediction/update loops.
- A Boolean vector of independent depth comparisons, then MSB-first mask packing;
  no SIMD-specific code, CPU dispatch or instruction-set requirement is added.

A Python callback samples each node's feature subset using the exact NumPy RNG
sequence used by the reference. This is per-node setup, not Python histogram
updates or split search. Its callback/binding overhead is included in timings.
All three implementations receive `protocol.seed` through explicit `random_state`.

Supported modes: `adaptive`, `mean`/`average`; absolute/relative variance criteria;
depth 0..128. The behavioral reference is FastODT: `depth=D` allows D shared splits
and leaves at depth D, with the root at zero. Maximum-depth leaves continue to
update statistics without attempting new splits.
Histogram inheritance is supported only with `max_feature=1`. Equiwidth histograms
(which need external range initialization) and custom histogram/criterion callables
are rejected. These modes are not used by the six configured paper datasets.
This initial binding does not serialize fitted C model state for pickling.

## Routing-only equivalence

Python xFastODT inherits FastODT's node class, histogram updates, split selection,
grace-period handling and child initialization. Its indexed containers refer to
the same pointer nodes. C duplicates those numerical rules using native storage.
Both indexed versions evaluate the registered depth comparisons before node lookup.
Prediction does not create nodes; missing indexed paths return the nearest existing
ancestor's value. Training applies cached depth splits and creates both children,
including inherited histograms when enabled, exactly as FastODT does.

This corrects the old xFastODT off-by-one depth limit and prediction-time node
creation. Old Python/C xFastODT results and the earlier synthetic size measurement
are therefore obsolete. Rebuild the library (ABI 2) and rerun before comparing costs.

## Memory definitions

The allocator counts every requested byte owned by the C learner: tree metadata,
split arrays, sparse hash table (including spare capacity), nodes, histogram arrays,
bin capacity and feature-selection scratch storage. `native_peak_memory_bytes`
includes transient old/new hash tables during growth. These counters exclude malloc
headers/fragmentation, shared-library code/static storage and borrowed input arrays.

`model_size_mb` = Pympler Python-wrapper size + live owned native bytes, in MiB.
`python_model_size_mb` and `native_model_size_mb` refer to that same largest final
device model; `native_peak_model_size_mb` is the maximum native peak across devices.
For other estimators, native bytes are zero if no explicit counter is provided:
that does not mean third-party estimators allocate no native memory. This change
does not validate scikit-learn native-tree accounting or fix process RSS floors.

JSON records node counts/routing depths, compiler identity/ABI, parameters, seed,
revision and environment. Process RSS remains a separate measurement. Compare
accuracy and model structure before interpreting memory or latency differences.

## Verification

`make native-check` runs a standalone C stress test under AddressSanitizer/UBSan.
`uv run python -m unittest tests.test_xfastodt_parity tests.test_native_xfastodt tests.test_experiment_benchmarks`
checks streamed predictions/splits against FastODT, node/histogram/RNG state for the
Python variant, missing indexed paths, equality boundaries,
feature subsets, inheritance, mature updates, paths beyond 64 bits, refitting,
input validation, explicit release, seed propagation and benchmark memory fields.
Default compilation uses `-O3 -std=c11 -fPIC -Wall -Wextra -Werror -ffp-contract=off`.
If changing `CC`/`NATIVE_CFLAGS` after a build, use `make -B native` to rebuild.
