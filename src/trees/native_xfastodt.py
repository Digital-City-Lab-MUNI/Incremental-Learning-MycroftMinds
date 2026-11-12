"""Python interface to the full C xFastODT learner; build with ``make native``.

FastODT's online statistics, histograms, split search and child initialization live
in C. Routing uses a Boolean decision vector and sparse index. A per-node setup
callback reproduces FastODT's NumPy feature sampling.
Native storage is reported explicitly rather than hidden behind a ctypes pointer.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from src.trees.constants import SplitCriterion

DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)
FEATURE_SELECTOR = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.py_object, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
)


class FeatureSamplingContext:
    """Keep the NumPy generator and callback failures alive for the native tree."""

    def __init__(self, random_state: Any) -> None:
        self.rng = np.random.default_rng(random_state)
        self.error: Exception | None = None


def select_features(context: FeatureSamplingContext, total: int, count: int, output: Any) -> int:
    """Match xnode_hf's seeded, sorted subset selection without training in Python."""
    try:
        seed = context.rng.integers(1_000_000)
        selected = np.sort(np.random.default_rng(seed).choice(total, size=count, replace=False))
        for index, feature in enumerate(selected):
            output[index] = int(feature)
        return 0
    except Exception as error:
        # Exceptions must not escape a ctypes callback and leave C with undefined results.
        context.error = error
        return -1


SELECT_FEATURES = FEATURE_SELECTOR(select_features)


@lru_cache(maxsize=1)
def native_library() -> ctypes.CDLL:
    """Load the explicitly built shared library and declare every C ABI signature."""
    path = Path(__file__).with_name("_native") / "libxfastodt.so"
    if not path.exists():
        raise ImportError(f"Native xFastODT library missing at {path}; run `make native` in the repository root")
    library = ctypes.CDLL(str(path))
    library.xfd_abi_version.argtypes = []
    library.xfd_abi_version.restype = ctypes.c_int
    if library.xfd_abi_version() != 2:
        raise ImportError("Native xFastODT ABI mismatch; rebuild with `make native`")
    library.xfd_build_info.argtypes = []
    library.xfd_build_info.restype = ctypes.c_char_p
    library.xfd_create.argtypes = [
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        FEATURE_SELECTOR,
        ctypes.py_object,
    ]
    library.xfd_create.restype = ctypes.c_void_p
    library.xfd_free.argtypes = [ctypes.c_void_p]
    library.xfd_free.restype = None
    library.xfd_partial_fit.argtypes = [ctypes.c_void_p, DOUBLE_POINTER, DOUBLE_POINTER, ctypes.c_size_t, ctypes.c_int]
    library.xfd_partial_fit.restype = ctypes.c_int
    library.xfd_predict.argtypes = [ctypes.c_void_p, DOUBLE_POINTER, ctypes.c_size_t, ctypes.c_int, DOUBLE_POINTER]
    library.xfd_predict.restype = ctypes.c_int
    for name in ("xfd_memory_bytes", "xfd_peak_memory_bytes", "xfd_node_count"):
        function = getattr(library, name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_size_t
    for name in ("xfd_current_depth", "xfd_mature", "xfd_spine_depth"):
        function = getattr(library, name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_int
    library.xfd_split_feature.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.xfd_split_feature.restype = ctypes.c_int
    library.xfd_split_value.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.xfd_split_value.restype = ctypes.c_double
    library.xfd_error.argtypes = [ctypes.c_void_p]
    library.xfd_error.restype = ctypes.c_char_p
    return library


class NativeXFastODT(BaseEstimator):
    """Online C regression tree with the paper xFastODT parameter interface.

    Supports adaptive and mean histograms and absolute/relative variance reduction.
    Depth 0..128 uses FastODT's convention (root at zero, leaves at depth D).
    Histogram inheritance requires max_feature=1 because
    inherited histograms and reselected feature subsets are inconsistent otherwise
    in the Python reference. Custom Python histogram/split callables are unsupported.
    """

    def __init__(
        self,
        depth: int = 25,
        grace_period: int = 10,
        split_confidence: float = 0.01,
        tie_threshold: float = 0.05,
        max_feature: float = 1.0,
        grace_period_depth: int = -1,
        bin_type: str = "adaptive",
        max_bins: int = 16,
        inherit_bins: bool = False,
        split_criterion: str = "vr_rel",
        random_state: Any = None,
        eagerness: float = 0.0,
    ) -> None:
        self.depth = depth
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.max_feature = max_feature
        self.grace_period_depth = grace_period_depth
        self.bin_type = bin_type
        self.max_bins = max_bins
        self.inherit_bins = inherit_bins
        self.split_criterion = split_criterion
        self.random_state = random_state
        self.eagerness = eagerness
        self.error_rates: list[float] = []
        self._handle: int | None = None
        self._initialize()

    def _initialize(self) -> None:
        for name, value, low, high in (
            ("depth", self.depth, 0, 128),
            ("grace_period", self.grace_period, 0, 2**64 - 1),
            ("grace_period_depth", self.grace_period_depth, -1, 2**31 - 1),
            ("max_bins", self.max_bins, 1, 2**31 - 2),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]; got {value!r}")
        if not np.isfinite(self.split_confidence) or not 0 < self.split_confidence <= 1:
            raise ValueError("split_confidence must be finite and in (0, 1]")
        if not np.isfinite(self.tie_threshold) or self.tie_threshold < 0:
            raise ValueError("tie_threshold must be finite and nonnegative")
        if not np.isfinite(self.max_feature) or not 0 < self.max_feature <= 1:
            raise ValueError("max_feature must be finite and in (0, 1]")
        if self.inherit_bins and self.max_feature != 1:
            raise ValueError("inherit_bins=True requires max_feature=1 for reference-compatible histogram ownership")
        if self.bin_type not in (None, "adaptive", "mean", "average"):
            raise ValueError("NativeXFastODT supports bin_type='adaptive' or 'mean'/'average'")
        if self.split_criterion not in (*SplitCriterion.ABS_ALIASES, *SplitCriterion.REL_ALIASES, None):
            raise ValueError("NativeXFastODT supports absolute or relative variance reduction aliases")
        self._sampling_context = FeatureSamplingContext(self.random_state)
        self._handle = native_library().xfd_create(
            int(self.depth),
            int(self.grace_period),
            float(self.split_confidence),
            float(self.tie_threshold),
            float(self.max_feature),
            int(self.grace_period_depth),
            int(self.bin_type in ("mean", "average")),
            int(self.max_bins),
            int(self.inherit_bins),
            int(self.split_criterion not in SplitCriterion.ABS_ALIASES),
            SELECT_FEATURES,
            self._sampling_context,
        )
        if not self._handle:
            raise MemoryError("Unable to allocate the native xFastODT model")

    def _require_open(self) -> int:
        if not self._handle:
            raise ValueError("NativeXFastODT is closed; create a new model")
        return self._handle

    def _check_status(self, status: int) -> None:
        if status == 0:
            return
        if self._sampling_context.error is not None:
            raise RuntimeError("Native feature sampling failed") from self._sampling_context.error
        message = native_library().xfd_error(self._handle).decode("utf-8")
        if "allocation" in message:
            self.close()
            raise MemoryError(message)
        raise ValueError(message)

    @staticmethod
    def _features(values: Any) -> np.ndarray:
        features = np.asarray(values, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.ndim != 2 or features.shape[1] < 1 or features.shape[1] >= 2**31:
            raise ValueError("X must be a 2D numeric array with at least one feature")
        return np.ascontiguousarray(features)

    def partial_fit(self, X: Any, y: Any, verbose: bool = False) -> NativeXFastODT:
        """Update every row online in C, retaining all histogram and node state."""
        features = self._features(X)
        targets = np.ascontiguousarray(np.asarray(y, dtype=np.float64).reshape(-1))
        if len(targets) != len(features):
            raise ValueError("X and y must have the same number of samples")
        status = native_library().xfd_partial_fit(
            self._require_open(),
            features.ctypes.data_as(DOUBLE_POINTER),
            targets.ctypes.data_as(DOUBLE_POINTER),
            len(features),
            features.shape[1],
        )
        self._check_status(status)
        self.n_features_in_ = features.shape[1]
        return self

    def fit(self, X: Any, y: Any) -> NativeXFastODT:
        """Reset the learner and train online over the supplied rows."""
        self.close()
        self._initialize()
        self.error_rates.clear()
        return self.partial_fit(X, y)

    def predict(self, X: Any) -> np.ndarray:
        """Predict by Boolean-vector index with read-only ancestor fallback."""
        features = self._features(X)
        output = np.empty(len(features), dtype=np.float64)
        status = native_library().xfd_predict(
            self._require_open(),
            features.ctypes.data_as(DOUBLE_POINTER),
            len(features),
            features.shape[1],
            output.ctypes.data_as(DOUBLE_POINTER),
        )
        self._check_status(status)
        return output

    def learn_one(self, x: Any, y: float) -> NativeXFastODT:
        """Update one array-like or insertion-ordered feature dictionary."""
        return self.partial_fit(list(x.values()) if isinstance(x, dict) else x, [y])

    def predict_one(self, x: Any) -> float:
        """Predict one array-like or insertion-ordered feature dictionary."""
        return float(self.predict(list(x.values()) if isinstance(x, dict) else x)[0])

    def measure_error(self, y_baseline: float, y_predicted: float) -> float:
        """Maintain the same 20-observation rolling MAE as Python xFastODT."""
        self.error_rates.append(abs(y_baseline - y_predicted))
        if len(self.error_rates) > 20:
            self.error_rates.pop(0)
        return float(np.mean(self.error_rates))

    @property
    def native_memory_bytes(self) -> int:
        """Requested live bytes owned by C, excluding allocator overhead and inputs."""
        return int(native_library().xfd_memory_bytes(self._handle)) if self._handle else 0

    @property
    def native_build_info(self) -> str:
        """Compiler identity and C ABI version used by this loaded library."""
        return native_library().xfd_build_info().decode("utf-8")

    @property
    def native_peak_memory_bytes(self) -> int:
        """Peak requested native bytes, including temporary hash-table growth."""
        return int(native_library().xfd_peak_memory_bytes(self._handle)) if self._handle else 0

    @property
    def n_nodes_(self) -> int:
        """Number of allocated sparse nodes; prediction never creates nodes."""
        return int(native_library().xfd_node_count(self._require_open()))

    @property
    def current_depth(self) -> int:
        """Number of active shared routing splits."""
        return int(native_library().xfd_current_depth(self._require_open()))

    @property
    def mature_tree(self) -> bool:
        """Whether structural growth stopped at the configured depth limit."""
        return bool(native_library().xfd_mature(self._require_open()))

    @property
    def splits_features(self) -> list[int | None]:
        """Expose registered split features for parity diagnostics."""
        handle = self._require_open()
        features = [native_library().xfd_split_feature(handle, depth) for depth in range(self.depth)]
        return [None if feature < 0 else feature for feature in features]

    @property
    def splits_values(self) -> list[float | None]:
        """Expose registered shared thresholds for parity diagnostics."""
        handle = self._require_open()
        return [
            None if feature is None else native_library().xfd_split_value(handle, depth)
            for depth, feature in enumerate(self.splits_features)
        ]

    def is_filled(self) -> bool:
        """Match FastODT's left-preferring filled-tree check."""
        return native_library().xfd_spine_depth(self._require_open()) == self.depth

    def how_deep(self) -> int:
        """Match FastODT's existing left-preferring depth helper."""
        return int(native_library().xfd_spine_depth(self._require_open()))

    def close(self) -> None:
        """Free all native node/histogram allocations; repeated calls are safe."""
        if self._handle:
            native_library().xfd_free(self._handle)
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, ImportError, TypeError):
            pass
