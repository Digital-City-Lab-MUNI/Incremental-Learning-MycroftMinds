"""Shared evaluation constants used across metrics and visualization.

Import this module as an alias (e.g., `import src.evaluation.constants as C`)
and access grouped constants like `C.Metric.RMSE`, `C.Event.SPIKES_IMMEDIATE`, etc.

Avoid importing project modules here to prevent circular dependencies.
"""


class Metric:
    """Metric name keys used in metrics dicts.

    Includes regression metrics and per-event sub-metrics.
    """

    RMSE = "rmse"
    MAE = "mae"
    MAPE = "mape"
    DIR_ACCURACY = "dir_accuracy"
    FIRST_DIFF_CORR = "first_diff_corr"

    # Per-event submetrics
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"


class Event:
    """Event buckets used for spike/drop detection results."""

    SPIKES_IMMEDIATE = "spikes_immediate"
    DROPS_IMMEDIATE = "drops_immediate"
    SPIKES_DELAYED = "spikes_delayed"
    DROPS_DELAYED = "drops_delayed"


class Agg:
    """Aggregation labels for metrics dictionaries."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    DAYS_28 = "28_days"


class Freq:
    """Pandas frequency aliases used for resampling/aggregation."""

    H = "H"
    D = "D"
    W = "W"
    H1 = "1H"
    D1 = "1D"
    W1 = "1W"
    D28 = "28D"


class EventMetric:
    """Composite keys for accessing per-event metric values in aggregated summaries."""

    SPIKES_IMMEDIATE_PRECISION = f"{Event.SPIKES_IMMEDIATE}.{Metric.PRECISION}"
    SPIKES_IMMEDIATE_RECALL = f"{Event.SPIKES_IMMEDIATE}.{Metric.RECALL}"
    SPIKES_IMMEDIATE_F1 = f"{Event.SPIKES_IMMEDIATE}.{Metric.F1}"

    DROPS_IMMEDIATE_PRECISION = f"{Event.DROPS_IMMEDIATE}.{Metric.PRECISION}"
    DROPS_IMMEDIATE_RECALL = f"{Event.DROPS_IMMEDIATE}.{Metric.RECALL}"
    DROPS_IMMEDIATE_F1 = f"{Event.DROPS_IMMEDIATE}.{Metric.F1}"

    SPIKES_DELAYED_PRECISION = f"{Event.SPIKES_DELAYED}.{Metric.PRECISION}"
    SPIKES_DELAYED_RECALL = f"{Event.SPIKES_DELAYED}.{Metric.RECALL}"
    SPIKES_DELAYED_F1 = f"{Event.SPIKES_DELAYED}.{Metric.F1}"

    DROPS_DELAYED_PRECISION = f"{Event.DROPS_DELAYED}.{Metric.PRECISION}"
    DROPS_DELAYED_RECALL = f"{Event.DROPS_DELAYED}.{Metric.RECALL}"
    DROPS_DELAYED_F1 = f"{Event.DROPS_DELAYED}.{Metric.F1}"


class Format:
    """Formatting constants for visualization and reporting."""

    DATETIME = "%Y-%m-%d %H:%M:%S"


class Axis:
    """Axis label constants for plots."""

    TIME = "Time"
    TOTAL_POWER_ACTIVE = "Total Power Active (kWh)"


class Markers:
    """Marker format strings used in plots."""

    ORIGINAL = "o"
    PREDICTED = "x"
    OUTLIERS = "ro"  # red circles with no face by default in code
    SPIKES = "^r"  # red triangles up
    DROPS = "vb"  # blue triangles down


class Labels:
    """Legend/label strings used in plots."""

    ORIGINAL = "Original Values"
    PREDICTED = "Predicted Values"
    OUTLIERS_TMPL = "Outliers (>{percent}% diff)"
    SPIKES = "Spikes"
    DROPS = "Drops"


class MarkerSize:
    """Common marker sizes used across plots."""

    SMALL = 6
    LARGE = 10


__all__ = [
    "Metric",
    "Event",
    "Agg",
    "Freq",
    "EventMetric",
    "Format",
    "Axis",
    "Markers",
    "Labels",
    "MarkerSize",
]
