"""Split criteria functions for incremental decision trees."""

import src.trees.constants as C

class BaseReduction:
    
    @classmethod
    def reduce(
        cls,
        left_stats: dict,
        right_stats: dict,
        total_n: int,
        total_sum_y: float,
        total_sum_y2: float,
    ) -> float:
        raise NotImplementedError("Subclasses must implement reduce()")

    @classmethod
    def reduce(
        cls,
        left_stats: dict,
        right_stats: dict,
        total_n: int,
        total_sum_y: float,
        total_sum_y2: float,
    ) -> float:
        raise NotImplementedError("Subclasses must implement R()")

class VarianceReduction:

    @classmethod
    def reduce(
        cls,
        left_stats: dict,
        right_stats: dict,
        total_n: int,
        total_sum_y: float,
        total_sum_y2: float,
    ) -> float:
        """Absolute variance reduction for a proposed split.

        Let :math:`Var_P` be the parent variance and :math:`Var_L`, :math:`Var_R`
        the child variances. The weighted child variance is
        ``Var_children = (n_L/ n_total)*Var_L + (n_R/ n_total)*Var_R`` and the
        reduction is ``Var_P - Var_children``.

        This absolute reduction is used with a Hoeffding bound range
        :math:`R = 0.25 (y_{max} - y_{min})^2` (Popoviciu upper bound) when applying
        the Hoeffding test.

        Parameters
        ----------
        left_stats : dict
            Dictionary with keys ``count``, ``sum_y``, ``sum_y2`` for the left
            partition.
        right_stats : dict
            Dictionary with keys ``count``, ``sum_y``, ``sum_y2`` for the right
            partition.
        total_n : int
            Total number of samples at the parent node.
        total_sum_y : float
            Sum of targets at the parent.
        total_sum_y2 : float
            Sum of squared targets at the parent.

        Returns
        -------
        float
            Absolute variance reduction (non-negative, can be 0 if no split or one
            side empty).
        """
        left_n = left_stats[C.StatKey.COUNT]
        right_n = right_stats[C.StatKey.COUNT]
        if left_n == 0 or right_n == 0:
            return 0

        # Parent variance
        var_parent = (total_sum_y2 / total_n) - (total_sum_y / total_n) ** 2

        # Left and right variances
        var_left = (left_stats[C.StatKey.SUM_Y2] / left_n) - (left_stats[C.StatKey.SUM_Y] / left_n) ** 2
        var_right = (right_stats[C.StatKey.SUM_Y2] / right_n) - (right_stats[C.StatKey.SUM_Y] / right_n) ** 2

        # Weighted variance after split
        var_children = (left_n / total_n) * var_left + (right_n / total_n) * var_right
        return var_parent - var_children

    @classmethod
    def R(cls, *args):
        y_max, y_min = args
        return 0.25 * (y_max - y_min) ** 2  # Popoviciu upper bound

class VarianceReductionRelative:

    @classmethod
    def reduce(
        cls,
        left: dict, right: dict, total_n: int, total_sum_y: float, total_sum_y2: float
    ) -> float:
        """Relative (percentage) variance reduction in ``[0, 1]``.

        Defined as ``(absolute_variance_reduction / parent_variance)`` with 0 for
        degenerate parent variance (no reduction at all). Useful when a fixed Hoeffding range of ``R=1``
        is desired (normalized criterion).

        Parameters
        ----------
        left : dict
            Left partition statistics (``count``, ``sum_y``, ``sum_y2``).
        right : dict
            Right partition statistics.
        total_n : int
            Total number of samples at the parent.
        total_sum_y : float
            Sum of targets at the parent.
        total_sum_y2 : float
            Sum of squared targets at the parent.

        Returns
        -------
        float
            Relative variance reduction (0 if parent variance is zero).
        """
        abs_gain = VarianceReduction.reduce(left, right, total_n, total_sum_y, total_sum_y2)
        var_parent = (total_sum_y2 / total_n) - (total_sum_y / total_n) ** 2

        return abs_gain / var_parent if var_parent > 0 else 0.0

    @classmethod
    def R(cls, *args):
        return 1.0


def get_split_criterion(name: str | None) -> callable:
    """Return a split criterion function by name.

    Parameters
    ----------
    name : str or None
        One of ``{'vr','vr_abs','reduction_abs','vr_rel','vr_relative',
        'reduction_rel'}``. ``None`` maps to relative reduction.

    Returns
    -------
    callable
        The variance reduction function (absolute or relative).

    Raises
    ------
    ValueError
        If the name is not recognized.
    """
    if name in C.SplitCriterion.ABS_ALIASES:
        return variance_reduction
    elif name in C.SplitCriterion.REL_ALIASES or name is None:
        return variance_reduction_relative
    else:
        raise ValueError(f"[ERROR]: Unknown split criterion: {name}")
