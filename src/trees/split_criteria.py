"""
Split criteria functions for incremental decision trees.
"""

def variance_reduction(left_stats: dict, right_stats: dict, total_n: int, total_sum_y: float, total_sum_y2: float) -> float:
    """
    Compute absolute variance reduction for regression splits. The R in HB should be [0 , (y_max-y_min)^2/4], based on formula

    Args:
        left_stats (dict): Statistics for the left split. Must contain 'count', 'sum_y', 'sum_y2'.
        right_stats (dict): Statistics for the right split. Must contain 'count', 'sum_y', 'sum_y2'.
        total_n (int): Total number of samples.
        total_sum_y (float): Sum of target values for all samples.
        total_sum_y2 (float): Sum of squared target values for all samples.

    Returns:
        float: Absolute variance reduction after the split.
    """
    
    left_n = left_stats['count']
    right_n = right_stats['count']
    if left_n == 0 or right_n == 0:
        return 0
    
    # Parent variance
    var_parent = (total_sum_y2 / total_n) - (total_sum_y / total_n) ** 2
    
    # Left and right variances
    var_left = (left_stats['sum_y2'] / left_n) - (left_stats['sum_y'] / left_n) ** 2
    var_right = (right_stats['sum_y2'] / right_n) - (right_stats['sum_y'] / right_n) ** 2
    
    # Weighted variance after split
    var_children = (left_n / total_n) * var_left + (right_n / total_n) * var_right
    return var_parent - var_children  


def variance_reduction_relative(left, right, total_n, total_sum_y, total_sum_y2):
    """
    Relative (= percentage) variance-reduction in [0 , 1]. The R in HB now should be ∈ [0 , 1]
    Args:
        left (dict): Statistics for the left split.
        right (dict): Statistics for the right split.
        total_n (int): Total number of samples.
        total_sum_y (float): Sum of target values for all samples.
        total_sum_y2 (float): Sum of squared target values for all samples.
    Returns:
        float: Relative variance reduction after the split.
    """
    abs_gain = variance_reduction(left, right, total_n, total_sum_y, total_sum_y2)
    var_parent = (total_sum_y2 / total_n) - (total_sum_y / total_n) ** 2

    return abs_gain / var_parent if var_parent > 0 else 0.0

def get_split_criterion(name):
    """
    Return a split criterion function by name.
    Supported names: 'vr', 'vr_abs', 'vr_rel', 'vr_relative'.
    """
    if name in ['vr', 'vr_abs', 'reduction_abs']:
        return variance_reduction
    elif name in ['vr_rel', 'vr_relative', 'reduction_rel'] or name is None:
        return variance_reduction_relative
    else:
        raise ValueError(f"[ERROR]: Unknown split criterion: {name}")
