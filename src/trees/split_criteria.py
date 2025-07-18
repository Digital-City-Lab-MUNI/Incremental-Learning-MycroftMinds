"""
Split criteria functions for incremental decision trees.
"""

def variance_reduction(left_stats: dict, right_stats: dict, total_n: int, total_sum_y: float, total_sum_y2: float) -> float:
    """
    Compute variance reduction for regression splits.
    left_stats, right_stats: dicts with keys 'count', 'sum_y', 'sum_y2'
    total_n: total number of samples
    total_sum_y: sum of y for all samples
    total_sum_y2: sum of y^2 for all samples
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
    return var_parent - var_children  # variance reduction

def get_split_criterion(name):
    """
    Return a split criterion function by name.
    Supported names: 'variance'.
    """
    if name == 'variance' or name is None:
        return variance_reduction
    else:
        raise ValueError(f"[ERROR]: Unknown split criterion: {name}")
