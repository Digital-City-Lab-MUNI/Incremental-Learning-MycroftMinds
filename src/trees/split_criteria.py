import numpy as np

def _variance_reduction(self, left_y, left_y_squared, left_n, right_y, right_y_squared, right_n):
        # Also known as MSE gain
        if left_n == 0 or right_n == 0:
            return 0
        # Parent variance
        var_parent = (self.sum_y_squared / self.samples_seen) - (self.sum_y / self.samples_seen) ** 2
        # Left and right variances
        var_left = (left_y_squared / left_n) - (left_y / left_n) ** 2
        var_right = (right_y_squared / right_n) - (right_y / right_n) ** 2
        # Weighted variance after split
        var_children = (left_n / self.samples_seen) * var_left + (right_n / self.samples_seen) * var_right

        return var_parent - var_children  # variance reduction
    
def _friedman_mse(self, left_y, left_n, right_y, right_n):
    
    if left_n == 0 or right_n == 0:
        return 0
    
    mean_left = left_y / left_n
    mean_right = right_y / right_n
    diff_squared = (mean_left - mean_right) **2

    return ((left_n * right_n) / (left_n + right_n)) * diff_squared

def _poisson_deviance(self,
                        left_y, left_y_log, left_n,
                        right_y, right_y_log, right_n):
    # Poisson deviance reduction
    # Requires: all y > 0, and self.sum_y_log = sum(y * log(y)) over the parent
    if left_n == 0 or right_n == 0:
        return 0.0

    # parent mean
    mu = self.sum_y / self.samples_seen
    # parent deviance
    D_parent = 2 * (self.sum_y_log - self.sum_y * np.log(mu))

    # left deviance
    mu_L = left_y / left_n
    D_left = 2 * (left_y_log - left_y * np.log(mu_L))

    # right deviance
    mu_R = right_y / right_n
    D_right = 2 * (right_y_log - right_y * np.log(mu_R))

    # weighted children deviance
    D_children = (left_n  / self.samples_seen) * D_left \
                + (right_n / self.samples_seen) * D_right

    return D_parent - D_children


def _mean_absolute_error(self, left_values, right_values):
    # MAE gain (reduction in absolute deviation around median)
    if len(left_values) == 0 or len(right_values) == 0:
        return 0.0

    # parent median & total absolute deviation
    parent = np.array(self.data_from_before_split) # is this correct variable to use? perhaps val would be better
    med_p = np.median(parent)
    mae_p = np.sum(np.abs(parent - med_p))

    # left median & deviation
    L = np.array(left_values)
    med_L = np.median(L)
    mae_L = np.sum(np.abs(L - med_L))

    # right median & deviation
    R = np.array(right_values)
    med_R = np.median(R)
    mae_R = np.sum(np.abs(R - med_R))

    # reduction in total absolute deviation
    return mae_p - (mae_L + mae_R)