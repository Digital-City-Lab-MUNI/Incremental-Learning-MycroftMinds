from collections import deque
import numpy as np
import math


class ADWIN:

    def __init__(self, delta=0.0002):
        self.delta = delta
        self.window = deque()
        self.n = 0
        self.change_detected = False
        self.max_windows = 100

    def add_sample(self, sample: float):
        """
        Add a sample to the ADWIN window and check for change detection.

        Parameters
        ----------
        sample : float
            The new sample to add to the window.

        Notes
        -----
        Adding a sample will not immediately trigger a change detection. If a change is
        detected, the window is updated accordingly.
        """
        self.window.append(sample)
        self.n += 1
        self.change_detected = False

        if len(self.window) > 2:
            for division_index in range(1, len(self.window)):
                window_list = np.array(self.window)
                left_part = window_list[:division_index]
                right_part = window_list[division_index:]

                mean_left = np.sum(left_part) / len(left_part)
                mean_right = np.sum(right_part) / len(right_part)

                e = self.bound(len(left_part), len(right_part))
                if abs(mean_left - mean_right) > e:
                    for _ in range(division_index):
                        self.window.popleft()
                        self.n -= 1
                    self.change_detected = True
                    return
        if len(self.window) > self.max_windows:
            self.window.popleft()
            self.n -= 1

    def bound(self, n_left, n_right):
        """
        Calculate the statistical bound for change detection.

        Parameters
        ----------
        n_left : int
            Number of samples in the left partition.
        n_right : int
            Number of samples in the right partition.

        Returns
        -------
        float
            The calculated bound value.
        """
        harmonic_mean_n = 1 / ((1 / n_left) + (1 / n_right))
        return math.sqrt(1 / (2 * harmonic_mean_n) / self.delta)

    def isDrift(self):
        """
        Check if a change (drift) has been detected.

        Returns
        -------
        bool
            True if change detected, False otherwise.
        """
        return self.change_detected
