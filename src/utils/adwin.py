from collections import deque
import numpy as np
import math

class ADWIN:

    def __init__(self, delta = 0.0002):
        self.delta = delta
        self.window = deque()
        self.n = 0
        self.change_detected = False

    def add_sample(self,sample):
        self.window.append(sample)
        self.n += 1
        self.change_detected = False

        if len(self.window) > 2:
            for division_index in range(1,len(self.window)):
                window_list = np.array(self.window)
                left_part = window_list[:division_index]
                right_part = window_list[division_index:]

                mean_left = np.sum(left_part)/len(left_part)
                mean_right = np.sum(right_part)/len(right_part)

                e = self.bound(len(left_part),len(right_part))
                if abs(mean_left - mean_right) > e:
                    for _ in range(division_index):
                        self.window.popleft()
                        self.n -= 1
                    
                    self.change_detected = True
                    return

    def bound(self,n_left,n_right):
        harmonic_mean_n = 1/((1/n_left)+(1/n_right))
        return math.sqrt(1/(2*harmonic_mean_n) / self.delta)
    
    def isDrift(self):
        return self.change_detected