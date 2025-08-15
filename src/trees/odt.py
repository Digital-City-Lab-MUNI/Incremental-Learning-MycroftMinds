from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from tree import BaseTree, Node
import numpy as np

class ODTree(BaseTree):

    def __init__(self,depth):
        super(ODTree, self).__init__(depth)
        raise NotImplementedError  # it is obsolete 
