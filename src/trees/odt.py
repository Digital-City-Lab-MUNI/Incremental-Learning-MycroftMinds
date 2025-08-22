"""Oblivious Decision Tree (ODT) implementation for regression.

Applies a single (feature, threshold) split uniformly across all nodes at a
given depth level, enabling fast traversal and compact models.
"""

from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from tree import BaseTree, Node
import numpy as np

class ODTree(BaseTree):
    """Oblivious depth-limited regression tree (same split per level)."""

    def __init__(self,depth):
        super(ODTree, self).__init__(depth)
        raise NotImplementedError  # it is obsolete 
