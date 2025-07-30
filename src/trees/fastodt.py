from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt 
from src.trees.tree import BaseTree, Node
from src.trees.node_hf import NodeHoeffdingBound
from src.utils.utils import ensure_2d_array
from src.trees.split_criteria import get_split_criterion
from src.trees.histogram import get_histogram_type
import numpy as np
import pandas as pd
import tqdm
import time
import os

class FastODT(BaseTree):
    def __init__(self, 
                 depth=25, 
                 grace_period=10, 
                 split_confidence=0.01, 
                 tie_threshold=0.05,
                 max_feature=1.0,
                 eagerness=0.0, 
                 grace_period_depth=-1, 
                 bin_type='adaptive', 
                 max_bins=16,
                 inherit_bins=False, 
                 split_criterion='vr_rel'):
        """
        Initialize the FastODT tree model with parameters.

        Parameters:
        ----------
        depth : int, optional (default=5)
            Maximum depth of the decision tree. Controls the number of splits from the root to a leaf.

        grace_period : int, optional (default=10)
            Number of samples a node should observe before attempting to split.

        split_confidence : float, optional (default=1.0)
            Confidence level for splitting criterion. A higher value makes splitting more conservative.

        max_feature : float, optional (default=1.0)
            Fraction of features to consider for splitting at each node.

        eagerness : float, optional (default=0.0)
            Parameter to control the eagerness to split for Hoeffding Bound nodes.

        grace_period_depth : int, optional (default=-1)
            Grace period value to control to how deep nodes grace period should still apply. 
            If set to 1, only first node will use grace_period, 
            If set to 100, first 100 nodes will use grace period,
            If set to -1, this feature is disabled (set to over 999999999999999 internally).

        bin_type : str or class, optional (default='adaptive')
            String name of histogram type (e.g., 'adaptive', 'equiwidth', 'mean') or a histogram class. If None or 'adaptive', uses AdaptiveHistogram.

        max_bins : int, optional (default=32)
            Maximum number of bins per feature per node.

        inherit_bins : bool, optional (default=False)
            If True, pass histograms from parent to children on split. If False, children start with empty histograms.

        split_criterion : str or callable, optional (default=None)
            String name of split criterion (e.g., 'variance') or a callable. If None, uses default for node type.

        Attributes:
        ----------
        root : Node
            The root node of the tree, initialized according to the chosen `option`.

        splits : list
            A list initialized to hold split information for each depth level.

        Raises:
        ------
        None
        """
        self.grace_period = grace_period  # Number of samples before checking for a split
        self.split_confidence = split_confidence  # Confidence level for splitting
        self.depth = depth
        self.splits_values = [None] * depth
        self.splits_features = [None] * depth
        self.depths_containers = []

        self.error_rates = np.zeros(20).tolist() 

        self.max_feature = max_feature
        self.grace_period_depth = grace_period_depth
        if self.grace_period_depth == -1:
            self.grace_period_depth = 999999999999999

        self.tie_threshold = tie_threshold
        if isinstance(bin_type, str) or bin_type is None:
            bin_type = get_histogram_type(bin_type)

        if isinstance(split_criterion, str):
            split_criterion = get_split_criterion(split_criterion)
        # create the root node
        self.depths_containers.append([])
        self.depths_containers[0].append(
            NodeHoeffdingBound(
                split_confidence=self.split_confidence,
                tie_threshold=self.tie_threshold,
                max_feature=self.max_feature,
                eagerness=eagerness,
                grace_period=grace_period,
                bin_type=bin_type,
                max_bins=max_bins,
                inherit_bins=inherit_bins,
                split_criterion=split_criterion
            )
        )

    def measure_error(self, y_baseline, y_predicted):
        self.error_rates.pop(0)
        self.error_rates.append(y_baseline - y_predicted)
        return np.mean(self.error_rates)

    def partial_fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        X = ensure_2d_array(X) # 10.1 -> [10.1] -> (1,1)
        y = np.asarray(y)
        if y.ndim == 0:
            y = np.array([y])
        if X.shape[0] != len(y):
            raise ValueError(f"X and y must have the same number of samples. X: {X.shape}, y: {y.shape}")

        self._update_tree_v(None, X, y, depth=0)

    def learn_one(self, x, y: float):
        """
        Incrementally update the tree with a single sample (x, y).
        x must be a 1D array-like structure that can be converted to a numpy array (or dict).
        """
        if isinstance(x, dict):
            x = np.asarray(list(x.values()))
        elif isinstance(x, pd.DataFrame):
            x = np.asarray(x.values)
        else:
            x = np.asarray(x)
        if x.ndim == 2 and x.shape[0] == 1:
            x = x[0]
        self._update_tree(self.root, x, y, depth=0)

    def predict_one(self, x):
        """
        Predict the output for a single sample x.
        x must be a 1D array-like structure that can be converted to a numpy array (or dict).
        """
        if isinstance(x, dict):
            x = np.asarray(list(x.values()))
        else:
            x = np.asarray(x)
        if x.ndim == 2 and x.shape[0] == 1:
            x = x[0]
        return self.predict(x)

    def _update_tree_v(self, node, x: np.ndarray, y: float, depth=0):
        # Check if we reached the maximum depth
        if depth >= self.depth:
            return

        active_splits_features = list(filter(None.__ne__, self.splits_features))
        active_splits_values = list(filter(None.__ne__, self.splits_values))

        x_split_values = x[:, active_splits_features]
        decision_array = x_split_values[:] > active_splits_values
        print(decision_array)

        for decision_vector in decision_array:
            if len(decision_vector) == 0:
                node_index = 0
                node_depth = 0
            else:
                # Convert to integer
                node_index = int(''.join(str(int(decision)) for decision in decision_vector), 2)
                node_depth = len(active_splits_features)

            node = self.depths_containers[node_depth][node_index]
            if node is None:
                node = self.__find_node(node_depth, node_index)

            node.update_stats(x, y)
    
            best_split, _ = node.find_best_split() # here happens hf bound calculation
            if best_split[0] is not None:
                self.splits_features[depth] = best_split
                self.depths_containers[depth] = []
                for _ in range(2**(depth)):
                    self.depths_containers[depth].append(None)
                left, right = node.produce_split(best_split, self.split_confidence)
                self.depths_containers[depth][left.index] = left
                self.depths_containers[depth][right.index] = right

    def __find_node(self, depth, index):
        node = self.depths_containers[depth][index]
        if node is None:
            parent_node = None
            # check if we are in left or right node
            if index % 2 == 0:
                parent_node = self.__find_node(depth - 1, index/2)
            else:
                parent_node = self.__find_node(depth - 1, index/2 - 1)
            node = parent_node
        return node

    def _update_tree(self, node, x: np.ndarray, y: float, depth=0):
        # Check if we reached the maximum depth
        if depth >= self.depth:
            return
        # if node is a leaf, update its stats and check for split
        if node.left is None and node.right is None:
            node.update_stats(x, y)
            if node.reached_grace_period() or self.grace_period_depth < depth:  # check if grace period is over
                self._attempt_split(node, depth)
        
        # traverse the tree to reach the correct leaf
        elif node.feature is not None:
            if x[node.feature] <= node.threshold:
                self._update_tree(node.left, x, y, depth + 1)
            elif x[node.feature] > node.threshold:
                self._update_tree(node.right, x, y, depth + 1)

    def _attempt_split(self, node, depth: int):
        best_split, _ = node.find_best_split() # here happens hf bound calculation
        if self.splits_features[depth] is not None:
            node.split(self.splits_features[depth], self.split_confidence)
            node.left.update_from_parent_node(node)
            node.right.update_from_parent_node(node)
        
        elif best_split[0] is not None:
            self.splits_features[depth] = best_split 
            node.split(best_split, self.split_confidence)
            node.left.update_from_parent_node(node)
            node.right.update_from_parent_node(node)

if __name__ == "__main__":

    X = np.load('./smoke_data/X.npy', allow_pickle=True)
    Y = np.load('./smoke_data/y.npy', allow_pickle=True)
    print(X.shape, Y.shape)
    X = X[:,2:]
    print(type(X), type(Y), X[0])

    # Test if model will be partially filled after receiving only one sample
    start = time.time()
    model_hf = FastODT(
        depth=20, 
        grace_period=10,
        split_confidence=0.0001
    )
    model_hf.partial_fit(X[:,:], Y, verbose=True)
    print(f"It took: {time.time() - start}s")
    # only one depth split at most should be created
    # assert(False == model_hf.is_filled())

    # start = time.time()
    # PRINT_BUFFOR_START = 3000
    # PRINT_BUFFOR_END = 3500
    # SAMPLES = X.shape[0]

    # # Generate synthetic data following a sine wave pattern
    # # T = np.linspace(0, 5, SAMPLES).reshape(-1, 1)  # Features: 100 points from 0 to 5
    # # Y = np.sin(T).ravel() + np.random.normal(0, 0.1, T.shape[0])  # Sine wave with noise
    # # X = - np.sin(T).ravel() + np.random.normal(0, 0.1, T.shape[0])  # Sine wave with noise
    # print(f"Working with: {SAMPLES} SAMPLES")
    # print(len(X),len(Y))
    # # depth = 20

    # # Train and predict with different depths
    # depths = [100]
    # predictions_cb = {}
    # predictions_hf = {}

    # for d in depths:
    #     split_confidence_cb = 0.0001
    #     split_confidence_hf = 0.0001
    #     grace_period = 1
    #     grace_period_depth = 1

    #     model_cb = FastODT(
    #         depth=d, 
    #         grace_period=grace_period,
    #         split_confidence=split_confidence_cb,
    #         max_feature=1.0,
    #         option="cb",
    #         grace_period_depth=grace_period_depth
    #     )
    #     model_hf = FastODT(
    #         depth=d, 
    #         grace_period=grace_period,
    #         split_confidence=split_confidence_hf,
    #         max_feature=1.0,
    #         option="hf",
    #         grace_period_depth=grace_period_depth
    #     )
    #     model_cb.partial_fit(X, Y, verbose=True)
    #     model_hf.partial_fit(X, Y, verbose=True)
    #     Y_pred_cb = model_cb.predict(X)
    #     Y_pred_hf = model_hf.predict(X)
    #     predictions_cb[d] = Y_pred_cb
    #     predictions_hf[d] = Y_pred_hf
    #     mae_cb = mean_absolute_error(Y, Y_pred_cb)
    #     mae_hf = mean_absolute_error(Y, Y_pred_hf)
    #     print(f"Depth {d} split confidence: {split_confidence_cb} - MAE: {mae_cb:.4f}")
    #     print(f"Depth {d} split confidence: {split_confidence_hf} - MAE: {mae_hf:.4f}")

    # print(f"Training for 1 label data (1D) took: {time.time()-start}s")
    # # Plot predicted values (vs true Y)
    # plt.figure(figsize=(12, 6))
    # plt.plot(Y[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Actual Y", color="black", linewidth=2)

    # for d in depths:
    #     plt.plot(predictions_hf[d][PRINT_BUFFOR_START:PRINT_BUFFOR_END], label=f"Depth {d} Prediction HF")
    #     plt.plot(predictions_cb[d][PRINT_BUFFOR_START:PRINT_BUFFOR_END], label=f"Depth {d} Prediction CB")

    # plt.xlabel(f"Samples")
    # plt.ylabel("Target Y")
    # plt.title(f"FastODT Regression on smoke data. HF {split_confidence_hf} CB: {split_confidence_cb}")
    # plt.ylim(0, 300)
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # Y = np.clip(Y, 0, 200)

    # start = time.time()
    # # # 2nd example - multifeature classification
    # # Generate Features
    # n_samples = Y.shape[0]
    
    # # Train and predict with different depths
    # depths = depths
    # predictions_cb = {}
    # predictions_hf = {}

    # for d in depths:
    #     split_confidence_cb = 0.001
    #     split_confidence_hf = 0.001
    #     grace_period = 10
    #     grace_period_depth=1

    #     model_cb = FastODT(
    #         depth=d, 
    #         grace_period=grace_period,
    #         split_confidence=split_confidence_cb,
    #         max_feature=1.0,
    #         option="cb",
    #         grace_period_depth = grace_period_depth
    #     )
    #     model_hf = FastODT(
    #         depth=d, 
    #         grace_period=grace_period,
    #         split_confidence=split_confidence_hf,
    #         max_feature=1.0,
    #         option="hf",
    #         grace_period_depth = grace_period_depth
    #     )
    #     model_cb.partial_fit(X, Y, verbose=True)
    #     model_hf.partial_fit(X, Y, verbose=True)
    #     Y_pred_cb = np.array([model_cb.predict(x) for x in X])
    #     Y_pred_hf = np.array([model_hf.predict(x) for x in X])
    #     predictions_cb[d] = Y_pred_cb
    #     predictions_hf[d] = Y_pred_hf
    #     mae_cb = mean_absolute_error(Y, Y_pred_cb)
    #     mae_hf = mean_absolute_error(Y, Y_pred_hf)
    #     print(f"Depth {d} split confidence: {split_confidence_cb} - MAE: {mae_cb:.4f}")
    #     print(f"Depth {d} split confidence: {split_confidence_hf} - MAE: {mae_hf:.4f}")

    # print(f"Training for 3 label data (1D) took: {time.time()-start}s")

    # # Plot predicted values (vs true Y)
    # plt.figure(figsize=(12, 6))
    # plt.plot(Y[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Actual Y", color="black", linewidth=2)

    # for d in depths:
    #     plt.plot(predictions_hf[d][PRINT_BUFFOR_START:PRINT_BUFFOR_END], label=f"Depth {d} Prediction HF")
    #     plt.plot(predictions_cb[d][PRINT_BUFFOR_START:PRINT_BUFFOR_END], label=f"Depth {d} Prediction CB")

    # plt.xlabel("T (synthetic time)")
    # plt.ylabel("Target Y")
    # plt.title(f"FastODT Regression on clipped smoke data. HF {split_confidence_hf} CB: {split_confidence_cb}")
    # plt.ylim(0, 300)
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
