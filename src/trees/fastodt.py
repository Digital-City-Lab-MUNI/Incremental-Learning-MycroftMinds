from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt 
from src.trees.tree import BaseTree, Node
from src.trees.node_hf import NodeHoeffdingBound
from src.trees.node_cb import NodeConfidenceBound
import numpy as np
import tqdm
import time
import os

class FastODT(BaseTree):
    def __init__(self, depth = 5, grace_period=10, split_confidence=1.0, option = "HF"):
        self.grace_period = grace_period  # Number of samples before checking for a split
        self.split_confidence = split_confidence  # Confidence level for splitting
        self.depth = depth
        self.splits = [None] * depth

        self.option = option
        nodes_option_list = ["hf","cb"]
        if self.option not in nodes_option_list:
            self.option = "hf"
            print(f"Wrong option in option: {option}, available options: {nodes_option_list}. Falling back to default: 'hf'.")

        if self.option == "hf":
            self.root = NodeHoeffdingBound(self.split_confidence)
        elif self.option == "cb":
            # self.split_confidence = np.log(1/self.split_confidence)
            self.root = NodeConfidenceBound()

    def partial_fit(self, X, y, verbose = False):
        if verbose:
            for i in tqdm.tqdm(range(len(X))):
                self._update_tree(self.root, X[i], y[i], depth = 0)
        else:
            for i in range(len(X)):
                self._update_tree(self.root, X[i], y[i], depth = 0)

    def _update_tree(self, node , x, y, depth = 0):

        if depth >= self.depth:
            return

        # print(node.left,node.right)
        if not isinstance(x, np.ndarray):
            x = np.array([x])

        if node.left is None and node.right is None:
            node.update_stats(x, y)
            if node.samples_seen >= self.grace_period:
                self._attempt_split(node, x, y, depth)
            # print(node.left,node.right)
        elif node.feature is not None:
            if x[node.feature] <= node.threshold:
                self._update_tree(node.left,x,y,depth + 1)

            elif x[node.feature] > node.threshold:
                self._update_tree(node.right,x,y,depth + 1)

    def _attempt_split(self, node, x, y, depth):
        if self.option == "hf":
            self._attempt_split_HF(node, x, y, depth)
        else:
            self._attempt_split_confidence(node, x, y, depth)

    def _attempt_split_HF(self, node, x, y, depth):
        best_split, _ = node.find_best_split()
        if self.splits[depth] is not None:
            node.split(self.splits[depth],self.split_confidence)
            node.left.update_from_parent_node(node)
            node.right.update_from_parent_node(node)
        
        elif best_split[0] is not None and best_split[1] is not None:
            self.splits[depth] = best_split 
            node.split(best_split,self.split_confidence)
            node.left.update_from_parent_node(node)
            node.right.update_from_parent_node(node)


    def _attempt_split_confidence(self, node, x, y, depth):
        best_split, best_reduction = node.find_best_split()

        if self.splits[depth] is not None:
            node.split(self.splits[depth])
            node.left.update_from_parent_node(node)
            node.right.update_from_parent_node(node)
        
        elif best_split and best_reduction > self.split_confidence:
            self.splits[depth] = best_split 
            node.split(best_split)
            node.left.update_from_parent_node(node)
            node.right.update_from_parent_node(node)


if __name__ == "__main__":

    X = np.load('./smoke_data/X.npy', allow_pickle=True)
    Y = np.load('./smoke_data/y.npy', allow_pickle=True)
    print(X.shape, Y.shape)
    X = X[:,2:]
    print(type(X), type(Y), X[0])
    start = time.time()
    PRINT_BUFFOR_START = 3000
    PRINT_BUFFOR_END = 3500
    SAMPLES = X.shape[0]

    # Generate synthetic data following a sine wave pattern
    # T = np.linspace(0, 5, SAMPLES).reshape(-1, 1)  # Features: 100 points from 0 to 5
    # Y = np.sin(T).ravel() + np.random.normal(0, 0.1, T.shape[0])  # Sine wave with noise
    # X = - np.sin(T).ravel() + np.random.normal(0, 0.1, T.shape[0])  # Sine wave with noise
    print(f"Working with: {SAMPLES} SAMPLES")
    print(len(X),len(Y))
    # depth = 20

    # Train and predict with different depths
    depths = [20, 150, 500]
    predictions_cb = {}
    predictions_hf = {}

    for d in depths:
        print(d)
        split_confidence_cb = 0.0001
        split_confidence_hf = 0.0001

        model_cb = FastODT(
            depth=d, 
            grace_period=10,
            split_confidence=split_confidence_cb,
            option="cb"
        )
        model_hf = FastODT(
            depth=d, 
            grace_period=10,
            split_confidence=split_confidence_hf,
            option="hf"
        )
        model_cb.partial_fit(X, Y, verbose=True)
        model_hf.partial_fit(X, Y, verbose=True)
        Y_pred_cb = np.array([model_cb.predict(x) for x in X])
        Y_pred_hf = np.array([model_hf.predict(x) for x in X])
        predictions_cb[d] = Y_pred_cb
        predictions_hf[d] = Y_pred_hf
        mae_cb = mean_absolute_error(Y, Y_pred_cb)
        mae_hf = mean_absolute_error(Y, Y_pred_hf)
        print(f"Depth {d} split confidence: {split_confidence_cb} - MAE: {mae_cb:.4f}")
        print(f"Depth {d} split confidence: {split_confidence_hf} - MAE: {mae_hf:.4f}")

    print(f"Training for 1 label data (1D) took: {time.time()-start}s")
    # Plot predicted values (vs true Y)
    plt.figure(figsize=(12, 6))
    print(Y[PRINT_BUFFOR_START:PRINT_BUFFOR_END].shape)
    plt.plot(Y[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Actual Y", color="black", linewidth=2)

    for d in depths:
        print(predictions_hf[d][PRINT_BUFFOR_START:PRINT_BUFFOR_END].shape)
        print(predictions_cb[d][PRINT_BUFFOR_START:PRINT_BUFFOR_END].shape)
        plt.plot(predictions_hf[d][PRINT_BUFFOR_START:PRINT_BUFFOR_END], label=f"Depth {d} Prediction HF")
        plt.plot(predictions_cb[d][PRINT_BUFFOR_START:PRINT_BUFFOR_END], label=f"Depth {d} Prediction CB")

    plt.xlabel(f"Samples")
    plt.ylabel("Target Y")
    plt.title(f"FastODT Regression on smoke data. HF {split_confidence_hf} CB: {split_confidence_cb}")
    plt.ylim(0, 300)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    Y = np.clip(Y, 0, 200)

    start = time.time()
    # # 2nd example - multifeature classification
    # Generate Features
    n_samples = Y.shape[0]
    
    # Train and predict with different depths
    depths = depths
    predictions_cb = {}
    predictions_hf = {}

    for d in depths:
        print(d)
        split_confidence_cb = 0.001
        split_confidence_hf = 0.001
        model_cb = FastODT(
            depth=d, 
            grace_period=10,
            split_confidence=split_confidence_cb,
            option="cb"
        )
        model_hf = FastODT(
            depth=d, 
            grace_period=10,
            split_confidence=split_confidence_hf,
            option="hf"
        )
        model_cb.partial_fit(X, Y, verbose=True)
        model_hf.partial_fit(X, Y, verbose=True)
        Y_pred_cb = np.array([model_cb.predict(x) for x in X])
        Y_pred_hf = np.array([model_hf.predict(x) for x in X])
        predictions_cb[d] = Y_pred_cb
        predictions_hf[d] = Y_pred_hf
        mae_cb = mean_absolute_error(Y, Y_pred_cb)
        mae_hf = mean_absolute_error(Y, Y_pred_hf)
        print(f"Depth {d} split confidence: {split_confidence_cb} - MAE: {mae_cb:.4f}")
        print(f"Depth {d} split confidence: {split_confidence_hf} - MAE: {mae_hf:.4f}")

    print(f"Training for 3 label data (1D) took: {time.time()-start}s")

    # Plot predicted values (vs true Y)
    plt.figure(figsize=(12, 6))
    plt.plot(Y[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Actual Y", color="black", linewidth=2)

    for d in depths:
        plt.plot(predictions_hf[d][PRINT_BUFFOR_START:PRINT_BUFFOR_END], label=f"Depth {d} Prediction HF")
        plt.plot(predictions_cb[d][PRINT_BUFFOR_START:PRINT_BUFFOR_END], label=f"Depth {d} Prediction CB")

    plt.xlabel("T (synthetic time)")
    plt.ylabel("Target Y")
    plt.title(f"FastODT Regression on clipped smoke data. HF {split_confidence_hf} CB: {split_confidence_cb}")
    plt.ylim(0, 300)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
