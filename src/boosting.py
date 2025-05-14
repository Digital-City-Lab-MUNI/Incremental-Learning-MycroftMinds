from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt 
from src.trees.fastodt import FastODT
from src.utils.adwin import ADWIN
import numpy as np
import tqdm
import time
import copy
import os

class IncubationBoost:

    def __init__(self, baseLearner, length = 5, incubationMode = "DEFAULT", verbose = True,  baseLearnerOptions = (5, 10, 1.0, "HF")):
        """
        Initializes IncubationBoost.

        Parameters:
            baseLearner (object): The base learning model, for example: FastODT.
            length (int, optional): The max length of boosting chain - after this nodes are being replaced instead of addded.
            incubationMode (str, optional): Mode of operation or configuration for incubation (e.g., "DEFAULT"). In "DEFAULT" algorithm add new node to the end,
                                            or replaces youngest tree. In "BREAK" mode algorithm replaces the first tree in the chain (the oldest one).
                                            "SHIFT_LEFT" shifts youngest nodes left, append new node to the end (it becomes last node), and removes oldest node.
                                            "SHIFT_RIGHT" shifts oldest nodes right, append new node to the start (it takes place of oldest node), and removes last node. 
            baseLearnerOptions (tuple, optional): A tuple of options or hyperparameters for the base learner. 
                                                Default is FastOOT setting (5, 10, 1.0, "HF") - (Depth, Grace_period, HF_bound, Mode).
        """
        self.chain = []
        self.length = length
        self.incubatingNode = None 
        self.baseLearner = baseLearner
        self.baseLearnerOptions = baseLearnerOptions
        self.incubationMode = incubationMode
        self.filling_counter = 0
        self.drift_detector = ADWIN(delta = 0.000002)
        self.under_drift_latch = False
        self.verbose = verbose

        try:
            # Check if running in a Jupyter notebook
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None and 'IPKernelApp' in ipython.config:
                self.tqdm = tqdm.notebook.tqdm
            else:
                self.tqdm = tqdm.tqdm
        except Exception:
            # Fallback in case something goes wrong
            self.tqdm = tqdm.tqdm


    def partial_fit(self,X,y):
        if self.verbose == False:
            for i in range(len(X)):
                self.__partial_fit(X[i],y[i])
        else:
            for i in self.tqdm(range(len(X))):
                self.__partial_fit(X[i],y[i])

    def __partial_fit(self,X,y):
        residual = y
        y_pred = 0

        self.drift_detector.add_sample(X)
        # self.under_drift_latch = True

        # detect if drift happens - if drift already happened then keep it latched (with +) until new tree is deployed
        self.under_drift_latch = bool(self.drift_detector.isDrift() + self.under_drift_latch)

        if len(self.chain) < self.length:
            for node in self.chain:
                __y_pred = node.predict(X)
                y_pred += __y_pred

            residual = y - y_pred
            hatchedNode = self.incubate(X,residual)
            if hatchedNode:
                self.chain.append(hatchedNode)

        elif self.under_drift_latch:
            if self.incubationMode == "DEFAULT":
                for node in self.chain[:-1]:
                    __y_pred = node.predict(X)
                    y_pred += __y_pred
                residual = y - y_pred

                hatchedNode = self.incubate(X,residual)
                if hatchedNode:
                    self.chain[self.length - 1] = hatchedNode
                    self.under_drift_latch = False

            elif self.incubationMode == "BREAK":
                hatchedNode = self.incubate(X,y)
                if hatchedNode:
                    self.chain[0] = hatchedNode

                    for node in self.chain[:-1]:
                        residual = y - y_pred
                        __y_pred = node.predict(X)
                        y_pred += __y_pred
                        self.under_drift_latch = False

            elif self.incubationMode == "SHIFT_LEFT":
                # Generate residual without first tree (which will go out of the scope after adding new tree)
                for node in self.chain[1:]: 
                    __y_pred = node.predict(X)
                    y_pred += __y_pred
                residual = y - y_pred

                hatchedNode = self.incubate(X,residual)
                if hatchedNode:
                    self.chain = self.chain[:-1]
                    self.chain.append(hatchedNode)
                    self.under_drift_latch = False

            elif self.incubationMode == "SHIFT_RIGHT":

                hatchedNode = self.incubate(X,y)
                if hatchedNode:
                    self.chain = self.chain[1:]
                    self.chain.insert(0,hatchedNode)
                    self.under_drift_latch = False


    def incubate(self, X, residual):
        if self.incubatingNode == None:
            self.incubatingNode = self.baseLearner(
                *self.baseLearnerOptions
            )
            self.incubatingNode.partial_fit([X], residual)

        else:
            self.incubatingNode.partial_fit([X], residual)

            if self.incubatingNode.is_filled():

                hatchedNode = self.incubatingNode
                self.incubatingNode = None
                return hatchedNode
            else:
                return None

    def predict(self,X):
        __y = 0.0
        for node in self.chain:
            __y += node.predict(X)
        return __y


if __name__=="__main__":
    split_confidence_hf = 0.0001
    n_trees = 3
    depth = 20
    grace_period = 10
    incubationBreak = IncubationBoost(
        FastODT,
        n_trees,
        incubationMode = "BREAK",
        verbose=True,
        baseLearnerOptions = (depth, grace_period, split_confidence_hf, "hf")
    )
    incubationDefault = IncubationBoost(
        FastODT,
        n_trees,
        incubationMode = "DEFAULT",
        verbose=True,
        baseLearnerOptions = (depth, grace_period, split_confidence_hf, "hf")
    )
    incubationShiftLeft = IncubationBoost(
        FastODT,
        n_trees,
        incubationMode = "SHIFT_LEFT",
        verbose=True,
        baseLearnerOptions = (depth, grace_period, split_confidence_hf, "hf")
    )
    incubationShiftRight = IncubationBoost(
        FastODT,
        n_trees,
        incubationMode = "SHIFT_RIGHT",
        verbose=True,
        baseLearnerOptions = (depth, grace_period, split_confidence_hf, "hf")
    )


    # [:,2:] to remove timestamp and device_id
    print(np.load('./data/first_batch/dev_4_hourly_wfv_test.npz', allow_pickle=True))
    X = np.load('./data/first_batch/dev_4_hourly_wfv_test.npz', allow_pickle=True)["X"][:,2:]
    Y = np.load('./data/first_batch/dev_4_hourly_wfv_test.npz', allow_pickle=True)["y"]
    # X = np.load('./smoke_data/X.npy', allow_pickle=True)[:,2:]
    # Y = np.load('./smoke_data/y.npy', allow_pickle=True)
    # Y = np.clip(Y,0,200)
    print(f"X: {X.shape}, Y: {Y.shape}")

    start = time.time()

    PRINT_BUFFOR_START = 0
    PRINT_BUFFOR_END = X.shape[0]
    SAMPLES = X.shape[0]

    print(f"Working with: {SAMPLES} SAMPLES")
    print(len(X),len(Y))
    # depth = 20

    incubationBreak.partial_fit(X, Y)
    incubationDefault.partial_fit(X, Y)
    incubationShiftLeft.partial_fit(X, Y)
    incubationShiftRight.partial_fit(X, Y)

    Y_pred_def = []
    Y_pred_break = []
    Y_pred_shift_left = []
    Y_pred_shift_right = []
    for x,y in zip(X,Y):
        Y_pred_def.append(incubationDefault.predict(x))
        Y_pred_break.append(incubationBreak.predict(x))
        Y_pred_shift_left.append(incubationShiftLeft.predict(x))
        Y_pred_shift_right.append(incubationShiftRight.predict(x))
    Y_pred_def = np.array(Y_pred_def)
    Y_pred_break = np.array(Y_pred_break)
    Y_pred_shift_left = np.array(Y_pred_shift_left)
    Y_pred_shift_right = np.array(Y_pred_shift_right)

    plt.figure(figsize=(12, 6))
    plt.plot(Y[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Actual Y", linewidth=2)
    plt.plot(Y_pred_def[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Predicted DEFAULT Y", linewidth=5)
    plt.plot(Y_pred_break[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Predicted BREAK Y", linewidth=4)
    plt.plot(Y_pred_shift_left[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Predicted SHIFT LEFT Y", linewidth=3)
    plt.plot(Y_pred_shift_right[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Predicted SHIFT RIGHT Y", linewidth=2)

    plt.xlabel(f"Samples")
    plt.ylabel("Target Y")
    plt.title(f"IncubationBoost Regression on smoke data. HF {split_confidence_hf}")
    # plt.ylim(0, 300)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
