from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt 
from src.trees.fastodt import FastODT
#from src.utils.adwin import ADWIN
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import tqdm
import time
import copy
import os

class IncubationBoost:

    def __init__(self, 
                 length: int = 5, 
                 learning_rate: float = 1.0, 
                 base_learner=FastODT, 
                 base_learner_params: dict = {'depth': 15, 'grace_period': 10, 'split_confidence': 0.01, 'max_feature': 1.0, 'grace_period_depth': -1, 'eagerness': 0.0, 'bin_type': 'adaptive', 'max_bins': 32, 'split_criterion': 'vr_rel'},
                 verbose: bool = True,
                 ):
        """
        Initializes IncubationBoost.

        Parameters:
            base_learner (object): The base learning model. Default is FastODT.
            length (int, optional): The max length of boosting chain - after this nodes are being replaced instead of addded.
            learning_rate (float, optional): The learning rate for the boosting process.
            base_learner_params (dict, optional): Options for the base learner, such as depth, grace_period, split_confidence, max_feature, grace_period_depth, eagerness, bin_type, max_bins, and split_criterion.
            
        """
        self.chain = []
        self.length: int = length
        self.learning_rate: float = learning_rate
        self.base_learner = base_learner
        self.base_learner_params: dict = base_learner_params
        self.incubating_node = None 
        #self.drift_detector = ADWIN(delta = 0.000002)
        # self.under_drift_latch = False
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


    def partial_fit(self, X, y):
        if self.verbose == False:
            for i in range(len(X)):
                self._partial_fit(X[i],y[i])
                # input()
        else:
            for i in self.tqdm(range(len(X))):
                self._partial_fit(X[i],y[i])
                # input()

    def _partial_fit(self, X, y):
        residual = y
        y_pred_strong = 0.0

        pruning_index = None
        prev_pruning_error_rate = 9999.0

        #self.drift_detector.add_sample(X)
        # self.under_drift_latch = True
        # detect if drift happens - if drift already happened then keep it latched (with +) until new tree is deployed
        #self.under_drift_latch = bool(self.drift_detector.isDrift() + self.under_drift_latch)
        
        # if the chain is empty, create the first node
        if len(self.chain) == 0:
            newNode = self.base_learner(
                **self.base_learner_params
            )
            self.chain.append(newNode)

        # Index-based pruning (eGBDT)
        for n, node in enumerate(self.chain):
            y_pred = node.predict(X) * (1.0 if n == 0 else self.learning_rate)  # first node is not multiplied by learning rate
            y_pred_strong += y_pred
            residual = y - y_pred_strong

            if node.is_filled():
                error_rate = node.measure_error(y, y_pred_strong)
                if prev_pruning_error_rate > error_rate:
                    pruning_index = n 
                    prev_pruning_error_rate = error_rate 
                else:
                    break
            else:
                node.partial_fit([X], residual)

        if pruning_index is not None:
            self.chain = self.chain[:pruning_index+1]

        hatched_node = self.incubate(X,residual)
        if hatched_node and len(self.chain) < self.length:
            self.chain.append(hatched_node)

    def incubate(self, X, residual):
        if self.incubating_node == None:
            self.incubating_node = self.base_learner(
                **self.base_learner_params
            )
            self.incubating_node.partial_fit([X], residual)

        else:
            self.incubating_node.partial_fit([X], residual)

            if self.incubating_node.is_filled():
                hatched_node = self.incubating_node
                self.incubating_node = None
                return hatched_node
            else:
                return None

    def predict(self,X):
        y = 0.0
        for n, node in enumerate(self.chain):
            y += node.predict(X) * (1.0 if n == 0 else self.learning_rate)
        return y


if __name__=="__main__":
    split_confidence_hf = 0.01
    n_trees = 5
    depth = 50
    grace_period = 1
    max_feature = 1.0
    eagerness = 1000
    learning_rate = 1.0
    grace_period_depth = -1
    incubationDefault = IncubationBoost(
        FastODT,
        n_trees,
        learning_rate = learning_rate,
        incubationMode = "DEFAULT",
        verbose=True,
        baseLearnerOptions = (depth, grace_period, split_confidence_hf, "hf", max_feature, eagerness, grace_period_depth)
    )
    incubationShiftLeft = IncubationBoost(
        FastODT,
        n_trees,
        learning_rate = learning_rate,
        incubationMode = "SHIFT_LEFT",
        verbose=True,
        baseLearnerOptions = (depth, grace_period, split_confidence_hf, "hf", max_feature, eagerness, grace_period_depth)
    )
    incubationShiftRight = IncubationBoost(
        FastODT,
        n_trees,
        learning_rate = learning_rate,
        incubationMode = "SHIFT_RIGHT",
        verbose=True,
        baseLearnerOptions = (depth, grace_period, split_confidence_hf, "hf", max_feature, eagerness, grace_period_depth)
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

    incubationDefault.partial_fit(X, Y)
    incubationShiftLeft.partial_fit(X, Y)
    incubationShiftRight.partial_fit(X, Y)

    Y_pred_def = []
    Y_pred_break = []
    Y_pred_shift_left = []
    Y_pred_shift_right = []
    
    for x,y in zip(X,Y):
        Y_pred_def.append(incubationDefault.predict(x))
        Y_pred_shift_left.append(incubationShiftLeft.predict(x))
        Y_pred_shift_right.append(incubationShiftRight.predict(x))
    Y_pred_def = np.array(Y_pred_def)
    Y_pred_break = np.array(Y_pred_break)
    Y_pred_shift_left = np.array(Y_pred_shift_left)
    Y_pred_shift_right = np.array(Y_pred_shift_right)

    plt.figure(figsize=(12, 6))
    plt.plot(Y[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Actual Y", linewidth=2)
    plt.plot(Y_pred_def[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Predicted DEFAULT Y", linewidth=5)
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
