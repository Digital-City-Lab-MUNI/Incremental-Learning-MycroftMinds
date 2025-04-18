from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt 
from src.trees.fastodt import FastODT
import numpy as np
import tqdm
import time
import copy
import os

class IncubationBoost:

    def __init__(self, baseLearner, length = 5, incubationMode = "DEFAULT",  baseLearnerOptions = (5, 10, 1.0, "HF")):
        self.chain = []
        self.length = length
        self.incubatingNode = None 
        self.baseLearner = baseLearner
        self.baseLearnerOptions = baseLearnerOptions
        self.incubationMode = incubationMode
        self.filling_counter = 0

    def partial_fit(self,X,y,verbose = False):
        if verbose == False:
            for i in range(len(X)):
                self.__partial_fit(X[i],y[i])
        else:
            for i in tqdm.tqdm(range(len(X))):
                self.__partial_fit(X[i],y[i])

    def __partial_fit(self,X,y):
        residual = y
        y_pred = 0
        if len(self.chain) < self.length:
            for node in self.chain:
                __y_pred = node.predict(X)
                y_pred += __y_pred

            residual = y - y_pred
            hatchedNode = self.incubate(X,residual)
            if hatchedNode:
                self.chain.append(hatchedNode)

        elif self.incubationMode == "DEFAULT":
            for node in self.chain[:-1]:
                __y_pred = node.predict(X)
                y_pred += __y_pred
            residual = y - y_pred

            hatchedNode = self.incubate(X,residual)
            if hatchedNode:
                self.chain[self.length - 1] = hatchedNode

        elif self.incubationMode == "BREAK":
            hatchedNode = self.incubate(X,y)
            if hatchedNode:
                self.chain[0] = hatchedNode

                for node in self.chain[:-1]:
                    residual = y - y_pred
                    __y_pred = node.predict(X)
                    y_pred += __y_pred


    def incubate(self, X, residual):
        if self.incubatingNode == None:
            self.incubatingNode = self.baseLearner(
                *self.baseLearnerOptions
            )
            self.incubatingNode.partial_fit(X, residual)

        else:
            self.incubatingNode.partial_fit(X, residual)

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
    depth = 10
    grace_period = 10
    incubationBreak = IncubationBoost(
        FastODT,
        n_trees,
        incubationMode = "BREAK",
        baseLearnerOptions = (depth, grace_period, split_confidence_hf, "hf")
    )
    incubationDefault = IncubationBoost(
        FastODT,
        n_trees,
        incubationMode = "DEFAULT",
        baseLearnerOptions = (depth, grace_period, split_confidence_hf, "hf")
    )

    # [:,2:] to remove timestamp and device_id
    X = np.load('./smoke_data/X.npy', allow_pickle=True)[:,2:]
    Y = np.load('./smoke_data/y.npy', allow_pickle=True)
    Y = np.clip(Y,0,200)
    print(f"X: {X.shape}, Y: {Y.shape}")

    start = time.time()

    PRINT_BUFFOR_START = -3500
    PRINT_BUFFOR_END = X.shape[0]
    SAMPLES = X.shape[0]

    print(f"Working with: {SAMPLES} SAMPLES")
    print(len(X),len(Y))
    # depth = 20

    incubationBreak.partial_fit(X, Y, verbose=True)
    incubationDefault.partial_fit(X, Y, verbose=True)

    Y_pred_def = []
    Y_pred_break = []
    for x,y in zip(X,Y):
        Y_pred_def.append(incubationBreak.predict(x))
        Y_pred_break.append(incubationDefault.predict(x))
    Y_pred_def = np.array(Y_pred_def)
    Y_pred_break = np.array(Y_pred_break)

    plt.figure(figsize=(12, 6))
    plt.plot(Y[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Actual Y", linewidth=2)
    plt.plot(Y_pred_break[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Predicted BREAK Y", linewidth=2)
    plt.plot(Y_pred_def[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Predicted DEFAULT Y", linewidth=2)

    plt.xlabel(f"Samples")
    plt.ylabel("Target Y")
    plt.title(f"IncubationBoost Regression on smoke data. HF {split_confidence_hf}")
    # plt.ylim(0, 300)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
