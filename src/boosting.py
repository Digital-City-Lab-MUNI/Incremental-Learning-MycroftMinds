from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt 
from src.trees.fastodt import FastODT
import numpy as np
import tqdm
import time
import copy
import os

class IncubationBoost:

    def __init__(self, baseLearner, length = 5, baseLearnerOptions = (5, 10, 1.0, "HF")):
        self.chain = []
        self.length = length
        self.incubatingNode = None 
        self.baseLearner = baseLearner
        self.baseLearnerOptions = baseLearnerOptions
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
                residual = y - y_pred
                __y_pred = node.predict(X)
                y_pred += __y_pred
        else:
            for node in self.chain[:-1]:
                residual = y - y_pred
                __y_pred = node.predict(X)
                y_pred += __y_pred

        if self.incubatingNode == None:
            self.incubatingNode = FastODT(
                *self.baseLearnerOptions
            )
            self.incubatingNode.partial_fit(X, residual)
            self.filling_counter = 0
        else:
            self.incubatingNode.partial_fit(X, residual)

            if self.incubatingNode.is_filled():
                self.filling_counter += 1

                if len(self.chain) < self.length:
                    self.chain.append(self.incubatingNode)
                else:
                    self.chain[self.length - 1] = self.incubatingNode
                self.incubatingNode = None
            else:
                print(f"not filled: {self.filling_counter}")

    def predict(self,X):
        __y = 0.0
        for node in self.chain:
            __y += node.predict(X)
        return __y
    

if __name__=="__main__":
    split_confidence_hf = 0.01
    incubation = IncubationBoost(
        FastODT,
        10,
        baseLearnerOptions = (10, 20, split_confidence_hf, "hf")
    )

    # [:,2:] to remove timestamp and device_id
    X = np.load('./smoke_data/X.npy', allow_pickle=True)[:,2:]
    Y = np.load('./smoke_data/y.npy', allow_pickle=True)
    Y = np.clip(Y,0,200)
    print(f"X: {X.shape}, Y: {Y.shape}")

    start = time.time()

    PRINT_BUFFOR_START = 3000
    PRINT_BUFFOR_END = 4500
    SAMPLES = X.shape[0]

    print(f"Working with: {SAMPLES} SAMPLES")
    print(len(X),len(Y))
    # depth = 20

    incubation.partial_fit(X, Y, verbose=True)

    Y_pred = []
    for x,y in zip(X,Y):
        print(f"=>>>> {y} <<<<=")
        Y_pred.append(incubation.predict(x))
    Y_pred = np.array(Y_pred)

    plt.figure(figsize=(12, 6))
    plt.plot(Y[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Actual Y", linewidth=2)
    plt.plot(Y_pred[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Predicted Y", linewidth=2)

    plt.xlabel(f"Samples")
    plt.ylabel("Target Y")
    plt.title(f"IncubationBoost Regression on smoke data. HF {split_confidence_hf}")
    # plt.ylim(0, 300)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
