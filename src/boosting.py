from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt 
from src.trees.fastodt import FastODT
import numpy as np
import tqdm
import time
import os

class IncubationBoost:

    def __init__(self, baseLearner, length = 5, baseLearnerOptions = (5, 10, 1.0, "HF")):
        self.chain = []
        self.length = length
        self.incubatingNode = None 
        self.baseLearner = baseLearner
        self.baseLearnerOptions = baseLearnerOptions

    def partial_fit(self,X,y,verbose = False):
        if verbose == False:
            for x in X:
                self.__partial_fit(x,y)
        else:
            for x in tqdm.tqdm(X):
                self.__partial_fit(x,y)

    def __partial_fit(self,x,y):
        __y = x

        if len(self.chain) < self.length:
            for node in self.chain:
                print(f"Before: {__y}")
                __y = node.predict(__y)
                print(f"After: {__y}")
        else:
            for node in self.chain[:-1]:
                print(f"Before: {__y}")
                __y = node.predict(__y)
                print(f"After: {__y}")

        if self.incubatingNode == None:
            self.incubatingNode = self.baseLearner(
                *self.baseLearnerOptions
            )
            self.incubatingNode.partial_fit([__y],y)

        else:
            self.incubatingNode.partial_fit([__y],y)
            __y = self.incubatingNode.predict(__y)

            if self.incubatingNode.is_filled():
                if len(self.chain) < self.length:
                    self.chain.append(self.incubatingNode)
                else:
                    self.chain[self.length - 1] = self.incubatingNode
                self.incubatingNode = None

    def predict(self,X):
        __y = X
        for node in self.chain:
            __y = node.predict(__y)
        return __y
    

if __name__=="__main__":
    split_confidence_hf = 1.0
    incubation = IncubationBoost(
        FastODT,
        2,
        baseLearnerOptions = (10, 10, split_confidence_hf, "hf")
    )

    # [:,2:] to remove timestamp and device_id
    X = np.load('./smoke_data/X.npy', allow_pickle=True)[:,2:]
    Y = np.load('./smoke_data/y.npy', allow_pickle=True)
    Y = np.clip(Y,0,200)
    print(f"X: {X.shape}, Y: {Y.shape}")

    start = time.time()

    PRINT_BUFFOR_START = 3000
    PRINT_BUFFOR_END = 3500
    SAMPLES = X.shape[0]

    print(f"Working with: {SAMPLES} SAMPLES")
    print(len(X),len(Y))
    # depth = 20

    incubation.partial_fit(X, Y, verbose=True)
    Y_pred = np.array([incubation.predict(x) for x in X])

    plt.figure(figsize=(12, 6))
    plt.plot(Y_pred[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Predicted Y", linewidth=2)
    plt.plot(Y[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Actual Y", linewidth=2)

    plt.xlabel(f"Samples")
    plt.ylabel("Target Y")
    plt.title(f"FastODT Regression on smoke data. HF {split_confidence_hf}")
    # plt.ylim(0, 300)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
