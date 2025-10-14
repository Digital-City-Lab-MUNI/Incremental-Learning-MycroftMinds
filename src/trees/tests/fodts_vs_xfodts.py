from pathlib import Path
import sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from src.trees.fastodt import FastODT
from src.trees.xfastodt import xFastODT

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def fodt_index_nodes(node, depth, result, current_depth=0, erron = True):

    if depth == current_depth:
        return

    # Ensure list exists for current depth
    if current_depth == len(result):
        result.append([])

    result[current_depth].append(node)
    
    if node is None:
        fodt_index_nodes(None, depth, result, current_depth + 1)
        fodt_index_nodes(None, depth, result, current_depth + 1)
    else:
        fodt_index_nodes(node.left, depth, result, current_depth + 1)
        fodt_index_nodes(node.right, depth, result, current_depth + 1)

def model_check(fast_model, xfast_model, depth, erron = False, similarity_break = True):
    similar = True
    fast_model_indexed = []
    fodt_index_nodes(fast_model.root, depth, fast_model_indexed)

    for d in range(depth):
        if not similar and similarity_break:
            break

        if not erron:
            print(f"^^^^^^^^^ {d} ^^^^^^^^^^")

        max_nodes = 2**d
        for n in range(max_nodes):

            fast_model_row = fast_model_indexed[d] 
            fast_node = fast_model_row[n]
            x_fast_node = xfast_model.depths_containers[d][n]

            if not erron:
                print(f"                             ")
                print(f"-------max_depth: {depth} d: {d} max_n: {max_nodes} n: {n} parent {n // 2}--------- 1")

            if not fast_node and x_fast_node:
                if erron:
                    print(f"-------max_depth: {depth} d: {d} max_n: {2**d} n: {n} parent {n // 2}--------- 2")
                if x_fast_node.threshold and x_fast_node.feature:
                    print("nodes are different")
                    print("x_fast_node exists but no fast_node")
                    similar = False

            elif fast_node and not x_fast_node:    
                if erron:
                    print(f"-------max_depth: {depth} d: {d} max_n: {2**d} n: {n} parent {n // 2}--------- 3")
                if fast_node.threshold and fast_node.feature:
                    print("nodes are different")
                    print("fast_node exists but no x_fast_node")
                    similar = False

            elif not fast_node and not x_fast_node:
                if not erron:
                    print(f"{GREEN}Both nodes are not existing.{RESET}")
                pass
            else:
                if fast_node.value == x_fast_node.value:
                    if not erron:
                        print(f"{GREEN}value is same: {fast_node.value == x_fast_node.value}{RESET}")
                else:
                    if erron:
                        print(f"-------max_depth: {depth} d: {d} max_n: {2**d} n: {n} parent {n // 2}--------- 4")

                    print(f"{RED}value is same: {fast_node.value == x_fast_node.value}{RESET}")
                    print(f"x_fast_node: {x_fast_node.value}")
                    print(f"fast_node: {fast_node.value}")
                    similar = False

                if d < depth - 1: 
                    if fast_node.feature == x_fast_node.feature:
                        if not erron:
                            print(f"{GREEN}feature is same: {fast_node.feature == x_fast_node.feature}{RESET}")
                    else:
                        if erron:
                            print(f"-------max_depth: {depth} d: {d} max_n: {2**d} n: {n} parent {n // 2}--------- 5")
                        if fast_node.feature is None or x_fast_node.feature is None:
                            print(f"{YELLOW}feature is same: {fast_node.feature == x_fast_node.feature}{RESET}")
                        else:
                            print(f"{RED}feature is same: {fast_node.feature == x_fast_node.feature}{RESET}")
                            similar = False
                        print(f"x_fast_node: {x_fast_node.feature}")
                        print(f"fast_node: {fast_node.feature}")
                        
                    if fast_node.threshold == x_fast_node.threshold:
                        if not erron:
                            print(f"{GREEN}threshold is same: {fast_node.threshold == x_fast_node.threshold}{RESET}")
                    else:
                        if erron:
                            print(f"-------max_depth: {depth} d: {d} max_n: {2**d} n: {n} parent {n // 2}--------- 6")
                        if fast_node.threshold is None or x_fast_node.threshold is None:
                            print(f"{YELLOW}threshold is same: {fast_node.threshold == x_fast_node.threshold}{RESET}")
                        else:
                            print(f"{RED}threshold is same: {fast_node.threshold == x_fast_node.threshold}{RESET}")
                            similar = False
                        print(f"x_fast_node: {x_fast_node.threshold}")
                        print(f"fast_node: {fast_node.threshold}")
    return similar

if __name__ == "__main__":
    project_root = ""

    smoke_dir = project_root + 'smoke_data/'
    X_path = smoke_dir + 'X.npy'
    y_path = smoke_dir + 'y.npy'
    print('Smoke data dir:', smoke_dir)

    X = np.load(X_path, allow_pickle=True)
    Y = np.load(y_path, allow_pickle=True)
    print('Raw shapes -> X:', X.shape, 'Y:', Y.shape)

    # drop the first two columns
    if X.ndim == 2 and X.shape[1] >= 3:
        X = X[:, 2:]
    print('After slicing -> X:', X.shape)
    print('Y stats -> min:', float(np.min(Y)), 'max:', float(np.max(Y)))

    # hyperparams for trees
    depth = 7
    grace_period = 10
    split_confidence = 0.01
    tie_threshold = 0.05
    max_feature = 1.0
    grace_period_depth = -1
    bin_type = 'mean'
    max_bins = 1
    inherit_bins = False
    split_criterion = 'vr_rel'

    fast_model = FastODT(
        depth=depth,
        grace_period=grace_period,
        split_confidence=split_confidence,
        tie_threshold=tie_threshold,
        max_feature=max_feature,
        grace_period_depth=grace_period_depth,
        bin_type=bin_type,
        max_bins=max_bins,
        inherit_bins=inherit_bins,
        split_criterion=split_criterion,
    )

    xfast_model = xFastODT(
        depth=depth,
        grace_period=grace_period,
        split_confidence=split_confidence,
        tie_threshold=tie_threshold,
        max_feature=max_feature,
        grace_period_depth=grace_period_depth,
        bin_type=bin_type,
        max_bins=max_bins,
        inherit_bins=inherit_bins,
        split_criterion=split_criterion,
    )
    print('Models initialized.')

    it = 0
    for x,y in zip(X,Y):
        print("========================================================")
        print(f'#{it}. Fitting sample x:', x[:5], 'y:', y)
        fast_model.learn_one(x, y)
        xfast_model.learn_one(x, y)
        it += 1
        if not model_check(fast_model, xfast_model, depth, erron = True):
            print(f"{RED} MODEL DIFFERS {RESET}")
            exit()

    t0 = time.time()
    fast_model.partial_fit(X, Y, verbose=False)
    fast_train_time = time.time() - t0

    t0 = time.time()
    xfast_model.partial_fit(X, Y, verbose=False)
    xfast_train_time = time.time() - t0

    print(f"Training times -> FastODT: {fast_train_time:.3f}s | xFastODT: {xfast_train_time:.3f}s")

    y_pred_fast = np.array([fast_model.predict(x) for x in X])
    y_pred_xfast = np.array(xfast_model.predict(X))

    mae_fast = mean_absolute_error(Y, y_pred_fast)
    mae_xfast = mean_absolute_error(Y, y_pred_xfast)
    print(f"MAE -> FastODT: {mae_fast:.4f} | xFastODT: {mae_xfast:.4f}")

    print("verify model [model diff]")
    model_check(fast_model, xfast_model, depth)

    