import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from LearnNSE import LearnNSE  # your LearnNSE class with pruning_type

# --- 1) Concept‐drift generators ---

def _sample_gaussian(mu0, mu1, cov, n_samples):
    half = n_samples // 2
    X0 = np.random.multivariate_normal(mu0, cov, half)
    X1 = np.random.multivariate_normal(mu1, cov, half)
    X = np.vstack([X0, X1])
    y = np.array([0]*half + [1]*half)
    perm = np.random.permutation(n_samples)
    return X[perm], y[perm]

def generate_sudden_drift(batch_idx, n_batches, n_samples=200):
    midpoint = n_batches // 2
    if batch_idx <= midpoint:
        mu0, mu1 = np.array([0,0]), np.array([2,2])
    else:
        mu0, mu1 = np.array([0,2]), np.array([2,0])
    return _sample_gaussian(mu0, mu1, np.eye(2)*0.5, n_samples)

def generate_incremental_drift(batch_idx, n_batches, n_samples=200):
    start_mu1, end_mu1 = np.array([2,2]), np.array([8,-4])
    alpha = batch_idx / (n_batches-1)
    mu1 = (1-alpha)*start_mu1 + alpha*end_mu1
    mu0 = np.array([0,0])
    return _sample_gaussian(mu0, mu1, np.eye(2)*0.5, n_samples)

def generate_gradual_drift(batch_idx, n_batches, n_samples=200):
    mu0_a, mu1_a = np.array([0,0]), np.array([2,2])
    mu0_b, mu1_b = np.array([0,2]), np.array([2,0])
    cov = np.eye(2)*0.5
    frac_b = batch_idx / (n_batches-1)
    n_b = int(n_samples * frac_b)
    n_a = n_samples - n_b

    # Concept A sample counts
    half_a0 = n_a // 2
    half_a1 = n_a - half_a0
    X0_a = np.random.multivariate_normal(mu0_a, cov, half_a0)
    X1_a = np.random.multivariate_normal(mu1_a, cov, half_a1)
    ya = np.array([0]*half_a0 + [1]*half_a1)

    # Concept B sample counts
    half_b0 = n_b // 2
    half_b1 = n_b - half_b0
    X0_b = np.random.multivariate_normal(mu0_b, cov, half_b0)
    X1_b = np.random.multivariate_normal(mu1_b, cov, half_b1)
    yb = np.array([0]*half_b0 + [1]*half_b1)

    # Combine & shuffle
    X = np.vstack([X0_a, X1_a, X0_b, X1_b])
    y = np.concatenate([ya, yb])
    perm = np.random.permutation(n_samples)
    return X[perm], y[perm]

# --- 2) Drift schedule ---
n_batches = 140
schedule = [
    (generate_incremental_drift, 0, 49),
    (generate_sudden_drift,      50, 89),
    (generate_gradual_drift,     90, 139),
]

def drift_fn(t):
    for fn, start, end in schedule:
        if start <= t <= end:
            return fn(t, n_batches)
    # fallback
    return generate_incremental_drift(t, n_batches)

# --- 3) Instantiate LearnNSE with pruning ---
ensemble_size = 30
nse_age   = LearnNSE(base_classifier=None, ensemble_size=ensemble_size, pruning_type='age')
nse_error = LearnNSE(base_classifier=None, ensemble_size=ensemble_size, pruning_type='error')

# --- 4) Static baselines on batch 0 ---
rf_baseline = RandomForestClassifier(n_estimators=ensemble_size, random_state=0)
gb_baseline = GradientBoostingClassifier(n_estimators=ensemble_size, random_state=0)
X0, y0 = drift_fn(0)
rf_baseline.fit(X0, y0)
gb_baseline.fit(X0, y0)

# --- 5) Online evaluation ---
acc_age, acc_error = [], []
acc_rf,  acc_gb    = [], []
size_age, size_error = [], []

for t in tqdm.tqdm(range(n_batches)):
    X_train, y_train = drift_fn(t)
    nse_age.fit(X_train, y_train)
    nse_error.fit(X_train, y_train)
    rf_baseline.fit(X_train, y_train)
    gb_baseline.fit(X_train, y_train)

    X_test, y_test = drift_fn(t+1)
    acc_age.append(nse_age.score(X_test, y_test))
    acc_error.append(nse_error.score(X_test, y_test))
    acc_rf.append(rf_baseline.score(X_test, y_test))
    acc_gb.append(gb_baseline.score(X_test, y_test))
    size_age.append(len(nse_age.models))
    size_error.append(len(nse_error.models))

# --- 6) Plotting ---
plt.figure(figsize=(10,4))
plt.plot(acc_age,   label='LearnNSE (age)',   marker='o', markevery=20)
plt.plot(acc_error, label='LearnNSE (error)', marker='x', markevery=20)
plt.plot(acc_rf,    '--', label='RF (online)',  color='gray')
plt.plot(acc_gb,    '--', label='GBRT (online)',color='orange')
plt.xlabel('Batch index')
plt.ylabel('Accuracy next batch')
plt.title('Mixed-Drift: Incremental → Sudden → Gradual')
plt.legend(ncol=2)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(size_age,   label='Size (age-prune)',   marker='o')
plt.plot(size_error, label='Size (error-prune)', marker='x')
plt.xlabel('Batch index')
plt.ylabel('Ensemble size')
plt.title('LearnNSE Growth under Mixed Drift')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 7) Summary ---
print("\nMean accuracy:")
print(f"LearnNSE (age)   : {np.mean(acc_age):.3f}")
print(f"LearnNSE (error) : {np.mean(acc_error):.3f}")
print(f"RF baseline      : {np.mean(acc_rf):.3f}")
print(f"GBRT baseline    : {np.mean(acc_gb):.3f}")
