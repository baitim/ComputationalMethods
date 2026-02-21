import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def fast_var(data):
    n = len(data)
    M = np.mean(data)
    RS = np.mean(data ** 2)
    return RS - M * M

def two_pass_var(data):
    M = np.mean(data)
    return np.mean((data - M) ** 2)

def online_var(data):
    n = len(data)
    M = data[0] * 0.0
    S = 0.0
    for i, x in enumerate(data, 1):
        delta = x - M
        M = M + delta / i
        S = S + delta * (x - M)
    return S / n

distributions = [
    ("mean=1, std=1", 1.0, 1.0),
    ("mean=10, std=0.1", 10.0, 0.1),
    ("mean=100, std=0.01", 100.0, 0.01)
]

methods = [
    ("Fast", fast_var),
    ("Two-pass", two_pass_var),
    ("Online", online_var)
]

dtypes = [("float32", np.float32), ("float64", np.float64)]

results = {label: {f"{m[0]}_{d[0]}": [] for m in methods for d in dtypes} for label, _, _ in distributions}

for label, mean, std in distributions:
    data64 = np.random.normal(mean, std, 1000)
    ref = np.var(data64, ddof=0)

    for mname, mfunc in methods:
        for dtype_name, dtype in dtypes:
            data = data64.astype(dtype)
            computed = mfunc(data)
            err = abs(computed - ref) / ref
            results[label][f"{mname}_{dtype_name}"] = err

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

for ax, (label, _, _) in zip(axes, distributions):
    errs = [results[label][f"{m}_{d}"] for m in ["Fast", "Two-pass", "Online"] for d in ["float32", "float64"]]
    x = np.arange(len(methods) * len(dtypes))
    width = 0.8
    bars = ax.bar(x, errs, width)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{d}" for m in ["Fast", "Two-pass", "Online"] for d in ["f32", "f64"]])
    ax.set_yscale('log')
    ax.set_title(label)
    ax.set_ylabel("Relative error")
    for bar, err in zip(bars, errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{err:.2e}", ha='center', va='bottom', fontsize=8)

plt.savefig("variance_errors.png")