import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def my_mean(x):
    s = 0.0
    for v in x:
        s += v
    return s / len(x)

def fast_var(data):
    n = len(data)
    M = my_mean(data)
    RS = my_mean(data ** 2)
    return RS - M * M

def two_pass_var(data):
    M = my_mean(data)
    s = 0.0
    for v in data:
        s += (v - M) ** 2
    return s / len(data)

def online_var(data):
    n = len(data)
    M = data[0] * 0.0
    S = 0.0
    for i, x in enumerate(data, 1):
        delta = x - M
        M = M + delta / i
        S = S + delta * (x - M)
    return S / n

def online_var_presentation(data):
    n = len(data)
    if n == 0:
        return 0.0
    M = data[0].astype(float)
    D = 0.0
    for i in range(2, n + 1):
        x = data[i-1]
        M_new = M + (x - M) / (i - 1)
        D_new = D + ((x - M) * (x - M_new) - D) / (i - 1)
        M, D = M_new, D_new
    return D

distributions = [
    ("mean=1, std=1", 1.0, 1.0),
    ("mean=10, std=0.1", 10.0, 0.1),
    ("mean=100, std=0.01", 100.0, 0.01)
]

true_vars = [1.0, 0.01, 0.0001]

methods = [
    ("Fast", fast_var),
    ("Two-pass", two_pass_var),
    ("Online (old)", online_var),
    ("Online (pres)", online_var_presentation)
]

dtypes = [("float32", np.float32), ("float64", np.float64)]

print("Отрицательные значения дисперсии:")
for (label, mean, std), true_var in zip(distributions, true_vars):
    data64 = np.random.normal(mean, std, 1000)
    for mname, mfunc in methods:
        for dtype_name, dtype in dtypes:
            data = data64.astype(dtype)
            computed = mfunc(data)
            if computed < 0:
                print(f"  {label}, {mname}_{dtype_name}: {computed:.6e}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (label, mean, std), true_var in zip(axes, distributions, true_vars):
    data64 = np.random.normal(mean, std, 1000)
    rel_errors = []
    for mname, mfunc in methods:
        for dtype_name, dtype in dtypes:
            data = data64.astype(dtype)
            computed = mfunc(data)
            err = abs(computed - true_var) / true_var
            rel_errors.append(err)
    x = np.arange(len(rel_errors))
    bars = ax.bar(x, rel_errors, width=0.8)
    ax.set_xticks(x)
    labels = [f"{m}\n{d}" for m in ["Fast", "Two-pass", "Online\n(old)", "Online\n(pres)"] for d in ["f32", "f64"]]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yscale('log')
    ax.set_title(label)
    ax.set_ylabel("Relative error (vs theoretical variance)")
    for bar, err in zip(bars, rel_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{err:.2e}",
                ha='center', va='bottom', fontsize=7, rotation=45)

plt.savefig("variance_errors.png", bbox_inches='tight')