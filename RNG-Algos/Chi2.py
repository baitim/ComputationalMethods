import numpy as np
import matplotlib.pyplot as plt
from math import gamma

def chi2_generator(n, size):
    u = np.random.random((size, n, 2))
    r = np.sqrt(-2 * np.log(u[:, :, 0]))
    theta = 2 * np.pi * u[:, :, 1]
    z = r * np.cos(theta)
    return np.sum(z**2, axis=1)

n = 5
size = 100000
sample = chi2_generator(n, size)

x = np.linspace(0, 25, 500)
theor = (x**(n/2 - 1) * np.exp(-x/2)) / (2**(n/2) * gamma(n/2))

plt.figure(figsize=(8, 5))
plt.hist(sample, bins=100, density=True, alpha=0.7, label='Выборка')
plt.plot(x, theor, 'r-', lw=2, label='Теоретическая плотность')
plt.xlabel('x')
plt.ylabel('Плотность')
plt.title(f'Распределение хи-квадрат с {n} степенями свободы')
plt.legend()
plt.grid(True)
plt.savefig("chi2.png")