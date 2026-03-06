import numpy as np
import matplotlib.pyplot as plt

def correlated_normal(rho, size):
    u = np.random.random((size, 2))
    r = np.sqrt(-2 * np.log(u[:, 0]))
    theta = 2 * np.pi * u[:, 1]
    x1 = r * np.cos(theta)
    u2 = np.random.random((size, 2))
    r2 = np.sqrt(-2 * np.log(u2[:, 0]))
    theta2 = 2 * np.pi * u2[:, 1]
    x2 = r2 * np.cos(theta2)
    y1 = x1
    y2 = rho * x1 + np.sqrt(1 - rho**2) * x2
    return y1, y2

rho = 0.7
size = 50000
Y1, Y2 = correlated_normal(rho, size)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(Y1, Y2, s=1, alpha=0.3)
plt.xlabel('Y1')
plt.ylabel('Y2')
plt.title(f'Корреляция ρ = {rho}')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.hist(Y1, bins=50, density=True, alpha=0.7)
x = np.linspace(-4, 4, 200)
plt.plot(x, 1/np.sqrt(2*np.pi)*np.exp(-x**2/2), 'r-', lw=2)
plt.xlabel('Y1')
plt.title('Маргинальное распределение Y1')

plt.subplot(1, 3, 3)
plt.hist(Y2, bins=50, density=True, alpha=0.7)
plt.plot(x, 1/np.sqrt(2*np.pi)*np.exp(-x**2/2), 'r-', lw=2)
plt.xlabel('Y2')
plt.title('Маргинальное распределение Y2')

plt.tight_layout()
plt.savefig("correlated.png")

rho_emp = np.corrcoef(Y1, Y2)[0, 1]
print(f'Эмпирическая корреляция: {rho_emp:.4f}')