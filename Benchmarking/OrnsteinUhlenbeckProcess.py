import numpy as np
import matplotlib.pyplot as plt

T = 1.0
v0 = 10.0
k = 10.0
theta = 50.0
q = 1.0
dt = 0.0001
n_steps = int(T / dt)
time = np.linspace(0, T, n_steps + 1)
n_paths = 1000

trajectories = np.zeros((n_paths, n_steps + 1))
trajectories[:, 0] = v0

for i in range(1, n_steps + 1):
    dW = np.sqrt(dt) * np.random.randn(n_paths)
    v_prev = trajectories[:, i - 1]
    v_prev = np.maximum(v_prev, 0)
    dv = k * (theta - v_prev) * dt + q * np.sqrt(v_prev) * dW
    v_new = v_prev + dv
    v_new = np.maximum(v_new, 0)
    trajectories[:, i] = v_new

plt.figure(figsize=(10, 6))
plt.plot(time, trajectories.T, color='blue', alpha=0.015, linewidth=0.15)
mean_traj = np.mean(trajectories, axis=0)
plt.plot(time, mean_traj, color='red', linewidth=2, label='Mean')
plt.xlabel('Time')
plt.ylabel('v(t)')
plt.title('1000 trajectories of CIR process (Heston volatility)')
plt.legend()
plt.grid(True)
plt.savefig("OrnsteinUhlenbeckTrajectories.png")
