import matplotlib.pyplot as plt
import numpy as np
from CPGs.CPG import hopf_oscillator
from scipy.integrate import solve_ivp
import math

# CPG 参数
alpha, mu, omega, k = 50, 1, np.pi / 3, 5
Z = np.array([0, mu, 0, -mu])
dt = 0.01
t_max = 10
steps = int(t_max / dt)

A = np.concatenate([
    np.array([1.0, 1.0]),        # A1: hip scaling
    np.ones(6) * 0.8,            # A2: knee scaling
    np.ones(6) * 1.0             # A3: ankle scaling
])

# 存储角度数据
hip_angles = {0: [], 1: []}
knee_angles = {0: [], 1: []}
ankle_angles = {0: [], 1: []}
time_data = []

for step in range(steps):
    time = step * dt
    time_data.append(time)

    # 获取当前 dy（Z 的导数）
    dZ = hopf_oscillator(0, Z, alpha, mu, omega, k)

    for group_idx in [0, 1]:
        z_y = Z[2 * group_idx + 1] / 1.23
        z_x = Z[2 * group_idx]
        hip = A[group_idx] * z_y

        dy = dZ[2 * group_idx + 1]  # 当前振荡器的 dy
        if dy >= 0:
            knee = A[2 + group_idx * 3] * (1 - z_y ** 2)
        else:
            knee = 0

        ankle = -A[8 + group_idx * 3] * knee

        hip_angles[group_idx].append(hip)
        knee_angles[group_idx].append(knee)
        ankle_angles[group_idx].append(ankle)

    # 更新 Z
    sol = solve_ivp(hopf_oscillator, [0, dt], Z, args=(alpha, mu, omega, k), method='RK45')
    Z = sol.y[:, -1]

# 绘图
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
joint_names = ['Hip', 'Knee', 'Ankle']
angle_data = [hip_angles, knee_angles, ankle_angles]

for i, (ax, joint, data) in enumerate(zip(axs, joint_names, angle_data)):
    ax.plot(time_data, data[0], label='Group 0 (legs 0,2,4)')
    # ax.plot(time_data, data[1], label='Group 1 (legs 1,3,5)')
    ax.set_ylabel(f'{joint} Angle')
    ax.grid(True)
    ax.legend()

axs[-1].set_xlabel('Time (s)')
plt.suptitle('Joint Angle Variations of Two Leg Groups')
plt.tight_layout()
plt.show()
