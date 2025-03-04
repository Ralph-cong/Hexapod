import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



def R(theta):
    """耦合矩阵 R，根据相位差 theta 返回旋转矩阵"""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


# Define the phase difference matrix (as provided in the image)
theta_matrix_degrees = np.array([
    [0, 180, 0, 180, 0, 180, 90, 270, 90, 270, 90, 270],
    [-180, 0, -180, 0, -180, 0, -90, 90, -90, 90, -90, 90],
    [0, 180, 0, 180, 0, 180, 90, 270, 90, 270, 90, 270],
    [-180, 0, -180, 0, -180, 0, -90, 90, -90, 90, -90, 90],
    [0, 180, 0, 180, 0, 180, 90, 270, 90, 270, 90, 270],
    [-180, 0, -180, 0, -180, 0, -90, 90, -90, 90, -90, 90],
    [-90, 90, -90, 90, -90, 90, 0, 180, 0, 180, 0, 180],
    [-270, -90, -270, -90, -270, -90, -180, 0, -180, 0, -180, 0],
    [-90, 90, -90, 90, -90, 90, 0, 180, 0, 180, 0, 180],
    [-270, -90, -270, -90, -270, -90, -180, 0, -180, 0, -180, 0],
    [-90, 90, -90, 90, -90, 90, 0, 180, 0, 180, 0, 180],
    [-270, -90, -270, -90, -270, -90, -180, 0, -180, 0, -180, 0]
])

# Convert degrees to radians for computation
theta_matrix = np.radians(theta_matrix_degrees)


def coupled_hopf(t, Z, N1, N2, alpha, beta, mu, omega1, omega2, k):
    """定义扩散耦合的 Hopf 振荡器系统"""
    dZ = np.zeros(2 * (N1 + N2))
    for i in range(N1 + N2):
        x, y = Z[2 * i], Z[2 * i + 1]

        if i < N1:
            dxdt = alpha * (mu**2 - x**2 - y**2) * x - omega1 * y
            dydt = beta * (mu**2 - x**2 - y**2) * y + omega1 * x

            for m in range(N1):
                if i != m:
                    xm, ym = Z[2 * m], Z[2 * m + 1]

                    # Use the phase difference from the matrix
                    theta_im = theta_matrix[i, m]

                    if xm**2 + ym**2 == 0:
                        continue

                    r_im = np.array([xm / np.sqrt(xm**2 + ym**2),
                                    ym / np.sqrt(xm**2 + ym**2)])
                    coupling = k * R(theta_im) @ r_im

                    dxdt += coupling[0]
                    dydt += coupling[1]

        if i >= N1:  # 如果是 N2 的振荡器，则使用 omega2
            dxdt = alpha * ((mu/2)**2 - x**2 - y**2) * x - omega2 * y
            dydt = beta * ((mu/2)**2 - x**2 - y**2) * y + omega2 * x

            m = i % N1
            xm, ym = Z[2 * m], Z[2 * m + 1]
            # Use the phase difference from the matrix
            theta_im = theta_matrix[i, m]

            # if xm**2 + ym**2 == 0:
            #     continue

            r_im = np.array([xm / np.sqrt(xm**2 + ym**2),
                            ym / np.sqrt(xm**2 + ym**2)])
            coupling = k * R(theta_im) @ r_im

            dxdt += coupling[0]
            dydt += coupling[1]

        dZ[2 * i] = dxdt
        dZ[2 * i + 1] = dydt
    return dZ


# 参数
N1 = 6  # 第一级振荡器的数量
N2 = 6  # 第二级振荡器的数量，考虑最后一级 yaw 角为定值
alpha, beta = 100, 100
mu, omega1, omega2, k = 3.14, 0.1, 0.2, 2  # 调整 omega2 为 omega1 的两倍

# 初始条件：为不同组指定不同初始条件
Z0 = []
for i in range(N1+N2):
    Z0.extend([0.5, 0.4])  # 另一组初始相位

# 时间范围
t_span = [0, 100]
t_eval = np.linspace(t_span[0], t_span[1], 200)

# 求解系统
sol = solve_ivp(coupled_hopf, t_span, Z0, args=(
    N1, N2, alpha, beta, mu, omega1, omega2, k), t_eval=t_eval)

# # 绘制相位图和极限环图
# fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# 绘制相位图和极限环图
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# 波形图
for i in range(N1+N2):
    if i >= N1:
        for j in range(200):
            sol.y[2*i][j] = max(0, sol.y[2*i][j]-0.2)

    axs[0].plot(sol.t, sol.y[2 * i], label=f'Oscillator {i + 1}')
axs[0].set_xlabel('Time (t)')
axs[0].set_ylabel('Displacement (x)')
axs[0].set_title('Displacement over Time')
axs[0].legend()
axs[0].grid(True)

# 极限环图
for i in range(N1+N2):
    axs[1].plot(sol.y[2 * i], sol.y[2 * i + 1], label=f'Oscillator {i + 1}')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].set_title('Limit Cycles')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
