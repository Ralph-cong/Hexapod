import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def R(theta):
    """耦合矩阵 R，根据相位差 theta 返回旋转矩阵"""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def coupled_hopf(t, Z, N, alpha, beta, mu, omega, k):
    """定义扩散耦合的 Hopf 振荡器系统"""
    dZ = np.zeros(2 * N)
    for i in range(N):
        x, y = Z[2 * i], Z[2 * i + 1]
        dxdt = alpha * (mu**2 - x**2 - y**2) * x - omega * y
        dydt = beta * (mu**2 - x**2 - y**2) * y + omega * x

        for m in range(N):
            if i != m:
                xm, ym = Z[2 * m], Z[2 * m + 1]

                if (i % 2 == m % 2):
                    theta_im = 0
                else:
                    theta_im = np.pi

                if xm**2 + ym**2 == 0:
                    continue

                r_im = np.array([0, (xm + ym) / np.sqrt(xm**2 + ym**2)])
                coupling = k * R(theta_im) @ r_im

                dxdt += coupling[0]
                dydt += coupling[1]

        dZ[2 * i] = dxdt
        dZ[2 * i + 1] = dydt
    return dZ


# 参数
N = 6  # 振荡器的数量
alpha, beta = 100, 100
mu, omega, k = 3.14, 0.1, 0.1

# 初始条件：为不同组指定不同初始条件
Z0 = []
for i in range(N):
    if i % 2 == 0:
        Z0.extend([0.5, 0.4])  # 一组初始相位
    else:
        Z0.extend([-0.5, -0.4])  # 另一组初始相位

# 时间范围
t_span = [0, 100]
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# 求解系统
sol = solve_ivp(coupled_hopf, t_span, Z0, args=(
    N, alpha, beta, mu, omega, k), t_eval=t_eval)

# 绘制相位图
fig, ax = plt.subplots()
for i in range(N):
    ax.plot(sol.t, sol.y[2 * i], label=f'Oscillator {i + 1}')
ax.set_xlabel('Time (t)')
ax.set_ylabel('Displacement (x)')
ax.set_title('Displacement over Time')
ax.legend()
ax.grid(True)
plt.show()
