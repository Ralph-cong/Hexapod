import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def R(theta):
    """耦合矩阵 R，根据相位差 theta 返回旋转矩阵"""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


# Define the phase difference matrix (as provided in the image)
theta_matrix_degrees = np.array([
    [0, 180, 0, 0, 0, 180, 0, 0, 0, 0, 0, 0],
    [-180, 0, -180, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 180, 0, 180, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -180, 0, -180, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 180, 0, 180, 0, 0, 0, 0, 0, 0],
    [-180, 0, 0, 0, -180, 0, 0, 0, 0, 0, 0, 0],
    [90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0]
])

# theta_matrix_degrees = np.array([
#     [0, 180, 90, 0, 180, 270, 0, 0, 0, 0, 0, 0],
#     [-180, 0, -90, -180, 0, 90, 0, 0, 0, 0, 0, 0],
#     [-90, 90, 0, -90, 90, 180, 0, 0, 0, 0, 0, 0],
#     [0, 180, 90, 0, 180, 270, 0, 0, 0, 0, 0, 0],
#     [-180, 0, -90, -180, 0, 90, 0, 0, 0, 0, 0, 0],
#     [-270, -90, -180, -270, -90, 0, 0, 0, 0, 0, 0, 0],
#     [-90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, -90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, -90, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, -90, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, -90, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, -90, 0, 0, 0, 0, 0, 0]
# ])

# 四足步态
# theta_matrix_degrees = np.array([
#     [0, 270, 180, 180, 90, 0, 0, 0, 0, 0, 0, 0],
#     [-270, 0, -90, -90, -180, -270, 0, 0, 0, 0, 0, 0],
#     [-180, 90, 0, 0, -90, -180, 0, 0, 0, 0, 0, 0],
#     [-180, 90, 0, 0, -90, -180, 0, 0, 0, 0, 0, 0],
#     [-90, 180, 90, 90, 0, -90, 0, 0, 0, 0, 0, 0],
#     [0, 270, 180, 180, 90, 0, 0, 0, 0, 0, 0, 0],
#     [-90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, -90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, -90, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, -90, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, -90, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, -90, 0, 0, 0, 0, 0, 0]
# ])

# Convert degrees to radians for computation
theta_matrix = np.radians(theta_matrix_degrees)


# def coupled_hopf(t, Z, N1, N2, alpha, beta, mu, omega1, omega2, k, eps):
#     """定义扩散耦合的 Hopf 振荡器系统"""
#     dZ = np.zeros(2 * (N1 + N2))
#     for i in range(N1 + N2):
#         x, y = Z[2 * i], Z[2 * i + 1]
#         if y >= 0:
#             omega1 = omega1/(1-eps)
#         elif y < 0:
#             omega1 = omega1/eps

#         if i < N1:
#             dxdt = alpha * (mu**2 - x**2 - y**2) * x - omega1 * y
#             dydt = beta * (mu**2 - x**2 - y**2) * y + omega1 * x

#             for m in range(N1):
#                 if i != m:
#                     xm, ym = Z[2 * m], Z[2 * m + 1]

#                     # Use the phase difference from the matrix
#                     theta_im = theta_matrix[i, m]

#                     if xm**2 + ym**2 == 0:
#                         continue

#                     r_im = np.array([xm / np.sqrt(xm**2 + ym**2),
#                                     ym / np.sqrt(xm**2 + ym**2)])
#                     coupling = k * R(theta_im) @ r_im

#                     dxdt += coupling[0]
#                     dydt += coupling[1]

#         if i >= N1:  # 如果是 N2 的振荡器，则使用 omega2
#             dxdt = alpha * ((mu/2)**2 - x**2 - y**2) * x - omega2 * y
#             dydt = beta * ((mu/2)**2 - x**2 - y**2) * y + omega2 * x

#             m = i % N1
#             xm, ym = Z[2 * m], Z[2 * m + 1]
#             # Use the phase difference from the matrix
#             theta_im = theta_matrix[i, m]

#             r_im = np.array([xm / np.sqrt(xm**2 + ym**2),
#                             ym / np.sqrt(xm**2 + ym**2)])
#             coupling = k * R(theta_im) @ r_im

#             dxdt += coupling[0]
#             dydt += coupling[1]

#         dZ[2 * i] = dxdt
#         dZ[2 * i + 1] = dydt
#     return dZ

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
            dxdt = alpha * ((mu)**2 - x**2 - y**2) * x - omega2 * y
            dydt = beta * ((mu)**2 - x**2 - y**2) * y + omega2 * x

            m = i % N1
            xm, ym = Z[2 * m], Z[2 * m + 1]
            # Use the phase difference from the matrix
            theta_im = theta_matrix[i, m]

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
alpha, beta = 150, 150
mu, omega1, omega2, k, eps = 3.14, 30, 30, 120, 0.5  # 调整 omega2 为 omega1 的两倍

# 初始条件：为不同组指定不同初始条件
Z0 = []
for i in range(3):
    Z0.extend([0, mu, 0, -mu])  # 另一组初始相位
for i in range(3):
    Z0.extend([-mu, 0,  mu, 0])  # 另一组初始相位

# 时间范围和步长
t_span = [0, 100]
dt = 1./1000
t_eval = np.arange(t_span[0], t_span[1], dt)

# 初始化状态
Z = np.array(Z0)

# 设置绘图
plt.ion()  # 打开交互模式
fig, ax = plt.subplots()
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# 保存时间和 Z[2*i] 的值以绘图
time_data = []
state_data = [[] for _ in range(N1 + N2)]

# 循环计算并输出每个时刻点的值
for t in t_eval:
    # 输出当前状态
    # print(f"Time: {t:.1f}, States: {Z}")

    # 保存当前时间和状态
    time_data.append(t)

    for i in range(N1 + N2):
        if i < N1:
            state_data[i].append(Z[2 * i])
        if i >= N1:
            # state_data[i].append(0)
            state_data[i].append(max(0, Z[2 * i]))
            # state_data[i].append(max(0, Z[2 * i] - 0.2))

    # 更新图形
    ax.clear()
    # for i in range(N1 + N2):
    #     ax.plot(time_data, state_data[i], label=f'Oscillator {i + 1}')
    ax.plot(time_data, state_data[0], label=f'Oscillator {1}')
    ax.plot(time_data, state_data[6], label=f'Oscillator {7}')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Displacement (x)')
    ax.set_title('Displacement over Time')
    ax.legend()
    ax.grid(True)

    # 设置更细的y轴刻度
    y_min, y_max = -3.6, 3.6  # 根据观察结果调整这些值
    yticks = np.arange(y_min, y_max, 0.1)  # 设置刻度间隔为0.1
    ax.set_yticks(yticks)

    plt.pause(0.01)

    # 求解下一个时间步长
    sol = solve_ivp(coupled_hopf, [
                    0, dt], Z, args=(N1, N2, alpha, beta, mu, omega1, omega2, k))

    # 更新状态
    Z = sol.y[:, -1]


plt.ioff()
plt.show()
