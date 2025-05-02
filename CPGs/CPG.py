import numpy as np

theta = np.array(
    [[0,np.pi],
    [-np.pi,0]]
)

def R(theta):
    """耦合矩阵 R，根据相位差 theta 返回旋转矩阵"""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def hopf_oscillator(t, Z, alpha, mu, omega, k):
    """定义扩散耦合的 Hopf 振荡器系统"""
    N = 2
    dZ = np.zeros(N*2)

    for i in range(N):
        x, y = Z[2 * i], Z[2 * i + 1]
        r_square = x**2 + y**2
        dx = alpha * (mu**2 - r_square) * x - omega * y
        dy = alpha * (mu**2 - r_square) * y + omega * x

        for j in range(N):
            if i != j:
                xj, yj = Z[2 * j], Z[2 * j + 1]
                r_ij = np.array([xj, yj])
                delta = k * R(theta[i][j]) @ r_ij

                dx += k * delta[0]
                dy += k * delta[1]

        dZ[2 * i] = dx
        dZ[2 * i + 1] = dy
    return dZ


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    # 参数
    alpha, mu, omega, k = 100, 3, np.pi, 10  

    # 初始条件：为不同组指定不同初始条件
    Z0 = [0, 0.2, 0, 0.3]

    # 时间范围和步长
    t_span = [0, 100]
    dt = 1./100
    t_eval = np.arange(t_span[0], t_span[1], dt)

    # 初始化状态
    Z = np.array(Z0)

    # 设置绘图
    plt.ion()  # 打开交互模式
    fig, ax = plt.subplots()
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # 保存时间和 Z[2*i] 的值以绘图
    time_data = []
    state_data = [[] for _ in range(2)]

    # 循环计算并输出每个时刻点的值
    for t in t_eval:

        # 保存当前时间和状态
        time_data.append(t)

        for i in range(2):
            state_data[i].append(Z[2 * i+1])

        # 更新图形
        ax.clear()
        for i in range(2):
            ax.plot(time_data, state_data[i], label=f'Oscillator {i + 1}')
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Displacement (x)')
        ax.set_title('Displacement over Time')
        ax.legend()
        ax.grid(True)

        # 设置更细的y轴刻度
        y_min, y_max = -3.6, 3.6  # 根据观察结果调整这些值
        yticks = np.arange(y_min, y_max, 1)  # 设置刻度间隔为0.1
        ax.set_yticks(yticks)

        plt.pause(0.01)

        # 求解下一个时间步长
        sol = solve_ivp(hopf_oscillator, [0, dt], Z, args=(alpha, mu, omega, k))

        # 更新状态
        Z = sol.y[:, -1]


    plt.ioff()
    plt.show()