import numpy as np


def hopf_oscillator(t, Z, alpha, mu, omega, k):
    """定义扩散耦合的 Hopf 振荡器系统"""
    N = 2
    dZ = np.zeros(N*2)

    for i in range(N):
        x, y = Z[2 * i], Z[2 * i + 1]
        r_square = x**2 + y**2
        dx = alpha * (mu**2 - r_square) * x - omega * y
        dy = alpha * (mu**2 - r_square) * y + omega * x

        # 耦合项
        theta = np.pi*(-1)**i
        delta = y*np.cos(theta) - x*np.sin(theta)
        dy += k * delta

        dZ[2 * i] = dx
        dZ[2 * i + 1] = dy
    return dZ


