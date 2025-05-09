import matplotlib.pyplot as plt
import numpy as np

# 读取数据
file_path = './assets/rpy_log.txt'
data = np.loadtxt(file_path, delimiter=',')

roll, pitch, yaw = data[:, 0], data[:, 1], data[:, 2]
timesteps = np.arange(len(roll))

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(timesteps, roll, label='Roll', color='r')
plt.plot(timesteps, pitch, label='Pitch', color='g')
plt.plot(timesteps, yaw, label='Yaw', color='b')
plt.xlabel('Timestep')
plt.ylabel('Angle (radians)')
plt.title('Hexapod Orientation over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
