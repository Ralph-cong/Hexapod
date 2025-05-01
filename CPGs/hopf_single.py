import time
import pybullet as p
import pybullet_data
import numpy as np
from scipy.integrate import solve_ivp


def R(theta):
    """耦合矩阵 R，根据相位差 theta 返回旋转矩阵"""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


# 相位差矩阵
# 三足
theta_matrix_degrees = np.array([
    [0, 180, 0, 0, 0, 180, 0, 0, 0, 0, 0, 0],
    [-180, 0, -180, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 180, 0, 180, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -180, 0, -180, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 180, 0, 180, 0, 0, 0, 0, 0, 0],
    [-180, 0, 0, 0, -180, 0, 0, 0, 0, 0, 0, 0],
    [-90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -90, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -90, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -90, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -90, 0, 0, 0, 0, 0, 0]
])

# four leg???编号不对
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

# # 四足,根据仿真编号写的，可以检查一下？
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

            r_im = np.array([xm / np.sqrt(xm**2 + ym**2),
                            ym / np.sqrt(xm**2 + ym**2)])
            coupling = k * R(theta_im) @ r_im

            dxdt += coupling[0]
            dydt += coupling[1]

        dZ[2 * i] = dxdt
        dZ[2 * i + 1] = dydt
    return dZ


# PyBullet 设置

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 设置时间步长，例如设置为1/240秒
# p.setTimeStep(0.001)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(0)

# Load the URDFs
# plane_id = p.loadURDF("plane100.urdf", useMaximalCoordinates=True)
plane_id = p.loadURDF("./Environment/custom_ground.urdf", basePosition=[0, 0, 0],useFixedBase=True)


p.changeDynamics(plane_id, -1, lateralFriction=6)
r_ind = p.loadURDF('./hexapod_34/urdf/hexapod_34.urdf', (0, 0, 0.3),
                   p.getQuaternionFromEuler([0, 0, 0]))

# num_joints = p.getNumJoints(r_ind)
# print(f"机器人关节总数: {num_joints}")
# for i in range(num_joints):
#     joint_info = p.getJointInfo(r_ind, i)
#     print(f"关节索引: {i}, 名称: {joint_info[1].decode('utf-8')}")

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0,
                             cameraPitch=-40, cameraTargetPosition=[0, -0.35, 0.5])

# 初始化振荡器参数和状态
N1, N2 = 6, 6
alpha, beta, mu, omega1, omega2, k = 100, 100, 3.14, 6, 12, 100
# 初始条件
Z = np.array([0, 0.3] * (N1 + N2))
t_current = 0
dt = 1./240  # 与 PyBullet 时间步匹配
peak1 = 3.42
peak2 = 1.9

par1 = [1, 1, 1, -1, -1, -1]


while True:
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
    sol = solve_ivp(coupled_hopf, [t_current, t_current+dt], Z,  args=(
        N1, N2, alpha, beta, mu, omega1, omega2, k))
    Z = sol.y[:, -1]  # 更新状态
    t_current += dt

    # 设置关节1到6的目标位置
    for index in range(0, 6):  # 根据模型的输出更新机器人的关节角
        target_position_1 = Z[2*index]/peak1*par1[index]*0.45
        p.setJointMotorControl2(bodyUniqueId=r_ind,
                                jointIndex=3 * index,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=target_position_1)
        target_position_2 = max(
            0, (Z[2 * (index+6)]/peak2 * 1 - 0.1))*(-1)+0.35
        p.setJointMotorControl2(bodyUniqueId=r_ind,
                                jointIndex=3 * index + 1,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=target_position_2)
        target_position_3 = max(0, (Z[2 * (index+6)]/peak2*0.95-0.1))-0.35
        p.setJointMotorControl2(bodyUniqueId=r_ind,
                                jointIndex=3 * index + 2,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=target_position_3)
    p.stepSimulation()
