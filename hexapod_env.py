import numpy as np
import pybullet_data
import pybullet as p

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
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
    [90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0]
])

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

                    # 计算分母
                    denominator = np.sqrt(xm**2 + ym**2)
                    # 防止分母为零
                    denominator = np.where(denominator == 0, 1e-8, denominator)

                    r_im = np.array([xm / denominator,
                                    ym / denominator])
                    coupling = k * R(theta_im) @ r_im

                    dxdt += coupling[0]
                    dydt += coupling[1]

        if i >= N1:  # 如果是 N2 的振荡器，则使用 omega2
            dxdt = alpha * (mu**2 - x**2 - y**2) * x - omega2 * y
            dydt = beta * (mu**2 - x**2 - y**2) * y + omega2 * x

            m = i % N1
            xm, ym = Z[2 * m], Z[2 * m + 1]
            # Use the phase difference from the matrix
            theta_im = theta_matrix[i, m]

            # 计算分母
            denominator = np.sqrt(xm**2 + ym**2)
            # 防止分母为零
            denominator = np.where(denominator == 0, 1e-8, denominator)

            r_im = np.array([xm / denominator,
                             ym / denominator])
            coupling = k * R(theta_im) @ r_im

            dxdt += coupling[0]
            dydt += coupling[1]

        dZ[2 * i] = dxdt
        dZ[2 * i + 1] = dydt
    return dZ


class HexapodCPGEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'none']}

    def __init__(self, render_mode='none'):

        self.camera_settings = {
            "target_position": [-0.2, 0, 0.3],
            "distance": 1.5,
            "yaw": -90,
            "pitch": -45
        }
        self.render_mode = render_mode

        if render_mode == 'human':
            p.connect(p.GUI)
            # Optionally disable the GUI
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        else:
            p.connect(p.DIRECT)  # Non-graphical version

        self.action_space = spaces.Box(
            np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1,
                     0.06, 0.06, 0.06]),  # 动作空间下限
            np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1]),  # 动作空间上限
            dtype=np.float32
        )

        num_controlled_joints = 18  # 每个腿3个关节，共6腿
        self.observation_space = spaces.Box(
            low=np.concatenate([
                -np.ones(num_controlled_joints),  # 归一化后的关节位置范围为 [-1, 1]
                -np.ones(3),  # 归一化后的线速度范围为 [-1, 1]
                -np.ones(3),  # 归一化后的角速度范围为 [-1, 1]
            ]),
            high=np.concatenate([
                np.ones(num_controlled_joints),
                np.ones(3),
                np.ones(3),
            ]),
            dtype=np.float32
        )

        self.mean_observation = np.zeros(num_controlled_joints+6)
        self.std_observation = np.ones(num_controlled_joints+6)
        self.alpha_update = 0.01  # 平滑常数

        # 初始化CPG模型参数
        self.N1, self.N2 = 6, 6
        self.current_step, self.dt = 0, 1./100  # 时间步长
        self.alpha, self.beta, self.mu, self.omega1, self.omega2, self.k = 100, 100, 3.14, 30, 30, 120
        self.text_id = None
        self.velocity_id = None
        self.angular_velocity_id = None

        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)  # 重置随机种子

        self.text_id = None
        self.velocity_id = None
        self.angular_velocity_id = None
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.loadURDF("plane100.urdf", basePosition=[0, 0, 0])

        self.rest_poses = [0, 0.65, -0.32]

        # 随机初始化机器人的位置
        # 随机初始化机器人的x和y位置
        # 仅生成x和y的随机值
        initial_position = self.np_random.uniform(-0.01, 0.01, size=2)

        # 将固定的高度值添加到x和y的随机位置上
        initial_position = np.append(
            initial_position, 0.19)  # 将高度值0.19添加到数组的最后

        initial_orientation = p.getQuaternionFromEuler([0, 0, -np.pi])

        self.robot_id = p.loadURDF(
            './robot/phantomx.urdf', initial_position, initial_orientation, useFixedBase=False)

        for i in range(0, 6):
            p.resetJointState(self.robot_id, 4*i+1, self.rest_poses[0])
            p.resetJointState(self.robot_id, 4*i+3, self.rest_poses[1])
            p.resetJointState(self.robot_id, 4*i+4, self.rest_poses[2])

        self.current_step = 0
        self.t = 0

        # 初始条件：为不同组指定不同初始条件
        Z0 = []
        for i in range(3):
            Z0.extend([0, self.mu, 0, -self.mu])  # 另一组初始相位
        for i in range(3):
            Z0.extend([-self.mu, 0,  self.mu, 0])  # 另一组初始相位

        self.Z = np.array(Z0)

        if self.render_mode == 'human':
            p.resetDebugVisualizerCamera(
                cameraDistance=self.camera_settings["distance"],
                cameraYaw=self.camera_settings["yaw"],
                cameraPitch=self.camera_settings["pitch"],
                cameraTargetPosition=self.camera_settings["target_position"]
            )
            # Ensure rendering is opened
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        elif self.render_mode == 'rgb_array':
            p.resetDebugVisualizerCamera(
                cameraDistance=self.camera_settings["distance"],
                cameraYaw=self.camera_settings["yaw"],
                cameraPitch=self.camera_settings["pitch"],
                cameraTargetPosition=self.camera_settings["target_position"]
            )
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        observation = self._get_observation()
        info = {"initial reset"}  # 你可以在这里添加更多有用的信息

        return observation, info

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # k1, k2, k3 = action
        # 解包动作，每个leg组的三个关节有独立的控制参数,共3*3=9个参数
        k1, k2, k3 = action[:3], action[3:6], action[6:9]
        self.Z = solve_ivp(coupled_hopf, [(self.t+self.current_step)*self.dt, (self.t+self.current_step+1)*self.dt], self.Z,
                           args=(self.N1, self.N2, self.alpha, self.beta, self.mu, self.omega1, self.omega2, self.k), method='RK45').y[:, -1]

        self.current_step += 1

        par1 = [-1, -1, -1, 1, 1, 1]
        position, _ = p.getBasePositionAndOrientation(self.robot_id)
        position = np.array(position)
        camera_target_p = position + self.camera_settings["target_position"]

        p.resetDebugVisualizerCamera(
            cameraDistance=self.camera_settings["distance"],
            cameraYaw=self.camera_settings["yaw"],
            cameraPitch=self.camera_settings["pitch"],
            cameraTargetPosition=camera_target_p)

    # 获取机器人基座的线速度和角速度
        # linear_velocity, angular_velocity = p.getBaseVelocity(self.robot_id)

        # 显示机器人的三维坐标
        # if self.text_id is None:
        #     self.text_id = p.addUserDebugText(f'Position: {position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}', [
        #         position[0], position[1]-0.1, position[2] + 0.1], textColorRGB=[1, 0, 0], textSize=1.5)
        #     self.velocity_id = p.addUserDebugText(f'Linear Vel: {linear_velocity[0]:.2f}, {linear_velocity[1]:.2f}, {linear_velocity[2]:.2f}',
        #                                           [position[0], position[1]-0.1, position[2] + 0.3], textColorRGB=[0, 1, 0], textSize=1.5)
        #     self.angular_velocity_id = p.addUserDebugText(f'Angular Vel: {angular_velocity[0]:.2f}, {angular_velocity[1]:.2f}, {angular_velocity[2]:.2f}',
        #                                                   [position[0], position[1]-0.1, position[2] + 0.5], textColorRGB=[0, 0, 1], textSize=1.5)
        # else:
        #     p.removeUserDebugItem(self.text_id)
        #     p.removeUserDebugItem(self.velocity_id)
        #     p.removeUserDebugItem(self.angular_velocity_id)
        #     self.text_id = p.addUserDebugText(f'Position: {position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}', [
        #         position[0], position[1]-0.1, position[2] + 0.1], textColorRGB=[1, 0, 0], textSize=1.5)
        #     self.velocity_id = p.addUserDebugText(f'Linear Vel: {linear_velocity[0]:.2f}, {linear_velocity[1]:.2f}, {linear_velocity[2]:.2f}',
        #                                           [position[0], position[1]-0.1, position[2] + 0.3], textColorRGB=[0, 1, 0], textSize=1.5)
        #     self.angular_velocity_id = p.addUserDebugText(f'Angular Vel: {angular_velocity[0]:.2f}, {angular_velocity[1]:.2f}, {angular_velocity[2]:.2f}',
        #                                                   [position[0], position[1]-0.1, position[2] + 0.5], textColorRGB=[0, 0, 1], textSize=1.5)

        # 应用动作到机器人关节
        for index in range(0, 6):
            group = index % 3  # 计算当前索引属于哪一组leg,1、4，2、5，3、6两两对称，共三组

            # 髋关节
            hip_target = k1[group]*self.Z[2*index] * \
                par1[index]+self.rest_poses[0]
            p.setJointMotorControl2(bodyUniqueId=self.robot_id, jointIndex=4 * index + 1,
                                    controlMode=p.POSITION_CONTROL, targetPosition=hip_target)

            # 膝关节
            knee_target = max(
                0, (k2[group]*self.Z[2 * (index+6)] - 0.1))*(-1)+self.rest_poses[1]
            p.setJointMotorControl2(bodyUniqueId=self.robot_id, jointIndex=4 * index + 3,
                                    controlMode=p.POSITION_CONTROL, targetPosition=knee_target)

            # 踝关节
            ankle_target = max(
                0, (k3[group]*self.Z[2 * (index+6)]-0.1))+self.rest_poses[2]
            p.setJointMotorControl2(bodyUniqueId=self.robot_id, jointIndex=4 * index + 4,
                                    controlMode=p.POSITION_CONTROL, targetPosition=ankle_target)

        p.stepSimulation()

        observation = self._get_observation()
        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        info = {}  # 可以添加额外信息
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        # 获取基座的速度和加速度
        base_velocity = p.getBaseVelocity(self.robot_id)
        linear_velocity = np.array(base_velocity[0])  # 基座的线速度
        angular_velocity = np.array(base_velocity[1])  # 基座的角速度

        # 获取特定关节的位置
        joint_positions = []
        for index in range(6):
            # 获取每个关节的位置信息，并跳过4*index+2
            joint_positions.append(
                p.getJointState(self.robot_id, 4*index+1)[0])
            joint_positions.append(
                p.getJointState(self.robot_id, 4*index+3)[0])
            joint_positions.append(
                p.getJointState(self.robot_id, 4*index+4)[0])

        # 将关节位置和速度信息合并成一个观察向量
        observation = np.concatenate(
            [joint_positions, linear_velocity, angular_velocity])

        # 更新移动均值和标准差
        self.mean_observation = self.alpha_update * observation + \
            (1 - self.alpha_update) * self.mean_observation
        self.std_observation = self.alpha_update * \
            np.square(observation - self.mean_observation) + \
            (1 - self.alpha_update) * self.std_observation

        # 归一化观察量
        self.std_observation[self.std_observation < 1e-8] = 1e-8  # 避免除以零
        normalized_observation = (
            observation - self.mean_observation) / np.sqrt(self.std_observation)

        return normalized_observation

    def _compute_reward(self):
        # 获取机器人当前速度和位置
        target_velocity = np.array([0.5, 0])
        base_velocity = p.getBaseVelocity(self.robot_id)
        linear_velocity = np.array(base_velocity[0])  # 线速度

        # 获取机器人当前位置（假设p.getPosition返回格式为[x, y, z]）
        position, _ = p.getBasePositionAndOrientation(self.robot_id)
        y_position = position[1]  # y轴位置

        # 提取 x 和 y 平面的速度分量
        linear_velocity_xy = linear_velocity[:2]

        # 1. 计算速度方向和目标速度方向的余弦相似度
        target_speed = np.linalg.norm(target_velocity)
        current_speed = np.linalg.norm(linear_velocity_xy)

        # 如果机器人静止或目标速度为零，处理边界情况
        if target_speed == 0 or current_speed == 0:
            direction_similarity = 1.0 if target_speed == current_speed else 0.0
        else:
            direction_similarity = np.dot(target_velocity, linear_velocity_xy) / (
                target_speed * current_speed + 1e-6
            )

        # 2. 计算速度幅值误差
        speed_error = abs(target_speed - current_speed)

        # 3. 奖励函数计算：方向相似性和速度匹配
        direction_reward = max(0, direction_similarity)  # 确保只有在正向时才奖励
        speed_reward = 1.0 - speed_error / (target_speed + 1e-6)  # 归一化速度误差奖励

        # 4. 计算y轴位置偏差惩罚
        y_position_penalty = -abs(y_position)  # 惩罚项与y轴位置的绝对值成正比

        # 5. 计算最终奖励
        reward = 0.8 * direction_reward + 0.2 * \
            speed_reward + y_position_penalty  # 加入y轴位置偏差惩罚项
        return reward

    def _check_terminated(self):
        # 检查是否达到了环境的自然结束条件
        _, orientation = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, _ = p.getEulerFromQuaternion(orientation)
        if abs(roll) > 0.5 or abs(pitch) > 0.5:  # 检查是否翻倒
            return True
        # 可以添加更多自然结束的条件，例如达到目标位置等
        return False

    def _check_truncated(self):
        # 检查是否因为步数限制或其他非自然原因而结束
        if self.current_step >= 1000:  # 假设最大步数为1000
            return True
        return False

    def render(self, mode='none'):
        if self.render_mode == 'none':
            return None
        elif self.render_mode in ['human', 'rgb_array']:
            # 只在需要时设置视图和投影矩阵
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.camera_settings["target_position"],
                distance=self.camera_settings["distance"],
                yaw=self.camera_settings["yaw"],
                pitch=self.camera_settings["pitch"],
                roll=self.camera_settings["roll"],
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=self.camera_settings["fov"],
                aspect=self.camera_settings["aspect"],
                nearVal=self.camera_settings["near"],
                farVal=self.camera_settings["far"]
            )
            (_, _, px, _, _) = p.getCameraImage(width=960, height=720, viewMatrix=view_matrix,
                                                projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (720, 960, 4))

            if mode == 'rgb_array':
                return rgb_array[:, :, :3]  # 返回RGB数组
            else:
                # 在‘human’模式下无需返回，PyBullet已进行渲染
                return None

    def close(self):  # 关闭连接
        # 检查 PyBullet 是否已连接
        if p.isConnected():
            p.disconnect()  # 只有连接时才执行断开操作
