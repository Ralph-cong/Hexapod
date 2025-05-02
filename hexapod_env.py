import numpy as np
import pybullet_data
import pybullet as p

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from scipy.integrate import solve_ivp
from CPGs.CPG import hopf_oscillator


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
            low = np.concatenate([
                np.array([0.8]),
                np.ones(14)*0.5,
                ]),
            high= np.concatenate([
                np.array([1.2]),
                np.ones(14)*1.5,
                ]),
                dtype=np.float64
        )

        self.observation_space = spaces.Box(
            low=-np.ones(59),
            high=np.ones(59),
            dtype=np.float64
        )

        self.init_pos = np.array([0.15, 0, 0.05])
        self.init_ori = np.array(p.getQuaternionFromEuler([0, 0, -np.pi/2]))
        self.goal = np.array([1.0, 0])  # 目标位置

        # CPG params
        self.alpha, self.mu, self.omega, self.k = 100, 3, np.pi, 10
        self.A = np.zeros(14)

        self.current_step, self.dt = 0, 1./100  # 时间步长

        self.max_h = 0.

        # reward weights
        self.w_h, self.w_th, self.w_d = 0.5, 0.3, 0.2

        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)  # 重置随机种子

        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # p.loadURDF("./assets/custom_ground.urdf", basePosition=[0, 0, 0],useFixedBase=True)
        p.loadURDF("plane.urdf", useMaximalCoordinates=True)
        self.mid_joint_value = [0, 0, 0]

        self.robot_id = p.loadURDF(
            './assets/robot/hexapod_34/urdf/hexapod_34.urdf', self.init_pos, self.init_ori, useFixedBase=False)

        for i in range(0, 6):
            p.resetJointState(self.robot_id, 3*i, self.mid_joint_value[0])
            p.resetJointState(self.robot_id, 3*i+1, self.mid_joint_value[1])
            p.resetJointState(self.robot_id, 3*i+2, self.mid_joint_value[2])

        self.current_step = 0
        self.last_position = np.array([0, 0, 0])

        self.Z = np.array([0, self.mu, 0, -self.mu])

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
        self.A = action[1:]
        self.Z = solve_ivp(hopf_oscillator, [(self.current_step)*self.dt, (self.current_step+1)*self.dt], self.Z,
                           args=(self.alpha, self.mu, action[0]*self.omega, self.k), method='RK45').y[:, -1]

        self.current_step += 1

        position, _ = p.getBasePositionAndOrientation(self.robot_id)
        position = np.array(position)
        self.last_position = position.copy()
        camera_target_p = position + self.camera_settings["target_position"]

        p.resetDebugVisualizerCamera(
            cameraDistance=self.camera_settings["distance"],
            cameraYaw=self.camera_settings["yaw"],
            cameraPitch=self.camera_settings["pitch"],
            cameraTargetPosition=camera_target_p)

        self.joint_mapping(self.A)
        p.stepSimulation()

        observation = self._get_observation()
        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        info = {} 
        return observation, reward, terminated, truncated, info

    def joint_mapping(self, A):
        A1, A2, A3 = A[0:2], A[2:8], A[8:14]
        for idx in range(0, 6):
            group_idx = idx % 2  # 计算当前索引属于哪一组leg, 0,2,4为一组，1,3,5为另一组
            # hip
            hip_target = A1[group_idx]*self.Z[2*group_idx]
            p.setJointMotorControl2(bodyUniqueId=self.robot_id, jointIndex=3 * idx,
                                    controlMode=p.POSITION_CONTROL, targetPosition=(hip_target+self.mid_joint_value[0]))
            # knee
            knee_target = max(
                0, (A2[idx]*(1-self.Z[2*group_idx])**2))
            p.setJointMotorControl2(bodyUniqueId=self.robot_id, jointIndex=3 * idx + 1,
                                    controlMode=p.POSITION_CONTROL, targetPosition=(knee_target+self.mid_joint_value[1]))
            # ankle
            ankle_target = -A3[idx]*knee_target
            p.setJointMotorControl2(bodyUniqueId=self.robot_id, jointIndex=3 * idx + 2,
                                    controlMode=p.POSITION_CONTROL, targetPosition=(ankle_target+self.mid_joint_value[2]))

    def _get_observation(self):
        # 获取特定关节的位置
        joint_pos = np.array([p.getJointState(self.robot_id, idx)[0] for idx in range(18)])/np.pi
        joint_vel = np.array([p.getJointState(self.robot_id, idx)[1] for idx in range(18)])/10

        # 获取基座的位置和朝向
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        orientation = np.array(p.getEulerFromQuaternion(orientation))

        # 将关节位置和速度信息合并成一个观察向量
        observation = np.concatenate(
            [joint_pos, joint_vel, self.goal, position, orientation, self.A, np.array([self.max_h])])

        return observation

    def _compute_reward(self):

        pos, ori = p.getBasePositionAndOrientation(self.robot_id)
        pos = np.array(pos)
        # Height Reward
        if pos[2] > self.max_h + 0.04:
            r_h = pos[2] - self.max_h
        else:
            r_h = 0

        # Direction Reward
        theta = np.arctan2(pos[1]-self.init_pos[1], pos[0]-self.init_pos[0])
        if theta < np.pi/6:
            r_theta = np.dot(self.goal-self.init_pos[:2], [np.cos(theta), np.sin(theta)])/np.linalg.norm(self.goal-self.init_pos[:2])
        else:
            r_theta = 0
        
        # Distance Reward
        d_n = np.linalg.norm(pos[:2]-self.init_pos[:2])
        d_t = np.linalg.norm(self.goal-self.init_pos[:2])
        r_d = np.exp(-(d_n-d_t))
        
        # Stability Reward
        roll, pitch, _ = p.getEulerFromQuaternion(ori)
        if pos[2] < self.max_h + 0.04 or abs(roll)>np.pi/6 or abs(pitch)>np.pi/6:
            r_s = -100
        else:
            r_s = 0

        # Forward Reward
        if pos[0]-self.last_position[0]< 0.01 and self.current_step > 0:
            r_f = -0.3
        else:
            r_f = 0

        reward = self.w_h*r_h + self.w_th*r_theta + self.w_d*r_d + r_s + r_f
        return reward

    def _check_terminated(self):
        # 检查是否达到了环境的自然结束条件
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, _ = p.getEulerFromQuaternion(orientation)

        # check if the robot has fallen
        if position[2] < self.max_h + 0.02 or abs(roll) > np.pi/3 or abs(pitch) > np.pi/3:  # 检查是否翻倒
            return True
        # check if the robot has moved too far in the Y-axis direction
        if abs(position[1]-self.init_pos[1]) > 0.8:
            return True
        # check if the robot has reached the goal
        if abs(position[0]-self.goal[0]) < 0.2:
            return True
        
        return False

    def _check_truncated(self):
        # 检查是否因为步数限制或其他非自然原因而结束
        if self.current_step >= 7000:  
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
