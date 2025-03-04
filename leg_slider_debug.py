import time
import pybullet as p
import pybullet_data
import numpy as np

# PyBullet 设置
client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

p.setGravity(0, 0, -10)
p.setRealTimeSimulation(0)

# 加载URDF模型
camera_settings = {
    "target_position": [-0.2, 0, 0.3],
    "distance": 1.5,
    "yaw": -90,
    "pitch": -40,
}
plane_id = p.loadURDF("plane.urdf", useMaximalCoordinates=True)
p.changeDynamics(plane_id, -1, lateralFriction=5)
robot_id = p.loadURDF('./robot/phantomx.urdf', (0, 0, 0.03),
                      p.getQuaternionFromEuler([0, 0, -np.pi]))
p.resetDebugVisualizerCamera(
    cameraDistance=camera_settings["distance"],
    cameraYaw=camera_settings["yaw"],
    cameraPitch=camera_settings["pitch"],
    cameraTargetPosition=camera_settings["target_position"]
)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

par = [-1, -1, -1, 1, 1, 1]

# # 创建滑条
# slider_ids = []
# for i in range(3):
#     slider_id = p.addUserDebugParameter(f'Z[{i}]', -1, 1, 0)
#     slider_ids.append(slider_id)

# # 跟踪显示的文本ID，以便可以更新文本而不是创建新的文本
# text_id = None

# while True:
#     p.stepSimulation()

#     # 更新关节控制
#     for index in range(6):
#         p.setJointMotorControl2(bodyUniqueId=robot_id,
#                                 jointIndex=4 * index + 1,
#                                 controlMode=p.POSITION_CONTROL,
#                                 targetPosition=p.readUserDebugParameter(slider_ids[0]) * par[index])

#         p.setJointMotorControl2(bodyUniqueId=robot_id,
#                                 jointIndex=4 * index + 3,
#                                 controlMode=p.POSITION_CONTROL,
#                                 targetPosition=p.readUserDebugParameter(slider_ids[1]))

#         p.setJointMotorControl2(bodyUniqueId=robot_id,
#                                 jointIndex=4 * index + 4,
#                                 controlMode=p.POSITION_CONTROL,
#                                 targetPosition=p.readUserDebugParameter(slider_ids[2]))


# 创建滑条
slider_ids = []
for leg_index in range(6):  # 六条腿
    for joint_index in range(3):  # 每条腿3个关节
        slider_id = p.addUserDebugParameter(
            f'Leg {leg_index} Joint {joint_index}', -1, 1, 0)
        slider_ids.append(slider_id)

# 跟踪显示的文本ID，以便可以更新文本而不是创建新的文本
text_id = None
velocity_id = None

while True:
    p.stepSimulation()

    # 更新关节控制
    for leg_index in range(6):  # 六条腿
        # 控制第1个关节（对应URDF中的第1个关节）
        target_position_1 = p.readUserDebugParameter(
            slider_ids[leg_index * 3])*par[leg_index]
        p.setJointMotorControl2(bodyUniqueId=robot_id,
                                jointIndex=4 * leg_index + 1,  # 第1个关节
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=target_position_1)

        # 控制第2个关节（对应URDF中的第3个关节）
        target_position_2 = p.readUserDebugParameter(
            slider_ids[leg_index * 3 + 1])
        p.setJointMotorControl2(bodyUniqueId=robot_id,
                                jointIndex=4 * leg_index + 3,  # 第3个关节
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=target_position_2)

        # 控制第3个关节（对应URDF中的第4个关节）
        target_position_3 = p.readUserDebugParameter(
            slider_ids[leg_index * 3 + 2])
        p.setJointMotorControl2(bodyUniqueId=robot_id,
                                jointIndex=4 * leg_index + 4,  # 第4个关节
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=target_position_3)
    # 获取机器人基座的位置和方向
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    # 获取机器人基座的线速度和角速度
    linear_velocity, angular_velocity = p.getBaseVelocity(robot_id)

    # 显示机器人的三维坐标
    if text_id is None:
        text_id = p.addUserDebugText(f'Position: {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}', [
                                     pos[0], pos[1], pos[2] + 0.1], textColorRGB=[1, 0, 0], textSize=1.5)
        # 添加速度的文本显示
        velocity_text = f'Linear Vel: {linear_velocity[0]:.2f}, {linear_velocity[1]:.2f}, {linear_velocity[2]:.2f}\n' \
            f'Angular Vel: {angular_velocity[0]: .2f}, {angular_velocity[1]: .2f}, {angular_velocity[2]: .2f}'
        velocity_id = p.addUserDebugText(velocity_text, [
            pos[0], pos[1], pos[2] + 0.3], textColorRGB=[0, 1, 0], textSize=1.5)
    else:
        p.removeUserDebugItem(text_id)
        p.removeUserDebugItem(velocity_id)
        text_id = p.addUserDebugText(f'Position: {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}', [
                                     pos[0], pos[1], pos[2] + 0.1], textColorRGB=[1, 0, 0], textSize=1.5)
        # 添加速度的文本显示
        velocity_text = f'Linear Vel: {linear_velocity[0]:.2f}, {linear_velocity[1]:.2f}, {linear_velocity[2]:.2f}\n' \
            f'Angular Vel: {angular_velocity[0]: .2f}, {angular_velocity[1]: .2f}, {angular_velocity[2]: .2f}'
        velocity_id = p.addUserDebugText(velocity_text, [
            pos[0], pos[1], pos[2] + 0.3], textColorRGB=[0, 1, 0], textSize=1.5)

    time.sleep(1. / 240.)
