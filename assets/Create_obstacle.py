import os
import random

OBSTACLE_DIR = "./assets"
OBSTACLE_URDF_PATH = os.path.join(OBSTACLE_DIR, "obstacle.urdf")

BLOCK_UNIT_W = 0.05  # 单位小块宽度/长度
BLOCK_UNIT_H = 0.01  # 单位小块高度

NUM_REGIONS_RANGE = (10, 10)  # 随机区域数量
REGION_XY_BLOCKS_RANGE = (1, 3)  # 每个区域在x/y方向上拼接的块数范围
REGION_POSITION_RANGE = 0.8  # 区域放置范围（±米）

def generate_region_link_and_joint(name, x, y, z, size_x, size_y, size_z):
    link = f"""
    <link name="{name}">
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{size_x} {size_y} {size_z}"/>
            </geometry>
            <material name="gray">
                <color rgba="0.5 0.5 0.5 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{size_x} {size_y} {size_z}"/>
            </geometry>
        </collision>
        <inertial>
            <mass value="0.1"/>
            <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
        </inertial>
    </link>
    """

    joint = f"""
    <joint name="{name}_joint" type="fixed">
        <parent link="base_link"/>
        <child link="{name}"/>
        <origin xyz="{x} {y} {z}" rpy="0 0 0"/>
    </joint>
    """
    return link, joint

def generate_obstacle_regions():
    if not os.path.exists(OBSTACLE_DIR):
        os.makedirs(OBSTACLE_DIR)

    links = []
    joints = []

    num_regions = random.randint(*NUM_REGIONS_RANGE)

    for region_id in range(num_regions):
        # 随机区域尺寸（以block为单位）
        num_x = random.randint(*REGION_XY_BLOCKS_RANGE)
        num_y = random.randint(*REGION_XY_BLOCKS_RANGE)

        size_x = num_x * BLOCK_UNIT_W
        size_y = num_y * BLOCK_UNIT_W
        size_z = BLOCK_UNIT_H

        # 放置坐标
        x = random.uniform(-REGION_POSITION_RANGE, 0)
        y = random.uniform(-REGION_POSITION_RANGE, REGION_POSITION_RANGE)
        z = size_z / 2  # 地面上方

        name = f"region_{region_id}"
        link, joint = generate_region_link_and_joint(name, x, y, z, size_x, size_y, size_z)
        links.append(link)
        joints.append(joint)

    urdf_content = f"""<?xml version="1.0" ?>
<robot name="obstacle_regions">
    <link name="base_link"/>
    {''.join(links)}
    {''.join(joints)}
</robot>
"""

    with open(OBSTACLE_URDF_PATH, "w") as f:
        f.write(urdf_content)

    print(f"Obstacle URDF written to: {OBSTACLE_URDF_PATH}")

if __name__ == "__main__":
    generate_obstacle_regions()
