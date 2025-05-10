import os
import random
import math

OBSTACLE_DIR = "./assets"
OBSTACLE_URDF_PATH = os.path.join(OBSTACLE_DIR, "obstacle.urdf")

BLOCK_UNIT_W = 0.05  # 单位小块宽度/长度
BLOCK_UNIT_H = 0.01  # 单位小块高度

NUM_REGIONS = 150  # 固定数量
REGION_XY_BLOCKS_RANGE = (1, 3)  # 每个区域在x/y方向上拼接的块数范围

PLACEMENT_WIDTH = 5.0
PLACEMENT_HEIGHT = 4.0
MIN_SEPARATION = 0.1  # 最小间隔

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

def regions_overlap(x1, y1, sx1, sy1, x2, y2, sx2, sy2, min_dist):
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    min_dx = (sx1 + sx2) / 2 + min_dist
    min_dy = (sy1 + sy2) / 2 + min_dist
    return dx < min_dx and dy < min_dy

def generate_obstacle_regions():
    if not os.path.exists(OBSTACLE_DIR):
        os.makedirs(OBSTACLE_DIR)

    links = []
    joints = []
    placed_regions = []

    attempts = 0
    max_attempts = 1000

    while len(placed_regions) < NUM_REGIONS and attempts < max_attempts:
        num_x = random.randint(*REGION_XY_BLOCKS_RANGE)
        num_y = random.randint(*REGION_XY_BLOCKS_RANGE)

        size_x = num_x * BLOCK_UNIT_W
        size_y = num_y * BLOCK_UNIT_W
        size_z = BLOCK_UNIT_H

        # x ∈ [0, PLACEMENT_WIDTH - size_x]
        # y ∈ [0, PLACEMENT_HEIGHT - size_y]
        x = random.uniform(size_x / 2, PLACEMENT_WIDTH - size_x / 2)
        y = random.uniform(size_y / 2, PLACEMENT_HEIGHT - size_y / 2)
        z = size_z / 2  # 地面上方

        conflict = False
        for (px, py, psx, psy) in placed_regions:
            if regions_overlap(x, y, size_x, size_y, px, py, psx, psy, MIN_SEPARATION):
                conflict = True
                break

        if not conflict:
            name = f"region_{len(placed_regions)}"
            link, joint = generate_region_link_and_joint(name, x - PLACEMENT_WIDTH / 2, y - PLACEMENT_HEIGHT / 2, z, size_x, size_y, size_z)
            links.append(link)
            joints.append(joint)
            placed_regions.append((x, y, size_x, size_y))

        attempts += 1

    if len(placed_regions) < NUM_REGIONS:
        print(f"Warning: Only placed {len(placed_regions)} regions out of {NUM_REGIONS}")

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
