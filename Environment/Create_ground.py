import numpy as np

tile_size = 0.1
rows = 40  # y 方向格子数量
cols = 15  # x 方向格子数量

rng = np.random.default_rng()
heights = rng.uniform(0.03, 0.1, size=(rows, cols))

urdf_lines = [
    '<?xml version="1.0" ?>',
    '<robot name="random_terrain">',
    '  <link name="base_link"/>'
]

for i in range(rows):
    for j in range(cols):
        height = heights[i, j]
        x = (j - cols / 2) * tile_size + tile_size / 2
        y = (i - rows / 2) * tile_size + tile_size / 2
        z = height / 2
        name = f"tile_{i}_{j}"

        # 设置蓝白相间颜色
        if (i + j) % 2 == 0:
            color = "0.59 0.96 1 1"  # 淡蓝色
        else:
            color = "1 1 1 1"      # 白色

        urdf_lines.append(f'''
  <link name="{name}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{tile_size} {tile_size} {height:.4f}"/>
      </geometry>
      <material name="color_{i}_{j}">
        <color rgba="{color}"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{tile_size} {tile_size} {height:.4f}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
  <joint name="joint_{name}" type="fixed">
    <parent link="base_link"/>
    <child link="{name}"/>
    <origin xyz="{x:.3f} {y:.3f} {z:.3f}" rpy="0 0 0"/>
  </joint>
        ''')


urdf_lines.append('</robot>')

with open("./Environment/custom_ground.urdf", "w") as f:
    f.writelines(urdf_lines)
