import numpy as np

tile_size = 0.1  # tile 的边长，单位：米
rows = 40  # y 方向格子数量
cols = 15  # x 方向格子数量
tile_mass = 0.1  # 每个 tile 的质量，单位：kg

rng = np.random.default_rng()
heights = rng.uniform(0.03, 0.1, size=(rows, cols))  # 随机生成高度

urdf_lines = [
    '<?xml version="1.0" ?>',
    '<robot name="random_terrain">',
    '  <link name="base_link">',
    '    <inertial>',
    '      <origin xyz="0 0 0" rpy="0 0 0"/>',
    '      <mass value="1.0"/>',
    '      <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0"/>',
    '    </inertial>',
    '  </link>'
]

for i in range(rows):
    for j in range(cols):
        height = heights[i, j]
        x = (j - cols / 2) * tile_size + tile_size / 2
        y = (i - rows / 2) * tile_size + tile_size / 2
        z = height / 2
        name = f"tile_{i}_{j}"

        # 设置蓝白相间颜色
        color = "0.59 0.96 1 1" if (i + j) % 2 == 0 else "1 1 1 1"

        # 计算惯性张量
        w = tile_size
        h = tile_size
        d = height
        ixx = (1/12) * tile_mass * (h**2 + d**2)
        iyy = (1/12) * tile_mass * (w**2 + d**2)
        izz = (1/12) * tile_mass * (w**2 + h**2)

        urdf_lines.append(f'''
  <link name="{name}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{w} {h} {d:.4f}"/>
      </geometry>
      <material name="color_{i}_{j}">
        <color rgba="{color}"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{w} {h} {d:.4f}"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{tile_mass}"/>
      <inertia ixx="{ixx:.6f}" ixy="0" ixz="0" iyy="{iyy:.6f}" iyz="0" izz="{izz:.6f}"/>
    </inertial>
  </link>
  <joint name="joint_{name}" type="fixed">
    <parent link="base_link"/>
    <child link="{name}"/>
    <origin xyz="{x:.3f} {y:.3f} {z:.3f}" rpy="0 0 0"/>
  </joint>
        ''')

urdf_lines.append('</robot>')

with open("./assets/custom_ground.urdf", "w") as f:
    f.writelines(urdf_lines)
