def generate_stair_urdf_single_link(
    urdf_path="./assets/stair_obstacle.urdf",
    num_steps=15,
    step_width=2.2,
    step_depth=0.2,
    step_height_increment=0.01,
    start_x=0,
    start_y=0
):
    urdf = [
        '<?xml version="1.0" ?>',
        '<robot name="stair_obstacle">',
        '  <link name="stairs">'
    ]

    for i in range(num_steps):
        height = (num_steps - i) * step_height_increment
        x_pos = start_x + i * step_depth
        y_pos = start_y
        z_pos = height / 2  # box center is at half the height

        # visual + collision combined
        visual = f'''
    <visual>
      <origin xyz="{x_pos} {y_pos} {z_pos}" rpy="0 0 0"/>
      <geometry>
        <box size="{step_depth} {step_width} {height}"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>'''
        collision = f'''
    <collision>
      <origin xyz="{x_pos} {y_pos} {z_pos}" rpy="0 0 0"/>
      <geometry>
        <box size="{step_depth} {step_width} {height}"/>
      </geometry>
    </collision>'''

        urdf.append(visual)
        urdf.append(collision)

    # inertial
    urdf.append('''
    <inertial>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </inertial>''')

    urdf.append('  </link>')
    urdf.append('</robot>')

    with open(urdf_path, 'w') as f:
        f.write('\n'.join(urdf))

    print(f"Single-link stair URDF saved to {urdf_path}")

# 运行脚本
generate_stair_urdf_single_link()
