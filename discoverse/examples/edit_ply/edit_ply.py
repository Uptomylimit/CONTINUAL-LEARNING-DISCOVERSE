import numpy as np
from plyfile import PlyData, PlyElement

import os
from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSERT_DIR

# Example usage
input_file = os.path.join(DISCOVERSE_ASSERT_DIR+"/3dgs", "object/bridge_es.ply")  # Replace with your input PLY file
output_file = os.path.join(DISCOVERSE_ASSERT_DIR+"/3dgs", "object/bridge_es2.ply")  # Replace with your desired output PLY file

ply_data = PlyData.read(input_file)
# print(ply_data)

# 提取点数据
vertices = ply_data['vertex']
points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

# 修改Y坐标
points[:, 0] += 0
points[:, 1] += 0.0025
points[:, 2] += 0.003

ply_data['vertex']['x'] = points[:, 0]
ply_data['vertex']['y'] = points[:, 1]
ply_data['vertex']['z'] = points[:, 2]

# 保存修改后的PLY文件
ply_data.write(output_file)

ply_data = PlyData.read(output_file)
# print(ply_data)

print(f"修改后的点云已保存到 {output_file}")
