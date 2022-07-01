import cupoch as cph
import copy
import time
import numpy as np
import os

import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../misc"))
import meshes


cubic_size = 1.0
voxel_resolution = 128.0
mesh = meshes.bunny()
target = cph.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
    mesh,
    voxel_size=cubic_size / voxel_resolution,
    min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),
    max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2),
)

dt = cph.geometry.DistanceTransform(cubic_size / voxel_resolution, int(voxel_resolution))
dt.compute_edt(target)
cph.visualization.draw_geometries([target, dt])
