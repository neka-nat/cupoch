import cupoch as cph
import copy
import time
import numpy as np
import os

import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../misc'))
import meshes


cubic_size = 1.0
voxel_resolution = 128.0
mesh = meshes.bunny()
target = cph.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
    mesh,
    voxel_size=cubic_size / voxel_resolution,
    min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),
    max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2))

vis = cph.visualization.Visualizer()
vis.create_window()
vis.add_geometry(target)
query = cph.geometry.LineSet.create_from_triangle_mesh(mesh)
query.translate([-0.15, 0.0, 0.0])

dh = 0.02
for i in range(15):
    query.translate([dh, 0.0, 0.0])
    ans = cph.collision.compute_intersection(target, query, 0.0)
    print(ans)
    target.paint_uniform_color([1.0, 1.0, 1.0])
    query.paint_uniform_color([1.0, 1.0, 1.0])
    target.paint_indexed_color(ans.get_first_collision_indices(), [1.0, 0.0, 0.0])
    query.paint_indexed_color(ans.get_second_collision_indices(), [0.0, 1.0, 0.0])
    if i == 0:
        vis.add_geometry(query)
    vis.update_geometry(target)
    vis.update_geometry(query)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.1)
vis.destroy_window()