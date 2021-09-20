import numpy as np
import cupoch as cph

import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../misc'))
import time
import meshes

kin = cph.kinematics.KinematicChain("../../testdata/tello.urdf")

cubic_size = 2.0
voxel_resolution = 128.0
mesh = meshes.bunny()
mesh = mesh.scale(7.0)

target = cph.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
    mesh,
    voxel_size=cubic_size / voxel_resolution,
    min_bound=(-cubic_size / 2, -cubic_size / 2, -cubic_size / 2),
    max_bound=(cubic_size / 2, cubic_size / 2, cubic_size / 2))
gp = cph.geometry.Graph.create_from_axis_aligned_bounding_box([-1.0, -1.0, -1.0],
                                                              [1.0, 1.0, 1.0],
                                                              [10, 20, 5])
planner = cph.planning.Pos3DPlanner(gp, 0.2)
planner.add_obstacle(target)
planner.update_graph()
path = planner.find_path(np.array([-0.7, -0.7, -0.7]), np.array([0.7, 0.7, 0.7]))
print(path)
pathline = cph.geometry.LineSet(path)
pathline.paint_uniform_color([1.0, 0.0, 0.0])
drone_trans = np.identity(4)
drone_trans[1:3, 1:3] = [[np.cos(np.pi * 0.5), np.sin(np.pi * 0.5)], [-np.sin(np.pi * 0.5), np.cos(np.pi * 0.5)]]
drone_trans[:3, 3] = path[0]
poses = kin.forward_kinematics({}, drone_trans)
geoms = kin.get_transformed_visual_geometry_map(poses)

vis = cph.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pathline)
vis.add_geometry(target)
for v in geoms.values():
    vis.add_geometry(v)
for p in path:
    drone_trans[:3, 3] = p
    poses = kin.forward_kinematics({}, drone_trans)
    geoms = kin.get_transformed_visual_geometry_map(poses)
    vis.clear_geometries()
    vis.add_geometry(pathline)
    vis.add_geometry(target)
    for v in geoms.values():
        vis.add_geometry(v)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(1)
vis.destroy_window()