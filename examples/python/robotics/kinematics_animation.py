import cupoch as cph
import numpy as np
import time

kin = cph.kinematics.KinematicChain("../../testdata/dobot/dobot.urdf")
joint_map = {'joint_linkAlinkA9': 0.0,
             'joint_linkAlinkB': 0.0}
poses = kin.forward_kinematics()
geoms = kin.get_transformed_visual_geometry_map(poses)

vis = cph.visualization.Visualizer()
vis.create_window()
for v in geoms.values():
    vis.add_geometry(v)

dh = np.pi * 0.05
for i in range(15):
    joint_map = {'joint_linkAlinkA9': dh * i,
                 'joint_linkAlinkB': dh * i}
    poses = kin.forward_kinematics(joint_map)
    geoms = kin.get_transformed_visual_geometry_map(poses)
    vis.clear_geometries()
    for v in geoms.values():
        vis.add_geometry(v)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.1)
vis.destroy_window()