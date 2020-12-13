import cupoch as cph

kin = cph.kinematics.KinematicChain("../../testdata/franka_panda/panda.urdf")
poses = kin.forward_kinematics()
geoms = kin.get_transformed_visual_geometry_map(poses)
cph.visualization.draw_geometries(list(geoms.values()))