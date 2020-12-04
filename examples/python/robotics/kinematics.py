import cupoch as cph

kin = cph.kinematics.KinematicChain("../../testdata/dobot/dobot.urdf")
poses = kin.forward_kinematics()
geoms = kin.get_transformed_visual_geometries(poses)
cph.visualization.draw_geometries(geoms)