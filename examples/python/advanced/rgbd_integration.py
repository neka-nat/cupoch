import cupoch as cph
from trajectory_io import *
import numpy as np

if __name__ == "__main__":
    camera_poses = read_trajectory("../../testdata/rgbd/odometry.log")
    volume = cph.integration.ScalableTSDFVolume(
        voxel_length=8.0 / 512.0,
        sdf_trunc=0.04,
        color_type=cph.integration.TSDFVolumeColorType.RGB8)

    for i in range(len(camera_poses)):
        print("Integrate {:d}-th image into the volume.".format(i))
        color = cph.io.read_image(
            "../../testdata/rgbd/color/{:05d}.jpg".format(i))
        depth = cph.io.read_image(
            "../../testdata/rgbd/depth/{:05d}.png".format(i))
        rgbd = cph.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
        volume.integrate(
            rgbd,
            cph.camera.PinholeCameraIntrinsic(
                cph.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
            np.linalg.inv(camera_poses[i].pose))

    print("Extract a point cloud from the volume and visualize it.")
    pcd = volume.extract_point_cloud()
    cph.visualization.draw_geometries([pcd])