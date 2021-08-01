import cupoch as cph
from trajectory_io import read_trajectory
import numpy as np

if __name__ == "__main__":
    camera_poses = read_trajectory("../../testdata/rgbd/odometry.log")
    camera_intrinsics = cph.camera.PinholeCameraIntrinsic(
        cph.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    volume = cph.integration.UniformTSDFVolume(
        length=8.0,
        resolution=512,
        sdf_trunc=0.04,
        color_type=cph.integration.TSDFVolumeColorType.RGB8,
    )

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
            camera_intrinsics,
            np.linalg.inv(camera_poses[i].pose),
        )

    print("Extract triangle mesh")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    cph.visualization.draw_geometries([mesh])

    print("Extract voxel-aligned debugging point cloud")
    voxel_pcd = volume.extract_voxel_point_cloud()
    cph.visualization.draw_geometries([voxel_pcd])

    print("Extract voxel-aligned debugging voxel grid")
    voxel_grid = volume.extract_voxel_grid()
    cph.visualization.draw_geometries([voxel_grid])

    print("Extract point cloud")
    pcd = volume.extract_point_cloud()
    cph.visualization.draw_geometries([pcd])

    print("Raycasting")
    pcd = volume.raycast(camera_intrinsics, np.linalg.inv(camera_poses[0].pose), 0.04)
    cph.visualization.draw_geometries([pcd])
