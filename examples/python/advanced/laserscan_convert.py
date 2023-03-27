import cupoch as cph
import numpy as np


if __name__ == "__main__":
    # From point cloud demo
    pcd = cph.io.read_point_cloud("../../testdata/fragment.ply")
    laserscan = cph.geometry.LaserScanBuffer.create_from_point_cloud(pcd, 0.01, 0, 10)
    print(laserscan.num_scans)
    converted_pcd = cph.geometry.PointCloud.create_from_laserscanbuffer(laserscan, 0.0, 1000.0)
    cph.visualization.draw_geometries([pcd, converted_pcd])

    # From depth image demo
    pinhole_camera_intrinsic = cph.io.read_pinhole_camera_intrinsic("../../testdata/camera_primesense.json")
    print(pinhole_camera_intrinsic.intrinsic_matrix)
    image = cph.io.read_image("../../testdata/rgbd/depth/00000.png")
    pcd = cph.geometry.PointCloud.create_from_depth_image(image, pinhole_camera_intrinsic, depth_scale=1000)
    laserscan = cph.geometry.LaserScanBuffer.create_from_depth_image(image, pinhole_camera_intrinsic, 0.01, 0, 10, depth_scale=1000)
    converted_pcd = cph.geometry.PointCloud.create_from_laserscanbuffer(laserscan, 0.0, 1000.0)
    cph.visualization.draw_geometries([pcd, converted_pcd])
