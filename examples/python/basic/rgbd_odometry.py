import cupoch as cph
import numpy as np

if __name__ == "__main__":
    pinhole_camera_intrinsic = cph.io.read_pinhole_camera_intrinsic("../../testdata/camera_primesense.json")
    print(pinhole_camera_intrinsic.intrinsic_matrix)

    source_color = cph.io.read_image("../../testdata/rgbd/color/00000.jpg")
    source_depth = cph.io.read_image("../../testdata/rgbd/depth/00000.png")
    target_color = cph.io.read_image("../../testdata/rgbd/color/00001.jpg")
    target_depth = cph.io.read_image("../../testdata/rgbd/depth/00001.png")
    source_rgbd_image = cph.geometry.RGBDImage.create_from_color_and_depth(source_color, source_depth)
    target_rgbd_image = cph.geometry.RGBDImage.create_from_color_and_depth(target_color, target_depth)
    target_pcd = cph.geometry.PointCloud.create_from_rgbd_image(target_rgbd_image, pinhole_camera_intrinsic)

    option = cph.odometry.OdometryOption()
    odo_init = np.identity(4)
    print(option)

    [success_color_term, trans_color_term, info] = cph.odometry.compute_rgbd_odometry(
        source_rgbd_image,
        target_rgbd_image,
        pinhole_camera_intrinsic,
        odo_init,
        cph.odometry.RGBDOdometryJacobianFromColorTerm(),
        option,
    )
    [success_hybrid_term, trans_hybrid_term, info] = cph.odometry.compute_rgbd_odometry(
        source_rgbd_image,
        target_rgbd_image,
        pinhole_camera_intrinsic,
        odo_init,
        cph.odometry.RGBDOdometryJacobianFromHybridTerm(),
        option,
    )

    if success_color_term:
        print("Using RGB-D Odometry")
        print(trans_color_term)
        source_pcd_color_term = cph.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, pinhole_camera_intrinsic
        )
        source_pcd_color_term.transform(trans_color_term)
        cph.visualization.draw_geometries([target_pcd, source_pcd_color_term])
    if success_hybrid_term:
        print("Using Hybrid RGB-D Odometry")
        print(trans_hybrid_term)
        source_pcd_hybrid_term = cph.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, pinhole_camera_intrinsic
        )
        source_pcd_hybrid_term.transform(trans_hybrid_term)
        cph.visualization.draw_geometries([target_pcd, source_pcd_hybrid_term])
