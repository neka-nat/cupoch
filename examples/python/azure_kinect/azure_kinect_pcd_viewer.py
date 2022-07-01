import numpy as np
from pyKinectAzure import pyKinectAzure, _k4a, _k4atypes
import cupoch as cph

module_path = "/usr/local/lib/libk4a.so"


def main():
    pyk4a = pyKinectAzure.pyKinectAzure(module_path)
    pyk4a.device_open()
    device_config = pyk4a.config
    device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = _k4a.K4A_DEPTH_MODE_NFOV_UNBINNED
    print(device_config)
    calib = _k4atypes.k4a_calibration_t()
    pyk4a.device_get_calibration(device_config.depth_mode, device_config.color_resolution, calib)
    p = calib.depth_camera_calibration.intrinsics.parameters
    intrinsic = cph.camera.PinholeCameraIntrinsic(
        calib.depth_camera_calibration.resolution_width,
        calib.depth_camera_calibration.resolution_height,
        p.param.fx,
        p.param.fy,
        p.param.cx,
        p.param.cy,
    )

    # Start cameras using modified configuration
    pyk4a.device_start_cameras(device_config)

    clipping_distance_in_meters = 3
    vis = cph.visualization.Visualizer()
    vis.create_window()
    pcd = cph.geometry.PointCloud()
    frame_count = 0
    while True:
        pyk4a.device_get_capture()
        depth_image_handle = pyk4a.capture_get_depth_image()
        if depth_image_handle:
            depth_image = pyk4a.image_convert_to_numpy(depth_image_handle)
            depth_image = cph.geometry.Image(depth_image)
            temp = cph.geometry.PointCloud.create_from_depth_image(depth_image, intrinsic)
            pcd.points = temp.points

            if frame_count == 0:
                vis.add_geometry(pcd)

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            frame_count += 1
            pyk4a.image_release(depth_image_handle)
        else:
            break
        pyk4a.capture_release()

    pyk4a.device_stop_cameras()
    pyk4a.device_close()


if __name__ == "__main__":
    main()
