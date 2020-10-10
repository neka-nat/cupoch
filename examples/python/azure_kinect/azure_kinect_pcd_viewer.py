import numpy as np
import pyk4a
from pyk4a import Config, PyK4A
import cupoch as cph

def main():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510

    vis = cph.visualization.Visualizer()
    vis.create_window()
    pcd = cph.geometry.PointCloud()
    frame_count = 0
    while True:
        capture = k4a.get_capture()
        if not np.any(capture.depth) or not np.any(capture.color):
            break
        points = capture.depth_point_cloud.reshape((-1, 3))
        colors = capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3)) / 255
        pcd.points = cph.utility.Vector3fVector(points)
        pcd.colors = cph.utility.Vector3fVector(colors)

        if frame_count == 0:
            vis.add_geometry(pcd)

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        frame_count += 1
    k4a.stop()

if __name__ == "__main__":
    main()