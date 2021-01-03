import sys
import pyrealsense2 as rs
import numpy as np
from enum import IntEnum

from datetime import datetime
import cupoch as x3d
x3d.initialize_allocator(x3d.PoolAllocation, 1000000000)


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def get_intrinsic_matrix(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = x3d.camera.PinholeCameraIntrinsic(640, 480, intrinsics.fx,
                                            intrinsics.fy, intrinsics.ppx,
                                            intrinsics.ppy)
    return out


if __name__ == "__main__":

    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    # Using preset HighAccuracy for recording
    depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    vis = x3d.visualization.Visualizer()
    vis.create_window()

    ocgd = x3d.geometry.OccupancyGrid(0.05, 512)
    ocgd.visualize_free_area = False
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    prev_rgbd_image = None
    option = x3d.odometry.OdometryOption()
    cur_trans = np.identity(4)

    # Streaming loop
    frame_count = 0
    try:
        while True:

            dt0 = datetime.now()

            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            intrinsic = x3d.camera.PinholeCameraIntrinsic(
                get_intrinsic_matrix(color_frame))

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_temp = np.array(aligned_depth_frame.get_data())
            depth_image = x3d.geometry.Image(depth_temp)
            color_temp = np.asarray(color_frame.get_data())
            color_image = x3d.geometry.Image(color_temp)

            rgbd_image = x3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image,
                depth_image)
            if not prev_rgbd_image is None:
                res, odo_trans, _ = x3d.odometry.compute_rgbd_odometry(
                                prev_rgbd_image, rgbd_image, intrinsic,
                                np.identity(4), x3d.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
                if res:
                    cur_trans = np.matmul(cur_trans, odo_trans)

            prev_rgbd_image = rgbd_image
            temp = x3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, intrinsic)
            temp.transform(np.matmul(cur_trans, flip_transform))
            temp = temp.voxel_down_sample(0.05)
            ocgd.insert(temp, cur_trans[:3, 3])

            if frame_count == 0:
                vis.add_geometry(ocgd)

            vis.update_geometry(ocgd)
            vis.poll_events()
            vis.update_renderer()

            dt1 = datetime.now()
            process_time = dt1 - dt0
            print("FPS: " + str(1 / process_time.total_seconds()))
            frame_count += 1

    finally:
        pipeline.stop()
    vis.destroy_window()